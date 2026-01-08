// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::{iter::zip, ptr::swap_nonoverlapping, sync::Arc};

use arrow_array::{
    ffi_stream::FFI_ArrowArrayStream, RecordBatch, RecordBatchIterator, RecordBatchReader,
};
use arrow_schema::{FieldRef, Schema};
use datafusion::physical_plan::PhysicalExpr;
use datafusion_common::{
    tree_node::{Transformed, TreeNode},
    Column, DFSchema, Result, ScalarValue,
};
use datafusion_expr::{
    expr::{AggregateFunction, FieldMetadata, NullTreatment, ScalarFunction, WindowFunction},
    utils::{expr_as_column_expr, find_aggregate_exprs, find_column_exprs, find_window_exprs},
    BinaryExpr, Cast, ColumnarValue, Expr, LogicalPlan, LogicalPlanBuilder,
    LogicalPlanBuilderOptions, Operator, WindowFunctionDefinition,
};
use savvy::{savvy, savvy_err, EnvironmentSexp};
use sedona::context::SedonaContext;

use crate::{
    context::InternalContext,
    ffi::{import_array, import_field},
};

#[savvy]
pub struct SedonaDBExpr {
    pub inner: Expr,
}

#[savvy]
impl SedonaDBExpr {
    fn display(&self) -> savvy::Result<savvy::Sexp> {
        format!("{}", self.inner).try_into()
    }

    fn debug_string(&self) -> savvy::Result<savvy::Sexp> {
        format!("{:?}", self.inner).try_into()
    }

    fn alias(&self, name: &str) -> savvy::Result<SedonaDBExpr> {
        let inner = self.inner.clone().alias_if_changed(name.to_string())?;
        Ok(Self { inner })
    }

    fn cast(&self, schema_xptr: savvy::Sexp) -> savvy::Result<SedonaDBExpr> {
        let field = import_field(schema_xptr)?;
        if let Some(type_name) = field.extension_type_name() {
            return Err(savvy_err!(
                "Can't cast to Arrow extension type '{type_name}'"
            ));
        }

        let inner = Expr::Cast(Cast::new(
            self.inner.clone().into(),
            field.data_type().clone(),
        ));

        Ok(Self { inner })
    }

    fn negate(&self) -> savvy::Result<SedonaDBExpr> {
        let inner = Expr::Negative(Box::new(self.inner.clone()));
        Ok(Self { inner })
    }
}

#[savvy]
pub struct SedonaDBExprFactory {
    pub ctx: Arc<SedonaContext>,
}

#[savvy]
impl SedonaDBExprFactory {
    fn new(ctx: &InternalContext) -> Self {
        Self {
            ctx: ctx.inner.clone(),
        }
    }

    fn literal(array_xptr: savvy::Sexp, schema_xptr: savvy::Sexp) -> savvy::Result<SedonaDBExpr> {
        let (field, array_ref) = import_array(array_xptr, schema_xptr)?;
        let metadata = if field.metadata().is_empty() {
            None
        } else {
            Some(FieldMetadata::new_from_field(&field))
        };

        let scalar_value = ScalarValue::try_from_array(&array_ref, 0)?;
        let inner = Expr::Literal(scalar_value, metadata);
        Ok(SedonaDBExpr { inner })
    }

    fn column(&self, name: &str, qualifier: Option<&str>) -> savvy::Result<SedonaDBExpr> {
        let inner = Expr::Column(Column::new(qualifier, name));
        Ok(SedonaDBExpr { inner })
    }

    fn binary(
        &self,
        op: &str,
        lhs: &SedonaDBExpr,
        rhs: &SedonaDBExpr,
    ) -> savvy::Result<SedonaDBExpr> {
        let operator = match op {
            "==" => Operator::Eq,
            "!=" => Operator::NotEq,
            ">" => Operator::Gt,
            ">=" => Operator::GtEq,
            "<" => Operator::Lt,
            "<=" => Operator::LtEq,
            "+" => Operator::Plus,
            "-" => Operator::Minus,
            "*" => Operator::Multiply,
            "/" => Operator::Divide,
            "&" => Operator::And,
            "|" => Operator::Or,
            other => return Err(savvy_err!("Unimplemented binary operation '{other}'")),
        };

        let inner = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(lhs.inner.clone()),
            operator,
            Box::new(rhs.inner.clone()),
        ));
        Ok(SedonaDBExpr { inner })
    }

    fn scalar_function(&self, name: &str, args: savvy::Sexp) -> savvy::Result<SedonaDBExpr> {
        if let Some(udf) = self.ctx.ctx.state().scalar_functions().get(name) {
            let args = Self::exprs(args)?;
            let inner = Expr::ScalarFunction(ScalarFunction::new_udf(udf.clone(), args));
            Ok(SedonaDBExpr { inner })
        } else {
            Err(savvy_err!("Scalar UDF '{name}' not found"))
        }
    }

    fn aggregate_function(
        &self,
        name: &str,
        args: savvy::Sexp,
        na_rm: Option<bool>,
        distinct: Option<bool>,
    ) -> savvy::Result<SedonaDBExpr> {
        if let Some(udf) = self.ctx.ctx.state().aggregate_functions().get(name) {
            let args = Self::exprs(args)?;
            let null_treatment = if na_rm.unwrap_or(true) {
                NullTreatment::IgnoreNulls
            } else {
                NullTreatment::RespectNulls
            };

            let inner = Expr::AggregateFunction(AggregateFunction::new_udf(
                udf.clone(),
                args,
                distinct.unwrap_or(false),
                None,   // filter
                vec![], // order by
                Some(null_treatment),
            ));

            Ok(SedonaDBExpr { inner })
        } else {
            Err(savvy_err!("Aggregate UDF '{name}' not found"))
        }
    }

    fn evaluate_scalar(
        &self,
        exprs_sexp: savvy::Sexp,
        stream_in: savvy::Sexp,
        stream_out: savvy::Sexp,
    ) -> savvy::Result<savvy::Sexp> {
        let out_void = unsafe { savvy_ffi::R_ExternalPtrAddr(stream_out.0) };
        if out_void.is_null() {
            return Err(savvy_err!("external pointer to null in evaluate()"));
        }

        let exprs = Self::exprs(exprs_sexp)?;
        let expr_names = exprs
            .iter()
            .map(|e| e.schema_name().to_string())
            .collect::<Vec<_>>();
        let reader_in = crate::ffi::import_array_stream(stream_in)?;

        let physical_exprs = exprs
            .into_iter()
            .map(|e| {
                self.ctx.ctx.create_physical_expr(
                    e,
                    &DFSchema::try_from(reader_in.schema().as_ref().clone())?,
                )
            })
            .collect::<datafusion_common::Result<Vec<Arc<dyn PhysicalExpr>>>>()?;

        let out_fields = physical_exprs
            .iter()
            .map(|e| e.return_field(&reader_in.schema()))
            .collect::<datafusion_common::Result<Vec<FieldRef>>>()?;
        let out_fields_named = zip(out_fields, expr_names)
            .map(|(f, name)| f.as_ref().clone().with_name(name))
            .collect::<Vec<_>>();
        let out_schema = Arc::new(Schema::new(out_fields_named));

        let mut out_batches = Vec::new();
        let mut size = 0;
        for batch in reader_in {
            let batch = batch?;
            size += batch.num_rows();
            let columns = physical_exprs
                .iter()
                .map(|e| e.evaluate(&batch))
                .collect::<datafusion_common::Result<Vec<ColumnarValue>>>()?;
            let out_batch = RecordBatch::try_new(
                out_schema.clone(),
                ColumnarValue::values_to_arrays(&columns)?,
            )?;
            out_batches.push(out_batch);
        }

        let reader = Box::new(RecordBatchIterator::new(
            out_batches.into_iter().map(Ok),
            out_schema,
        ));
        let mut ffi_stream = FFI_ArrowArrayStream::new(reader);
        let ffi_out = out_void as *mut FFI_ArrowArrayStream;
        unsafe { swap_nonoverlapping(&mut ffi_stream, ffi_out, 1) };

        savvy::Sexp::try_from(size as f64)
    }
}

impl SedonaDBExprFactory {
    pub fn exprs(exprs_sexp: savvy::Sexp) -> savvy::Result<Vec<Expr>> {
        savvy::ListSexp::try_from(exprs_sexp)?
            .iter()
            .map(|(_, item)| -> savvy::Result<Expr> {
                // item here is the Environment wrapper around the external pointer
                let expr_wrapper: &SedonaDBExpr = EnvironmentSexp::try_from(item)?.try_into()?;
                Ok(expr_wrapper.inner.clone())
            })
            .collect()
    }

    pub fn select(
        base_plan: LogicalPlan,
        exprs: Vec<Expr>,
        group_by_exprs: Vec<Expr>,
    ) -> datafusion_common::Result<LogicalPlan> {
        // Translated from DataFusion's SQL SELECT -> LogicalPlan constructor
        // https://github.com/apache/datafusion/blob/102caeb2261c5ae006c201546cf74769d80ceff8/datafusion/sql/src/select.rs#L890-L1098

        // First, find aggregates in SELECT
        let aggr_exprs = find_aggregate_exprs(&exprs);

        // Determine if we should use aggregation or window functions
        // If we have an explicit GROUP BY or can infer one, use aggregation
        // Otherwise, treat aggregates as window functions
        let use_aggregation = if !group_by_exprs.is_empty() {
            true
        } else if !aggr_exprs.is_empty() {
            // Try to infer GROUP BY from columns outside aggregates
            let all_columns = find_column_exprs(&exprs);
            let agg_columns = find_column_exprs(&aggr_exprs);
            let non_agg_columns: Vec<_> = all_columns
                .into_iter()
                .filter(|col| !agg_columns.contains(col))
                .collect();
            !non_agg_columns.is_empty()
        } else {
            false
        };

        // Process aggregation if appropriate
        let (plan, select_exprs_post_aggr) = if use_aggregation && !aggr_exprs.is_empty() {
            // We have aggregates with a valid GROUP BY, create aggregate plan
            let result = Self::aggregate(&base_plan, &exprs, group_by_exprs, &aggr_exprs)?;
            (result.plan, result.select_exprs)
        } else if !aggr_exprs.is_empty() {
            // We have aggregates but no valid GROUP BY - convert to window functions
            // First resolve column references to be fully qualified
            let exprs_resolved: Vec<Expr> = exprs
                .iter()
                .map(|expr| Self::resolve_columns(expr, &base_plan))
                .collect::<Result<Vec<_>>>()?;

            let exprs_with_windows = Self::aggregates_to_window_functions(&exprs_resolved)?;
            (base_plan, exprs_with_windows)
        } else {
            // No aggregation
            (base_plan, exprs.clone())
        };

        // All of the window expressions (includes aggregates converted to windows)
        let window_func_exprs = find_window_exprs(&select_exprs_post_aggr);

        // Process window functions after aggregation
        let plan = if window_func_exprs.is_empty() {
            plan
        } else {
            // Resolve columns in window expressions to be fully qualified
            let window_func_exprs: Vec<Expr> = window_func_exprs
                .iter()
                .map(|expr| Self::resolve_columns(expr, &plan))
                .collect::<Result<Vec<_>>>()?;

            let plan = LogicalPlanBuilder::window_plan(plan, window_func_exprs.clone())?;

            // Re-write the projection
            let select_exprs_post_aggr = select_exprs_post_aggr
                .iter()
                .map(|expr| Self::rebase_expr(expr, &window_func_exprs, &plan))
                .collect::<Result<Vec<Expr>>>()?;

            // Final projection
            LogicalPlanBuilder::from(plan)
                .project(select_exprs_post_aggr)?
                .build()?
        };

        // Final projection if no windows
        if window_func_exprs.is_empty() {
            LogicalPlanBuilder::from(plan)
                .project(select_exprs_post_aggr)?
                .build()
        } else {
            Ok(plan)
        }
    }

    /// Helper function to rebase expressions to reference columns from the plan.
    /// Simplified version of datafusion-sql's rebase_expr (which is pub(crate)).
    fn rebase_expr(expr: &Expr, base_exprs: &[Expr], plan: &LogicalPlan) -> Result<Expr> {
        let result = expr.clone().transform_down(|nested_expr| {
            if base_exprs.contains(&nested_expr) {
                Ok(Transformed::yes(expr_as_column_expr(&nested_expr, plan)?))
            } else {
                Ok(Transformed::no(nested_expr))
            }
        })?;
        Ok(result.data)
    }

    /// Helper function to resolve column references to fully qualified columns.
    /// Simplified version of datafusion-sql's resolve_columns (which is pub(crate)).
    fn resolve_columns(expr: &Expr, plan: &LogicalPlan) -> Result<Expr> {
        let result = expr.clone().transform_up(|nested_expr| {
            match nested_expr {
                Expr::Column(col) => {
                    let (qualifier, field) = plan.schema().qualified_field_from_column(&col)?;
                    Ok(Transformed::yes(Expr::Column(Column::from((
                        qualifier, field,
                    )))))
                }
                _ => {
                    // keep recursing
                    Ok(Transformed::no(nested_expr))
                }
            }
        })?;
        Ok(result.data)
    }

    /// Convert aggregate functions to window functions with empty OVER clause
    fn aggregates_to_window_functions(exprs: &[Expr]) -> Result<Vec<Expr>> {
        exprs
            .iter()
            .map(|expr| {
                expr.clone()
                    .transform_up(|nested_expr| {
                        match nested_expr {
                            Expr::AggregateFunction(agg) => {
                                // Convert to window function with empty OVER ()
                                let window_func =
                                    Expr::WindowFunction(Box::new(WindowFunction::new(
                                        WindowFunctionDefinition::AggregateUDF(agg.func.clone()),
                                        agg.params.args,
                                    )));
                                Ok(Transformed::yes(window_func))
                            }
                            _ => Ok(Transformed::no(nested_expr)),
                        }
                    })
                    .map(|t| t.data)
            })
            .collect()
    }

    /// Create an aggregate plan from the given input, group by, and aggregate expressions.
    /// Based on DataFusion's aggregate() method.
    /// https://github.com/apache/datafusion/blob/102caeb2261c5ae006c201546cf74769d80ceff8/datafusion/sql/src/select.rs#L652-L764
    fn aggregate(
        input: &LogicalPlan,
        select_exprs: &[Expr],
        group_by_exprs: Vec<Expr>,
        aggr_exprs: &[Expr],
    ) -> Result<AggregatePlanResult> {
        // If group_by_exprs is empty, we need to extract column references from
        // select_exprs that are NOT inside aggregate functions
        let group_by_exprs = if group_by_exprs.is_empty() {
            // Find all columns referenced in select expressions
            let all_columns = find_column_exprs(select_exprs);

            // Find columns that are inside aggregate expressions
            let agg_columns = find_column_exprs(aggr_exprs);

            // Keep only columns that are NOT inside aggregates
            all_columns
                .into_iter()
                .filter(|col| !agg_columns.contains(col))
                .collect::<Vec<_>>()
        } else {
            group_by_exprs
        };

        // Create the aggregate plan
        let options = LogicalPlanBuilderOptions::new().with_add_implicit_group_by_exprs(true);
        let plan = LogicalPlanBuilder::from(input.clone())
            .with_options(options)
            .aggregate(group_by_exprs, aggr_exprs.to_vec())?
            .build()?;

        // Get the group_by_exprs and aggr_exprs from the constructed plan
        // (they may have been modified by implicit group by logic)
        let (group_by_exprs, aggr_exprs_from_plan) = if let LogicalPlan::Aggregate(agg) = &plan {
            (&agg.group_expr, &agg.aggr_expr)
        } else {
            unreachable!();
        };

        // Combine the original grouping and aggregate expressions into one list
        let mut aggr_projection_exprs = vec![];
        for expr in group_by_exprs {
            aggr_projection_exprs.push(expr.clone());
        }
        aggr_projection_exprs.extend_from_slice(aggr_exprs_from_plan);

        // Now attempt to resolve columns and replace with fully-qualified columns
        let aggr_projection_exprs = aggr_projection_exprs
            .iter()
            .map(|expr| Self::resolve_columns(expr, input))
            .collect::<Result<Vec<Expr>>>()?;

        // Resolve columns in select expressions too, so qualifiers match when rebasing
        let select_exprs_resolved = select_exprs
            .iter()
            .map(|expr| Self::resolve_columns(expr, input))
            .collect::<Result<Vec<Expr>>>()?;

        // Re-write the projection
        let select_exprs_post_aggr = select_exprs_resolved
            .iter()
            .map(|expr| Self::rebase_expr(expr, &aggr_projection_exprs, input))
            .collect::<Result<Vec<Expr>>>()?;

        Ok(AggregatePlanResult {
            plan,
            select_exprs: select_exprs_post_aggr,
        })
    }
}

impl TryFrom<EnvironmentSexp> for &SedonaDBExpr {
    type Error = savvy::Error;

    fn try_from(env: EnvironmentSexp) -> Result<Self, Self::Error> {
        env.get(".ptr")?
            .map(<&SedonaDBExpr>::try_from)
            .transpose()?
            .ok_or(savvy_err!("Invalid SedonaDBExpr object."))
    }
}

/// Result of the `aggregate` function, containing the aggregate plan and
/// rewritten expressions that reference the aggregate output columns.
/// https://github.com/apache/datafusion/blob/102caeb2261c5ae006c201546cf74769d80ceff8/datafusion/sql/src/select.rs#L55-L68
struct AggregatePlanResult {
    /// The aggregate logical plan
    plan: LogicalPlan,
    /// SELECT expressions rewritten to reference aggregate output columns
    select_exprs: Vec<Expr>,
}
