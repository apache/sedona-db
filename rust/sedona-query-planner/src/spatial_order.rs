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

//! Use registered SD_Order overloads for sort keys.

use std::sync::Arc;

use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{DFSchema, Result};
use datafusion_expr::expr_rewriter::NamePreserver;
use datafusion_expr::{Expr, ExprSchemable, LogicalPlan, ScalarUDF, SortExpr};
use datafusion_optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::datatypes::SedonaType;

#[derive(Default, Debug)]
pub struct SpatialOrderOptimizerRule;

impl OptimizerRule for SpatialOrderOptimizerRule {
    fn name(&self) -> &str {
        "sedona.spatial_order"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if !matches!(
            plan,
            LogicalPlan::Sort(_) | LogicalPlan::Aggregate(_) | LogicalPlan::Window(_)
        ) {
            return Ok(Transformed::no(plan));
        }
        let Some(registry) = config.function_registry() else {
            return Ok(Transformed::no(plan));
        };
        let Ok(order) = registry.udf("sd_order") else {
            return Ok(Transformed::no(plan));
        };
        // Cache the session's UDF and its Sedona implementation once per rewrite,
        // so all sort keys share overload resolution without repeated lookups.
        let Some(order_impl) = order.inner().downcast_ref::<SedonaScalarUDF>() else {
            return Ok(Transformed::no(plan));
        };

        match plan {
            LogicalPlan::Sort(mut sort) => {
                let changed =
                    rewrite_sorts(&mut sort.expr, sort.input.schema(), &order, order_impl)?;
                Ok(Transformed::new_transformed(
                    LogicalPlan::Sort(sort),
                    changed,
                ))
            }
            LogicalPlan::Aggregate(_) | LogicalPlan::Window(_) => {
                let schema = plan.inputs()[0].schema().clone();
                let names = NamePreserver::new(&plan);
                plan.map_expressions(|expr| {
                    let name = names.save(&expr);
                    expr.transform_up(|mut expr| {
                        let changed = match &mut expr {
                            Expr::AggregateFunction(function) => rewrite_sorts(
                                &mut function.params.order_by,
                                &schema,
                                &order,
                                order_impl,
                            )?,
                            Expr::WindowFunction(function) => rewrite_sorts(
                                &mut function.params.order_by,
                                &schema,
                                &order,
                                order_impl,
                            )?,
                            _ => false,
                        };
                        Ok(Transformed::new_transformed(expr, changed))
                    })?
                    .map_data(|expr| Ok(name.restore(expr)))
                })
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

fn rewrite_sorts(
    sorts: &mut [SortExpr],
    schema: &DFSchema,
    order: &Arc<ScalarUDF>,
    order_impl: &SedonaScalarUDF,
) -> Result<bool> {
    let mut changed = false;
    for sort in sorts {
        // Use the field to retain extension metadata for custom types.
        let (_, field) = sort.expr.to_field(schema)?;
        let arg_type = SedonaType::from_storage_field(&field)?;
        let scalar = match &sort.expr {
            Expr::Literal(value, _) => Some(value),
            _ => None,
        };
        if order_impl.resolve_kernel(&[arg_type], &[scalar])?.is_some() {
            sort.expr = order.call(vec![sort.expr.clone()]);
            changed = true;
        }
    }
    Ok(changed)
}
