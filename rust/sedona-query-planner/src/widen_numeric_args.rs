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
//! Cast integer and decimal arguments of Sedona functions to `Float64` when
//! that is the only way a kernel accepts them.
//!
//! A [`SedonaScalarUDF`] matches its kernels against Sedona types, which
//! DataFusion's own type coercion cannot see (`coerce_types` is handed storage
//! types, where geometry, geography and plain binary all read as `Binary`).
//! Kernels that declare an exact `Float64` argument, such as the ones imported
//! from s2geography, therefore reject the `Int32` or `Decimal128` a Spark
//! Connect client sends for the literals `1` and `1.5`. This analyzer asks the
//! function which arguments it would accept widened to `Float64` (see
//! [`SedonaScalarUDF::widened_arg_types`]) and wraps exactly those in an
//! explicit [`Expr::Cast`], so the casts appear in `EXPLAIN`, a literal is cast
//! once when the plan is simplified rather than on every batch, and the kernel
//! runs on the types it matched.

use std::sync::Arc;

use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{DFSchema, Result, ScalarValue};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::utils::merge_schema;
use datafusion_expr::{Cast, Expr, ExprSchemable, LogicalPlan};
use datafusion_optimizer::AnalyzerRule;
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::datatypes::SedonaType;

/// Analyzer rule that casts the integer and decimal arguments of a Sedona
/// function to `Float64` when no kernel accepts them as given.
#[derive(Debug, Default)]
pub struct WidenNumericArgs;

impl AnalyzerRule for WidenNumericArgs {
    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_up_with_subqueries(|plan| {
            let schema = merge_schema(&plan.inputs());
            plan.map_expressions(|expr| {
                expr.transform_up(|expr| widen_scalar_function(expr, &schema))
            })
        })
        .data()
    }

    fn name(&self) -> &str {
        "sedona.widen_numeric_args"
    }
}

fn widen_scalar_function(expr: Expr, schema: &DFSchema) -> Result<Transformed<Expr>> {
    let Expr::ScalarFunction(ScalarFunction { func, args }) = &expr else {
        return Ok(Transformed::no(expr));
    };
    let Some(udf) = func.inner().downcast_ref::<SedonaScalarUDF>() else {
        return Ok(Transformed::no(expr));
    };

    let arg_types = args
        .iter()
        .map(|arg| {
            let (_, field) = arg.to_field(schema)?;
            SedonaType::from_storage_field(&field)
        })
        .collect::<Result<Vec<_>>>()?;
    let scalars: Vec<Option<&ScalarValue>> = args
        .iter()
        .map(|arg| match arg {
            Expr::Literal(value, _) => Some(value),
            _ => None,
        })
        .collect();

    let Some(widened_types) = udf.widened_arg_types(&arg_types, &scalars)? else {
        return Ok(Transformed::no(expr));
    };

    let widened_args = args
        .iter()
        .zip(arg_types.iter().zip(&widened_types))
        .map(|(arg, (arg_type, widened_type))| {
            if arg_type == widened_type {
                arg.clone()
            } else {
                Expr::Cast(Cast::new(
                    Box::new(arg.clone()),
                    widened_type.storage_type().clone(),
                ))
            }
        })
        .collect();

    Ok(Transformed::yes(Expr::ScalarFunction(
        ScalarFunction::new_udf(Arc::clone(func), widened_args),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::DataType;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::{SessionContext, col, lit};
    use datafusion::{assert_batches_eq, assert_batches_sorted_eq};
    use datafusion_expr::{ColumnarValue, LogicalPlanBuilder, ScalarUDF, cast};
    use sedona_expr::scalar_udf::SimpleSedonaScalarKernel;
    use sedona_schema::matchers::ArgMatcher;

    use crate::optimizer::register_sedona_analyzer_rules;

    /// A function with one kernel that accepts exactly Float64 and echoes it back
    fn float64_only() -> ScalarUDF {
        SedonaScalarUDF::from_impl(
            "float64_only",
            SimpleSedonaScalarKernel::new_ref(
                ArgMatcher::new(
                    vec![ArgMatcher::is_arrow(DataType::Float64)],
                    SedonaType::Arrow(DataType::Float64),
                ),
                Arc::new(|arg_types, args| {
                    assert_eq!(arg_types, &[SedonaType::Arrow(DataType::Float64)]);
                    Ok(args[0].clone())
                }),
            ),
        )
        .into()
    }

    /// A function with one kernel that accepts any numeric type and reports its name
    fn numeric_any() -> ScalarUDF {
        SedonaScalarUDF::from_impl(
            "numeric_any",
            SimpleSedonaScalarKernel::new_ref(
                ArgMatcher::new(
                    vec![ArgMatcher::is_numeric()],
                    SedonaType::Arrow(DataType::Utf8),
                ),
                Arc::new(|arg_types, _| {
                    Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                        arg_types[0].logical_type_name(),
                    ))))
                }),
            ),
        )
        .into()
    }

    fn project_over_one_column(value: Expr, projection: impl Fn(Expr) -> Expr) -> LogicalPlan {
        LogicalPlanBuilder::values(vec![vec![value]])
            .unwrap()
            .project(vec![projection(col("column1"))])
            .unwrap()
            .build()
            .unwrap()
    }

    #[test]
    fn casts_only_the_arguments_the_function_asks_for() {
        let udf = Arc::new(float64_only());
        let plan = project_over_one_column(lit(1_i32), |column| udf.call(vec![column]));
        // The schema already names Float64 because the function resolved its return
        // type against the widened argument when the projection was built
        assert_eq!(plan.schema().field(0).data_type(), &DataType::Float64);

        let analyzed = WidenNumericArgs
            .analyze(plan, &ConfigOptions::default())
            .unwrap();
        let expected = project_over_one_column(lit(1_i32), |column| {
            udf.call(vec![cast(column, DataType::Float64)])
        });
        assert_eq!(analyzed, expected);
        assert_eq!(analyzed.schema().field(0).data_type(), &DataType::Float64);
    }

    #[test]
    fn leaves_arguments_a_kernel_accepts_alone() {
        for (udf, value) in [
            (Arc::new(float64_only()), lit(1.5_f64)),
            (
                Arc::new(numeric_any()),
                lit(ScalarValue::Decimal128(Some(150), 10, 2)),
            ),
        ] {
            let plan = project_over_one_column(value, |column| udf.call(vec![column]));
            let analyzed = WidenNumericArgs
                .analyze(plan.clone(), &ConfigOptions::default())
                .unwrap();
            assert_eq!(analyzed, plan);
        }
    }

    #[tokio::test]
    async fn casts_show_in_the_plan_and_the_kernel_runs_on_float64() {
        let builder =
            register_sedona_analyzer_rules(SessionStateBuilder::new_with_default_features())
                .unwrap();
        let ctx = SessionContext::new_with_state(builder.build());
        ctx.register_udf(float64_only());
        ctx.register_udf(numeric_any());

        // A column argument keeps an explicit cast in the plan
        let df = ctx
            .sql("SELECT float64_only(x) AS widened FROM (VALUES (1), (2)) AS t(x)")
            .await
            .unwrap();
        let plan = df.clone().into_optimized_plan().unwrap().to_string();
        assert!(
            plan.contains("float64_only(CAST(t.x AS Float64))"),
            "{plan}"
        );
        let batches = df.collect().await.unwrap();
        assert_batches_sorted_eq!(
            [
                "+---------+",
                "| widened |",
                "+---------+",
                "| 1.0     |",
                "| 2.0     |",
                "+---------+",
            ],
            &batches
        );

        // A literal argument is cast once while the plan is simplified, so no cast
        // remains in the optimized plan
        let df = ctx
            .sql("SELECT float64_only(CAST(1.5 AS DECIMAL(10, 2))) AS widened")
            .await
            .unwrap();
        let plan = df.clone().into_optimized_plan().unwrap().to_string();
        assert!(!plan.contains("CAST"), "{plan}");
        let batches = df.collect().await.unwrap();
        assert_batches_eq!(
            [
                "+---------+",
                "| widened |",
                "+---------+",
                "| 1.5     |",
                "+---------+",
            ],
            &batches
        );

        // A kernel that accepts the decimal as given sees the decimal
        let batches = ctx
            .sql("SELECT numeric_any(CAST(1.5 AS DECIMAL(10, 2))) AS as_given")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_batches_eq!(
            [
                "+------------+",
                "| as_given   |",
                "+------------+",
                "| decimal128 |",
                "+------------+",
            ],
            &batches
        );
    }
}
