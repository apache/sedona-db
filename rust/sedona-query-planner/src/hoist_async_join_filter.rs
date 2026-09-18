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

//! Logical optimizer rule that keeps async scalar UDF calls out of inner join
//! filters.
//!
//! DataFusion's physical planner hoists async scalar UDFs into `AsyncFuncExec`
//! only for `Projection`, `Filter` and `Aggregate` expressions. A join's
//! non-equi `filter` is evaluated synchronously inside the join operator
//! (`NestedLoopJoinExec`, `HashJoinExec`, `SpatialJoinExec`), so an async call
//! left there fails at execution with
//! `Internal error: async functions should not be called directly`.
//!
//! `PushDownFilter` is how one gets there. A `WHERE` on the output of an
//! async raster function over a join — `WHERE s.count > 0` with
//! `RS_ZonalStatsAll(rast, geom) AS s` — is rewritten in terms of the join
//! inputs and, because it references both sides, moved into the inner join's
//! filter. For an inner join that filter is a plain post-join predicate, so
//! this rule moves the conjuncts that call an async UDF back out into a
//! `Filter` above the join, where the physical planner handles them.
//!
//! Ordering: it runs after [`EnsureLoadedOptimizerRule`], because the raster
//! argument of a `needs_pixels` call only becomes an async `RS_EnsureLoaded`
//! call there, and before `SpatialJoinLogicalRewrite`, which turns the join
//! into an extension node whose filter this rule doesn't touch. On the next
//! optimizer pass `PushDownFilter` pushes the predicate back down and this
//! rule hoists it again; the pass is a net no-op, so DataFusion's plan
//! signature check ends the fixpoint.
//!
//! Outer joins are left alone: their filter decides which rows are
//! null-extended, so it can't become a post-filter. `PushDownFilter` never
//! moves a `WHERE` predicate into an outer join's filter anyway; an async call
//! written directly in an outer join's `ON` clause is unsupported.
//!
//! [`EnsureLoadedOptimizerRule`]: crate::ensure_loaded::EnsureLoadedOptimizerRule

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_expr::async_udf::AsyncScalarUDF;
use datafusion_expr::utils::{conjunction, split_conjunction_owned};
use datafusion_expr::{Expr, Filter, JoinType, LogicalPlan};
use datafusion_optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

/// Logical optimizer rule moving async-UDF conjuncts of an inner join's filter
/// into a `Filter` above the join. See the module docs.
#[derive(Default, Debug)]
pub struct HoistAsyncJoinFilterRule;

impl OptimizerRule for HoistAsyncJoinFilterRule {
    fn name(&self) -> &str {
        "sedona.hoist_async_join_filter"
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
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let mut join = match plan {
            LogicalPlan::Join(join) if join.join_type == JoinType::Inner => join,
            other => return Ok(Transformed::no(other)),
        };
        let Some(filter) = join.filter.take() else {
            return Ok(Transformed::no(LogicalPlan::Join(join)));
        };

        let (hoist, keep): (Vec<Expr>, Vec<Expr>) = split_conjunction_owned(filter.clone())
            .into_iter()
            .partition(calls_async_udf);
        let Some(hoisted) = conjunction(hoist) else {
            join.filter = Some(filter);
            return Ok(Transformed::no(LogicalPlan::Join(join)));
        };

        join.filter = conjunction(keep);
        let filter = Filter::try_new(hoisted, Arc::new(LogicalPlan::Join(join)))?;
        Ok(Transformed::yes(LogicalPlan::Filter(filter)))
    }
}

/// True if any call anywhere in `expr` is to an [`AsyncScalarUDF`].
fn calls_async_udf(expr: &Expr) -> bool {
    expr.exists(|e| {
        Ok(matches!(
            e,
            Expr::ScalarFunction(call)
                if call.func.inner().downcast_ref::<AsyncScalarUDF>().is_some()
        ))
    })
    .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::hash::{Hash, Hasher};

    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use datafusion_common::DFSchema;
    use datafusion_common::tree_node::TreeNodeRecursion;
    use datafusion_expr::async_udf::AsyncScalarUDFImpl;
    use datafusion_expr::expr::ScalarFunction;
    use datafusion_expr::{
        ColumnarValue, EmptyRelation, LogicalPlanBuilder, ScalarFunctionArgs, ScalarUDF,
        ScalarUDFImpl, Signature, Volatility, col, lit,
    };
    use datafusion_optimizer::OptimizerContext;

    /// A fake async UDF: Int64 -> Int64.
    #[derive(Debug)]
    struct FakeAsyncUdf {
        signature: Signature,
    }

    impl PartialEq for FakeAsyncUdf {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl Eq for FakeAsyncUdf {}
    impl Hash for FakeAsyncUdf {
        fn hash<H: Hasher>(&self, state: &mut H) {
            "fake_async".hash(state);
        }
    }

    impl ScalarUDFImpl for FakeAsyncUdf {
        fn name(&self) -> &str {
            "fake_async"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int64)
        }

        fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            unreachable!("stub; never executed")
        }
    }

    #[async_trait]
    impl AsyncScalarUDFImpl for FakeAsyncUdf {
        async fn invoke_async_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            unreachable!("stub; never executed")
        }
    }

    fn fake_async_udf() -> ScalarUDF {
        AsyncScalarUDF::new(Arc::new(FakeAsyncUdf {
            signature: Signature::exact(vec![DataType::Int64], Volatility::Stable),
        }))
        .into_scalar_udf()
    }

    fn fake_async(arg: Expr) -> Expr {
        Expr::ScalarFunction(ScalarFunction {
            func: Arc::new(fake_async_udf()),
            args: vec![arg],
        })
    }

    fn scan(name: &str, column: &str) -> LogicalPlan {
        let schema = Schema::new(vec![Field::new(column, DataType::Int64, true)]);
        let schema = DFSchema::try_from_qualified_schema(name, &schema).unwrap();
        LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(schema),
        })
    }

    fn inner_join_with_filter(filter: Expr) -> LogicalPlan {
        LogicalPlanBuilder::from(scan("a", "x"))
            .join_on(scan("b", "y"), JoinType::Inner, [filter])
            .unwrap()
            .build()
            .unwrap()
    }

    fn rewrite(plan: LogicalPlan) -> Transformed<LogicalPlan> {
        HoistAsyncJoinFilterRule
            .rewrite(plan, &OptimizerContext::new())
            .unwrap()
    }

    #[test]
    fn hoists_async_conjuncts_above_the_join() {
        let async_pred = fake_async(col("a.x")).gt(col("b.y"));
        let sync_pred = col("a.x").not_eq(col("b.y"));
        let out = rewrite(inner_join_with_filter(
            async_pred.clone().and(sync_pred.clone()),
        ));

        assert!(out.transformed);
        let LogicalPlan::Filter(filter) = &out.data else {
            panic!(
                "expected Filter above the join, got {}",
                out.data.display_indent()
            );
        };
        assert_eq!(filter.predicate, async_pred);
        let LogicalPlan::Join(join) = filter.input.as_ref() else {
            panic!(
                "expected the join under the Filter, got {}",
                out.data.display_indent()
            );
        };
        assert_eq!(join.filter, Some(sync_pred));
    }

    #[test]
    fn hoists_the_whole_filter_when_every_conjunct_is_async() {
        let async_pred = fake_async(col("a.x")).gt(col("b.y"));
        let out = rewrite(inner_join_with_filter(async_pred.clone()));

        let LogicalPlan::Filter(filter) = &out.data else {
            panic!(
                "expected Filter above the join, got {}",
                out.data.display_indent()
            );
        };
        assert_eq!(filter.predicate, async_pred);
        let LogicalPlan::Join(join) = filter.input.as_ref() else {
            panic!("expected the join under the Filter");
        };
        assert_eq!(join.filter, None, "the join keeps no filter");
    }

    #[test]
    fn leaves_sync_join_filters_alone() {
        let out = rewrite(inner_join_with_filter(col("a.x").gt(col("b.y"))));
        assert!(!out.transformed);
        assert!(matches!(out.data, LogicalPlan::Join(_)));
    }

    #[test]
    fn leaves_outer_join_filters_alone() {
        // The filter of an outer join decides which rows get null-extended;
        // it is not a post-join predicate and must stay put.
        let plan = LogicalPlanBuilder::from(scan("a", "x"))
            .join_on(
                scan("b", "y"),
                JoinType::Left,
                [fake_async(col("a.x")).gt(col("b.y"))],
            )
            .unwrap()
            .build()
            .unwrap();
        let out = rewrite(plan);
        assert!(!out.transformed);
        assert!(matches!(out.data, LogicalPlan::Join(_)));
    }

    #[test]
    fn preserves_the_join_schema() {
        let out = rewrite(inner_join_with_filter(fake_async(col("a.x")).gt(lit(0))));
        let expected = inner_join_with_filter(lit(true));
        assert_eq!(out.data.schema(), expected.schema());
    }

    /// The reported shape through the whole optimizer, to its fixpoint: a
    /// WHERE over the output of an async call over a join. PushDownFilter
    /// moves it into the join filter, this rule moves it back out, and the
    /// physical planner then hoists the call into an AsyncFuncExec instead of
    /// leaving it in the join operator.
    #[tokio::test]
    async fn optimizer_keeps_async_predicate_above_the_join() {
        use crate::optimizer::register_ensure_loaded_optimizer;
        use datafusion::catalog::MemTable;
        use datafusion::execution::session_state::SessionStateBuilder;
        use datafusion::physical_plan::displayable;
        use datafusion::prelude::SessionContext;

        let builder =
            register_ensure_loaded_optimizer(SessionStateBuilder::new().with_default_features())
                .unwrap();
        let ctx = SessionContext::new_with_state(builder.build());
        ctx.register_udf(fake_async_udf());
        for (name, column) in [("a", "x"), ("b", "y")] {
            let schema = Arc::new(Schema::new(vec![Field::new(column, DataType::Int64, true)]));
            let table = MemTable::try_new(schema, vec![vec![]]).unwrap();
            ctx.register_table(name, Arc::new(table)).unwrap();
        }

        let df = ctx
            .sql(
                "SELECT x, v FROM (SELECT a.x, b.y, fake_async(a.x) AS v FROM a, b) \
                 WHERE v > y",
            )
            .await
            .unwrap();

        // Independent of `calls_async_udf`: look for the call by name.
        let optimized = df.clone().into_optimized_plan().unwrap();
        let mut async_in_join_filter = false;
        optimized
            .apply(|node| {
                if let LogicalPlan::Join(join) = node
                    && join
                        .filter
                        .as_ref()
                        .is_some_and(|filter| format!("{filter}").contains("fake_async"))
                {
                    async_in_join_filter = true;
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
        assert!(
            !async_in_join_filter,
            "async call must not stay in the join filter: {}",
            optimized.display_indent()
        );

        // Every physical mention of the call is inside an AsyncFuncExec: not
        // the join operator, and not the projection DataFusion pushes below a
        // nested-loop join for its filter's sub-expressions.
        let physical = df.create_physical_plan().await.unwrap();
        let physical_str = displayable(physical.as_ref()).indent(true).to_string();
        let mentions: Vec<&str> = physical_str
            .lines()
            .filter(|line| line.contains("fake_async"))
            .collect();
        assert!(
            !mentions.is_empty() && mentions.iter().all(|line| line.contains("AsyncFuncExec")),
            "{physical_str}"
        );
    }
}
