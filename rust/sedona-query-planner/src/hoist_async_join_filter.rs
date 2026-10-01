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
//! Only conjuncts whose async calls are evaluated unconditionally move.
//! `AsyncFuncExec` evaluates an async call for every row of every batch,
//! whereas the synchronous evaluator the join uses skips some sub-expressions:
//! every `CASE` branch after the first `WHEN` condition runs only on the rows
//! that reach it, the right side of `AND`/`OR` is short-circuited, and a
//! `ScalarUDF` with `short_circuits()` declares arguments that may not be
//! evaluated. Hoisting an async call from such a position would run it for
//! rows the sync path never touches: extra I/O at best (`RS_EnsureLoaded`
//! fetching rasters an unreached `ELSE` names), an error from those rows at
//! worst. A conjunct with a conditionally evaluated async call therefore
//! stays in the join filter, where it behaves as before this rule: fine while
//! the branch is never taken, and DataFusion's "async functions should not be
//! called directly" error once it is. Evaluating async calls under a selection
//! is DataFusion's to add (apache/datafusion#16520, "Join expression").
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
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_expr::async_udf::AsyncScalarUDF;
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::utils::{conjunction, split_conjunction_owned};
use datafusion_expr::{BinaryExpr, Expr, Filter, JoinType, LogicalPlan, Operator};
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
            .partition(hoistable);
        let Some(hoisted) = conjunction(hoist) else {
            join.filter = Some(filter);
            return Ok(Transformed::no(LogicalPlan::Join(join)));
        };

        join.filter = conjunction(keep);
        let filter = Filter::try_new(hoisted, Arc::new(LogicalPlan::Join(join)))?;
        Ok(Transformed::yes(LogicalPlan::Filter(filter)))
    }
}

/// A conjunct moves above the join when it calls an async UDF and every such
/// call is evaluated unconditionally, so evaluating it for every row (as
/// `AsyncFuncExec` does) is what the join would have done anyway.
fn hoistable(expr: &Expr) -> bool {
    calls_async_udf(expr) && !has_conditional_async_call(expr)
}

fn is_async(call: &ScalarFunction) -> bool {
    call.func.inner().downcast_ref::<AsyncScalarUDF>().is_some()
}

/// True if any call anywhere in `expr` is to an [`AsyncScalarUDF`].
fn calls_async_udf(expr: &Expr) -> bool {
    expr.exists(|e| Ok(matches!(e, Expr::ScalarFunction(call) if is_async(call))))
        .unwrap_or(false)
}

/// True if `expr` contains an async UDF call that the synchronous evaluator
/// may skip for some rows or batches. Mirrors DataFusion's physical
/// evaluators: `CaseExpr` runs the base expression and the first `WHEN`
/// condition on the whole batch and every other branch on the rows that
/// reach it; `BinaryExpr` short-circuits the right side of `AND`/`OR`; a
/// `ScalarUDF` with `short_circuits()` reports its lazily evaluated arguments
/// through `conditional_arguments()`.
fn has_conditional_async_call(expr: &Expr) -> bool {
    fn walk(expr: &Expr, conditional: bool) -> Result<bool> {
        match expr {
            Expr::ScalarFunction(call) if is_async(call) && conditional => Ok(true),
            Expr::Case(case) => {
                if let Some(base) = &case.expr
                    && walk(base, conditional)?
                {
                    return Ok(true);
                }
                for (i, (when, then)) in case.when_then_expr.iter().enumerate() {
                    // With a base expression the first WHEN value is compared
                    // only on rows where the base is not null.
                    let when_conditional = conditional || case.expr.is_some() || i > 0;
                    if walk(when, when_conditional)? || walk(then, true)? {
                        return Ok(true);
                    }
                }
                match &case.else_expr {
                    Some(else_expr) => walk(else_expr, true),
                    None => Ok(false),
                }
            }
            Expr::BinaryExpr(BinaryExpr {
                left,
                op: Operator::And | Operator::Or,
                right,
            }) => Ok(walk(left, conditional)? || walk(right, true)?),
            Expr::ScalarFunction(call) if call.func.short_circuits() => {
                let (eager, lazy) = call
                    .func
                    .conditional_arguments(&call.args)
                    .unwrap_or_else(|| (call.args.iter().collect(), vec![]));
                for arg in eager {
                    if walk(arg, conditional)? {
                        return Ok(true);
                    }
                }
                for arg in lazy {
                    if walk(arg, true)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            other => {
                let mut found = false;
                other.apply_children(|child| {
                    found = walk(child, conditional)?;
                    Ok(if found {
                        TreeNodeRecursion::Stop
                    } else {
                        TreeNodeRecursion::Continue
                    })
                })?;
                Ok(found)
            }
        }
    }
    // The walk cannot fail; if it somehow did, keeping the conjunct in the
    // join is the behaviour-preserving answer.
    walk(expr, false).unwrap_or(true)
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

    /// `CASE WHEN a.x < b.y THEN 1 ELSE fake_async(a.x + b.y) END > 0`: the
    /// async call sits in an `ELSE` the sync evaluator reaches only for rows
    /// where the condition is false.
    fn case_guarded_async_pred() -> Expr {
        Expr::Case(datafusion_expr::expr::Case::new(
            None,
            vec![(Box::new(col("a.x").lt(col("b.y"))), Box::new(lit(1_i64)))],
            Some(Box::new(fake_async(col("a.x") + col("b.y")))),
        ))
        .gt(lit(0_i64))
    }

    fn join_filter_of(plan: &LogicalPlan) -> Option<Expr> {
        match plan {
            LogicalPlan::Join(join) => join.filter.clone(),
            LogicalPlan::Filter(filter) => join_filter_of(&filter.input),
            other => panic!("unexpected plan shape: {}", other.display_indent()),
        }
    }

    #[test]
    fn keeps_conditionally_evaluated_async_calls_in_the_join() {
        // The CASE conjunct stays in the join; the unconditional one moves.
        let unconditional = fake_async(col("a.x")).gt(col("b.y"));
        let out = rewrite(inner_join_with_filter(
            case_guarded_async_pred().and(unconditional.clone()),
        ));

        assert!(out.transformed);
        let LogicalPlan::Filter(filter) = &out.data else {
            panic!(
                "expected Filter above the join, got {}",
                out.data.display_indent()
            );
        };
        assert_eq!(filter.predicate, unconditional);
        assert_eq!(join_filter_of(&out.data), Some(case_guarded_async_pred()));
    }

    #[test]
    fn leaves_a_join_whose_only_async_call_is_conditional_alone() {
        let out = rewrite(inner_join_with_filter(case_guarded_async_pred()));
        assert!(!out.transformed);
        assert!(matches!(out.data, LogicalPlan::Join(_)));
    }

    #[test]
    fn hoists_an_async_call_in_the_first_when_condition() {
        // The first WHEN condition is evaluated on the whole batch.
        let pred = Expr::Case(datafusion_expr::expr::Case::new(
            None,
            vec![(
                Box::new(fake_async(col("a.x")).gt(col("b.y"))),
                Box::new(lit(true)),
            )],
            Some(Box::new(lit(false))),
        ));
        let out = rewrite(inner_join_with_filter(pred.clone()));
        assert!(out.transformed);
        let LogicalPlan::Filter(filter) = &out.data else {
            panic!(
                "expected Filter above the join, got {}",
                out.data.display_indent()
            );
        };
        assert_eq!(filter.predicate, pred);
    }

    #[test]
    fn treats_the_right_side_of_and_or_as_conditional() {
        // Left side: evaluated for every row, hoisted.
        let left = fake_async(col("a.x"))
            .gt(col("b.y"))
            .or(col("a.x").gt(col("b.y")));
        assert!(rewrite(inner_join_with_filter(left)).transformed);
        // Right side: short-circuited by the sync evaluator, kept.
        let right = col("a.x")
            .gt(col("b.y"))
            .or(fake_async(col("a.x")).gt(col("b.y")));
        assert!(!rewrite(inner_join_with_filter(right)).transformed);
    }

    #[test]
    fn treats_lazy_arguments_of_short_circuit_udfs_as_conditional() {
        use datafusion::functions::core::coalesce;

        // coalesce declares its first argument eager and the rest lazy.
        let first = Expr::ScalarFunction(ScalarFunction {
            func: coalesce(),
            args: vec![fake_async(col("a.x")), lit(0_i64)],
        })
        .gt(col("b.y"));
        assert!(rewrite(inner_join_with_filter(first)).transformed);
        let second = Expr::ScalarFunction(ScalarFunction {
            func: coalesce(),
            args: vec![col("a.x"), fake_async(col("a.x"))],
        })
        .gt(col("b.y"));
        assert!(!rewrite(inner_join_with_filter(second)).transformed);
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

    /// Regression from review: an async call in a CASE branch the sync
    /// evaluator never takes must not run. Before the conditional check the
    /// whole conjunct was hoisted and `AsyncFuncExec` evaluated the call for
    /// all four pairs, hitting the stub's `unreachable!`; in the join filter
    /// the `ELSE` is never reached and the query returns all four rows.
    #[tokio::test]
    async fn case_keeps_unselected_async_branch() {
        use arrow::{array::Int64Array, record_batch::RecordBatch};
        use datafusion::{
            catalog::MemTable, execution::session_state::SessionStateBuilder,
            prelude::SessionContext,
        };

        let builder = crate::optimizer::register_ensure_loaded_optimizer(
            SessionStateBuilder::new().with_default_features(),
        )
        .unwrap();
        let ctx = SessionContext::new_with_state(builder.build());
        ctx.register_udf(fake_async_udf());
        for (name, column, values) in [("a", "x", vec![1_i64, 2]), ("b", "y", vec![10_i64, 20])] {
            let schema = Arc::new(Schema::new(vec![Field::new(
                column,
                DataType::Int64,
                false,
            )]));
            let batch =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))])
                    .unwrap();
            ctx.register_table(
                name,
                Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap()),
            )
            .unwrap();
        }
        let batches = ctx
            .sql(
                "SELECT a.x, b.y FROM a JOIN b ON \
                 CASE WHEN a.x < b.y THEN CAST(1 AS BIGINT) \
                 ELSE fake_async(a.x + b.y) END > 0",
            )
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);
    }
}
