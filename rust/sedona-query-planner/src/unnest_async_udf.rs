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

//! Logical optimizer rule that moves an async scalar UDF call nested inside
//! another async call's arguments into a projection below.
//!
//! DataFusion's physical planner evaluates the async calls of a `Projection`,
//! `Filter` or `Aggregate` in an `AsyncFuncExec` below the node, but it can't
//! plan one async call inside another's arguments: only the innermost is
//! extracted, the outer is invoked synchronously and fails with "async
//! functions should not be called directly" (apache/datafusion#20031). So for
//!
//! ```text
//! Projection: rs_value(rs_ensureloaded(rs_setsrid(rs_ensureloaded(rast), 4326)), ..)
//! ```
//!
//! this rule computes the inner call in a projection of its own and refers to
//! its output column instead:
//!
//! ```text
//! Projection: rs_value(rs_ensureloaded(rs_setsrid(__sd_async_0, 4326)), ..)
//!   Projection: <input columns>, rs_ensureloaded(rast) AS __sd_async_0
//! ```
//!
//! A `Filter` gets a projection above it as well, restoring its schema.
//! Identical nested calls in one node share a column, and calls nested
//! several levels deep produce one projection per level. The replaced call is
//! aliased to its own name, so no enclosing expression is renamed. A call
//! wrapped in `sd_restore_metadata` (see [`crate::wrap_async_udf`]) moves
//! together with the wrapper, so the column keeps its extension metadata.
//!
//! Evaluation: `AsyncFuncExec` already evaluates every async call of the node
//! for every input row, including calls in a `CASE` branch, the right side of
//! `AND`/`OR` or a lazily evaluated argument. The projection below sees the
//! same rows, so moving a call there evaluates it exactly as often as
//! DataFusion would if it weren't nested, and can't raise an error the node
//! wouldn't raise. Join filters are left alone: the physical planner plans no
//! async calls there at all.
//!
//! Ordering: register it after `OptimizeProjections`, which would merge the
//! single-use projection straight back into the node. On the next optimizer
//! pass that merge happens and this rule splits the node again the same way
//! (column names are deterministic), so the plan signature check ends the
//! fixpoint with the nesting removed.

use std::sync::Arc;

use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{Column, DFSchema, Result};
use datafusion_expr::{Expr, LogicalPlan, Projection};
use datafusion_optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

use crate::restore_metadata::RESTORE_METADATA_NAME;

/// Prefix of the columns this rule adds.
const COLUMN_PREFIX: &str = "__sd_async_";

/// Logical optimizer rule moving nested async scalar UDF calls into a
/// projection below. See the module docs.
#[derive(Default, Debug)]
pub struct UnnestAsyncUdfRule;

impl OptimizerRule for UnnestAsyncUdfRule {
    fn name(&self) -> &str {
        "sedona.unnest_async_udf"
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
        unnest(plan, &mut 0)
    }
}

/// Split `plan` (a `Projection`, `Filter` or `Aggregate`) so that no async
/// call in its expressions has another async call in its arguments. `next` is
/// the first column number to try, shared across the levels of one node so
/// their columns never collide.
fn unnest(plan: LogicalPlan, next: &mut usize) -> Result<Transformed<LogicalPlan>> {
    if !matches!(
        plan,
        LogicalPlan::Projection(_) | LogicalPlan::Filter(_) | LogicalPlan::Aggregate(_)
    ) {
        return Ok(Transformed::no(plan));
    }

    let mut nested = Vec::new();
    for expr in plan.expressions() {
        collect_nested(&expr, false, &mut nested)?;
    }
    if nested.is_empty() {
        return Ok(Transformed::no(plan));
    }

    let input = plan.inputs()[0].clone();
    let names: Vec<String> = nested
        .iter()
        .map(|_| fresh_name(input.schema(), next))
        .collect();

    let mut child_exprs: Vec<Expr> = input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    child_exprs.extend(
        nested
            .iter()
            .zip(&names)
            .map(|(call, name)| call.clone().alias(name)),
    );
    let child = LogicalPlan::Projection(Projection::try_new(child_exprs, Arc::new(input))?);
    // A nested call may itself contain nested calls; split the child too.
    let child = unnest(child, next)?.data;

    let exprs = plan
        .expressions()
        .into_iter()
        .map(|expr| replace_nested(expr, &nested, &names))
        .collect::<Result<Vec<_>>>()?;
    let original_schema = Arc::clone(plan.schema());
    let rewritten = plan.with_new_exprs(exprs, vec![child])?;

    let rewritten = match rewritten {
        // A Filter passes the new columns through; project them away again.
        LogicalPlan::Filter(_) => {
            let columns = original_schema.columns().into_iter().map(Expr::Column);
            LogicalPlan::Projection(Projection::try_new(columns.collect(), Arc::new(rewritten))?)
        }
        other => other,
    };
    Ok(Transformed::yes(rewritten))
}

/// The arguments of `expr` if it is an async call, or an async call wrapped in
/// `sd_restore_metadata`. The two move as one unit.
fn async_call_args(expr: &Expr) -> Option<&[Expr]> {
    let Expr::ScalarFunction(call) = expr else {
        return None;
    };
    if call.func.as_async().is_some() {
        return Some(&call.args);
    }
    if call.func.name() == RESTORE_METADATA_NAME
        && let [Expr::ScalarFunction(inner)] = call.args.as_slice()
        && inner.func.as_async().is_some()
    {
        return Some(&inner.args);
    }
    None
}

/// Collect, without duplicates, the outermost async calls of `expr` that sit
/// inside another async call's arguments. Calls nested further inside those
/// are left for the projection they move to.
fn collect_nested(expr: &Expr, in_async_args: bool, out: &mut Vec<Expr>) -> Result<()> {
    if let Some(args) = async_call_args(expr) {
        if in_async_args {
            if !out.contains(expr) {
                out.push(expr.clone());
            }
        } else {
            for arg in args {
                collect_nested(arg, true, out)?;
            }
        }
        return Ok(());
    }
    expr.apply_children(|child| {
        collect_nested(child, in_async_args, out)?;
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(())
}

/// Replace every occurrence of a `nested` call in `expr` with a reference to
/// its column, aliased to the call's name so enclosing names don't change.
fn replace_nested(expr: Expr, nested: &[Expr], names: &[String]) -> Result<Expr> {
    let column = |call: &Expr| {
        nested
            .iter()
            .position(|n| n == call)
            .map(|i| Expr::Column(Column::new_unqualified(&names[i])))
    };
    expr.transform_down(|expr| {
        if let Expr::Alias(mut alias) = expr {
            // Keep the alias (and its name); replace what it names.
            let Some(column) = column(&alias.expr) else {
                return Ok(Transformed::no(Expr::Alias(alias)));
            };
            alias.expr = Box::new(column);
            return Ok(Transformed::new(
                Expr::Alias(alias),
                true,
                TreeNodeRecursion::Jump,
            ));
        }
        if let Some(column) = column(&expr) {
            let name = expr.schema_name().to_string();
            return Ok(Transformed::new(
                column.alias(name),
                true,
                TreeNodeRecursion::Jump,
            ));
        }
        Ok(Transformed::no(expr))
    })
    .map(|t| t.data)
}

/// The first `__sd_async_N` (from `*next` on) that `schema` doesn't use.
fn fresh_name(schema: &DFSchema, next: &mut usize) -> String {
    loop {
        let name = format!("{COLUMN_PREFIX}{next}");
        *next += 1;
        if !schema.fields().iter().any(|f| f.name() == &name) {
            return name;
        }
    }
}

/// Whether `expr` has an async call inside another async call's arguments.
#[cfg(test)]
pub(crate) fn has_nested_async_call(expr: &Expr) -> bool {
    let mut nested = Vec::new();
    collect_nested(expr, false, &mut nested).unwrap();
    !nested.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::hash::{Hash, Hasher};

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use datafusion::catalog::MemTable;
    use datafusion::execution::session_state::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
    use datafusion_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
    };

    use crate::optimizer::register_ensure_loaded_optimizer;

    /// An async `Int64 -> Int64` UDF adding one, so queries can run.
    #[derive(Debug)]
    struct AsyncAddOne {
        signature: Signature,
    }

    impl PartialEq for AsyncAddOne {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl Eq for AsyncAddOne {}
    impl Hash for AsyncAddOne {
        fn hash<H: Hasher>(&self, state: &mut H) {
            "add_one".hash(state);
        }
    }

    impl ScalarUDFImpl for AsyncAddOne {
        fn name(&self) -> &str {
            "add_one"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int64)
        }

        fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            unreachable!("async UDF; DataFusion calls invoke_async_with_args")
        }
    }

    #[async_trait]
    impl AsyncScalarUDFImpl for AsyncAddOne {
        async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            let arrays = ColumnarValue::values_to_arrays(&args.args)?;
            let out: Int64Array = arrays[0].as_primitive::<Int64Type>().unary(|v| v + 1);
            Ok(ColumnarValue::Array(Arc::new(out)))
        }
    }

    /// A context with the Sedona optimizer rules, `add_one`, and a table `t`
    /// with the Int64 columns `columns`, holding `values` in each.
    fn context(columns: &[&str], values: Vec<i64>) -> SessionContext {
        let builder =
            register_ensure_loaded_optimizer(SessionStateBuilder::new().with_default_features())
                .unwrap();
        let ctx = SessionContext::new_with_state(builder.build());
        ctx.register_udf(
            AsyncScalarUDF::new(Arc::new(AsyncAddOne {
                signature: Signature::exact(vec![DataType::Int64], Volatility::Stable),
            }))
            .into_scalar_udf(),
        );
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|name| Field::new(*name, DataType::Int64, false))
                .collect::<Vec<_>>(),
        ));
        let array: ArrayRef = Arc::new(Int64Array::from(values));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&array); columns.len()])
                .unwrap();
        let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
        ctx.register_table("t", Arc::new(table)).unwrap();
        ctx
    }

    /// Optimize and run `sql`, asserting the optimized plan has no nested
    /// async call. Returns the plan and the first output column.
    async fn run(ctx: &SessionContext, sql: &str) -> (LogicalPlan, Vec<Option<i64>>) {
        let df = ctx.sql(sql).await.unwrap();
        let plan = df.clone().into_optimized_plan().unwrap();
        plan.apply(|node| {
            node.apply_expressions(|expr| {
                assert!(
                    !has_nested_async_call(expr),
                    "nested async call left in:\n{}",
                    plan.display_indent()
                );
                Ok(TreeNodeRecursion::Continue)
            })
        })
        .unwrap();
        let batches = df.collect().await.unwrap();
        let values = batches
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().iter())
            .collect();
        (plan, values)
    }

    /// The columns this rule defined in `plan`.
    fn hoisted_columns(plan: &LogicalPlan) -> Vec<String> {
        let mut names = vec![];
        plan.apply(|node| {
            if let LogicalPlan::Projection(projection) = node {
                names.extend(projection.expr.iter().filter_map(|expr| match expr {
                    Expr::Alias(alias) if alias.name.starts_with(COLUMN_PREFIX) => {
                        Some(alias.name.clone())
                    }
                    _ => None,
                }));
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        names
    }

    #[tokio::test]
    async fn projection() {
        let ctx = context(&["x"], vec![-5, 7]);
        let (plan, values) = run(&ctx, "SELECT add_one(abs(add_one(x))) AS v FROM t").await;
        assert_eq!(values, vec![Some(5), Some(9)]);
        assert_eq!(hoisted_columns(&plan), vec!["__sd_async_0"]);
    }

    #[tokio::test]
    async fn keeps_the_output_names() {
        let ctx = context(&["x"], vec![1]);
        let df = ctx.sql("SELECT add_one(add_one(x)) FROM t").await.unwrap();
        let before = df.logical_plan().schema().as_ref().clone();
        let after = df.into_optimized_plan().unwrap().schema().as_ref().clone();
        assert_eq!(before, after);
        assert_eq!(after.field(0).name(), "add_one(add_one(t.x))");
    }

    #[tokio::test]
    async fn several_levels() {
        let ctx = context(&["x"], vec![-5, 7]);
        let (plan, values) = run(&ctx, "SELECT add_one(add_one(add_one(x))) AS v FROM t").await;
        assert_eq!(values, vec![Some(-2), Some(10)]);
        assert_eq!(hoisted_columns(&plan).len(), 2, "{}", plan.display_indent());
    }

    #[tokio::test]
    async fn identical_calls_share_a_column() {
        // Applied to the unoptimized plan, so CSE hasn't deduped them first.
        let ctx = context(&["x"], vec![1]);
        let plan = ctx
            .sql("SELECT add_one(abs(add_one(x))) + add_one(add_one(x) * 2) AS v FROM t")
            .await
            .unwrap()
            .into_unoptimized_plan();
        let out = plan
            .transform_up(|node| UnnestAsyncUdfRule.rewrite(node, &ctx.state()))
            .unwrap()
            .data;
        assert_eq!(
            hoisted_columns(&out),
            vec!["__sd_async_0"],
            "{}",
            out.display_indent()
        );
        let results = ctx
            .execute_logical_plan(out)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(
            results[0].column(0).as_primitive::<Int64Type>().value(0),
            3 + 5
        );
    }

    #[tokio::test]
    async fn conditional_positions() {
        // The nested calls sit in a CASE branch and on the right of an AND.
        let ctx = context(&["x"], vec![-5, 7]);
        let (_, values) = run(
            &ctx,
            "SELECT CASE WHEN x > 0 AND add_one(add_one(x)) > 0 \
             THEN add_one(abs(add_one(x))) ELSE 0 END AS v FROM t",
        )
        .await;
        assert_eq!(values, vec![Some(0), Some(9)]);
    }

    #[tokio::test]
    async fn filter() {
        let ctx = context(&["x"], vec![-5, 7]);
        let df = ctx
            .sql("SELECT x FROM t WHERE add_one(add_one(x)) > 0")
            .await
            .unwrap();
        let before = df.logical_plan().schema().as_ref().clone();
        let (plan, values) = run(&ctx, "SELECT x FROM t WHERE add_one(add_one(x)) > 0").await;
        assert_eq!(values, vec![Some(7)]);
        assert_eq!(plan.schema().as_ref(), &before);
    }

    #[tokio::test]
    async fn filter_on_one_side_of_a_join() {
        let ctx = context(&["x"], vec![-5, 7]);
        let (_, values) = run(
            &ctx,
            "SELECT a.x FROM t a JOIN t b ON a.x = b.x WHERE add_one(add_one(a.x)) > 0",
        )
        .await;
        assert_eq!(values, vec![Some(7)]);
    }

    #[tokio::test]
    async fn aggregate() {
        let ctx = context(&["x"], vec![-5, 7]);
        let (_, values) = run(&ctx, "SELECT sum(add_one(add_one(x))) AS v FROM t").await;
        assert_eq!(values, vec![Some(-3 + 9)]);
    }

    #[tokio::test]
    async fn skips_a_name_the_input_uses() {
        let ctx = context(&["__sd_async_0"], vec![-5, 7]);
        let (plan, values) = run(
            &ctx,
            "SELECT __sd_async_0 + add_one(add_one(__sd_async_0)) AS v FROM t",
        )
        .await;
        assert_eq!(values, vec![Some(-5 - 3), Some(7 + 9)]);
        assert!(
            hoisted_columns(&plan).contains(&"__sd_async_1".to_string()),
            "{}",
            plan.display_indent()
        );
    }

    #[tokio::test]
    async fn rewrite_is_idempotent() {
        let ctx = context(&["x"], vec![1]);
        let plan = ctx
            .sql("SELECT add_one(abs(add_one(x))) AS v FROM t WHERE add_one(add_one(x)) > 0")
            .await
            .unwrap()
            .into_optimized_plan()
            .unwrap();
        let again = plan
            .clone()
            .transform_up(|node| UnnestAsyncUdfRule.rewrite(node, &ctx.state()))
            .unwrap();
        assert!(!again.transformed, "{}", again.data.display_indent());
    }

    #[tokio::test]
    async fn leaves_a_join_filter_alone() {
        // The physical planner plans no async calls in a join filter, so there
        // is nothing to split there.
        let ctx = context(&["x"], vec![1]);
        let df = ctx
            .sql("SELECT a.x FROM t a JOIN t b ON add_one(add_one(a.x)) > b.x")
            .await
            .unwrap();
        let mut join = None;
        df.logical_plan()
            .apply(|node| {
                if let LogicalPlan::Join(j) = node {
                    assert!(j.filter.is_some());
                    join = Some(node.clone());
                }
                Ok(TreeNodeRecursion::Continue)
            })
            .unwrap();
        let out = UnnestAsyncUdfRule
            .rewrite(join.unwrap(), &ctx.state())
            .unwrap();
        assert!(!out.transformed);
    }
}
