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

use std::sync::Arc;

use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::output_requirements::{
    OutputRequirementExec, OutputRequirements,
};
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_physical_expr::{Distribution, LexOrdering, OrderingRequirements};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::projection::{ProjectionExec, update_ordering};
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

/// Keep a global sort's requirements above order-preserving projections.
///
/// DataFusion adds its output requirement at the sort, allowing a computed
/// projection above it to repartition the sorted output. When the output still
/// exposes all the sort expressions, require them at the root instead, mapping
/// them through projections so renamed or reordered columns have the correct
/// indices.
#[derive(Debug)]
pub(crate) struct PreserveGlobalOrder;

impl PhysicalOptimizerRule for PreserveGlobalOrder {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if plan.downcast_ref::<OutputRequirementExec>().is_none()
            && plan.downcast_ref::<SortExec>().is_none()
            && plan.downcast_ref::<SortPreservingMergeExec>().is_none()
            && let Some(ordering) = projected_global_order(&plan)?
        {
            return Ok(Arc::new(OutputRequirementExec::new(
                plan,
                Some(OrderingRequirements::from(ordering)),
                Distribution::SinglePartition,
                None,
            )));
        }

        OutputRequirements::new_add_mode().optimize(plan, config)
    }

    fn name(&self) -> &str {
        "sedona.preserve_global_order"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn projected_global_order(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<LexOrdering>> {
    if let Some(sort) = plan.downcast_ref::<SortExec>() {
        return Ok((!sort.preserve_partitioning()).then(|| sort.expr().clone()));
    }
    if let Some(sort) = plan.downcast_ref::<SortPreservingMergeExec>() {
        return Ok(Some(sort.expr().clone()));
    }

    let children = plan.children();
    if children.len() == 1
        && plan.maintains_input_order()[0]
        && plan.required_input_ordering()[0]
            .as_ref()
            .is_none_or(|order| matches!(order, OrderingRequirements::Soft(_)))
    {
        let Some(ordering) = projected_global_order(children[0])? else {
            return Ok(None);
        };
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            return update_ordering(ordering, projection.expr());
        }
        if plan.schema() == children[0].schema() {
            return Ok(Some(ordering));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int64Array, RecordBatch};
    use datafusion::dataframe::DataFrame;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::{SessionConfig, SessionContext, col, lit};
    use datafusion_common::config::ConfigOptions;
    use datafusion_physical_expr::expressions::{Column, Literal};
    use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties, collect};

    use super::*;
    use crate::optimizer::register_vendored_optimizer_rules;

    fn context() -> SessionContext {
        let config = SessionConfig::new()
            .with_target_partitions(4)
            .with_batch_size(128);
        let state = register_vendored_optimizer_rules(
            SessionStateBuilder::new_with_default_features().with_config(config),
        )
        .unwrap()
        .build();
        SessionContext::new_with_state(state)
    }

    fn numbers(ctx: &SessionContext) -> DataFrame {
        let batch = RecordBatch::try_from_iter(vec![(
            "a",
            Arc::new(Int64Array::from_iter_values(0..10000)) as ArrayRef,
        )])
        .unwrap();
        ctx.read_batch(batch).unwrap()
    }

    async fn assert_values(df: DataFrame, ctx: &SessionContext, column: usize, expected: Vec<i64>) {
        let plan = df.create_physical_plan().await.unwrap();
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        let batches = collect(plan, ctx.task_ctx()).await.unwrap();
        let actual: Vec<_> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(column)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn sorted_projection_preserves_global_order() {
        let ctx = context();
        let sorted = numbers(&ctx)
            .sort(vec![col("a").sort(false, false)])
            .unwrap();
        let projected = sorted
            .select(vec![col("a"), (col("a") + lit(1)).alias("b")])
            .unwrap();
        assert_values(projected, &ctx, 0, (0..10000).rev().collect()).await;
    }

    #[tokio::test]
    async fn sorted_projection_can_rename_and_reorder_columns() {
        let ctx = context();
        let projected = numbers(&ctx)
            .sort(vec![col("a").sort(false, false)])
            .unwrap()
            .select(vec![(col("a") + lit(1)).alias("b"), col("a").alias("key")])
            .unwrap()
            .filter(col("b").gt(lit(10)))
            .unwrap();
        assert_values(projected, &ctx, 1, (10..10000).rev().collect()).await;
    }

    #[tokio::test]
    async fn sorted_projection_preserves_limit_and_offset() {
        let ctx = context();
        let projected = numbers(&ctx)
            .sort(vec![col("a").sort(false, false)])
            .unwrap()
            .limit(5, Some(10))
            .unwrap()
            .select(vec![col("a"), (col("a") + lit(1)).alias("b")])
            .unwrap();
        assert_values(projected, &ctx, 0, (9985..9995).rev().collect()).await;
    }

    #[tokio::test]
    async fn sorted_projection_preserves_multiple_sort_keys_and_nulls() {
        let ctx = context();
        let groups = Int64Array::from_iter(
            (0..10000).map(|i| if i % 11 == 0 { None } else { Some(i % 10) }),
        );
        let batch = RecordBatch::try_from_iter(vec![
            ("group", Arc::new(groups) as ArrayRef),
            (
                "a",
                Arc::new(Int64Array::from_iter_values(0..10000)) as ArrayRef,
            ),
        ])
        .unwrap();
        let projected = ctx
            .read_batch(batch)
            .unwrap()
            .sort(vec![
                col("group").sort(true, true),
                col("a").sort(false, false),
            ])
            .unwrap()
            .select(vec![
                (col("a") + lit(1)).alias("b"),
                col("a").alias("key"),
                col("group").alias("renamed_group"),
            ])
            .unwrap();
        let mut expected: Vec<i64> = (0..10000).collect();
        expected.sort_by_key(|&i| {
            (
                if i % 11 == 0 { None } else { Some(i % 10) },
                std::cmp::Reverse(i),
            )
        });
        assert_values(projected, &ctx, 1, expected).await;
    }

    #[tokio::test]
    async fn unordered_projection_still_runs_in_parallel() {
        let ctx = context();
        let plan = numbers(&ctx)
            .select(vec![col("a"), (col("a") + lit(1)).alias("b")])
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        assert_eq!(plan.output_partitioning().partition_count(), 4);
    }

    async fn projected_sort(
        keep_sort_columns: usize,
        preserve_partitioning: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let ctx = context();
        let input = numbers(&ctx)
            .select(vec![col("a"), (col("a") + lit(1)).alias("b")])
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let a = Arc::new(Column::new("a", 0)) as Arc<dyn PhysicalExpr>;
        let b = Arc::new(Column::new("b", 1)) as Arc<dyn PhysicalExpr>;
        let sort = Arc::new(
            SortExec::new(
                LexOrdering::new(vec![
                    PhysicalSortExpr::new_default(Arc::clone(&a)),
                    PhysicalSortExpr::new_default(Arc::clone(&b)),
                ])
                .unwrap(),
                input,
            )
            .with_preserve_partitioning(preserve_partitioning),
        );
        let mut expr = vec![(a, "key".to_string()), (b, "other_key".to_string())];
        expr.truncate(keep_sort_columns);
        if expr.is_empty() {
            expr.push((Arc::new(Literal::new(1i64.into())), "constant".to_string()));
        }
        Arc::new(ProjectionExec::try_new(expr, sort).unwrap())
    }

    fn requirement_count(plan: &Arc<dyn ExecutionPlan>) -> usize {
        usize::from(plan.downcast_ref::<OutputRequirementExec>().is_some())
            + plan
                .children()
                .iter()
                .map(|child| requirement_count(child))
                .sum::<usize>()
    }

    #[tokio::test]
    async fn repeated_optimization_does_not_duplicate_requirements() {
        let config = ConfigOptions::new();
        let plan = PreserveGlobalOrder
            .optimize(projected_sort(2, false).await, &config)
            .unwrap();
        assert!(plan.downcast_ref::<OutputRequirementExec>().is_some());
        let plan = PreserveGlobalOrder.optimize(plan, &config).unwrap();
        assert_eq!(requirement_count(&plan), 1);
    }

    #[tokio::test]
    async fn dropping_sort_keys_keeps_datafusions_requirement_at_the_sort() {
        // A prefix (or none) of the sort keys cannot represent the full row
        // order. Leave these plans to DataFusion rather than promoting a weaker
        // requirement that could shuffle rows tied on the remaining keys.
        let config = ConfigOptions::new();
        for keep_sort_columns in [0, 1] {
            let plan = PreserveGlobalOrder
                .optimize(projected_sort(keep_sort_columns, false).await, &config)
                .unwrap();
            assert!(plan.downcast_ref::<ProjectionExec>().is_some());
            let plan = PreserveGlobalOrder.optimize(plan, &config).unwrap();
            assert_eq!(requirement_count(&plan), 1);
        }
    }

    #[tokio::test]
    async fn partition_local_sort_does_not_become_a_global_requirement() {
        let plan = PreserveGlobalOrder
            .optimize(projected_sort(2, true).await, &ConfigOptions::new())
            .unwrap();
        assert!(plan.downcast_ref::<ProjectionExec>().is_some());
        assert_eq!(requirement_count(&plan), 1);
    }
}
