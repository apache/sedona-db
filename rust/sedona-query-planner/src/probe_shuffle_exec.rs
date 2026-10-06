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

//! [`ProbeShuffleExec`] — a round-robin repartitioning wrapper that is invisible
//! to DataFusion's `EnforceDistribution` / `EnforceSorting` optimizer passes.
//!
//! Those passes unconditionally strip every [`RepartitionExec`] before
//! re-evaluating distribution requirements.  Because `SpatialJoinExec` reports
//! `UnspecifiedDistribution` for its inputs, a bare `RepartitionExec` that was
//! inserted by the extension planner is removed and never re-added.
//!
//! `ProbeShuffleExec` wraps a hidden, internal `RepartitionExec` so that:
//! * **Optimizer passes** see an opaque node (not a `RepartitionExec`) and leave
//!   it alone.
//! * **`children()` / `with_new_children()`** expose the *original* input so
//!   the rest of the optimizer tree can still be rewritten normally.
//! * **`execute()`** delegates to the internal `RepartitionExec` which performs
//!   the actual round-robin shuffle.
//!
//! `RoundRobinBatch` deals out whole batches, so a probe input that arrives as
//! only a few batches (for example a raster catalog: a few thousand small rows
//! read from a single Parquet row group) lands on only a few partitions, and
//! everything downstream of the join runs on those partitions alone. Optionally
//! (see [`ProbeShuffleExec::try_new_with_batch_split`]), each input batch is first
//! cut into zero-copy row slices so that even a single batch is spread over every
//! output partition. The slicing step lives behind the same opaque node, so the
//! optimizer can neither move nor strip it.

use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, Statistics, internal_err, plan_err};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
};
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties,
};
use futures::StreamExt;

/// A round-robin repartitioning node that is invisible to DataFusion's
/// physical optimizer passes.
///
/// See [module-level documentation](self) for motivation and design.
#[derive(Debug)]
pub struct ProbeShuffleExec {
    /// The probe input; this is the node's only child as seen by the optimizer.
    input: Arc<dyn ExecutionPlan>,
    /// When set, input batches are cut into row slices of at least this many
    /// rows before the round-robin. See [`Self::try_new_with_batch_split`].
    batch_split_min_rows: Option<usize>,
    /// The repartition that does the work. Its input is `input`, or a hidden
    /// [`SplitBatchesExec`] over `input` when batch splitting is enabled.
    inner_repartition: RepartitionExec,
}

impl ProbeShuffleExec {
    /// Create a new [`ProbeShuffleExec`] that round-robin repartitions `input`
    /// into the same number of output partitions as `input`. This will ensure
    /// that the probe workload of a spatial join will be evenly distributed.
    /// More importantly, shuffled probe side data will be less likely to
    /// cause skew issues when out-of-core, spatial partitioned spatial join is enabled,
    /// especially when the input probe data is sorted by their spatial locations.
    pub fn try_new(input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        let num_partitions = input.output_partitioning().partition_count();
        Self::try_new_with_options(input, num_partitions, None)
    }

    /// Like [`Self::try_new`], but each input batch is first cut into up to
    /// `num_partitions` zero-copy row slices of at least `min_rows` rows, so a
    /// probe input made of a few batches still reaches every output partition.
    ///
    /// A batch of `n` rows becomes `clamp(n / min_rows, 1, num_partitions)`
    /// slices of near-equal size, so no slice is smaller than `min_rows` unless
    /// the batch itself is, and a batch never yields more slices than there are
    /// partitions. Only operators whose per-row downstream work is heavy relative
    /// to the per-batch overhead (such as the raster join) should enable this.
    pub fn try_new_with_batch_split(
        input: Arc<dyn ExecutionPlan>,
        min_rows: usize,
    ) -> Result<Self> {
        let num_partitions = input.output_partitioning().partition_count();
        Self::try_new_with_options(input, num_partitions, Some(min_rows.max(1)))
    }

    fn try_new_with_options(
        input: Arc<dyn ExecutionPlan>,
        num_partitions: usize,
        batch_split_min_rows: Option<usize>,
    ) -> Result<Self> {
        let repartition_input: Arc<dyn ExecutionPlan> = match batch_split_min_rows {
            Some(min_rows) => Arc::new(SplitBatchesExec {
                input: Arc::clone(&input),
                max_slices: num_partitions.max(1),
                min_rows,
            }),
            None => Arc::clone(&input),
        };
        let inner_repartition = RepartitionExec::try_new(
            repartition_input,
            Partitioning::RoundRobinBatch(num_partitions),
        )?;
        Ok(Self {
            input,
            batch_split_min_rows,
            inner_repartition,
        })
    }

    /// Try to wrap the given [`RepartitionExec`] `plan` with [`ProbeShuffleExec`].
    ///
    /// The result deals whole batches (no batch splitting).
    pub fn try_wrap_repartition(plan: Arc<dyn ExecutionPlan>) -> Result<Self> {
        let Some(repartition_exec) = plan.downcast_ref::<RepartitionExec>() else {
            return plan_err!(
                "ProbeShuffleExec can only wrap RepartitionExec, but got {}",
                plan.name()
            );
        };
        Ok(Self {
            input: Arc::clone(repartition_exec.input()),
            batch_split_min_rows: None,
            inner_repartition: repartition_exec.clone(),
        })
    }

    /// Number of output partitions.
    pub fn num_partitions(&self) -> usize {
        self.inner_repartition
            .properties()
            .output_partitioning()
            .partition_count()
    }

    /// The minimum slice size when input batches are split before the
    /// round-robin, or `None` when whole batches are dealt.
    pub fn batch_split_min_rows(&self) -> Option<usize> {
        self.batch_split_min_rows
    }
}

impl DisplayAs for ProbeShuffleExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ProbeShuffleExec: partitioning=RoundRobinBatch({})",
                    self.num_partitions()
                )?;
                if let Some(min_rows) = self.batch_split_min_rows {
                    write!(f, ", split_batches_min_rows={min_rows}")?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                write!(f, "partitioning=RoundRobinBatch({})", self.num_partitions())?;
                if let Some(min_rows) = self.batch_split_min_rows {
                    write!(f, "\nsplit_batches_min_rows={min_rows}")?;
                }
                Ok(())
            }
        }
    }
}

impl ExecutionPlan for ProbeShuffleExec {
    fn name(&self) -> &str {
        "ProbeShuffleExec"
    }

    fn properties(&self) -> &std::sync::Arc<PlanProperties> {
        self.inner_repartition.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!(
                "ProbeShuffleExec expects exactly 1 child, got {}",
                children.len()
            );
        }
        let child = children.remove(0);
        let num_partitions = child.output_partitioning().partition_count();
        Ok(Arc::new(Self::try_new_with_options(
            child,
            num_partitions,
            self.batch_split_min_rows,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.inner_repartition.execute(partition, context)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.inner_repartition.maintains_input_order()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        self.inner_repartition.benefits_from_input_partitioning()
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        self.inner_repartition.cardinality_effect()
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.inner_repartition.metrics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        self.inner_repartition.partition_statistics(partition)
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        // Ask a plain repartition over the visible input, so the projection is
        // pushed below the (hidden) batch split rather than between it and the
        // repartition.
        let plain = RepartitionExec::try_new(
            Arc::clone(&self.input),
            self.inner_repartition.partitioning().clone(),
        )?;
        let Some(new_repartition) = plain.try_swapping_with_projection(projection)? else {
            return Ok(None);
        };
        let Some(new_repartition) = new_repartition.downcast_ref::<RepartitionExec>() else {
            return internal_err!(
                "RepartitionExec projection swap did not return a RepartitionExec"
            );
        };
        Ok(Some(Arc::new(Self::try_new_with_options(
            Arc::clone(new_repartition.input()),
            self.num_partitions(),
            self.batch_split_min_rows,
        )?)))
    }

    fn gather_filters_for_pushdown(
        &self,
        phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        self.inner_repartition
            .gather_filters_for_pushdown(phase, parent_filters, config)
    }

    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        self.inner_repartition
            .handle_child_pushdown_result(phase, child_pushdown_result, config)
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        config: &ConfigOptions,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if self
            .inner_repartition
            .repartitioned(target_partitions, config)?
            .is_none()
        {
            return Ok(None);
        }
        // Rebuild rather than wrap the repartitioned inner node: the hidden batch
        // split caps its slice count at the number of output partitions.
        Ok(Some(Arc::new(Self::try_new_with_options(
            Arc::clone(&self.input),
            target_partitions,
            self.batch_split_min_rows,
        )?)))
    }
}

/// Cuts each input batch into up to `max_slices` zero-copy row slices of at
/// least `min_rows` rows (see [`ProbeShuffleExec::try_new_with_batch_split`]).
///
/// Only ever used as the input of the [`RepartitionExec`] hidden inside a
/// [`ProbeShuffleExec`]; it never appears in a plan the optimizer can see.
#[derive(Debug)]
struct SplitBatchesExec {
    input: Arc<dyn ExecutionPlan>,
    max_slices: usize,
    min_rows: usize,
}

impl SplitBatchesExec {
    fn split(batch: RecordBatch, max_slices: usize, min_rows: usize) -> Vec<RecordBatch> {
        let num_rows = batch.num_rows();
        let num_slices = (num_rows / min_rows).clamp(1, max_slices);
        if num_slices == 1 {
            return vec![batch];
        }
        // Near-equal slices: the first `num_rows % num_slices` get one extra row.
        let base = num_rows / num_slices;
        let extra = num_rows % num_slices;
        let mut offset = 0;
        (0..num_slices)
            .map(|i| {
                let len = base + usize::from(i < extra);
                let slice = batch.slice(offset, len);
                offset += len;
                slice
            })
            .collect()
    }
}

impl DisplayAs for SplitBatchesExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SplitBatchesExec: max_slices={}, min_rows={}",
            self.max_slices, self.min_rows
        )
    }
}

impl ExecutionPlan for SplitBatchesExec {
    fn name(&self) -> &str {
        "SplitBatchesExec"
    }

    fn properties(&self) -> &std::sync::Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!(
                "SplitBatchesExec expects exactly 1 child, got {}",
                children.len()
            );
        }
        Ok(Arc::new(Self {
            input: children.remove(0),
            max_slices: self.max_slices,
            min_rows: self.min_rows,
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let (max_slices, min_rows) = (self.max_slices, self.min_rows);
        let stream = self
            .input
            .execute(partition, context)?
            .flat_map(move |batch| {
                let slices = match batch {
                    Ok(batch) => Self::split(batch, max_slices, min_rows)
                        .into_iter()
                        .map(Ok)
                        .collect(),
                    Err(e) => vec![Err(e)],
                };
                futures::stream::iter(slices)
            });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.input.schema(),
            stream,
        )))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        self.input.partition_statistics(partition)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::{collect_partitioned, displayable};
    use datafusion::prelude::SessionContext;
    use datafusion_common::config::ConfigOptions;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::projection::ProjectionExec;

    use super::{ProbeShuffleExec, SplitBatchesExec};

    #[test]
    fn split_rule() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));

        // (num_rows, max_slices, min_rows) -> slice lengths
        for (num_rows, max_slices, min_rows, expected) in [
            // Smaller than min_rows: left whole.
            (0, 12, 64, vec![0]),
            (63, 12, 64, vec![63]),
            // Room for only two slices of at least 64 rows.
            (130, 12, 64, vec![65, 65]),
            // A big batch is capped at one slice per partition, near-equal sizes.
            (2000, 12, 64, [vec![167; 8], vec![166; 4]].concat()),
            (8192, 4, 64, vec![2048; 4]),
            // A single partition never splits.
            (8192, 1, 64, vec![8192]),
        ] {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from_iter_values(0..num_rows))],
            )
            .unwrap();
            let slices = SplitBatchesExec::split(batch, max_slices, min_rows);
            let lengths: Vec<usize> = slices.iter().map(|b| b.num_rows()).collect();
            assert_eq!(lengths, expected, "{num_rows} rows, {max_slices} slices");

            // Slices are contiguous and in order.
            let values: Vec<i32> = slices
                .iter()
                .flat_map(|b| {
                    let column = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    column.values().to_vec()
                })
                .collect();
            assert_eq!(values, (0..num_rows).collect::<Vec<_>>());
        }
    }

    /// A probe input whose only batch sits in one partition reaches every
    /// output partition when batch splitting is on, and only one without it.
    #[tokio::test]
    async fn single_batch_spreads_over_all_partitions() {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..1000))],
        )
        .unwrap();
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch], vec![], vec![], vec![]], schema, None)
                .unwrap();
        let task_ctx = SessionContext::new().task_ctx();

        let whole = Arc::new(ProbeShuffleExec::try_new(input.clone()).unwrap());
        let split = Arc::new(ProbeShuffleExec::try_new_with_batch_split(input, 64).unwrap());
        assert_eq!(whole.batch_split_min_rows(), None);
        assert_eq!(split.batch_split_min_rows(), Some(64));

        let whole_rows: Vec<usize> = collect_partitioned(whole, task_ctx.clone())
            .await
            .unwrap()
            .iter()
            .map(|p| p.iter().map(|b| b.num_rows()).sum())
            .collect();
        let mut nonempty = whole_rows.iter().filter(|&&n| n > 0).count();
        assert_eq!(nonempty, 1, "{whole_rows:?}");

        let split_partitions = collect_partitioned(split, task_ctx).await.unwrap();
        let split_rows: Vec<usize> = split_partitions
            .iter()
            .map(|p| p.iter().map(|b| b.num_rows()).sum())
            .collect();
        nonempty = split_rows.iter().filter(|&&n| n > 0).count();
        assert_eq!(nonempty, 4, "{split_rows:?}");
        assert_eq!(split_rows, vec![250; 4]);

        // Same rows overall.
        let mut values: Vec<i32> = split_partitions
            .iter()
            .flatten()
            .flat_map(|b| {
                let column = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                column.values().to_vec()
            })
            .collect();
        values.sort_unstable();
        assert_eq!(values, (0..1000).collect::<Vec<_>>());
    }

    /// Plan rewrites keep the split setting and never expose the hidden split
    /// node as a child.
    #[test]
    fn plan_rewrites_keep_batch_split() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let input = MemorySourceConfig::try_new_exec(&[vec![], vec![]], schema, None).unwrap();
        let input: Arc<dyn ExecutionPlan> = input;
        let exec = Arc::new(ProbeShuffleExec::try_new_with_batch_split(input.clone(), 64).unwrap());

        assert_eq!(
            displayable(exec.as_ref()).one_line().to_string(),
            "ProbeShuffleExec: partitioning=RoundRobinBatch(2), split_batches_min_rows=64\n"
        );
        assert_eq!(
            displayable(&ProbeShuffleExec::try_new(input.clone()).unwrap())
                .one_line()
                .to_string(),
            "ProbeShuffleExec: partitioning=RoundRobinBatch(2)\n"
        );
        assert!(Arc::ptr_eq(exec.children()[0], &input));

        let rebuilt = exec.clone().with_new_children(vec![input.clone()]).unwrap();
        let rebuilt = rebuilt.downcast_ref::<ProbeShuffleExec>().unwrap();
        assert_eq!(rebuilt.batch_split_min_rows(), Some(64));
        assert!(Arc::ptr_eq(rebuilt.children()[0], &input));

        let repartitioned = exec
            .repartitioned(8, &ConfigOptions::default())
            .unwrap()
            .unwrap();
        let repartitioned = repartitioned.downcast_ref::<ProbeShuffleExec>().unwrap();
        assert_eq!(repartitioned.num_partitions(), 8);
        assert_eq!(repartitioned.batch_split_min_rows(), Some(64));
        assert!(Arc::ptr_eq(repartitioned.children()[0], &input));

        // A narrowing projection is pushed below the shuffle, directly onto the
        // original input.
        let projection = ProjectionExec::try_new(
            vec![(Arc::new(Column::new("a", 0)) as _, "a".to_string())],
            exec.clone(),
        )
        .unwrap();
        let swapped = exec
            .try_swapping_with_projection(&projection)
            .unwrap()
            .unwrap();
        let swapped = swapped.downcast_ref::<ProbeShuffleExec>().unwrap();
        assert_eq!(swapped.batch_split_min_rows(), Some(64));
        assert_eq!(swapped.num_partitions(), 2);
        let child_projection = swapped.children()[0]
            .downcast_ref::<ProjectionExec>()
            .unwrap();
        assert!(Arc::ptr_eq(child_projection.input(), &input));
    }
}
