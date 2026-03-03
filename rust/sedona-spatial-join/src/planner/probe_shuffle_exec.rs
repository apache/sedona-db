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

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion_common::config::ConfigOptions;
use datafusion_common::{internal_err, plan_err, Result, Statistics};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
};
use datafusion_physical_plan::metrics::MetricsSet;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties,
};

/// A round-robin repartitioning node that is invisible to DataFusion's
/// physical optimizer passes.
///
/// See [module-level documentation](self) for motivation and design.
#[derive(Debug)]
pub struct ProbeShuffleExec {
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
        let inner_repartition = RepartitionExec::try_new(
            Arc::clone(&input),
            Partitioning::RoundRobinBatch(num_partitions),
        )?;
        Ok(Self { inner_repartition })
    }

    /// Try to wrap the given [`RepartitionExec`] `plan` with [`ProbeShuffleExec`].
    pub fn try_wrap_repartition(plan: Arc<dyn ExecutionPlan>) -> Result<Self> {
        let Some(repartition_exec) = plan.as_any().downcast_ref::<RepartitionExec>() else {
            return plan_err!(
                "ProbeShuffleExec can only wrap RepartitionExec, but got {}",
                plan.name()
            );
        };
        Ok(Self {
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
}

impl DisplayAs for ProbeShuffleExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "ProbeShuffleExec: partitioning=RoundRobinBatch({})",
                    self.num_partitions()
                )
            }
            DisplayFormatType::TreeRender => {
                write!(f, "partitioning=RoundRobinBatch({})", self.num_partitions())
            }
        }
    }
}

impl ExecutionPlan for ProbeShuffleExec {
    fn name(&self) -> &str {
        "ProbeShuffleExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        self.inner_repartition.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![self.inner_repartition.input()]
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
        Ok(Arc::new(Self::try_new(child)?))
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

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        self.inner_repartition.partition_statistics(partition)
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(new_repartition) = self
            .inner_repartition
            .try_swapping_with_projection(projection)?
        else {
            return Ok(None);
        };
        let new_plan = Self::try_wrap_repartition(new_repartition)?;
        Ok(Some(Arc::new(new_plan)))
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
        let Some(plan) = self
            .inner_repartition
            .repartitioned(target_partitions, config)?
        else {
            return Ok(None);
        };
        let new_plan = Self::try_wrap_repartition(plan)?;
        Ok(Some(Arc::new(new_plan)))
    }
}
