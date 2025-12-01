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
use std::{any::Any, sync::Arc};

use datafusion_common::{Result, Statistics};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::{
    execution_plan::CardinalityEffect,
    metrics::MetricsSet,
    projection::ProjectionExec,
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};

/// Wrapper around a [ProjectionExec] that isolates a projection from optimizer rules
///
/// Without this wrapper, optimizer rules attempt to push the bbox projection required
/// for GeoParquet 1.1 writes into a different place in the plan that DataFusion
/// does not seem to propagate correctly <https://github.com/apache/sedona-db/issues/379>.
#[derive(Debug, Clone)]
pub struct OpaqueProjectExec {
    pub inner: ProjectionExec,
}

impl ExecutionPlan for OpaqueProjectExec {

    fn name(&self) -> &'static str {
        "OpaqueProjectExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        self.inner.properties()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.inner.maintains_input_order()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        self.inner.benefits_from_input_partitioning()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inner.children()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let new_inner = Arc::new(self.inner.clone()).with_new_children(children)?;
        Ok(Arc::new(Self {
            inner: new_inner
                .as_any()
                .downcast_ref::<ProjectionExec>()
                .unwrap()
                .clone(),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.inner.execute(partition, context)
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.inner.metrics()
    }

    #[allow(deprecated)]
    fn statistics(&self) -> Result<Statistics> {
        self.inner.statistics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        self.inner.partition_statistics(partition)
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.inner.supports_limit_pushdown()
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        self.inner.cardinality_effect()
    }
}

impl DisplayAs for OpaqueProjectExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.inner.fmt_as(t, f)
    }
}
