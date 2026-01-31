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

use datafusion_physical_plan::metrics::{
    self, ExecutionPlanMetricsSet, MetricBuilder, SpillMetrics,
};

#[derive(Clone, Debug)]
pub(crate) struct ProbeStreamMetrics {
    pub probe_input_batches: metrics::Count,
    pub probe_input_rows: metrics::Count,
    pub repartition_time: metrics::Time,
    pub spill_metrics: SpillMetrics,
}

impl ProbeStreamMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            probe_input_batches: MetricBuilder::new(metrics)
                .counter("probe_input_batches", partition),
            probe_input_rows: MetricBuilder::new(metrics).counter("probe_input_rows", partition),
            repartition_time: MetricBuilder::new(metrics)
                .subset_time("partition_probe_time", partition),
            spill_metrics: SpillMetrics::new(metrics, partition),
        }
    }
}

pub(crate) mod first_pass_stream;
pub(crate) mod non_partitioned_stream;
pub(crate) mod partitioned_stream_provider;
