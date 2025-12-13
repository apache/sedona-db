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

use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{memory_pool::MemoryConsumer, SendableRecordBatchStream, TaskContext};
use datafusion_expr::JoinType;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use sedona_common::SedonaOptions;

use crate::{
    index::{
        BuildSideBatchesCollector, CollectBuildSideMetrics, SpatialIndex, SpatialIndexBuilder,
        SpatialJoinBuildMetrics,
    },
    operand_evaluator::create_operand_evaluator,
    spatial_predicate::SpatialPredicate,
};

/// Build a spatial index from the build side streams.
///
/// This function reads the `concurrent_build_side_collection` configuration from the context
/// to determine whether to collect build side partitions concurrently (using spawned tasks)
/// or sequentially (for JNI/embedded contexts without async runtime support).
#[allow(clippy::too_many_arguments)]
pub async fn build_index(
    context: Arc<TaskContext>,
    build_schema: SchemaRef,
    build_streams: Vec<SendableRecordBatchStream>,
    spatial_predicate: SpatialPredicate,
    join_type: JoinType,
    probe_threads_count: usize,
    metrics: ExecutionPlanMetricsSet,
) -> Result<SpatialIndex> {
    let session_config = context.session_config();
    let sedona_options = session_config
        .options()
        .extensions
        .get::<SedonaOptions>()
        .cloned()
        .unwrap_or_default();
    let concurrent = sedona_options.spatial_join.concurrent_build_side_collection;
    let memory_pool = context.memory_pool();
    let evaluator =
        create_operand_evaluator(&spatial_predicate, sedona_options.spatial_join.clone());
    let collector = BuildSideBatchesCollector::new(evaluator);
    let num_partitions = build_streams.len();
    let mut collect_metrics_vec = Vec::with_capacity(num_partitions);
    let mut reservations = Vec::with_capacity(num_partitions);
    for k in 0..num_partitions {
        let consumer =
            MemoryConsumer::new(format!("SpatialJoinCollectBuildSide[{k}]")).with_can_spill(true);
        let reservation = consumer.register(memory_pool);
        reservations.push(reservation);
        collect_metrics_vec.push(CollectBuildSideMetrics::new(k, &metrics));
    }

    let build_partitions = if concurrent {
        // Collect partitions concurrently using collect_all which spawns tasks
        collector
            .collect_all(build_streams, reservations, collect_metrics_vec)
            .await?
    } else {
        // Collect partitions sequentially (for JNI/embedded contexts)
        let mut partitions = Vec::with_capacity(num_partitions);
        for ((stream, reservation), metrics) in build_streams
            .into_iter()
            .zip(reservations)
            .zip(&collect_metrics_vec)
        {
            let partition = collector.collect(stream, reservation, metrics).await?;
            partitions.push(partition);
        }
        partitions
    };

    let contains_external_stream = build_partitions
        .iter()
        .any(|partition| partition.build_side_batch_stream.is_external());
    if !contains_external_stream {
        let mut index_builder = SpatialIndexBuilder::new(
            build_schema,
            spatial_predicate,
            sedona_options.spatial_join,
            join_type,
            probe_threads_count,
            Arc::clone(memory_pool),
            SpatialJoinBuildMetrics::new(0, &metrics),
        )?;
        index_builder.add_partitions(build_partitions).await?;
        index_builder.finish()
    } else {
        Err(DataFusionError::ResourcesExhausted("Memory limit exceeded while collecting indexed data. External spatial index builder is not yet implemented.".to_string()))
    }
}
