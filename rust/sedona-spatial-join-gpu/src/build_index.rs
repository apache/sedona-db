use crate::index::{
    BuildSideBatchesCollector, CollectBuildSideMetrics, SpatialIndex, SpatialIndexBuilder,
    SpatialJoinBuildMetrics,
};
use crate::operand_evaluator::create_operand_evaluator;
use crate::spatial_predicate::SpatialPredicate;
use crate::GpuSpatialJoinConfig;
use datafusion_common::{DataFusionError, JoinType};
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use parking_lot::RwLock;
use sedona_common::SedonaOptions;
use std::sync::Arc;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

pub async fn build_index(
    context: Arc<TaskContext>,
    build_streams: Vec<SendableRecordBatchStream>,
    spatial_predicate: SpatialPredicate,
    join_type: JoinType,
    probe_threads_count: usize,
    metrics: ExecutionPlanMetricsSet,
    _gpu_join_config: GpuSpatialJoinConfig,
) -> datafusion_common::Result<Arc<SpatialIndex>> {
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
            spatial_predicate,
            sedona_options.spatial_join,
            join_type,
            probe_threads_count,
            SpatialJoinBuildMetrics::new(0, &metrics),
        );
        index_builder.add_partitions(build_partitions).await?;
        let res = index_builder.finish();
        Ok(Arc::new(res?))
    } else {
        Err(DataFusionError::ResourcesExhausted("Memory limit exceeded while collecting indexed data. External spatial index builder is not yet implemented.".to_string()))
    }
}
