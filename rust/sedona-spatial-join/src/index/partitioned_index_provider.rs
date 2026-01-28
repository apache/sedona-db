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

use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result, SharedResult};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::JoinType;
use futures::StreamExt;
use parking_lot::Mutex;
use sedona_common::{sedona_internal_err, SpatialJoinOptions};
use std::ops::DerefMut;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::evaluated_batch::evaluated_batch_stream::external::ExternalEvaluatedBatchStream;
use crate::index::BuildPartition;
use crate::partitioning::stream_repartitioner::{SpilledPartition, SpilledPartitions};
use crate::utils::disposable_async_cell::DisposableAsyncCell;
use crate::{
    index::{SpatialIndex, SpatialIndexBuilder, SpatialJoinBuildMetrics},
    partitioning::SpatialPartition,
    spatial_predicate::SpatialPredicate,
};

pub(crate) struct PartitionedIndexProvider {
    schema: SchemaRef,
    spatial_predicate: SpatialPredicate,
    options: SpatialJoinOptions,
    join_type: JoinType,
    probe_threads_count: usize,
    metrics: SpatialJoinBuildMetrics,

    /// Data on the build side to build index for
    data: BuildSideData,

    /// Async cells for indexes, one per regular partition
    index_cells: Vec<DisposableAsyncCell<SharedResult<Arc<SpatialIndex>>>>,

    /// The memory reserved in the build side collection phase. We'll hold them until
    /// we don't need to build spatial indexes.
    _reservations: Vec<MemoryReservation>,
}

pub(crate) enum BuildSideData {
    SinglePartition(Mutex<Option<Vec<BuildPartition>>>),
    MultiPartition(Mutex<SpilledPartitions>),
}

impl PartitionedIndexProvider {
    #[allow(clippy::too_many_arguments)]
    pub fn new_multi_partition(
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        partitioned_spill_files: SpilledPartitions,
        metrics: SpatialJoinBuildMetrics,
        reservations: Vec<MemoryReservation>,
    ) -> Self {
        let num_partitions = partitioned_spill_files.num_regular_partitions();
        let index_cells = (0..num_partitions)
            .map(|_| DisposableAsyncCell::new())
            .collect();
        Self {
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
            data: BuildSideData::MultiPartition(Mutex::new(partitioned_spill_files)),
            index_cells,
            _reservations: reservations,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_single_partition(
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        mut build_partitions: Vec<BuildPartition>,
        metrics: SpatialJoinBuildMetrics,
    ) -> Self {
        let reservations = build_partitions
            .iter_mut()
            .map(|p| p.reservation.take())
            .collect();
        let index_cells = vec![DisposableAsyncCell::new()];
        Self {
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
            data: BuildSideData::SinglePartition(Mutex::new(Some(build_partitions))),
            index_cells,
            _reservations: reservations,
        }
    }

    pub fn new_empty(
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Self {
        let build_partitions = Vec::new();
        Self::new_single_partition(
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            build_partitions,
            metrics,
        )
    }

    pub fn num_regular_partitions(&self) -> usize {
        self.index_cells.len()
    }

    pub async fn build_or_wait_for_index(
        &self,
        partition_id: u32,
    ) -> Option<Result<Arc<SpatialIndex>>> {
        let cell = match self.index_cells.get(partition_id as usize) {
            Some(cell) => cell,
            None => {
                return Some(sedona_internal_err!(
                    "partition_id {} exceeds {} partitions",
                    partition_id,
                    self.index_cells.len()
                ))
            }
        };
        if !cell.is_empty() {
            return get_index_from_cell(cell).await;
        }

        let res_index = {
            let opt_res_index = self.maybe_build_index(partition_id).await;
            match opt_res_index {
                Some(res_index) => res_index,
                None => {
                    // The build side data for building the index has already been consumed by someone else,
                    // we just need to wait for the task consumed the data to finish building the index.
                    return get_index_from_cell(cell).await;
                }
            }
        };

        match res_index {
            Ok(idx) => {
                if let Err(e) = cell.set(Ok(Arc::clone(&idx))) {
                    // This is probably because the cell has been disposed. No one
                    // will get the index from the cell so this failure is not a big deal.
                    log::debug!("Cannot set the index into the async cell: {:?}", e);
                }
                Some(Ok(idx))
            }
            Err(err) => {
                let err_arc = Arc::new(err);
                if let Err(e) = cell.set(Err(Arc::clone(&err_arc))) {
                    log::debug!(
                        "Cannot set the index build error into the async cell: {:?}",
                        e
                    );
                }
                Some(Err(DataFusionError::Shared(err_arc)))
            }
        }
    }

    async fn maybe_build_index(&self, partition_id: u32) -> Option<Result<Arc<SpatialIndex>>> {
        match &self.data {
            BuildSideData::SinglePartition(build_partition_opt) => {
                if partition_id != 0 {
                    return Some(sedona_internal_err!(
                        "partition_id for single-partition index is not 0"
                    ));
                }

                // consume the build side data for building the index
                let build_partition_opt = {
                    let mut locked = build_partition_opt.lock();
                    std::mem::take(locked.deref_mut())
                };

                let Some(build_partition) = build_partition_opt else {
                    // already consumed by previous attempts, the result should be present in the channel.
                    return None;
                };
                Some(self.build_index_for_single_partition(build_partition).await)
            }
            BuildSideData::MultiPartition(partitioned_spill_files) => {
                // consume this partition of build side data for building index
                let spilled_partition = {
                    let mut locked = partitioned_spill_files.lock();
                    let partition = SpatialPartition::Regular(partition_id);
                    if !locked.can_take_spilled_partition(partition) {
                        // already consumed by previous attempts, the result should be present in the channel.
                        return None;
                    }
                    match locked.take_spilled_partition(partition) {
                        Ok(spilled_partition) => spilled_partition,
                        Err(e) => return Some(Err(e)),
                    }
                };
                Some(
                    self.build_index_for_spilled_partition(spilled_partition)
                        .await,
                )
            }
        }
    }

    #[cfg(test)]
    pub async fn wait_for_index(&self, partition_id: u32) -> Option<Result<Arc<SpatialIndex>>> {
        let cell = match self.index_cells.get(partition_id as usize) {
            Some(cell) => cell,
            None => {
                return Some(sedona_internal_err!(
                    "partition_id {} exceeds {} partitions",
                    partition_id,
                    self.index_cells.len()
                ))
            }
        };

        get_index_from_cell(cell).await
    }

    pub fn dispose_index(&self, partition_id: u32) {
        if let Some(cell) = self.index_cells.get(partition_id as usize) {
            cell.dispose();
        }
    }

    pub fn num_loaded_indexes(&self) -> usize {
        self.index_cells
            .iter()
            .filter(|index_cell| index_cell.is_set())
            .count()
    }

    async fn build_index_for_single_partition(
        &self,
        build_partitions: Vec<BuildPartition>,
    ) -> Result<Arc<SpatialIndex>> {
        let mut index_builder = SpatialIndexBuilder::new(
            Arc::clone(&self.schema),
            self.spatial_predicate.clone(),
            self.options.clone(),
            self.join_type,
            self.probe_threads_count,
            self.metrics.clone(),
        )?;

        for build_partition in build_partitions {
            let stream = build_partition.build_side_batch_stream;
            let geo_statistics = build_partition.geo_statistics;
            index_builder.add_stream(stream, geo_statistics).await?;
        }

        let index = index_builder.finish()?;
        Ok(Arc::new(index))
    }

    async fn build_index_for_spilled_partition(
        &self,
        spilled_partition: SpilledPartition,
    ) -> Result<Arc<SpatialIndex>> {
        let mut index_builder = SpatialIndexBuilder::new(
            Arc::clone(&self.schema),
            self.spatial_predicate.clone(),
            self.options.clone(),
            self.join_type,
            self.probe_threads_count,
            self.metrics.clone(),
        )?;

        // Spawn tasks to load indexed batches from spilled files concurrently
        let (spill_files, geo_statistics, _) = spilled_partition.into_inner();
        let mut join_set: JoinSet<Result<(), DataFusionError>> = JoinSet::new();
        let (tx, mut rx) = mpsc::channel(spill_files.len() * 2 + 1);
        for spill_file in spill_files {
            let tx = tx.clone();
            join_set.spawn(async move {
                let result = async {
                    let mut stream = ExternalEvaluatedBatchStream::try_from_spill_file(spill_file)?;
                    while let Some(batch) = stream.next().await {
                        let indexed_batch = batch?;
                        if tx.send(Ok(indexed_batch)).await.is_err() {
                            return Ok(());
                        }
                    }
                    Ok::<(), DataFusionError>(())
                }
                .await;
                if let Err(e) = result {
                    let _ = tx.send(Err(e)).await;
                }
                Ok(())
            });
        }
        drop(tx);

        // Collect the loaded indexed batches and add them to the index builder
        while let Some(res) = rx.recv().await {
            let batch = res?;
            index_builder.add_batch(batch)?;
        }

        // Ensure all tasks completed successfully
        while let Some(res) = join_set.join_next().await {
            if let Err(e) = res {
                if e.is_panic() {
                    std::panic::resume_unwind(e.into_panic());
                }
                return Err(DataFusionError::External(Box::new(e)));
            }
        }

        index_builder.merge_stats(geo_statistics);

        let index = index_builder.finish()?;
        Ok(Arc::new(index))
    }
}

async fn get_index_from_cell(
    cell: &DisposableAsyncCell<SharedResult<Arc<SpatialIndex>>>,
) -> Option<Result<Arc<SpatialIndex>>> {
    match cell.get().await {
        Some(Ok(index)) => Some(Ok(index)),
        Some(Err(shared_err)) => Some(Err(DataFusionError::Shared(shared_err))),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operand_evaluator::EvaluatedGeometryArray;
    use crate::partitioning::partition_slots::PartitionSlots;
    use crate::utils::bbox_sampler::BoundingBoxSamples;
    use crate::{
        evaluated_batch::{
            evaluated_batch_stream::{
                in_mem::InMemoryEvaluatedBatchStream, SendableEvaluatedBatchStream,
            },
            EvaluatedBatch,
        },
        index::CollectBuildSideMetrics,
    };
    use arrow_array::{ArrayRef, BinaryArray, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::config::SpillCompression;
    use datafusion_common::{DataFusionError, Result};
    use datafusion_execution::{
        memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool},
        runtime_env::RuntimeEnv,
    };
    use datafusion_expr::JoinType;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
    use sedona_expr::statistics::GeoStatistics;
    use sedona_functions::st_analyze_agg::AnalyzeAccumulator;
    use sedona_geometry::analyze::analyze_geometry;
    use sedona_schema::datatypes::WKB_GEOMETRY;

    use crate::evaluated_batch::spill::EvaluatedBatchSpillWriter;
    use crate::partitioning::stream_repartitioner::{SpilledPartition, SpilledPartitions};
    use crate::spatial_predicate::{RelationPredicate, SpatialRelationType};

    fn sample_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("geom", DataType::Binary, true),
            Field::new("id", DataType::Int32, false),
        ]))
    }

    fn point_wkb(x: f64, y: f64) -> Vec<u8> {
        let mut buf = vec![1u8, 1, 0, 0, 0];
        buf.extend_from_slice(&x.to_le_bytes());
        buf.extend_from_slice(&y.to_le_bytes());
        buf
    }

    fn sample_batch(ids: &[i32], wkbs: Vec<Option<Vec<u8>>>) -> Result<EvaluatedBatch> {
        assert_eq!(ids.len(), wkbs.len());
        let geom_values: Vec<Option<&[u8]>> = wkbs
            .iter()
            .map(|opt| opt.as_ref().map(|wkb| wkb.as_slice()))
            .collect();
        let geom_array: ArrayRef = Arc::new(BinaryArray::from_opt_vec(geom_values));
        let id_array: ArrayRef = Arc::new(Int32Array::from(ids.to_vec()));
        let batch = RecordBatch::try_new(sample_schema(), vec![geom_array.clone(), id_array])?;
        let geom = EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY)?;
        Ok(EvaluatedBatch {
            batch,
            geom_array: geom,
        })
    }

    fn predicate() -> SpatialPredicate {
        SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 0)),
            SpatialRelationType::Intersects,
        ))
    }

    fn geo_stats_from_batches(batches: &[EvaluatedBatch]) -> Result<GeoStatistics> {
        let mut analyzer = AnalyzeAccumulator::new(WKB_GEOMETRY, WKB_GEOMETRY);
        for batch in batches {
            for wkb in batch.geom_array.wkbs().iter().flatten() {
                let summary =
                    analyze_geometry(wkb).map_err(|e| DataFusionError::External(Box::new(e)))?;
                analyzer.ingest_geometry_summary(&summary);
            }
        }
        Ok(analyzer.finish())
    }

    fn new_reservation(memory_pool: Arc<dyn MemoryPool>) -> MemoryReservation {
        let consumer = MemoryConsumer::new("PartitionedIndexProviderTest");
        consumer.register(&memory_pool)
    }

    fn build_partition_from_batches(
        memory_pool: Arc<dyn MemoryPool>,
        batches: Vec<EvaluatedBatch>,
    ) -> Result<BuildPartition> {
        let schema = batches
            .first()
            .map(|batch| batch.schema())
            .unwrap_or_else(|| Arc::new(Schema::empty()));
        let geo_statistics = geo_stats_from_batches(&batches)?;
        let num_rows = batches.iter().map(|batch| batch.num_rows()).sum();
        let mut estimated_usage = 0;
        for batch in &batches {
            estimated_usage += batch.in_mem_size()?;
        }
        let stream: SendableEvaluatedBatchStream =
            Box::pin(InMemoryEvaluatedBatchStream::new(schema, batches));
        Ok(BuildPartition {
            num_rows,
            build_side_batch_stream: stream,
            geo_statistics,
            bbox_samples: BoundingBoxSamples::empty(),
            estimated_spatial_index_memory_usage: estimated_usage,
            reservation: new_reservation(memory_pool),
            metrics: CollectBuildSideMetrics::new(0, &ExecutionPlanMetricsSet::new()),
        })
    }

    fn spill_partition_from_batches(
        runtime_env: Arc<RuntimeEnv>,
        batches: Vec<EvaluatedBatch>,
    ) -> Result<SpilledPartition> {
        if batches.is_empty() {
            return Ok(SpilledPartition::empty());
        }
        let schema = batches[0].schema();
        let sedona_type = batches[0].geom_array.sedona_type.clone();
        let mut writer = EvaluatedBatchSpillWriter::try_new(
            runtime_env,
            schema,
            &sedona_type,
            "partitioned-index-provider-test",
            SpillCompression::Uncompressed,
            SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0),
            None,
        )?;
        let mut num_rows = 0;
        for batch in &batches {
            num_rows += batch.num_rows();
            writer.append(batch)?;
        }
        let geo_statistics = geo_stats_from_batches(&batches)?;
        let spill_file = writer.finish()?;
        Ok(SpilledPartition::new(
            vec![Arc::new(spill_file)],
            geo_statistics,
            num_rows,
        ))
    }

    fn make_spilled_partitions(
        runtime_env: Arc<RuntimeEnv>,
        partitions: Vec<Vec<EvaluatedBatch>>,
    ) -> Result<SpilledPartitions> {
        let slots = PartitionSlots::new(partitions.len());
        let mut spilled = Vec::with_capacity(slots.total_slots());
        for partition_batches in partitions {
            spilled.push(spill_partition_from_batches(
                Arc::clone(&runtime_env),
                partition_batches,
            )?);
        }
        spilled.push(SpilledPartition::empty());
        spilled.push(SpilledPartition::empty());
        Ok(SpilledPartitions::new(slots, spilled))
    }

    #[tokio::test]
    async fn single_partition_builds_once_and_is_cached() -> Result<()> {
        let memory_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let batches = vec![sample_batch(
            &[1, 2],
            vec![Some(point_wkb(10.0, 10.0)), Some(point_wkb(20.0, 20.0))],
        )?];
        let build_partition = build_partition_from_batches(Arc::clone(&memory_pool), batches)?;
        let metrics = ExecutionPlanMetricsSet::new();
        let provider = PartitionedIndexProvider::new_single_partition(
            sample_schema(),
            predicate(),
            SpatialJoinOptions::default(),
            JoinType::Inner,
            1,
            vec![build_partition],
            SpatialJoinBuildMetrics::new(0, &metrics),
        );

        let first_index = provider
            .build_or_wait_for_index(0)
            .await
            .expect("partition exists")?;
        assert_eq!(first_index.indexed_batches.len(), 1);
        assert_eq!(provider.num_loaded_indexes(), 1);

        let cached_index = provider
            .wait_for_index(0)
            .await
            .expect("cached value must remain accessible")?;
        assert!(Arc::ptr_eq(&first_index, &cached_index));
        Ok(())
    }

    #[tokio::test]
    async fn multi_partition_concurrent_requests_share_indexes() -> Result<()> {
        let memory_pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let runtime_env = Arc::new(RuntimeEnv::default());
        let partition_batches = vec![
            vec![sample_batch(&[10], vec![Some(point_wkb(0.0, 0.0))])?],
            vec![sample_batch(&[20], vec![Some(point_wkb(50.0, 50.0))])?],
        ];
        let spilled_partitions = make_spilled_partitions(runtime_env, partition_batches)?;
        let metrics = ExecutionPlanMetricsSet::new();
        let provider = Arc::new(PartitionedIndexProvider::new_multi_partition(
            sample_schema(),
            predicate(),
            SpatialJoinOptions::default(),
            JoinType::Inner,
            1,
            spilled_partitions,
            SpatialJoinBuildMetrics::new(0, &metrics),
            vec![new_reservation(Arc::clone(&memory_pool))],
        ));

        let (idx_one, idx_two) = tokio::join!(
            provider.build_or_wait_for_index(0),
            provider.build_or_wait_for_index(0)
        );
        let idx_one = idx_one.expect("partition exists")?;
        let idx_two = idx_two.expect("partition exists")?;
        assert!(Arc::ptr_eq(&idx_one, &idx_two));
        assert_eq!(idx_one.indexed_batches.len(), 1);

        let second_partition = provider
            .build_or_wait_for_index(1)
            .await
            .expect("second partition exists")?;
        assert_eq!(second_partition.indexed_batches.len(), 1);
        assert_eq!(provider.num_loaded_indexes(), 2);
        Ok(())
    }
}
