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

//! Streaming spatial partitioning utilities.
//!
//! This module provides helpers that repartition an [`EvaluatedBatch`] stream into
//! spatial spill files using a [`SpatialPartitioner`]. Each regular partition, along
//! with the None and Multi partitions, gets at most one spill file which can
//! later be replayed via [`SendableEvaluatedBatchStream`].

use std::sync::Arc;

use crate::{
    evaluated_batch::{
        evaluated_batch_stream::SendableEvaluatedBatchStream, spill::EvaluatedBatchSpillWriter,
        EvaluatedBatch,
    },
    operand_evaluator::EvaluatedGeometryArray,
    partitioning::{
        partition_slots::PartitionSlots, util::geo_rect_to_bbox, PartitionedSide, SpatialPartition,
        SpatialPartitioner,
    },
    utils::arrow_utils::{compact_array, compact_batch},
};
use arrow::compute::interleave as arrow_interleave;
use arrow::compute::interleave_record_batch;
use arrow_array::{Array, ArrayRef, RecordBatch};
use datafusion::config::SpillCompression;
use datafusion_common::{Result, ScalarValue};
use datafusion_execution::{disk_manager::RefCountedTempFile, runtime_env::RuntimeEnv};
use datafusion_expr::ColumnarValue;
use datafusion_physical_plan::metrics::SpillMetrics;
use futures::StreamExt;
use sedona_common::sedona_internal_err;
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::st_analyze_agg::AnalyzeAccumulator;
use sedona_geometry::bounding_box::BoundingBox;
use sedona_geometry::interval::IntervalTrait;
use sedona_schema::datatypes::WKB_GEOMETRY;

/// Result emitted after a stream is spatially repartitioned.
#[derive(Debug)]
pub struct SpilledPartitions {
    slots: PartitionSlots,
    partitions: Vec<Option<SpilledPartition>>,
}

/// Spill metadata captured for a single spatial partition.
#[derive(Debug, Clone)]
pub struct SpilledPartition {
    spill_files: Vec<Arc<RefCountedTempFile>>,
    geo_statistics: GeoStatistics,
    num_rows: usize,
}

impl SpilledPartition {
    pub fn new(
        spill_files: Vec<Arc<RefCountedTempFile>>,
        geo_statistics: GeoStatistics,
        num_rows: usize,
    ) -> Self {
        Self {
            spill_files,
            geo_statistics,
            num_rows,
        }
    }

    pub fn empty() -> Self {
        Self::new(Vec::new(), GeoStatistics::empty(), 0)
    }

    pub fn spill_files(&self) -> &[Arc<RefCountedTempFile>] {
        &self.spill_files
    }

    pub fn geo_statistics(&self) -> &GeoStatistics {
        &self.geo_statistics
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn bounding_box(&self) -> Option<&BoundingBox> {
        self.geo_statistics.bbox()
    }

    pub fn into_spill_files(self) -> Vec<Arc<RefCountedTempFile>> {
        self.spill_files
    }

    pub fn into_inner(self) -> (Vec<Arc<RefCountedTempFile>>, GeoStatistics, usize) {
        (self.spill_files, self.geo_statistics, self.num_rows)
    }
}

impl SpilledPartitions {
    pub fn new(slots: PartitionSlots, partitions: Vec<SpilledPartition>) -> Self {
        assert_eq!(partitions.len(), slots.total_slots());
        let partitions = partitions.into_iter().map(Some).collect();
        Self { slots, partitions }
    }

    /// Number of regular partitions
    pub fn num_regular_partitions(&self) -> usize {
        self.slots.num_regular_partitions()
    }

    /// Get the slots for mapping spatial partitions to sequential 0-based indexes
    pub fn slots(&self) -> PartitionSlots {
        self.slots
    }

    /// Count of spill files that were actually materialized.
    pub fn spill_file_count(&self) -> usize {
        self.partitions
            .iter()
            .map(|partition| partition.as_ref().map_or(0, |p| p.spill_files().len()))
            .sum()
    }

    /// Retrieve the spilled partition for a given spatial partition.
    pub fn spilled_partition(&self, partition: SpatialPartition) -> Result<&SpilledPartition> {
        let Some(slot) = self.slots.slot(partition) else {
            return sedona_internal_err!(
                "Invalid partition {:?} for {} regular partitions",
                partition,
                self.slots.num_regular_partitions()
            );
        };
        match &self.partitions[slot] {
            Some(spilled_partition) => Ok(spilled_partition),
            None => sedona_internal_err!(
                "Spilled partition {:?} has already been taken away",
                partition
            ),
        }
    }

    /// Consume this structure into concrete spilled partitions.
    pub fn into_spilled_partitions(self) -> Result<Vec<SpilledPartition>> {
        let mut partitions = Vec::with_capacity(self.partitions.len());
        for partition_opt in self.partitions {
            match partition_opt {
                Some(partition) => partitions.push(partition),
                None => {
                    return sedona_internal_err!(
                        "Some of the spilled partitions have already been taken away"
                    )
                }
            }
        }
        Ok(partitions)
    }

    /// Get a clone of the spill files in specified partition without consuming it. This is
    /// for retrieving the Multi partition, which may be scanned multiple times.
    pub fn get_spilled_partition(&self, partition: SpatialPartition) -> Result<SpilledPartition> {
        let Some(slot) = self.slots.slot(partition) else {
            return sedona_internal_err!(
                "Invalid partition {:?} for {} regular partitions",
                partition,
                self.slots.num_regular_partitions()
            );
        };
        match &self.partitions[slot] {
            Some(spilled_partition) => Ok(spilled_partition.clone()),
            None => sedona_internal_err!(
                "Spilled partition {:?} has already been taken away",
                partition
            ),
        }
    }

    /// Take the spill files in specified partition from it without consuming this value
    pub fn take_spilled_partition(
        &mut self,
        partition: SpatialPartition,
    ) -> Result<SpilledPartition> {
        let Some(slot) = self.slots.slot(partition) else {
            return sedona_internal_err!(
                "Invalid partition {:?} for {} regular partitions",
                partition,
                self.slots.num_regular_partitions()
            );
        };
        match std::mem::take(&mut self.partitions[slot]) {
            Some(spilled_partition) => Ok(spilled_partition),
            None => sedona_internal_err!(
                "Spilled partition {:?} has already been taken away",
                partition
            ),
        }
    }

    /// Is the spill files still present and can be taken away
    pub fn can_take_spilled_partition(&self, partition: SpatialPartition) -> bool {
        let Some(slot) = self.slots.slot(partition) else {
            return false;
        };
        self.partitions[slot].is_some()
    }

    /// Write debug info for this spilled partitions
    pub fn debug_print(&self, f: &mut impl std::fmt::Write) -> std::fmt::Result {
        for k in 0..self.slots.total_slots() {
            if let Some(spilled_partition) = &self.partitions[k] {
                let bbox_str = if let Some(bbox) = spilled_partition.bounding_box() {
                    format!(
                        "x: [{:.6}, {:.6}], y: [{:.6}, {:.6}]",
                        bbox.x().lo(),
                        bbox.x().hi(),
                        bbox.y().lo(),
                        bbox.y().hi()
                    )
                } else {
                    "None".to_string()
                };
                let spill_files = spilled_partition.spill_files();
                let spill_file_sizes = spill_files
                    .iter()
                    .map(|sp| {
                        sp.inner()
                            .as_file()
                            .metadata()
                            .map(|m| m.len())
                            .unwrap_or(0)
                    })
                    .collect::<Vec<_>>();
                writeln!(
                    f,
                    "Partition {:?}: {} spill file(s), num non-empty geoms: {:?}, bbox: {}, spill file sizes: {:?}",
                    self.slots.partition(k),
                    spilled_partition.spill_files().len(),
                    spilled_partition
                        .geo_statistics()
                        .total_geometries()
                        .unwrap_or_default(),
                    bbox_str,
                    spill_file_sizes,
                )?;
            } else {
                writeln!(f, "Partition {}: already taken away", k)?;
            }
        }
        Ok(())
    }
}

/// Stateful helper that incrementally repartitions [`EvaluatedBatch`] values into
/// spill files while keeping writers open across batches.
pub struct StreamRepartitioner {
    runtime_env: Arc<RuntimeEnv>,
    partitioner: Arc<dyn SpatialPartitioner>,
    partitioned_side: PartitionedSide,
    slots: PartitionSlots,
    /// Spill files for each spatial partition.
    /// The None and Multi partitions should be None when repartitioning the build side.
    spill_registry: Vec<Option<EvaluatedBatchSpillWriter>>,
    /// Geospatial statistics for each spatial partition.
    geo_stats_accumulators: Vec<AnalyzeAccumulator>,
    /// Number of rows in each spatial partition.
    num_rows: Vec<usize>,
    slot_assignments: Vec<Vec<(usize, usize)>>,
    row_assignments_buffer: Vec<SpatialPartition>,
    spill_compression: SpillCompression,
    spill_metrics: SpillMetrics,
    buffer_bytes_threshold: usize,
    target_batch_size: usize,
    spilled_batch_in_memory_size_threshold: Option<usize>,
    pending_batches: Vec<EvaluatedBatch>,
    pending_bytes: usize,
}

impl StreamRepartitioner {
    /// Create a new repartitioner that targets the provided spatial partitioner.
    pub fn new(
        runtime_env: Arc<RuntimeEnv>,
        partitioner: Arc<dyn SpatialPartitioner>,
        partitioned_side: PartitionedSide,
        spill_compression: SpillCompression,
        spill_metrics: SpillMetrics,
        buffer_bytes_threshold: usize,
        target_batch_size: usize,
        spilled_batch_in_memory_size_threshold: Option<usize>,
    ) -> Self {
        let slots = PartitionSlots::new(partitioner.num_regular_partitions());
        let slot_count = slots.total_slots();
        Self {
            runtime_env,
            partitioner,
            partitioned_side,
            slots,
            spill_registry: (0..slot_count).map(|_| None).collect(),
            geo_stats_accumulators: (0..slot_count)
                .map(|_| AnalyzeAccumulator::new(WKB_GEOMETRY, WKB_GEOMETRY))
                .collect(),
            num_rows: vec![0; slot_count],
            slot_assignments: (0..slot_count).map(|_| Vec::new()).collect(),
            row_assignments_buffer: Vec::new(),
            spill_compression,
            spill_metrics,
            buffer_bytes_threshold,
            target_batch_size,
            spilled_batch_in_memory_size_threshold,
            pending_batches: Vec::new(),
            pending_bytes: 0,
        }
    }

    /// Route a single evaluated batch into its corresponding spill writers.
    pub fn repartition_batch(&mut self, batch: EvaluatedBatch) -> Result<()> {
        let mut row_assignments = std::mem::take(&mut self.row_assignments_buffer);
        assign_rows(
            &batch,
            self.partitioner.as_ref(),
            self.partitioned_side,
            &mut row_assignments,
        )?;
        self.insert_repartitioned_batch(batch, &row_assignments)?;
        self.row_assignments_buffer = row_assignments;
        Ok(())
    }

    /// Insert batch with row assignments into the repartitioner. The spatial partitioner
    /// does not need to be invoked in this method. This is useful when the batch has
    /// already been partitioned by calling assign_rows.
    pub fn insert_repartitioned_batch(
        &mut self,
        batch: EvaluatedBatch,
        row_assignments: &[SpatialPartition],
    ) -> Result<()> {
        let batch_idx = self.pending_batches.len();
        self.pending_bytes += batch.in_mem_size()?;
        self.pending_batches.push(batch);
        let batch_ref = &self.pending_batches[batch_idx];
        assert_eq!(row_assignments.len(), batch_ref.num_rows());
        for (row_idx, partition) in row_assignments.iter().enumerate() {
            let Some(slot_idx) = self.slots.slot(*partition) else {
                return sedona_internal_err!(
                    "Invalid partition {:?} for {} regular partitions",
                    partition,
                    self.slots.num_regular_partitions()
                );
            };
            if let Some(wkb) = batch_ref.wkb(row_idx) {
                self.geo_stats_accumulators[slot_idx].update_statistics(wkb)?;
            }
            self.slot_assignments[slot_idx].push((batch_idx, row_idx));
            self.num_rows[slot_idx] += 1;
        }
        let threshold = self.buffer_bytes_threshold;
        if threshold == 0 || self.pending_bytes >= threshold {
            self.flush_pending_batches()?;
        }
        Ok(())
    }

    fn flush_pending_batches(&mut self) -> Result<()> {
        if self.pending_batches.is_empty() {
            debug_assert!(self
                .slot_assignments
                .iter()
                .all(|assignments| assignments.is_empty()));
            return Ok(());
        }

        let pending_batches = std::mem::take(&mut self.pending_batches);
        self.pending_bytes = 0;

        let record_batches: Vec<&RecordBatch> =
            pending_batches.iter().map(|batch| &batch.batch).collect();
        let geom_arrays: Vec<&EvaluatedGeometryArray> = pending_batches
            .iter()
            .map(|batch| &batch.geom_array)
            .collect();

        let mut slot_assignments = std::mem::take(&mut self.slot_assignments);

        for (slot_idx, assignments) in slot_assignments.iter_mut().enumerate() {
            if assignments.is_empty() {
                continue;
            }
            let chunk_cap = if self.target_batch_size == 0 {
                assignments.len()
            } else {
                self.target_batch_size
            }
            .max(1);
            for chunk in assignments.chunks(chunk_cap) {
                let sliced_batch =
                    interleave_evaluated_batch(&record_batches, &geom_arrays, chunk)?;
                let writer = self.ensure_writer(slot_idx, &sliced_batch)?;
                writer.append(&sliced_batch)?;
            }

            assignments.clear();
        }

        self.slot_assignments = slot_assignments;
        Ok(())
    }

    /// Seal every partition and return their associated spill files and bounds.
    pub fn finish(mut self) -> Result<SpilledPartitions> {
        self.flush_pending_batches()?;
        let slot_count = self.slots.total_slots();
        let mut spilled_partition_vec = Vec::with_capacity(slot_count);
        for ((writer_opt, accumulator), num_rows) in self
            .spill_registry
            .into_iter()
            .zip(self.geo_stats_accumulators.into_iter())
            .zip(self.num_rows.into_iter())
        {
            let spilled_partition = if let Some(writer) = writer_opt {
                let spill_files = vec![Arc::new(writer.finish()?)];
                let geo_statistics = accumulator.finish();
                SpilledPartition::new(spill_files, geo_statistics, num_rows)
            } else {
                SpilledPartition::empty()
            };
            spilled_partition_vec.push(spilled_partition);
        }

        Ok(SpilledPartitions::new(self.slots, spilled_partition_vec))
    }

    fn ensure_writer(
        &mut self,
        slot_idx: usize,
        batch: &EvaluatedBatch,
    ) -> Result<&mut EvaluatedBatchSpillWriter> {
        if self.spill_registry[slot_idx].is_none() {
            self.spill_registry[slot_idx] = Some(EvaluatedBatchSpillWriter::try_new(
                Arc::clone(&self.runtime_env),
                batch.schema(),
                &batch.geom_array.sedona_type,
                "streaming repartitioner",
                self.spill_compression,
                self.spill_metrics.clone(),
                self.spilled_batch_in_memory_size_threshold,
            )?);
        }
        Ok(self.spill_registry[slot_idx]
            .as_mut()
            .expect("writer inserted above"))
    }
}

/// Repartition evaluated batches into per-partition spill files.
#[allow(clippy::too_many_arguments)]
pub async fn repartition_evaluated_batches(
    runtime_env: Arc<RuntimeEnv>,
    mut stream: SendableEvaluatedBatchStream,
    partitioner: Arc<dyn SpatialPartitioner>,
    partitioned_side: PartitionedSide,
    spill_compression: SpillCompression,
    spill_metrics: SpillMetrics,
    buffer_bytes_threshold: usize,
    target_batch_size: usize,
    spilled_batch_in_memory_size_threshold: Option<usize>,
) -> Result<SpilledPartitions> {
    let mut repartitioner = StreamRepartitioner::new(
        runtime_env,
        partitioner,
        partitioned_side,
        spill_compression,
        spill_metrics,
        buffer_bytes_threshold,
        target_batch_size,
        spilled_batch_in_memory_size_threshold,
    );
    while let Some(batch_result) = stream.next().await {
        let batch = batch_result?;
        repartitioner.repartition_batch(batch)?;
    }
    repartitioner.finish()
}

/// Populate `assignments` with the spatial partition for every row in `batch`,
/// reusing the provided buffer to avoid repeated allocations. The vector length
/// after this call matches `batch.rects().len()` and each entry records which
/// [`SpatialPartition`] the corresponding row belongs to.
pub(crate) fn assign_rows(
    batch: &EvaluatedBatch,
    partitioner: &dyn SpatialPartitioner,
    partitioned_side: PartitionedSide,
    assignments: &mut Vec<SpatialPartition>,
) -> Result<()> {
    assignments.clear();
    assignments.reserve(batch.rects().len());

    match partitioned_side {
        PartitionedSide::BuildSide => {
            let mut cnt = 0;
            let num_regular_partitions = partitioner.num_regular_partitions() as u32;
            for rect_opt in batch.rects() {
                let partition = match rect_opt {
                    Some(rect) => partitioner.partition_no_multi(&geo_rect_to_bbox(rect))?,
                    None => {
                        // Round-robin empty geometries through regular partitions to avoid
                        // overloading a single slot when the build side is mostly empty.
                        let p = SpatialPartition::Regular(cnt);
                        cnt = (cnt + 1) % num_regular_partitions;
                        p
                    }
                };
                assignments.push(partition);
            }
        }
        PartitionedSide::ProbeSide => {
            for rect_opt in batch.rects() {
                let partition = match rect_opt {
                    Some(rect) => partitioner.partition(&geo_rect_to_bbox(rect))?,
                    None => SpatialPartition::None,
                };
                assignments.push(partition);
            }
        }
    }

    Ok(())
}

/// Build a new [`EvaluatedBatch`] by interleaving rows from the provided
/// `record_batches`/`geom_arrays` inputs according to `assignments`. Each pair
/// in `assignments` identifies the source batch index and row index that should
/// appear in the output in order, ensuring the geometry metadata stays aligned
/// with the Arrow row data.
pub(crate) fn interleave_evaluated_batch(
    record_batches: &[&RecordBatch],
    geom_arrays: &[&EvaluatedGeometryArray],
    indices: &[(usize, usize)],
) -> Result<EvaluatedBatch> {
    if record_batches.is_empty() || geom_arrays.is_empty() {
        return sedona_internal_err!("interleave_evaluated_batch requires at least one batch");
    }
    let batch = interleave_record_batch(record_batches, indices)?;
    let batch = compact_batch(batch)?;
    let geom_array = interleave_geometry_array(geom_arrays, indices)?;
    Ok(EvaluatedBatch { batch, geom_array })
}

fn interleave_geometry_array(
    geom_arrays: &[&EvaluatedGeometryArray],
    indices: &[(usize, usize)],
) -> Result<EvaluatedGeometryArray> {
    if geom_arrays.is_empty() {
        return sedona_internal_err!("interleave_geometry_array requires at least one batch");
    }
    let sedona_type = &geom_arrays[0].sedona_type;
    let value_refs: Vec<&dyn Array> = geom_arrays
        .iter()
        .map(|geom| geom.geometry_array.as_ref())
        .collect();
    let geometry_array = arrow_interleave(&value_refs, indices)?;
    let (geometry_array, _) = compact_array(geometry_array)?;

    let distance = interleave_distance_columns(geom_arrays, indices)?;

    let mut result = EvaluatedGeometryArray::try_new(geometry_array, sedona_type)?;
    result.distance = distance;
    Ok(result)
}

fn interleave_distance_columns(
    geom_arrays: &[&EvaluatedGeometryArray],
    assignments: &[(usize, usize)],
) -> Result<Option<ColumnarValue>> {
    // Check consistency and determine if we need array conversion
    let mut first_value: Option<&ColumnarValue> = None;
    let mut needs_array = false;
    let mut all_null = true;
    let mut first_scalar: Option<&ScalarValue> = None;

    for geom in geom_arrays {
        match &geom.distance {
            Some(value) => {
                if first_value.is_none() {
                    first_value = Some(value);
                }

                match value {
                    ColumnarValue::Array(array) => {
                        needs_array = true;
                        if all_null && array.logical_null_count() != array.len() {
                            all_null = false;
                        }
                    }
                    ColumnarValue::Scalar(scalar) => {
                        if let Some(first) = first_scalar {
                            if first != scalar {
                                needs_array = true;
                            }
                        } else {
                            first_scalar = Some(scalar);
                        }
                        if !scalar.is_null() {
                            all_null = false;
                        }
                    }
                }
            }
            None => {
                if first_value.is_some() && !all_null {
                    return sedona_internal_err!("Inconsistent distance metadata across batches");
                }
            }
        }
    }

    if all_null {
        return Ok(None);
    }

    let Some(distance_value) = first_value else {
        return Ok(None);
    };

    // If all scalars match, return scalar
    if !needs_array {
        if let ColumnarValue::Scalar(value) = distance_value {
            return Ok(Some(ColumnarValue::Scalar(value.clone())));
        }
    }

    // Convert to arrays and interleave
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(geom_arrays.len());
    for geom in geom_arrays {
        match &geom.distance {
            Some(ColumnarValue::Array(array)) => arrays.push(array.clone()),
            Some(ColumnarValue::Scalar(value)) => {
                arrays.push(value.to_array_of_size(geom.geometry_array.len())?);
            }
            None => {
                return sedona_internal_err!("Inconsistent distance metadata across batches");
            }
        }
    }

    let array_refs: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
    let array = arrow_interleave(&array_refs, assignments)?;
    Ok(Some(ColumnarValue::Array(array)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, BinaryArray, Int32Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
    use sedona_geometry::bounding_box::BoundingBox;
    use sedona_geometry::interval::IntervalTrait;
    use sedona_schema::datatypes::WKB_GEOMETRY;

    use crate::{
        evaluated_batch::{
            evaluated_batch_stream::in_mem::InMemoryEvaluatedBatchStream,
            spill::EvaluatedBatchSpillReader,
        },
        partitioning::flat::FlatPartitioner,
    };

    const BUFFER_BYTES: usize = 8 * 1024 * 1024;
    const TARGET_BATCH_SIZE: usize = 4096;

    fn sample_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
    }

    fn sample_batch(ids: &[i32], wkbs: Vec<Option<Vec<u8>>>) -> Result<EvaluatedBatch> {
        assert_eq!(ids.len(), wkbs.len());
        let id_array = Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef;
        let batch = RecordBatch::try_new(sample_schema(), vec![id_array])?;
        let geom_values: Vec<Option<&[u8]>> = wkbs
            .iter()
            .map(|wkb_opt| wkb_opt.as_ref().map(|wkb| wkb.as_slice()))
            .collect();
        let geom_array: ArrayRef = Arc::new(BinaryArray::from(geom_values));
        let geom = EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY)?;
        Ok(EvaluatedBatch {
            batch,
            geom_array: geom,
        })
    }

    fn point_wkb(x: f64, y: f64) -> Vec<u8> {
        let mut buf = vec![1u8, 1, 0, 0, 0];
        buf.extend_from_slice(&x.to_le_bytes());
        buf.extend_from_slice(&y.to_le_bytes());
        buf
    }

    fn rect_wkb(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<u8> {
        assert!(min_x <= max_x, "min_x must be <= max_x");
        assert!(min_y <= max_y, "min_y must be <= max_y");
        let mut buf = Vec::with_capacity(1 + 4 + 4 + 4 + 5 * 16);
        buf.push(1u8); // little endian
        buf.extend_from_slice(&3u32.to_le_bytes()); // polygon type
        buf.extend_from_slice(&1u32.to_le_bytes()); // single ring
        buf.extend_from_slice(&5u32.to_le_bytes()); // five coordinates (closed ring)
        let coords = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ];
        for (x, y) in coords {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
        }
        buf
    }

    fn read_ids(file: &RefCountedTempFile) -> Result<Vec<i32>> {
        let mut reader = EvaluatedBatchSpillReader::try_new(file)?;
        let mut ids = Vec::new();
        while let Some(batch) = reader.next_batch() {
            let batch = batch?;
            let array = batch
                .batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..array.len() {
                ids.push(array.value(i));
            }
        }
        Ok(ids)
    }

    fn read_batch_row_counts(file: &RefCountedTempFile) -> Result<Vec<usize>> {
        let mut reader = EvaluatedBatchSpillReader::try_new(file)?;
        let mut counts = Vec::new();
        while let Some(batch) = reader.next_batch() {
            let batch = batch?;
            counts.push(batch.batch.num_rows());
        }
        Ok(counts)
    }

    fn bbox_limits(bbox: &BoundingBox) -> (f64, f64, f64, f64) {
        (bbox.x().lo(), bbox.x().hi(), bbox.y().lo(), bbox.y().hi())
    }

    #[tokio::test]
    async fn repartition_basic() -> Result<()> {
        let wkbs = vec![
            Some(point_wkb(10.0, 10.0)),
            Some(point_wkb(60.0, 10.0)),
            Some(point_wkb(150.0, 10.0)),
        ];
        let batch = sample_batch(&[0, 1, 2], wkbs)?;
        let schema = batch.schema();
        let stream: SendableEvaluatedBatchStream =
            Box::pin(InMemoryEvaluatedBatchStream::new(schema, vec![batch]));

        let partitions = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
        ];
        let partitioner = Arc::new(FlatPartitioner::try_new(partitions)?);
        let runtime_env = Arc::new(RuntimeEnv::default());
        let metrics = SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0);

        let result = repartition_evaluated_batches(
            runtime_env,
            stream,
            partitioner,
            PartitionedSide::ProbeSide,
            SpillCompression::Uncompressed,
            metrics,
            BUFFER_BYTES,
            TARGET_BATCH_SIZE,
            None,
        )
        .await?;

        assert_eq!(result.spill_file_count(), 3);
        assert_eq!(
            read_ids(
                &result
                    .spilled_partition(SpatialPartition::Regular(0))?
                    .spill_files()[0]
            )?,
            vec![0]
        );
        assert_eq!(
            read_ids(
                &result
                    .spilled_partition(SpatialPartition::Regular(1))?
                    .spill_files()[0]
            )?,
            vec![1]
        );
        assert_eq!(
            read_ids(
                &result
                    .spilled_partition(SpatialPartition::None)?
                    .spill_files()[0]
            )?,
            vec![2]
        );

        assert_eq!(
            bbox_limits(
                result
                    .spilled_partition(SpatialPartition::Regular(0))?
                    .bounding_box()
                    .unwrap()
            ),
            (10.0, 10.0, 10.0, 10.0)
        );
        assert_eq!(
            bbox_limits(
                result
                    .spilled_partition(SpatialPartition::Regular(1))?
                    .bounding_box()
                    .unwrap()
            ),
            (60.0, 60.0, 10.0, 10.0)
        );
        assert_eq!(
            bbox_limits(
                result
                    .spilled_partition(SpatialPartition::None)?
                    .bounding_box()
                    .unwrap()
            ),
            (150.0, 150.0, 10.0, 10.0)
        );
        Ok(())
    }

    #[tokio::test]
    async fn repartition_multi_and_none() -> Result<()> {
        let wkbs = vec![Some(rect_wkb(25.0, 0.0, 75.0, 20.0)), None];
        let batch = sample_batch(&[0, 1], wkbs)?;
        let schema = batch.schema();
        let stream: SendableEvaluatedBatchStream =
            Box::pin(InMemoryEvaluatedBatchStream::new(schema, vec![batch]));

        let partitions = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
        ];
        let partitioner = Arc::new(FlatPartitioner::try_new(partitions)?);
        let runtime_env = Arc::new(RuntimeEnv::default());
        let metrics = SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0);

        let result = repartition_evaluated_batches(
            runtime_env,
            stream,
            partitioner,
            PartitionedSide::ProbeSide,
            SpillCompression::Uncompressed,
            metrics,
            BUFFER_BYTES,
            TARGET_BATCH_SIZE,
            None,
        )
        .await?;

        assert_eq!(result.spill_file_count(), 2);
        assert_eq!(
            read_ids(
                &result
                    .spilled_partition(SpatialPartition::Multi)?
                    .spill_files()[0]
            )?,
            vec![0]
        );
        assert_eq!(
            read_ids(
                &result
                    .spilled_partition(SpatialPartition::None)?
                    .spill_files()[0]
            )?,
            vec![1]
        );
        assert_eq!(
            bbox_limits(
                result
                    .spilled_partition(SpatialPartition::Multi)?
                    .bounding_box()
                    .unwrap()
            ),
            (25.0, 75.0, 0.0, 20.0)
        );
        let none_bound = result
            .spilled_partition(SpatialPartition::None)?
            .bounding_box()
            .expect("Geo stats should exist for None partition");
        assert!(none_bound.x().is_empty());
        assert!(none_bound.y().is_empty());
        Ok(())
    }

    #[test]
    fn streaming_repartitioner_finishes_partitions() -> Result<()> {
        let wkbs = vec![Some(point_wkb(10.0, 10.0)), Some(point_wkb(60.0, 10.0))];
        let batch = sample_batch(&[0, 1], wkbs)?;
        let partitions = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
        ];
        let partitioner = Arc::new(FlatPartitioner::try_new(partitions)?);
        let runtime_env = Arc::new(RuntimeEnv::default());
        let spill_metrics = SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let mut repartitioner = StreamRepartitioner::new(
            runtime_env,
            partitioner,
            PartitionedSide::ProbeSide,
            SpillCompression::Uncompressed,
            spill_metrics,
            0,
            TARGET_BATCH_SIZE,
            None,
        );

        repartitioner.repartition_batch(batch)?;
        let result = repartitioner.finish()?;
        assert!(result
            .spilled_partition(SpatialPartition::None)?
            .spill_files()
            .is_empty());
        assert_eq!(
            read_ids(
                &result
                    .spilled_partition(SpatialPartition::Regular(0))?
                    .spill_files()[0]
            )?,
            vec![0]
        );
        assert_eq!(
            read_ids(
                &result
                    .spilled_partition(SpatialPartition::Regular(1))?
                    .spill_files()[0]
            )?,
            vec![1]
        );
        Ok(())
    }

    #[test]
    fn streaming_repartitioner_buffers_until_threshold() -> Result<()> {
        let batch_a = sample_batch(&[0], vec![Some(point_wkb(10.0, 10.0))])?;
        let batch_b = sample_batch(&[1], vec![Some(point_wkb(20.0, 10.0))])?;
        let partitions = vec![BoundingBox::xy((0.0, 50.0), (0.0, 50.0))];
        let partitioner = Arc::new(FlatPartitioner::try_new(partitions)?);
        let runtime_env = Arc::new(RuntimeEnv::default());
        let spill_metrics = SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let mut repartitioner = StreamRepartitioner::new(
            runtime_env,
            partitioner,
            PartitionedSide::ProbeSide,
            SpillCompression::Uncompressed,
            spill_metrics,
            usize::MAX,
            TARGET_BATCH_SIZE,
            None,
        );

        repartitioner.repartition_batch(batch_a)?;
        repartitioner.repartition_batch(batch_b)?;
        let result = repartitioner.finish()?;
        assert_eq!(
            read_ids(
                &result
                    .spilled_partition(SpatialPartition::Regular(0))?
                    .spill_files()[0]
            )?,
            vec![0, 1]
        );
        Ok(())
    }

    #[test]
    fn streaming_repartitioner_respects_target_batch_size() -> Result<()> {
        let batch_a = sample_batch(&[0], vec![Some(point_wkb(10.0, 10.0))])?;
        let batch_b = sample_batch(&[1], vec![Some(point_wkb(20.0, 10.0))])?;
        let partitions = vec![BoundingBox::xy((0.0, 50.0), (0.0, 50.0))];
        let partitioner = Arc::new(FlatPartitioner::try_new(partitions)?);
        let runtime_env = Arc::new(RuntimeEnv::default());
        let spill_metrics = SpillMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
        let mut repartitioner = StreamRepartitioner::new(
            runtime_env,
            partitioner,
            PartitionedSide::ProbeSide,
            SpillCompression::Uncompressed,
            spill_metrics,
            usize::MAX,
            1,
            None,
        );

        repartitioner.repartition_batch(batch_a)?;
        repartitioner.repartition_batch(batch_b)?;
        let result = repartitioner.finish()?;
        let counts = read_batch_row_counts(
            &result
                .spilled_partition(SpatialPartition::Regular(0))?
                .spill_files()[0],
        )?;
        assert_eq!(counts, vec![1, 1]);
        Ok(())
    }

    fn make_geom_array_with_distance(
        wkbs: Vec<Vec<u8>>,
        distance: Option<ColumnarValue>,
    ) -> Result<EvaluatedGeometryArray> {
        let geom_array: ArrayRef = Arc::new(BinaryArray::from(
            wkbs.iter()
                .map(|wkb| Some(wkb.as_slice()))
                .collect::<Vec<_>>(),
        ));
        let mut geom = EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY)?;
        geom.distance = distance;
        Ok(geom)
    }

    #[test]
    fn interleave_distance_none() -> Result<()> {
        let wkbs1 = vec![point_wkb(10.0, 10.0), point_wkb(20.0, 20.0)];
        let wkbs2 = vec![point_wkb(30.0, 30.0)];

        let geom1 = make_geom_array_with_distance(wkbs1, None)?;
        let geom2 = make_geom_array_with_distance(wkbs2, None)?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = interleave_distance_columns(&geom_arrays, &assignments)?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn interleave_distance_uniform_scalar() -> Result<()> {
        let wkbs1 = vec![point_wkb(10.0, 10.0), point_wkb(20.0, 20.0)];
        let wkbs2 = vec![point_wkb(30.0, 30.0)];

        let scalar = ScalarValue::Float64(Some(5.0));
        let geom1 =
            make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Scalar(scalar.clone())))?;
        let geom2 =
            make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Scalar(scalar.clone())))?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = interleave_distance_columns(&geom_arrays, &assignments)?;
        assert!(matches!(result, Some(ColumnarValue::Scalar(_))));
        if let Some(ColumnarValue::Scalar(value)) = result {
            assert_eq!(value, ScalarValue::Float64(Some(5.0)));
        }
        Ok(())
    }

    #[test]
    fn interleave_distance_different_scalars() -> Result<()> {
        use arrow_array::Float64Array;

        let wkbs1 = vec![point_wkb(10.0, 10.0), point_wkb(20.0, 20.0)];
        let wkbs2 = vec![point_wkb(30.0, 30.0)];

        let scalar1 = ScalarValue::Float64(Some(5.0));
        let scalar2 = ScalarValue::Float64(Some(10.0));
        let geom1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Scalar(scalar1)))?;
        let geom2 = make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Scalar(scalar2)))?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = interleave_distance_columns(&geom_arrays, &assignments)?;
        assert!(matches!(result, Some(ColumnarValue::Array(_))));
        if let Some(ColumnarValue::Array(array)) = result {
            let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(float_array.len(), 3);
            assert_eq!(float_array.value(0), 5.0);
            assert_eq!(float_array.value(1), 10.0);
            assert_eq!(float_array.value(2), 5.0);
        }
        Ok(())
    }

    #[test]
    fn interleave_distance_arrays() -> Result<()> {
        use arrow_array::Float64Array;

        let wkbs1 = vec![point_wkb(10.0, 10.0), point_wkb(20.0, 20.0)];
        let wkbs2 = vec![point_wkb(30.0, 30.0)];

        let array1: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let array2: ArrayRef = Arc::new(Float64Array::from(vec![3.0]));
        let geom1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Array(array1)))?;
        let geom2 = make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Array(array2)))?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = interleave_distance_columns(&geom_arrays, &assignments)?;
        assert!(matches!(result, Some(ColumnarValue::Array(_))));
        if let Some(ColumnarValue::Array(array)) = result {
            let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(float_array.len(), 3);
            assert_eq!(float_array.value(0), 1.0);
            assert_eq!(float_array.value(1), 3.0);
            assert_eq!(float_array.value(2), 2.0);
        }
        Ok(())
    }

    #[test]
    fn interleave_distance_mixed_scalar_and_array() -> Result<()> {
        use arrow_array::Float64Array;

        let wkbs1 = vec![point_wkb(10.0, 10.0), point_wkb(20.0, 20.0)];
        let wkbs2 = vec![point_wkb(30.0, 30.0)];

        let scalar = ScalarValue::Float64(Some(5.0));
        let array: ArrayRef = Arc::new(Float64Array::from(vec![10.0]));
        let geom1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Scalar(scalar)))?;
        let geom2 = make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Array(array)))?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        let result = interleave_distance_columns(&geom_arrays, &assignments)?;
        assert!(matches!(result, Some(ColumnarValue::Array(_))));
        if let Some(ColumnarValue::Array(array)) = result {
            let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(float_array.len(), 3);
            assert_eq!(float_array.value(0), 5.0);
            assert_eq!(float_array.value(1), 10.0);
            assert_eq!(float_array.value(2), 5.0);
        }
        Ok(())
    }

    #[test]
    fn interleave_evaluated_batch_empty_assignments() -> Result<()> {
        let batch_a = sample_batch(&[0], vec![Some(point_wkb(10.0, 10.0))])?;
        let batch_b = sample_batch(&[1], vec![Some(point_wkb(20.0, 20.0))])?;
        let record_batches = vec![&batch_a.batch, &batch_b.batch];
        let geom_arrays = vec![&batch_a.geom_array, &batch_b.geom_array];

        let result = interleave_evaluated_batch(&record_batches, &geom_arrays, &[])?;
        assert_eq!(result.batch.num_rows(), 0);
        assert_eq!(result.geom_array.geometry_array.len(), 0);
        assert!(result.geom_array.rects.is_empty());
        assert!(result.geom_array.distance.is_none());
        Ok(())
    }

    #[test]
    fn interleave_distance_inconsistent_metadata() -> Result<()> {
        let wkbs1 = vec![point_wkb(10.0, 10.0)];
        let wkbs2 = vec![point_wkb(20.0, 20.0)];

        let scalar = ScalarValue::Float64(Some(5.0));
        let geom1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Scalar(scalar)))?;
        let geom2 = make_geom_array_with_distance(wkbs2, None)?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0)];

        let result = interleave_distance_columns(&geom_arrays, &assignments);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Inconsistent distance metadata"));
        }
        Ok(())
    }

    #[test]
    fn interleave_binary_view_array() -> Result<()> {
        use arrow_array::BinaryViewArray;
        use sedona_schema::crs::Crs;
        use sedona_schema::datatypes::{Edges, SedonaType};
        let wkb_view_geometry = SedonaType::WkbView(Edges::Planar, Crs::None);

        let wkbs1 = [point_wkb(10.0, 10.0), point_wkb(20.0, 20.0)];
        let wkbs2 = [point_wkb(30.0, 30.0)];

        // Create BinaryViewArray
        let array1 = BinaryViewArray::from_iter(wkbs1.iter().map(|w| Some(w.as_slice())));
        let array2 = BinaryViewArray::from_iter(wkbs2.iter().map(|w| Some(w.as_slice())));

        let geom1 = EvaluatedGeometryArray::try_new(Arc::new(array1), &wkb_view_geometry)?;
        let geom2 = EvaluatedGeometryArray::try_new(Arc::new(array2), &wkb_view_geometry)?;

        let geom_arrays = vec![&geom1, &geom2];
        let assignments = vec![(0, 0), (1, 0), (0, 1)];

        // Create dummy record batches
        let batch1 = RecordBatch::try_new(
            sample_schema(),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )?;
        let batch2 =
            RecordBatch::try_new(sample_schema(), vec![Arc::new(Int32Array::from(vec![3]))])?;
        let record_batches = vec![&batch1, &batch2];

        let result = interleave_evaluated_batch(&record_batches, &geom_arrays, &assignments)?;

        // Check if the result geometry array is BinaryViewArray
        let geom_array = result.geom_array.geometry_array;
        assert!(geom_array
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .is_some());

        // Check values
        let view_array = geom_array
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();
        assert_eq!(view_array.len(), 3);
        assert_eq!(view_array.value(0), wkbs1[0].as_slice());
        assert_eq!(view_array.value(1), wkbs2[0].as_slice());
        assert_eq!(view_array.value(2), wkbs1[1].as_slice());

        Ok(())
    }

    #[test]
    fn interleave_distance_mixed_none_and_null() -> Result<()> {
        use arrow_array::Float64Array;

        let wkbs1 = vec![point_wkb(10.0, 10.0)];
        let wkbs2 = vec![point_wkb(20.0, 20.0)];
        let wkbs3 = vec![point_wkb(30.0, 30.0)];

        let null_array = Arc::new(Float64Array::new_null(1));
        let ega1 = make_geom_array_with_distance(wkbs1, Some(ColumnarValue::Array(null_array)))?;

        let null_scalar = ScalarValue::Float64(None);
        let ega2 = make_geom_array_with_distance(wkbs2, Some(ColumnarValue::Scalar(null_scalar)))?;

        let ega3 = make_geom_array_with_distance(wkbs3, None)?;

        let vec_ega = vec![&ega1, &ega2, &ega3];
        let assignments = vec![(0, 0), (1, 0), (2, 0)];

        let result = interleave_distance_columns(&vec_ega, &assignments)?;
        assert!(result.is_none());
        Ok(())
    }
}
