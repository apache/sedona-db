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
use crate::Result;
use arrow::compute::take;
use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Schema};
use sedona_libgpuspatial::{GpuSpatialContext, SpatialPredicate};
use std::sync::Arc;
use std::time::Instant;

/// GPU backend for spatial operations
#[allow(dead_code)]
pub struct GpuBackend {
    device_id: i32,
    gpu_context: Option<GpuSpatialContext>,
}

#[allow(dead_code)]
impl GpuBackend {
    pub fn new(device_id: i32) -> Result<Self> {
        Ok(Self {
            device_id,
            gpu_context: None,
        })
    }

    pub fn init(&mut self) -> Result<()> {
        // Initialize GPU context
        println!(
            "[GPU Join] Initializing GPU context (device {})",
            self.device_id
        );
        match GpuSpatialContext::new() {
            Ok(mut ctx) => {
                ctx.init().map_err(|e| {
                    crate::Error::GpuInit(format!("Failed to initialize GPU context: {e:?}"))
                })?;
                self.gpu_context = Some(ctx);
                println!("[GPU Join] GPU context initialized successfully");
                Ok(())
            }
            Err(e) => {
                log::warn!("GPU not available: {e:?}");
                println!("[GPU Join] Warning: GPU not available: {e:?}");
                // Gracefully handle GPU not being available
                Ok(())
            }
        }
    }

    /// Convert BinaryView array to Binary array for GPU processing
    /// OPTIMIZATION: Use Arrow's optimized cast instead of manual iteration
    fn ensure_binary_array(array: &ArrayRef) -> Result<ArrayRef> {
        match array.data_type() {
            DataType::BinaryView => {
                // OPTIMIZATION: Use Arrow's cast which is much faster than manual iteration
                use arrow::compute::cast;
                cast(array.as_ref(), &DataType::Binary).map_err(crate::Error::Arrow)
            }
            DataType::Binary | DataType::LargeBinary => {
                // Already in correct format
                Ok(array.clone())
            }
            _ => Err(crate::Error::GpuSpatial(format!(
                "Expected Binary/BinaryView array, got {:?}",
                array.data_type()
            ))),
        }
    }

    pub fn spatial_join(
        &mut self,
        left_batch: &RecordBatch,
        right_batch: &RecordBatch,
        left_geom_col: usize,
        right_geom_col: usize,
        predicate: SpatialPredicate,
    ) -> Result<RecordBatch> {
        let gpu_ctx = match &mut self.gpu_context {
            Some(ctx) => ctx,
            None => {
                return Err(crate::Error::GpuInit(
                    "GPU context not available - falling back to CPU".into(),
                ));
            }
        };

        // Extract geometry columns from both batches
        let left_geom = left_batch.column(left_geom_col);
        let right_geom = right_batch.column(right_geom_col);

        log::info!(
            "GPU spatial join: left_batch={} rows, right_batch={} rows, left_geom type={:?}, right_geom type={:?}",
            left_batch.num_rows(),
            right_batch.num_rows(),
            left_geom.data_type(),
            right_geom.data_type()
        );

        // Convert BinaryView to Binary if needed
        let left_geom = Self::ensure_binary_array(left_geom)?;
        let right_geom = Self::ensure_binary_array(right_geom)?;

        log::info!(
            "After conversion: left_geom type={:?} len={}, right_geom type={:?} len={}",
            left_geom.data_type(),
            left_geom.len(),
            right_geom.data_type(),
            right_geom.len()
        );

        // OPTIMIZATION: Remove clones - Arc is cheap to clone, but avoid if possible
        match gpu_ctx.spatial_join(left_geom.clone(), right_geom.clone(), predicate) {
            Ok((build_indices, stream_indices)) => {
                // Create result record batch from the join indices
                self.create_result_batch(left_batch, right_batch, &build_indices, &stream_indices)
            }
            Err(e) => Err(crate::Error::GpuSpatial(format!(
                "GPU spatial join failed: {e:?}"
            ))),
        }
    }

    /// Create result RecordBatch from join indices
    fn create_result_batch(
        &self,
        left_batch: &RecordBatch,
        right_batch: &RecordBatch,
        build_indices: &[u32],
        stream_indices: &[u32],
    ) -> Result<RecordBatch> {
        if build_indices.len() != stream_indices.len() {
            return Err(crate::Error::GpuSpatial(
                "Mismatched join result lengths".into(),
            ));
        }

        let num_matches = build_indices.len();
        if num_matches == 0 {
            // Return empty result with combined schema
            let combined_schema =
                self.create_combined_schema(&left_batch.schema(), &right_batch.schema())?;
            return Ok(RecordBatch::new_empty(Arc::new(combined_schema)));
        }

        let materialize_start = Instant::now();

        // Build arrays for left side (build indices)
        // OPTIMIZATION: Create index arrays once and reuse for all columns
        let build_idx_array = UInt32Array::from(build_indices.to_vec());
        let stream_idx_array = UInt32Array::from(stream_indices.to_vec());

        let mut left_arrays: Vec<ArrayRef> = Vec::new();
        for i in 0..left_batch.num_columns() {
            let column = left_batch.column(i);
            let selected = take(column.as_ref(), &build_idx_array, None)?;
            left_arrays.push(selected);
        }

        // Build arrays for right side (stream indices)
        let mut right_arrays: Vec<ArrayRef> = Vec::new();
        for i in 0..right_batch.num_columns() {
            let column = right_batch.column(i);
            let selected = take(column.as_ref(), &stream_idx_array, None)?;
            right_arrays.push(selected);
        }

        // Combine arrays and create schema
        let mut all_arrays = left_arrays;
        all_arrays.extend(right_arrays);

        let combined_schema =
            self.create_combined_schema(&left_batch.schema(), &right_batch.schema())?;

        let result = RecordBatch::try_new(Arc::new(combined_schema), all_arrays)?;
        let materialize_elapsed = materialize_start.elapsed();
        println!(
            "[GPU Join] Result batch materialized in {:.3}s: {} rows, {} columns",
            materialize_elapsed.as_secs_f64(),
            result.num_rows(),
            result.num_columns()
        );

        Ok(result)
    }

    /// Create combined schema for join result
    fn create_combined_schema(
        &self,
        left_schema: &Schema,
        right_schema: &Schema,
    ) -> Result<Schema> {
        // Combine schemas directly without prefixes to match exec.rs schema creation
        let mut fields = left_schema.fields().to_vec();
        fields.extend_from_slice(right_schema.fields());
        Ok(Schema::new(fields))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_creation() {
        let backend = GpuBackend::new(0);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_gpu_backend_initialization() {
        let mut backend = GpuBackend::new(0).unwrap();
        let result = backend.init();

        #[cfg(gpu_available)]
        assert!(result.is_ok());
        #[cfg(not(gpu_available))]
        assert!(result.is_ok()); // Should still succeed but with no GPU context
    }
}
