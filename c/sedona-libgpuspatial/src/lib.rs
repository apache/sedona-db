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

// Module declarations
#[cfg(gpu_available)]
pub mod error;
#[cfg(gpu_available)]
mod libgpuspatial;
#[cfg(gpu_available)]
mod libgpuspatial_glue_bindgen;

// Import Array trait for len() method (used in gpu_available code)
#[cfg(gpu_available)]
use arrow_array::Array;

// Re-exports for GPU functionality
#[cfg(gpu_available)]
pub use error::GpuSpatialError;
#[cfg(gpu_available)]
pub use libgpuspatial::{GpuSpatialJoinerWrapper, GpuSpatialPredicateWrapper};
#[cfg(gpu_available)]
pub use libgpuspatial_glue_bindgen::GpuSpatialJoinerContext;

// Mark GPU types as Send for thread safety
// SAFETY: The GPU library is designed to be used from multiple threads.
// Each thread gets its own context, and the underlying GPU library handles thread safety.
// The raw pointers inside are managed by the C++ library which ensures proper synchronization.
#[cfg(gpu_available)]
unsafe impl Send for GpuSpatialJoinerContext {}

#[cfg(gpu_available)]
unsafe impl Send for libgpuspatial_glue_bindgen::GpuSpatialJoiner {}

#[cfg(gpu_available)]
unsafe impl Send for GpuSpatialJoinerWrapper {}

// Error type for non-GPU builds
#[cfg(not(gpu_available))]
#[derive(Debug, thiserror::Error)]
pub enum GpuSpatialError {
    #[error("GPU not available - CUDA not found during build")]
    GpuNotAvailable,
}

pub type Result<T> = std::result::Result<T, GpuSpatialError>;

/// High-level wrapper for GPU spatial operations
pub struct GpuSpatialContext {
    #[cfg(gpu_available)]
    joiner: Option<GpuSpatialJoinerWrapper>,
    #[cfg(gpu_available)]
    context: Option<GpuSpatialJoinerContext>,
    initialized: bool,
}

impl GpuSpatialContext {
    pub fn new() -> Result<Self> {
        #[cfg(not(gpu_available))]
        {
            Err(GpuSpatialError::GpuNotAvailable)
        }

        #[cfg(gpu_available)]
        {
            Ok(Self {
                joiner: None,
                context: None,
                initialized: false,
            })
        }
    }

    pub fn init(&mut self) -> Result<()> {
        #[cfg(not(gpu_available))]
        {
            Err(GpuSpatialError::GpuNotAvailable)
        }

        #[cfg(gpu_available)]
        {
            let mut joiner = GpuSpatialJoinerWrapper::new();

            // Get PTX path from OUT_DIR
            let out_path = std::path::PathBuf::from(env!("OUT_DIR"));
            let ptx_root = out_path.join("share/gpuspatial/shaders");
            let ptx_root_str = ptx_root
                .to_str()
                .ok_or_else(|| GpuSpatialError::Init("Invalid PTX path".to_string()))?;

            // Initialize with concurrency of 1 for now
            joiner.init(1, ptx_root_str)?;

            // Create context
            let mut ctx = GpuSpatialJoinerContext {
                last_error: std::ptr::null(),
                private_data: std::ptr::null_mut(),
                build_indices: std::ptr::null_mut(),
                stream_indices: std::ptr::null_mut(),
            };
            joiner.create_context(&mut ctx);

            self.joiner = Some(joiner);
            self.context = Some(ctx);
            self.initialized = true;
            Ok(())
        }
    }

    #[cfg(gpu_available)]
    pub fn get_joiner_mut(&mut self) -> Option<&mut GpuSpatialJoinerWrapper> {
        self.joiner.as_mut()
    }

    #[cfg(gpu_available)]
    pub fn get_context_mut(&mut self) -> Option<&mut GpuSpatialJoinerContext> {
        self.context.as_mut()
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Perform spatial join between two geometry arrays
    pub fn spatial_join(
        &mut self,
        left_geom: arrow_array::ArrayRef,
        right_geom: arrow_array::ArrayRef,
        predicate: SpatialPredicate,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        #[cfg(not(gpu_available))]
        {
            let _ = (left_geom, right_geom, predicate);
            Err(GpuSpatialError::GpuNotAvailable)
        }

        #[cfg(gpu_available)]
        {
            if !self.initialized {
                return Err(GpuSpatialError::Init("Context not initialized".into()));
            }

            let joiner = self
                .joiner
                .as_mut()
                .ok_or_else(|| GpuSpatialError::Init("GPU joiner not available".into()))?;

            // Clear previous build data
            joiner.clear();

            // Push build data (left side)
            log::info!(
                "DEBUG: Pushing {} geometries to GPU (build side)",
                left_geom.len()
            );
            log::info!("DEBUG: Left array data type: {:?}", left_geom.data_type());
            if let Some(binary_arr) = left_geom
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
            {
                log::info!("DEBUG: Left binary array has {} values", binary_arr.len());
                if binary_arr.len() > 0 {
                    let first_wkb = binary_arr.value(0);
                    log::info!(
                        "DEBUG: First left WKB length: {}, first bytes: {:?}",
                        first_wkb.len(),
                        &first_wkb[..8.min(first_wkb.len())]
                    );
                }
            }

            //             println!("[GPU Join] Phase 1: Transferring {} left geometries to GPU memory", left_geom.len());
            // let transfer_start = std::time::Instant::now();
            joiner.push_build(&left_geom, 0, left_geom.len() as i64)?;
            // let transfer_elapsed = transfer_start.elapsed();
            //             println!("[GPU Join] Phase 1 complete: Data transfer in {:.3}s", transfer_elapsed.as_secs_f64());

            //             println!("[GPU Join] Phase 2: Building BVH spatial index on GPU");
            // let bvh_start = std::time::Instant::now();
            joiner.finish_building()?;
            // let bvh_elapsed = bvh_start.elapsed();
            //             println!("[GPU Join] Phase 2 complete: BVH index built in {:.3}s", bvh_elapsed.as_secs_f64());

            // Recreate context after building (required by libgpuspatial)
            let mut new_context = libgpuspatial_glue_bindgen::GpuSpatialJoinerContext {
                last_error: std::ptr::null(),
                private_data: std::ptr::null_mut(),
                build_indices: std::ptr::null_mut(),
                stream_indices: std::ptr::null_mut(),
            };
            joiner.create_context(&mut new_context);
            self.context = Some(new_context);
            let context = self.context.as_mut().unwrap();
            // Push stream data (right side) and perform join
            //             log::info!("DEBUG: Pushing {} geometries to GPU (stream side)", right_geom.len());
            //             log::info!("DEBUG: Right array data type: {:?}", right_geom.data_type());
            /*
            if let Some(binary_arr) = right_geom
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
            {
                //                 log::info!("DEBUG: Right binary array has {} values", binary_arr.len());
                if binary_arr.len() > 0 {
                     let first_wkb = binary_arr.value(0);
                                         log::info!("DEBUG: First right WKB length: {}, first bytes: {:?}",
                                             first_wkb.len(), &first_wkb[..8.min(first_wkb.len())]);
                }
            }
            */
            //             println!("[GPU Join] Phase 3: Executing spatial join kernel on GPU ({} right geometries)", right_geom.len());
            // let kernel_start = std::time::Instant::now();
            let gpu_predicate = predicate.into();
            joiner.push_stream(
                context,
                &right_geom,
                0,
                right_geom.len() as i64,
                gpu_predicate,
                0, // array_index_offset
            )?;

            // Get results
            let build_indices = joiner.get_build_indices_buffer(context).to_vec();
            let stream_indices = joiner.get_stream_indices_buffer(context).to_vec();
            // let kernel_elapsed = kernel_start.elapsed();

            //             println!("[GPU Join] Phase 3 complete: Join kernel executed in {:.3}s, {} result pairs",
            //                 kernel_elapsed.as_secs_f64(), build_indices.len());
            //             log::info!("DEBUG: Retrieved {} build indices, {} stream indices",
            //                 build_indices.len(), stream_indices.len());

            // Debug: Print first few and max indices
            /*
            if !build_indices.is_empty() {
                let max_build = *build_indices.iter().max().unwrap();
                let min_build = *build_indices.iter().min().unwrap();
                                 println!("DEBUG Rust: build_indices len={}, min={}, max={}, first 5={:?}",
                                     build_indices.len(), min_build, max_build, &build_indices[..5.min(build_indices.len())]);
            }
            if !stream_indices.is_empty() {
                let max_stream = *stream_indices.iter().max().unwrap();
                let min_stream = *stream_indices.iter().min().unwrap();
                                 println!("DEBUG Rust: stream_indices len={}, min={}, max={}, first 5={:?}",
                                     stream_indices.len(), min_stream, max_stream, &stream_indices[..5.min(stream_indices.len())]);
            }
            */
            Ok((build_indices, stream_indices))
        }
    }
}

/// Spatial predicates for GPU operations
#[repr(u32)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum SpatialPredicate {
    Equals = 0,
    Disjoint = 1,
    Touches = 2,
    Contains = 3,
    Covers = 4,
    Intersects = 5,
    Within = 6,
    CoveredBy = 7,
}

#[cfg(gpu_available)]
impl From<SpatialPredicate> for GpuSpatialPredicateWrapper {
    fn from(pred: SpatialPredicate) -> Self {
        match pred {
            SpatialPredicate::Equals => GpuSpatialPredicateWrapper::Equals,
            SpatialPredicate::Disjoint => GpuSpatialPredicateWrapper::Disjoint,
            SpatialPredicate::Touches => GpuSpatialPredicateWrapper::Touches,
            SpatialPredicate::Contains => GpuSpatialPredicateWrapper::Contains,
            SpatialPredicate::Covers => GpuSpatialPredicateWrapper::Covers,
            SpatialPredicate::Intersects => GpuSpatialPredicateWrapper::Intersects,
            SpatialPredicate::Within => GpuSpatialPredicateWrapper::Within,
            SpatialPredicate::CoveredBy => GpuSpatialPredicateWrapper::CoveredBy,
        }
    }
}

// Cleanup implementation
impl Drop for GpuSpatialContext {
    fn drop(&mut self) {
        #[cfg(gpu_available)]
        {
            if let (Some(mut joiner), Some(mut ctx)) = (self.joiner.take(), self.context.take()) {
                joiner.destroy_context(&mut ctx);
                joiner.release();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = GpuSpatialContext::new();
        #[cfg(gpu_available)]
        assert!(ctx.is_ok());
        #[cfg(not(gpu_available))]
        assert!(ctx.is_err());
    }
}
