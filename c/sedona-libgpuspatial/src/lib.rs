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

use arrow_schema::DataType;
use geo_types::Rect;

mod error;
#[cfg(gpu_available)]
mod libgpuspatial;
mod libgpuspatial_glue_bindgen;
mod options;
mod predicate;

pub use error::GpuSpatialError;
pub use options::GpuSpatialOptions;
pub use predicate::GpuSpatialRelationPredicate;
pub use sys::{GpuSpatialIndex, GpuSpatialRefiner};

#[cfg(gpu_available)]
mod sys {
    use super::libgpuspatial;
    use super::*;
    use libgpuspatial::GpuSpatialRuntimeWrapper;
    use std::sync::{Arc, Mutex};

    pub type Result<T> = std::result::Result<T, GpuSpatialError>;

    // Global Runtime State
    unsafe impl Send for GpuSpatialRuntimeWrapper {}
    unsafe impl Sync for GpuSpatialRuntimeWrapper {}

    static GLOBAL_GPUSPATIAL_RUNTIME: Mutex<Option<Arc<GpuSpatialRuntimeWrapper>>> =
        Mutex::new(None);
    /// Handles initialization of the GPU runtime.
    pub struct SpatialContext {
        runtime: Arc<GpuSpatialRuntimeWrapper>,
    }

    impl SpatialContext {
        pub fn try_new(options: &GpuSpatialOptions) -> Result<Self> {
            // Lock the mutex globally
            let mut guard = GLOBAL_GPUSPATIAL_RUNTIME
                .lock()
                .map_err(|_| GpuSpatialError::Init("Global mutex poisoned".into()))?;

            // Check if it already exists
            if let Some(existing_runtime) = guard.as_ref() {
                if existing_runtime.device_id != options.device_id {
                    return Err(GpuSpatialError::Init(format!(
                        "Runtime conflict: Initialized on Device {}, requested Device {}.",
                        existing_runtime.device_id, options.device_id
                    )));
                }
                // Return the existing one
                return Ok(Self {
                    runtime: existing_runtime.clone(),
                });
            }

            let out_path = std::path::PathBuf::from(env!("OUT_DIR"));
            let ptx_root = out_path.join("share/gpuspatial/shaders");
            let ptx_root_str = ptx_root
                .to_str()
                .ok_or_else(|| GpuSpatialError::Init("Invalid PTX path".to_string()))?;

            let wrapper = libgpuspatial::GpuSpatialRuntimeWrapper::try_new(
                options.device_id,
                ptx_root_str,
                options.cuda_use_memory_pool,
                options.cuda_memory_pool_init_percent,
            )?;

            let arc_wrapper = Arc::new(wrapper);

            *guard = Some(arc_wrapper.clone());

            Ok(Self {
                runtime: arc_wrapper,
            })
        }
    }

    /// A GPU-accelerated spatial index for 2D rectangles in FP32.
    /// Once built, the index is immutable and can be safely shared across threads for read-only probe operations.
    pub struct GpuSpatialIndex {
        inner: libgpuspatial::FloatIndex2D,
    }

    impl GpuSpatialIndex {
        /// Creates a new GPU spatial index with the specified options.
        /// This initializes the GPU runtime if it hasn't been initialized yet.
        pub fn try_new(options: &GpuSpatialOptions) -> Result<Self> {
            let ctx = SpatialContext::try_new(options)?;
            let inner = libgpuspatial::FloatIndex2D::try_new(ctx.runtime, options.concurrency)?;
            Ok(Self { inner })
        }

        /// Clears any previously inserted data from the builder, allowing it to be reused for building a new index.
        pub fn clear(&mut self) {
            self.inner.clear()
        }

        /// Inserts a batch of bounding boxes into the index.
        /// Each rectangle is represented as a `Rect<f32>` with minimum and maximum x and y coordinates.
        /// This method accumulates these rectangles until `finish_building` is called to finalize the index.
        /// The method can be called multiple times to insert data in batches before finalizing.
        pub fn push_build(&mut self, rects: &[Rect<f32>]) -> Result<()> {
            // Re-interpreting Rect<f32> as flat f32 array (xmin, ymin, xmax, ymax)
            let raw_ptr = rects.as_ptr() as *const f32;
            self.inner.push_build(raw_ptr, rects.len() as u32)
        }

        /// Finalizes the building process and returns an immutable spatial index that can be probed.
        pub fn finish_building(&mut self) -> Result<()> {
            self.inner.finish_building()
        }

        /// Probes the spatial index with a batch of rectangles and returns pairs of matching indices from the build and probe sets.
        pub fn probe(&self, rects: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>)> {
            let raw_ptr = rects.as_ptr() as *const f32;
            self.inner.probe(raw_ptr, rects.len() as u32)
        }
    }

    /// A GPU-accelerated spatial refiner that can perform various spatial relation tests (e.g., Intersects, Contains) between two sets of geometries.
    pub struct GpuSpatialRefiner {
        inner: libgpuspatial::Refiner,
    }

    impl GpuSpatialRefiner {
        /// Creates a new GPU spatial refiner with the specified options.
        /// This initializes the GPU runtime if it hasn't been initialized yet.
        pub fn try_new(options: &GpuSpatialOptions) -> Result<Self> {
            let ctx = SpatialContext::try_new(options)?;
            let inner = libgpuspatial::Refiner::try_new(
                ctx.runtime,
                options.concurrency,
                options.compress_bvh,
                options.pipeline_batches,
            )?;
            Ok(Self { inner })
        }

        /// Initializes the schema for the refiner based on the data types of the build and probe geometries.
        /// This allows the refiner to understand how to interpret the geometry data for both sets.
        pub fn init_schema(&mut self, build: &DataType, probe: &DataType) -> Result<()> {
            self.inner.init_schema(build, probe)
        }

        /// Clears any previously inserted data from the refiner, allowing it to be reused for building a new set of geometries.
        pub fn clear(&mut self) {
            self.inner.clear()
        }

        /// Inserts a batch of geometries into the refiner for the build side.
        /// The geometries are provided as an Arrow array reference, and the refiner will process them according to the initialized schema.
        /// This method accumulates these geometries until `finish_building` is called to finalize the refiner.
        pub fn push_build(&mut self, array: &arrow_array::ArrayRef) -> Result<()> {
            self.inner.push_build(array)
        }

        /// Finalizes the building process and prepares the refiner for refinement operations.
        /// After this call, the refiner is ready to perform spatial relation tests against probe geometries.
        pub fn finish_building(&mut self) -> Result<()> {
            self.inner.finish_building()
        }

        /// Refines the candidate pairs of geometries based on the specified spatial relation predicate.
        /// The probe geometries are provided as an Arrow array reference,
        /// and the method updates the provided vectors of build and probe indices to
        /// include only those pairs that satisfy the spatial relation predicate.
        pub fn refine(
            &self,
            probe: &arrow_array::ArrayRef,
            pred: GpuSpatialRelationPredicate,
            build_indices: &mut Vec<u32>,
            probe_indices: &mut Vec<u32>,
        ) -> Result<()> {
            self.inner.refine(probe, pred, build_indices, probe_indices)
        }
    }
}

#[cfg(not(gpu_available))]
mod sys {
    use super::*;
    pub type Result<T> = std::result::Result<T, crate::error::GpuSpatialError>;

    pub struct GpuSpatialIndex;
    pub struct GpuSpatialRefiner;

    impl GpuSpatialIndex {
        pub fn try_new(_opts: &GpuSpatialOptions) -> Result<Self> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn clear(&mut self) {}
        pub fn push_build(&mut self, _r: &[Rect<f32>]) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn finish_building(self) -> Result<GpuSpatialIndex> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn probe(&self, _r: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>)> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
    }

    impl GpuSpatialRefiner {
        pub fn try_new(_opts: &GpuSpatialOptions) -> Result<Self> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn init_schema(&mut self, _b: &DataType, _p: &DataType) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn clear(&mut self) {}
        pub fn push_build(&mut self, _arr: &arrow_array::ArrayRef) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn finish_building(&mut self) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn refine(
            &self,
            _p: &arrow_array::ArrayRef,
            _pr: GpuSpatialRelationPredicate,
            _build_indices: &mut Vec<u32>,
            _probe_indices: &mut Vec<u32>,
        ) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
    }
}

#[cfg(gpu_available)]
#[cfg(test)]
mod tests {
    use super::*;
    use geo::{BoundingRect, Point, Polygon};
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_array_storage;
    use wkt::TryFromWkt;

    #[test]
    fn test_spatial_index() {
        let options = GpuSpatialOptions {
            concurrency: 1,
            device_id: 0,
            compress_bvh: false,
            pipeline_batches: 1,
            cuda_use_memory_pool: true,
            cuda_memory_pool_init_percent: 10,
        };

        // 1. Create Builder
        let mut index = GpuSpatialIndex::try_new(&options).expect("Failed to create builder");

        let polygon_values = &[
            Some("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
            Some("POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))"),
        ];
        let rects: Vec<Rect<f32>> = polygon_values
            .iter()
            .map(|w| {
                Polygon::try_from_wkt_str(w.unwrap())
                    .unwrap()
                    .bounding_rect()
                    .unwrap()
            })
            .collect();

        // 2. Insert Data
        index.push_build(&rects).expect("Failed to insert");

        // 3. Finish (Consumes Builder -> Returns Index)
        index.finish_building().expect("Failed to finish building");

        // 4. Probe (Index is immutable and safe)
        let point_values = &[Some("POINT (30 20)")];
        let points: Vec<Rect<f32>> = point_values
            .iter()
            .map(|w| Point::try_from_wkt_str(w.unwrap()).unwrap().bounding_rect())
            .collect();

        let (build_idx, probe_idx) = index.probe(&points).unwrap();

        assert!(!build_idx.is_empty());
        assert_eq!(build_idx.len(), probe_idx.len());
    }

    #[test]
    fn test_spatial_refiner() {
        let options = GpuSpatialOptions {
            concurrency: 1,
            device_id: 0,
            compress_bvh: false,
            pipeline_batches: 1,
            cuda_use_memory_pool: true,
            cuda_memory_pool_init_percent: 10,
        };

        // 1. Create Refiner Builder
        let mut refiner =
            GpuSpatialRefiner::try_new(&options).expect("Failed to create refiner builder");

        let polygon_values = &[Some("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")];
        let polygons = create_array_storage(polygon_values, &WKB_GEOMETRY);

        let point_values = &[Some("POINT (30 20)")];
        let points = create_array_storage(point_values, &WKB_GEOMETRY);

        // 2. Build Refiner
        refiner
            .init_schema(polygons.data_type(), points.data_type())
            .unwrap();

        refiner.push_build(&polygons).unwrap();

        // 3. Finish (Consumes Builder -> Returns Refiner)
        refiner.finish_building().expect("Failed to finish refiner");

        // 4. Use Refiner
        let mut build_idx = vec![0];
        let mut probe_idx = vec![0];

        refiner
            .refine(
                &points,
                GpuSpatialRelationPredicate::Intersects,
                &mut build_idx,
                &mut probe_idx,
            )
            .unwrap();
    }
}
