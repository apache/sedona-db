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
use geo::Rect;

mod error;
#[cfg(gpu_available)]
mod libgpuspatial;
mod libgpuspatial_glue_bindgen;
mod options;
mod predicate;

pub use error::GpuSpatialError;
pub use options::GpuSpatialOptions;
pub use predicate::GpuSpatialRelationPredicate;

#[cfg(gpu_available)]
mod sys {
    use super::libgpuspatial;
    use super::libgpuspatial_glue_bindgen;
    use super::*;
    use std::sync::{Arc, Mutex};

    pub type Result<T> = std::result::Result<T, GpuSpatialError>;

    // Direct type aliases to C++ wrappers
    pub type IndexBuilderInner = libgpuspatial::FloatIndex2DBuilder;
    pub type IndexInner = libgpuspatial::SharedFloatIndex2D;

    // Refiner aliases
    pub type RefinerBuilderInner = libgpuspatial::GpuSpatialRefinerBuilder;
    pub type RefinerInner = libgpuspatial::GpuSpatialRefinerWrapper;

    use libgpuspatial::GpuSpatialRuntimeWrapper;

    // Global Runtime State
    unsafe impl Send for libgpuspatial_glue_bindgen::GpuSpatialRuntime {}
    unsafe impl Sync for libgpuspatial_glue_bindgen::GpuSpatialRuntime {}

    static GLOBAL_GPUSPATIAL_RUNTIME: Mutex<Option<Arc<Mutex<GpuSpatialRuntimeWrapper>>>> =
        Mutex::new(None);

    /// Handles initialization of the GPU runtime.
    pub struct SpatialContext {
        runtime: Arc<Mutex<GpuSpatialRuntimeWrapper>>,
    }

    impl SpatialContext {
        pub fn try_new(options: &GpuSpatialOptions) -> Result<Self> {
            let mut global_runtime_guard = GLOBAL_GPUSPATIAL_RUNTIME.lock().unwrap();

            if global_runtime_guard.is_none() {
                let out_path = std::path::PathBuf::from(env!("OUT_DIR"));
                let ptx_root = out_path.join("share/gpuspatial/shaders");
                let ptx_root_str = ptx_root
                    .to_str()
                    .ok_or_else(|| GpuSpatialError::Init("Invalid PTX path".to_string()))?;

                let runtime = GpuSpatialRuntimeWrapper::try_new(
                    options.device_id,
                    ptx_root_str,
                    options.cuda_use_memory_pool,
                    options.cuda_memory_pool_init_percent,
                )?;
                *global_runtime_guard = Some(Arc::new(Mutex::new(runtime)));
            }

            Ok(Self {
                runtime: global_runtime_guard.as_ref().unwrap().clone(),
            })
        }

        pub fn runtime(&self) -> Arc<Mutex<GpuSpatialRuntimeWrapper>> {
            self.runtime.clone()
        }
    }

    pub struct IndexBuilderImpl {
        inner: IndexBuilderInner,
    }

    impl IndexBuilderImpl {
        pub fn try_new(options: &GpuSpatialOptions) -> Result<Self> {
            let ctx = SpatialContext::try_new(options)?;
            let inner = IndexBuilderInner::try_new(ctx.runtime(), options.concurrency)?;
            Ok(Self { inner })
        }

        pub fn clear(&mut self) {
            self.inner.clear()
        }
        pub fn push_build(&mut self, rects: &[Rect<f32>]) -> Result<()> {
            unsafe {
                self.inner
                    .push_build(rects.as_ptr() as *const f32, rects.len() as u32)
            }
        }

        pub fn finish_building(self) -> Result<IndexImpl> {
            let index = self.inner.finish()?;
            Ok(IndexImpl { inner: index })
        }
    }

    pub struct IndexImpl {
        inner: IndexInner,
    }

    impl IndexImpl {
        pub fn probe(&self, rects: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>)> {
            // 1. Create a thread-local context
            let mut ctx = self.inner.create_context()?;

            // 2. Perform the probe using the context
            unsafe {
                self.inner
                    .probe(&mut ctx, rects.as_ptr() as *const f32, rects.len() as u32)?;
            }

            // 3. Extract results from the context
            Ok((
                ctx.get_build_indices_buffer().to_vec(),
                ctx.get_probe_indices_buffer().to_vec(),
            ))
        }
    }

    pub struct RefinerBuilderImpl {
        inner: RefinerBuilderInner,
    }

    impl RefinerBuilderImpl {
        pub fn try_new(options: &GpuSpatialOptions) -> Result<Self> {
            let ctx = SpatialContext::try_new(options)?;
            let inner = RefinerBuilderInner::try_new(
                ctx.runtime(),
                options.concurrency,
                options.compress_bvh,
                options.pipeline_batches,
            )?;
            Ok(Self { inner })
        }

        pub fn init_schema(&mut self, build: &DataType, probe: &DataType) -> Result<()> {
            self.inner.init_schema(build, probe)
        }

        pub fn clear(&mut self) {
            self.inner.clear()
        }
        pub fn push_build(&mut self, array: &arrow_array::ArrayRef) -> Result<()> {
            self.inner.push_build(array)
        }

        pub fn finish_building(self) -> Result<RefinerImpl> {
            let refiner_wrapper = self.inner.finish_building()?;
            Ok(RefinerImpl {
                inner: refiner_wrapper,
            })
        }
    }

    pub struct RefinerImpl {
        inner: RefinerInner,
    }

    impl RefinerImpl {
        pub fn refine(
            &self,
            probe: &arrow_array::ArrayRef,
            pred: GpuSpatialRelationPredicate,
            bi: &mut Vec<u32>,
            pi: &mut Vec<u32>,
        ) -> Result<()> {
            self.inner.refine(probe, pred, bi, pi)
        }
    }
}

#[cfg(not(gpu_available))]
mod sys {
    use super::*;
    pub type Result<T> = std::result::Result<T, crate::error::GpuSpatialError>;

    pub struct IndexBuilderImpl;
    pub struct IndexImpl;
    pub struct RefinerBuilderImpl;
    pub struct RefinerImpl;

    impl IndexBuilderImpl {
        pub fn try_new(_opts: &GpuSpatialOptions) -> Result<Self> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn clear(&mut self) {}
        pub fn push_build(&mut self, _r: &[Rect<f32>]) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn finish_building(self) -> Result<IndexImpl> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
    }
    impl IndexImpl {
        pub fn probe(&self, _r: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>)> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
    }
    impl RefinerBuilderImpl {
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
        pub fn finish_building(self) -> Result<RefinerImpl> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
    }
    impl RefinerImpl {
        pub fn refine(
            &self,
            _p: &arrow_array::ArrayRef,
            _pr: GpuSpatialRelationPredicate,
            _bi: &mut Vec<u32>,
            _pi: &mut Vec<u32>,
        ) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
    }
}

/// Builder for creating a GPU Spatial Index.
pub struct GpuSpatialIndexBuilder {
    inner: sys::IndexBuilderImpl,
}

impl GpuSpatialIndexBuilder {
    /// Create a new builder with the specified options.
    pub fn try_new(options: GpuSpatialOptions) -> Result<Self, GpuSpatialError> {
        Ok(Self {
            inner: sys::IndexBuilderImpl::try_new(&options)?,
        })
    }
    /// Clear any existing data from the builder.
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Add bounding boxes to the index construction.
    pub fn push_build(&mut self, rects: &[Rect<f32>]) -> Result<(), GpuSpatialError> {
        self.inner.push_build(rects)
    }

    /// Finalize construction and return the immutable, queryable Index.
    pub fn finish_building(self) -> Result<GpuSpatialIndex, GpuSpatialError> {
        Ok(GpuSpatialIndex {
            inner: self.inner.finish_building()?,
        })
    }
}

/// A constructed, immutable GPU Spatial Index ready for probing.
pub struct GpuSpatialIndex {
    inner: sys::IndexImpl,
}

impl GpuSpatialIndex {
    /// Probe the index with query rectangles.
    pub fn probe(&self, rects: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>), GpuSpatialError> {
        self.inner.probe(rects)
    }
}

/// Builder for creating a GPU Spatial Refiner.
pub struct GpuSpatialRefinerBuilder {
    inner: sys::RefinerBuilderImpl,
}

impl GpuSpatialRefinerBuilder {
    /// Create a new refiner builder with the specified options.
    pub fn try_new(options: GpuSpatialOptions) -> Result<Self, GpuSpatialError> {
        Ok(Self {
            inner: sys::RefinerBuilderImpl::try_new(&options)?,
        })
    }

    /// Initialize the refiner schema with the build and probe data types.
    pub fn init_schema(
        &mut self,
        build_data_type: &DataType,
        probe_data_type: &DataType,
    ) -> Result<(), GpuSpatialError> {
        self.inner.init_schema(build_data_type, probe_data_type)
    }
    /// Clear any existing build-side data from the builder.
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// Add build-side data to the refiner construction.
    pub fn push_build(&mut self, array: &arrow_array::ArrayRef) -> Result<(), GpuSpatialError> {
        self.inner.push_build(array)
    }

    /// Finalize construction and return the immutable, queryable Refiner.
    pub fn finish_building(self) -> Result<GpuSpatialRefiner, GpuSpatialError> {
        Ok(GpuSpatialRefiner {
            inner: self.inner.finish_building()?,
        })
    }
}

pub struct GpuSpatialRefiner {
    inner: sys::RefinerImpl,
}

impl GpuSpatialRefiner {
    /// Refine candidate pairs using the specified spatial predicate.
    pub fn refine(
        &self,
        probe_array: &arrow_array::ArrayRef,
        predicate: GpuSpatialRelationPredicate,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<(), GpuSpatialError> {
        self.inner
            .refine(probe_array, predicate, build_indices, probe_indices)
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
        let mut builder =
            GpuSpatialIndexBuilder::try_new(options).expect("Failed to create builder");

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
        builder.push_build(&rects).expect("Failed to insert");

        // 3. Finish (Consumes Builder -> Returns Index)
        let index = builder
            .finish_building()
            .expect("Failed to finish building");

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
        let mut builder =
            GpuSpatialRefinerBuilder::try_new(options).expect("Failed to create refiner builder");

        let polygon_values = &[Some("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))")];
        let polygons = create_array_storage(polygon_values, &WKB_GEOMETRY);

        let point_values = &[Some("POINT (30 20)")];
        let points = create_array_storage(point_values, &WKB_GEOMETRY);

        // 2. Build Refiner
        builder
            .init_schema(polygons.data_type(), points.data_type())
            .unwrap();
        builder.push_build(&polygons).unwrap();

        // 3. Finish (Consumes Builder -> Returns Refiner)
        let refiner = builder.finish_building().expect("Failed to finish refiner");

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
