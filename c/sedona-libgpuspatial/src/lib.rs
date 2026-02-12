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
// Re-export to the users
pub use error::GpuSpatialError;
pub use options::GpuSpatialOptions;
pub use predicate::GpuSpatialRelationPredicate;

#[cfg(gpu_available)]
mod sys {
    use super::libgpuspatial;
    use super::libgpuspatial_glue_bindgen;
    use super::*;

    pub type Result<T> = std::result::Result<T, GpuSpatialError>;

    use std::sync::{Arc, Mutex};

    use libgpuspatial::{
        FloatIndex2DBuilder, GpuSpatialRefinerWrapper, GpuSpatialRuntimeWrapper, SharedFloatIndex2D,
    };
    // To be used in the global state. We need to ensure these are Send + Sync since they will be shared across threads.
    unsafe impl Send for libgpuspatial_glue_bindgen::GpuSpatialRuntime {}
    unsafe impl Sync for libgpuspatial_glue_bindgen::GpuSpatialRuntime {}

    // -- Global State --
    static GLOBAL_GPUSPATIAL_RUNTIME: Mutex<Option<Arc<Mutex<GpuSpatialRuntimeWrapper>>>> =
        Mutex::new(None);

    // -- The Actual Implementation Struct --
    pub struct SpatialImpl {
        runtime: Option<Arc<Mutex<GpuSpatialRuntimeWrapper>>>,
        // Index state is either building (Builder) or built (Shared)
        index_builder: Option<FloatIndex2DBuilder>,
        index: Option<SharedFloatIndex2D>,
        refiner: Option<GpuSpatialRefinerWrapper>,
    }

    impl SpatialImpl {
        pub fn new() -> Result<Self> {
            Ok(Self {
                runtime: None,
                index_builder: None,
                index: None,
                refiner: None,
            })
        }

        pub fn init(&mut self, options: GpuSpatialOptions) -> Result<()> {
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

            let runtime_ref = global_runtime_guard.as_ref().unwrap().clone();
            self.runtime = Some(runtime_ref);

            // Create the builder
            let builder = FloatIndex2DBuilder::try_new(
                self.runtime.as_ref().unwrap().clone(),
                options.concurrency,
            )?;
            self.index_builder = Some(builder);
            self.index = None; // Reset built index if any

            // Create refiner
            let refiner = GpuSpatialRefinerWrapper::try_new(
                self.runtime.as_ref().unwrap().clone(),
                options.concurrency,
                options.compress_bvh,
                options.pipeline_batches,
            )?;
            self.refiner = Some(refiner);

            Ok(())
        }

        pub fn index_clear(&mut self) -> Result<()> {
            // We can only clear the builder. If we have a finished index, we can't clear it.
            // If the user wants to restart, they should conceptually re-init or we need
            // to store config to recreate the builder.
            // For now, assuming this is called during build phase.
            if let Some(builder) = self.index_builder.as_mut() {
                builder.clear();
                Ok(())
            } else {
                Err(GpuSpatialError::Init(
                    "Cannot clear index: Index is already built or not initialized".into(),
                ))
            }
        }

        pub fn index_push_build(&mut self, rects: &[Rect<f32>]) -> Result<()> {
            let builder = self
                .index_builder
                .as_mut()
                .ok_or_else(|| GpuSpatialError::Init("GPU index builder not available".into()))?;

            unsafe { builder.push_build(rects.as_ptr() as *const f32, rects.len() as u32) }
        }

        pub fn index_finish_building(&mut self) -> Result<()> {
            // Take the builder out, consume it, and store the shared index
            let builder = self
                .index_builder
                .take()
                .ok_or_else(|| GpuSpatialError::Init("GPU index builder not available".into()))?;

            let shared_index = builder.finish()?;
            self.index = Some(shared_index);
            Ok(())
        }

        pub fn probe(&self, rects: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>)> {
            let index = self
                .index
                .as_ref()
                .ok_or_else(|| GpuSpatialError::Init("GPU index not built".into()))?;

            // Create a thread-local context wrapper
            let mut ctx_wrapper = index.create_context()?;

            unsafe {
                ctx_wrapper.probe(rects.as_ptr() as *const f32, rects.len() as u32)?;
            }

            let build_indices = ctx_wrapper.get_build_indices_buffer().to_vec();
            let probe_indices = ctx_wrapper.get_probe_indices_buffer().to_vec();

            Ok((build_indices, probe_indices))
            // ctx_wrapper is dropped here, which calls destroy_context in C
        }

        pub fn refiner_clear(&mut self) -> Result<()> {
            let refiner = self
                .refiner
                .as_mut()
                .ok_or_else(|| GpuSpatialError::Init("GPU refiner is not available".into()))?;
            refiner.clear();
            Ok(())
        }

        pub fn refiner_init_schema(
            &mut self,
            build_data_type: &DataType,
            probe_data_type: &DataType,
        ) -> Result<()> {
            let refiner = self
                .refiner
                .as_mut()
                .ok_or_else(|| GpuSpatialError::Init("GPU refiner not available".into()))?;
            refiner.init_schema(build_data_type, probe_data_type)
        }

        pub fn refiner_push_build(&mut self, array: &arrow_array::ArrayRef) -> Result<()> {
            let refiner = self
                .refiner
                .as_mut()
                .ok_or_else(|| GpuSpatialError::Init("GPU refiner not available".into()))?;
            refiner.push_build(array)
        }

        pub fn refiner_finish_building(&mut self) -> Result<()> {
            let refiner = self
                .refiner
                .as_mut()
                .ok_or_else(|| GpuSpatialError::Init("GPU refiner not available".into()))?;
            refiner.finish_building()
        }

        pub fn refine(
            &self,
            probe_array: &arrow_array::ArrayRef,
            predicate: GpuSpatialRelationPredicate,
            build_indices: &mut Vec<u32>,
            probe_indices: &mut Vec<u32>,
        ) -> Result<()> {
            let refiner = self
                .refiner
                .as_ref()
                .ok_or_else(|| GpuSpatialError::Init("GPU refiner not available".into()))?;

            refiner.refine(probe_array, predicate, build_indices, probe_indices)
        }
    }
}

#[cfg(not(gpu_available))]
mod sys {
    use super::*;

    pub struct SpatialImpl;

    impl SpatialImpl {
        pub fn new() -> std::result::Result<Self, GpuSpatialError> {
            Err(GpuSpatialError::GpuNotAvailable)
        }

        pub fn init(&mut self, _options: GpuSpatialOptions) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn index_clear(&mut self) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn index_push_build(&mut self, _rects: &[Rect<f32>]) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn index_finish_building(&mut self) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn probe(&self, _rects: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>)> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn refiner_clear(&mut self) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn refiner_init_schema(&mut self, _build: &DataType, _probe: &DataType) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn refiner_push_build(&mut self, _array: &arrow_array::ArrayRef) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn refiner_finish_building(&mut self) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        pub fn refine(
            &self,
            _probe: &arrow_array::ArrayRef,
            _pred: GpuSpatialRelationPredicate,
            _bi: &mut Vec<u32>,
            _pi: &mut Vec<u32>,
        ) -> Result<()> {
            Err(GpuSpatialError::GpuNotAvailable)
        }
    }
}

/// High-level wrapper for GPU spatial operations
pub struct GpuSpatial {
    inner: sys::SpatialImpl,
}

impl GpuSpatial {
    pub fn new() -> Result<Self, GpuSpatialError> {
        Ok(Self {
            inner: sys::SpatialImpl::new()?,
        })
    }

    pub fn init(&mut self, options: GpuSpatialOptions) -> Result<(), GpuSpatialError> {
        self.inner.init(options)
    }

    pub fn index_clear(&mut self) -> Result<(), GpuSpatialError> {
        self.inner.index_clear()
    }

    pub fn index_push_build(&mut self, rects: &[Rect<f32>]) -> Result<(), GpuSpatialError> {
        self.inner.index_push_build(rects)
    }

    pub fn index_finish_building(&mut self) -> Result<(), GpuSpatialError> {
        self.inner.index_finish_building()
    }

    pub fn probe(&self, rects: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>), GpuSpatialError> {
        self.inner.probe(rects)
    }

    pub fn refiner_clear(&mut self) -> Result<(), GpuSpatialError> {
        self.inner.refiner_clear()
    }

    pub fn refiner_init_schema(
        &mut self,
        build_data_type: &DataType,
        probe_data_type: &DataType,
    ) -> Result<(), GpuSpatialError> {
        self.inner
            .refiner_init_schema(build_data_type, probe_data_type)
    }

    pub fn refiner_push_build(
        &mut self,
        array: &arrow_array::ArrayRef,
    ) -> Result<(), GpuSpatialError> {
        self.inner.refiner_push_build(array)
    }

    pub fn refiner_finish_building(&mut self) -> Result<(), GpuSpatialError> {
        self.inner.refiner_finish_building()
    }

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
    use arrow_array::Array;
    use geo::{BoundingRect, Intersects, Point, Polygon};
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_geos::register::scalar_kernels;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::{Edges, SedonaType, WKB_GEOMETRY};
    use sedona_testing::create::create_array_storage;
    use sedona_testing::testers::ScalarUdfTester;
    use wkt::TryFromWkt;

    pub fn compute_expected_intersections(
        vec_a: &[Rect<f32>],
        vec_b: &[Rect<f32>],
    ) -> (Vec<u32>, Vec<u32>) {
        let mut ids_a = Vec::new();
        let mut ids_b = Vec::new();

        for (i, rect_a) in vec_a.iter().enumerate() {
            for (j, rect_b) in vec_b.iter().enumerate() {
                if rect_a.intersects(rect_b) {
                    ids_a.push(i as u32);
                    ids_b.push(j as u32);
                }
            }
        }
        (ids_a, ids_b)
    }

    fn compute_expected_pip_results(
        polygons: &[Option<&str>],
        points: &[Option<&str>],
    ) -> (Vec<u32>, Vec<u32>) {
        let kernels = scalar_kernels();
        let st_intersects = kernels
            .into_iter()
            .find(|(name, _)| *name == "st_intersects")
            .map(|(_, kernel_ref)| kernel_ref)
            .unwrap();

        let sedona_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let udf = SedonaScalarUDF::from_impl("st_intersects", st_intersects);
        let tester =
            ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type.clone()]);

        let mut ans_build_indices: Vec<u32> = Vec::new();
        let mut ans_probe_indices: Vec<u32> = Vec::new();

        for (poly_index, poly) in polygons.iter().enumerate() {
            for (point_index, point) in points.iter().enumerate() {
                let result = tester
                    .invoke_scalar_scalar(poly.unwrap(), point.unwrap())
                    .unwrap();
                if result == true.into() {
                    ans_build_indices.push(poly_index as u32);
                    ans_probe_indices.push(point_index as u32);
                }
            }
        }
        ans_build_indices.sort();
        ans_probe_indices.sort();
        (ans_build_indices, ans_probe_indices)
    }

    #[test]
    fn test_spatial_index() {
        let mut gs = GpuSpatial::new().unwrap();
        let options = GpuSpatialOptions {
            concurrency: 1,
            device_id: 0,
            compress_bvh: false,
            pipeline_batches: 1,
            cuda_use_memory_pool: true,
            cuda_memory_pool_init_percent: 10,
        };
        gs.init(options).expect("Failed to initialize GpuSpatial");

        let polygon_values = &[
            Some("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
            Some("POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))"),
            Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 3 2, 3 3, 2 3, 2 2), (6 6, 8 6, 8 8, 6 8, 6 6))"),
            Some("POLYGON ((30 0, 60 20, 50 50, 10 50, 0 20, 30 0), (20 30, 25 40, 15 40, 20 30), (30 30, 35 40, 25 40, 30 30), (40 30, 45 40, 35 40, 40 30))"),
            Some("POLYGON ((40 0, 50 30, 80 20, 90 70, 60 90, 30 80, 20 40, 40 0), (50 20, 65 30, 60 50, 45 40, 50 20), (30 60, 50 70, 45 80, 30 60))"),
        ];
        let rects: Vec<Rect<f32>> = polygon_values
            .iter()
            .filter_map(|opt_wkt| {
                let wkt_str = opt_wkt.as_ref()?;
                let polygon: Polygon<f32> = Polygon::try_from_wkt_str(wkt_str).ok()?;
                polygon.bounding_rect()
            })
            .collect();
        gs.index_push_build(&rects)
            .expect("Failed to push build data");
        gs.index_finish_building()
            .expect("Failed to finish building");
        let point_values = &[
            Some("POINT (30 20)"),
            Some("POINT (20 20)"),
            Some("POINT (1 1)"),
            Some("POINT (70 70)"),
            Some("POINT (55 35)"),
        ];
        let points: Vec<Rect<f32>> = point_values
            .iter()
            .map(|opt_wkt| -> Rect<f32> {
                let wkt_str = opt_wkt.unwrap();
                let point: Point<f32> = Point::try_from_wkt_str(wkt_str).ok().unwrap();
                point.bounding_rect()
            })
            .collect();
        let (mut build_indices, mut probe_indices) = gs.probe(&points).unwrap();
        build_indices.sort();
        probe_indices.sort();

        let (mut ans_build_indices, mut ans_probe_indices) =
            compute_expected_intersections(&rects, &points);

        ans_build_indices.sort();
        ans_probe_indices.sort();

        assert_eq!(build_indices, ans_build_indices);
        assert_eq!(probe_indices, ans_probe_indices);
    }
    #[test]
    fn test_spatial_refiner() {
        let mut gs = GpuSpatial::new().unwrap();
        let options = GpuSpatialOptions {
            concurrency: 1,
            device_id: 0,
            compress_bvh: false,
            pipeline_batches: 1,
            cuda_use_memory_pool: true,
            cuda_memory_pool_init_percent: 10,
        };
        gs.init(options).expect("Failed to initialize GpuSpatial");

        let polygon_values = &[
            Some("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
            Some("POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))"),
            Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 3 2, 3 3, 2 3, 2 2), (6 6, 8 6, 8 8, 6 8, 6 6))"),
            Some("POLYGON ((30 0, 60 20, 50 50, 10 50, 0 20, 30 0), (20 30, 25 40, 15 40, 20 30), (30 30, 35 40, 25 40, 30 30), (40 30, 45 40, 35 40, 40 30))"),
            Some("POLYGON ((40 0, 50 30, 80 20, 90 70, 60 90, 30 80, 20 40, 40 0), (50 20, 65 30, 60 50, 45 40, 50 20), (30 60, 50 70, 45 80, 30 60))"),
        ];

        let polygons = create_array_storage(polygon_values, &WKB_GEOMETRY);

        let rects: Vec<Rect<f32>> = polygon_values
            .iter()
            .map(|opt_wkt| -> Rect<f32> {
                let wkt_str = opt_wkt.unwrap();
                let polygon: Polygon<f32> = Polygon::try_from_wkt_str(wkt_str).ok().unwrap();
                polygon.bounding_rect().unwrap()
            })
            .collect();
        gs.index_push_build(&rects)
            .expect("Failed to push build data");
        gs.index_finish_building()
            .expect("Failed to finish building");

        let point_values = &[
            Some("POINT (30 20)"),
            Some("POINT (20 20)"),
            Some("POINT (1 1)"),
            Some("POINT (70 70)"),
            Some("POINT (55 35)"),
        ];
        let points = create_array_storage(point_values, &WKB_GEOMETRY);
        let point_rects: Vec<Rect<f32>> = point_values
            .iter()
            .map(|wkt| -> Rect<f32> {
                let wkt_str = wkt.unwrap();
                let point: Point<f32> = Point::try_from_wkt_str(wkt_str).unwrap();
                point.bounding_rect()
            })
            .collect();

        // 1. Get GPU Results
        let (mut build_indices, mut probe_indices) = gs.probe(&point_rects).unwrap();

        gs.refiner_init_schema(polygons.data_type(), points.data_type())
            .expect("Failed to init schema");
        gs.refiner_push_build(&polygons)
            .expect("Failed to push build");
        gs.refiner_finish_building()
            .expect("Failed to finish building refiner");
        gs.refine(
            &points,
            GpuSpatialRelationPredicate::Intersects,
            &mut build_indices,
            &mut probe_indices,
        )
        .expect("Failed to refine results");

        build_indices.sort();
        probe_indices.sort();

        // 2. Get CPU Expected Results
        let (ans_build_indices, ans_probe_indices) =
            compute_expected_pip_results(polygon_values, point_values);

        // 3. Compare
        assert_eq!(build_indices, ans_build_indices);
        assert_eq!(probe_indices, ans_probe_indices);
    }
}
