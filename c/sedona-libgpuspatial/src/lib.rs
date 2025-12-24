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

#[cfg(gpu_available)]
use std::sync::{Arc, Mutex};
// Import Array trait for len() method (used in gpu_available code)
use geo::Rect;
// Re-exports for GPU functionality
#[cfg(gpu_available)]
pub use error::GpuSpatialError;
#[cfg(gpu_available)]
pub use libgpuspatial::{
    GpuSpatialIndexFloat2DWrapper, GpuSpatialRTEngineWrapper, GpuSpatialRefinerWrapper,
    GpuSpatialRelationPredicateWrapper,
};
#[cfg(gpu_available)]
pub use libgpuspatial_glue_bindgen::GpuSpatialIndexContext;
#[cfg(gpu_available)]
use nvml_wrapper::Nvml;

// Mark GPU types as Send for thread safety
// SAFETY: The GPU library is designed to be used from multiple threads.
// Each thread gets its own context, and the underlying GPU library handles thread safety.
// The raw pointers inside are managed by the C++ library which ensures proper synchronization.
#[cfg(gpu_available)]
unsafe impl Send for GpuSpatialIndexContext {}
#[cfg(gpu_available)]
unsafe impl Send for libgpuspatial_glue_bindgen::GpuSpatialRTEngine {}
#[cfg(gpu_available)]
unsafe impl Sync for libgpuspatial_glue_bindgen::GpuSpatialRTEngine {}

#[cfg(gpu_available)]
unsafe impl Send for libgpuspatial_glue_bindgen::GpuSpatialIndexFloat2D {}
#[cfg(gpu_available)]
unsafe impl Send for libgpuspatial_glue_bindgen::GpuSpatialRefiner {}

#[cfg(gpu_available)]
unsafe impl Sync for libgpuspatial_glue_bindgen::GpuSpatialIndexFloat2D {}

#[cfg(gpu_available)]
unsafe impl Sync for libgpuspatial_glue_bindgen::GpuSpatialRefiner {}

// Error type for non-GPU builds
#[cfg(not(gpu_available))]
#[derive(Debug, thiserror::Error)]
pub enum GpuSpatialError {
    #[error("GPU not available - CUDA not found during build")]
    GpuNotAvailable,
}

pub type Result<T> = std::result::Result<T, GpuSpatialError>;

/// Spatial predicates for GPU operations
#[repr(u32)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum GpuSpatialRelationPredicate {
    Equals = 0,
    Disjoint = 1,
    Touches = 2,
    Contains = 3,
    Covers = 4,
    Intersects = 5,
    Within = 6,
    CoveredBy = 7,
}

impl std::fmt::Display for GpuSpatialRelationPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuSpatialRelationPredicate::Equals => write!(f, "equals"),
            GpuSpatialRelationPredicate::Disjoint => write!(f, "disjoint"),
            GpuSpatialRelationPredicate::Touches => write!(f, "touches"),
            GpuSpatialRelationPredicate::Contains => write!(f, "contains"),
            GpuSpatialRelationPredicate::Covers => write!(f, "covers"),
            GpuSpatialRelationPredicate::Intersects => write!(f, "intersects"),
            GpuSpatialRelationPredicate::Within => write!(f, "within"),
            GpuSpatialRelationPredicate::CoveredBy => write!(f, "coveredby"),
        }
    }
}

#[cfg(gpu_available)]
impl From<GpuSpatialRelationPredicate> for GpuSpatialRelationPredicateWrapper {
    fn from(pred: GpuSpatialRelationPredicate) -> Self {
        match pred {
            GpuSpatialRelationPredicate::Equals => GpuSpatialRelationPredicateWrapper::Equals,
            GpuSpatialRelationPredicate::Disjoint => GpuSpatialRelationPredicateWrapper::Disjoint,
            GpuSpatialRelationPredicate::Touches => GpuSpatialRelationPredicateWrapper::Touches,
            GpuSpatialRelationPredicate::Contains => GpuSpatialRelationPredicateWrapper::Contains,
            GpuSpatialRelationPredicate::Covers => GpuSpatialRelationPredicateWrapper::Covers,
            GpuSpatialRelationPredicate::Intersects => {
                GpuSpatialRelationPredicateWrapper::Intersects
            }
            GpuSpatialRelationPredicate::Within => GpuSpatialRelationPredicateWrapper::Within,
            GpuSpatialRelationPredicate::CoveredBy => GpuSpatialRelationPredicateWrapper::CoveredBy,
        }
    }
}

/// High-level wrapper for GPU spatial operations
pub struct GpuSpatial {
    #[cfg(gpu_available)]
    rt_engine: Option<Arc<Mutex<GpuSpatialRTEngineWrapper>>>,
    #[cfg(gpu_available)]
    index: Option<GpuSpatialIndexFloat2DWrapper>,
    #[cfg(gpu_available)]
    refiner: Option<GpuSpatialRefinerWrapper>,
    initialized: bool,
}

impl GpuSpatial {
    pub fn new() -> Result<Self> {
        #[cfg(not(gpu_available))]
        {
            Err(GpuSpatialError::GpuNotAvailable)
        }

        #[cfg(gpu_available)]
        {
            Ok(Self {
                rt_engine: None,
                index: None,
                refiner: None,
                initialized: false,
            })
        }
    }

    pub fn init(&mut self, concurrency: u32, device_id: i32) -> Result<()> {
        #[cfg(not(gpu_available))]
        {
            let _ = (concurrency, device_id);
            Err(GpuSpatialError::GpuNotAvailable)
        }

        #[cfg(gpu_available)]
        {
            // Get PTX path from OUT_DIR
            let out_path = std::path::PathBuf::from(env!("OUT_DIR"));
            let ptx_root = out_path.join("share/gpuspatial/shaders");
            let ptx_root_str = ptx_root
                .to_str()
                .ok_or_else(|| GpuSpatialError::Init("Invalid PTX path".to_string()))?;

            let rt_engine = GpuSpatialRTEngineWrapper::try_new(device_id, ptx_root_str)?;

            self.rt_engine = Some(Arc::new(Mutex::new(rt_engine)));

            let index = GpuSpatialIndexFloat2DWrapper::try_new(
                self.rt_engine.as_ref().unwrap(),
                concurrency,
            )?;

            self.index = Some(index);

            let refiner =
                GpuSpatialRefinerWrapper::try_new(self.rt_engine.as_ref().unwrap(), concurrency)?;
            self.refiner = Some(refiner);

            self.initialized = true;
            Ok(())
        }
    }

    pub fn is_gpu_available() -> bool {
        #[cfg(not(gpu_available))]
        {
            false
        }
        #[cfg(gpu_available)]
        {
            let nvml = match Nvml::init() {
                Ok(instance) => instance,
                Err(_) => return false,
            };

            // Check if the device count is greater than zero
            match nvml.device_count() {
                Ok(count) => count > 0,
                Err(_) => false,
            }
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Clear previous build data
    pub fn clear(&mut self) -> Result<()> {
        #[cfg(not(gpu_available))]
        {
            Err(GpuSpatialError::GpuNotAvailable)
        }
        #[cfg(gpu_available)]
        {
            if !self.initialized {
                return Err(GpuSpatialError::Init("GpuSpatial not initialized".into()));
            }

            let index = self
                .index
                .as_mut()
                .ok_or_else(|| GpuSpatialError::Init("GPU index is not available".into()))?;

            // Clear previous build data
            index.clear();
            Ok(())
        }
    }

    pub fn push_build(&mut self, rects: &[Rect<f32>]) -> Result<()> {
        #[cfg(not(gpu_available))]
        {
            let _ = rects;
            Err(GpuSpatialError::GpuNotAvailable)
        }
        #[cfg(gpu_available)]
        {
            let index = self
                .index
                .as_mut()
                .ok_or_else(|| GpuSpatialError::Init("GPU index not available".into()))?;

            unsafe { index.push_build(rects.as_ptr() as *const f32, rects.len() as u32) }
        }
    }

    pub fn finish_building(&mut self) -> Result<()> {
        #[cfg(not(gpu_available))]
        return Err(GpuSpatialError::GpuNotAvailable);

        #[cfg(gpu_available)]
        self.index
            .as_mut()
            .ok_or_else(|| GpuSpatialError::Init("GPU index not available".into()))?
            .finish_building()
    }

    pub fn probe(&self, rects: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>)> {
        #[cfg(not(gpu_available))]
        {
            let _ = rects;
            Err(GpuSpatialError::GpuNotAvailable)
        }

        #[cfg(gpu_available)]
        {
            let index = self
                .index
                .as_ref()
                .ok_or_else(|| GpuSpatialError::Init("GPU index not available".into()))?;
            // Create context
            let mut ctx = GpuSpatialIndexContext {
                last_error: std::ptr::null(),
                build_indices: std::ptr::null_mut(),
                probe_indices: std::ptr::null_mut(),
            };
            index.create_context(&mut ctx);

            // Push stream data (probe side) and perform join
            unsafe {
                index.probe(&mut ctx, rects.as_ptr() as *const f32, rects.len() as u32)?;
            }

            // Get results
            let build_indices = index.get_build_indices_buffer(&mut ctx).to_vec();
            let probe_indices = index.get_probe_indices_buffer(&mut ctx).to_vec();
            index.destroy_context(&mut ctx);
            Ok((build_indices, probe_indices))
        }
    }

    pub fn load_build_array(&mut self, array: &arrow_array::ArrayRef) -> Result<()> {
        #[cfg(not(gpu_available))]
        {
            let _ = array;
            Err(GpuSpatialError::GpuNotAvailable)
        }
        #[cfg(gpu_available)]
        {
            let refiner = self
                .refiner
                .as_ref()
                .ok_or_else(|| GpuSpatialError::Init("GPU refiner not available".into()))?;

            refiner.load_build_array(array)
        }
    }

    pub fn refine_loaded(
        &self,
        probe_array: &arrow_array::ArrayRef,
        predicate: GpuSpatialRelationPredicate,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<()> {
        #[cfg(not(gpu_available))]
        {
            let _ = (probe_array, predicate, build_indices, probe_indices);
            Err(GpuSpatialError::GpuNotAvailable)
        }
        #[cfg(gpu_available)]
        {
            let refiner = self
                .refiner
                .as_ref()
                .ok_or_else(|| GpuSpatialError::Init("GPU refiner not available".into()))?;

            refiner.refine_loaded(
                probe_array,
                GpuSpatialRelationPredicateWrapper::from(predicate),
                build_indices,
                probe_indices,
            )
        }
    }

    pub fn refine(
        &self,
        array1: &arrow_array::ArrayRef,
        array2: &arrow_array::ArrayRef,
        predicate: GpuSpatialRelationPredicate,
        indices1: &mut Vec<u32>,
        indices2: &mut Vec<u32>,
    ) -> Result<()> {
        #[cfg(not(gpu_available))]
        {
            let _ = (array1, array2, predicate, indices1, indices2);
            Err(GpuSpatialError::GpuNotAvailable)
        }
        #[cfg(gpu_available)]
        {
            let refiner = self
                .refiner
                .as_ref()
                .ok_or_else(|| GpuSpatialError::Init("GPU refiner not available".into()))?;

            refiner.refine(
                array1,
                array2,
                GpuSpatialRelationPredicateWrapper::from(predicate),
                indices1,
                indices2,
            )
        }
    }
}

#[cfg(gpu_available)]
#[cfg(test)]
mod tests {
    use super::*;
    use geo::{BoundingRect, Intersects, Point, Polygon};
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_geos::register::scalar_kernels;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::{Edges, SedonaType, WKB_GEOMETRY};
    use sedona_testing::create::create_array_storage;
    use sedona_testing::testers::ScalarUdfTester;
    use wkt::TryFromWkt;

    pub fn find_intersection_pairs(
        vec_a: &[Rect<f32>],
        vec_b: &[Rect<f32>],
    ) -> (Vec<u32>, Vec<u32>) {
        let mut ids_a = Vec::new();
        let mut ids_b = Vec::new();

        // Iterate through A with index 'i'
        for (i, rect_a) in vec_a.iter().enumerate() {
            // Only proceed if 'a' exists
            // Iterate through B with index 'j'
            for (j, rect_b) in vec_b.iter().enumerate() {
                // Check if 'b' exists and intersects 'a'
                if rect_a.intersects(rect_b) {
                    ids_a.push(i as u32);
                    ids_b.push(j as u32);
                }
            }
        }

        (ids_a, ids_b)
    }
    #[test]
    fn test_spatial_index() {
        let mut gs = GpuSpatial::new().unwrap();
        gs.init(1, 0).expect("Failed to initialize GpuSpatial");

        let polygon_values =  &[
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
        gs.push_build(&rects).expect("Failed to push build data");
        gs.finish_building().expect("Failed to finish building");
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
            find_intersection_pairs(&rects, &points);

        ans_build_indices.sort();
        ans_probe_indices.sort();

        assert_eq!(build_indices, ans_build_indices);
        assert_eq!(probe_indices, ans_probe_indices);
    }

    #[test]
    fn test_spatial_refiner() {
        let mut gs = GpuSpatial::new().unwrap();
        gs.init(1, 0).expect("Failed to initialize GpuSpatial");

        let polygon_values =  &[
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
        gs.push_build(&rects).expect("Failed to push build data");
        gs.finish_building().expect("Failed to finish building");
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
        let (mut build_indices, mut probe_indices) = gs.probe(&point_rects).unwrap();

        gs.refine(
            &polygons,
            &points,
            GpuSpatialRelationPredicate::Intersects,
            &mut build_indices,
            &mut probe_indices,
        )
        .expect("Failed to refine results");

        build_indices.sort();
        probe_indices.sort();

        let kernels = scalar_kernels();

        // Iterate through the vector and find the one named "st_intersects"
        let st_intersects = kernels
            .into_iter()
            .find(|(name, _)| *name == "st_intersects")
            .map(|(_, kernel_ref)| kernel_ref)
            .unwrap();

        let sedona_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let udf = SedonaScalarUDF::from_kernel("st_intersects", st_intersects);
        let tester =
            ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type.clone()]);

        let mut ans_build_indices: Vec<u32> = Vec::new();
        let mut ans_probe_indices: Vec<u32> = Vec::new();

        for (poly_index, poly) in polygon_values.iter().enumerate() {
            for (point_index, point) in point_values.iter().enumerate() {
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

        assert_eq!(build_indices, ans_build_indices);
        assert_eq!(probe_indices, ans_probe_indices);
    }
}
