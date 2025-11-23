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

use crate::error::GpuSpatialError;
use crate::libgpuspatial_glue_bindgen::*;
use arrow_array::{ffi::FFI_ArrowArray, ArrayRef};
use std::convert::TryFrom;
use std::ffi::CString;
use std::mem::transmute;
use std::os::raw::{c_uint, c_void};

pub struct GpuSpatialJoinerWrapper {
    joiner: GpuSpatialJoiner,
}

#[repr(u32)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum GpuSpatialPredicateWrapper {
    Equals = 0,
    Disjoint = 1,
    Touches = 2,
    Contains = 3,
    Covers = 4,
    Intersects = 5,
    Within = 6,
    CoveredBy = 7,
}

impl TryFrom<c_uint> for GpuSpatialPredicateWrapper {
    type Error = &'static str;

    fn try_from(v: c_uint) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(GpuSpatialPredicateWrapper::Equals),
            1 => Ok(GpuSpatialPredicateWrapper::Disjoint),
            2 => Ok(GpuSpatialPredicateWrapper::Touches),
            3 => Ok(GpuSpatialPredicateWrapper::Contains),
            4 => Ok(GpuSpatialPredicateWrapper::Covers),
            5 => Ok(GpuSpatialPredicateWrapper::Intersects),
            6 => Ok(GpuSpatialPredicateWrapper::Within),
            7 => Ok(GpuSpatialPredicateWrapper::CoveredBy),
            _ => Err("Invalid GpuSpatialPredicate value"),
        }
    }
}

impl Default for GpuSpatialJoinerWrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuSpatialJoinerWrapper {
    pub fn new() -> Self {
        GpuSpatialJoinerWrapper {
            joiner: GpuSpatialJoiner {
                init: None,
                clear: None,
                create_context: None,
                destroy_context: None,
                push_build: None,
                finish_building: None,
                push_stream: None,
                get_build_indices_buffer: None,
                get_stream_indices_buffer: None,
                release: None,
                private_data: std::ptr::null_mut(),
                last_error: std::ptr::null(),
            },
        }
    }

    /// # Initializes the GpuSpatialJoiner
    /// This function should only be called once per joiner instance.
    ///
    /// # Arguments
    /// * `concurrency` - How many threads will call the joiner concurrently.
    /// * `ptx_root` - The root directory for PTX files.
    pub fn init(&mut self, concurrency: u32, ptx_root: &str) -> Result<(), GpuSpatialError> {
        let joiner_ptr: *mut GpuSpatialJoiner = &mut self.joiner;

        unsafe {
            // Set function pointers to the C functions
            GpuSpatialJoinerCreate(joiner_ptr);
        }

        if let Some(init_fn) = self.joiner.init {
            let c_ptx_root = CString::new(ptx_root).expect("CString::new failed");

            let mut config = GpuSpatialJoinerConfig {
                concurrency,
                ptx_root: c_ptx_root.as_ptr(),
            };

            // This is an unsafe call because it's calling a C function from the bindings.
            unsafe {
                if init_fn(&self.joiner as *const _ as *mut _, &mut config) != 0 {
                    let error_message = self.joiner.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    return Err(GpuSpatialError::Init(error_string));
                }
            }
        }
        Ok(())
    }

    /// # Clears the GpuSpatialJoiner
    /// This function clears the internal state of the joiner.
    /// By calling this function, the pushed build data will be cleared.
    /// You should call this function to reuse the joiner
    /// instead of building a new one because creating a new joiner is expensive.
    /// **This method is not thread-safe and should be called from a single thread.**
    pub fn clear(&mut self) {
        if let Some(clear_fn) = self.joiner.clear {
            unsafe {
                clear_fn(&mut self.joiner as *mut _);
            }
        }
    }

    /// # Pushes an array of WKBs to the build side of the joiner
    /// This function can be called multiple times to push multiple arrays.
    /// The joiner will internally parse the WKBs and build a spatial index.
    /// After pushing all build data, you must call `finish_building()` to build the
    /// spatial index.
    /// **This method is not thread-safe and should be called from a single thread.**
    /// # Arguments
    /// * `array` - The array of WKBs to push.
    /// * `offset` - The offset of the array to push.
    /// * `length` - The length of the array to push.
    pub fn push_build(
        &mut self,
        array: &ArrayRef,
        offset: i64,
        length: i64,
    ) -> Result<(), GpuSpatialError> {
        log::info!(
            "DEBUG FFI: push_build called with offset={}, length={}",
            offset,
            length
        );
        log::info!(
            "DEBUG FFI: Array length={}, null_count={}",
            array.len(),
            array.null_count()
        );

        // 1. Convert the single ArrayRef to its FFI representation
        let (ffi_array, _) = arrow_array::ffi::to_ffi(&array.to_data())?;

        log::info!("DEBUG FFI: FFI conversion successful");
        log::info!("DEBUG FFI: FFI array null_count={}", ffi_array.null_count());

        // 2. Get the raw pointer to the FFI_ArrowArray struct
        // let arrow_ptr = &mut ffi_array as *mut FFI_ArrowArray as *mut ArrowArray;

        if let Some(push_build_fn) = self.joiner.push_build {
            unsafe {
                let ffi_array_ptr: *const ArrowArray =
                    transmute(&ffi_array as *const FFI_ArrowArray);
                log::info!("DEBUG FFI: Calling C++ push_build function");
                if push_build_fn(
                    &mut self.joiner as *mut _,
                    std::ptr::null_mut(), // schema is unused currently
                    ffi_array_ptr as *mut _,
                    offset,
                    length,
                ) != 0
                {
                    let error_message = self.joiner.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    log::error!("DEBUG FFI: push_build failed: {}", error_string);
                    return Err(GpuSpatialError::PushBuild(error_string));
                }
                log::info!("DEBUG FFI: push_build C++ call succeeded");
            }
        }
        Ok(())
    }

    /// # Finishes building the spatial index
    /// This function must be called after all build data has been pushed
    /// using `push_build()`. It builds the spatial index internally on the GPU.
    /// After calling this function, the joiner is ready to accept stream data
    /// for spatial join operations.
    /// **This method is not thread-safe and should be called from a single thread.**
    pub fn finish_building(&mut self) -> Result<(), GpuSpatialError> {
        if let Some(finish_building_fn) = self.joiner.finish_building {
            unsafe {
                if finish_building_fn(&mut self.joiner as *mut _) != 0 {
                    let error_message = self.joiner.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    return Err(GpuSpatialError::FinishBuild(error_string));
                }
            }
        }
        Ok(())
    }

    /// # Creates a context for a thread to perform spatial joins
    /// This function initializes a context that holds thread-specific data for spatial joins and
    /// pointers to buffers that store the results of spatial joins.
    /// Each thread that performs spatial joins should have its own context.
    /// The context is passed to PushStream calls to perform spatial joins.
    /// The context must be created after the joiner has been initialized.
    /// It is encouraged to create reuse the context within the same thread to reduce resource allocation overhead.
    /// The context can be destroyed by calling the `destroy_context` function pointer in the `GpuSpatialJoiner` struct.
    /// The context should be destroyed before destroying the joiner.
    /// **This method is thread-safe.**
    pub fn create_context(&mut self, ctx: &mut GpuSpatialJoinerContext) {
        if let Some(create_context_fn) = self.joiner.create_context {
            unsafe {
                create_context_fn(&mut self.joiner as *mut _, ctx as *mut _);
            }
        }
    }

    pub fn destroy_context(&mut self, ctx: &mut GpuSpatialJoinerContext) {
        if let Some(destroy_context_fn) = self.joiner.destroy_context {
            unsafe {
                destroy_context_fn(ctx as *mut _);
            }
        }
    }

    pub fn push_stream(
        &mut self,
        ctx: &mut GpuSpatialJoinerContext,
        array: &ArrayRef,
        offset: i64,
        length: i64,
        predicate: GpuSpatialPredicateWrapper,
        array_index_offset: i32,
    ) -> Result<(), GpuSpatialError> {
        log::info!(
            "DEBUG FFI: push_stream called with offset={}, length={}, predicate={:?}",
            offset,
            length,
            predicate
        );
        log::info!(
            "DEBUG FFI: Array length={}, null_count={}",
            array.len(),
            array.null_count()
        );

        // 1. Convert the single ArrayRef to its FFI representation
        let (ffi_array, _) = arrow_array::ffi::to_ffi(&array.to_data())?;

        log::info!("DEBUG FFI: FFI conversion successful");
        log::info!("DEBUG FFI: FFI array null_count={}", ffi_array.null_count());

        // 2. Get the raw pointer to the FFI_ArrowArray struct
        // let arrow_ptr = &mut ffi_array as *mut FFI_ArrowArray as *mut ArrowArray;

        if let Some(push_stream_fn) = self.joiner.push_stream {
            unsafe {
                let ffi_array_ptr: *const ArrowArray =
                    transmute(&ffi_array as *const FFI_ArrowArray);
                log::info!("DEBUG FFI: Calling C++ push_stream function");
                if push_stream_fn(
                    &mut self.joiner as *mut _,
                    ctx as *mut _,
                    std::ptr::null_mut(), // schema is unused currently
                    ffi_array_ptr as *mut _,
                    offset,
                    length,
                    predicate as c_uint,
                    array_index_offset,
                ) != 0
                {
                    let error_message = ctx.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    log::error!("DEBUG FFI: push_stream failed: {}", error_string);
                    return Err(GpuSpatialError::PushStream(error_string));
                }
                log::info!("DEBUG FFI: push_stream C++ call succeeded");
            }
        }
        Ok(())
    }

    pub fn get_build_indices_buffer(&self, ctx: &mut GpuSpatialJoinerContext) -> &[u32] {
        if let Some(get_build_indices_buffer_fn) = self.joiner.get_build_indices_buffer {
            let mut build_indices_ptr: *mut c_void = std::ptr::null_mut();
            let mut build_indices_len: u32 = 0;

            unsafe {
                get_build_indices_buffer_fn(
                    ctx as *mut _,
                    &mut build_indices_ptr as *mut *mut c_void,
                    &mut build_indices_len as *mut u32,
                );

                // Check length first - empty vectors return empty slice
                if build_indices_len == 0 {
                    return &[];
                }

                // Validate pointer (should not be null if length > 0)
                if build_indices_ptr.is_null() {
                    return &[];
                }

                // Convert the raw pointer to a slice. This is safe to do because
                // we've validated the pointer is non-null and length is valid.
                let typed_ptr = build_indices_ptr as *const u32;

                // Safety: We've checked ptr is non-null and len > 0
                return std::slice::from_raw_parts(typed_ptr, build_indices_len as usize);
            }
        }
        &[]
    }

    pub fn get_stream_indices_buffer(&self, ctx: &mut GpuSpatialJoinerContext) -> &[u32] {
        if let Some(get_stream_indices_buffer_fn) = self.joiner.get_stream_indices_buffer {
            let mut stream_indices_ptr: *mut c_void = std::ptr::null_mut();
            let mut stream_indices_len: u32 = 0;

            unsafe {
                get_stream_indices_buffer_fn(
                    ctx as *mut _,
                    &mut stream_indices_ptr as *mut *mut c_void,
                    &mut stream_indices_len as *mut u32,
                );

                // Check length first - empty vectors return empty slice
                if stream_indices_len == 0 {
                    return &[];
                }

                // Validate pointer (should not be null if length > 0)
                if stream_indices_ptr.is_null() {
                    return &[];
                }

                // Convert the raw pointer to a slice. This is safe to do because
                // we've validated the pointer is non-null and length is valid.
                let typed_ptr = stream_indices_ptr as *const u32;

                // Safety: We've checked ptr is non-null and len > 0
                return std::slice::from_raw_parts(typed_ptr, stream_indices_len as usize);
            }
        }
        &[]
    }

    pub fn release(&mut self) {
        // Call the release function if it exists
        if let Some(release_fn) = self.joiner.release {
            unsafe {
                release_fn(&mut self.joiner as *mut _);
            }
        }
    }
}

impl Drop for GpuSpatialJoinerWrapper {
    fn drop(&mut self) {
        // Call the release function if it exists
        if let Some(release_fn) = self.joiner.release {
            unsafe {
                release_fn(&mut self.joiner as *mut _);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use std::env;
    use std::path::PathBuf;
    // use arrow_array::{create_array as arrow_array, ArrayRef};
    use sedona_geos::register::scalar_kernels;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::{Edges, SedonaType, WKB_GEOMETRY};
    use sedona_testing::create::create_array_storage;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn test_gpu_joiner_end2end() {
        let mut joiner = GpuSpatialJoinerWrapper::new();

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        let ptx_root = out_path.join("share/gpuspatial/shaders");

        joiner
            .init(
                1,
                ptx_root.to_str().expect("Failed to convert path to string"),
            )
            .expect("Failed to init GpuSpatialJoiner");

        let polygon_values =  &[
            Some("POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
            Some("POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))"),
            Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 3 2, 3 3, 2 3, 2 2), (6 6, 8 6, 8 8, 6 8, 6 6))"),
            Some("POLYGON ((30 0, 60 20, 50 50, 10 50, 0 20, 30 0), (20 30, 25 40, 15 40, 20 30), (30 30, 35 40, 25 40, 30 30), (40 30, 45 40, 35 40, 40 30))"),
            Some("POLYGON ((40 0, 50 30, 80 20, 90 70, 60 90, 30 80, 20 40, 40 0), (50 20, 65 30, 60 50, 45 40, 50 20), (30 60, 50 70, 45 80, 30 60))"),
        ];
        let polygons = create_array_storage(polygon_values, &WKB_GEOMETRY);

        // Let the gpusaptial joiner to parse WKBs and get building boxes
        joiner
            .push_build(&polygons, 0, polygons.len().try_into().unwrap())
            .expect("Failed to push building");
        // Build a spatial index for Build internally on GPU
        joiner.finish_building().expect("Failed to finish building");

        // Each thread that performs spatial joins should have its own context.
        // The context is passed to PushStream calls to perform spatial joins.
        let mut ctx = GpuSpatialJoinerContext {
            last_error: std::ptr::null(),
            private_data: std::ptr::null_mut(),
            build_indices: std::ptr::null_mut(),
            stream_indices: std::ptr::null_mut(),
        };

        joiner.create_context(&mut ctx);

        let point_values = &[
            Some("POINT (30 20)"), // poly0
            Some("POINT (20 20)"), // poly1
            Some("POINT (1 1)"),   // poly2
            Some("POINT (70 70)"),
            Some("POINT (55 35)"), // poly4
        ];
        let points = create_array_storage(point_values, &WKB_GEOMETRY);

        // array_index_offset offsets the result of stream indices
        let array_index_offset = 0;
        joiner
            .push_stream(
                &mut ctx,
                &points,
                0,
                points.len().try_into().unwrap(),
                GpuSpatialPredicateWrapper::Intersects,
                array_index_offset,
            )
            .expect("Failed to push building");

        let build_indices = joiner.get_build_indices_buffer(&mut ctx);
        let stream_indices = joiner.get_stream_indices_buffer(&mut ctx);

        let mut result_pairs: Vec<(u32, u32)> = Vec::new();

        for (build_index, stream_index) in build_indices.iter().zip(stream_indices.iter()) {
            result_pairs.push((*build_index, *stream_index));
        }

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

        let mut answer_pairs: Vec<(u32, u32)> = Vec::new();

        for (poly_index, poly) in polygon_values.iter().enumerate() {
            for (point_index, point) in point_values.iter().enumerate() {
                let result = tester
                    .invoke_scalar_scalar(poly.unwrap(), point.unwrap())
                    .unwrap();
                if result == true.into() {
                    answer_pairs.push((poly_index as u32, point_index as u32));
                }
            }
        }

        // Sort both vectors. The default sort on tuples compares element by element.
        result_pairs.sort();
        answer_pairs.sort();

        // Assert that the two sorted vectors are equal.
        assert_eq!(result_pairs, answer_pairs);

        joiner.destroy_context(&mut ctx);
        joiner.release();
    }
}
