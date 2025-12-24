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
use arrow_array::{ffi::FFI_ArrowArray, Array, ArrayRef};
use arrow_schema::ffi::FFI_ArrowSchema;
use std::convert::TryFrom;
use std::ffi::CString;
use std::mem::transmute;
use std::os::raw::{c_uint, c_void};
use std::sync::{Arc, Mutex};

pub struct GpuSpatialRTEngineWrapper {
    rt_engine: GpuSpatialRTEngine,
    device_id: i32,
}

impl GpuSpatialRTEngineWrapper {
    /// # Initializes the GpuSpatialRTEngine
    /// This function should only be called once per engine instance.
    /// # Arguments
    /// * `device_id` - The GPU device ID to use.
    /// * `ptx_root` - The root directory for PTX files.
    pub fn try_new(
        device_id: i32,
        ptx_root: &str,
    ) -> Result<GpuSpatialRTEngineWrapper, GpuSpatialError> {
        let mut rt_engine = GpuSpatialRTEngine {
            init: None,
            release: None,
            private_data: std::ptr::null_mut(),
            last_error: std::ptr::null(),
        };

        unsafe {
            // Set function pointers to the C functions
            GpuSpatialRTEngineCreate(&mut rt_engine);
        }

        if let Some(init_fn) = rt_engine.init {
            let c_ptx_root = CString::new(ptx_root).expect("CString::new failed");

            let mut config = GpuSpatialRTEngineConfig {
                device_id,
                ptx_root: c_ptx_root.as_ptr(),
            };

            // This is an unsafe call because it's calling a C function from the bindings.
            unsafe {
                if init_fn(&rt_engine as *const _ as *mut _, &mut config) != 0 {
                    let error_message = rt_engine.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    return Err(GpuSpatialError::Init(error_string));
                }
            }
        }
        Ok(GpuSpatialRTEngineWrapper {
            rt_engine,
            device_id,
        })
    }
}

impl Default for GpuSpatialRTEngineWrapper {
    fn default() -> Self {
        GpuSpatialRTEngineWrapper {
            rt_engine: GpuSpatialRTEngine {
                init: None,
                release: None,
                private_data: std::ptr::null_mut(),
                last_error: std::ptr::null(),
            },
            device_id: -1,
        }
    }
}

impl Drop for GpuSpatialRTEngineWrapper {
    fn drop(&mut self) {
        // Call the release function if it exists
        if let Some(release_fn) = self.rt_engine.release {
            unsafe {
                release_fn(&mut self.rt_engine as *mut _);
            }
        }
    }
}

pub struct GpuSpatialIndexFloat2DWrapper {
    index: GpuSpatialIndexFloat2D,
    _rt_engine: Arc<Mutex<GpuSpatialRTEngineWrapper>>, // Keep a reference to the RT engine to ensure it lives as long as the index
}

impl GpuSpatialIndexFloat2DWrapper {
    /// # Initializes the GpuSpatialJoiner
    /// This function should only be called once per joiner instance.
    ///
    /// # Arguments
    /// * `rt_engine` - The ray-tracing engine to use for GPU operations.
    /// * `concurrency` - How many threads will call the joiner concurrently.
    pub fn try_new(
        rt_engine: &Arc<Mutex<GpuSpatialRTEngineWrapper>>,
        concurrency: u32,
    ) -> Result<Self, GpuSpatialError> {
        let mut index = GpuSpatialIndexFloat2D {
            init: None,
            clear: None,
            create_context: None,
            destroy_context: None,
            push_build: None,
            finish_building: None,
            probe: None,
            get_build_indices_buffer: None,
            get_probe_indices_buffer: None,
            release: None,
            private_data: std::ptr::null_mut(),
            last_error: std::ptr::null(),
        };

        unsafe {
            // Set function pointers to the C functions
            GpuSpatialIndexFloat2DCreate(&mut index);
        }

        if let Some(init_fn) = index.init {
            let mut engine_guard = rt_engine
                .lock()
                .map_err(|_| GpuSpatialError::Init("Failed to acquire mutex lock".to_string()))?;

            let mut config = GpuSpatialIndexConfig {
                rt_engine: &mut engine_guard.rt_engine,
                concurrency,
                device_id: engine_guard.device_id,
            };

            // This is an unsafe call because it's calling a C function from the bindings.
            unsafe {
                if init_fn(&index as *const _ as *mut _, &mut config) != 0 {
                    let error_message = index.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    return Err(GpuSpatialError::Init(error_string));
                }
            }
        }
        Ok(GpuSpatialIndexFloat2DWrapper {
            index,
            _rt_engine: rt_engine.clone(),
        })
    }

    /// # Clears the GpuSpatialJoiner
    /// This function clears the internal state of the joiner.
    /// By calling this function, the pushed build data will be cleared.
    /// You should call this function to reuse the joiner
    /// instead of building a new one because creating a new joiner is expensive.
    /// **This method is not thread-safe and should be called from a single thread.**
    pub fn clear(&mut self) {
        if let Some(clear_fn) = self.index.clear {
            unsafe {
                clear_fn(&mut self.index as *mut _);
            }
        }
    }

    /// # Pushes an array of rectangles to the build side of the joiner
    /// This function can be called multiple times to push multiple arrays.
    /// The joiner will internally parse the rectangles and build a spatial index.
    /// After pushing all build data, you must call `finish_building()` to build the
    /// spatial index.
    /// **This method is not thread-safe and should be called from a single thread.**
    /// # Arguments
    /// * `buf` - The array pointer to the rectangles to push.
    /// * `n_rects` - The number of rectangles in the array.
    /// # Safety
    /// This function is unsafe because it takes a raw pointer to the rectangles.
    ///
    pub unsafe fn push_build(
        &mut self,
        buf: *const f32,
        n_rects: u32,
    ) -> Result<(), GpuSpatialError> {
        log::debug!("DEBUG FFI: push_build called with length={}", n_rects);

        if let Some(push_build_fn) = self.index.push_build {
            unsafe {
                if push_build_fn(&mut self.index as *mut _, buf, n_rects) != 0 {
                    let error_message = self.index.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    log::error!("DEBUG FFI: push_build failed: {}", error_string);
                    return Err(GpuSpatialError::PushBuild(error_string));
                }
                log::debug!("DEBUG FFI: push_build C++ call succeeded");
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
        if let Some(finish_building_fn) = self.index.finish_building {
            unsafe {
                if finish_building_fn(&mut self.index as *mut _) != 0 {
                    let error_message = self.index.last_error;
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
    pub fn create_context(&self, ctx: &mut GpuSpatialIndexContext) {
        if let Some(create_context_fn) = self.index.create_context {
            unsafe {
                // Cast the shared reference to a raw pointer, then to a mutable raw pointer
                let index_ptr = &self.index as *const _ as *mut _;
                create_context_fn(index_ptr, ctx as *mut _);
            }
        }
    }

    pub fn destroy_context(&self, ctx: &mut GpuSpatialIndexContext) {
        if let Some(destroy_context_fn) = self.index.destroy_context {
            unsafe {
                destroy_context_fn(ctx as *mut _);
            }
        }
    }

    /// # Probes an array of rectangles against the built spatial index
    /// This function probes an array of rectangles against the spatial index built
    /// using `push_build()` and `finish_building()`. It finds all pairs of rectangles
    /// that satisfy the spatial relation defined by the index.
    /// The results are stored in the context passed to the function.
    /// **This method is thread-safe if each thread uses its own context.**
    /// # Arguments
    /// * `ctx` - The context for the thread performing the spatial join.
    /// * `buf` - A pointer to the array of rectangles to probe.
    /// * `n_rects` - The number of rectangles in the array.
    /// # Safety
    /// This function is unsafe because it takes a raw pointer to the rectangles.
    pub unsafe fn probe(
        &self,
        ctx: &mut GpuSpatialIndexContext,
        buf: *const f32,
        n_rects: u32,
    ) -> Result<(), GpuSpatialError> {
        log::debug!("DEBUG FFI: probe called with length={}", n_rects);

        if let Some(probe_fn) = self.index.probe {
            unsafe {
                if probe_fn(
                    &self.index as *const _ as *mut _,
                    ctx as *mut _,
                    buf,
                    n_rects,
                ) != 0
                {
                    let error_message = ctx.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    log::error!("DEBUG FFI: probe failed: {}", error_string);
                    return Err(GpuSpatialError::PushStream(error_string));
                }
                log::debug!("DEBUG FFI: probe C++ call succeeded");
            }
        }
        Ok(())
    }

    pub fn get_build_indices_buffer(&self, ctx: &mut GpuSpatialIndexContext) -> &[u32] {
        if let Some(get_build_indices_buffer_fn) = self.index.get_build_indices_buffer {
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

    pub fn get_probe_indices_buffer(&self, ctx: &mut GpuSpatialIndexContext) -> &[u32] {
        if let Some(get_probe_indices_buffer_fn) = self.index.get_probe_indices_buffer {
            let mut probe_indices_ptr: *mut c_void = std::ptr::null_mut();
            let mut probe_indices_len: u32 = 0;

            unsafe {
                get_probe_indices_buffer_fn(
                    ctx as *mut _,
                    &mut probe_indices_ptr as *mut *mut c_void,
                    &mut probe_indices_len as *mut u32,
                );

                // Check length first - empty vectors return empty slice
                if probe_indices_len == 0 {
                    return &[];
                }

                // Validate pointer (should not be null if length > 0)
                if probe_indices_ptr.is_null() {
                    return &[];
                }

                // Convert the raw pointer to a slice. This is safe to do because
                // we've validated the pointer is non-null and length is valid.
                let typed_ptr = probe_indices_ptr as *const u32;

                // Safety: We've checked ptr is non-null and len > 0
                return std::slice::from_raw_parts(typed_ptr, probe_indices_len as usize);
            }
        }
        &[]
    }
}

impl Default for GpuSpatialIndexFloat2DWrapper {
    fn default() -> Self {
        GpuSpatialIndexFloat2DWrapper {
            index: GpuSpatialIndexFloat2D {
                init: None,
                clear: None,
                create_context: None,
                destroy_context: None,
                push_build: None,
                finish_building: None,
                probe: None,
                get_build_indices_buffer: None,
                get_probe_indices_buffer: None,
                release: None,
                private_data: std::ptr::null_mut(),
                last_error: std::ptr::null(),
            },
            _rt_engine: Arc::new(Mutex::new(GpuSpatialRTEngineWrapper::default())),
        }
    }
}

impl Drop for GpuSpatialIndexFloat2DWrapper {
    fn drop(&mut self) {
        // Call the release function if it exists
        if let Some(release_fn) = self.index.release {
            unsafe {
                release_fn(&mut self.index as *mut _);
            }
        }
    }
}

#[repr(u32)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum GpuSpatialRelationPredicateWrapper {
    Equals = 0,
    Disjoint = 1,
    Touches = 2,
    Contains = 3,
    Covers = 4,
    Intersects = 5,
    Within = 6,
    CoveredBy = 7,
}

impl TryFrom<c_uint> for GpuSpatialRelationPredicateWrapper {
    type Error = &'static str;

    fn try_from(v: c_uint) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(GpuSpatialRelationPredicateWrapper::Equals),
            1 => Ok(GpuSpatialRelationPredicateWrapper::Disjoint),
            2 => Ok(GpuSpatialRelationPredicateWrapper::Touches),
            3 => Ok(GpuSpatialRelationPredicateWrapper::Contains),
            4 => Ok(GpuSpatialRelationPredicateWrapper::Covers),
            5 => Ok(GpuSpatialRelationPredicateWrapper::Intersects),
            6 => Ok(GpuSpatialRelationPredicateWrapper::Within),
            7 => Ok(GpuSpatialRelationPredicateWrapper::CoveredBy),
            _ => Err("Invalid GpuSpatialPredicate value"),
        }
    }
}

pub struct GpuSpatialRefinerWrapper {
    refiner: GpuSpatialRefiner,
    _rt_engine: Arc<Mutex<GpuSpatialRTEngineWrapper>>, // Keep a reference to the RT engine to ensure it lives as long as the refiner
}

impl GpuSpatialRefinerWrapper {
    /// # Initializes the GpuSpatialJoiner
    /// This function should only be called once per joiner instance.
    ///
    /// # Arguments
    /// * `concurrency` - How many threads will call the joiner concurrently.
    /// * `ptx_root` - The root directory for PTX files.
    pub fn try_new(
        rt_engine: &Arc<Mutex<GpuSpatialRTEngineWrapper>>,
        concurrency: u32,
    ) -> Result<Self, GpuSpatialError> {
        let mut refiner = GpuSpatialRefiner {
            init: None,
            load_build_array: None,
            refine_loaded: None,
            refine: None,
            release: None,
            private_data: std::ptr::null_mut(),
            last_error: std::ptr::null(),
        };

        unsafe {
            // Set function pointers to the C functions
            GpuSpatialRefinerCreate(&mut refiner);
        }

        if let Some(init_fn) = refiner.init {
            let mut engine_guard = rt_engine
                .lock()
                .map_err(|_| GpuSpatialError::Init("Failed to acquire mutex lock".to_string()))?;

            let mut config = GpuSpatialRefinerConfig {
                rt_engine: &mut engine_guard.rt_engine,
                concurrency,
                device_id: engine_guard.device_id,
            };

            // This is an unsafe call because it's calling a C function from the bindings.
            unsafe {
                if init_fn(&refiner as *const _ as *mut _, &mut config) != 0 {
                    let error_message = refiner.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    return Err(GpuSpatialError::Init(error_string));
                }
            }
        }
        Ok(GpuSpatialRefinerWrapper {
            refiner,
            _rt_engine: rt_engine.clone(),
        })
    }

    /// # Loads a build array into the GPU spatial refiner
    /// This function loads an array of geometries into the GPU spatial refiner
    /// for parsing and loading on the GPU side.
    /// # Arguments
    /// * `array` - The array of geometries to load.
    /// # Returns
    /// * `Result<(), GpuSpatialError>` - Ok if successful, Err if an error occurred.
    pub fn load_build_array(&self, array: &ArrayRef) -> Result<(), GpuSpatialError> {
        log::debug!(
            "DEBUG FFI: load_build_array called with array={}",
            array.len(),
        );

        let (ffi_array, ffi_schema) = arrow_array::ffi::to_ffi(&array.to_data())?;
        log::debug!("DEBUG FFI: FFI conversion successful");
        if let Some(load_fn) = self.refiner.load_build_array {
            unsafe {
                let ffi_array_ptr: *const ArrowArray =
                    transmute(&ffi_array as *const FFI_ArrowArray);
                let ffi_schema_ptr: *const ArrowSchema =
                    transmute(&ffi_schema as *const FFI_ArrowSchema);
                log::debug!("DEBUG FFI: Calling C++ refine function");
                let mut new_len: u32 = 0;
                if load_fn(
                    &self.refiner as *const _ as *mut _,
                    ffi_schema_ptr as *mut _,
                    ffi_array_ptr as *mut _,
                ) != 0
                {
                    let error_message = self.refiner.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    log::error!("DEBUG FFI: load_build_array failed: {}", error_string);
                    return Err(GpuSpatialError::PushStream(error_string));
                }
                log::debug!("DEBUG FFI: load_build_array C++ call succeeded");
            }
        }
        Ok(())
    }

    /// # Refines candidate pairs using the GPU spatial refiner
    /// This function refines candidate pairs of geometries using the GPU spatial refiner.
    /// It takes the probe side array of geometries and a predicate, and outputs the refined pairs of
    /// indices that satisfy the predicate.
    /// # Arguments
    /// * `array` - The array of geometries on the probe side.
    /// * `predicate` - The spatial relation predicate to use for refinement.
    /// * `build_indices` - The input/output vector of indices for the first array.
    /// * `probe_indices` - The input/output vector of indices for the second array.
    /// # Returns
    /// * `Result<(), GpuSpatialError>` - Ok if successful, Err if an error occurred.
    pub fn refine_loaded(
        &self,
        array: &ArrayRef,
        predicate: GpuSpatialRelationPredicateWrapper,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<(), GpuSpatialError> {
        log::debug!(
            "DEBUG FFI: refine called with array={}, indices={}, predicate={:?}",
            array.len(),
            build_indices.len(),
            predicate
        );

        let (ffi_array, ffi_schema) = arrow_array::ffi::to_ffi(&array.to_data())?;

        log::debug!("DEBUG FFI: FFI conversion successful");

        if let Some(refine_fn) = self.refiner.refine_loaded {
            unsafe {
                let ffi_array_ptr: *const ArrowArray =
                    transmute(&ffi_array as *const FFI_ArrowArray);
                let ffi_schema_ptr: *const ArrowSchema =
                    transmute(&ffi_schema as *const FFI_ArrowSchema);
                log::debug!("DEBUG FFI: Calling C++ refine function");
                let mut new_len: u32 = 0;
                if refine_fn(
                    &self.refiner as *const _ as *mut _,
                    ffi_schema_ptr as *mut _,
                    ffi_array_ptr as *mut _,
                    predicate as c_uint,
                    build_indices.as_mut_ptr(),
                    probe_indices.as_mut_ptr(),
                    build_indices.len() as u32,
                    &mut new_len as *mut u32,
                ) != 0
                {
                    let error_message = self.refiner.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    log::error!("DEBUG FFI: refine failed: {}", error_string);
                    return Err(GpuSpatialError::PushStream(error_string));
                }
                log::debug!("DEBUG FFI: refine C++ call succeeded");
                // Update the lengths of the output index vectors
                build_indices.truncate(new_len as usize);
                probe_indices.truncate(new_len as usize);
            }
        }
        Ok(())
    }
    /// # Refines candidate pairs using the GPU spatial refiner
    /// This function refines candidate pairs of geometries using the GPU spatial refiner.
    /// It takes two arrays of geometries and a predicate, and outputs the refined pairs of
    /// indices that satisfy the predicate.
    /// # Arguments
    /// * `array1` - The first array of geometries.
    /// * `array2` - The second array of geometries.
    /// * `predicate` - The spatial relation predicate to use for refinement.
    /// * `indices1` - The input/output vector of indices for the first array.
    /// * `indices2` - The input/output vector of indices for the second array.
    /// # Returns
    /// * `Result<(), GpuSpatialError>` - Ok if successful, Err if an error occurred.
    pub fn refine(
        &self,
        array1: &ArrayRef,
        array2: &ArrayRef,
        predicate: GpuSpatialRelationPredicateWrapper,
        indices1: &mut Vec<u32>,
        indices2: &mut Vec<u32>,
    ) -> Result<(), GpuSpatialError> {
        log::debug!(
            "DEBUG FFI: refine called with array1={}, array2={}, indices={}, predicate={:?}",
            array1.len(),
            array2.len(),
            indices1.len(),
            predicate
        );

        let (ffi_array1, ffi_schema1) = arrow_array::ffi::to_ffi(&array1.to_data())?;
        let (ffi_array2, ffi_schema2) = arrow_array::ffi::to_ffi(&array2.to_data())?;

        log::debug!("DEBUG FFI: FFI conversion successful");

        if let Some(refine_fn) = self.refiner.refine {
            unsafe {
                let ffi_array1_ptr: *const ArrowArray =
                    transmute(&ffi_array1 as *const FFI_ArrowArray);
                let ffi_schema1_ptr: *const ArrowSchema =
                    transmute(&ffi_schema1 as *const FFI_ArrowSchema);
                let ffi_array2_ptr: *const ArrowArray =
                    transmute(&ffi_array2 as *const FFI_ArrowArray);
                let ffi_schema2_ptr: *const ArrowSchema =
                    transmute(&ffi_schema2 as *const FFI_ArrowSchema);
                log::debug!("DEBUG FFI: Calling C++ refine function");
                let mut new_len: u32 = 0;
                if refine_fn(
                    &self.refiner as *const _ as *mut _,
                    ffi_schema1_ptr as *mut _,
                    ffi_array1_ptr as *mut _,
                    ffi_schema2_ptr as *mut _,
                    ffi_array2_ptr as *mut _,
                    predicate as c_uint,
                    indices1.as_mut_ptr(),
                    indices2.as_mut_ptr(),
                    indices1.len() as u32,
                    &mut new_len as *mut u32,
                ) != 0
                {
                    let error_message = self.refiner.last_error;
                    let c_str = std::ffi::CStr::from_ptr(error_message);
                    let error_string = c_str.to_string_lossy().into_owned();
                    log::error!("DEBUG FFI: refine failed: {}", error_string);
                    return Err(GpuSpatialError::PushStream(error_string));
                }
                log::debug!("DEBUG FFI: refine C++ call succeeded");
                // Update the lengths of the output index vectors
                indices1.truncate(new_len as usize);
                indices2.truncate(new_len as usize);
            }
        }
        Ok(())
    }
}

impl Default for GpuSpatialRefinerWrapper {
    fn default() -> Self {
        GpuSpatialRefinerWrapper {
            refiner: GpuSpatialRefiner {
                init: None,
                load_build_array: None,
                refine_loaded: None,
                refine: None,
                release: None,
                private_data: std::ptr::null_mut(),
                last_error: std::ptr::null(),
            },
            _rt_engine: Arc::new(Mutex::new(GpuSpatialRTEngineWrapper::default())),
        }
    }
}

impl Drop for GpuSpatialRefinerWrapper {
    fn drop(&mut self) {
        // Call the release function if it exists
        if let Some(release_fn) = self.refiner.release {
            unsafe {
                release_fn(&mut self.refiner as *mut _);
            }
        }
    }
}
