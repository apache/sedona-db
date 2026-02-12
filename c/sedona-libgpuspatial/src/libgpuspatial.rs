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
#[cfg(gpu_available)]
use crate::libgpuspatial_glue_bindgen::*;
use crate::predicate::GpuSpatialRelationPredicate;
use arrow_array::{Array, ArrayRef};
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::DataType;
use std::convert::TryFrom;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::{Arc, Mutex};

// ----------------------------------------------------------------------
// Runtime Wrapper
// ----------------------------------------------------------------------

pub struct GpuSpatialRuntimeWrapper {
    runtime: GpuSpatialRuntime,
}

impl GpuSpatialRuntimeWrapper {
    pub fn try_new(
        device_id: i32,
        ptx_root: &str,
        use_cuda_memory_pool: bool,
        cuda_memory_pool_init_precent: i32,
    ) -> Result<GpuSpatialRuntimeWrapper, GpuSpatialError> {
        let mut runtime = GpuSpatialRuntime {
            init: None,
            release: None,
            get_last_error: None,
            private_data: std::ptr::null_mut(),
        };

        unsafe {
            GpuSpatialRuntimeCreate(&mut runtime);
        }

        if let Some(init_fn) = runtime.init {
            let c_ptx_root = CString::new(ptx_root).map_err(|_| {
                GpuSpatialError::Init("Failed to convert ptx_root to CString".into())
            })?;

            let mut config = GpuSpatialRuntimeConfig {
                device_id,
                ptx_root: c_ptx_root.as_ptr(),
                use_cuda_memory_pool,
                cuda_memory_pool_init_precent,
            };

            unsafe {
                let get_last_error = runtime.get_last_error;
                let runtime_ptr = &mut runtime as *mut GpuSpatialRuntime;

                check_ffi_call(
                    move || init_fn(runtime_ptr as *mut _, &mut config),
                    get_last_error,
                    runtime_ptr,
                    GpuSpatialError::Init,
                )?;
            }
        }

        Ok(GpuSpatialRuntimeWrapper { runtime })
    }
}

impl Drop for GpuSpatialRuntimeWrapper {
    fn drop(&mut self) {
        if let Some(release_fn) = self.runtime.release {
            unsafe {
                release_fn(&mut self.runtime as *mut _);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Spatial Index - Internal Wrapper
// ----------------------------------------------------------------------

/// Internal wrapper that manages the lifecycle of the C `SedonaFloatIndex2D` struct.
/// It is wrapped in an `Arc` by the public structs to ensure thread safety.
struct FloatIndex2DWrapper {
    index: SedonaFloatIndex2D,
    // Keep a reference to the RT engine to ensure it lives as long as the index
    _runtime: Arc<Mutex<GpuSpatialRuntimeWrapper>>,
}

// The C library is designed for thread safety when used correctly (separate contexts per thread)
unsafe impl Send for FloatIndex2DWrapper {}
unsafe impl Sync for FloatIndex2DWrapper {}

impl Drop for FloatIndex2DWrapper {
    fn drop(&mut self) {
        if let Some(release_fn) = self.index.release {
            unsafe {
                release_fn(&mut self.index as *mut _);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Spatial Index - Builder
// ----------------------------------------------------------------------

/// Builder for the Spatial Index. This struct has exclusive ownership
/// and is not thread-safe (Send but not Sync) because building is a
/// single-threaded operation.
pub struct FloatIndex2DBuilder {
    inner: FloatIndex2DWrapper,
}

impl FloatIndex2DBuilder {
    pub fn try_new(
        runtime: Arc<Mutex<GpuSpatialRuntimeWrapper>>,
        concurrency: u32,
    ) -> Result<Self, GpuSpatialError> {
        let mut index = SedonaFloatIndex2D {
            clear: None,
            create_context: None,
            destroy_context: None,
            push_build: None,
            finish_building: None,
            probe: None,
            get_build_indices_buffer: None,
            get_probe_indices_buffer: None,
            get_last_error: None,
            context_get_last_error: None,
            release: None,
            private_data: std::ptr::null_mut(),
        };

        let mut engine_guard = runtime
            .lock()
            .map_err(|_| GpuSpatialError::Init("Failed to acquire mutex lock".to_string()))?;

        let config = GpuSpatialIndexConfig {
            runtime: &mut engine_guard.runtime,
            concurrency,
        };

        unsafe {
            if GpuSpatialIndexFloat2DCreate(&mut index, &config) != 0 {
                let msg = if let Some(get_err) = index.get_last_error {
                    CStr::from_ptr(get_err(&index as *const _ as *mut _))
                        .to_string_lossy()
                        .into_owned()
                } else {
                    "Unknown error during Index Create".into()
                };
                return Err(GpuSpatialError::Init(msg));
            }
        }

        Ok(FloatIndex2DBuilder {
            inner: FloatIndex2DWrapper {
                index,
                _runtime: runtime.clone(),
            },
        })
    }

    pub fn clear(&mut self) {
        if let Some(clear_fn) = self.inner.index.clear {
            unsafe {
                clear_fn(&mut self.inner.index as *mut _);
            }
        }
    }

    pub unsafe fn push_build(
        &mut self,
        buf: *const f32,
        n_rects: u32,
    ) -> Result<(), GpuSpatialError> {
        if let Some(push_build_fn) = self.inner.index.push_build {
            let get_last_error = self.inner.index.get_last_error;
            let index_ptr = &mut self.inner.index as *mut _;

            check_ffi_call(
                move || push_build_fn(index_ptr, buf, n_rects),
                get_last_error,
                index_ptr,
                GpuSpatialError::PushBuild,
            )?;
        }
        Ok(())
    }

    /// Consumes the builder and returns a shared, thread-safe index wrapper.
    pub fn finish(mut self) -> Result<SharedFloatIndex2D, GpuSpatialError> {
        if let Some(finish_building_fn) = self.inner.index.finish_building {
            // Extract to local vars
            let get_last_error = self.inner.index.get_last_error;
            let index_ptr = &mut self.inner.index as *mut _;

            unsafe {
                check_ffi_call(
                    move || finish_building_fn(index_ptr),
                    get_last_error,
                    index_ptr,
                    GpuSpatialError::FinishBuild,
                )?;
            }
        }

        Ok(SharedFloatIndex2D {
            inner: Arc::new(self.inner),
        })
    }
}

// ----------------------------------------------------------------------
// Spatial Index - Shared Read-Only Index
// ----------------------------------------------------------------------

/// Thread-safe wrapper around the built index.
/// Used to spawn thread-local contexts for probing.
#[derive(Clone)]
pub struct SharedFloatIndex2D {
    inner: Arc<FloatIndex2DWrapper>,
}

unsafe impl Send for SharedFloatIndex2D {}
unsafe impl Sync for SharedFloatIndex2D {}

impl SharedFloatIndex2D {
    pub fn create_context(&self) -> Result<FloatIndex2DContext, GpuSpatialError> {
        let mut ctx = SedonaSpatialIndexContext {
            private_data: std::ptr::null_mut(),
        };

        if let Some(create_context_fn) = self.inner.index.create_context {
            unsafe {
                create_context_fn(&mut ctx);
            }
        }

        Ok(FloatIndex2DContext {
            inner: self.inner.clone(),
            context: ctx,
        })
    }
}

// ----------------------------------------------------------------------
// Spatial Index - Thread Local Context
// ----------------------------------------------------------------------

/// Thread-local context for probing the index.
/// This struct is Send (can be moved between threads) but NOT Sync.
pub struct FloatIndex2DContext {
    inner: Arc<FloatIndex2DWrapper>,
    context: SedonaSpatialIndexContext,
}

unsafe impl Send for FloatIndex2DContext {}

impl FloatIndex2DContext {
    pub unsafe fn probe(&mut self, buf: *const f32, n_rects: u32) -> Result<(), GpuSpatialError> {
        if let Some(probe_fn) = self.inner.index.probe {
            // We need a mutable pointer to the index for the C API, even though conceptually it's "probing".
            // Since `inner` is wrapped in an Arc, we cast the const pointer to mut.
            // SAFETY: The C library handles concurrent probes via the separate `context` objects.
            let index_ptr = &self.inner.index as *const _ as *mut SedonaFloatIndex2D;

            if probe_fn(index_ptr, &mut self.context, buf, n_rects) != 0 {
                let error_string =
                    if let Some(get_ctx_err) = self.inner.index.context_get_last_error {
                        CStr::from_ptr(get_ctx_err(&mut self.context))
                            .to_string_lossy()
                            .into_owned()
                    } else {
                        "Unknown context error".to_string()
                    };
                return Err(GpuSpatialError::Probe(error_string));
            }
        }
        Ok(())
    }

    fn get_indices_buffer_helper(
        &mut self,
        func: Option<unsafe extern "C" fn(*mut SedonaSpatialIndexContext, *mut *mut u32, *mut u32)>,
    ) -> &[u32] {
        if let Some(f) = func {
            let mut ptr: *mut u32 = std::ptr::null_mut();
            let mut len: u32 = 0;
            unsafe {
                f(&mut self.context, &mut ptr, &mut len);
                if len > 0 && !ptr.is_null() {
                    return std::slice::from_raw_parts(ptr, len as usize);
                }
            }
        }
        &[]
    }

    pub fn get_build_indices_buffer(&mut self) -> &[u32] {
        self.get_indices_buffer_helper(self.inner.index.get_build_indices_buffer)
    }

    pub fn get_probe_indices_buffer(&mut self) -> &[u32] {
        self.get_indices_buffer_helper(self.inner.index.get_probe_indices_buffer)
    }
}

impl Drop for FloatIndex2DContext {
    fn drop(&mut self) {
        if let Some(destroy_context_fn) = self.inner.index.destroy_context {
            unsafe {
                destroy_context_fn(&mut self.context);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Refiner Wrapper
// ----------------------------------------------------------------------

pub struct GpuSpatialRefinerWrapper {
    refiner: SedonaSpatialRefiner,
    _runtime: Arc<Mutex<GpuSpatialRuntimeWrapper>>,
}

impl GpuSpatialRefinerWrapper {
    pub fn try_new(
        runtime: Arc<Mutex<GpuSpatialRuntimeWrapper>>,
        concurrency: u32,
        compress_bvh: bool,
        pipeline_batches: u32,
    ) -> Result<Self, GpuSpatialError> {
        let mut refiner = SedonaSpatialRefiner {
            clear: None,
            init_schema: None,
            push_build: None,
            finish_building: None,
            refine: None,
            get_last_error: None,
            release: None,
            private_data: std::ptr::null_mut(),
        };

        let mut engine_guard = runtime
            .lock()
            .map_err(|_| GpuSpatialError::Init("Failed to acquire mutex lock".to_string()))?;

        let config = GpuSpatialRefinerConfig {
            runtime: &mut engine_guard.runtime,
            concurrency,
            compress_bvh,
            pipeline_batches,
        };

        unsafe {
            if GpuSpatialRefinerCreate(&mut refiner, &config) != 0 {
                let msg = if let Some(get_err) = refiner.get_last_error {
                    CStr::from_ptr(get_err(&refiner as *const _ as *mut _))
                        .to_string_lossy()
                        .into_owned()
                } else {
                    "Unknown error during Refiner Create".into()
                };
                return Err(GpuSpatialError::Init(msg));
            }
        }

        Ok(GpuSpatialRefinerWrapper {
            refiner,
            _runtime: runtime.clone(),
        })
    }

    pub fn init_schema(
        &mut self,
        build_data_type: &DataType,
        probe_data_type: &DataType,
    ) -> Result<(), GpuSpatialError> {
        let ffi_build_schema = FFI_ArrowSchema::try_from(build_data_type)?;
        let ffi_probe_schema = FFI_ArrowSchema::try_from(probe_data_type)?;

        if let Some(init_schema_fn) = self.refiner.init_schema {
            let ffi_build_ptr = &ffi_build_schema as *const _ as *const ArrowSchema;
            let ffi_probe_ptr = &ffi_probe_schema as *const _ as *const ArrowSchema;

            let get_last_error = self.refiner.get_last_error;
            // Changed: Safely cast mutable reference to *mut
            let refiner_ptr = &mut self.refiner as *mut _;

            unsafe {
                check_ffi_call(
                    move || init_schema_fn(refiner_ptr, ffi_build_ptr, ffi_probe_ptr),
                    get_last_error,
                    refiner_ptr,
                    GpuSpatialError::Init,
                )?;
            }
        }
        Ok(())
    }

    pub fn clear(&mut self) {
        if let Some(clear_fn) = self.refiner.clear {
            unsafe {
                clear_fn(&mut self.refiner as *mut _);
            }
        }
    }

    pub fn push_build(&mut self, array: &ArrayRef) -> Result<(), GpuSpatialError> {
        let (ffi_array, _ffi_schema) = arrow_array::ffi::to_ffi(&array.to_data())?;

        if let Some(push_build_fn) = self.refiner.push_build {
            let ffi_array_ptr = &ffi_array as *const _ as *const ArrowArray;

            let get_last_error = self.refiner.get_last_error;
            let refiner_ptr = &mut self.refiner as *mut _;

            unsafe {
                check_ffi_call(
                    move || push_build_fn(refiner_ptr, ffi_array_ptr),
                    get_last_error,
                    refiner_ptr,
                    GpuSpatialError::PushBuild,
                )?;
            }
        }
        Ok(())
    }

    pub fn finish_building(&mut self) -> Result<(), GpuSpatialError> {
        if let Some(finish_building_fn) = self.refiner.finish_building {
            let get_last_error = self.refiner.get_last_error;
            let refiner_ptr = &mut self.refiner as *mut _;

            unsafe {
                check_ffi_call(
                    move || finish_building_fn(refiner_ptr),
                    get_last_error,
                    refiner_ptr,
                    GpuSpatialError::FinishBuild,
                )?;
            }
        }
        Ok(())
    }

    pub fn refine(
        &self,
        probe_array: &ArrayRef,
        predicate: GpuSpatialRelationPredicate,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<(), GpuSpatialError> {
        let (ffi_array, _ffi_schema) = arrow_array::ffi::to_ffi(&probe_array.to_data())?;

        if let Some(refine_fn) = self.refiner.refine {
            let ffi_array_ptr = &ffi_array as *const _ as *const ArrowArray;
            let mut new_len: u32 = 0;

            let get_last_error = self.refiner.get_last_error;
            let refiner_ptr = &self.refiner as *const _ as *mut _;

            let build_ptr = build_indices.as_mut_ptr();
            let probe_ptr = probe_indices.as_mut_ptr();
            let capacity = build_indices.len() as u32;
            let new_len_ptr = &mut new_len as *mut u32;
            let predicate_int = predicate.as_c_uint();

            unsafe {
                check_ffi_call(
                    move || {
                        refine_fn(
                            refiner_ptr,
                            ffi_array_ptr as *mut _,
                            predicate_int,
                            build_ptr,
                            probe_ptr,
                            capacity,
                            new_len_ptr,
                        )
                    },
                    get_last_error,
                    refiner_ptr,
                    GpuSpatialError::Refine,
                )?;
            }

            build_indices.truncate(new_len as usize);
            probe_indices.truncate(new_len as usize);
        }
        Ok(())
    }
}

impl Drop for GpuSpatialRefinerWrapper {
    fn drop(&mut self) {
        if let Some(release_fn) = self.refiner.release {
            unsafe {
                release_fn(&mut self.refiner as *mut _);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------

// Define the exact signature of the C error-getting function
type ErrorFn<T> = unsafe extern "C" fn(*mut T) -> *const c_char;

/// Helper to handle the common pattern of calling a C function returning an int status,
/// checking if it failed, and retrieving the error message if so.
unsafe fn check_ffi_call<T, F, ErrMap>(
    call_fn: F,
    get_error_fn: Option<ErrorFn<T>>,
    obj_ptr: *mut T,
    err_mapper: ErrMap,
) -> Result<(), GpuSpatialError>
where
    F: FnOnce() -> i32,
    ErrMap: FnOnce(String) -> GpuSpatialError,
{
    if call_fn() != 0 {
        let error_string = if let Some(get_err) = get_error_fn {
            let err_ptr = get_err(obj_ptr);
            if !err_ptr.is_null() {
                CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
            } else {
                "Unknown error (null error message)".to_string()
            }
        } else {
            "Unknown error (get_last_error not available)".to_string()
        };

        return Err(err_mapper(error_string));
    }
    Ok(())
}
