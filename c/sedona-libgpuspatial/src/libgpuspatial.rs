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
use std::cell::UnsafeCell;
use std::convert::TryFrom;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;

pub struct GpuSpatialRuntimeWrapper {
    runtime: UnsafeCell<GpuSpatialRuntime>,
    /// Store which device the runtime is created on
    pub device_id: i32,
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
        } else {
            return Err(GpuSpatialError::Init("init function is None".to_string()));
        }

        Ok(GpuSpatialRuntimeWrapper {
            runtime: UnsafeCell::new(runtime),
            device_id,
        })
    }
}

impl Drop for GpuSpatialRuntimeWrapper {
    fn drop(&mut self) {
        let runtime = self.runtime.get_mut();
        let release_fn = runtime.release.expect("release function is None");
        unsafe {
            release_fn(runtime as *mut _);
        }
    }
}

/// Internal wrapper that manages the lifecycle of the C `SedonaFloatIndex2D` struct.
/// It is wrapped in an `Arc` by the public structs to ensure thread safety.
struct FloatIndex2DWrapper {
    index: SedonaFloatIndex2D,
    // Keep a reference to the RT engine to ensure it lives as long as the index
    _runtime: Arc<GpuSpatialRuntimeWrapper>,
}

impl Drop for FloatIndex2DWrapper {
    fn drop(&mut self) {
        let release_fn = self.index.release.expect("release function is None");
        unsafe {
            release_fn(&mut self.index as *mut _);
        }
    }
}

pub struct FloatIndex2D {
    inner: FloatIndex2DWrapper,
}

impl FloatIndex2D {
    pub fn try_new(
        runtime: Arc<GpuSpatialRuntimeWrapper>,
        concurrency: u32,
    ) -> Result<Self, GpuSpatialError> {
        let mut index = SedonaFloatIndex2D {
            clear: None,
            create_context: None,
            destroy_context: None,
            push_build: None,
            finish_building: None,
            probe: None,
            get_last_error: None,
            context_get_last_error: None,
            release: None,
            private_data: std::ptr::null_mut(),
        };

        let config = GpuSpatialIndexConfig {
            runtime: runtime.runtime.get(),
            concurrency,
        };

        unsafe {
            if GpuSpatialIndexFloat2DCreate(&mut index, &config) != 0 {
                return Err(GpuSpatialError::Init("Index Create failed".into()));
            }
        }

        Ok(Self {
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

    pub fn push_build(&mut self, buf: *const f32, n_rects: u32) -> Result<(), GpuSpatialError> {
        let push_fn =
            self.inner.index.push_build.ok_or_else(|| {
                GpuSpatialError::PushBuild("push_build function is None".to_string())
            })?;
        let get_last_error = self.inner.index.get_last_error;
        let index_ptr = &mut self.inner.index as *mut SedonaFloatIndex2D;

        unsafe {
            check_ffi_call(
                move || push_fn(index_ptr, buf, n_rects),
                get_last_error,
                index_ptr,
                GpuSpatialError::PushBuild,
            )
        }
    }

    pub fn finish_building(&mut self) -> Result<(), GpuSpatialError> {
        let finish_fn = self
            .inner
            .index
            .finish_building
            .ok_or_else(|| GpuSpatialError::FinishBuild("finish_building missing".into()))?;
        let get_last_error = self.inner.index.get_last_error;
        let index_ptr = &mut self.inner.index as *mut SedonaFloatIndex2D;

        unsafe {
            check_ffi_call(
                move || finish_fn(&mut self.inner.index),
                get_last_error,
                index_ptr,
                GpuSpatialError::FinishBuild,
            )
        }
    }

    pub fn probe(
        &self,
        buf: *const f32,
        n_rects: u32,
    ) -> Result<(Vec<u32>, Vec<u32>), GpuSpatialError> {
        let probe_fn = self
            .inner
            .index
            .probe
            .ok_or_else(|| GpuSpatialError::Probe("probe function is None".into()))?;
        let create_context_fn = self.inner.index.create_context;
        let destroy_context_fn = self.inner.index.destroy_context;
        let context_err_fn = self.inner.index.context_get_last_error;
        let index_ptr = &self.inner.index as *const _ as *mut SedonaFloatIndex2D;

        let mut ctx = SedonaSpatialIndexContext {
            private_data: std::ptr::null_mut(),
        };
        let mut state = ProbeState {
            results: (Vec::new(), Vec::new()),
            error: None,
        };

        unsafe {
            if let Some(create_ctx) = create_context_fn {
                create_ctx(&mut ctx);
            }

            let status = probe_fn(
                index_ptr,
                &mut ctx,
                buf,
                n_rects,
                Some(probe_callback_wrapper),
                &mut state as *mut _ as *mut c_void,
            );

            if status != 0 {
                let error_string = if let Some(get_ctx_err) = context_err_fn {
                    CStr::from_ptr(get_ctx_err(&mut ctx))
                        .to_string_lossy()
                        .into_owned()
                } else {
                    "Unknown context error during probe".to_string()
                };

                if let Some(destroy_ctx) = destroy_context_fn {
                    destroy_ctx(&mut ctx);
                }
                return Err(GpuSpatialError::Probe(error_string));
            }

            if let Some(callback_error) = state.error {
                if let Some(destroy_ctx) = destroy_context_fn {
                    destroy_ctx(&mut ctx);
                }
                return Err(callback_error);
            }

            if let Some(destroy_ctx) = destroy_context_fn {
                destroy_ctx(&mut ctx);
            }
        }

        Ok(state.results)
    }
}

struct RefinerWrapper {
    refiner: SedonaSpatialRefiner,
    _runtime: Arc<GpuSpatialRuntimeWrapper>,
}

impl Drop for RefinerWrapper {
    fn drop(&mut self) {
        let release_fn = self.refiner.release.expect("release function is None");
        unsafe {
            release_fn(&mut self.refiner as *mut _);
        }
    }
}
pub struct Refiner {
    inner: RefinerWrapper,
}

impl Refiner {
    pub fn try_new(
        runtime: Arc<GpuSpatialRuntimeWrapper>,
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

        let config = GpuSpatialRefinerConfig {
            runtime: runtime.runtime.get(),
            concurrency,
            compress_bvh,
            pipeline_batches,
        };

        unsafe {
            GpuSpatialRefinerCreate(&mut refiner, &config);
        }

        Ok(Self {
            inner: RefinerWrapper {
                refiner,
                _runtime: runtime.clone(),
            },
        })
    }

    pub fn init_schema(
        &mut self,
        build_dt: &DataType,
        probe_dt: &DataType,
    ) -> Result<(), GpuSpatialError> {
        let build_ffi = FFI_ArrowSchema::try_from(build_dt)?;
        let probe_ffi = FFI_ArrowSchema::try_from(probe_dt)?;
        let init_fn = self.inner.refiner.init_schema.unwrap();
        let get_last_error = self.inner.refiner.get_last_error;
        let refiner_ptr = &mut self.inner.refiner as *mut SedonaSpatialRefiner;

        unsafe {
            check_ffi_call(
                || {
                    init_fn(
                        &mut self.inner.refiner,
                        &build_ffi as *const _ as *const _,
                        &probe_ffi as *const _ as *const _,
                    )
                },
                get_last_error,
                refiner_ptr,
                GpuSpatialError::Init,
            )
        }
    }

    pub fn push_build(&mut self, array: &ArrayRef) -> Result<(), GpuSpatialError> {
        let (ffi_array, _) = arrow_array::ffi::to_ffi(&array.to_data())?;
        let push_fn = self.inner.refiner.push_build.unwrap();
        let get_last_error = self.inner.refiner.get_last_error;
        let refiner_ptr = &mut self.inner.refiner as *mut SedonaSpatialRefiner;

        unsafe {
            check_ffi_call(
                || push_fn(&mut self.inner.refiner, &ffi_array as *const _ as *const _),
                get_last_error,
                refiner_ptr,
                GpuSpatialError::PushBuild,
            )
        }
    }

    pub fn clear(&mut self) {
        if let Some(clear_fn) = self.inner.refiner.clear {
            unsafe {
                clear_fn(&mut self.inner.refiner as *mut _);
            }
        }
    }

    pub fn finish_building(&mut self) -> Result<(), GpuSpatialError> {
        let finish_fn = self.inner.refiner.finish_building.unwrap();
        let get_last_error = self.inner.refiner.get_last_error;
        let refiner_ptr = &mut self.inner.refiner as *mut SedonaSpatialRefiner;

        unsafe {
            check_ffi_call(
                || finish_fn(&mut self.inner.refiner),
                get_last_error,
                refiner_ptr,
                GpuSpatialError::FinishBuild,
            )
        }
    }

    pub fn refine(
        &self,
        array: &ArrayRef,
        predicate: GpuSpatialRelationPredicate,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<(), GpuSpatialError> {
        let (ffi_array, _) = arrow_array::ffi::to_ffi(&array.to_data())?;
        let refine_fn = self.inner.refiner.refine.unwrap();
        let mut new_len: u32 = 0;

        unsafe {
            check_ffi_call(
                || {
                    refine_fn(
                        &self.inner.refiner as *const _ as *mut _,
                        &ffi_array as *const _ as *mut _,
                        predicate.as_c_uint(),
                        build_indices.as_mut_ptr(),
                        probe_indices.as_mut_ptr(),
                        build_indices.len() as u32,
                        &mut new_len,
                    )
                },
                self.inner.refiner.get_last_error,
                &self.inner.refiner as *const _ as *mut _,
                GpuSpatialError::Refine,
            )?;
        }
        build_indices.truncate(new_len as usize);
        probe_indices.truncate(new_len as usize);
        Ok(())
    }
}

// ----------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------

// Define the exact signature of the C error-getting function
type ErrorFn<T> = unsafe extern "C" fn(*mut T) -> *const c_char;
struct ProbeState {
    results: (Vec<u32>, Vec<u32>),
    error: Option<GpuSpatialError>,
}
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

unsafe extern "C" fn probe_callback_wrapper(
    build_indices: *const u32,
    probe_indices: *const u32,
    length: u32,
    user_data: *mut c_void,
) {
    // Cast user_data back to our state struct
    let state = &mut *(user_data as *mut ProbeState);

    // Short-circuit: If an error already occurred in a previous call,
    // stop processing to save time and prevent overwriting the error.
    if state.error.is_some() {
        return;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if length > 0 {
            let build_slice = std::slice::from_raw_parts(build_indices, length as usize);
            let probe_slice = std::slice::from_raw_parts(probe_indices, length as usize);

            state.results.0.extend_from_slice(build_slice);
            state.results.1.extend_from_slice(probe_slice);
        }
    }));

    // If a panic occurred, capture it as an error
    if let Err(payload) = result {
        // Try to extract the panic message
        let msg = if let Some(s) = payload.downcast_ref::<&str>() {
            format!("Panic in callback: {}", s)
        } else if let Some(s) = payload.downcast_ref::<String>() {
            format!("Panic in callback: {}", s)
        } else {
            "Unknown panic in callback".to_string()
        };

        state.error = Some(GpuSpatialError::Probe(msg));
    }
}
