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
use arrow_array::{Array, ArrayRef};
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::DataType;
use std::convert::TryFrom;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_uint};
use std::sync::{Arc, Mutex};

// ----------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------
/// Helper to handle the common pattern of calling a C function returning an int status,
/// checking if it failed, and retrieving the error message if so.
///
/// T: The type of the object (Runtime, Index, Refiner) being operated on.
unsafe fn check_ffi_call<T, F, ErrMap>(
    call_fn: F,
    get_error_fn: Option<unsafe extern "C" fn(*mut T) -> *const c_char>,
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

        log::error!("GpuSpatial FFI Error: {}", error_string);
        return Err(err_mapper(error_string));
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Runtime Wrapper
// ----------------------------------------------------------------------

pub struct GpuSpatialRuntimeWrapper {
    runtime: GpuSpatialRuntime,
}

impl GpuSpatialRuntimeWrapper {
    /// Initializes the GpuSpatialRuntime.
    /// This function should only be called once per engine instance.
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
                check_ffi_call(
                    || init_fn(&runtime as *const _ as *mut _, &mut config),
                    runtime.get_last_error,
                    &runtime as *const _ as *mut _,
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
// Spatial Index Wrapper
// ----------------------------------------------------------------------

pub struct GpuSpatialIndexFloat2DWrapper {
    index: SedonaFloatIndex2D,
    // Keep a reference to the RT engine to ensure it lives as long as the index
    _runtime: Arc<Mutex<GpuSpatialRuntimeWrapper>>,
}

impl GpuSpatialIndexFloat2DWrapper {
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
                // Can't use check_ffi_call helper here easily because 'index' isn't fully initialized/wrapped yet
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

        Ok(GpuSpatialIndexFloat2DWrapper {
            index,
            _runtime: runtime.clone(),
        })
    }

    pub fn clear(&mut self) {
        if let Some(clear_fn) = self.index.clear {
            unsafe {
                clear_fn(&mut self.index as *mut _);
            }
        }
    }

    pub unsafe fn push_build(
        &mut self,
        buf: *const f32,
        n_rects: u32,
    ) -> Result<(), GpuSpatialError> {
        if let Some(push_build_fn) = self.index.push_build {
            let get_last_error = self.index.get_last_error;
            let index_ptr = &mut self.index as *mut _;

            check_ffi_call(
                move || push_build_fn(index_ptr, buf, n_rects),
                get_last_error,
                index_ptr,
                GpuSpatialError::PushBuild,
            )?;
        }
        Ok(())
    }

    pub fn finish_building(&mut self) -> Result<(), GpuSpatialError> {
        if let Some(finish_building_fn) = self.index.finish_building {
            let get_last_error = self.index.get_last_error;
            let index_ptr = &mut self.index as *mut _;

            unsafe {
                check_ffi_call(
                    move || finish_building_fn(index_ptr),
                    get_last_error,
                    index_ptr,
                    GpuSpatialError::FinishBuild,
                )?;
            }
        }
        Ok(())
    }

    pub fn create_context(&self, ctx: &mut SedonaSpatialIndexContext) {
        if let Some(create_context_fn) = self.index.create_context {
            unsafe {
                create_context_fn(ctx as *mut _);
            }
        }
    }

    pub fn destroy_context(&self, ctx: &mut SedonaSpatialIndexContext) {
        if let Some(destroy_context_fn) = self.index.destroy_context {
            unsafe {
                destroy_context_fn(ctx as *mut _);
            }
        }
    }

    pub unsafe fn probe(
        &self,
        ctx: &mut SedonaSpatialIndexContext,
        buf: *const f32,
        n_rects: u32,
    ) -> Result<(), GpuSpatialError> {
        if let Some(probe_fn) = self.index.probe {
            // Note: probe uses context_get_last_error, not the index one
            if probe_fn(
                &self.index as *const _ as *mut _,
                ctx as *mut _,
                buf,
                n_rects,
            ) != 0
            {
                let error_string = if let Some(get_ctx_err) = self.index.context_get_last_error {
                    CStr::from_ptr(get_ctx_err(ctx))
                        .to_string_lossy()
                        .into_owned()
                } else {
                    "Unknown context error".to_string()
                };
                log::error!("DEBUG FFI: probe failed: {}", error_string);
                return Err(GpuSpatialError::Probe(error_string));
            }
        }
        Ok(())
    }

    fn get_indices_buffer_helper(
        &self,
        ctx: &mut SedonaSpatialIndexContext,
        func: Option<unsafe extern "C" fn(*mut SedonaSpatialIndexContext, *mut *mut u32, *mut u32)>,
    ) -> &[u32] {
        if let Some(f) = func {
            let mut ptr: *mut u32 = std::ptr::null_mut();
            let mut len: u32 = 0;
            unsafe {
                f(ctx, &mut ptr, &mut len);
                if len > 0 && !ptr.is_null() {
                    return std::slice::from_raw_parts(ptr, len as usize);
                }
            }
        }
        &[]
    }

    pub fn get_build_indices_buffer(&self, ctx: &mut SedonaSpatialIndexContext) -> &[u32] {
        self.get_indices_buffer_helper(ctx, self.index.get_build_indices_buffer)
    }

    pub fn get_probe_indices_buffer(&self, ctx: &mut SedonaSpatialIndexContext) -> &[u32] {
        self.get_indices_buffer_helper(ctx, self.index.get_probe_indices_buffer)
    }
}

impl Drop for GpuSpatialIndexFloat2DWrapper {
    fn drop(&mut self) {
        if let Some(release_fn) = self.index.release {
            unsafe {
                release_fn(&mut self.index as *mut _);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Predicate Wrapper
// ----------------------------------------------------------------------

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
            0 => Ok(Self::Equals),
            1 => Ok(Self::Disjoint),
            2 => Ok(Self::Touches),
            3 => Ok(Self::Contains),
            4 => Ok(Self::Covers),
            5 => Ok(Self::Intersects),
            6 => Ok(Self::Within),
            7 => Ok(Self::CoveredBy),
            _ => Err("Invalid GpuSpatialPredicate value"),
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
        &self,
        build_data_type: &DataType,
        probe_data_type: &DataType,
    ) -> Result<(), GpuSpatialError> {
        let ffi_build_schema = FFI_ArrowSchema::try_from(build_data_type)?;
        let ffi_probe_schema = FFI_ArrowSchema::try_from(probe_data_type)?;

        if let Some(init_schema_fn) = self.refiner.init_schema {
            // Use pointer casting instead of transmute for safer FFI
            let ffi_build_ptr = &ffi_build_schema as *const _ as *const ArrowSchema;
            let ffi_probe_ptr = &ffi_probe_schema as *const _ as *const ArrowSchema;

            unsafe {
                check_ffi_call(
                    || {
                        init_schema_fn(
                            &self.refiner as *const _ as *mut _,
                            ffi_build_ptr,
                            ffi_probe_ptr,
                        )
                    },
                    self.refiner.get_last_error,
                    &self.refiner as *const _ as *mut _,
                    GpuSpatialError::Init,
                )?;
            }
        }
        Ok(())
    }

    pub fn clear(&self) {
        if let Some(clear_fn) = self.refiner.clear {
            unsafe {
                clear_fn(&self.refiner as *const _ as *mut _);
            }
        }
    }

    pub fn push_build(&self, array: &ArrayRef) -> Result<(), GpuSpatialError> {
        // Keep ffi_array alive until the C function returns
        let (ffi_array, _ffi_schema) = arrow_array::ffi::to_ffi(&array.to_data())?;

        if let Some(push_build_fn) = self.refiner.push_build {
            let ffi_array_ptr = &ffi_array as *const _ as *const ArrowArray;
            unsafe {
                check_ffi_call(
                    || push_build_fn(&self.refiner as *const _ as *mut _, ffi_array_ptr as *mut _),
                    self.refiner.get_last_error,
                    &self.refiner as *const _ as *mut _,
                    GpuSpatialError::PushBuild,
                )?;
            }
        }
        Ok(())
    }

    pub fn finish_building(&self) -> Result<(), GpuSpatialError> {
        if let Some(finish_building_fn) = self.refiner.finish_building {
            unsafe {
                check_ffi_call(
                    || finish_building_fn(&self.refiner as *const _ as *mut _),
                    self.refiner.get_last_error,
                    &self.refiner as *const _ as *mut _,
                    GpuSpatialError::FinishBuild,
                )?;
            }
        }
        Ok(())
    }

    pub fn refine(
        &self,
        probe_array: &ArrayRef,
        predicate: GpuSpatialRelationPredicateWrapper,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<(), GpuSpatialError> {
        let (ffi_array, _ffi_schema) = arrow_array::ffi::to_ffi(&probe_array.to_data())?;

        if let Some(refine_fn) = self.refiner.refine {
            let ffi_array_ptr = &ffi_array as *const _ as *const ArrowArray;
            let mut new_len: u32 = 0;

            unsafe {
                check_ffi_call(
                    || {
                        refine_fn(
                            &self.refiner as *const _ as *mut _,
                            ffi_array_ptr as *mut _,
                            predicate as c_uint,
                            build_indices.as_mut_ptr(),
                            probe_indices.as_mut_ptr(),
                            build_indices.len() as u32,
                            &mut new_len as *mut u32,
                        )
                    },
                    self.refiner.get_last_error,
                    &self.refiner as *const _ as *mut _,
                    GpuSpatialError::Refine,
                )?;
            }

            // Update the lengths of the output index vectors based on GPU result
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
