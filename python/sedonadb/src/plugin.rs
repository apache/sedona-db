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

//! Host-side plumbing for cross-extension UDTF registration via
//! `datafusion-ffi` — the same wire format `datafusion-python` uses.
//!
//! The host calls the plugin's `__datafusion_table_function__(session)`
//! with a session capsule carrying an [`FFI_LogicalExtensionCodec`].
//! The plugin returns a capsule named `"datafusion_table_function"`
//! wrapping an [`FFI_TableFunction`], which the host converts to an
//! `Arc<dyn TableFunctionImpl>` and registers on the session.

use std::sync::Arc;

use datafusion::execution::context::SessionContext;
use datafusion::execution::{TaskContext, TaskContextProvider};
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::udtf::FFI_TableFunction;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::CStr;
use std::ptr::NonNull;

use crate::error::PySedonaError;

const CODEC_CAPSULE_NAME: &CStr = c"datafusion_logical_extension_codec";
const UDTF_CAPSULE_NAME: &CStr = c"datafusion_table_function";

pub(crate) const UDTF_ATTR: &str = "__datafusion_table_function__";

#[derive(Debug)]
pub(crate) struct SessionTaskContextProvider {
    task_ctx: Arc<TaskContext>,
}

impl SessionTaskContextProvider {
    pub fn new(ctx: &SessionContext) -> Self {
        Self {
            task_ctx: ctx.task_ctx(),
        }
    }
}

impl TaskContextProvider for SessionTaskContextProvider {
    fn task_ctx(&self) -> Arc<TaskContext> {
        self.task_ctx.clone()
    }
}

/// Build the session capsule the host hands to the plugin.
///
/// The codec holds a `Weak<dyn TaskContextProvider>`, so `provider`
/// must outlive any UDTF registered through this capsule — see
/// `InternalContext::udtf_task_provider`.
pub(crate) fn create_session_capsule<'py>(
    py: Python<'py>,
    provider: &Arc<dyn TaskContextProvider + Send + Sync>,
) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
    // FFI_TaskContextProvider's `From` only needs TaskContextProvider;
    // drop the Send + Sync (carried for pyclass) at the boundary.
    let provider: Arc<dyn TaskContextProvider> = provider.clone();
    let codec = FFI_LogicalExtensionCodec::new_default(&provider);
    PyCapsule::new(py, codec, Some(CODEC_CAPSULE_NAME.to_owned())).map_err(PySedonaError::from)
}

pub(crate) fn ffi_table_function_from_capsule(
    capsule: &Bound<'_, PyCapsule>,
) -> Result<Arc<dyn datafusion::catalog::TableFunctionImpl>, PySedonaError> {
    let name = capsule
        .name()?
        .ok_or_else(|| PySedonaError::SedonaPython("UDTF capsule has no name".to_string()))?;
    if name != UDTF_CAPSULE_NAME {
        return Err(PySedonaError::SedonaPython(format!(
            "UDTF capsule name mismatch: expected {UDTF_CAPSULE_NAME:?}, got {name:?}"
        )));
    }
    let ptr = capsule.pointer() as *mut FFI_TableFunction;
    let ptr = NonNull::new(ptr)
        .ok_or_else(|| PySedonaError::SedonaPython("UDTF capsule pointer is null".to_string()))?;
    // SAFETY: name-matched above; clone() runs the FFI release-aware
    // hook so we own an independent struct that `From` unwraps.
    let ffi: FFI_TableFunction = unsafe { ptr.as_ref().clone() };
    Ok(ffi.into())
}
