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

//! Host-side plumbing for cross-extension UDTF registration.
//!
//! Format-specific Python plugins (e.g. `sedonadb-zarr`) build their
//! UDTF in their own PyO3 extension and hand it across the cdylib
//! boundary as a [`datafusion_ffi::udtf::FFI_TableFunction`] inside a
//! [`PyCapsule`] — the same wire format the
//! `datafusion-python` ecosystem uses for FFI table functions.
//!
//! The plugin exposes a `__datafusion_table_function__(session)`
//! method on its Python-visible spec class. The host calls that method
//! with a session capsule carrying an [`FFI_LogicalExtensionCodec`];
//! the plugin uses the codec to construct its `FFI_TableFunction` and
//! returns it inside a capsule named `"datafusion_table_function"`.
//! The host then converts the FFI struct into a regular
//! `Arc<dyn TableFunctionImpl>` (the conversion in `datafusion-ffi`
//! takes a same-library fast path or wraps a `ForeignTableFunction`
//! depending on the marker id) and registers it on the session.

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

/// Capsule name carried in the session capsule the host passes to the
/// plugin. Matches `datafusion-python`'s convention so plugins built
/// against either ecosystem can register on a sedonadb context.
pub const CODEC_CAPSULE_NAME: &CStr = c"datafusion_logical_extension_codec";

/// Capsule name returned by the plugin's
/// `__datafusion_table_function__`. Matches `datafusion-python`.
pub const UDTF_CAPSULE_NAME: &CStr = c"datafusion_table_function";

/// Python attribute the host invokes on the plugin's spec class.
pub const UDTF_ATTR: &str = "__datafusion_table_function__";

/// Adapt a sedonadb [`SessionContext`] to [`TaskContextProvider`].
///
/// The plugin's `FFI_TableFunction::call` needs a [`TaskContext`] at
/// runtime so it can deserialise the argument `Expr`s. We give it
/// access to ours via this thin wrapper — held as an `Arc` so the FFI
/// codec can clone it across the boundary.
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

/// Build a session capsule the host hands to the plugin.
///
/// The capsule carries an [`FFI_LogicalExtensionCodec`] over which the
/// plugin's `FFI_TableFunction` will serialise expressions.
pub(crate) fn create_session_capsule<'py>(
    py: Python<'py>,
    ctx: &SessionContext,
) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
    let provider: Arc<dyn TaskContextProvider> = Arc::new(SessionTaskContextProvider::new(ctx));
    let codec = FFI_LogicalExtensionCodec::new_default(&provider);
    PyCapsule::new(py, codec, Some(CODEC_CAPSULE_NAME.to_owned())).map_err(PySedonaError::from)
}

/// Extract an [`FFI_TableFunction`] from the capsule returned by the
/// plugin's `__datafusion_table_function__` method, and convert it into
/// a registrable `Arc<dyn TableFunctionImpl>`.
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
    // SAFETY: the capsule's payload is an `FFI_TableFunction` (name
    // matched above). PyO3 keeps the value alive for the lifetime of
    // the capsule; `clone()` runs the FFI release-aware clone hook so
    // we end up owning an independent FFI struct, which `From` then
    // unwraps into an `Arc<dyn TableFunctionImpl>` (same-cdylib fast
    // path or `ForeignTableFunction` wrapper, depending on marker).
    let ffi: FFI_TableFunction = unsafe { ptr.as_ref().clone() };
    Ok(ffi.into())
}
