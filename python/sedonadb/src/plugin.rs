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

//! Plugin handoff types shared by sedonadb and its Python plugin
//! crates (e.g. `sedonadb-zarr`).
//!
//! PyO3's `#[pyclass]` extraction uses a per-cdylib type-id static,
//! so a plugin can't extract sedonadb's `InternalContext` directly.
//! UDTF implementations cross the extension boundary inside a
//! [`PyCapsule`](pyo3::types::PyCapsule) wrapping an [`UdtfCapsule`].

use std::ffi::CStr;
use std::sync::Arc;

use datafusion::catalog::TableFunctionImpl;

/// Capsule name carried in [`pyo3::types::PyCapsule::name`].
///
/// The trailing `.v1` is an ABI tag: if the layout of [`UdtfCapsule`]
/// or the `TableFunctionImpl` vtable ever changes (e.g. a datafusion
/// major bump), bump the suffix so a pre-built plugin wheel paired
/// with a newer host fails fast at registration time instead of
/// silently corrupting on call.
pub const UDTF_CAPSULE_NAME: &CStr = c"sedonadb.udtf.v1";

/// Magic sentinel at offset 0 of [`UdtfCapsule`].
///
/// The capsule name is a public string; any Python caller can build a
/// capsule with that name pointing at arbitrary memory. The magic
/// gives the consumer a second, harder-to-forge check before it
/// dereferences the payload as an `Arc<dyn TableFunctionImpl>`.
pub const UDTF_CAPSULE_MAGIC: u64 = 0x5345_444F_4E41_5544;

/// Payload stored inside the UDTF [`pyo3::types::PyCapsule`].
///
/// `#[repr(C)]` guarantees `magic` is at offset 0 regardless of
/// alignment, so a stale or forged capsule can be rejected via a
/// single field read before touching `udtf`. The producer constructs
/// one of these; the consumer recovers it via
/// [`pyo3::types::PyCapsule::reference`].
#[repr(C)]
pub struct UdtfCapsule {
    pub magic: u64,
    pub udtf: Arc<dyn TableFunctionImpl>,
}

impl UdtfCapsule {
    /// Build a capsule payload around a [`TableFunctionImpl`].
    pub fn new(udtf: Arc<dyn TableFunctionImpl>) -> Self {
        Self {
            magic: UDTF_CAPSULE_MAGIC,
            udtf,
        }
    }
}
