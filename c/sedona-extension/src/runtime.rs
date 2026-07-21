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

//! A Tokio runtime owner that shuts down in the background on drop.

use std::mem::ManuallyDrop;
use std::ops::Deref;

use tokio::runtime::Runtime;

/// Owns a Tokio [`Runtime`] and shuts it down in the background when dropped.
///
/// The default `Runtime::drop` blocks the dropping thread while it joins every
/// worker thread. When a runtime is shared through `Arc<RuntimeHandle>`, the
/// final reference can drop on any thread — including one attached to the
/// CPython interpreter (an ordinary decref or a cyclic-GC finalization). A
/// blocking native join keeps that thread from reaching a bytecode safe point,
/// which stalls interpreter-wide stop-the-world operations under a
/// free-threaded build. Dropping through this handle instead calls
/// [`Runtime::shutdown_background`], which returns immediately and lets the
/// worker threads wind down detached. All work submitted through
/// [`Runtime::block_on`] has already completed by the time the last handle
/// drops, so nothing is left in flight for the background shutdown to abandon.
///
/// [`RuntimeHandle`] dereferences to the wrapped [`Runtime`], so callers use it
/// exactly as they would the runtime itself.
pub struct RuntimeHandle {
    runtime: ManuallyDrop<Runtime>,
}

impl RuntimeHandle {
    /// Wrap a [`Runtime`] so that dropping it shuts down in the background.
    pub fn new(runtime: Runtime) -> Self {
        Self {
            runtime: ManuallyDrop::new(runtime),
        }
    }
}

impl Deref for RuntimeHandle {
    type Target = Runtime;

    fn deref(&self) -> &Runtime {
        &self.runtime
    }
}

impl Drop for RuntimeHandle {
    fn drop(&mut self) {
        // SAFETY: `runtime` is initialized in `new` and taken exactly once,
        // here in `drop`; it is never accessed again afterward.
        let runtime = unsafe { ManuallyDrop::take(&mut self.runtime) };
        runtime.shutdown_background();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn test_runtime() -> Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn deref_exposes_runtime() {
        let handle = RuntimeHandle::new(test_runtime());
        // `block_on` is a `Runtime` method reached through `Deref`; every call
        // site relies on this coercion instead of touching the runtime directly.
        let answer = handle.block_on(async { 1 + 1 });
        assert_eq!(answer, 2);
    }

    #[test]
    fn drop_runs_cleanly_when_shared() {
        let handle = Arc::new(RuntimeHandle::new(test_runtime()));
        let clone = handle.clone();

        // Dropping references in turn must leave the wrapped runtime taken
        // exactly once by the final drop (no double-take, no panic, no leak).
        drop(handle);
        drop(clone);
    }
}
