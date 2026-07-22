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

//! A Tokio runtime owner that drains gracefully off the interpreter thread.

use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

use tokio::runtime::Runtime;

/// Process-global set of janitor threads, each draining one dropped runtime to
/// completion.
///
/// [`RuntimeHandle::drop`] spawns a janitor and pushes its handle here;
/// [`RuntimeHandle::new`] sweeps out the janitors that have finished. Whatever
/// remains is a runtime still draining, so a trash can that never empties is a
/// visible signal that shutdowns are not completing — the alternative to a
/// silent leak.
fn trash() -> &'static Mutex<Vec<JoinHandle<()>>> {
    static TRASH: OnceLock<Mutex<Vec<JoinHandle<()>>>> = OnceLock::new();
    TRASH.get_or_init(|| Mutex::new(Vec::new()))
}

/// Owns a Tokio [`Runtime`] and drains it on a dedicated OS thread when dropped.
///
/// The default `Runtime::drop` blocks the dropping thread while it joins every
/// worker thread. When a runtime is shared through `Arc<RuntimeHandle>`, the
/// final reference can drop on any thread — including one attached to the
/// CPython interpreter (an ordinary decref or a cyclic-GC finalization). A
/// blocking native join keeps that thread from reaching a bytecode safe point,
/// which stalls interpreter-wide stop-the-world operations under a free-threaded
/// build.
///
/// Dropping through this handle moves the runtime onto a freshly spawned plain
/// OS thread — never a Tokio thread, never attached to the interpreter — and
/// runs the ordinary blocking `Runtime::drop` there. That janitor thread waits
/// for every task and worker to wind down (a graceful shutdown that abandons no
/// work), but because it carries no CPython thread state it is invisible to the
/// stop-the-world and may block harmlessly. The dropping thread itself only
/// spawns the janitor and records its handle, so it returns immediately and can
/// still reach a safe point.
///
/// [`RuntimeHandle`] dereferences to the wrapped [`Runtime`], so callers use it
/// exactly as they would the runtime itself.
pub struct RuntimeHandle {
    runtime: ManuallyDrop<Runtime>,
}

impl RuntimeHandle {
    /// Wrap a [`Runtime`] so that dropping it drains the runtime on a dedicated
    /// OS thread.
    ///
    /// Constructing a handle first sweeps the trash can, dropping the handles of
    /// janitor threads that have already finished draining an earlier runtime.
    pub fn new(runtime: Runtime) -> Self {
        // Recover a poisoned lock rather than panic: janitors only move a
        // runtime out on their own thread, so the trash can's contents stay
        // valid to sweep even if some thread panicked while holding the lock.
        let mut trash = trash()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        trash.retain(|janitor| !janitor.is_finished());
        drop(trash);

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

        // Run the blocking `Runtime::drop` (which joins every worker thread) on
        // a dedicated OS thread. That thread is never attached to the CPython
        // interpreter, so its blocking join cannot stall the free-threaded
        // stop-the-world; the shutdown stays graceful — it waits for tasks to
        // finish rather than abandoning them — while this thread returns at once
        // and can reach a safe point.
        let janitor = std::thread::spawn(move || drop(runtime));

        // Recover a poisoned lock rather than panic inside `drop`.
        let mut trash = trash()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        trash.push(janitor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    fn test_runtime() -> Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    /// Take every janitor currently in the trash can and join it, returning the
    /// number joined.
    ///
    /// Joining waits for each runtime's graceful drop to finish, which turns the
    /// otherwise-detached shutdown into a deterministic synchronization point
    /// for tests — no wall-clock waiting. The trash can is process-global and
    /// tests run in parallel, so this may also join janitors that other tests
    /// enqueued; that is harmless (their runtimes drain and their handles are
    /// removed), and tests therefore assert about their own runtime rather than
    /// an exact global count.
    fn drain_janitors() -> usize {
        let janitors: Vec<JoinHandle<()>> = {
            let mut trash = trash()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::mem::take(&mut *trash)
        };
        let count = janitors.len();
        for janitor in janitors {
            janitor
                .join()
                .expect("janitor thread panicked while draining a runtime");
        }
        count
    }

    #[test]
    fn deref_exposes_runtime() {
        let handle = RuntimeHandle::new(test_runtime());
        // `block_on` is a `Runtime` method reached through `Deref`; every call
        // site relies on this coercion instead of touching the runtime directly.
        let answer = handle.block_on(async { 1 + 1 });
        assert_eq!(answer, 2);
        drain_janitors();
    }

    #[test]
    fn drop_runs_cleanly_when_shared() {
        let handle = Arc::new(RuntimeHandle::new(test_runtime()));
        let clone = handle.clone();

        // Dropping references in turn must leave the wrapped runtime taken
        // exactly once by the final drop (no double-take, no panic, no leak).
        drop(handle);
        drop(clone);

        // The final drop hands the runtime to a janitor thread; join it so the
        // graceful shutdown completes without a panic.
        drain_janitors();
    }

    #[test]
    fn drop_drains_runtime_on_a_janitor_thread() {
        // A guard whose `Drop` fires only when the runtime tears down: it rides
        // inside a queued task, so it is released when the janitor thread drops
        // the runtime. Observing it proves the graceful shutdown actually ran on
        // the janitor thread rather than being skipped.
        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));
        let handle = {
            let runtime = test_runtime();
            let guard = DropFlag(dropped.clone());
            // Nothing ever drives this current-thread runtime, so the task never
            // runs; the guard lives on inside the queued task until the janitor
            // thread drops the runtime (and with it the task).
            runtime.spawn(async move {
                let _guard = guard;
            });
            RuntimeHandle::new(runtime)
        };

        // The runtime is still alive, so the guard has not been released yet.
        assert!(!dropped.load(Ordering::SeqCst));

        drop(handle);

        // Draining joins the janitor, deterministically waiting for the runtime
        // to finish draining; only then is the guard guaranteed released.
        drain_janitors();
        assert!(
            dropped.load(Ordering::SeqCst),
            "janitor thread should have dropped the runtime"
        );
    }

    #[test]
    fn new_sweeps_finished_janitors_without_blocking() {
        // Enqueue a janitor and wait for it to finish draining its runtime.
        drop(RuntimeHandle::new(test_runtime()));
        drain_janitors();

        // Constructing another handle runs the trash-can sweep. It must neither
        // panic nor block on any janitor, and the handle it returns must be a
        // fully usable runtime.
        let handle = RuntimeHandle::new(test_runtime());
        let answer = handle.block_on(async { 40 + 2 });
        assert_eq!(answer, 42);
        drain_janitors();
    }
}
