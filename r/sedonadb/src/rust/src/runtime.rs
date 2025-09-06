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
use std::{
    future::Future,
    sync::{OnceLock, RwLock},
    time::Duration,
};

use savvy::{savvy, savvy_err, savvy_init};
use savvy_ffi::R_NilValue;
use tokio::{runtime::Runtime, time::sleep};

use crate::error::RSedonaError;

pub fn wait_for_future_captured_r<F>(runtime: &Runtime, fut: F) -> Result<F::Output, RSedonaError>
where
    F: Future + Send + 'static,
    F::Output: Send,
{
    const INTERVAL_CHECK_SIGNALS: Duration = Duration::from_millis(1_000);
    let handle = runtime.spawn(async move {
        tokio::pin!(fut);
        loop {
            tokio::select! {
                res = &mut fut => break Ok(res),
                _ = sleep(INTERVAL_CHECK_SIGNALS) => {
                    let handle = R.get().unwrap().rt().spawn(async move {
                        R.get().unwrap().check_signals()
                    });
                    handle.await??;
                }
            }
        }
    });

    R.get().unwrap().rt().block_on(handle)?
}

struct RRuntime {
    rt: Runtime,
    check_interrupts_call: RwLock<Option<savvy::Sexp>>,
    pkg_env: RwLock<Option<savvy::Sexp>>,
}

unsafe impl Send for RRuntime {}

unsafe impl Sync for RRuntime {}

impl RRuntime {
    pub fn try_new() -> Result<Self, RSedonaError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Self {
            rt,
            check_interrupts_call: RwLock::new(None),
            pkg_env: RwLock::new(None),
        })
    }

    pub fn rt(&self) -> &Runtime {
        &self.rt
    }

    pub fn check_signals(&self) -> Result<(), RSedonaError> {
        let (maybe_call, maybe_env) = match (
            self.check_interrupts_call.try_read(),
            self.pkg_env.try_read(),
        ) {
            (Ok(call), Ok(env)) => (call, env),
            _ => {
                return Err(RSedonaError::Internal(
                    "Check interrupts call could not be read".to_string(),
                ));
            }
        };

        let (call, env) = match (maybe_call.as_ref(), maybe_env.as_ref()) {
            (Some(call), Some(env)) => (call, env),
            _ => {
                return Err(RSedonaError::Internal(
                    "Check interrupts not set".to_string(),
                ));
            }
        };

        unsafe {
            // We could save this error for future things that can actually fail (like evaluating
            // R UDFs)
            let is_interrupted_sexp = savvy::unwind_protect(|| savvy_ffi::Rf_eval(call.0, env.0))
                .expect(
                "Check interrupts function must not error (i.e., must be wrapped in tryCatch())",
            );

            let is_interrupted: bool = savvy::Sexp(is_interrupted_sexp)
                .try_into()
                .expect("Check interrupts function must return a bool");

            if is_interrupted {
                Err(RSedonaError::Interrupted)
            } else {
                Ok(())
            }
        }
    }
}

static R: OnceLock<RRuntime> = OnceLock::new();

#[savvy_init]
fn init_r_runtime(_dll_info: *mut savvy_ffi::DllInfo) -> savvy::Result<()> {
    R.get_or_init(|| RRuntime::try_new().unwrap());
    Ok(())
}

#[savvy]
fn init_r_runtime_interrupts(
    interrupts_call: savvy::Sexp,
    pkg_env: savvy::Sexp,
) -> savvy::Result<()> {
    if let Some(r) = R.get() {
        unsafe {
            savvy::unwind_protect(|| {
                savvy_ffi::R_PreserveObject(interrupts_call.0);
                savvy_ffi::R_PreserveObject(pkg_env.0);
                R_NilValue
            })?;
        }

        r.check_interrupts_call
            .try_write()?
            .replace(interrupts_call);
        r.pkg_env.try_write()?.replace(pkg_env);

        Ok(())
    } else {
        Err(savvy_err!("R runtime not initialized"))
    }
}
