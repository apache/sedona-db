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
use std::{future::Future, time::Duration};

use pyo3::Python;
use tokio::{runtime::Runtime, time::sleep};

use crate::error::PySedonaError;

// Adapted from datafusion-python:
// https://github.com/apache/datafusion-python/blob/cbe845b1e840c78f7a9fc4d83d184a1e6f35f47c/src/utils.rs#L64
pub fn wait_for_future<F>(py: Python, runtime: &Runtime, fut: F) -> Result<F::Output, PySedonaError>
where
    F: Future + Send,
    F::Output: Send,
{
    const INTERVAL_CHECK_SIGNALS: Duration = Duration::from_millis(1_000);

    py.allow_threads(|| {
        runtime.block_on(async {
            tokio::pin!(fut);
            loop {
                tokio::select! {
                    res = &mut fut => break Ok(res),
                    _ = sleep(INTERVAL_CHECK_SIGNALS) => {
                        Python::with_gil(|py| py.check_signals())?;
                    }
                }
            }
        })
    })
}

// A version of the above except for use from an arbitrary Rust function instead
// of from somewhere that had already acquired the GIL.
pub fn wait_for_future_from_rust<F>(runtime: &Runtime, fut: F) -> Result<F::Output, PySedonaError>
where
    F: Future + Send,
    F::Output: Send,
{
    const INTERVAL_CHECK_SIGNALS: Duration = Duration::from_millis(1_000);

    runtime.block_on(async {
        tokio::pin!(fut);
        loop {
            tokio::select! {
                res = &mut fut => break Ok(res),
                _ = sleep(INTERVAL_CHECK_SIGNALS) => {
                    Python::with_gil(|py| py.check_signals())?;
                }
            }
        }
    })
}
