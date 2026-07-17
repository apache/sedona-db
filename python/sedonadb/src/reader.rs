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
use std::sync::Arc;
use std::time::Duration;

use datafusion::execution::SendableRecordBatchStream;
use pyo3::Python;
use sedona_extension::streaming::StreamingRecordBatchReader;
use tokio::runtime::Runtime;

/// Interval for checking Python signals during batch fetches.
const INTERVAL_CHECK_SIGNALS: Duration = Duration::from_millis(2_000);

/// Create a Python-aware [StreamingRecordBatchReader] that:
/// - Skips empty batches
/// - Periodically checks for Python cancellation signals (Ctrl+C)
///
/// The reader will check Python signals every 2 seconds during batch fetches,
/// allowing users to interrupt long-running queries.
///
/// Takes an `Arc<Runtime>` to ensure the runtime stays alive for the lifetime
/// of the reader, preventing "Worker thread terminated" errors.
pub fn new_py_streaming_reader(
    stream: SendableRecordBatchStream,
    runtime: Arc<Runtime>,
) -> StreamingRecordBatchReader {
    // Create a cancel checker that checks Python signals
    let cancel_checker: Box<dyn Fn() -> bool + Send + Sync> = Box::new(|| {
        Python::attach(|py| {
            // Run `pass` to process any pending signals, then check for errors
            if py.run(cr"pass", None, None).is_err() {
                return true;
            }
            py.check_signals().is_err()
        })
    });

    StreamingRecordBatchReader::with_cancel_checker(
        stream,
        runtime,
        cancel_checker,
        INTERVAL_CHECK_SIGNALS,
    )
    .with_skip_empty_batches(true)
}
