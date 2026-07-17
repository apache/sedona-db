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

//! Utilities for streaming record batches across FFI boundaries.
//!
//! This module provides [`StreamingRecordBatchReader`] for exporting DataFusion streams
//! over FFI, and [`ffi_stream_to_sendable`] for importing FFI streams back into DataFusion.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion_common::{exec_err, Result};
use datafusion_physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use tokio::runtime::Runtime;

/// A cancellation check callback for FFI operations.
///
/// Returns `true` if the operation should be cancelled, `false` to continue.
/// This allows Python, R, ADBC, and other runtimes to integrate their own
/// cancellation mechanisms.
pub type CancelChecker = Box<dyn Fn() -> bool + Send + Sync>;

/// Create a cancellation error for FFI stream operations.
///
/// This is the single source of truth for cancellation errors, ensuring
/// consistent error format across all cancellation sites.
fn cancellation_arrow_error() -> ArrowError {
    ArrowError::ExternalError(Box::new(std::io::Error::new(
        std::io::ErrorKind::Interrupted,
        "Operation cancelled",
    )))
}

/// Result type for batch fetching from the worker thread.
type BatchResult = Option<std::result::Result<RecordBatch, ArrowError>>;

/// Worker thread state for streaming record batch reads.
///
/// Uses eager prefetch: the worker fetches batches into a bounded buffer as fast
/// as possible, providing one-batch pipelining. This allows producer fetch and
/// consumer processing to overlap, roughly halving per-batch wall time compared
/// to a zero-capacity rendezvous protocol.
struct StreamWorker {
    /// Channel to receive prefetched batch results.
    /// The worker eagerly pushes batches here; bounded capacity provides backpressure.
    batch_rx: std::sync::mpsc::Receiver<BatchResult>,
    /// Cancellation flag shared with worker thread.
    /// When set to true, the worker will abort the current fetch and exit.
    cancel_flag: Arc<AtomicBool>,
    /// Worker thread handle for cleanup.
    _handle: std::thread::JoinHandle<()>,
}

/// A RecordBatchReader that lazily polls a SendableRecordBatchStream.
///
/// This allows exporting a DataFusion stream over FFI without collecting all
/// batches upfront. Used when exporting execution plans across FFI boundaries.
///
/// Internally uses a dedicated worker thread to avoid blocking issues when
/// called from within a tokio runtime context.
///
/// # Runtime Requirements
///
/// For responsive mid-batch cancellation, a **multi-thread runtime** is recommended.
/// On current_thread runtimes, `Handle::block_on` does not drive the time driver,
/// so the cancellation timer cannot fire while waiting for `stream.next()`. In this
/// case, cancellation only takes effect between batch fetches (via reader-side
/// `recv_timeout` and the cancel flag check before each fetch).
///
/// If your use case requires responsive cancellation during long-running batch
/// fetches, ensure the runtime passed to [`StreamingRecordBatchReader::new`] is
/// created with [`tokio::runtime::Builder::new_multi_thread`].
pub struct StreamingRecordBatchReader {
    schema: SchemaRef,
    /// Stream and runtime, wrapped in Option so they can be moved to the worker.
    /// We hold Arc<Runtime> instead of just Handle to keep the runtime alive
    /// as long as this reader exists.
    stream_and_runtime: Option<(SendableRecordBatchStream, Arc<Runtime>)>,
    /// Worker thread state, lazily initialized on first fetch.
    worker: Option<StreamWorker>,
    cancel_checker: Option<CancelChecker>,
    cancelled: bool,
    skip_empty_batches: bool,
    periodic_check_interval: Option<std::time::Duration>,
}

impl StreamingRecordBatchReader {
    /// Create a new StreamingRecordBatchReader from a SendableRecordBatchStream.
    ///
    /// Takes an `Arc<Runtime>` to ensure the runtime stays alive for the lifetime
    /// of the reader. This prevents "Worker thread terminated" errors when the
    /// runtime would otherwise be dropped while the stream is still being read.
    pub fn new(stream: SendableRecordBatchStream, runtime: Arc<Runtime>) -> Self {
        Self {
            schema: stream.schema(),
            stream_and_runtime: Some((stream, runtime)),
            worker: None,
            cancel_checker: None,
            cancelled: false,
            skip_empty_batches: false,
            periodic_check_interval: None,
        }
    }

    /// Create a new StreamingRecordBatchReader with a cancellation checker.
    ///
    /// The cancellation checker is called periodically at `check_interval` while
    /// waiting for batches to be fetched. If it returns `true`, iteration stops
    /// with a cancellation error on the next call and `None` on subsequent calls.
    ///
    /// The `check_interval` controls how frequently the checker is called. This
    /// is useful for Python where we need to periodically check for signals
    /// (Ctrl+C) during long-running operations.
    ///
    /// Takes an `Arc<Runtime>` to ensure the runtime stays alive for the lifetime
    /// of the reader.
    pub fn with_cancel_checker(
        stream: SendableRecordBatchStream,
        runtime: Arc<Runtime>,
        cancel_checker: CancelChecker,
        check_interval: Duration,
    ) -> Self {
        Self {
            schema: stream.schema(),
            stream_and_runtime: Some((stream, runtime)),
            worker: None,
            cancel_checker: Some(cancel_checker),
            cancelled: false,
            skip_empty_batches: false,
            periodic_check_interval: Some(check_interval),
        }
    }

    /// Set whether to skip empty batches (batches with 0 rows).
    ///
    /// When enabled, the iterator will automatically skip over any batches
    /// that have no rows and continue to the next batch.
    pub fn with_skip_empty_batches(mut self, skip: bool) -> Self {
        self.skip_empty_batches = skip;
        self
    }

    /// Ensure the worker thread is running, spawning it if necessary.
    fn ensure_worker(&mut self) {
        if self.worker.is_some() {
            return;
        }

        // Take ownership of stream and runtime
        let Some((stream, runtime)) = self.stream_and_runtime.take() else {
            return;
        };

        // Create bounded channel for prefetched batches.
        // Capacity 2 allows one-batch pipelining: while consumer processes batch N,
        // worker can fetch batch N+1 and have it ready. This roughly halves per-batch
        // wall time (max(T_produce, T_consume) instead of T_produce + T_consume).
        let (batch_tx, batch_rx) = std::sync::mpsc::sync_channel::<BatchResult>(2);

        // Cancellation flag - when set, worker will abort current fetch
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let worker_cancel_flag = cancel_flag.clone();

        // Spawn the worker thread - eagerly fetches batches into the buffer
        let handle = std::thread::spawn(move || {
            let mut stream = stream;

            loop {
                // Check if cancelled before starting fetch
                if worker_cancel_flag.load(Ordering::Relaxed) {
                    let _ = batch_tx.send(Some(Err(cancellation_arrow_error())));
                    break;
                }

                let result = runtime.handle().block_on(async {
                    // Poll cancellation periodically while waiting for the next batch.
                    // This allows us to abort the fetch if cancellation is requested.
                    let cancel_check = async {
                        loop {
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            if worker_cancel_flag.load(Ordering::Relaxed) {
                                return;
                            }
                        }
                    };

                    tokio::select! {
                        biased;
                        // Check cancellation first
                        _ = cancel_check => {
                            Some(Err(ArrowError::ExternalError(Box::new(std::io::Error::new(
                                std::io::ErrorKind::Interrupted,
                                "Operation cancelled",
                            )))))
                        }
                        // Then try to get the next batch
                        batch = stream.next() => {
                            match batch {
                                Some(Ok(batch)) => Some(Ok(batch)),
                                Some(Err(e)) => Some(Err(ArrowError::ExternalError(Box::new(e)))),
                                None => None,
                            }
                        }
                    }
                });

                let is_end = result.is_none();
                let is_cancelled = worker_cancel_flag.load(Ordering::Relaxed);

                // Send eagerly - blocks if buffer is full (backpressure)
                // If send fails, the reader was dropped
                if batch_tx.send(result).is_err() || is_end || is_cancelled {
                    break;
                }
            }
        });

        self.worker = Some(StreamWorker {
            batch_rx,
            cancel_flag,
            _handle: handle,
        });
    }

    fn fetch_next_batch(&mut self) -> Option<std::result::Result<RecordBatch, ArrowError>> {
        self.ensure_worker();

        let Some(worker) = &self.worker else {
            return Some(Err(ArrowError::InvalidArgumentError(
                "Worker not initialized".to_string(),
            )));
        };

        // Receive from the prefetch buffer, with optional periodic cancellation checking
        match &self.periodic_check_interval {
            Some(interval) => {
                let interval = *interval;
                loop {
                    match worker.batch_rx.recv_timeout(interval) {
                        Ok(result) => return result,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            // Check for cancellation
                            if let Some(ref checker) = self.cancel_checker {
                                if checker() {
                                    self.cancelled = true;
                                    // Signal the worker to abort the current fetch
                                    // This will cause the worker's select! to see the
                                    // cancellation and drop the stream future.
                                    worker.cancel_flag.store(true, Ordering::Relaxed);
                                    // Wait briefly for the worker to send its response
                                    // then return the cancellation error
                                    let _ = worker
                                        .batch_rx
                                        .recv_timeout(std::time::Duration::from_millis(200));
                                    return Some(Err(cancellation_arrow_error()));
                                }
                            }
                            // Continue waiting
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                            return Some(Err(ArrowError::InvalidArgumentError(
                                "Worker thread terminated".to_string(),
                            )));
                        }
                    }
                }
            }
            None => {
                // Simple blocking receive from prefetch buffer
                match worker.batch_rx.recv() {
                    Ok(result) => result,
                    Err(_) => Some(Err(ArrowError::InvalidArgumentError(
                        "Worker thread terminated".to_string(),
                    ))),
                }
            }
        }
    }
}

impl Iterator for StreamingRecordBatchReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        // If already cancelled, return None to stop iteration
        if self.cancelled {
            return None;
        }

        loop {
            match self.fetch_next_batch() {
                Some(Ok(batch)) => {
                    if self.skip_empty_batches && batch.num_rows() == 0 {
                        continue;
                    }
                    return Some(Ok(batch));
                }
                Some(Err(e)) => {
                    return Some(Err(e));
                }
                None => return None,
            }
        }
    }
}

impl RecordBatchReader for StreamingRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Drop for StreamingRecordBatchReader {
    fn drop(&mut self) {
        if let Some(worker) = &self.worker {
            // Ensure the worker wakes up and exits promptly if it is blocked.
            worker.cancel_flag.store(true, Ordering::Relaxed);
        }
    }
}

/// Convert an FFI ArrowArrayStream into a SendableRecordBatchStream with cancellation support.
///
/// Uses a dedicated OS thread to read from the synchronous FFI stream, sending
/// batches through an async channel. This avoids blocking tokio worker threads,
/// which would otherwise cause deadlocks if the producer side also needs the
/// same runtime via `block_on()` (especially fatal with current-thread
/// runtimes, but problematic with any shared runtime).
///
/// The cancellation checker is called periodically (at most once per `check_interval`)
/// before yielding batches. If it returns `true`, the stream yields a cancellation error.
/// If `check_interval` is `None`, the checker is called before every batch.
///
/// # Safety
///
/// The caller must ensure that the FFI stream pointer is valid and properly
/// initialized.
pub unsafe fn ffi_stream_to_sendable(
    ffi_stream: &mut FFI_ArrowArrayStream,
    cancel_checker: Option<CancelChecker>,
    check_interval: Option<Duration>,
) -> Result<SendableRecordBatchStream> {
    let reader = arrow_array::ffi_stream::ArrowArrayStreamReader::from_raw(ffi_stream)?;
    let schema = reader.schema();

    // Use an async channel with capacity 2 for one-batch pipelining.
    // The dedicated thread reads blocking batches from FFI, the async stream
    // receives them without blocking tokio workers.
    let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<RecordBatch, ArrowError>>(2);

    // Dedicated thread for blocking reads - keeps blocking I/O off tokio workers
    std::thread::spawn(move || {
        for batch in reader {
            // blocking_send blocks this thread if buffer is full (backpressure)
            if tx.blocking_send(batch).is_err() {
                break; // Receiver dropped, stop reading
            }
        }
    });

    // Track last check time for periodic cancellation checking
    let mut last_check = Instant::now();

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx).map(move |result| {
        // Check for cancellation periodically (or every batch if no interval)
        if let Some(ref checker) = cancel_checker {
            let should_check = match check_interval {
                Some(interval) => {
                    let now = Instant::now();
                    if now.duration_since(last_check) >= interval {
                        last_check = now;
                        true
                    } else {
                        false
                    }
                }
                None => true, // No interval means check every batch
            };

            if should_check && checker() {
                return exec_err!("Operation cancelled");
            }
        }
        result.map_err(|e| datafusion_common::DataFusionError::ArrowError(Box::new(e), None))
    });

    Ok(Box::pin(
        datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// Create a test runtime wrapped in Arc.
    fn test_runtime() -> Arc<Runtime> {
        Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        )
    }

    /// Create a slow stream that yields batches with a configurable delay.
    fn create_slow_stream(
        num_batches: usize,
        delay_ms: u64,
    ) -> (SchemaRef, SendableRecordBatchStream) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let schema_clone = schema.clone();

        let stream = futures::stream::iter(0..num_batches).then(move |i| {
            let schema = schema_clone.clone();
            async move {
                if delay_ms > 0 {
                    // Use std::thread::sleep because tokio::time::sleep requires
                    // the runtime to poll, which can deadlock with block_on
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
                let array = Int32Array::from(vec![i as i32]);
                Ok(RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap())
            }
        });

        (
            schema.clone(),
            Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream),
            ),
        )
    }

    #[test]
    fn test_streaming_reader_basic() {
        let runtime = test_runtime();
        let (_schema, stream) = create_slow_stream(5, 10);

        let reader = StreamingRecordBatchReader::new(stream, runtime);
        let batches: Vec<_> = reader.collect();

        assert_eq!(batches.len(), 5);
        for (i, batch) in batches.iter().enumerate() {
            let batch = batch.as_ref().unwrap();
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            assert_eq!(array.value(0), i as i32);
        }
    }

    #[test]
    fn test_streaming_reader_cancel() {
        let runtime = test_runtime();
        // 10 batches, 50ms each = 500ms total
        let (_schema, stream) = create_slow_stream(10, 50);

        // Cancel after 175ms - should allow ~3 batches (at 50ms each) to complete
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();
        let cancel_checker: CancelChecker =
            Box::new(move || cancelled_clone.load(Ordering::SeqCst));

        let reader = StreamingRecordBatchReader::with_cancel_checker(
            stream,
            runtime,
            cancel_checker,
            Duration::from_millis(25),
        );

        // Set cancel flag after 175ms from a separate thread
        let cancelled_setter = cancelled.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(175));
            cancelled_setter.store(true, Ordering::SeqCst);
        });

        let batches: Vec<_> = reader.collect();

        // Should have ~3 successful batches + 1 cancellation error
        // (timing can vary slightly, so we check for at least 2 and at most 5)
        assert!(
            batches.len() >= 2 && batches.len() <= 5,
            "expected 2-5 batches, got {}",
            batches.len()
        );

        // All but the last should be Ok
        for (i, batch) in batches.iter().take(batches.len() - 1).enumerate() {
            assert!(batch.is_ok(), "batch {} should be Ok", i);
        }

        // Last one should be a cancellation error
        let last = batches.last().unwrap();
        assert!(last.is_err());
        let err = last.as_ref().unwrap_err();
        assert!(
            err.to_string().contains("cancelled"),
            "error should mention cancellation: {}",
            err
        );
    }

    #[test]
    fn test_ffi_stream_to_sendable_basic() {
        use arrow_array::ffi_stream::FFI_ArrowArrayStream;

        let runtime = test_runtime();
        let (_schema, stream) = create_slow_stream(5, 10);

        // Export to FFI stream
        let reader = StreamingRecordBatchReader::new(stream, runtime.clone());
        let mut ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));

        // Import back
        let imported = unsafe { ffi_stream_to_sendable(&mut ffi_stream, None, None).unwrap() };

        // Collect results
        let batches: Vec<_> = runtime.block_on(imported.collect::<Vec<_>>());

        assert_eq!(batches.len(), 5);
        for (i, batch) in batches.iter().enumerate() {
            let batch = batch.as_ref().unwrap();
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            assert_eq!(array.value(0), i as i32);
        }
    }

    #[test]
    fn test_ffi_stream_to_sendable_cancel() {
        use arrow_array::ffi_stream::FFI_ArrowArrayStream;

        let runtime = test_runtime();
        let (_schema, stream) = create_slow_stream(10, 50);

        // Export to FFI stream (no cancellation on export side)
        let reader = StreamingRecordBatchReader::new(stream, runtime.clone());
        let mut ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));

        // Cancel after reading 3 batches on import side
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();
        let cancel_checker: CancelChecker =
            Box::new(move || cancelled_clone.load(Ordering::SeqCst));

        let imported =
            unsafe { ffi_stream_to_sendable(&mut ffi_stream, Some(cancel_checker), None).unwrap() };

        // Collect with cancellation after 3 batches, stop on first error
        let batches = runtime.block_on(async {
            let mut batches = Vec::new();
            let mut stream = imported;
            while let Some(result) = stream.next().await {
                let is_err = result.is_err();
                batches.push(result);
                if batches.len() == 3 {
                    cancelled.store(true, Ordering::SeqCst);
                }
                if is_err {
                    break; // Stop on first error
                }
            }
            batches
        });

        // Should have 3 successful batches + 1 cancellation error
        assert_eq!(batches.len(), 4);

        // First 3 should be Ok
        for (i, batch) in batches.iter().enumerate().take(3) {
            assert!(batch.is_ok(), "batch {} should be Ok", i);
        }

        // Last one should be a cancellation error
        let last = batches.last().unwrap();
        assert!(last.is_err());
        let err = last.as_ref().unwrap_err();
        assert!(
            err.to_string().contains("cancelled"),
            "error should mention cancellation: {}",
            err
        );
    }

    /// Create a stream that yields some empty batches.
    fn create_stream_with_empty_batches(
        batch_row_counts: Vec<usize>,
    ) -> (SchemaRef, SendableRecordBatchStream) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let schema_clone = schema.clone();

        let stream = futures::stream::iter(batch_row_counts.into_iter().enumerate()).map(
            move |(i, row_count)| {
                let schema = schema_clone.clone();
                let array: Int32Array = (0..row_count).map(|_| i as i32).collect();
                Ok(RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap())
            },
        );

        (
            schema.clone(),
            Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream),
            ),
        )
    }

    #[test]
    fn test_streaming_reader_skip_empty_batches() {
        let runtime = test_runtime();
        // Create stream with: 2 rows, 0 rows, 3 rows, 0 rows, 0 rows, 1 row
        let (_schema, stream) = create_stream_with_empty_batches(vec![2, 0, 3, 0, 0, 1]);

        let reader = StreamingRecordBatchReader::new(stream, runtime).with_skip_empty_batches(true);
        let batches: Vec<_> = reader.collect();

        // Should only get 3 batches (the non-empty ones)
        assert_eq!(batches.len(), 3);

        // Check row counts: 2, 3, 1
        assert_eq!(batches[0].as_ref().unwrap().num_rows(), 2);
        assert_eq!(batches[1].as_ref().unwrap().num_rows(), 3);
        assert_eq!(batches[2].as_ref().unwrap().num_rows(), 1);
    }

    #[test]
    fn test_streaming_reader_no_skip_empty_batches() {
        let runtime = test_runtime();
        // Create stream with: 2 rows, 0 rows, 3 rows
        let (_schema, stream) = create_stream_with_empty_batches(vec![2, 0, 3]);

        // Default: skip_empty_batches is false
        let reader = StreamingRecordBatchReader::new(stream, runtime);
        let batches: Vec<_> = reader.collect();

        // Should get all 3 batches including the empty one
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].as_ref().unwrap().num_rows(), 2);
        assert_eq!(batches[1].as_ref().unwrap().num_rows(), 0);
        assert_eq!(batches[2].as_ref().unwrap().num_rows(), 3);
    }

    #[test]
    fn test_streaming_reader_periodic_check_interval() {
        let runtime = test_runtime();
        // Create a slow stream where each batch takes 200ms
        let (_schema, stream) = create_slow_stream(5, 200);

        // Cancel flag - will be set after 150ms
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();
        let cancel_checker: CancelChecker =
            Box::new(move || cancelled_clone.load(Ordering::SeqCst));

        let reader = StreamingRecordBatchReader::with_cancel_checker(
            stream,
            runtime,
            cancel_checker,
            Duration::from_millis(50),
        );

        // Spawn a task to set cancelled after 150ms
        let cancelled_setter = cancelled.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            cancelled_setter.store(true, Ordering::SeqCst);
        });

        let batches: Vec<_> = reader.collect();

        // The first batch takes 200ms but we cancel at 150ms during the fetch
        // With periodic_check_interval of 50ms, the check happens at 50ms, 100ms, 150ms
        // At 150ms the cancel check should trigger
        // So we should get 0 successful batches + 1 cancellation error
        assert!(
            batches.len() <= 2,
            "expected at most 1 batch + 1 error, got {}",
            batches.len()
        );

        // At least one should be an error
        let has_cancel_error = batches
            .iter()
            .any(|b| b.is_err() && b.as_ref().unwrap_err().to_string().contains("cancelled"));
        assert!(
            has_cancel_error,
            "should have a cancellation error in batches: {:?}",
            batches
        );
    }
}
