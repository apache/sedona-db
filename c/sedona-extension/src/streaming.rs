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

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion_common::{exec_err, Result};
use datafusion_physical_plan::SendableRecordBatchStream;
use futures::StreamExt;

/// A cancellation check callback for FFI operations.
///
/// Returns `true` if the operation should be cancelled, `false` to continue.
/// This allows Python, R, ADBC, and other runtimes to integrate their own
/// cancellation mechanisms.
pub type CancelChecker = Box<dyn Fn() -> bool + Send + Sync>;

/// Result type for batch fetching from the worker thread.
type BatchResult = Option<std::result::Result<RecordBatch, ArrowError>>;

/// Worker thread state for streaming record batch reads.
struct StreamWorker {
    /// Channel to request the next batch (send () to request).
    request_tx: std::sync::mpsc::SyncSender<()>,
    /// Channel to receive batch results.
    response_rx: std::sync::mpsc::Receiver<BatchResult>,
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
pub struct StreamingRecordBatchReader {
    schema: SchemaRef,
    /// Stream and runtime, wrapped in Option so they can be moved to the worker.
    stream_and_runtime: Option<(SendableRecordBatchStream, tokio::runtime::Handle)>,
    /// Worker thread state, lazily initialized on first fetch.
    worker: Option<StreamWorker>,
    cancel_checker: Option<CancelChecker>,
    cancelled: bool,
    skip_empty_batches: bool,
    periodic_check_interval: Option<std::time::Duration>,
}

impl StreamingRecordBatchReader {
    /// Create a new StreamingRecordBatchReader from a SendableRecordBatchStream.
    pub fn new(stream: SendableRecordBatchStream, runtime: tokio::runtime::Handle) -> Self {
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
    /// The cancellation checker is called before each batch is fetched. If it
    /// returns `true`, iteration stops with a cancellation error on the next
    /// call and `None` on subsequent calls.
    pub fn with_cancel_checker(
        stream: SendableRecordBatchStream,
        runtime: tokio::runtime::Handle,
        cancel_checker: CancelChecker,
    ) -> Self {
        Self {
            schema: stream.schema(),
            stream_and_runtime: Some((stream, runtime)),
            worker: None,
            cancel_checker: Some(cancel_checker),
            cancelled: false,
            skip_empty_batches: false,
            periodic_check_interval: None,
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

    /// Set a periodic interval for checking cancellation during batch fetches.
    ///
    /// When set, the cancel checker will be called periodically at this interval
    /// even while waiting for a single batch to be fetched. This is useful for
    /// Python where we need to periodically check for signals (Ctrl+C) during
    /// long-running operations.
    ///
    /// Without this, the cancel checker is only called between batch fetches.
    pub fn with_periodic_check_interval(mut self, interval: std::time::Duration) -> Self {
        self.periodic_check_interval = Some(interval);
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

        // Create channels for communication
        // Use bounded channel with size 0 (rendezvous) for backpressure
        let (request_tx, request_rx) = std::sync::mpsc::sync_channel::<()>(0);
        // Use bounded channel with size 1 for response to prevent deadlock if reader
        // stops early due to periodic cancellation before receiving the response
        let (response_tx, response_rx) = std::sync::mpsc::sync_channel::<BatchResult>(1);

        // Spawn the worker thread
        let handle = std::thread::spawn(move || {
            let mut stream = stream;
            // Process requests until the channel is closed
            while request_rx.recv().is_ok() {
                let result = runtime.block_on(async {
                    match stream.next().await {
                        Some(Ok(batch)) => Some(Ok(batch)),
                        Some(Err(e)) => Some(Err(ArrowError::ExternalError(Box::new(e)))),
                        None => None,
                    }
                });

                // Send the result back; if send fails, the reader was dropped
                if response_tx.send(result).is_err() {
                    break;
                }
            }
        });

        self.worker = Some(StreamWorker {
            request_tx,
            response_rx,
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

        // Request the next batch
        if worker.request_tx.send(()).is_err() {
            return Some(Err(ArrowError::InvalidArgumentError(
                "Worker thread terminated".to_string(),
            )));
        }

        // Wait for the response, with optional periodic cancellation checking
        match &self.periodic_check_interval {
            Some(interval) => {
                let interval = *interval;
                loop {
                    match worker.response_rx.recv_timeout(interval) {
                        Ok(result) => return result,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            // Check for cancellation
                            if let Some(ref checker) = self.cancel_checker {
                                if checker() {
                                    return Some(Err(ArrowError::ExternalError(Box::new(
                                        std::io::Error::new(
                                            std::io::ErrorKind::Interrupted,
                                            "Operation cancelled",
                                        ),
                                    ))));
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
                // Simple blocking receive
                match worker.response_rx.recv() {
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

        // Check for cancellation before fetching the next batch
        // (periodic checking during fetch is handled by fetch_next_batch if configured)
        if let Some(ref checker) = self.cancel_checker {
            if checker() {
                self.cancelled = true;
                return Some(Err(ArrowError::ExternalError(Box::new(
                    std::io::Error::new(std::io::ErrorKind::Interrupted, "Operation cancelled"),
                ))));
            }
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
                    // Check if this was a cancellation error from periodic checking
                    if let ArrowError::ExternalError(box_err) = &e {
                        if let Some(io_err) = box_err.downcast_ref::<std::io::Error>() {
                            if io_err.kind() == std::io::ErrorKind::Interrupted {
                                self.cancelled = true;
                            }
                        }
                    }
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

/// Convert an FFI ArrowArrayStream into a SendableRecordBatchStream.
///
/// This is the inverse of StreamingRecordBatchReader - it takes an FFI stream
/// and converts it back to a DataFusion stream. Used when importing execution
/// plans from across FFI boundaries.
///
/// # Safety
///
/// The caller must ensure that the FFI stream pointer is valid and properly
/// initialized.
pub unsafe fn ffi_stream_to_sendable(
    ffi_stream: &mut FFI_ArrowArrayStream,
) -> Result<SendableRecordBatchStream> {
    ffi_stream_to_sendable_with_cancel(ffi_stream, None)
}

/// Convert an FFI ArrowArrayStream into a SendableRecordBatchStream with cancellation support.
///
/// The cancellation checker is called before each batch is read. If it returns
/// `true`, the stream yields a cancellation error.
///
/// # Safety
///
/// The caller must ensure that the FFI stream pointer is valid and properly
/// initialized.
pub unsafe fn ffi_stream_to_sendable_with_cancel(
    ffi_stream: &mut FFI_ArrowArrayStream,
    cancel_checker: Option<CancelChecker>,
) -> Result<SendableRecordBatchStream> {
    let reader = arrow_array::ffi_stream::ArrowArrayStreamReader::from_raw(ffi_stream)?;

    let schema = reader.schema();
    let stream = futures::stream::iter(reader).map(move |result| {
        // Check for cancellation before yielding each batch
        if let Some(ref checker) = cancel_checker {
            if checker() {
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

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

    #[tokio::test]
    async fn test_streaming_reader_basic() {
        let runtime = tokio::runtime::Handle::current();
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

    #[tokio::test]
    async fn test_streaming_reader_cancel() {
        let runtime = tokio::runtime::Handle::current();
        let (_schema, stream) = create_slow_stream(10, 50);

        // Cancel after reading 3 batches
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let cancel_checker: CancelChecker = Box::new(move || {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            count >= 3
        });

        let reader =
            StreamingRecordBatchReader::with_cancel_checker(stream, runtime, cancel_checker);
        let batches: Vec<_> = reader.collect();

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

    #[tokio::test]
    async fn test_ffi_stream_to_sendable_basic() {
        use arrow_array::ffi_stream::FFI_ArrowArrayStream;

        let runtime = tokio::runtime::Handle::current();
        let (_schema, stream) = create_slow_stream(5, 10);

        // Export to FFI stream
        let reader = StreamingRecordBatchReader::new(stream, runtime.clone());
        let mut ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));

        // Import back
        let imported = unsafe { ffi_stream_to_sendable(&mut ffi_stream).unwrap() };

        // Collect results
        let batches: Vec<_> = imported.collect::<Vec<_>>().await;

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

    #[tokio::test]
    async fn test_ffi_stream_to_sendable_cancel() {
        use arrow_array::ffi_stream::FFI_ArrowArrayStream;

        let runtime = tokio::runtime::Handle::current();
        let (_schema, stream) = create_slow_stream(10, 50);

        // Export to FFI stream (no cancellation on export side)
        let reader = StreamingRecordBatchReader::new(stream, runtime.clone());
        let mut ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));

        // Cancel after reading 3 batches on import side
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();
        let cancel_checker: CancelChecker =
            Box::new(move || cancelled_clone.load(Ordering::SeqCst));

        let imported = unsafe {
            ffi_stream_to_sendable_with_cancel(&mut ffi_stream, Some(cancel_checker)).unwrap()
        };

        // Collect with cancellation after 3 batches, stop on first error
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

    #[tokio::test]
    async fn test_streaming_reader_skip_empty_batches() {
        let runtime = tokio::runtime::Handle::current();
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

    #[tokio::test]
    async fn test_streaming_reader_no_skip_empty_batches() {
        let runtime = tokio::runtime::Handle::current();
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

    #[tokio::test]
    async fn test_streaming_reader_periodic_check_interval() {
        let runtime = tokio::runtime::Handle::current();
        // Create a slow stream where each batch takes 200ms
        let (_schema, stream) = create_slow_stream(5, 200);

        // Cancel flag - will be set after 150ms
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();
        let cancel_checker: CancelChecker =
            Box::new(move || cancelled_clone.load(Ordering::SeqCst));

        let reader =
            StreamingRecordBatchReader::with_cancel_checker(stream, runtime, cancel_checker)
                .with_periodic_check_interval(Duration::from_millis(50));

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
