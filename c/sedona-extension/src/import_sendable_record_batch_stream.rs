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

//! Import an async device stream from FFI into a [`SendableRecordBatchStream`].
//!
//! This module allows Rust code to consume async record batch streams provided by
//! C/FFI producers via the Arrow C Device Data Interface.
//!
//! # Usage
//!
//! 1. Create an [`ImportedAsyncDeviceStream`] which provides an [`FFI_ArrowAsyncDeviceStreamHandler`]
//! 2. Pass the handler pointer to the FFI producer
//! 3. Use the stream as a normal [`SendableRecordBatchStream`]
//!
//! ```ignore
//! use sedona_extension::import_sendable_record_batch_stream::ImportedAsyncDeviceStream;
//!
//! let (stream, handler) = ImportedAsyncDeviceStream::new(16);
//!
//! // Pass handler to FFI producer...
//! // ffi_producer_start(handler.as_ptr());
//!
//! // Consume the stream
//! while let Some(batch) = stream.next().await {
//!     let batch = batch?;
//!     // process batch
//! }
//! ```

use std::ffi::{c_int, c_void, CStr};
use std::pin::Pin;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use arrow_array::ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{RecordBatch, StructArray};
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::RecordBatchStream;
use futures::channel::mpsc;
use futures::task::AtomicWaker;
use futures::Stream;

use crate::extension_ffi::{
    FFI_ArrowAsyncDeviceStreamHandler, FFI_ArrowAsyncProducer, FFI_ArrowAsyncTask,
    FFI_ArrowDeviceArray, ARROW_DEVICE_CPU,
};

/// Messages sent from FFI callbacks to the stream.
enum StreamMessage {
    /// Schema received from producer.
    Schema(SchemaRef),
    /// A record batch task received.
    Task(FFI_ArrowAsyncTask),
    /// End of stream (NULL task received).
    EndOfStream,
    /// Error from producer.
    Error(ArrowError),
}

/// State shared between callbacks and stream for producer management.
/// This is the only state that requires mutex protection.
struct ProducerState {
    /// Producer pointer for requesting more data.
    producer: *mut FFI_ArrowAsyncProducer,
    /// Number of batches to request at a time (for back-pressure).
    prefetch_count: u64,
}

// Safety: Producer pointer is only dereferenced while holding the lock
unsafe impl Send for ProducerState {}
unsafe impl Sync for ProducerState {}

/// Shared state between the handler callbacks and the stream.
struct ImportedStreamState {
    /// Sender for messages from callbacks to stream (lock-free).
    sender: mpsc::UnboundedSender<StreamMessage>,
    /// Waker for the stream (lock-free).
    waker: AtomicWaker,
    /// Whether the stream has ended.
    ended: AtomicBool,
    /// Number of outstanding requests (for back-pressure).
    pending_requests: AtomicU64,
    /// Set to true when the stream is dropped but producer hasn't called release yet.
    abandoned: AtomicBool,
    /// Set to true when handler_release is called (handler was freed by producer).
    handler_released: AtomicBool,
    /// Producer state (needs mutex for FFI calls).
    producer_state: Mutex<ProducerState>,
}

// Safety: All fields are Send+Sync (atomics, Mutex, channel sender).
// The producer pointer inside ProducerState is protected by the Mutex.
unsafe impl Send for ImportedStreamState {}
unsafe impl Sync for ImportedStreamState {}

impl ImportedStreamState {
    fn new(prefetch_count: u64, sender: mpsc::UnboundedSender<StreamMessage>) -> Self {
        Self {
            sender,
            waker: AtomicWaker::new(),
            ended: AtomicBool::new(false),
            pending_requests: AtomicU64::new(0),
            abandoned: AtomicBool::new(false),
            handler_released: AtomicBool::new(false),
            producer_state: Mutex::new(ProducerState {
                producer: null_mut(),
                prefetch_count,
            }),
        }
    }

    fn wake(&self) {
        self.waker.wake();
    }

    /// Request more data from the producer if needed.
    fn maybe_request_more(&self) {
        if self.ended.load(Ordering::Acquire) {
            return;
        }

        // Lock cannot be poisoned: we never panic while holding it
        let producer_state = self.producer_state.lock().expect("producer_state mutex poisoned");
        if producer_state.producer.is_null() {
            return;
        }

        let pending = self.pending_requests.load(Ordering::Acquire);
        let prefetch = producer_state.prefetch_count;

        // Request more when we're running low
        if pending < prefetch / 2 {
            let to_request = prefetch - pending;
            if let Some(request_fn) = unsafe { (*producer_state.producer).request } {
                unsafe { request_fn(producer_state.producer, to_request) };
                self.pending_requests.fetch_add(to_request, Ordering::Release);
            }
        }
    }

    fn set_producer(&self, producer: *mut FFI_ArrowAsyncProducer) {
        // Lock cannot be poisoned: we never panic while holding it
        let mut state = self.producer_state.lock().expect("producer_state mutex poisoned");
        state.producer = producer;
    }

    fn clear_producer(&self) {
        // Lock cannot be poisoned: we never panic while holding it
        let mut state = self.producer_state.lock().expect("producer_state mutex poisoned");
        state.producer = null_mut();
    }

    fn cancel(&self) {
        // Lock cannot be poisoned: we never panic while holding it
        let state = self.producer_state.lock().expect("producer_state mutex poisoned");
        if !state.producer.is_null() {
            if let Some(cancel_fn) = unsafe { (*state.producer).cancel } {
                unsafe { cancel_fn(state.producer) };
            }
        }
    }
}

/// A [`RecordBatchStream`] that consumes data from an FFI async producer.
///
/// Create with [`ImportedAsyncDeviceStream::new`], then pass the handler pointer
/// to the FFI producer. The stream will yield batches as the producer sends them.
///
/// # Dropping
///
/// When the stream is dropped, it signals cancellation to the producer. The handler
/// remains valid until the producer calls its `release` callback. This allows the
/// idiomatic DataFusion pattern of dropping streams to cancel queries.
///
/// # Performance
///
/// Uses a lock-free channel for message passing from FFI callbacks to the stream.
/// Only acquires a mutex when requesting more batches from the producer (approximately
/// once per `prefetch_count / 2` batches).
pub struct ImportedAsyncDeviceStream {
    state: Arc<ImportedStreamState>,
    /// Receiver for messages from callbacks (lock-free).
    receiver: mpsc::UnboundedReceiver<StreamMessage>,
    /// Cached schema for the RecordBatchStream trait.
    schema: Option<SchemaRef>,
    /// DataType for converting FFI arrays.
    schema_struct_type: Option<DataType>,
}

// Safety: The state uses atomic operations and mutex-protected producer access.
unsafe impl Send for ImportedAsyncDeviceStream {}

/// RAII wrapper for the FFI handler.
///
/// This wrapper ensures the handler is properly cleaned up even if the FFI producer
/// never calls `release()`. It provides safe access to the raw pointer for FFI calls.
///
/// # Usage
///
/// ```ignore
/// let (stream, handler) = ImportedAsyncDeviceStream::new(16);
///
/// // Pass raw pointer to FFI producer
/// ffi_producer_start(handler.as_ptr());
///
/// // Handler is automatically cleaned up when dropped (if producer didn't release it)
/// ```
pub struct AsyncDeviceStreamHandler {
    /// Raw pointer to the handler (heap-allocated).
    ptr: *mut FFI_ArrowAsyncDeviceStreamHandler,
    /// Shared state with the stream - needed to check if already released.
    state: Arc<ImportedStreamState>,
}

// Safety: The handler pointer is only accessed from one thread at a time.
// The state Arc provides synchronization.
unsafe impl Send for AsyncDeviceStreamHandler {}

impl AsyncDeviceStreamHandler {
    /// Get the raw pointer to pass to FFI code.
    ///
    /// The pointer remains valid until either:
    /// - The FFI producer calls the handler's `release` callback, or
    /// - This wrapper is dropped
    #[inline]
    pub fn as_ptr(&self) -> *mut FFI_ArrowAsyncDeviceStreamHandler {
        self.ptr
    }
}

impl Drop for AsyncDeviceStreamHandler {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }

        // Check via shared state if the handler was already released by the producer.
        // We MUST check this before touching self.ptr because handler_release frees it.
        if self.state.handler_released.load(Ordering::Acquire) {
            // Producer already called release and freed the handler
            return;
        }

        // Handler was never released by producer - clean it up ourselves.
        // This can happen if the producer never connected or crashed.
        //
        // We need to:
        // 1. Mark the stream as ended
        // 2. Drop the Arc reference in private_data
        // 3. Free the handler struct

        let handler = unsafe { &mut *self.ptr };

        self.state.ended.store(true, Ordering::Release);

        if !handler.private_data.is_null() {
            let state_ptr = handler.private_data as *const ImportedStreamState;
            let _ = unsafe { Arc::from_raw(state_ptr) };
            handler.private_data = null_mut();
        }

        // Clear release to prevent FFI_ArrowAsyncDeviceStreamHandler's Drop
        // from calling it (which would double-free)
        handler.release = None;

        let _ = unsafe { Box::from_raw(self.ptr) };
    }
}

impl ImportedAsyncDeviceStream {
    /// Create a new imported stream.
    ///
    /// Returns the stream and an RAII handler wrapper that should be used to pass
    /// the handler pointer to the FFI producer.
    ///
    /// # Arguments
    ///
    /// * `prefetch_count` - Number of batches to request ahead for back-pressure.
    ///   A larger value reduces latency but uses more memory.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (stream, handler) = ImportedAsyncDeviceStream::new(16);
    ///
    /// // Pass to FFI producer
    /// ffi_producer_start(handler.as_ptr());
    ///
    /// // Stream the data
    /// while let Some(batch) = stream.next().await {
    ///     // ...
    /// }
    /// // Handler is automatically cleaned up when dropped
    /// ```
    pub fn new(prefetch_count: u64) -> (Self, AsyncDeviceStreamHandler) {
        let (sender, receiver) = mpsc::unbounded();
        let state = Arc::new(ImportedStreamState::new(prefetch_count, sender));

        // Handler is heap-allocated and will be freed when:
        // 1. Producer calls release(), or
        // 2. AsyncDeviceStreamHandler is dropped (fallback cleanup)
        let handler_ptr = Box::into_raw(Box::new(FFI_ArrowAsyncDeviceStreamHandler {
            on_schema: Some(handler_on_schema),
            on_next_task: Some(handler_on_next_task),
            on_error: Some(handler_on_error),
            release: Some(handler_release),
            producer: null_mut(),
            private_data: Arc::into_raw(Arc::clone(&state)) as *mut c_void,
        }));

        let stream = Self {
            state: Arc::clone(&state),
            receiver,
            schema: None,
            schema_struct_type: None,
        };

        let handler = AsyncDeviceStreamHandler {
            ptr: handler_ptr,
            state,
        };

        (stream, handler)
    }

    /// Convert this stream into a [`SendableRecordBatchStream`].
    ///
    /// This is a convenience method equivalent to `Box::pin(stream)`.
    pub fn into_sendable(self) -> datafusion_execution::SendableRecordBatchStream {
        Box::pin(self)
    }

    /// Cancel the stream, signaling to the producer to stop sending data.
    pub fn cancel(&self) {
        self.state.cancel();
    }

    /// Poll for the next message, handling schema initialization.
    fn poll_next_inner(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<RecordBatch>>> {
        // Register waker (lock-free)
        self.state.waker.register(cx.waker());

        // Poll the channel for messages first (lock-free)
        // This ensures we drain all messages before returning None due to ended flag
        match Pin::new(&mut self.receiver).poll_next(cx) {
            Poll::Ready(Some(msg)) => match msg {
                StreamMessage::Schema(schema) => {
                    self.schema_struct_type =
                        Some(DataType::Struct(schema.fields().iter().cloned().collect()));
                    self.schema = Some(schema);

                    // Request initial batch of data (acquires lock)
                    self.state.maybe_request_more();

                    // Continue polling for actual data
                    self.poll_next_inner(cx)
                }
                StreamMessage::Task(mut task) => {
                    // Decrement pending (lock-free)
                    self.state
                        .pending_requests
                        .fetch_sub(1, Ordering::Release);
                    // Maybe request more (acquires lock only if needed)
                    self.state.maybe_request_more();

                    // Extract data from the task
                    let batch = self.extract_batch_from_task(&mut task);
                    Poll::Ready(Some(batch))
                }
                StreamMessage::EndOfStream => {
                    self.state.ended.store(true, Ordering::Release);
                    Poll::Ready(None)
                }
                StreamMessage::Error(e) => {
                    self.state.ended.store(true, Ordering::Release);
                    Poll::Ready(Some(Err(e.into())))
                }
            },
            Poll::Ready(None) => {
                // Channel closed (shouldn't happen in normal operation)
                self.state.ended.store(true, Ordering::Release);
                Poll::Ready(None)
            }
            Poll::Pending => {
                // No messages in channel - check if stream has ended (lock-free)
                if self.state.ended.load(Ordering::Acquire) {
                    Poll::Ready(None)
                } else {
                    Poll::Pending
                }
            }
        }
    }

    /// Extract a RecordBatch from an FFI task.
    fn extract_batch_from_task(&self, task: &mut FFI_ArrowAsyncTask) -> Result<RecordBatch> {
        let Some(extract_data) = task.extract_data else {
            return Err(ArrowError::CDataInterface("extract_data is null".to_string()).into());
        };

        let mut device_array = FFI_ArrowDeviceArray {
            array: FFI_ArrowArray::empty(),
            device_id: 0,
            device_type: 0,
            sync_event: null_mut(),
        };

        let ret = unsafe { extract_data(task, &mut device_array) };
        if ret != 0 {
            return Err(ArrowError::CDataInterface("extract_data failed".to_string()).into());
        }

        // Only CPU device supported
        if device_array.device_type != ARROW_DEVICE_CPU {
            return Err(ArrowError::CDataInterface(format!(
                "Unsupported device type: {}",
                device_array.device_type
            ))
            .into());
        }

        // Convert to RecordBatch
        let Some(ref struct_type) = self.schema_struct_type else {
            return Err(ArrowError::CDataInterface("Schema not yet received".to_string()).into());
        };

        let array_data =
            unsafe { from_ffi_and_data_type(device_array.array, struct_type.clone())? };
        let struct_array: StructArray = array_data.into();
        Ok(struct_array.into())
    }
}

impl Stream for ImportedAsyncDeviceStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.poll_next_inner(cx)
    }
}

impl RecordBatchStream for ImportedAsyncDeviceStream {
    fn schema(&self) -> SchemaRef {
        self.schema
            .clone()
            .unwrap_or_else(|| Arc::new(Schema::empty()))
    }
}

impl Drop for ImportedAsyncDeviceStream {
    fn drop(&mut self) {
        // Mark as abandoned so callbacks know to return errors (lock-free)
        self.state.abandoned.store(true, Ordering::Release);

        // Signal cancellation to the producer (acquires lock)
        self.state.cancel();
    }
}

// FFI callback implementations

unsafe extern "C" fn handler_on_schema(
    self_: *mut FFI_ArrowAsyncDeviceStreamHandler,
    schema: *mut FFI_ArrowSchema,
) -> c_int {
    if self_.is_null() || schema.is_null() {
        return 1;
    }

    let handler = &mut *self_;
    if handler.private_data.is_null() {
        return 1;
    }

    // Get shared state
    let state_ptr = handler.private_data as *const ImportedStreamState;
    let state_arc = Arc::from_raw(state_ptr);

    // If stream was dropped, tell producer to stop (lock-free check)
    if state_arc.abandoned.load(Ordering::Acquire) {
        let _ = Arc::into_raw(state_arc);
        return 1; // Error code signals producer to stop and release
    }

    // Store the producer pointer (acquires lock)
    state_arc.set_producer(handler.producer);

    // Import the schema
    let ffi_schema = std::ptr::read(schema);
    let result = match Schema::try_from(&ffi_schema) {
        Ok(s) => {
            // Send through channel (lock-free)
            let _ = state_arc.sender.unbounded_send(StreamMessage::Schema(Arc::new(s)));
            state_arc.wake();
            0
        }
        Err(e) => {
            let _ = state_arc.sender.unbounded_send(StreamMessage::Error(e));
            state_arc.wake();
            1
        }
    };

    // Don't drop the Arc, just forget about this reference
    let _ = Arc::into_raw(state_arc);
    result
}

unsafe extern "C" fn handler_on_next_task(
    self_: *mut FFI_ArrowAsyncDeviceStreamHandler,
    task: *mut FFI_ArrowAsyncTask,
    _metadata: *const std::ffi::c_char,
) -> c_int {
    if self_.is_null() {
        return 1;
    }

    let handler = &*self_;
    if handler.private_data.is_null() {
        return 1;
    }

    let state_ptr = handler.private_data as *const ImportedStreamState;
    let state_arc = Arc::from_raw(state_ptr);

    // If stream was dropped, tell producer to stop (lock-free check)
    if state_arc.abandoned.load(Ordering::Acquire) {
        let _ = Arc::into_raw(state_arc);
        return 1; // Error code signals producer to stop and release
    }

    // Send message through channel (lock-free)
    if task.is_null() {
        // NULL task signals end of stream
        let _ = state_arc.sender.unbounded_send(StreamMessage::EndOfStream);
    } else {
        // Take ownership of the task by copying it
        let task_copy = std::ptr::read(task);
        let _ = state_arc.sender.unbounded_send(StreamMessage::Task(task_copy));
    }
    state_arc.wake();

    let _ = Arc::into_raw(state_arc);
    0
}

unsafe extern "C" fn handler_on_error(
    self_: *mut FFI_ArrowAsyncDeviceStreamHandler,
    code: c_int,
    message: *const std::ffi::c_char,
    _metadata: *const std::ffi::c_char,
) {
    if self_.is_null() {
        return;
    }

    let handler = &*self_;
    if handler.private_data.is_null() {
        return;
    }

    let state_ptr = handler.private_data as *const ImportedStreamState;
    let state_arc = Arc::from_raw(state_ptr);

    let error_msg = if message.is_null() {
        format!("FFI error code {}", code)
    } else {
        let c_str = CStr::from_ptr(message);
        c_str.to_string_lossy().into_owned()
    };

    // Send error through channel (lock-free)
    let _ = state_arc
        .sender
        .unbounded_send(StreamMessage::Error(ArrowError::CDataInterface(error_msg)));
    state_arc.ended.store(true, Ordering::Release);
    state_arc.wake();

    let _ = Arc::into_raw(state_arc);
}

unsafe extern "C" fn handler_release(self_: *mut FFI_ArrowAsyncDeviceStreamHandler) {
    if self_.is_null() {
        return;
    }

    let handler = &mut *self_;
    if handler.private_data.is_null() {
        // Already released - free the handler struct
        // Clear release to prevent Drop from calling us again
        handler.release = None;
        let _ = Box::from_raw(self_);
        return;
    }

    // Get the state and clear the producer pointer so stream Drop won't use it
    let state_ptr = handler.private_data as *const ImportedStreamState;
    let state_arc = Arc::from_raw(state_ptr);

    state_arc.clear_producer();
    state_arc.ended.store(true, Ordering::Release);

    // Mark handler as released so AsyncDeviceStreamHandler::drop knows not to free it
    state_arc.handler_released.store(true, Ordering::Release);

    // Drop the Arc reference (will be freed when stream also drops its reference)
    drop(state_arc);
    handler.private_data = null_mut();

    // Clear release to prevent Drop impl from calling us again, then free
    handler.release = None;
    let _ = Box::from_raw(self_);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export_sendable_record_batch_stream::drive_stream_to_handler;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::Field;
    use datafusion_execution::RecordBatchStream;
    use futures::StreamExt;
    use std::pin::Pin;
    use std::sync::Arc;

    /// Create a test schema with id and name columns.
    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    /// Create a test batch with the given row count.
    fn make_batch(schema: &SchemaRef, start_id: i32, count: i32) -> RecordBatch {
        let ids: Vec<i32> = (start_id..start_id + count).collect();
        let names: Vec<Option<&str>> = ids
            .iter()
            .map(|i| Some(if i % 2 == 0 { "even" } else { "odd" }))
            .collect();

        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    /// A simple SendableRecordBatchStream from a list of batches.
    struct TestStream {
        schema: SchemaRef,
        batches: std::collections::VecDeque<Result<RecordBatch>>,
    }

    impl TestStream {
        fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Pin<Box<Self>> {
            Box::pin(Self {
                schema,
                batches: batches.into_iter().map(Ok).collect(),
            })
        }

        fn with_error(
            schema: SchemaRef,
            batches: Vec<RecordBatch>,
            error_after: usize,
        ) -> Pin<Box<Self>> {
            let mut results: std::collections::VecDeque<Result<RecordBatch>> =
                batches.into_iter().map(Ok).collect();
            if error_after < results.len() {
                results.insert(
                    error_after,
                    Err(datafusion_common::DataFusionError::External(
                        "Test error".into(),
                    )),
                );
            } else {
                results.push_back(Err(datafusion_common::DataFusionError::External(
                    "Test error".into(),
                )));
            }
            Box::pin(Self {
                schema,
                batches: results,
            })
        }
    }

    impl Stream for TestStream {
        type Item = Result<RecordBatch>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            let this = self.as_mut().get_mut();
            Poll::Ready(this.batches.pop_front())
        }
    }

    impl RecordBatchStream for TestStream {
        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }
    }

    #[tokio::test]
    async fn test_empty_stream() {
        let schema = test_schema();
        let source_stream = TestStream::new(schema.clone(), vec![]);

        let (mut consumer, handler) = ImportedAsyncDeviceStream::new(4);

        let consumer_future = async {
            let mut received = vec![];
            while let Some(result) = consumer.next().await {
                received.push(result);
            }
            (received, consumer.schema())
        };

        let producer_future = drive_stream_to_handler(source_stream, handler.as_ptr());

        let ((received, result_schema), _) = futures::join!(consumer_future, producer_future);

        assert!(received.is_empty());
        assert_eq!(result_schema.fields().len(), 2);
    }

    #[tokio::test]
    async fn test_single_batch() {
        let schema = test_schema();
        let batch = make_batch(&schema, 1, 5);
        let source_stream = TestStream::new(schema.clone(), vec![batch.clone()]);

        let (mut consumer, handler) = ImportedAsyncDeviceStream::new(4);

        let consumer_future = async {
            let mut received = vec![];
            while let Some(result) = consumer.next().await {
                received.push(result);
            }
            received
        };

        let producer_future = drive_stream_to_handler(source_stream, handler.as_ptr());

        let (received, _) = futures::join!(consumer_future, producer_future);

        assert_eq!(received.len(), 1);
        let received_batch = received.into_iter().next().unwrap().unwrap();
        assert_eq!(received_batch.num_rows(), 5);
        assert_eq!(received_batch.schema(), batch.schema());

        // Verify data
        let ids = received_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn test_multiple_batches_with_backpressure() {
        let schema = test_schema();

        // Create more batches than the prefetch count
        let prefetch_count = 2u64;
        let num_batches = 10;
        let batches: Vec<RecordBatch> = (0..num_batches)
            .map(|i| make_batch(&schema, i * 3, 3))
            .collect();
        let expected_total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        let source_stream = TestStream::new(schema.clone(), batches);

        let (mut consumer, handler) = ImportedAsyncDeviceStream::new(prefetch_count);

        let consumer_future = async {
            let mut received = vec![];
            while let Some(result) = consumer.next().await {
                received.push(result);
            }
            received
        };

        let producer_future = drive_stream_to_handler(source_stream, handler.as_ptr());

        let (received, _) = futures::join!(consumer_future, producer_future);

        assert_eq!(received.len(), num_batches as usize);
        let total_rows: usize = received.into_iter().map(|r| r.unwrap().num_rows()).sum();
        assert_eq!(total_rows, expected_total_rows);
    }

    #[tokio::test]
    async fn test_stream_error() {
        let schema = test_schema();
        let batches = vec![make_batch(&schema, 1, 3), make_batch(&schema, 4, 3)];

        // Error after first batch
        let source_stream = TestStream::with_error(schema.clone(), batches, 1);

        let (mut consumer, handler) = ImportedAsyncDeviceStream::new(4);

        let consumer_future = async {
            let mut received = vec![];
            while let Some(result) = consumer.next().await {
                received.push(result);
            }
            received
        };

        let producer_future = drive_stream_to_handler(source_stream, handler.as_ptr());

        let (received, _) = futures::join!(consumer_future, producer_future);

        // First batch should succeed
        assert!(!received.is_empty());
        assert!(received[0].is_ok());

        // Should have an error somewhere
        let has_error = received.iter().any(|r| r.is_err());
        assert!(has_error, "Expected an error in the stream");
    }

    #[tokio::test]
    async fn test_cancellation() {
        let schema = test_schema();

        // Create a large number of batches
        let batches: Vec<RecordBatch> = (0..100).map(|i| make_batch(&schema, i * 3, 3)).collect();

        let source_stream = TestStream::new(schema.clone(), batches);

        let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);

        let consumer_future = async {
            let mut count = 0;
            while let Some(result) = consumer.next().await {
                result.expect("should not error");
                count += 1;
                if count >= 3 {
                    // Cancel after receiving 3 batches
                    consumer.cancel();
                    break;
                }
            }
            count
        };

        let producer_future = drive_stream_to_handler(source_stream, handler.as_ptr());

        let (count, _) = futures::join!(consumer_future, producer_future);

        // Should have received at least 3 batches before cancellation
        assert!(count >= 3);
    }
}
