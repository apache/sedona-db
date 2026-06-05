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
//! let (stream, handler_ptr) = ImportedAsyncDeviceStream::new(16);
//!
//! // Pass handler_ptr to FFI producer...
//! // ffi_producer_start(handler_ptr);
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
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use arrow_array::ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{RecordBatch, StructArray};
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::RecordBatchStream;
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

/// Shared state between the handler callbacks and the stream.
struct ImportedStreamState {
    /// The schema, set by on_schema callback.
    schema: Option<SchemaRef>,
    /// Pending messages from callbacks.
    messages: Vec<StreamMessage>,
    /// Waker to wake the stream when new data arrives.
    waker: Option<Waker>,
    /// Whether the stream has ended.
    ended: bool,
    /// Producer pointer for requesting more data.
    producer: *mut FFI_ArrowAsyncProducer,
    /// Number of batches to request at a time (for back-pressure).
    prefetch_count: u64,
    /// Number of outstanding requests.
    pending_requests: u64,
}

// Safety: We ensure proper synchronization via Mutex
unsafe impl Send for ImportedStreamState {}

impl ImportedStreamState {
    fn new(prefetch_count: u64) -> Self {
        Self {
            schema: None,
            messages: Vec::new(),
            waker: None,
            ended: false,
            producer: null_mut(),
            prefetch_count,
            pending_requests: 0,
        }
    }

    fn wake(&self) {
        if let Some(ref waker) = self.waker {
            waker.wake_by_ref();
        }
    }

    /// Request more data from the producer if needed.
    fn maybe_request_more(&mut self) {
        if self.ended || self.producer.is_null() {
            return;
        }

        // Request more when we're running low
        if self.pending_requests < self.prefetch_count / 2 {
            let to_request = self.prefetch_count - self.pending_requests;
            if let Some(request_fn) = unsafe { (*self.producer).request } {
                unsafe { request_fn(self.producer, to_request) };
                self.pending_requests += to_request;
            }
        }
    }
}

/// A [`RecordBatchStream`] that consumes data from an FFI async producer.
///
/// Create with [`ImportedAsyncDeviceStream::new`], then pass the handler pointer
/// to the FFI producer. The stream will yield batches as the producer sends them.
pub struct ImportedAsyncDeviceStream {
    state: Arc<Mutex<ImportedStreamState>>,
    /// The handler - kept alive for the lifetime of the stream.
    /// Box is used to get a stable address.
    _handler: Box<FFI_ArrowAsyncDeviceStreamHandler>,
    /// Cached schema for the RecordBatchStream trait.
    schema: Option<SchemaRef>,
    /// DataType for converting FFI arrays.
    schema_struct_type: Option<DataType>,
}

// Safety: The handler contains raw pointers but they are only accessed from
// the thread that polls the stream. The Arc<Mutex<_>> provides synchronization
// for the shared state accessed by callbacks.
unsafe impl Send for ImportedAsyncDeviceStream {}

impl ImportedAsyncDeviceStream {
    /// Create a new imported stream.
    ///
    /// Returns the stream and a pointer to the handler that should be passed to the FFI producer.
    ///
    /// # Arguments
    ///
    /// * `prefetch_count` - Number of batches to request ahead for back-pressure.
    ///   A larger value reduces latency but uses more memory.
    ///
    /// # Safety
    ///
    /// The returned handler pointer is valid for the lifetime of the stream.
    /// The FFI producer must not use the handler after the stream is dropped.
    pub fn new(prefetch_count: u64) -> (Self, *mut FFI_ArrowAsyncDeviceStreamHandler) {
        let state = Arc::new(Mutex::new(ImportedStreamState::new(prefetch_count)));

        let handler = Box::new(FFI_ArrowAsyncDeviceStreamHandler {
            on_schema: Some(handler_on_schema),
            on_next_task: Some(handler_on_next_task),
            on_error: Some(handler_on_error),
            release: Some(handler_release),
            producer: null_mut(),
            private_data: Arc::into_raw(Arc::clone(&state)) as *mut c_void,
        });

        let handler_ptr = handler.as_ref() as *const _ as *mut FFI_ArrowAsyncDeviceStreamHandler;

        let stream = Self {
            state,
            _handler: handler,
            schema: None,
            schema_struct_type: None,
        };

        (stream, handler_ptr)
    }

    /// Convert this stream into a [`SendableRecordBatchStream`].
    ///
    /// This is a convenience method equivalent to `Box::pin(stream)`.
    pub fn into_sendable(self) -> datafusion_execution::SendableRecordBatchStream {
        Box::pin(self)
    }

    /// Cancel the stream, signaling to the producer to stop sending data.
    pub fn cancel(&self) {
        let state = self.state.lock().unwrap();
        if !state.producer.is_null() {
            if let Some(cancel_fn) = unsafe { (*state.producer).cancel } {
                unsafe { cancel_fn(state.producer) };
            }
        }
    }

    /// Poll for the next message, handling schema initialization.
    fn poll_next_inner(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<RecordBatch>>> {
        let mut state = self.state.lock().unwrap();

        // Register waker for when new data arrives
        state.waker = Some(cx.waker().clone());

        // Process any pending messages
        if let Some(msg) = state.messages.pop() {
            match msg {
                StreamMessage::Schema(schema) => {
                    self.schema_struct_type =
                        Some(DataType::Struct(schema.fields().iter().cloned().collect()));
                    self.schema = Some(schema.clone());
                    state.schema = Some(schema);

                    // Request initial batch of data
                    state.maybe_request_more();

                    // Continue polling for actual data
                    drop(state);
                    return self.poll_next_inner(cx);
                }
                StreamMessage::Task(mut task) => {
                    state.pending_requests = state.pending_requests.saturating_sub(1);
                    state.maybe_request_more();
                    drop(state);

                    // Extract data from the task
                    let batch = self.extract_batch_from_task(&mut task);
                    return Poll::Ready(Some(batch));
                }
                StreamMessage::EndOfStream => {
                    state.ended = true;
                    return Poll::Ready(None);
                }
                StreamMessage::Error(e) => {
                    state.ended = true;
                    return Poll::Ready(Some(Err(e.into())));
                }
            }
        }

        // Check if stream has ended
        if state.ended {
            return Poll::Ready(None);
        }

        // No messages available, wait for callback
        Poll::Pending
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

    // Store the producer pointer for later use
    let state_ptr = handler.private_data as *const Mutex<ImportedStreamState>;
    let state_arc = Arc::from_raw(state_ptr);

    let result = {
        let mut state = state_arc.lock().unwrap();
        state.producer = handler.producer;

        // Import the schema
        let ffi_schema = std::ptr::read(schema);
        match Schema::try_from(&ffi_schema) {
            Ok(s) => {
                state.messages.push(StreamMessage::Schema(Arc::new(s)));
                state.wake();
                0
            }
            Err(e) => {
                state.messages.push(StreamMessage::Error(e));
                state.wake();
                1
            }
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

    let state_ptr = handler.private_data as *const Mutex<ImportedStreamState>;
    let state_arc = Arc::from_raw(state_ptr);

    {
        let mut state = state_arc.lock().unwrap();

        if task.is_null() {
            // NULL task signals end of stream
            state.messages.push(StreamMessage::EndOfStream);
        } else {
            // Take ownership of the task by copying it
            let task_copy = std::ptr::read(task);
            state.messages.push(StreamMessage::Task(task_copy));
        }
        state.wake();
    }

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

    let state_ptr = handler.private_data as *const Mutex<ImportedStreamState>;
    let state_arc = Arc::from_raw(state_ptr);

    {
        let mut state = state_arc.lock().unwrap();

        let error_msg = if message.is_null() {
            format!("FFI error code {}", code)
        } else {
            let c_str = CStr::from_ptr(message);
            c_str.to_string_lossy().into_owned()
        };

        state
            .messages
            .push(StreamMessage::Error(ArrowError::CDataInterface(error_msg)));
        state.ended = true;
        state.wake();
    }

    let _ = Arc::into_raw(state_arc);
}

unsafe extern "C" fn handler_release(self_: *mut FFI_ArrowAsyncDeviceStreamHandler) {
    if self_.is_null() {
        return;
    }

    let handler = &mut *self_;
    if handler.private_data.is_null() {
        return;
    }

    // Drop our Arc reference
    let state_ptr = handler.private_data as *const Mutex<ImportedStreamState>;
    let _ = Arc::from_raw(state_ptr);
    handler.private_data = null_mut();
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

    /// A simple SendableRecordBatchStream that yields once before each batch.
    /// This ensures proper async interleaving between producer and consumer.
    struct TestStream {
        schema: SchemaRef,
        batches: std::collections::VecDeque<Result<RecordBatch>>,
        yielded: bool,
    }

    impl TestStream {
        fn new(schema: SchemaRef, batches: Vec<RecordBatch>) -> Pin<Box<Self>> {
            Box::pin(Self {
                schema,
                batches: batches.into_iter().map(Ok).collect(),
                yielded: false, // Start not yielded so first poll yields
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
                yielded: false, // Start not yielded so first poll yields
            })
        }
    }

    impl Stream for TestStream {
        type Item = Result<RecordBatch>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            let this = self.as_mut().get_mut();
            if this.yielded {
                // After yielding, return data on next poll
                this.yielded = false;
                if let Some(batch) = this.batches.pop_front() {
                    Poll::Ready(Some(batch))
                } else {
                    Poll::Ready(None)
                }
            } else {
                // Haven't yielded yet - yield once to give other tasks a chance
                this.yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
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

        let (mut consumer, handler_ptr) = ImportedAsyncDeviceStream::new(4);

        let consumer_future = async {
            let mut received = vec![];
            while let Some(result) = consumer.next().await {
                received.push(result);
            }
            (received, consumer.schema())
        };

        let producer_future = drive_stream_to_handler(source_stream, handler_ptr);

        let ((received, result_schema), _) = futures::join!(consumer_future, producer_future);

        assert!(received.is_empty());
        assert_eq!(result_schema.fields().len(), 2);
    }

    #[tokio::test]
    async fn test_single_batch() {
        let schema = test_schema();
        let batch = make_batch(&schema, 1, 5);
        let source_stream = TestStream::new(schema.clone(), vec![batch.clone()]);

        let (mut consumer, handler_ptr) = ImportedAsyncDeviceStream::new(4);

        let consumer_future = async {
            let mut received = vec![];
            while let Some(result) = consumer.next().await {
                received.push(result);
            }
            received
        };

        let producer_future = drive_stream_to_handler(source_stream, handler_ptr);

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

        let (mut consumer, handler_ptr) = ImportedAsyncDeviceStream::new(prefetch_count);

        let consumer_future = async {
            let mut received = vec![];
            while let Some(result) = consumer.next().await {
                received.push(result);
            }
            received
        };

        let producer_future = drive_stream_to_handler(source_stream, handler_ptr);

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

        let (mut consumer, handler_ptr) = ImportedAsyncDeviceStream::new(4);

        let consumer_future = async {
            let mut received = vec![];
            while let Some(result) = consumer.next().await {
                received.push(result);
            }
            received
        };

        let producer_future = drive_stream_to_handler(source_stream, handler_ptr);

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

        let (mut consumer, handler_ptr) = ImportedAsyncDeviceStream::new(2);

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

        let producer_future = drive_stream_to_handler(source_stream, handler_ptr);

        let (count, _) = futures::join!(consumer_future, producer_future);

        // Should have received at least 3 batches before cancellation
        assert!(count >= 3);
    }
}
