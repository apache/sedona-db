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

//! FFI wrapper for [`SendableRecordBatchStream`] using the Arrow C Device Data Interface.
//!
//! This module provides interoperability between DataFusion's async record batch streams
//! and C code via the [`FFI_ArrowAsyncDeviceStreamHandler`] interface.
//!
//! The implementation follows the Arrow Async Device Stream specification, including:
//! - Back-pressure via `request()` callback
//! - Consumer-initiated cancellation via `cancel()` callback
//! - Proper end-of-stream signaling

use std::ffi::{c_int, c_void, CString};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Poll, Waker};

use arrow_array::ffi::{to_ffi, FFI_ArrowSchema};
use arrow_array::RecordBatch;
use futures::StreamExt;

pub use datafusion_execution::SendableRecordBatchStream;

use crate::extension_ffi::{
    FFI_ArrowAsyncDeviceStreamHandler, FFI_ArrowAsyncProducer, FFI_ArrowAsyncTask,
    FFI_ArrowDeviceArray, ARROW_DEVICE_CPU,
};

/// Yields control back to the async executor once, then resumes.
/// This allows other tasks (e.g., the consumer) to make progress.
async fn yield_once() {
    let mut yielded = false;
    futures::future::poll_fn(|cx| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await
}

/// Shared state between the async driver and the FFI producer callbacks.
struct ProducerState {
    /// Number of batches requested by consumer (back-pressure).
    requested: AtomicU64,
    /// Set to true when consumer calls cancel().
    cancelled: AtomicBool,
    /// Waker to wake when requests are available or cancelled.
    waker: std::sync::Mutex<Option<Waker>>,
}

impl ProducerState {
    fn new() -> Self {
        Self {
            requested: AtomicU64::new(0),
            cancelled: AtomicBool::new(false),
            waker: std::sync::Mutex::new(None),
        }
    }

    fn request(&self, n: u64) {
        self.requested.fetch_add(n, Ordering::SeqCst);
        self.wake();
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.wake();
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Try to consume one request slot. Returns true if successful.
    fn try_consume_request(&self) -> bool {
        loop {
            let current = self.requested.load(Ordering::SeqCst);
            if current == 0 {
                return false;
            }
            if self
                .requested
                .compare_exchange(current, current - 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return true;
            }
        }
    }

    fn has_requests(&self) -> bool {
        self.requested.load(Ordering::SeqCst) > 0
    }

    fn register_waker(&self, waker: Waker) {
        *self.waker.lock().unwrap() = Some(waker);
    }

    fn wake(&self) {
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }
}

/// Drives a [`SendableRecordBatchStream`] and pushes results to an [`FFI_ArrowAsyncDeviceStreamHandler`].
///
/// This function implements the full Arrow Async Device Stream producer protocol:
/// - Creates an `FFI_ArrowAsyncProducer` and sets it on the handler before calling `on_schema`
/// - Respects back-pressure: waits for consumer to call `producer.request(n)` before sending batches
/// - Handles consumer cancellation via `producer.cancel()`
/// - Signals end-of-stream by calling `on_next_task` with `NULL`
/// - Calls `on_error` if an error occurs (including cancellation)
///
/// # Safety
///
/// The `handler` pointer must be valid and point to a properly initialized
/// [`FFI_ArrowAsyncDeviceStreamHandler`].
///
/// # Example
///
/// ```ignore
/// use sedona_extension::sendable_record_batch_stream::drive_stream_to_handler;
///
/// // The consumer should call handler.producer.request(n) to receive batches
/// drive_stream_to_handler(stream, handler).await;
/// ```
pub async fn drive_stream_to_handler(
    stream: SendableRecordBatchStream,
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
) {
    if handler.is_null() {
        return;
    }

    // Create shared state for producer callbacks
    let state = Arc::new(ProducerState::new());

    // Create the producer
    let mut producer = create_producer(Arc::clone(&state));

    // Use a guard to handle unexpected drops (panics)
    let mut guard = StreamDriverGuard::new(handler);

    let _result = drive_stream_inner(stream, handler, &mut producer, state).await;

    // Normal completion (success or error) - disarm the guard and release handler
    guard.disarm();
    release_handler(handler);
}

/// Inner implementation that returns a Result for cleaner control flow.
async fn drive_stream_inner(
    mut stream: SendableRecordBatchStream,
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
    producer: &mut FFI_ArrowAsyncProducer,
    state: Arc<ProducerState>,
) -> Result<(), ()> {
    let handler_ref = unsafe { &mut *handler };

    // Set the producer on the handler BEFORE calling on_schema (per Arrow spec)
    handler_ref.producer = producer;

    // Send the schema
    let schema = stream.schema();
    let ffi_schema = match FFI_ArrowSchema::try_from(schema.as_ref()) {
        Ok(s) => s,
        Err(e) => {
            call_on_error(handler_ref, 1, &e.to_string());
            return Err(());
        }
    };

    if let Some(on_schema) = handler_ref.on_schema {
        let mut boxed_schema = Box::new(ffi_schema);
        let ret = unsafe { on_schema(handler, boxed_schema.as_mut()) };
        if ret != 0 {
            call_on_error(handler_ref, ret, "on_schema callback failed");
            return Err(());
        }
        // The handler now owns the schema
        std::mem::forget(boxed_schema);
    }

    // Stream batches with back-pressure
    loop {
        // Check for cancellation
        if state.is_cancelled() {
            // Per Arrow spec: successful cancel should NOT call on_error,
            // just signal end of stream and release
            signal_end_of_stream(handler_ref, handler);
            return Ok(());
        }

        // Wait for consumer to request batches (back-pressure)
        // Use poll_fn to properly register our waker with ProducerState
        futures::future::poll_fn(|cx| {
            if state.has_requests() || state.is_cancelled() {
                Poll::Ready(())
            } else {
                state.register_waker(cx.waker().clone());
                Poll::Pending
            }
        })
        .await;

        // Re-check cancellation after waiting
        if state.is_cancelled() {
            signal_end_of_stream(handler_ref, handler);
            return Ok(());
        }

        // Consume one request slot
        if !state.try_consume_request() {
            continue;
        }

        // Get next batch
        match stream.next().await {
            Some(Ok(batch)) => {
                if let Err(e) = send_batch_to_handler(handler_ref, handler, batch) {
                    call_on_error(handler_ref, 1, &e);
                    return Err(());
                }
                // Yield to allow consumer to process the batch.
                // This enables single-task usage (producer and consumer in same task).
                yield_once().await;
            }
            Some(Err(e)) => {
                call_on_error(handler_ref, 1, &e.to_string());
                return Err(());
            }
            None => {
                // End of stream
                signal_end_of_stream(handler_ref, handler);
                return Ok(());
            }
        }
    }
}

/// Create an FFI producer with request/cancel callbacks.
fn create_producer(state: Arc<ProducerState>) -> FFI_ArrowAsyncProducer {
    // Convert Arc to raw pointer - this increments the refcount
    let private_data = Arc::into_raw(state) as *mut c_void;

    FFI_ArrowAsyncProducer {
        device_type: ARROW_DEVICE_CPU,
        request: Some(producer_request),
        cancel: Some(producer_cancel),
        release: Some(producer_release),
        additional_metadata: std::ptr::null(),
        private_data,
    }
}

unsafe extern "C" fn producer_request(self_: *mut FFI_ArrowAsyncProducer, n: u64) {
    if self_.is_null() {
        return;
    }
    let producer = &*self_;
    if producer.private_data.is_null() {
        return;
    }
    let state = &*(producer.private_data as *const ProducerState);
    state.request(n);
}

unsafe extern "C" fn producer_cancel(self_: *mut FFI_ArrowAsyncProducer) {
    if self_.is_null() {
        return;
    }
    let producer = &*self_;
    if producer.private_data.is_null() {
        return;
    }
    let state = &*(producer.private_data as *const ProducerState);
    state.cancel();
}

unsafe extern "C" fn producer_release(self_: *mut FFI_ArrowAsyncProducer) {
    if self_.is_null() {
        return;
    }
    let producer = &mut *self_;
    if producer.private_data.is_null() {
        return;
    }
    // Drop the Arc<ProducerState>
    let _ = Arc::from_raw(producer.private_data as *const ProducerState);
    producer.private_data = null_mut();
}

/// Guard that calls `on_error` if dropped without being disarmed.
struct StreamDriverGuard {
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
    armed: bool,
}

impl StreamDriverGuard {
    fn new(handler: *mut FFI_ArrowAsyncDeviceStreamHandler) -> Self {
        Self {
            handler,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for StreamDriverGuard {
    fn drop(&mut self) {
        if self.armed && !self.handler.is_null() {
            let handler_ref = unsafe { &mut *self.handler };
            call_on_error(
                handler_ref,
                libc::ECANCELED,
                "Stream driver dropped unexpectedly",
            );
            // Still need to release the handler
            release_handler(self.handler);
        }
    }
}

/// Signal end of stream by calling on_next_task with NULL.
fn signal_end_of_stream(
    handler_ref: &mut FFI_ArrowAsyncDeviceStreamHandler,
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
) {
    if let Some(on_next_task) = handler_ref.on_next_task {
        // Per Arrow spec: pass NULL task pointer to signal end of stream
        unsafe {
            on_next_task(handler, std::ptr::null_mut(), std::ptr::null());
        }
    }
}

/// Release the handler (call its release callback).
fn release_handler(handler: *mut FFI_ArrowAsyncDeviceStreamHandler) {
    if handler.is_null() {
        return;
    }
    let handler_ref = unsafe { &*handler };
    if let Some(release) = handler_ref.release {
        unsafe { release(handler) };
    }
}

fn send_batch_to_handler(
    handler_ref: &mut FFI_ArrowAsyncDeviceStreamHandler,
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
    batch: RecordBatch,
) -> Result<(), String> {
    let Some(on_next_task) = handler_ref.on_next_task else {
        return Err("on_next_task callback is null".to_string());
    };

    // Create a task that wraps the batch
    let task = create_async_task(batch)?;
    let mut boxed_task = Box::new(task);

    let ret = unsafe { on_next_task(handler, boxed_task.as_mut(), std::ptr::null()) };
    if ret != 0 {
        return Err("on_next_task callback failed".to_string());
    }

    // The handler now owns the task
    std::mem::forget(boxed_task);
    Ok(())
}

fn call_on_error(handler: &mut FFI_ArrowAsyncDeviceStreamHandler, code: c_int, message: &str) {
    if let Some(on_error) = handler.on_error {
        let c_message = CString::new(message).unwrap_or_else(|_| CString::new("error").unwrap());
        unsafe {
            on_error(
                handler,
                code,
                c_message.as_ptr(),
                std::ptr::null(), // no metadata
            );
        }
    }
}

/// Private data for an async task wrapping a RecordBatch
struct AsyncTaskPrivate {
    batch: Option<RecordBatch>,
}

fn create_async_task(batch: RecordBatch) -> Result<FFI_ArrowAsyncTask, String> {
    let private_data = Box::new(AsyncTaskPrivate { batch: Some(batch) });

    Ok(FFI_ArrowAsyncTask {
        extract_data: Some(async_task_extract_data),
        private_data: Box::into_raw(private_data) as *mut c_void,
    })
}

unsafe extern "C" fn async_task_extract_data(
    self_: *mut FFI_ArrowAsyncTask,
    out: *mut FFI_ArrowDeviceArray,
) -> c_int {
    if self_.is_null() || out.is_null() {
        return 1;
    }

    let task = &mut *self_;
    if task.private_data.is_null() {
        return 1;
    }

    let private = &mut *(task.private_data as *mut AsyncTaskPrivate);

    // Take the batch (can only extract once)
    let Some(batch) = private.batch.take() else {
        return 1;
    };

    // Convert to struct array and then to FFI
    let struct_array: arrow_array::StructArray = batch.into();
    let (ffi_array, _ffi_schema) = match to_ffi(&struct_array.into()) {
        Ok(result) => result,
        Err(_) => return 1,
    };

    // Create device array (CPU)
    let device_array = FFI_ArrowDeviceArray::from(ffi_array);
    std::ptr::write(out, device_array);
    0
}

impl Drop for FFI_ArrowAsyncTask {
    fn drop(&mut self) {
        if !self.private_data.is_null() {
            // Drop the private data
            let _ = unsafe { Box::from_raw(self.private_data as *mut AsyncTaskPrivate) };
            self.private_data = null_mut();
        }
    }
}
