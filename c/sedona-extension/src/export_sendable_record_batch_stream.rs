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
use std::future::Future;
use std::ptr::null_mut;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::task::Poll;

use arrow_array::ffi::{to_ffi, FFI_ArrowSchema};
use arrow_array::RecordBatch;
use futures::task::AtomicWaker;
use futures::StreamExt;

pub use datafusion_execution::SendableRecordBatchStream;

use crate::extension::{
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
    waker: AtomicWaker,
}

impl ProducerState {
    fn new() -> Self {
        Self {
            requested: AtomicU64::new(0),
            cancelled: AtomicBool::new(false),
            waker: AtomicWaker::new(),
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

    fn wake(&self) {
        self.waker.wake();
    }
}

/// Drives a [`SendableRecordBatchStream`] and pushes results to an [`FFI_ArrowAsyncDeviceStreamHandler`].
///
/// This function implements the full Arrow Async Device Stream producer protocol:
/// - Creates an `FFI_ArrowAsyncProducer` and sets it on the handler before calling `on_schema`
/// - Respects back-pressure: waits for consumer to call `producer.request(n)` before sending batches
/// - Handles consumer cancellation via `producer.cancel()`
/// - Signals end-of-stream by calling `on_next_task` with `NULL`
/// - Calls `on_error` if the producer encounters an error
///
/// # Safety
///
/// A non-null `handler` must point to a properly initialized
/// [`FFI_ArrowAsyncDeviceStreamHandler`] that remains valid until its release
/// callback is invoked. This function takes exclusive responsibility for
/// invoking and releasing the handler, including if the returned future is
/// dropped without being polled. The caller must not release it independently.
/// Its callbacks and private state must support serialized calls from any
/// thread, since the returned future is `Send`.
///
/// # Example
///
/// ```ignore
/// use sedona_extension::export_sendable_record_batch_stream::drive_stream_to_handler;
///
/// // The consumer should call handler.producer.request(n) to receive batches
/// unsafe { drive_stream_to_handler(stream, handler) }.await;
/// ```
///
/// Raw handler ownership must be explicitly acknowledged by the caller:
///
/// ```compile_fail,E0133
/// use sedona_extension::export_sendable_record_batch_stream::drive_stream_to_handler;
/// use sedona_extension::extension::FFI_ArrowAsyncDeviceStreamHandler;
/// use datafusion_execution::SendableRecordBatchStream;
/// fn start(stream: SendableRecordBatchStream, handler: *mut FFI_ArrowAsyncDeviceStreamHandler) {
///     let _driver = drive_stream_to_handler(stream, handler);
/// }
/// ```
pub unsafe fn drive_stream_to_handler(
    stream: SendableRecordBatchStream,
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
) -> impl Future<Output = ()> + Send {
    // Construct the handler guard before creating the future so dropping an
    // unpolled spawned task still releases the handler.
    let handler = SendableHandlerGuard::new(handler);

    async move {
        let Some(handler) = handler else {
            return;
        };

        let state = Arc::new(ProducerState::new());
        let producer = SendableProducerGuard::new(create_producer(Arc::clone(&state)));
        let mut resources = SendableDriverResources { handler, producer };

        let _result = drive_stream_inner(
            stream,
            resources.handler_address(),
            resources.producer_address(),
            state,
        )
        .await;

        // Normal completion (success or callback rejection).
        resources.release_handler();
    }
}

/// Inner implementation that returns a Result for cleaner control flow.
async fn drive_stream_inner(
    mut stream: SendableRecordBatchStream,
    handler_address: usize,
    producer_address: usize,
    state: Arc<ProducerState>,
) -> Result<(), ()> {
    // Set the producer on the handler BEFORE calling on_schema (per Arrow spec)
    unsafe {
        (*(handler_address as *mut FFI_ArrowAsyncDeviceStreamHandler)).producer =
            producer_address as *mut FFI_ArrowAsyncProducer;
    }

    // Send the schema
    let schema = stream.schema();
    let mut ffi_schema = match FFI_ArrowSchema::try_from(schema.as_ref()) {
        Ok(s) => s,
        Err(e) => {
            call_on_error(handler_address, 1, &e.to_string());
            return Err(());
        }
    };

    let on_schema =
        unsafe { (*(handler_address as *mut FFI_ArrowAsyncDeviceStreamHandler)).on_schema };
    if let Some(on_schema) = on_schema {
        let ret = unsafe {
            on_schema(
                handler_address as *mut FFI_ArrowAsyncDeviceStreamHandler,
                &mut ffi_schema,
            )
        };
        // Ownership of the schema contents transfers to the handler when the
        // callback is invoked, regardless of its return value.
        std::mem::forget(ffi_schema);
        if ret != 0 {
            // A rejected callback may only be followed by release.
            return Err(());
        }
    }

    // Stream batches with back-pressure
    loop {
        // Check for cancellation
        if state.is_cancelled() {
            // Per Arrow spec: successful cancel should NOT call on_error,
            // just signal end of stream and release
            signal_end_of_stream(handler_address);
            return Ok(());
        }

        // Wait for consumer to request batches (back-pressure)
        // Register before checking the state so a concurrent request cannot be
        // lost between the state check and waker registration.
        futures::future::poll_fn(|cx| {
            state.waker.register(cx.waker());
            if state.has_requests() || state.is_cancelled() {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        })
        .await;

        // Re-check cancellation after waiting
        if state.is_cancelled() {
            signal_end_of_stream(handler_address);
            return Ok(());
        }

        // Consume one request slot
        if !state.try_consume_request() {
            continue;
        }

        // Poll cancellation and the source together. A source is allowed to
        // remain pending indefinitely, so cancellation must wake and interrupt
        // this wait instead of only being observed between batches.
        let next = futures::future::poll_fn(|cx| {
            state.waker.register(cx.waker());
            if state.is_cancelled() {
                return Poll::Ready(Err(()));
            }

            match stream.poll_next_unpin(cx) {
                Poll::Ready(batch) => Poll::Ready(Ok(batch)),
                Poll::Pending if state.is_cancelled() => Poll::Ready(Err(())),
                Poll::Pending => Poll::Pending,
            }
        })
        .await;

        let next = match next {
            Ok(next) => next,
            Err(()) => {
                signal_end_of_stream(handler_address);
                return Ok(());
            }
        };

        match next {
            Some(Ok(batch)) => {
                match send_batch_to_handler(handler_address, batch) {
                    Ok(()) => {}
                    Err(SendBatchError::HandlerRejected) => {
                        // A rejected callback may only be followed by release.
                        return Err(());
                    }
                    Err(SendBatchError::Producer(error)) => {
                        call_on_error(handler_address, 1, &error);
                        return Err(());
                    }
                }
                // Yield to allow consumer to process the batch.
                // This enables single-task usage (producer and consumer in same task).
                yield_once().await;
            }
            Some(Err(e)) => {
                call_on_error(handler_address, 1, &e.to_string());
                return Err(());
            }
            None => {
                // End of stream
                signal_end_of_stream(handler_address);
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
        additional_metadata: std::ptr::null(),
        private_data,
    }
}

unsafe extern "C" fn producer_request(self_: *mut FFI_ArrowAsyncProducer, n: i64) {
    if self_.is_null() {
        return;
    }
    let producer = &*self_;
    if producer.private_data.is_null() {
        return;
    }
    let state = &*(producer.private_data as *const ProducerState);
    if n > 0 {
        state.request(n as u64);
    }
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

/// Owns this implementation's producer while allowing the async driver to move
/// between Tokio worker threads. The box keeps the pointer exposed to the
/// consumer stable even when this wrapper moves.
struct SendableProducerGuard(Box<FFI_ArrowAsyncProducer>);

impl SendableProducerGuard {
    fn new(producer: FFI_ArrowAsyncProducer) -> Self {
        Self(Box::new(producer))
    }

    fn address(&mut self) -> usize {
        self.0.as_mut() as *mut FFI_ArrowAsyncProducer as usize
    }
}

// SAFETY: This wrapper is only constructed for the producer defined in this
// module. Its private state is an Arc containing atomics and an AtomicWaker,
// its callbacks are thread-safe, and the boxed producer has a stable address.
unsafe impl Send for SendableProducerGuard {}

impl Drop for SendableProducerGuard {
    fn drop(&mut self) {
        unsafe { producer_release(self.0.as_mut()) };
    }
}

/// Owns the producer-side right to invoke and release one foreign handler.
/// Dropping an armed guard reports cancellation and releases it exactly once.
struct SendableHandlerGuard {
    handler: NonNull<FFI_ArrowAsyncDeviceStreamHandler>,
    armed: bool,
}

impl SendableHandlerGuard {
    fn new(handler: *mut FFI_ArrowAsyncDeviceStreamHandler) -> Option<Self> {
        NonNull::new(handler).map(|handler| Self {
            handler,
            armed: true,
        })
    }

    fn address(&self) -> usize {
        self.handler.as_ptr() as usize
    }

    fn release(&mut self) {
        if !self.armed {
            return;
        }

        // Disarm before invoking foreign code: after release returns the
        // handler may already have been freed.
        self.armed = false;
        let handler = self.handler.as_ptr();
        let release = unsafe { (*handler).release };
        if let Some(release) = release {
            unsafe { release(handler) };
        }
    }
}

// SAFETY: The guard is uniquely owned by the producer task. Arrow requires
// handler callbacks to be serialized but permits consecutive callbacks to run
// on different threads. No reference into the handler is retained across an
// await or a thread migration.
unsafe impl Send for SendableHandlerGuard {}

impl Drop for SendableHandlerGuard {
    fn drop(&mut self) {
        if self.armed {
            call_on_error(
                self.address(),
                libc::ECANCELED,
                "Stream driver dropped unexpectedly",
            );
            self.release();
        }
    }
}

/// Keeps the handler alive through its release callback, then drops the
/// producer. Field order is significant because Rust drops fields in
/// declaration order.
struct SendableDriverResources {
    handler: SendableHandlerGuard,
    producer: SendableProducerGuard,
}

impl SendableDriverResources {
    fn handler_address(&self) -> usize {
        self.handler.address()
    }

    fn producer_address(&mut self) -> usize {
        self.producer.address()
    }

    fn release_handler(&mut self) {
        self.handler.release();
    }
}

/// Signal end of stream by calling on_next_task with NULL.
fn signal_end_of_stream(handler_address: usize) {
    let handler = handler_address as *mut FFI_ArrowAsyncDeviceStreamHandler;
    let on_next_task = unsafe { (*handler).on_next_task };
    if let Some(on_next_task) = on_next_task {
        // Per Arrow spec: pass NULL task pointer to signal end of stream
        unsafe {
            on_next_task(handler, std::ptr::null_mut(), std::ptr::null());
        }
    }
}

enum SendBatchError {
    HandlerRejected,
    Producer(String),
}

fn send_batch_to_handler(handler_address: usize, batch: RecordBatch) -> Result<(), SendBatchError> {
    let handler = handler_address as *mut FFI_ArrowAsyncDeviceStreamHandler;
    let on_next_task = unsafe { (*handler).on_next_task };
    let Some(on_next_task) = on_next_task else {
        return Err(SendBatchError::Producer(
            "on_next_task callback is null".to_string(),
        ));
    };

    // Create a task that wraps the batch
    // The handler moves the task contents if it accepts the callback. Keep the
    // stack wrapper from dropping the moved private data in that case.
    let mut task = create_async_task(batch).map_err(SendBatchError::Producer)?;
    let ret = unsafe { on_next_task(handler, &mut task, std::ptr::null()) };
    if ret != 0 {
        return Err(SendBatchError::HandlerRejected);
    }
    std::mem::forget(task);
    Ok(())
}

fn call_on_error(handler_address: usize, code: c_int, message: &str) {
    let handler = handler_address as *mut FFI_ArrowAsyncDeviceStreamHandler;
    let on_error = unsafe { (*handler).on_error };
    if let Some(on_error) = on_error {
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
    if self_.is_null() {
        return 1;
    }

    let task = &mut *self_;
    if task.private_data.is_null() {
        return 1;
    }

    // extract_data is the task's one-shot cleanup callback. Take ownership of
    // the private allocation before doing any work so all exit paths release it.
    let mut private = Box::from_raw(task.private_data as *mut AsyncTaskPrivate);
    task.private_data = null_mut();
    task.extract_data = None;

    // A null output means the consumer is abandoning the task.
    if out.is_null() {
        return 0;
    }

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
        // ArrowAsyncTask private_data is producer-owned. Consumers must use
        // extract_data(NULL), even when abandoning the task, so foreign tasks
        // are cleaned up by the producer that created them.
        if let Some(extract_data) = self.extract_data.take() {
            unsafe {
                extract_data(self, null_mut());
            }
        }
    }
}
