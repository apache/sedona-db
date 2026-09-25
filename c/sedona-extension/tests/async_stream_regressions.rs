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

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::RecordBatchStream;
use futures::channel::oneshot;
use futures::{Stream, StreamExt};
use sedona_extension::export_sendable_record_batch_stream::drive_stream_to_handler;
use sedona_extension::extension::{
    FFI_ArrowAsyncDeviceStreamHandler, FFI_ArrowAsyncProducer, FFI_ArrowAsyncTask,
    FFI_ArrowDeviceArray, ARROW_DEVICE_CPU,
};
use sedona_extension::import_sendable_record_batch_stream::ImportedAsyncDeviceStream;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int32,
        false,
    )]))
}

struct PendingStream {
    schema: SchemaRef,
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Stream for PendingStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        Poll::Pending
    }
}

impl RecordBatchStream for PendingStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Drop for PendingStream {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct DelayedEmptyStream {
    schema: SchemaRef,
    ready: oneshot::Receiver<()>,
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Stream for DelayedEmptyStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.polls.fetch_add(1, Ordering::SeqCst);
        match Pin::new(&mut self.ready).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(_) => Poll::Ready(None),
        }
    }
}

impl RecordBatchStream for DelayedEmptyStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Drop for DelayedEmptyStream {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn request_racing_waker_registration_is_not_lost() {
    use std::task::{RawWaker, RawWakerVTable, Waker};

    struct WakeState {
        producer: AtomicPtr<FFI_ArrowAsyncProducer>,
        wakes: AtomicUsize,
    }

    unsafe fn clone_waker(data: *const ()) -> RawWaker {
        let state = &*(data as *const WakeState);
        let producer = state.producer.load(Ordering::SeqCst);
        if !producer.is_null() {
            ((*producer).request.unwrap())(producer, 1);
        }
        Arc::increment_strong_count(data as *const WakeState);
        RawWaker::new(data, &VTABLE)
    }

    unsafe fn wake(data: *const ()) {
        let state = Arc::from_raw(data as *const WakeState);
        state.wakes.fetch_add(1, Ordering::SeqCst);
    }

    unsafe fn wake_by_ref(data: *const ()) {
        (*(data as *const WakeState))
            .wakes
            .fetch_add(1, Ordering::SeqCst);
    }

    unsafe fn drop_waker(data: *const ()) {
        drop(Arc::from_raw(data as *const WakeState));
    }

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone_waker, wake, wake_by_ref, drop_waker);

    unsafe extern "C" fn on_schema(
        handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
        schema: *mut arrow_array::ffi::FFI_ArrowSchema,
    ) -> std::ffi::c_int {
        drop(arrow_array::ffi::FFI_ArrowSchema::from_raw(schema));
        (*((*handler).private_data as *const WakeState))
            .producer
            .store((*handler).producer, Ordering::SeqCst);
        0
    }

    let wake_state = Arc::new(WakeState {
        producer: AtomicPtr::new(std::ptr::null_mut()),
        wakes: AtomicUsize::new(0),
    });
    let waker = unsafe {
        Waker::from_raw(RawWaker::new(
            Arc::into_raw(wake_state.clone()).cast(),
            &VTABLE,
        ))
    };
    let mut handler = FFI_ArrowAsyncDeviceStreamHandler {
        on_schema: Some(on_schema),
        on_next_task: None,
        on_error: None,
        release: None,
        producer: std::ptr::null_mut(),
        private_data: Arc::as_ptr(&wake_state) as *mut _,
    };
    let source_polls = Arc::new(AtomicUsize::new(0));
    let source = Box::pin(PendingStream {
        schema: schema(),
        polls: source_polls.clone(),
        drops: Arc::new(AtomicUsize::new(0)),
    });
    let mut driver = Box::pin(unsafe { drive_stream_to_handler(source, &mut handler) });
    let mut cx = Context::from_waker(&waker);
    let result = driver.as_mut().poll(&mut cx);

    assert!(
        result.is_ready()
            || wake_state.wakes.load(Ordering::SeqCst) > 0
            || source_polls.load(Ordering::SeqCst) > 0,
        "request must schedule another poll or be observed before returning Pending"
    );
}

#[test]
fn cancellation_interrupts_pending_source() {
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let source = Box::pin(PendingStream {
        schema: schema(),
        polls: polls.clone(),
        drops: drops.clone(),
    });
    let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);
    let mut driver = Box::pin(unsafe { drive_stream_to_handler(source, handler.as_ptr()) });
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());

    assert!(driver.as_mut().poll(&mut cx).is_pending());
    assert!(Pin::new(&mut consumer).poll_next(&mut cx).is_pending());
    assert!(driver.as_mut().poll(&mut cx).is_pending());
    assert_eq!(polls.load(Ordering::SeqCst), 1);
    consumer.cancel();

    assert!(driver.as_mut().poll(&mut cx).is_ready());
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cancellation_before_first_poll_does_not_poll_source() {
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let source = Box::pin(PendingStream {
        schema: schema(),
        polls: polls.clone(),
        drops: drops.clone(),
    });
    let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);
    consumer.cancel();
    // SAFETY: into_raw transfers the handler to this driver's sole ownership.
    unsafe { drive_stream_to_handler(source, handler.into_raw()) }.await;

    assert!(consumer.next().await.is_none());
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn prefetch_one_requests_a_batch() {
    let polls = Arc::new(AtomicUsize::new(0));
    let source = Box::pin(PendingStream {
        schema: schema(),
        polls: polls.clone(),
        drops: Arc::new(AtomicUsize::new(0)),
    });
    let (mut consumer, handler) = ImportedAsyncDeviceStream::new(1);
    let mut driver = Box::pin(unsafe { drive_stream_to_handler(source, handler.as_ptr()) });
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());

    assert!(driver.as_mut().poll(&mut cx).is_pending());
    assert!(Pin::new(&mut consumer).poll_next(&mut cx).is_pending());
    assert!(driver.as_mut().poll(&mut cx).is_pending());
    assert_eq!(polls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn delayed_empty_stream_completes_without_a_batch() {
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let (ready_tx, ready_rx) = oneshot::channel();
    let source = Box::pin(DelayedEmptyStream {
        schema: schema(),
        ready: ready_rx,
        polls: polls.clone(),
        drops: drops.clone(),
    });
    let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);

    let consumption = async {
        assert!(consumer.next().await.is_none());
    };
    let release_source = async {
        // Ensure the driver and consumer both observe the pending phase before
        // allowing the source to complete without producing a batch.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        ready_tx.send(()).unwrap();
    };

    futures::join!(
        consumption,
        unsafe { drive_stream_to_handler(source, handler.as_ptr()) },
        release_source
    );

    assert!(polls.load(Ordering::SeqCst) >= 2);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

static FOREIGN_TASK_CLEANUPS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn foreign_request(_: *mut FFI_ArrowAsyncProducer, _: i64) {}
unsafe extern "C" fn foreign_cancel(_: *mut FFI_ArrowAsyncProducer) {}
unsafe extern "C" fn foreign_extract(
    _: *mut FFI_ArrowAsyncTask,
    out: *mut FFI_ArrowDeviceArray,
) -> std::ffi::c_int {
    assert!(out.is_null());
    FOREIGN_TASK_CLEANUPS.fetch_add(1, Ordering::SeqCst);
    0
}

#[test]
fn abandoning_a_foreign_task_calls_extract_null() {
    FOREIGN_TASK_CLEANUPS.store(0, Ordering::SeqCst);
    let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);
    let mut producer = FFI_ArrowAsyncProducer {
        device_type: ARROW_DEVICE_CPU,
        request: Some(foreign_request),
        cancel: Some(foreign_cancel),
        additional_metadata: std::ptr::null(),
        private_data: std::ptr::null_mut(),
    };
    let ptr = handler.as_ptr();
    unsafe {
        (*ptr).producer = &mut producer;
    }
    let mut ffi_schema = arrow_array::ffi::FFI_ArrowSchema::try_from(schema().as_ref()).unwrap();
    assert_eq!(
        unsafe { ((*ptr).on_schema.unwrap())(ptr, &mut ffi_schema) },
        0
    );
    assert!(ffi_schema.release.is_none());
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());
    assert!(Pin::new(&mut consumer).poll_next(&mut cx).is_pending());

    let mut task = FFI_ArrowAsyncTask {
        extract_data: Some(foreign_extract),
        private_data: std::ptr::null_mut(),
    };
    assert_eq!(
        unsafe { ((*ptr).on_next_task.unwrap())(ptr, &mut task, std::ptr::null()) },
        0
    );
    std::mem::forget(task);
    drop(consumer);
    unsafe {
        ((*ptr).release.unwrap())(ptr);
    }

    assert_eq!(FOREIGN_TASK_CLEANUPS.load(Ordering::SeqCst), 1);
}

struct ReadyStream {
    schema: SchemaRef,
    batch: Option<RecordBatch>,
}

impl Stream for ReadyStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.batch.take().map(Ok))
    }
}

impl RecordBatchStream for ReadyStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[derive(Default)]
struct RejectingHandlerState {
    schema_calls: AtomicUsize,
    next_calls: AtomicUsize,
    error_calls: AtomicUsize,
    release_calls: AtomicUsize,
}

unsafe extern "C" fn reject_schema(
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
    schema: *mut arrow_array::ffi::FFI_ArrowSchema,
) -> std::ffi::c_int {
    drop(arrow_array::ffi::FFI_ArrowSchema::from_raw(schema));
    let state = &*((*handler).private_data as *const RejectingHandlerState);
    state.schema_calls.fetch_add(1, Ordering::SeqCst);
    libc::EINVAL
}

unsafe extern "C" fn accept_schema_and_request_one(
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
    schema: *mut arrow_array::ffi::FFI_ArrowSchema,
) -> std::ffi::c_int {
    drop(arrow_array::ffi::FFI_ArrowSchema::from_raw(schema));
    let state = &*((*handler).private_data as *const RejectingHandlerState);
    state.schema_calls.fetch_add(1, Ordering::SeqCst);
    let producer = (*handler).producer;
    ((*producer).request.unwrap())(producer, 1);
    0
}

unsafe extern "C" fn reject_task(
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
    _task: *mut FFI_ArrowAsyncTask,
    _metadata: *const std::ffi::c_char,
) -> std::ffi::c_int {
    let state = &*((*handler).private_data as *const RejectingHandlerState);
    state.next_calls.fetch_add(1, Ordering::SeqCst);
    libc::EINVAL
}

unsafe extern "C" fn count_error(
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
    _code: std::ffi::c_int,
    _message: *const std::ffi::c_char,
    _metadata: *const std::ffi::c_char,
) {
    let state = &*((*handler).private_data as *const RejectingHandlerState);
    state.error_calls.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn count_release(handler: *mut FFI_ArrowAsyncDeviceStreamHandler) {
    let state = &*((*handler).private_data as *const RejectingHandlerState);
    state.release_calls.fetch_add(1, Ordering::SeqCst);
    (*handler).release = None;
}

#[test]
fn unpolled_sendable_driver_releases_handler_once() {
    fn assert_send<T: Send>(_: &T) {}

    let state = RejectingHandlerState::default();
    let mut handler = FFI_ArrowAsyncDeviceStreamHandler {
        on_schema: Some(reject_schema),
        on_next_task: Some(reject_task),
        on_error: Some(count_error),
        release: Some(count_release),
        producer: std::ptr::null_mut(),
        private_data: (&state as *const RejectingHandlerState).cast_mut().cast(),
    };
    let source_drops = Arc::new(AtomicUsize::new(0));
    let source = Box::pin(PendingStream {
        schema: schema(),
        polls: Arc::new(AtomicUsize::new(0)),
        drops: source_drops.clone(),
    });

    let driver = unsafe { drive_stream_to_handler(source, &mut handler) };
    assert_send(&driver);
    drop(driver);

    assert_eq!(state.schema_calls.load(Ordering::SeqCst), 0);
    assert_eq!(state.next_calls.load(Ordering::SeqCst), 0);
    assert_eq!(state.error_calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.release_calls.load(Ordering::SeqCst), 1);
    assert_eq!(source_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn runtime_shutdown_before_first_poll_releases_handler_once() {
    let state = RejectingHandlerState::default();
    let mut handler = FFI_ArrowAsyncDeviceStreamHandler {
        on_schema: Some(reject_schema),
        on_next_task: Some(reject_task),
        on_error: Some(count_error),
        release: Some(count_release),
        producer: std::ptr::null_mut(),
        private_data: (&state as *const RejectingHandlerState).cast_mut().cast(),
    };
    let source_drops = Arc::new(AtomicUsize::new(0));
    let source = Box::pin(PendingStream {
        schema: schema(),
        polls: Arc::new(AtomicUsize::new(0)),
        drops: source_drops.clone(),
    });
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    runtime.spawn(unsafe { drive_stream_to_handler(source, &mut handler) });
    drop(runtime);

    assert_eq!(state.schema_calls.load(Ordering::SeqCst), 0);
    assert_eq!(state.next_calls.load(Ordering::SeqCst), 0);
    assert_eq!(state.error_calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.release_calls.load(Ordering::SeqCst), 1);
    assert_eq!(source_drops.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn rejected_schema_is_consumed_and_only_followed_by_release() {
    let state = RejectingHandlerState::default();
    let mut handler = FFI_ArrowAsyncDeviceStreamHandler {
        on_schema: Some(reject_schema),
        on_next_task: Some(reject_task),
        on_error: Some(count_error),
        release: Some(count_release),
        producer: std::ptr::null_mut(),
        private_data: (&state as *const RejectingHandlerState).cast_mut().cast(),
    };
    let source = Box::pin(ReadyStream {
        schema: schema(),
        batch: None,
    });

    unsafe { drive_stream_to_handler(source, &mut handler) }.await;

    assert_eq!(state.schema_calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.next_calls.load(Ordering::SeqCst), 0);
    assert_eq!(state.error_calls.load(Ordering::SeqCst), 0);
    assert_eq!(state.release_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn rejected_task_is_only_followed_by_release() {
    let state = RejectingHandlerState::default();
    let mut handler = FFI_ArrowAsyncDeviceStreamHandler {
        on_schema: Some(accept_schema_and_request_one),
        on_next_task: Some(reject_task),
        on_error: Some(count_error),
        release: Some(count_release),
        producer: std::ptr::null_mut(),
        private_data: (&state as *const RejectingHandlerState).cast_mut().cast(),
    };
    let batch = RecordBatch::try_new(schema(), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let source = Box::pin(ReadyStream {
        schema: schema(),
        batch: Some(batch),
    });

    unsafe { drive_stream_to_handler(source, &mut handler) }.await;

    assert_eq!(state.schema_calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.next_calls.load(Ordering::SeqCst), 1);
    assert_eq!(state.error_calls.load(Ordering::SeqCst), 0);
    assert_eq!(state.release_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn roundtrip_preserves_schema_metadata() {
    let metadata = HashMap::from([("review_source".to_owned(), "table-v1".to_owned())]);
    let schema = Arc::new(schema().as_ref().clone().with_metadata(metadata));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
    let source = Box::pin(ReadyStream {
        schema: schema.clone(),
        batch: Some(batch),
    });
    let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);
    let consumption = async {
        let mut batches = vec![];
        while let Some(batch) = consumer.next().await {
            batches.push(batch.unwrap());
        }
        (batches, consumer.schema())
    };

    let ((batches, actual_schema), _) = futures::join!(consumption, unsafe {
        drive_stream_to_handler(source, handler.as_ptr())
    });

    assert_eq!(actual_schema, schema);
    assert_eq!(batches[0].schema(), schema);
}
