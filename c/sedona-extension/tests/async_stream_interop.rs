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

// Foreign callbacks follow the C ABI without Sedona-specific cleanup behavior.
use std::ffi::c_int;
use std::pin::Pin;
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow_array::ffi::{to_ffi, FFI_ArrowSchema};
use arrow_array::{Int32Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_execution::TaskContext;
use datafusion_physical_plan::{empty::EmptyExec, ExecutionPlan};
use futures::task::{waker, ArcWake};
use futures::{Stream, StreamExt};
use sedona_extension::execution_plan::{ExportedExecutionPlan, ImportedSedonaCExec};
use sedona_extension::extension::{
    FFI_ArrowAsyncDeviceStreamHandler, FFI_ArrowAsyncProducer, FFI_ArrowAsyncTask,
    FFI_ArrowDeviceArray, ARROW_DEVICE_CPU,
};
use sedona_extension::import_sendable_record_batch_stream::ImportedAsyncDeviceStream;
use sedona_extension::runtime::RuntimeHandle;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, false)]))
}
unsafe extern "C" fn request(_: *mut FFI_ArrowAsyncProducer, _: i64) {}
unsafe extern "C" fn cancel(_: *mut FFI_ArrowAsyncProducer) {}
fn producer() -> FFI_ArrowAsyncProducer {
    FFI_ArrowAsyncProducer {
        device_type: ARROW_DEVICE_CPU,
        request: Some(request),
        cancel: Some(cancel),
        additional_metadata: null(),
        private_data: null_mut(),
    }
}
unsafe fn connect(
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
    producer: &mut FFI_ArrowAsyncProducer,
) {
    (*handler).producer = producer;
    let mut exported = FFI_ArrowSchema::try_from(schema().as_ref()).unwrap();
    assert_eq!(((*handler).on_schema.unwrap())(handler, &mut exported), 0);
    assert!(exported.release.is_none());
}

#[test]
fn imported_schema_must_mark_original_released() {
    let (_consumer, handler) = ImportedAsyncDeviceStream::new(2);
    let mut producer = producer();
    let ptr = handler.as_ptr();
    unsafe {
        (*ptr).producer = &mut producer;
    }
    let mut exported = FFI_ArrowSchema::try_from(schema().as_ref()).unwrap();
    unsafe {
        assert_eq!(((*ptr).on_schema.unwrap())(ptr, &mut exported), 0);
    }
    let released = exported.release.is_none();
    // Keep a regression from double-freeing before the assertion can report it.
    std::mem::forget(exported);
    unsafe {
        ((*ptr).release.unwrap())(ptr);
    }
    assert!(
        released,
        "C++ SchemaExportGuard will release already-freed schema contents"
    );
}

struct ForeignTask {
    calls: AtomicUsize,
    fail: bool,
}
unsafe extern "C" fn extract(
    task: *mut FFI_ArrowAsyncTask,
    out: *mut FFI_ArrowDeviceArray,
) -> c_int {
    // Like Arrow C++ extract_data, leave the task's fields untouched.
    // Keep the counter allocation live so a double invocation is observable safely.
    let state = &*((*task).private_data as *const ForeignTask);
    state.calls.fetch_add(1, Ordering::SeqCst);
    if out.is_null() {
        return 0;
    }
    if state.fail {
        return libc::EINVAL;
    }
    let batch = RecordBatch::try_new(schema(), vec![Arc::new(Int32Array::from(vec![42]))]).unwrap();
    let data = StructArray::from(batch).into();
    let (array, _) = to_ffi(&data).unwrap();
    std::ptr::write(out, FFI_ArrowDeviceArray::from(array));
    0
}
fn check_extract_once(fail: bool) {
    let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);
    let mut producer = producer();
    let ptr = handler.as_ptr();
    unsafe {
        connect(ptr, &mut producer);
    }
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());
    assert!(Pin::new(&mut consumer).poll_next(&mut cx).is_pending());
    let state = ForeignTask {
        calls: AtomicUsize::new(0),
        fail,
    };
    let mut task = FFI_ArrowAsyncTask {
        extract_data: Some(extract),
        private_data: (&state as *const ForeignTask).cast_mut().cast(),
    };
    unsafe {
        assert_eq!(((*ptr).on_next_task.unwrap())(ptr, &mut task, null()), 0);
    }
    std::mem::forget(task);
    let result = Pin::new(&mut consumer).poll_next(&mut cx);
    match result {
        Poll::Ready(Some(result)) => {
            assert_eq!(result.is_err(), fail);
            if let Ok(batch) = result {
                assert_eq!(batch.num_rows(), 1);
                let values = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                assert_eq!(values.value(0), 42);
            }
        }
        other => panic!("expected task result, got {other:?}"),
    }
    unsafe {
        ((*ptr).release.unwrap())(ptr);
    }
    assert_eq!(
        state.calls.load(Ordering::SeqCst),
        1,
        "extract_data must be called exactly once"
    );
}
#[test]
fn foreign_successful_task_is_extracted_once() {
    check_extract_once(false);
}
#[test]
fn foreign_failed_task_is_extracted_once() {
    check_extract_once(true);
}

#[derive(Default)]
struct WakeCounter(AtomicUsize);
impl ArcWake for WakeCounter {
    fn wake_by_ref(this: &Arc<Self>) {
        this.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn release_without_eos_must_wake_waiting_consumer() {
    let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);
    let mut producer = producer();
    let ptr = handler.as_ptr();
    unsafe {
        connect(ptr, &mut producer);
    }
    let wakes = Arc::new(WakeCounter::default());
    let waker = waker(wakes.clone());
    let mut cx = Context::from_waker(&waker);
    assert!(Pin::new(&mut consumer).poll_next(&mut cx).is_pending());
    consumer.cancel();
    wakes.0.store(0, Ordering::SeqCst);
    // A conforming producer may finish cancellation with release alone.
    unsafe {
        ((*ptr).release.unwrap())(ptr);
    }
    assert!(
        wakes.0.load(Ordering::SeqCst) > 0,
        "consumer is left asleep after release"
    );
    assert!(matches!(
        Pin::new(&mut consumer).poll_next(&mut cx),
        Poll::Ready(None)
    ));
}

#[test]
fn stream_keeps_producer_runtime_alive_after_plan_drop() {
    let runtime = Arc::new(RuntimeHandle::new(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
    ));
    let ctx = Arc::new(TaskContext::default());
    let exported = ExportedExecutionPlan::new(
        Arc::new(EmptyExec::new(schema())),
        ctx.clone(),
        runtime.clone(),
    );
    let imported = ImportedSedonaCExec::try_new(exported.into())
        .unwrap()
        .with_async_execution(true);
    let mut consumer = imported.execute(0, ctx).unwrap();
    let weak = Arc::downgrade(&runtime);
    drop(runtime);
    drop(imported);

    // Recover a temporary driver for this current-thread runtime. The task
    // itself must own the runtime before and throughout stream consumption.
    let runtime = weak.upgrade().expect("stream lost its producer runtime");
    runtime.block_on(async {
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(2), consumer.next())
                .await
                .unwrap()
                .is_none()
        );
    });
    drop(consumer);
    drop(runtime);
    assert!(
        weak.upgrade().is_none(),
        "finished stream retained its runtime"
    );
}

#[test]
fn dropping_stream_releases_retained_runtime() {
    let runtime = Arc::new(RuntimeHandle::new(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
    ));
    let ctx = Arc::new(TaskContext::default());
    let exported = ExportedExecutionPlan::new(
        Arc::new(EmptyExec::new(schema())),
        ctx.clone(),
        runtime.clone(),
    );
    let imported = ImportedSedonaCExec::try_new(exported.into())
        .unwrap()
        .with_async_execution(true);
    let consumer = imported.execute(0, ctx).unwrap();
    let weak = Arc::downgrade(&runtime);
    drop(imported);
    drop(consumer);
    runtime.block_on(async {
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while Arc::strong_count(&runtime) > 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled driver retained its runtime");
    });
    drop(runtime);
    assert!(weak.upgrade().is_none());
}

#[test]
fn queued_batches_are_drained_before_error_or_eof() {
    for fail in [false, true] {
        let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);
        let mut producer = producer();
        let ptr = handler.as_ptr();
        unsafe {
            connect(ptr, &mut producer);
        }
        let mut cx = Context::from_waker(futures::task::noop_waker_ref());
        assert!(Pin::new(&mut consumer).poll_next(&mut cx).is_pending());
        let state = ForeignTask {
            calls: AtomicUsize::new(0),
            fail: false,
        };
        for _ in 0..2 {
            let mut task = FFI_ArrowAsyncTask {
                extract_data: Some(extract),
                private_data: (&state as *const ForeignTask).cast_mut().cast(),
            };
            unsafe {
                assert_eq!(((*ptr).on_next_task.unwrap())(ptr, &mut task, null()), 0);
            }
            std::mem::forget(task);
        }
        unsafe {
            if fail {
                ((*ptr).on_error.unwrap())(ptr, libc::EIO, c"expected error".as_ptr(), null());
            } else {
                assert_eq!(((*ptr).on_next_task.unwrap())(ptr, null_mut(), null()), 0);
            }
            ((*ptr).release.unwrap())(ptr);
        }
        // Extract tasks after the producer has already released the handler.
        for _ in 0..2 {
            assert!(matches!(
                Pin::new(&mut consumer).poll_next(&mut cx),
                Poll::Ready(Some(Ok(_)))
            ));
        }
        if fail {
            let Poll::Ready(Some(Err(error))) = Pin::new(&mut consumer).poll_next(&mut cx) else {
                panic!("producer error was lost");
            };
            assert!(error.to_string().contains("expected error"));
        }
        assert!(matches!(
            Pin::new(&mut consumer).poll_next(&mut cx),
            Poll::Ready(None)
        ));
        assert_eq!(state.calls.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn cancellation_before_schema_rejects_the_connection() {
    let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);
    consumer.cancel();
    consumer.cancel();
    let mut producer = producer();
    let ptr = handler.as_ptr();
    unsafe {
        (*ptr).producer = &mut producer;
    }
    let mut exported = FFI_ArrowSchema::try_from(schema().as_ref()).unwrap();
    let result = unsafe { ((*ptr).on_schema.unwrap())(ptr, &mut exported) };
    let released = exported.release.is_none();
    std::mem::forget(exported);
    // A rejected on_schema must only be followed by release.
    unsafe {
        ((*ptr).release.unwrap())(ptr);
    }
    assert_eq!(result, libc::ECANCELED);
    assert!(released, "rejected schema was not consumed");
    let mut cx = Context::from_waker(futures::task::noop_waker_ref());
    assert!(matches!(
        Pin::new(&mut consumer).poll_next(&mut cx),
        Poll::Ready(None)
    ));
}

#[test]
fn racing_error_must_not_be_reported_as_clean_eof() {
    let (send, receive) = std::sync::mpsc::channel::<usize>();
    let (done_tx, done_rx) = std::sync::mpsc::sync_channel(0);
    let producer = std::thread::spawn(move || {
        while let Ok(address) = receive.recv() {
            let ptr = address as *mut FFI_ArrowAsyncDeviceStreamHandler;
            unsafe {
                ((*ptr).on_error.unwrap())(ptr, libc::EIO, c"expected error".as_ptr(), null());
                ((*ptr).release.unwrap())(ptr);
            }
            done_tx.send(()).unwrap();
        }
    });
    let mut lost = 0;
    for _ in 0..50_000 {
        let (mut consumer, handler) = ImportedAsyncDeviceStream::new(2);
        let address = handler.into_raw() as usize;
        send.send(address).unwrap();
        let mut cx = Context::from_waker(futures::task::noop_waker_ref());
        loop {
            match Pin::new(&mut consumer).poll_next(&mut cx) {
                Poll::Ready(None) => {
                    lost += 1;
                    break;
                }
                Poll::Ready(Some(Err(_))) => break,
                Poll::Ready(Some(Ok(_))) => panic!("unexpected batch"),
                Poll::Pending => std::hint::spin_loop(),
            }
        }
        done_rx.recv().unwrap();
    }
    drop(send);
    producer.join().unwrap();
    assert_eq!(
        lost, 0,
        "on_error raced with the ended flag and became clean EOF"
    );
}
