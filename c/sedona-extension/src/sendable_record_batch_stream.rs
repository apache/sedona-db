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

use std::ffi::{c_int, c_void, CString};
use std::ptr::null_mut;

use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::RecordBatch;
use futures::StreamExt;

use datafusion_execution::SendableRecordBatchStream;

use crate::extension_ffi::{
    FFI_ArrowAsyncDeviceStreamHandler, FFI_ArrowAsyncTask, FFI_ArrowDeviceArray, ARROW_DEVICE_CPU,
};

/// Drives a [`SendableRecordBatchStream`] and pushes results to an [`FFI_ArrowAsyncDeviceStreamHandler`].
///
/// This function consumes the stream and calls the handler's callbacks:
/// - `on_schema` is called first with the stream's schema
/// - `on_next_task` is called for each record batch
/// - `on_error` is called if an error occurs
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
/// // Assuming you have a SendableRecordBatchStream and a handler
/// drive_stream_to_handler(stream, handler).await;
/// ```
pub async fn drive_stream_to_handler(
    mut stream: SendableRecordBatchStream,
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
) {
    if handler.is_null() {
        return;
    }

    let handler_ref = unsafe { &mut *handler };

    // First, send the schema
    let schema = stream.schema();
    let ffi_schema = match FFI_ArrowSchema::try_from(schema.as_ref()) {
        Ok(s) => s,
        Err(e) => {
            call_on_error(handler_ref, 1, &e.to_string());
            return;
        }
    };

    if let Some(on_schema) = handler_ref.on_schema {
        // We need to box the schema so it lives long enough
        let mut boxed_schema = Box::new(ffi_schema);
        let ret = unsafe { on_schema(handler, boxed_schema.as_mut()) };
        if ret != 0 {
            call_on_error(handler_ref, ret, "on_schema callback failed");
            return;
        }
        // The handler now owns the schema, so we forget it
        std::mem::forget(boxed_schema);
    }

    // Stream batches
    while let Some(result) = stream.next().await {
        match result {
            Ok(batch) => {
                if let Err(e) = send_batch_to_handler(handler_ref, handler, batch) {
                    call_on_error(handler_ref, 1, &e);
                    return;
                }
            }
            Err(e) => {
                call_on_error(handler_ref, 1, &e.to_string());
                return;
            }
        }
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
    let (ffi_array, _ffi_schema) = match arrow_array::ffi::to_ffi(&struct_array.into()) {
        Ok(result) => result,
        Err(_) => return 1,
    };

    // Create device array (CPU)
    let device_array = FFI_ArrowDeviceArray {
        array: ffi_array,
        device_id: -1,
        device_type: ARROW_DEVICE_CPU,
        sync_event: null_mut(),
    };

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
