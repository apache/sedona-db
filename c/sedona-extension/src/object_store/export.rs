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

use std::ffi::c_int;
use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use arrow_array::RecordBatch;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::{StreamExt, TryStreamExt};
use object_store::{path::Path, GetOptions, ObjectStore};

use super::bounded::{failure, invalid};
use super::ffi::*;
use super::wire::{self, ListEntry, ReadMetadata, Request, WireError};
use crate::export_sendable_record_batch_stream::drive_stream_to_handler;
use crate::extension::{FFI_ArrowAsyncDeviceStreamHandler, SedonaCError};
use crate::runtime::RuntimeHandle;

/// Export a configured store without exposing its credentials or Rust ABI.
/// The runtime must be actively driven (normally a multi-thread runtime).
pub struct ExportedObjectStore {
    store: Arc<dyn ObjectStore>,
    runtime: Arc<RuntimeHandle>,
}
impl ExportedObjectStore {
    pub fn new(store: Arc<dyn ObjectStore>, runtime: Arc<RuntimeHandle>) -> Self {
        Self { store, runtime }
    }
    pub fn into_ffi(self) -> SedonaCObjectStore {
        SedonaCObjectStore {
            abi_version: OBJECT_STORE_ABI_VERSION,
            capabilities: OBJECT_STORE_READ | OBJECT_STORE_LIST | OBJECT_STORE_LIST_WITH_DELIMITER,
            call: Some(call),
            release: Some(release_store),
            private_data: Box::into_raw(Box::new(self)).cast(),
            reserved: [0; 4],
        }
    }
}

struct ExportedStream {
    stream: Option<SendableRecordBatchStream>,
    runtime: Arc<RuntimeHandle>,
}
fn export_stream(
    stream: SendableRecordBatchStream,
    runtime: Arc<RuntimeHandle>,
) -> SedonaCObjectStoreStream {
    SedonaCObjectStoreStream {
        start: Some(start_stream),
        release: Some(release_stream),
        private_data: Box::into_raw(Box::new(ExportedStream {
            stream: Some(stream),
            runtime,
        }))
        .cast(),
    }
}
fn export_buffer(bytes: Vec<u8>) -> SedonaCObjectStoreBuffer {
    let bytes = Box::new(bytes);
    SedonaCObjectStoreBuffer {
        data: bytes.as_ptr(),
        len: bytes.len(),
        release: Some(release_buffer),
        private_data: Box::into_raw(bytes).cast(),
    }
}
unsafe extern "C" fn release_buffer(value: *mut SedonaCObjectStoreBuffer) {
    let value = &mut *value;
    drop(Box::from_raw(value.private_data.cast::<Vec<u8>>()));
    value.data = std::ptr::null();
    value.len = 0;
    value.private_data = std::ptr::null_mut();
    value.release = None;
}
unsafe extern "C" fn release_stream(value: *mut SedonaCObjectStoreStream) {
    let value = &mut *value;
    drop(Box::from_raw(value.private_data.cast::<ExportedStream>()));
    value.private_data = std::ptr::null_mut();
    value.release = None;
    value.start = None;
}
unsafe extern "C" fn release_store(value: *mut SedonaCObjectStore) {
    let value = &mut *value;
    drop(Box::from_raw(
        value.private_data.cast::<ExportedObjectStore>(),
    ));
    value.private_data = std::ptr::null_mut();
    value.release = None;
    value.call = None;
}
fn write_error(out: *mut SedonaCError, error: object_store::Error) -> c_int {
    let error = WireError::from_error(error);
    unsafe {
        crate::extension::write_ffi_error(out, &error.message);
    }
    error.code
}

unsafe extern "C" fn start_stream(
    value: *mut SedonaCObjectStoreStream,
    handler: *mut FFI_ArrowAsyncDeviceStreamHandler,
    err: *mut SedonaCError,
) -> c_int {
    if value.is_null() || handler.is_null() {
        return write_error(err, invalid("null stream/handler"));
    }
    let value = &mut *value;
    if value.private_data.is_null() {
        return write_error(err, invalid("released stream"));
    }
    let state = &mut *value.private_data.cast::<ExportedStream>();
    let Some(stream) = state.stream.take() else {
        return write_error(err, invalid("stream already started"));
    };
    // This transfers the handler even if the task is never polled. No callbacks
    // can run on the caller's stack: the existing async driver owns all of them.
    let driver = drive_stream_to_handler(stream, handler);
    let runtime = state.runtime.clone();
    state.runtime.spawn(async move {
        let _runtime = runtime;
        driver.await;
    });
    0
}

/// A synchronous wait for a single result. Cancellation aborts the spawned store
/// future; dropping JoinHandle by itself would silently detach the I/O.
fn bounded_get(
    state: &ExportedObjectStore,
    context: &SedonaCObjectStoreCall,
    path: Path,
    options: GetOptions,
) -> object_store::Result<object_store::GetResult> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let store = state.store.clone();
    let runtime = state.runtime.clone();
    let task = state.runtime.spawn(async move {
        let _runtime = runtime;
        let result = store.get_opts(&path, options).await;
        let _ = sender.send(result);
    });
    struct Abort(tokio::task::JoinHandle<()>);
    impl Drop for Abort {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let _abort = Abort(task);
    let deadline = Instant::now()
        .checked_add(Duration::from_millis(context.timeout_ms))
        .ok_or_else(|| invalid("invalid call timeout"))?;
    loop {
        if context
            .is_cancelled
            .is_some_and(|check| unsafe { check(context.private_data) != 0 })
        {
            return Err(failure(
                io::ErrorKind::Interrupted,
                "ObjectStore call cancelled",
            ));
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(failure(
                io::ErrorKind::TimedOut,
                "ObjectStore call timed out",
            ));
        }
        match receiver.recv_timeout(remaining.min(Duration::from_millis(10))) {
            Ok(result) => return result,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(invalid("ObjectStore runtime stopped before completion"))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
    }
}

unsafe extern "C" fn call(
    value: *const SedonaCObjectStore,
    context: *const SedonaCObjectStoreCall,
    request: *const u8,
    request_len: usize,
    out: *mut SedonaCObjectStoreResponse,
    err: *mut SedonaCError,
) -> c_int {
    if value.is_null()
        || context.is_null()
        || out.is_null()
        || request.is_null()
        || request_len > OBJECT_STORE_MAX_BYTES
    {
        return write_error(err, invalid("invalid ObjectStore call arguments"));
    }
    let value = &*value;
    if value.private_data.is_null() {
        return write_error(err, invalid("released ObjectStore"));
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let request = wire::decode(std::slice::from_raw_parts(request, request_len))?;
        execute(
            &*value.private_data.cast::<ExportedObjectStore>(),
            &*context,
            request,
        )
    }));
    match result {
        Ok(Ok(response)) => {
            *out = response;
            0
        }
        Ok(Err(error)) => write_error(err, error),
        Err(_) => write_error(err, invalid("ObjectStore provider panicked")),
    }
}
fn execute(
    state: &ExportedObjectStore,
    context: &SedonaCObjectStoreCall,
    request: Request,
) -> object_store::Result<SedonaCObjectStoreResponse> {
    if context.timeout_ms == 0 {
        return Err(failure(
            io::ErrorKind::TimedOut,
            "ObjectStore call expired before start",
        ));
    }
    if context
        .is_cancelled
        .is_some_and(|check| unsafe { check(context.private_data) != 0 })
    {
        return Err(failure(
            io::ErrorKind::Interrupted,
            "ObjectStore call cancelled before start",
        ));
    }
    match request {
        Request::Get { location, options } => {
            let head = options.head;
            let result = bounded_get(state, context, Path::parse(location)?, options.try_into()?)?;
            let metadata = wire::encode(&ReadMetadata {
                meta: result.meta.clone().into(),
                range_start: result.range.start,
                range_end: result.range.end,
                attributes: ReadMetadata::attributes(&result.attributes)?,
            })?;
            let stream = if head {
                SedonaCObjectStoreStream::default()
            } else {
                let schema = wire::schema("bytes");
                let body = result.into_stream();
                // Split provider chunks to bound the wire representation, preserving
                // ordering. The underlying store may itself allocate larger chunks.
                let chunks = body
                    .map_ok(|bytes| {
                        futures::stream::iter((0..bytes.len()).step_by(OBJECT_STORE_MAX_BYTES).map(
                            move |start| {
                                Ok(bytes.slice(
                                    start..(start + OBJECT_STORE_MAX_BYTES).min(bytes.len()),
                                ))
                            },
                        ))
                    })
                    .try_flatten();
                let encode_schema = schema.clone();
                let batches = chunks.map(move |chunk| match chunk {
                    Ok(bytes) => Ok(wire::payload_batch(
                        encode_schema.clone(),
                        &[bytes.to_vec()],
                    )),
                    Err(error) => Ok(wire::error_batch(encode_schema.clone(), error)),
                });
                export_stream(
                    Box::pin(RecordBatchStreamAdapter::new(schema, batches)),
                    state.runtime.clone(),
                )
            };
            Ok(SedonaCObjectStoreResponse {
                metadata: export_buffer(metadata),
                stream,
            })
        }
        Request::List {
            prefix,
            offset,
            delimiter,
        } => {
            let prefix = prefix.map(Path::parse).transpose()?;
            let offset = offset.map(Path::parse).transpose()?;
            if delimiter && offset.is_some() {
                return Err(invalid("delimiter listing does not support an offset"));
            }
            let store = state.store.clone();
            let entries: futures::stream::BoxStream<'static, object_store::Result<ListEntry>> =
                if delimiter {
                    futures::stream::once(async move {
                        let result = store.list_with_delimiter(prefix.as_ref()).await?;
                        Ok::<_, object_store::Error>(futures::stream::iter(
                            result
                                .objects
                                .into_iter()
                                .map(|meta| Ok(ListEntry::Object { meta: meta.into() }))
                                .chain(result.common_prefixes.into_iter().map(|path| {
                                    Ok(ListEntry::Prefix {
                                        path: path.to_string(),
                                    })
                                })),
                        ))
                    })
                    .try_flatten()
                    .boxed()
                } else {
                    let listing = match &offset {
                        Some(offset) => store.list_with_offset(prefix.as_ref(), offset),
                        None => store.list(prefix.as_ref()),
                    };
                    listing
                        .map_ok(|meta| ListEntry::Object { meta: meta.into() })
                        .boxed()
                };
            let schema = wire::schema("list");
            let encode_schema = schema.clone();
            // ready_chunks batches available items without waiting to fill a page.
            let batches = entries
                .ready_chunks(OBJECT_STORE_MAX_ROWS)
                .flat_map(move |entries| {
                    let mut batches: Vec<datafusion_common::Result<RecordBatch>> = Vec::new();
                    let mut payloads = Vec::new();
                    let mut size = 0;
                    for entry in entries {
                        match entry.and_then(|entry| wire::encode(&entry)) {
                            Ok(payload) => {
                                if size + payload.len() > OBJECT_STORE_MAX_BYTES {
                                    batches.push(Ok(wire::payload_batch(
                                        encode_schema.clone(),
                                        &payloads,
                                    )));
                                    payloads.clear();
                                    size = 0;
                                }
                                size += payload.len();
                                payloads.push(payload);
                            }
                            Err(error) => {
                                if !payloads.is_empty() {
                                    batches.push(Ok(wire::payload_batch(
                                        encode_schema.clone(),
                                        &payloads,
                                    )));
                                    payloads.clear();
                                }
                                batches.push(Ok(wire::error_batch(encode_schema.clone(), error)));
                                break;
                            }
                        }
                    }
                    if !payloads.is_empty() {
                        batches.push(Ok(wire::payload_batch(encode_schema.clone(), &payloads)));
                    }
                    futures::stream::iter(batches)
                });
            Ok(SedonaCObjectStoreResponse {
                metadata: SedonaCObjectStoreBuffer::default(),
                stream: export_stream(
                    Box::pin(RecordBatchStreamAdapter::new(schema, batches)),
                    state.runtime.clone(),
                ),
            })
        }
    }
}
