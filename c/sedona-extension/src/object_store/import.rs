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

use std::ffi::{c_int, c_void};
use std::fmt;
use std::io;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use bytes::Bytes;
use futures::channel::oneshot;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use object_store::{
    path::Path, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload,
    ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use tokio::sync::OwnedSemaphorePermit;

use super::bounded::{
    self, invalid, unsupported, CancelOnDrop, Cancellation, ObjectStoreExecutor,
    ObjectStoreOptions, ResponseStream,
};
use super::ffi::*;
use super::wire::{self, ListEntry, ReadMetadata, Request, WireError};
use crate::extension::SedonaCError;
use crate::import_sendable_record_batch_stream::ImportedAsyncDeviceStream;

struct ImportedState {
    raw: Option<SedonaCObjectStore>,
    executor: Arc<ObjectStoreExecutor>,
    permit: Option<OwnedSemaphorePermit>,
    owner: Arc<dyn Send + Sync>,
}
impl Drop for ImportedState {
    fn drop(&mut self) {
        let mut raw = self.raw.take().unwrap();
        let permit = self.permit.take();
        let owner = self.owner.clone();
        self.executor.spawn(async move {
            // Admission and library ownership survive a stuck release callback.
            let (_permit, _owner) = (permit, owner);
            if let Some(release) = raw.release {
                unsafe {
                    release(&mut raw);
                }
            }
        });
    }
}
struct OperationOwner {
    _store: Arc<ImportedState>,
    _permit: OwnedSemaphorePermit,
}

/// Rust ObjectStore backed by a v1 C store. Clones share ownership and capacity,
/// but every operation has independent cancellation and deadlines.
#[derive(Clone)]
pub struct ImportedObjectStore {
    state: Arc<ImportedState>,
    options: ObjectStoreOptions,
}
impl ImportedObjectStore {
    /// Move a raw store on success, leaving an empty value in its place. On error,
    /// ownership remains with the caller, including when admission is exhausted.
    /// `owner` keeps the supplying dynamic library loaded through all callbacks.
    ///
    /// # Safety
    /// The raw store and its results must obey the documented C ownership and
    /// thread-safety contract. `owner` must keep its code loaded until dropped.
    /// Native memory corruption and callbacks after release cannot be sandboxed.
    pub unsafe fn try_new(
        raw: &mut SedonaCObjectStore,
        owner: Arc<dyn Send + Sync>,
        options: ObjectStoreOptions,
    ) -> object_store::Result<Self> {
        let executor =
            ObjectStoreExecutor::shared().map_err(|e| bounded::failure(e.kind(), e.to_string()))?;
        Self::try_new_with_executor(raw, owner, options, executor)
    }

    /// Import using an explicitly shared foreign executor. See [`Self::try_new`].
    ///
    /// # Safety
    /// The same validity, ownership, and library-lifetime requirements apply.
    pub unsafe fn try_new_with_executor(
        raw: &mut SedonaCObjectStore,
        owner: Arc<dyn Send + Sync>,
        options: ObjectStoreOptions,
        executor: Arc<ObjectStoreExecutor>,
    ) -> object_store::Result<Self> {
        options.validate()?;
        if raw.abi_version != OBJECT_STORE_ABI_VERSION {
            return Err(unsupported(format!(
                "unsupported ObjectStore ABI version {}",
                raw.abi_version
            )));
        }
        if raw.release.is_none() || raw.call.is_none() || raw.reserved != [0; 4] {
            return Err(invalid(
                "uninitialized ObjectStore or nonzero reserved fields",
            ));
        }
        let permit = executor.admit()?;
        Ok(Self {
            state: Arc::new(ImportedState {
                raw: Some(std::mem::take(raw)),
                executor,
                permit: Some(permit),
                owner,
            }),
            options,
        })
    }

    pub fn capabilities(&self) -> u64 {
        self.state.raw.as_ref().unwrap().capabilities
    }

    async fn open(
        &self,
        request: Request,
    ) -> object_store::Result<(Vec<u8>, ResponseStream<Vec<Vec<u8>>>)> {
        let now = Instant::now();
        let operation_deadline = now + self.options.operation_timeout;
        let call_deadline = (now + self.options.call_timeout).min(operation_deadline);
        let required = match &request {
            Request::Get { .. } => OBJECT_STORE_READ,
            Request::List {
                delimiter: true, ..
            } => OBJECT_STORE_LIST_WITH_DELIMITER,
            Request::List { .. } => OBJECT_STORE_LIST,
        };
        if self.capabilities() & required == 0 {
            return Err(unsupported("ObjectStore operation is not supported"));
        }
        let request_bytes = wire::encode(&request)?;
        let permit = self.state.executor.admit()?;
        let owner = Arc::new(OperationOwner {
            _store: self.state.clone(),
            _permit: permit,
        });
        let cancel = Arc::new(Cancellation::default());
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let (opened, opening) = oneshot::channel();
        let stream = ResponseStream {
            receiver,
            cancel: CancelOnDrop(cancel.clone()),
            deadline: bounded::timer(operation_deadline),
            idle_timeout: self.options.idle_timeout,
            idle: None,
            finished: false,
        };
        let state = self.state.clone();
        self.state.executor.spawn(async move {
            let _owner = owner.clone();
            let result = open_foreign(
                &state,
                &cancel,
                call_deadline,
                &request,
                &request_bytes,
                owner,
            );
            let (metadata, source, expected_len) = match result {
                Ok(result) => result,
                Err(error) => {
                    let _ = opened.send(Err(error));
                    return;
                }
            };
            if opened.send(Ok(metadata)).is_err() {
                return;
            }
            if let Some(mut source) = source {
                let schema = wire::schema(if matches!(request, Request::Get { .. }) {
                    "bytes"
                } else {
                    "list"
                });
                let mut received = 0u64;
                loop {
                    let next =
                        futures::future::select(Box::pin(cancel.cancelled()), source.next()).await;
                    let batch = match next {
                        futures::future::Either::Left(_) => break,
                        futures::future::Either::Right((batch, _)) => batch,
                    };
                    let result = match batch {
                        Some(Ok(batch)) => wire::decode_batch(batch, &schema).and_then(|values| {
                            if let Some(expected) = expected_len {
                                for value in &values {
                                    received = received
                                        .checked_add(value.len() as u64)
                                        .ok_or_else(|| invalid("body length overflow"))?;
                                }
                                if received > expected {
                                    return Err(invalid("body exceeds the returned range"));
                                }
                            }
                            Ok(values)
                        }),
                        Some(Err(error)) => {
                            Err(invalid(format!("ObjectStore stream protocol: {error}")))
                        }
                        None => {
                            if expected_len.is_some_and(|expected| expected != received) {
                                let _ = sender
                                    .send(Err(invalid("body shorter than the returned range")))
                                    .await;
                            }
                            break;
                        }
                    };
                    let empty = match &result {
                        Ok(values) => values.iter().all(Vec::is_empty),
                        Err(_) => false,
                    };
                    if empty {
                        continue;
                    }
                    let failed = result.is_err();
                    match futures::future::select(
                        Box::pin(cancel.cancelled()),
                        Box::pin(sender.send(result)),
                    )
                    .await
                    {
                        futures::future::Either::Left(_) => break,
                        futures::future::Either::Right((Err(_), _)) => break,
                        _ => {}
                    }
                    if failed {
                        break;
                    }
                }
                // Publish completion before foreign cleanup: a stuck cancel or
                // task destructor must not hide an already received EOF.
                drop(sender);
                // source (and any queued Arrow tasks) is dropped on this executor.
            }
        });
        let metadata = bounded::wait(opening, call_deadline)
            .await?
            .map_err(|_| invalid("foreign ObjectStore worker stopped during startup"))??;
        Ok((metadata, stream))
    }

    fn list_entries(
        &self,
        prefix: Option<&Path>,
        offset: Option<&Path>,
        delimiter: bool,
    ) -> BoxStream<'static, object_store::Result<ListEntry>> {
        let this = self.clone();
        let request = Request::List {
            prefix: prefix.map(ToString::to_string),
            offset: offset.map(ToString::to_string),
            delimiter,
        };
        futures::stream::once(async move {
            let (_, stream) = this.open(request).await?;
            Ok::<_, object_store::Error>(
                stream
                    .map_ok(|values| {
                        futures::stream::iter(
                            values
                                .into_iter()
                                .map(|value| wire::decode::<ListEntry>(&value)),
                        )
                    })
                    .try_flatten(),
            )
        })
        .try_flatten()
        .boxed()
    }
}

unsafe extern "C" fn is_cancelled(value: *const c_void) -> c_int {
    i32::from((&*value.cast::<Cancellation>()).is_cancelled())
}

fn open_foreign(
    state: &ImportedState,
    cancel: &Arc<Cancellation>,
    deadline: Instant,
    request: &Request,
    request_bytes: &[u8],
    owner: Arc<OperationOwner>,
) -> object_store::Result<(Vec<u8>, Option<ImportedAsyncDeviceStream>, Option<u64>)> {
    if cancel.is_cancelled() {
        return Err(bounded::failure(
            io::ErrorKind::Interrupted,
            "ObjectStore call cancelled before dispatch",
        ));
    }
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(bounded::failure(
            io::ErrorKind::TimedOut,
            "ObjectStore call expired before dispatch",
        ));
    }
    let context = SedonaCObjectStoreCall {
        timeout_ms: remaining.as_millis().max(1).min(u64::MAX as u128) as u64,
        is_cancelled: Some(is_cancelled),
        private_data: Arc::as_ptr(cancel).cast(),
    };
    let raw = state.raw.as_ref().unwrap();
    let mut response = SedonaCObjectStoreResponse::default();
    let mut error = SedonaCError::default();
    let code = unsafe {
        raw.call.unwrap()(
            raw,
            &context,
            request_bytes.as_ptr(),
            request_bytes.len(),
            &mut response,
            &mut error,
        )
    };
    let path = match request {
        Request::Get { location, .. } => location.clone(),
        _ => String::new(),
    };
    if code != 0 {
        if error.err_len as usize > OBJECT_STORE_MAX_BYTES {
            return Err(invalid("oversized ObjectStore error"));
        }
        return Err(WireError {
            code,
            path,
            message: error.to_string(),
        }
        .into_error());
    }
    if response.metadata.len > OBJECT_STORE_MAX_BYTES
        || (response.metadata.len > 0 && response.metadata.data.is_null())
    {
        return Err(invalid("invalid ObjectStore metadata buffer"));
    }
    let metadata = if response.metadata.len == 0 {
        Vec::new()
    } else {
        unsafe {
            std::slice::from_raw_parts(response.metadata.data, response.metadata.len).to_vec()
        }
    };
    let (needs_stream, expected_len) = match request {
        Request::Get { options, .. } => {
            let meta: ReadMetadata = wire::decode(&metadata)?;
            if meta.range_start > meta.range_end || meta.range_end > meta.meta.size {
                return Err(invalid("invalid GET response metadata"));
            }
            (!options.head, Some(meta.range_end - meta.range_start))
        }
        Request::List { .. } => (true, None),
    };
    if !needs_stream {
        if response.stream.start.is_some() {
            return Err(invalid("HEAD must not return a body"));
        }
        return Ok((metadata, None, None));
    }
    let Some(start) = response.stream.start else {
        return Err(invalid("missing ObjectStore stream"));
    };
    if response.stream.release.is_none() {
        return Err(invalid("missing stream release callback"));
    }
    let kind = if matches!(request, Request::Get { .. }) {
        "bytes"
    } else {
        "list"
    };
    let (source, handler) = ImportedAsyncDeviceStream::new_with_owner(wire::schema(kind), owner);
    let code = unsafe { start(&mut response.stream, handler.as_ptr(), &mut error) };
    if code != 0 {
        if error.err_len as usize > OBJECT_STORE_MAX_BYTES {
            return Err(invalid("oversized ObjectStore error"));
        }
        return Err(WireError {
            code,
            path,
            message: error.to_string(),
        }
        .into_error());
    }
    let _ = handler.into_raw();
    // response (including the now-started handle and metadata buffer) is released
    // here, before exposing successful startup to the caller.
    Ok((metadata, Some(source), expected_len))
}

impl fmt::Display for ImportedObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SedonaCObjectStore(v1)")
    }
}
impl fmt::Debug for ImportedObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImportedObjectStore")
            .field("capabilities", &self.capabilities())
            .field("options", &self.options)
            .finish()
    }
}
#[async_trait]
impl ObjectStore for ImportedObjectStore {
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        let (metadata, stream) = self
            .open(Request::Get {
                location: location.to_string(),
                options: options.try_into()?,
            })
            .await?;
        let metadata: ReadMetadata = wire::decode(&metadata)?;
        let bytes = stream
            .map_ok(|values| {
                futures::stream::iter(values.into_iter().map(|value| Ok(Bytes::from(value))))
            })
            .try_flatten()
            .boxed();
        Ok(GetResult {
            payload: GetResultPayload::Stream(bytes),
            meta: metadata.meta.try_into()?,
            range: metadata.range_start..metadata.range_end,
            attributes: metadata
                .attributes
                .into_iter()
                .map(|(key, value)| (object_store::Attribute::from(key), value))
                .collect(),
        })
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.list_entries(prefix, None, false)
            .and_then(|entry| async move {
                match entry {
                    ListEntry::Object { meta } => meta.try_into(),
                    ListEntry::Prefix { .. } => Err(invalid("prefix in recursive listing")),
                }
            })
            .boxed()
    }
    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.list_entries(prefix, Some(offset), false)
            .and_then(|entry| async move {
                match entry {
                    ListEntry::Object { meta } => meta.try_into(),
                    ListEntry::Prefix { .. } => Err(invalid("prefix in recursive listing")),
                }
            })
            .boxed()
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        let mut stream = self.list_entries(prefix, None, true);
        let mut result = ListResult {
            common_prefixes: Vec::new(),
            objects: Vec::new(),
        };
        while let Some(entry) = stream.try_next().await? {
            match entry {
                ListEntry::Object { meta } => result.objects.push(meta.try_into()?),
                ListEntry::Prefix { path } => result.common_prefixes.push(Path::parse(path)?),
            }
        }
        Ok(result)
    }
    async fn put_opts(
        &self,
        _: &Path,
        _: PutPayload,
        _: PutOptions,
    ) -> object_store::Result<PutResult> {
        Err(unsupported("read-only FFI ObjectStore"))
    }
    async fn put_multipart_opts(
        &self,
        _: &Path,
        _: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        Err(unsupported("read-only FFI ObjectStore"))
    }
    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        locations
            .map(|entry| entry.and_then(|_| Err(unsupported("read-only FFI ObjectStore"))))
            .boxed()
    }
    async fn copy_opts(&self, _: &Path, _: &Path, _: CopyOptions) -> object_store::Result<()> {
        Err(unsupported("read-only FFI ObjectStore"))
    }
}
