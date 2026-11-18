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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{stream::BoxStream, StreamExt, TryStreamExt};
use object_store::{
    memory::InMemory, path::Path, Attribute, Attributes, CopyOptions, GetOptions, GetRange,
    GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use sedona_extension::extension::SedonaCError;
use sedona_extension::object_store::ffi::*;
use sedona_extension::object_store::{
    ExportedObjectStore, ImportedObjectStore, ObjectStoreExecutor, ObjectStoreOptions,
};
use sedona_extension::runtime::RuntimeHandle;

fn runtime() -> Arc<RuntimeHandle> {
    Arc::new(RuntimeHandle::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap(),
    ))
}
fn import(
    store: Arc<dyn ObjectStore>,
    runtime: Arc<RuntimeHandle>,
    options: ObjectStoreOptions,
) -> ImportedObjectStore {
    let mut raw = ExportedObjectStore::new(store, runtime).into_ffi();
    unsafe { ImportedObjectStore::try_new(&mut raw, Arc::new(()), options).unwrap() }
}
fn options() -> ObjectStoreOptions {
    ObjectStoreOptions {
        call_timeout: Duration::from_millis(200),
        idle_timeout: Duration::from_millis(100),
        operation_timeout: Duration::from_secs(3),
    }
}
fn assert_kind(error: object_store::Error, expected: io::ErrorKind) {
    let object_store::Error::Generic { source, .. } = error else {
        panic!("expected Generic error: {error}")
    };
    assert_eq!(source.downcast_ref::<io::Error>().unwrap().kind(), expected);
}
async fn until(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}

#[test]
fn reads_ranges_attributes_and_errors_without_a_caller_runtime() {
    let rt = runtime();
    let store = Arc::new(InMemory::new());
    let path = Path::from("data/points.laz");
    let data = Bytes::from(vec![42; OBJECT_STORE_MAX_BYTES * 2 + 17]);
    let attributes = Attributes::from_iter([
        (Attribute::ContentType, "application/octet-stream"),
        (Attribute::Metadata("custom".into()), "value"),
    ]);
    rt.block_on(store.put_opts(
        &path,
        data.clone().into(),
        PutOptions {
            attributes: attributes.clone(),
            ..Default::default()
        },
    ))
    .unwrap();
    let expected = rt.block_on(store.head(&path)).unwrap();
    let imported = import(store, rt, ObjectStoreOptions::default());
    futures::executor::block_on(async {
        assert_eq!(imported.head(&path).await.unwrap(), expected);
        let result = imported.get(&path).await.unwrap();
        assert_eq!(result.attributes, attributes);
        assert_eq!(result.range, 0..data.len() as u64);
        assert_eq!(result.bytes().await.unwrap(), data);
        for (range, expected) in [
            (GetRange::Bounded(7..19), data.slice(7..19)),
            (
                GetRange::Offset(data.len() as u64 - 9),
                data.slice(data.len() - 9..),
            ),
            (GetRange::Suffix(3), data.slice(data.len() - 3..)),
        ] {
            assert_eq!(
                imported
                    .get_opts(
                        &path,
                        GetOptions {
                            range: Some(range),
                            ..Default::default()
                        }
                    )
                    .await
                    .unwrap()
                    .bytes()
                    .await
                    .unwrap(),
                expected
            );
        }
        assert!(matches!(
            imported.head(&Path::from("missing")).await,
            Err(object_store::Error::NotFound { .. })
        ));
        assert!(matches!(
            imported
                .get_opts(
                    &path,
                    GetOptions {
                        if_match: Some("wrong".into()),
                        ..Default::default()
                    }
                )
                .await,
            Err(object_store::Error::Precondition { .. })
        ));
        assert!(matches!(
            imported
                .get_opts(
                    &path,
                    GetOptions {
                        if_none_match: expected.e_tag,
                        ..Default::default()
                    }
                )
                .await,
            Err(object_store::Error::NotModified { .. })
        ));
        assert!(matches!(
            imported.put(&path, Bytes::new().into()).await,
            Err(object_store::Error::NotSupported { .. })
        ));
        let mut extensions = object_store::Extensions::new();
        extensions.insert(123u32);
        assert!(matches!(
            imported
                .get_opts(
                    &path,
                    GetOptions {
                        extensions,
                        ..Default::default()
                    }
                )
                .await,
            Err(object_store::Error::NotSupported { .. })
        ));
    });
}

#[tokio::test]
async fn listing_batches_offsets_delimiters_and_empty_objects() {
    let rt = runtime();
    let store = Arc::new(InMemory::new());
    for i in 0..300 {
        store
            .put(&Path::from(format!("data/{i:04}")), Bytes::new().into())
            .await
            .unwrap();
    }
    store
        .put(&Path::from("data/subdir/file"), Bytes::new().into())
        .await
        .unwrap();
    let imported = import(store.clone(), rt, ObjectStoreOptions::default());
    assert_eq!(
        imported
            .get(&Path::from("data/0000"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap(),
        Bytes::new()
    );
    let prefix = Path::from("data");
    let actual: Vec<_> = imported.list(Some(&prefix)).try_collect().await.unwrap();
    assert_eq!(actual.len(), 301);
    let offset = Path::from("data/0250");
    let actual: Vec<_> = imported
        .list_with_offset(Some(&prefix), &offset)
        .try_collect()
        .await
        .unwrap();
    let expected: Vec<_> = store
        .list_with_offset(Some(&prefix), &offset)
        .try_collect()
        .await
        .unwrap();
    assert_eq!(actual, expected);
    let actual = imported.list_with_delimiter(Some(&prefix)).await.unwrap();
    assert_eq!(actual.objects.len(), 300);
    assert_eq!(actual.common_prefixes, vec![Path::from("data/subdir")]);
    assert!(imported
        .list(Some(&Path::from("absent")))
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .is_empty());
}

#[derive(Debug, Clone, Copy)]
enum Behavior {
    PendingOpen,
    PendingBody,
    PendingList,
    BodyError,
    ShortBody,
    Versioned,
    Trickle,
}
#[derive(Debug)]
struct Fixture {
    memory: InMemory,
    behavior: Behavior,
    started: AtomicUsize,
    dropped: Arc<AtomicUsize>,
}
impl std::fmt::Display for Fixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fixture")
    }
}
struct DropCount(Arc<AtomicUsize>);
impl Drop for DropCount {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
impl Fixture {
    async fn new(behavior: Behavior) -> Arc<Self> {
        let memory = InMemory::new();
        memory
            .put(&Path::from("file"), Bytes::from_static(b"hello").into())
            .await
            .unwrap();
        Arc::new(Self {
            memory,
            behavior,
            started: AtomicUsize::new(0),
            dropped: Arc::new(AtomicUsize::new(0)),
        })
    }
}
#[async_trait]
impl ObjectStore for Fixture {
    async fn get_opts(&self, path: &Path, options: GetOptions) -> object_store::Result<GetResult> {
        self.started.fetch_add(1, Ordering::SeqCst);
        if matches!(self.behavior, Behavior::PendingOpen) {
            let _drop = DropCount(self.dropped.clone());
            return futures::future::pending().await;
        }
        let head = options.head;
        if matches!(self.behavior, Behavior::Versioned) {
            assert_eq!(options.version.as_deref(), Some("v123"));
            assert_eq!(options.if_modified_since.unwrap().timestamp(), 1234);
            assert_eq!(options.if_unmodified_since.unwrap().timestamp(), 5678);
            let mut result = self.memory.get(path).await?;
            result.meta.version = Some("v123".into());
            return Ok(result);
        }
        let mut result = self.memory.get_opts(path, options).await?;
        if !head {
            result.payload = match self.behavior {
                Behavior::PendingBody => {
                    let dropped = self.dropped.clone();
                    GetResultPayload::Stream(
                        futures::stream::once(async move {
                            let _drop = DropCount(dropped);
                            futures::future::pending().await
                        })
                        .boxed(),
                    )
                }
                Behavior::Trickle => GetResultPayload::Stream(
                    futures::stream::iter(0..5)
                        .then(|_| async {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            Ok(Bytes::from_static(b"x"))
                        })
                        .boxed(),
                ),
                Behavior::BodyError => GetResultPayload::Stream(
                    futures::stream::once(async {
                        Err(object_store::Error::PermissionDenied {
                            path: "file".into(),
                            source: "denied while reading".into(),
                        })
                    })
                    .boxed(),
                ),
                Behavior::ShortBody => GetResultPayload::Stream(
                    futures::stream::iter([Ok(Bytes::from_static(b"hi"))]).boxed(),
                ),
                _ => result.payload,
            };
        }
        Ok(result)
    }
    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        if matches!(self.behavior, Behavior::PendingList) {
            let dropped = self.dropped.clone();
            return futures::stream::once(async move {
                let _drop = DropCount(dropped);
                futures::future::pending().await
            })
            .boxed();
        }
        self.memory.list(prefix)
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        if matches!(self.behavior, Behavior::PendingList) {
            let _drop = DropCount(self.dropped.clone());
            return futures::future::pending().await;
        }
        self.memory.list_with_delimiter(prefix).await
    }
    async fn put_opts(
        &self,
        p: &Path,
        b: PutPayload,
        o: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.memory.put_opts(p, b, o).await
    }
    async fn put_multipart_opts(
        &self,
        p: &Path,
        o: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.memory.put_multipart_opts(p, o).await
    }
    fn delete_stream(
        &self,
        p: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.memory.delete_stream(p)
    }
    async fn copy_opts(&self, a: &Path, b: &Path, o: CopyOptions) -> object_store::Result<()> {
        self.memory.copy_opts(a, b, o).await
    }
}

#[tokio::test]
async fn version_and_date_conditions_cross_the_boundary() {
    let store = Fixture::new(Behavior::Versioned).await;
    let imported = import(store, runtime(), ObjectStoreOptions::default());
    let result = imported
        .get_opts(
            &Path::from("file"),
            GetOptions {
                version: Some("v123".into()),
                if_modified_since: Some(chrono::DateTime::from_timestamp(1234, 0).unwrap()),
                if_unmodified_since: Some(chrono::DateTime::from_timestamp(5678, 0).unwrap()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(result.meta.version.as_deref(), Some("v123"));
}

#[tokio::test]
async fn timeout_and_dropping_calls_abort_the_exported_future() {
    let store = Fixture::new(Behavior::PendingOpen).await;
    let imported = import(store.clone(), runtime(), options());
    assert_kind(
        imported.head(&Path::from("file")).await.unwrap_err(),
        io::ErrorKind::TimedOut,
    );
    until(|| store.dropped.load(Ordering::SeqCst) == 1).await;
    let task = tokio::spawn(async move { imported.head(&Path::from("file")).await });
    until(|| store.started.load(Ordering::SeqCst) == 2).await;
    task.abort();
    until(|| store.dropped.load(Ordering::SeqCst) == 2).await;
}

#[tokio::test]
async fn idle_body_and_list_timeouts_cancel_native_sources() {
    for behavior in [Behavior::PendingBody, Behavior::PendingList] {
        let store = Fixture::new(behavior).await;
        let imported = import(store.clone(), runtime(), options());
        let error = if matches!(behavior, Behavior::PendingBody) {
            imported
                .get(&Path::from("file"))
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap_err()
        } else {
            imported
                .list(None)
                .try_collect::<Vec<_>>()
                .await
                .unwrap_err()
        };
        assert_kind(error, io::ErrorKind::TimedOut);
        until(|| store.dropped.load(Ordering::SeqCst) == 1).await;
        if matches!(behavior, Behavior::PendingList) {
            assert_kind(
                imported.list_with_delimiter(None).await.unwrap_err(),
                io::ErrorKind::TimedOut,
            );
            until(|| store.dropped.load(Ordering::SeqCst) == 2).await;
        }
    }
}

#[tokio::test]
async fn body_errors_remain_typed_and_truncation_is_detected() {
    for behavior in [Behavior::BodyError, Behavior::ShortBody] {
        let imported = import(Fixture::new(behavior).await, runtime(), options());
        let error = imported
            .get(&Path::from("file"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap_err();
        match behavior {
            Behavior::BodyError => assert!(matches!(
                error,
                object_store::Error::PermissionDenied { .. }
            )),
            _ => assert_kind(error, io::ErrorKind::InvalidData),
        }
    }
}

#[tokio::test]
async fn pending_streams_do_not_occupy_foreign_threads() {
    let store = Fixture::new(Behavior::PendingBody).await;
    let pool = ObjectStoreExecutor::new(1, 32).unwrap();
    let mut raw = ExportedObjectStore::new(store.clone(), runtime()).into_ffi();
    let imported = unsafe {
        ImportedObjectStore::try_new_with_executor(
            &mut raw,
            Arc::new(()),
            ObjectStoreOptions::default(),
            pool,
        )
        .unwrap()
    };
    let mut bodies = Vec::new();
    for _ in 0..16 {
        bodies.push(imported.get(&Path::from("file")).await.unwrap());
    }
    tokio::time::timeout(Duration::from_secs(3), imported.head(&Path::from("file")))
        .await
        .unwrap()
        .unwrap();
    // Dropping a body only signals local cancellation; its foreign cleanup is asynchronous.
    drop(bodies);
    until(|| store.dropped.load(Ordering::SeqCst) > 0).await;
}

#[derive(Default)]
struct Gate {
    entered: AtomicUsize,
    open: Mutex<bool>,
    ready: Condvar,
}
impl Gate {
    fn block(&self) {
        self.entered.fetch_add(1, Ordering::SeqCst);
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.ready.wait(open).unwrap();
        }
    }
    fn unblock(&self) {
        *self.open.lock().unwrap() = true;
        self.ready.notify_all();
    }
}
struct ForeignFixture {
    original: SedonaCObjectStore,
    gate: Arc<Gate>,
    released: Arc<AtomicBool>,
}
unsafe extern "C" fn stalled_call(
    value: *const SedonaCObjectStore,
    context: *const SedonaCObjectStoreCall,
    request: *const u8,
    len: usize,
    out: *mut SedonaCObjectStoreResponse,
    error: *mut SedonaCError,
) -> c_int {
    let state = &*(*value).private_data.cast::<ForeignFixture>();
    state.gate.block();
    state.original.call.unwrap()(&state.original, context, request, len, out, error)
}
unsafe extern "C" fn release_fixture(value: *mut SedonaCObjectStore) {
    let mut state = Box::from_raw((*value).private_data.cast::<ForeignFixture>());
    state.original.release.unwrap()(&mut state.original);
    state.released.store(true, Ordering::SeqCst);
    (*value).release = None;
}

#[tokio::test]
async fn stalled_calls_and_queued_calls_expire_without_freeing_live_arguments() {
    let rt = runtime();
    let gate = Arc::new(Gate::default());
    let released = Arc::new(AtomicBool::new(false));
    let fixture = ForeignFixture {
        original: ExportedObjectStore::new(Arc::new(InMemory::new()), rt).into_ffi(),
        gate: gate.clone(),
        released: released.clone(),
    };
    let mut raw = SedonaCObjectStore {
        abi_version: 1,
        capabilities: OBJECT_STORE_READ,
        call: Some(stalled_call),
        release: Some(release_fixture),
        private_data: Box::into_raw(Box::new(fixture)).cast(),
        reserved: [0; 4],
    };
    let pool = ObjectStoreExecutor::new(1, 3).unwrap();
    let imported = unsafe {
        ImportedObjectStore::try_new_with_executor(&mut raw, Arc::new(()), options(), pool).unwrap()
    };
    assert_kind(
        imported.head(&Path::from("file")).await.unwrap_err(),
        io::ErrorKind::TimedOut,
    );
    assert_eq!(gate.entered.load(Ordering::SeqCst), 1);
    assert_kind(
        imported.head(&Path::from("file")).await.unwrap_err(),
        io::ErrorKind::TimedOut,
    );
    assert_kind(
        imported.head(&Path::from("file")).await.unwrap_err(),
        io::ErrorKind::WouldBlock,
    );
    drop(imported);
    assert!(!released.load(Ordering::SeqCst));
    gate.unblock();
    until(|| released.load(Ordering::SeqCst)).await;
    // The queued call expired before dispatch and never entered the foreign callback.
    assert_eq!(gate.entered.load(Ordering::SeqCst), 1);
}

#[test]
fn invalid_import_preserves_caller_ownership() {
    let mut raw = ExportedObjectStore::new(Arc::new(InMemory::new()), runtime()).into_ffi();
    raw.abi_version = 999;
    assert!(unsafe {
        ImportedObjectStore::try_new(&mut raw, Arc::new(()), ObjectStoreOptions::default())
    }
    .is_err());
    assert!(raw.release.is_some());
    unsafe {
        raw.release.unwrap()(&mut raw);
    }
}

// Each permanent stall runs in a subprocess. Even a regression in cancellation
// must fail this test rather than hang the parent test runner or its shutdown.
#[test]
fn permanently_stuck_callbacks_cannot_hold_the_query() {
    const CHILD: &str = "SEDONA_OBJECT_STORE_STALL";
    if let Ok(mode) = std::env::var(CHILD) {
        let rt = runtime();
        rt.block_on(async {
            let pool = ObjectStoreExecutor::new(1, 4).unwrap();
            let mut raw = hostile_store(mode.clone());
            let imported = unsafe {
                ImportedObjectStore::try_new_with_executor(&mut raw, Arc::new(()), options(), pool)
                    .unwrap()
            };
            let start = Instant::now();
            if mode == "store_release" {
                drop(imported);
                until(|| STALLED.load(Ordering::SeqCst)).await;
            } else if mode == "cancel_after_eos" {
                assert_eq!(
                    imported
                        .get(&Path::from("file"))
                        .await
                        .unwrap()
                        .bytes()
                        .await
                        .unwrap(),
                    Bytes::from_static(b"x")
                );
                drop(imported);
                until(|| STALLED.load(Ordering::SeqCst)).await;
            } else {
                let error = match imported.get(&Path::from("file")).await {
                    Err(error) => error,
                    Ok(result) => result.bytes().await.unwrap_err(),
                };
                assert_kind(error, io::ErrorKind::TimedOut);
                drop(imported);
                until(|| STALLED.load(Ordering::SeqCst)).await;
            }
            assert!(start.elapsed() < Duration::from_secs(2));
        });
        // Native code cannot be killed safely in-process. The parent bounds the
        // process lifetime; this deliberately avoids waiting for a stuck worker.
        std::process::exit(0);
    }
    for mode in [
        "call",
        "start",
        "request",
        "extract",
        "cancel",
        "metadata_release",
        "stream_release",
        "store_release",
        "array_release",
        "error_release",
        "cancel_after_eos",
    ] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "permanently_stuck_callbacks_cannot_hold_the_query",
                "--nocapture",
            ])
            .env(CHILD, mode)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "child {mode}: {status}");
                break;
            }
            if Instant::now() > deadline {
                child.kill().unwrap();
                let _ = child.wait();
                panic!("query hung in {mode}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

static STALLED: AtomicBool = AtomicBool::new(false);
fn stall() -> ! {
    STALLED.store(true, Ordering::SeqCst);
    loop {
        std::thread::park();
    }
}
fn hostile_store(mode: String) -> SedonaCObjectStore {
    SedonaCObjectStore {
        abi_version: 1,
        capabilities: OBJECT_STORE_READ,
        call: Some(hostile_call),
        release: Some(hostile_release),
        private_data: Box::into_raw(Box::new(mode)).cast(),
        reserved: [0; 4],
    }
}
unsafe extern "C" fn hostile_release(value: *mut SedonaCObjectStore) {
    let mode = Box::from_raw((*value).private_data.cast::<String>());
    if mode.as_str() == "store_release" {
        stall();
    }
    (*value).release = None;
}
unsafe extern "C" fn hostile_call(
    value: *const SedonaCObjectStore,
    _: *const SedonaCObjectStoreCall,
    _: *const u8,
    _: usize,
    out: *mut SedonaCObjectStoreResponse,
    err: *mut SedonaCError,
) -> c_int {
    let mode = &*(*value).private_data.cast::<String>();
    if mode == "call" {
        stall();
    }
    if mode == "error_release" {
        *err = SedonaCError::new("failed");
        (*err).release = Some(stalled_error_release);
        return 1;
    }
    static META: &[u8] = br#"{"meta":{"location":"file","last_modified":"2026-01-01T00:00:00Z","size":1,"e_tag":null,"version":null},"range_start":0,"range_end":1,"attributes":[]}"#;
    (*out).metadata = SedonaCObjectStoreBuffer {
        data: META.as_ptr(),
        len: META.len(),
        release: if mode == "metadata_release" {
            Some(stalled_buffer_release)
        } else {
            None
        },
        private_data: std::ptr::null_mut(),
    };
    (*out).stream = SedonaCObjectStoreStream {
        start: Some(hostile_start),
        release: Some(hostile_stream_release),
        private_data: Box::into_raw(Box::new(mode.clone())).cast(),
    };
    0
}
unsafe extern "C" fn stalled_error_release(_: *mut SedonaCError) {
    stall();
}
unsafe extern "C" fn stalled_buffer_release(_: *mut SedonaCObjectStoreBuffer) {
    stall();
}
unsafe extern "C" fn hostile_stream_release(value: *mut SedonaCObjectStoreStream) {
    let mode = Box::from_raw((*value).private_data.cast::<String>());
    if mode.as_str() == "stream_release" {
        stall();
    }
    (*value).release = None;
}
struct HostileProducer {
    mode: String,
    handler: *mut sedona_extension::extension::FFI_ArrowAsyncDeviceStreamHandler,
}
unsafe extern "C" fn hostile_start(
    value: *mut SedonaCObjectStoreStream,
    handler: *mut sedona_extension::extension::FFI_ArrowAsyncDeviceStreamHandler,
    _: *mut SedonaCError,
) -> c_int {
    use sedona_extension::extension::FFI_ArrowAsyncProducer;
    let mode = &*(*value).private_data.cast::<String>();
    if mode == "start" {
        stall();
    }
    if mode == "premature_release" {
        ((*handler).release.unwrap())(handler);
        return 0;
    }
    let producer = Box::new(FFI_ArrowAsyncProducer {
        device_type: 1,
        request: Some(hostile_request),
        cancel: Some(hostile_cancel),
        additional_metadata: std::ptr::null(),
        private_data: Box::into_raw(Box::new(HostileProducer {
            mode: mode.clone(),
            handler,
        }))
        .cast(),
    });
    (*handler).producer = Box::into_raw(producer);
    let schema = arrow_schema::Schema::new_with_metadata(
        vec![
            arrow_schema::Field::new("payload", arrow_schema::DataType::Binary, true),
            arrow_schema::Field::new("error", arrow_schema::DataType::Utf8, true),
        ],
        [
            ("sedona.object_store.version".into(), "1".into()),
            ("sedona.object_store.kind".into(), "bytes".into()),
        ]
        .into(),
    );
    let mut schema = arrow_array::ffi::FFI_ArrowSchema::try_from(&schema).unwrap();
    ((*handler).on_schema.unwrap())(handler, &mut schema);
    0
}
unsafe extern "C" fn hostile_request(
    producer: *mut sedona_extension::extension::FFI_ArrowAsyncProducer,
    _: i64,
) {
    let state = &*(*producer).private_data.cast::<HostileProducer>();
    if state.mode == "request" {
        stall();
    }
    if ["extract", "array_release", "cancel_after_eos"].contains(&state.mode.as_str()) {
        let extract = if state.mode == "extract" {
            stalled_extract
        } else {
            ready_extract
        };
        let mut task = sedona_extension::extension::FFI_ArrowAsyncTask {
            extract_data: Some(extract),
            private_data: state as *const HostileProducer as *mut std::ffi::c_void,
        };
        ((*state.handler).on_next_task.unwrap())(state.handler, &mut task, std::ptr::null());
        // Callback accepted ownership of the task.
        task.extract_data = None;
        if state.mode == "cancel_after_eos" {
            ((*state.handler).on_next_task.unwrap())(
                state.handler,
                std::ptr::null_mut(),
                std::ptr::null(),
            );
        }
    }
}
unsafe extern "C" fn hostile_cancel(
    producer: *mut sedona_extension::extension::FFI_ArrowAsyncProducer,
) {
    let state = &*(*producer).private_data.cast::<HostileProducer>();
    if state.mode == "cancel" || state.mode == "cancel_after_eos" {
        stall();
    }
    // The subprocess owns these intentionally non-cooperative producer fixtures.
}
unsafe extern "C" fn stalled_extract(
    _: *mut sedona_extension::extension::FFI_ArrowAsyncTask,
    _: *mut sedona_extension::extension::FFI_ArrowDeviceArray,
) -> c_int {
    stall();
}

#[tokio::test]
async fn release_without_completion_is_a_protocol_error() {
    let mut raw = hostile_store("premature_release".into());
    let imported =
        unsafe { ImportedObjectStore::try_new(&mut raw, Arc::new(()), options()).unwrap() };
    let error = imported
        .get(&Path::from("file"))
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap_err();
    assert!(error.to_string().contains("released before completion"));
}

#[tokio::test]
async fn overall_deadline_is_not_reset_by_progress() {
    let store = Fixture::new(Behavior::Trickle).await;
    let imported = import(
        store,
        runtime(),
        ObjectStoreOptions {
            operation_timeout: Duration::from_millis(250),
            idle_timeout: Duration::from_millis(200),
            call_timeout: Duration::from_secs(2),
        },
    );
    assert_kind(
        imported
            .get(&Path::from("file"))
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap_err(),
        io::ErrorKind::TimedOut,
    );
}

#[tokio::test]
async fn consumer_backpressure_is_not_an_idle_timeout() {
    let store = Arc::new(InMemory::new());
    let path = Path::from("file");
    store
        .put(
            &path,
            Bytes::from(vec![7; OBJECT_STORE_MAX_BYTES * 3]).into(),
        )
        .await
        .unwrap();
    let imported = import(store, runtime(), options());
    let mut stream = imported.get(&path).await.unwrap().into_stream();
    assert_eq!(
        stream.try_next().await.unwrap().unwrap().len(),
        OBJECT_STORE_MAX_BYTES
    );
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(
        stream.try_next().await.unwrap().unwrap().len(),
        OBJECT_STORE_MAX_BYTES
    );
    assert_eq!(
        stream.try_next().await.unwrap().unwrap().len(),
        OBJECT_STORE_MAX_BYTES
    );
    assert!(stream.try_next().await.unwrap().is_none());
}

#[tokio::test]
async fn body_retains_store_runtime_and_library_until_cleanup() {
    let store = Arc::new(InMemory::new());
    store
        .put(&Path::from("file"), Bytes::from_static(b"hello").into())
        .await
        .unwrap();
    let rt = runtime();
    let weak_runtime = Arc::downgrade(&rt);
    let owner = Arc::new(AtomicBool::new(false));
    let weak_owner = Arc::downgrade(&owner);
    let mut raw = ExportedObjectStore::new(store, rt).into_ffi();
    let imported = unsafe {
        ImportedObjectStore::try_new(&mut raw, owner, ObjectStoreOptions::default()).unwrap()
    };
    let result = imported.get(&Path::from("file")).await.unwrap();
    drop(imported);
    assert_eq!(result.bytes().await.unwrap(), Bytes::from_static(b"hello"));
    until(|| weak_runtime.upgrade().is_none() && weak_owner.upgrade().is_none()).await;
}

#[repr(C)]
struct TestArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *const *const std::ffi::c_void,
    children: *mut *mut TestArrowArray,
    dictionary: *mut TestArrowArray,
    release: Option<unsafe extern "C" fn(*mut TestArrowArray)>,
    private_data: *mut std::ffi::c_void,
}
unsafe extern "C" fn stalled_array_release(_: *mut TestArrowArray) {
    stall();
}
unsafe extern "C" fn ready_extract(
    task: *mut sedona_extension::extension::FFI_ArrowAsyncTask,
    out: *mut sedona_extension::extension::FFI_ArrowDeviceArray,
) -> c_int {
    if out.is_null() {
        return 0;
    }
    let state = &*(*task).private_data.cast::<HostileProducer>();
    let fields = vec![
        arrow_schema::Field::new("payload", arrow_schema::DataType::Binary, true),
        arrow_schema::Field::new("error", arrow_schema::DataType::Utf8, true),
    ];
    let batch = arrow_array::RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(fields)),
        vec![
            Arc::new(arrow_array::BinaryArray::from(vec![Some(b"x".as_slice())])),
            Arc::new(arrow_array::StringArray::from(vec![None::<&str>])),
        ],
    )
    .unwrap();
    let array = arrow_array::StructArray::from(batch);
    let (mut array, _) = arrow_array::ffi::to_ffi(&array.into()).unwrap();
    if state.mode == "array_release" {
        // Install a genuinely foreign release callback using the public C layout.
        let layout = (&mut array as *mut arrow_array::ffi::FFI_ArrowArray).cast::<TestArrowArray>();
        (*layout).release = Some(stalled_array_release);
    }
    std::ptr::write(
        out,
        sedona_extension::extension::FFI_ArrowDeviceArray::from(array),
    );
    0
}

#[tokio::test]
async fn prefixed_store_preserves_underlying_get_metadata() {
    let memory = Arc::new(InMemory::new());
    let store = Arc::new(object_store::prefix::PrefixStore::new(memory, "base"));
    let path = Path::from("file");
    store
        .put(&path, Bytes::from_static(b"hello").into())
        .await
        .unwrap();
    let expected = store.head(&path).await.unwrap();
    let imported = import(store, runtime(), ObjectStoreOptions::default());
    assert_eq!(imported.head(&path).await.unwrap(), expected);
    assert_eq!(
        imported.get_range(&path, 1..3).await.unwrap(),
        Bytes::from_static(b"el")
    );
}

#[tokio::test]
async fn local_file_payloads_are_streamed_with_ranges() {
    let directory = tempfile::tempdir().unwrap();
    let store =
        Arc::new(object_store::local::LocalFileSystem::new_with_prefix(directory.path()).unwrap());
    let path = Path::from("file");
    let data = Bytes::from(vec![17; OBJECT_STORE_MAX_BYTES + 31]);
    store.put(&path, data.clone().into()).await.unwrap();
    let imported = import(store, runtime(), ObjectStoreOptions::default());
    assert_eq!(
        imported.get(&path).await.unwrap().bytes().await.unwrap(),
        data
    );
    assert_eq!(
        imported.get_range(&path, 9..100).await.unwrap(),
        data.slice(9..100)
    );
}
