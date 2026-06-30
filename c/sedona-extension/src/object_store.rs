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

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt::{Debug, Display, Formatter};
use std::ptr::{null, null_mut};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::{DataFusionError, Result};
use futures::channel::oneshot;
use futures::stream::BoxStream;
use object_store::{
    path::Path, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use sedona_common::sedona_internal_err;
use tokio::runtime::Handle;

use crate::extension::{SedonaCAsyncResultHandler, SedonaCObjectStore};

/// Wrapper around a [SedonaCObjectStore] that implements [ObjectStore]
///
/// This is the means by which an ObjectStore implementation may be imported from a
/// C implementation.
pub struct ImportedObjectStore {
    inner: SedonaCObjectStore,
}

impl TryFrom<SedonaCObjectStore> for ImportedObjectStore {
    type Error = DataFusionError;

    fn try_from(value: SedonaCObjectStore) -> Result<Self> {
        match (
            &value.display,
            &value.debug,
            &value.delete,
            &value.copy,
            &value.copy_if_not_exists,
            &value.release,
        ) {
            (Some(_), Some(_), Some(_), Some(_), Some(_), Some(_)) => Ok(Self { inner: value }),
            _ => {
                sedona_internal_err!("Can't import released or uninitialized SedonaCObjectStore")
            }
        }
    }
}

impl Display for ImportedObjectStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(display_fn) = self.inner.display {
            let c_str = unsafe { display_fn(&self.inner) };
            if c_str.is_null() {
                write!(f, "<null>")
            } else {
                let rust_str = unsafe { CStr::from_ptr(c_str) }.to_string_lossy();
                write!(f, "{}", rust_str)
            }
        } else {
            write!(f, "ImportedObjectStore(<invalid>)")
        }
    }
}

impl Debug for ImportedObjectStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(debug_fn) = self.inner.debug {
            let c_str = unsafe { debug_fn(&self.inner) };
            if c_str.is_null() {
                f.debug_struct("ImportedObjectStore")
                    .field("inner", &"<null>")
                    .finish()
            } else {
                let rust_str = unsafe { CStr::from_ptr(c_str) }.to_string_lossy();
                f.debug_struct("ImportedObjectStore")
                    .field("inner", &rust_str)
                    .finish()
            }
        } else {
            f.debug_struct("ImportedObjectStore")
                .field("inner", &"<invalid>")
                .finish()
        }
    }
}

// -----------------------------------------------------------------------------
// Imported Async Handler (Rust creates, C calls back)
// -----------------------------------------------------------------------------

/// Handler for async results when importing a C object store.
///
/// This is created by the Rust import side and passed to C. When the async
/// operation completes, C calls back into this handler.
struct ImportedAsyncResultHandler {
    sender: Option<oneshot::Sender<std::result::Result<(), (c_int, String)>>>,
}

impl ImportedAsyncResultHandler {
    fn new(sender: oneshot::Sender<std::result::Result<(), (c_int, String)>>) -> Self {
        Self {
            sender: Some(sender),
        }
    }
}

impl From<ImportedAsyncResultHandler> for SedonaCAsyncResultHandler {
    fn from(value: ImportedAsyncResultHandler) -> Self {
        let box_value = Box::new(value);
        Self {
            on_success: Some(c_imported_handler_on_success),
            on_error: Some(c_imported_handler_on_error),
            release: Some(c_imported_handler_release),
            private_data: Box::leak(box_value) as *mut ImportedAsyncResultHandler as *mut c_void,
        }
    }
}

/// C callback for success (called by C implementation)
unsafe extern "C" fn c_imported_handler_on_success(
    self_: *mut SedonaCAsyncResultHandler,
    _data: *const u8,
    _data_len: usize,
) {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ImportedAsyncResultHandler)
        .as_mut()
        .unwrap();

    // Send success through the channel
    if let Some(sender) = private_data.sender.take() {
        let _ = sender.send(Ok(()));
    }
}

/// C callback for error (called by C implementation)
unsafe extern "C" fn c_imported_handler_on_error(
    self_: *mut SedonaCAsyncResultHandler,
    code: c_int,
    message: *const c_char,
) {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ImportedAsyncResultHandler)
        .as_mut()
        .unwrap();

    // Extract error message
    let msg = if message.is_null() {
        String::new()
    } else {
        CStr::from_ptr(message).to_string_lossy().into_owned()
    };

    // Send error through the channel
    if let Some(sender) = private_data.sender.take() {
        let _ = sender.send(Err((code, msg)));
    }
}

/// C callback for release (called by C implementation after on_success/on_error)
unsafe extern "C" fn c_imported_handler_release(self_: *mut SedonaCAsyncResultHandler) {
    assert!(!self_.is_null());
    let self_ref = self_.as_mut().unwrap();

    assert!(!self_ref.private_data.is_null());
    let boxed = Box::from_raw(self_ref.private_data as *mut ImportedAsyncResultHandler);
    drop(boxed);

    self_ref.private_data = null_mut();
    self_ref.release = None;
}

#[async_trait]
impl ObjectStore for ImportedObjectStore {
    async fn put_opts(
        &self,
        _location: &Path,
        _payload: PutPayload,
        _opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        todo!()
    }

    async fn put_multipart_opts(
        &self,
        _location: &Path,
        _opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        todo!()
    }

    async fn get_opts(
        &self,
        _location: &Path,
        _options: GetOptions,
    ) -> object_store::Result<GetResult> {
        todo!()
    }

    async fn delete(&self, location: &Path) -> object_store::Result<()> {
        // Convert location to C string
        let location_cstr =
            CString::new(location.as_ref()).map_err(|e| object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: e.into(),
            })?;

        // Create a oneshot channel to receive the result
        let (tx, rx) = oneshot::channel::<std::result::Result<(), (c_int, String)>>();

        // Create the handler that signals the channel
        let handler = ImportedAsyncResultHandler::new(tx);
        let mut ffi_handler: SedonaCAsyncResultHandler = handler.into();

        // Call the C delete function
        let delete_fn = self.inner.delete.expect("delete callback missing");
        let code = unsafe { delete_fn(&self.inner, location_cstr.as_ptr(), &mut ffi_handler) };

        if code != 0 {
            // Operation failed to start - clean up handler manually
            // Note: we don't call release since the C side didn't accept the handler
            std::mem::forget(ffi_handler);
            return Err(object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: format!("delete failed to start: errno {}", code).into(),
            });
        }

        // Await the result from the callback
        // Note: the handler is now owned by the C side, which will call release
        std::mem::forget(ffi_handler);

        match rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err((code, msg))) => Err(object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: format!("delete failed ({}): {}", code, msg).into(),
            }),
            Err(_) => Err(object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: "delete callback channel cancelled".into(),
            }),
        }
    }

    fn list(&self, _prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        todo!()
    }

    async fn list_with_delimiter(
        &self,
        _prefix: Option<&Path>,
    ) -> object_store::Result<ListResult> {
        todo!()
    }

    async fn copy(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        // Convert paths to C strings
        let from_cstr = CString::new(from.as_ref()).map_err(|e| object_store::Error::Generic {
            store: "ImportedObjectStore",
            source: e.into(),
        })?;
        let to_cstr = CString::new(to.as_ref()).map_err(|e| object_store::Error::Generic {
            store: "ImportedObjectStore",
            source: e.into(),
        })?;

        // Create a oneshot channel to receive the result
        let (tx, rx) = oneshot::channel::<std::result::Result<(), (c_int, String)>>();

        // Create the handler that signals the channel
        let handler = ImportedAsyncResultHandler::new(tx);
        let mut ffi_handler: SedonaCAsyncResultHandler = handler.into();

        // Call the C copy function
        let copy_fn = self.inner.copy.expect("copy callback missing");
        let code = unsafe {
            copy_fn(
                &self.inner,
                from_cstr.as_ptr(),
                to_cstr.as_ptr(),
                &mut ffi_handler,
            )
        };

        if code != 0 {
            std::mem::forget(ffi_handler);
            return Err(object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: format!("copy failed to start: errno {}", code).into(),
            });
        }

        std::mem::forget(ffi_handler);

        match rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err((code, msg))) => Err(object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: format!("copy failed ({}): {}", code, msg).into(),
            }),
            Err(_) => Err(object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: "copy callback channel cancelled".into(),
            }),
        }
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        // Convert paths to C strings
        let from_cstr = CString::new(from.as_ref()).map_err(|e| object_store::Error::Generic {
            store: "ImportedObjectStore",
            source: e.into(),
        })?;
        let to_cstr = CString::new(to.as_ref()).map_err(|e| object_store::Error::Generic {
            store: "ImportedObjectStore",
            source: e.into(),
        })?;

        // Create a oneshot channel to receive the result
        let (tx, rx) = oneshot::channel::<std::result::Result<(), (c_int, String)>>();

        // Create the handler that signals the channel
        let handler = ImportedAsyncResultHandler::new(tx);
        let mut ffi_handler: SedonaCAsyncResultHandler = handler.into();

        // Call the C copy_if_not_exists function
        let copy_fn = self
            .inner
            .copy_if_not_exists
            .expect("copy_if_not_exists callback missing");
        let code = unsafe {
            copy_fn(
                &self.inner,
                from_cstr.as_ptr(),
                to_cstr.as_ptr(),
                &mut ffi_handler,
            )
        };

        if code != 0 {
            std::mem::forget(ffi_handler);
            return Err(object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: format!("copy_if_not_exists failed to start: errno {}", code).into(),
            });
        }

        std::mem::forget(ffi_handler);

        match rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err((code, msg))) => Err(object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: format!("copy_if_not_exists failed ({}): {}", code, msg).into(),
            }),
            Err(_) => Err(object_store::Error::Generic {
                store: "ImportedObjectStore",
                source: "copy_if_not_exists callback channel cancelled".into(),
            }),
        }
    }
}

/// Wrapper around an [Arc<dyn ObjectStore>] that may be used to export an existing
/// ObjectStore across an FFI boundary using the [SedonaCObjectStore]
pub struct ExportedObjectStore {
    inner: Arc<dyn ObjectStore>,
    runtime_handle: Handle,
    display_cache: CString,
    debug_cache: CString,
}

impl ExportedObjectStore {
    /// Create a new ExportedObjectStore with an explicit runtime handle.
    ///
    /// The runtime handle is used to spawn async operations when C calls
    /// into this object store.
    pub fn new(store: Arc<dyn ObjectStore>, runtime_handle: Handle) -> Self {
        let display_str = format!("{}", store);
        let debug_str = format!("{:?}", store);
        ExportedObjectStore {
            inner: store,
            runtime_handle,
            display_cache: CString::new(display_str).unwrap_or_default(),
            debug_cache: CString::new(debug_str).unwrap_or_default(),
        }
    }
}

impl From<Arc<dyn ObjectStore>> for ExportedObjectStore {
    /// Create an ExportedObjectStore using the current tokio runtime.
    ///
    /// # Panics
    /// Panics if not called from within a tokio runtime context.
    fn from(value: Arc<dyn ObjectStore>) -> Self {
        Self::new(value, Handle::current())
    }
}

impl From<ExportedObjectStore> for SedonaCObjectStore {
    fn from(value: ExportedObjectStore) -> Self {
        let box_value = Box::new(value);
        Self {
            display: Some(c_object_store_display),
            debug: Some(c_object_store_debug),
            delete: Some(c_object_store_delete),
            copy: Some(c_object_store_copy),
            copy_if_not_exists: Some(c_object_store_copy_if_not_exists),
            release: Some(c_object_store_release),
            private_data: Box::leak(box_value) as *mut ExportedObjectStore as *mut c_void,
        }
    }
}

/// C callable wrapper to get the display string
unsafe extern "C" fn c_object_store_display(self_: *const SedonaCObjectStore) -> *const c_char {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedObjectStore)
        .as_ref()
        .unwrap();
    private_data.display_cache.as_ptr()
}

/// C callable wrapper to get the debug string
unsafe extern "C" fn c_object_store_debug(self_: *const SedonaCObjectStore) -> *const c_char {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedObjectStore)
        .as_ref()
        .unwrap();
    private_data.debug_cache.as_ptr()
}

/// C callable wrapper for async delete operation
///
/// This spawns the async delete on the runtime and calls the handler callbacks
/// when complete.
unsafe extern "C" fn c_object_store_delete(
    self_: *const SedonaCObjectStore,
    location: *const c_char,
    handler: *mut SedonaCAsyncResultHandler,
) -> c_int {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedObjectStore)
        .as_ref()
        .unwrap();

    // Validate handler
    if handler.is_null() {
        return libc::EINVAL;
    }
    let handler_ref = handler.as_ref().unwrap();
    if handler_ref.on_success.is_none()
        || handler_ref.on_error.is_none()
        || handler_ref.release.is_none()
    {
        return libc::EINVAL;
    }

    // Convert location to Rust Path
    if location.is_null() {
        return libc::EINVAL;
    }
    let location_str = match CStr::from_ptr(location).to_str() {
        Ok(s) => s,
        Err(_) => return libc::EINVAL,
    };
    let path = Path::from(location_str);

    // Clone what we need for the async task
    let store = private_data.inner.clone();
    let handle = private_data.runtime_handle.clone();

    // Wrap the handler pointer in a Send-safe wrapper
    let handler_wrapper = SendableHandler(handler);

    // Spawn the async work - only capture the wrapper, not the raw pointer
    handle.spawn(async move {
        let result = store.delete(&path).await;

        // Now extract the handler pointer after the await
        handler_wrapper.complete(result);
    });

    0 // Operation accepted
}

/// C callable wrapper for async copy operation
///
/// This spawns the async copy on the runtime and calls the handler callbacks
/// when complete.
unsafe extern "C" fn c_object_store_copy(
    self_: *const SedonaCObjectStore,
    from: *const c_char,
    to: *const c_char,
    handler: *mut SedonaCAsyncResultHandler,
) -> c_int {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedObjectStore)
        .as_ref()
        .unwrap();

    // Validate handler
    if handler.is_null() {
        return libc::EINVAL;
    }
    let handler_ref = handler.as_ref().unwrap();
    if handler_ref.on_success.is_none()
        || handler_ref.on_error.is_none()
        || handler_ref.release.is_none()
    {
        return libc::EINVAL;
    }

    // Convert paths to Rust Path
    if from.is_null() || to.is_null() {
        return libc::EINVAL;
    }
    let from_str = match CStr::from_ptr(from).to_str() {
        Ok(s) => s,
        Err(_) => return libc::EINVAL,
    };
    let to_str = match CStr::from_ptr(to).to_str() {
        Ok(s) => s,
        Err(_) => return libc::EINVAL,
    };
    let from_path = Path::from(from_str);
    let to_path = Path::from(to_str);

    // Clone what we need for the async task
    let store = private_data.inner.clone();
    let handle = private_data.runtime_handle.clone();

    // Wrap the handler pointer in a Send-safe wrapper
    let handler_wrapper = SendableHandler(handler);

    // Spawn the async work
    handle.spawn(async move {
        let result = store.copy(&from_path, &to_path).await;
        handler_wrapper.complete(result);
    });

    0 // Operation accepted
}

/// C callable wrapper for async copy_if_not_exists operation
///
/// This spawns the async copy_if_not_exists on the runtime and calls the handler
/// callbacks when complete.
unsafe extern "C" fn c_object_store_copy_if_not_exists(
    self_: *const SedonaCObjectStore,
    from: *const c_char,
    to: *const c_char,
    handler: *mut SedonaCAsyncResultHandler,
) -> c_int {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedObjectStore)
        .as_ref()
        .unwrap();

    // Validate handler
    if handler.is_null() {
        return libc::EINVAL;
    }
    let handler_ref = handler.as_ref().unwrap();
    if handler_ref.on_success.is_none()
        || handler_ref.on_error.is_none()
        || handler_ref.release.is_none()
    {
        return libc::EINVAL;
    }

    // Convert paths to Rust Path
    if from.is_null() || to.is_null() {
        return libc::EINVAL;
    }
    let from_str = match CStr::from_ptr(from).to_str() {
        Ok(s) => s,
        Err(_) => return libc::EINVAL,
    };
    let to_str = match CStr::from_ptr(to).to_str() {
        Ok(s) => s,
        Err(_) => return libc::EINVAL,
    };
    let from_path = Path::from(from_str);
    let to_path = Path::from(to_str);

    // Clone what we need for the async task
    let store = private_data.inner.clone();
    let handle = private_data.runtime_handle.clone();

    // Wrap the handler pointer in a Send-safe wrapper
    let handler_wrapper = SendableHandler(handler);

    // Spawn the async work
    handle.spawn(async move {
        let result = store.copy_if_not_exists(&from_path, &to_path).await;
        handler_wrapper.complete(result);
    });

    0 // Operation accepted
}

/// Wrapper to allow sending a handler pointer across threads.
///
/// # Safety
/// The caller must ensure the handler pointer remains valid until the
/// async operation completes and callbacks are invoked.
struct SendableHandler(*mut SedonaCAsyncResultHandler);
unsafe impl Send for SendableHandler {}

impl SendableHandler {
    /// Complete the async operation and call the appropriate callbacks.
    ///
    /// # Safety
    /// The handler pointer must still be valid.
    unsafe fn complete(self, result: object_store::Result<()>) {
        let handler = self.0;

        // Call the appropriate callback
        match result {
            Ok(()) => {
                if let Some(on_success) = (*handler).on_success {
                    on_success(handler, null(), 0);
                }
            }
            Err(e) => {
                let msg = CString::new(e.to_string()).unwrap_or_default();
                if let Some(on_error) = (*handler).on_error {
                    on_error(handler, libc::EIO, msg.as_ptr());
                }
            }
        }

        // Always call release
        if let Some(release) = (*handler).release {
            release(handler);
        }
    }
}

/// C callable wrapper called when this value is dropped via FFI
unsafe extern "C" fn c_object_store_release(self_: *mut SedonaCObjectStore) {
    assert!(!self_.is_null());
    let self_ref = self_.as_mut().unwrap();

    assert!(!self_ref.private_data.is_null());
    let boxed = Box::from_raw(self_ref.private_data as *mut ExportedObjectStore);
    drop(boxed);

    self_ref.private_data = null_mut();
    self_ref.release = None;
}

#[cfg(test)]
mod test {
    use super::*;
    use object_store::memory::InMemory;
    use object_store::PutPayload;

    #[tokio::test]
    async fn test_export_import_display_debug() {
        // Create an in-memory object store and export it
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let original_display = format!("{}", store);
        let original_debug = format!("{:?}", store);

        let exported = ExportedObjectStore::from(store);
        let ffi_store: SedonaCObjectStore = exported.into();

        // Import it back
        let imported = ImportedObjectStore::try_from(ffi_store).unwrap();

        // Verify Display works correctly (passes through unchanged)
        let imported_display = format!("{}", imported);
        assert_eq!(original_display, imported_display);

        // Verify Debug wraps in ImportedObjectStore
        let imported_debug = format!("{:?}", imported);
        assert!(imported_debug.contains("ImportedObjectStore"));
        assert!(imported_debug.contains(&original_debug));
    }

    #[test]
    fn test_invalid_import() {
        // Create an empty/released SedonaCObjectStore
        let empty = SedonaCObjectStore::default();
        let result = ImportedObjectStore::try_from(empty);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_roundtrip() {
        // Create an in-memory object store with a file
        let store = Arc::new(InMemory::new());
        let path = Path::from("test/file.txt");
        store
            .put(&path, PutPayload::from_static(b"hello"))
            .await
            .unwrap();

        // Verify the file exists
        assert!(store.get(&path).await.is_ok());

        // Export and import the store
        let exported = ExportedObjectStore::from(store.clone() as Arc<dyn ObjectStore>);
        let ffi_store: SedonaCObjectStore = exported.into();
        let imported = ImportedObjectStore::try_from(ffi_store).unwrap();

        // Delete through the FFI boundary
        imported.delete(&path).await.unwrap();

        // Verify the file is gone (check via original store reference)
        assert!(store.get(&path).await.is_err());
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        // Create an empty in-memory object store
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        // Export and import
        let exported = ExportedObjectStore::from(store);
        let ffi_store: SedonaCObjectStore = exported.into();
        let imported = ImportedObjectStore::try_from(ffi_store).unwrap();

        // Delete a non-existent file
        // Note: InMemory store succeeds even for non-existent files
        let path = Path::from("nonexistent.txt");
        let result = imported.delete(&path).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_copy_roundtrip() {
        // Create an in-memory object store with a file
        let store = Arc::new(InMemory::new());
        let src_path = Path::from("source.txt");
        let dst_path = Path::from("dest.txt");
        store
            .put(&src_path, PutPayload::from_static(b"hello copy"))
            .await
            .unwrap();

        // Export and import the store
        let exported = ExportedObjectStore::from(store.clone() as Arc<dyn ObjectStore>);
        let ffi_store: SedonaCObjectStore = exported.into();
        let imported = ImportedObjectStore::try_from(ffi_store).unwrap();

        // Copy through the FFI boundary
        imported.copy(&src_path, &dst_path).await.unwrap();

        // Verify both files exist with same content
        let src_data = store.get(&src_path).await.unwrap().bytes().await.unwrap();
        let dst_data = store.get(&dst_path).await.unwrap().bytes().await.unwrap();
        assert_eq!(src_data, dst_data);
        assert_eq!(&src_data[..], b"hello copy");
    }

    #[tokio::test]
    async fn test_copy_if_not_exists_success() {
        // Create an in-memory object store with a file
        let store = Arc::new(InMemory::new());
        let src_path = Path::from("source.txt");
        let dst_path = Path::from("new_dest.txt");
        store
            .put(&src_path, PutPayload::from_static(b"hello"))
            .await
            .unwrap();

        // Export and import the store
        let exported = ExportedObjectStore::from(store.clone() as Arc<dyn ObjectStore>);
        let ffi_store: SedonaCObjectStore = exported.into();
        let imported = ImportedObjectStore::try_from(ffi_store).unwrap();

        // Copy should succeed since dest doesn't exist
        imported
            .copy_if_not_exists(&src_path, &dst_path)
            .await
            .unwrap();

        // Verify dest was created
        assert!(store.get(&dst_path).await.is_ok());
    }

    #[tokio::test]
    async fn test_copy_if_not_exists_fails_when_exists() {
        // Create an in-memory object store with source and dest files
        let store = Arc::new(InMemory::new());
        let src_path = Path::from("source.txt");
        let dst_path = Path::from("existing_dest.txt");
        store
            .put(&src_path, PutPayload::from_static(b"source"))
            .await
            .unwrap();
        store
            .put(&dst_path, PutPayload::from_static(b"existing"))
            .await
            .unwrap();

        // Export and import the store
        let exported = ExportedObjectStore::from(store.clone() as Arc<dyn ObjectStore>);
        let ffi_store: SedonaCObjectStore = exported.into();
        let imported = ImportedObjectStore::try_from(ffi_store).unwrap();

        // Copy should fail since dest exists
        let result = imported.copy_if_not_exists(&src_path, &dst_path).await;
        assert!(result.is_err());

        // Verify dest was not overwritten
        let dst_data = store.get(&dst_path).await.unwrap().bytes().await.unwrap();
        assert_eq!(&dst_data[..], b"existing");
    }
}
