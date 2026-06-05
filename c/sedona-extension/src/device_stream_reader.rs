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

use std::ffi::CStr;
use std::sync::Arc;

use arrow_array::{
    ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema},
    ffi_stream::FFI_ArrowArrayStream,
    RecordBatch, RecordBatchReader, StructArray,
};
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};

use crate::extension_ffi::{FFI_ArrowDeviceArray, FFI_ArrowDeviceArrayStream, ARROW_DEVICE_CPU};

pub struct DeviceStreamReader {
    inner: FFI_ArrowDeviceArrayStream,
    schema: SchemaRef,
    schema_struct_type: DataType,
}

impl DeviceStreamReader {
    pub fn try_new(mut inner: FFI_ArrowDeviceArrayStream) -> Result<Self, ArrowError> {
        let get_schema = inner
            .get_schema
            .ok_or_else(|| ArrowError::CDataInterface("get_schema callback is null".to_string()))?;

        let mut ffi_schema = FFI_ArrowSchema::empty();
        let ret = unsafe { get_schema(&mut inner, &mut ffi_schema) };
        if ret != 0 {
            let error_msg = Self::get_last_error_static(&mut inner);
            return Err(ArrowError::CDataInterface(error_msg));
        }

        let schema = Schema::try_from(&ffi_schema)?;
        let schema_struct_type = DataType::Struct(schema.fields().iter().cloned().collect());

        Ok(Self {
            inner,
            schema: Arc::new(schema),
            schema_struct_type,
        })
    }

    fn get_last_error_static(inner: &mut FFI_ArrowDeviceArrayStream) -> String {
        if let Some(get_last_error) = inner.get_last_error {
            let err_ptr = unsafe { get_last_error(inner) };
            if !err_ptr.is_null() {
                let c_str = unsafe { CStr::from_ptr(err_ptr) };
                return c_str.to_string_lossy().into_owned();
            }
        }
        "Unknown error".to_string()
    }

    fn get_last_error(&mut self) -> String {
        Self::get_last_error_static(&mut self.inner)
    }
}

impl Iterator for DeviceStreamReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(get_next) = self.inner.get_next else {
            return Some(Err(ArrowError::CDataInterface(
                "get_next() is null".to_string(),
            )));
        };

        let mut device_array = FFI_ArrowDeviceArray {
            array: FFI_ArrowArray::empty(),
            device_id: 0,
            device_type: 0,
            sync_event: std::ptr::null_mut(),
        };

        let ret = unsafe { get_next(&mut self.inner, &mut device_array) };
        if ret != 0 {
            return Some(Err(ArrowError::CDataInterface(self.get_last_error())));
        }

        // Check if the stream is exhausted (release is null means empty/end of stream)
        if device_array.array.is_released() {
            return None;
        }

        // Convert device array to regular array (only supports CPU for now)
        let ffi_array: FFI_ArrowArray = match device_array.try_into() {
            Ok(arr) => arr,
            Err(e) => return Some(Err(e)),
        };

        // Import the array data
        let array_data =
            match unsafe { from_ffi_and_data_type(ffi_array, self.schema_struct_type.clone()) } {
                Ok(array_data) => array_data,
                Err(e) => return Some(Err(e)),
            };

        // Create RecordBatch from StructArray
        let struct_array: StructArray = array_data.into();
        Some(Ok(struct_array.into()))
    }
}

impl RecordBatchReader for DeviceStreamReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl From<FFI_ArrowArrayStream> for FFI_ArrowDeviceArrayStream {
    fn from(value: FFI_ArrowArrayStream) -> Self {
        let private_data = Box::new(ExportedDeviceReaderPrivate { inner: value });

        FFI_ArrowDeviceArrayStream {
            device_type: ARROW_DEVICE_CPU,
            get_schema: Some(exported_reader_get_schema),
            get_next: Some(exported_reader_get_next),
            get_last_error: Some(exported_reader_get_last_error),
            release: Some(exported_reader_release),
            private_data: Box::into_raw(private_data) as *mut std::ffi::c_void,
        }
    }
}

struct ExportedDeviceReaderPrivate {
    inner: FFI_ArrowArrayStream,
}

unsafe extern "C" fn exported_reader_get_schema(
    self_: *mut FFI_ArrowDeviceArrayStream,
    out: *mut FFI_ArrowSchema,
) -> std::ffi::c_int {
    if self_.is_null() || out.is_null() {
        return 1;
    }

    let private = &mut *((*self_).private_data as *mut ExportedDeviceReaderPrivate);
    let inner = &mut private.inner as *mut FFI_ArrowArrayStream as *mut ArrowArrayStreamInternal;

    if let Some(get_schema) = (*inner).get_schema {
        get_schema(inner, out)
    } else {
        1
    }
}

unsafe extern "C" fn exported_reader_get_next(
    self_: *mut FFI_ArrowDeviceArrayStream,
    out: *mut FFI_ArrowDeviceArray,
) -> std::ffi::c_int {
    if self_.is_null() || out.is_null() {
        return 1;
    }

    let private = &mut *((*self_).private_data as *mut ExportedDeviceReaderPrivate);
    let inner = &mut private.inner as *mut FFI_ArrowArrayStream as *mut ArrowArrayStreamInternal;

    let Some(get_next) = (*inner).get_next else {
        return 1;
    };

    let mut ffi_array = FFI_ArrowArray::empty();
    let ret = get_next(inner, &mut ffi_array);
    if ret != 0 {
        return ret;
    }

    // Wrap the array in a device array
    let device_array = FFI_ArrowDeviceArray {
        array: ffi_array,
        device_id: -1,
        device_type: ARROW_DEVICE_CPU,
        sync_event: std::ptr::null_mut(),
    };
    std::ptr::write(out, device_array);
    0
}

unsafe extern "C" fn exported_reader_get_last_error(
    self_: *mut FFI_ArrowDeviceArrayStream,
) -> *const std::ffi::c_char {
    if self_.is_null() {
        return std::ptr::null();
    }

    let private = &mut *((*self_).private_data as *mut ExportedDeviceReaderPrivate);
    let inner = &mut private.inner as *mut FFI_ArrowArrayStream as *mut ArrowArrayStreamInternal;

    if let Some(get_last_error) = (*inner).get_last_error {
        get_last_error(inner)
    } else {
        std::ptr::null()
    }
}

unsafe extern "C" fn exported_reader_release(self_: *mut FFI_ArrowDeviceArrayStream) {
    if self_.is_null() {
        return;
    }

    let stream = &mut *self_;
    if stream.private_data.is_null() {
        return;
    }

    // Drop the private data (which will drop the inner FFI_ArrowArrayStream)
    let _ = Box::from_raw(stream.private_data as *mut ExportedDeviceReaderPrivate);
    stream.private_data = std::ptr::null_mut();
    stream.release = None;
}

/// Internal representation of FFI_ArrowArrayStream with public fields.
/// Duplicated here because arrow-rs doesn't expose the fields publicly.
#[repr(C)]
struct ArrowArrayStreamInternal {
    get_schema: Option<
        unsafe extern "C" fn(
            self_: *mut ArrowArrayStreamInternal,
            out: *mut FFI_ArrowSchema,
        ) -> std::ffi::c_int,
    >,
    get_next: Option<
        unsafe extern "C" fn(
            self_: *mut ArrowArrayStreamInternal,
            out: *mut FFI_ArrowArray,
        ) -> std::ffi::c_int,
    >,
    get_last_error: Option<
        unsafe extern "C" fn(self_: *mut ArrowArrayStreamInternal) -> *const std::ffi::c_char,
    >,
    release: Option<unsafe extern "C" fn(self_: *mut ArrowArrayStreamInternal)>,
    private_data: *mut std::ffi::c_void,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatchIterator, StringArray};
    use arrow_schema::Field;

    fn make_test_batches() -> (SchemaRef, Vec<RecordBatch>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), None])),
            ],
        )
        .unwrap();

        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![4, 5])),
                Arc::new(StringArray::from(vec![Some("d"), Some("e")])),
            ],
        )
        .unwrap();

        (schema, vec![batch1, batch2])
    }

    #[test]
    fn test_roundtrip_two_batches() {
        let (schema, batches) = make_test_batches();
        let original_batches = batches.clone();

        // Create a RecordBatchReader
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema.clone());

        // Export to FFI_ArrowArrayStream, then to FFI_ArrowDeviceArrayStream
        let array_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        let device_stream: FFI_ArrowDeviceArrayStream = array_stream.into();

        // Import back via DeviceStreamReader
        let imported_reader = DeviceStreamReader::try_new(device_stream).unwrap();

        // Verify schema
        assert_eq!(imported_reader.schema(), schema);

        // Collect and verify batches
        let imported_batches: Vec<RecordBatch> =
            imported_reader.into_iter().map(|r| r.unwrap()).collect();

        assert_eq!(imported_batches.len(), original_batches.len());
        for (imported, original) in imported_batches.iter().zip(original_batches.iter()) {
            assert_eq!(imported, original);
        }
    }

    /// A RecordBatchReader that yields one batch then errors
    struct ErroringReader {
        schema: SchemaRef,
        yielded_first: bool,
        first_batch: Option<RecordBatch>,
    }

    impl ErroringReader {
        fn new(schema: SchemaRef, first_batch: RecordBatch) -> Self {
            Self {
                schema,
                yielded_first: false,
                first_batch: Some(first_batch),
            }
        }
    }

    impl Iterator for ErroringReader {
        type Item = Result<RecordBatch, ArrowError>;

        fn next(&mut self) -> Option<Self::Item> {
            if !self.yielded_first {
                self.yielded_first = true;
                Some(Ok(self.first_batch.take().unwrap()))
            } else {
                Some(Err(ArrowError::ComputeError(
                    "intentional test error".to_string(),
                )))
            }
        }
    }

    impl RecordBatchReader for ErroringReader {
        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }
    }

    #[test]
    fn test_roundtrip_with_error() {
        let (schema, batches) = make_test_batches();
        let first_batch = batches[0].clone();

        // Create an erroring reader
        let reader = ErroringReader::new(schema.clone(), first_batch.clone());

        // Export to FFI_ArrowArrayStream, then to FFI_ArrowDeviceArrayStream
        let array_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        let device_stream: FFI_ArrowDeviceArrayStream = array_stream.into();

        // Import back via DeviceStreamReader
        let mut imported_reader = DeviceStreamReader::try_new(device_stream).unwrap();

        // First batch should succeed
        let result1 = imported_reader.next();
        assert!(result1.is_some());
        let batch1 = result1.unwrap();
        assert!(batch1.is_ok());
        assert_eq!(batch1.unwrap(), first_batch);

        // Second call should return an error
        let result2 = imported_reader.next();
        assert!(result2.is_some());
        let batch2 = result2.unwrap();
        assert!(batch2.is_err());
        let err = batch2.unwrap_err();
        assert!(err.to_string().contains("intentional test error"));
    }
}
