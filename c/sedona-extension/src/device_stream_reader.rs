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
    RecordBatch, RecordBatchReader, StructArray,
};
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};

use crate::extension_ffi::{FFI_ArrowDeviceArray, FFI_ArrowDeviceArrayStream};

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

struct ExportedDeviceReaderPrivate {
    inner: Box<dyn RecordBatchReader + Send>
}

fn exported_reader_get_schema(
            self_: *mut FFI_ArrowDeviceArrayStream,
            out: *mut FFI_ArrowSchema,
        ) -> std::ffi::c_int {

        }


