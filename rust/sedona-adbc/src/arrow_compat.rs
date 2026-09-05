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

use std::sync::Arc;

use arrow_array::{Array, RecordBatch, RecordBatchReader, StructArray};
use arrow_array_adbc::{
    RecordBatch as AdbcRecordBatch, RecordBatchReader as AdbcRecordBatchReader,
    StructArray as AdbcStructArray,
};
use arrow_schema::{ArrowError, DataType, Schema};
use arrow_schema_adbc::{
    ArrowError as AdbcArrowError, DataType as AdbcDataType, Schema as AdbcSchema,
};

pub(crate) fn convert_schema(schema: &Schema) -> Result<AdbcSchema, AdbcArrowError> {
    let mut ffi_schema =
        arrow_schema::ffi::FFI_ArrowSchema::try_from(&DataType::Struct(schema.fields().clone()))
            .map_err(convert_error)?;
    let adbc_ffi_schema = unsafe {
        arrow_schema_adbc::ffi::FFI_ArrowSchema::from_raw(
            (&mut ffi_schema as *mut arrow_schema::ffi::FFI_ArrowSchema).cast(),
        )
    };

    match AdbcDataType::try_from(&adbc_ffi_schema)? {
        AdbcDataType::Struct(fields) => Ok(AdbcSchema::new(fields)),
        data_type => Err(AdbcArrowError::ParseError(format!(
            "expected struct schema, got {data_type:?}"
        ))),
    }
}

fn convert_batch(batch: RecordBatch) -> Result<AdbcRecordBatch, AdbcArrowError> {
    let struct_array = StructArray::from(batch);
    let (mut ffi_array, mut ffi_schema) =
        arrow_array::ffi::to_ffi(&struct_array.to_data()).map_err(convert_error)?;
    let adbc_ffi_array = unsafe {
        arrow_array_adbc::ffi::FFI_ArrowArray::from_raw(
            (&mut ffi_array as *mut arrow_array::ffi::FFI_ArrowArray).cast(),
        )
    };
    let adbc_ffi_schema = unsafe {
        arrow_schema_adbc::ffi::FFI_ArrowSchema::from_raw(
            (&mut ffi_schema as *mut arrow_schema::ffi::FFI_ArrowSchema).cast(),
        )
    };
    let data = unsafe { arrow_array_adbc::ffi::from_ffi(adbc_ffi_array, &adbc_ffi_schema) }?;
    Ok(AdbcRecordBatch::from(AdbcStructArray::from(data)))
}

fn convert_error(error: ArrowError) -> AdbcArrowError {
    AdbcArrowError::ExternalError(Box::new(error))
}

pub(crate) struct ArrowVersionAdapter<R> {
    inner: R,
    schema: Arc<AdbcSchema>,
}

impl<R: RecordBatchReader> ArrowVersionAdapter<R> {
    pub(crate) fn try_new(inner: R) -> Result<Self, AdbcArrowError> {
        let schema = Arc::new(convert_schema(inner.schema().as_ref())?);
        Ok(Self { inner, schema })
    }
}

impl<R: RecordBatchReader> Iterator for ArrowVersionAdapter<R> {
    type Item = Result<AdbcRecordBatch, AdbcArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|batch| batch.map_err(convert_error).and_then(convert_batch))
    }
}

impl<R: RecordBatchReader> AdbcRecordBatchReader for ArrowVersionAdapter<R> {
    fn schema(&self) -> Arc<AdbcSchema> {
        self.schema.clone()
    }
}
