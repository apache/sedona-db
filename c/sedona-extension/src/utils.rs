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

//! Utilities for converting between Arrow FFI streams and DataFusion streams.

use std::ffi::{c_int, CString};
use std::ptr::null_mut;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{ArrowError, DataType, Field, SchemaRef};
use datafusion_common::{exec_err, Result};
use datafusion_physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::extension::{
    SedonaCError, SedonaCExecutionPlan, SedonaCExecutionPlanArgs, SedonaCTableProvider,
};

/// Success return code for FFI functions.
pub const ERRNO_OK: c_int = 0;

/// Get the schema for a property from a [SedonaCExecutionPlan].
///
/// Returns the DataType describing the property's data type.
/// If `get_property_schema` is not implemented, defaults to Binary.
fn get_plan_property_data_type(plan: &SedonaCExecutionPlan, property: &str) -> Result<DataType> {
    let Some(get_property_schema) = plan.get_property_schema else {
        // Default to Binary if get_property_schema is not implemented
        return Ok(DataType::Binary);
    };

    let property_cstr = CString::new(property)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid property name: {}", e))?;

    let mut ffi_schema = FFI_ArrowSchema::empty();
    let mut err = SedonaCError::default();

    let code =
        unsafe { get_property_schema(plan, property_cstr.as_ptr(), &mut ffi_schema, &mut err) };

    if code != ERRNO_OK {
        return sedona_internal_err!("Failed to get property schema for '{}': {}", property, err);
    }

    // Try to convert the FFI schema to a Field
    let field = Field::try_from(&ffi_schema).map_err(|e| {
        sedona_internal_datafusion_err!("Failed to parse property schema for '{}': {}", property, e)
    })?;
    Ok(field.data_type().clone())
}

/// Call `get_property` on a [SedonaCExecutionPlan] and deserialize the result.
///
/// This handles the common pattern of:
/// 1. Extracting the get_property function pointer
/// 2. Calling it with a property name and optional serializable args
/// 3. Parsing the binary array result
/// 4. Deserializing from JSON to the target type
///
/// # Arguments
///
/// * `plan` - The execution plan to query
/// * `property` - The property name to retrieve
/// * `args` - Optional arguments to pass (will be serialized to JSON bytes)
///
/// # Errors
///
/// Returns an error if:
/// - The plan does not have a `get_property` callback
/// - The FFI call fails
/// - The result cannot be parsed as a binary array
/// - The JSON deserialization fails
pub fn get_plan_property<T, A>(
    plan: &SedonaCExecutionPlan,
    property: &str,
    args: Option<&A>,
) -> Result<T>
where
    T: DeserializeOwned,
    A: Serialize,
{
    let Some(get_property) = plan.get_property else {
        return sedona_internal_err!("SedonaCExecutionPlan does not have get_property");
    };

    let property_cstr = CString::new(property)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid property name: {}", e))?;

    // Serialize args if provided
    let args_bytes = match args {
        Some(a) => serde_json::to_vec(a)
            .map_err(|e| sedona_internal_datafusion_err!("Failed to serialize args: {}", e))?,
        None => Vec::new(),
    };

    let mut ffi_args = SedonaCExecutionPlanArgs {
        args: if args_bytes.is_empty() {
            std::ptr::null()
        } else {
            args_bytes.as_ptr()
        },
        args_len: args_bytes.len(),
        exec_plans: std::ptr::null(),
        num_exec_plans: 0,
        exprs: std::ptr::null(),
        num_exprs: 0,
        reserved: null_mut(),
    };

    let mut ffi_array = arrow_array::ffi::FFI_ArrowArray::empty();
    let mut err = SedonaCError::default();

    let code = unsafe {
        get_property(
            plan,
            property_cstr.as_ptr(),
            &mut ffi_args,
            &mut ffi_array,
            &mut err,
        )
    };

    if code != ERRNO_OK {
        return sedona_internal_err!("Failed to get property '{}': {}", property, err);
    }

    // Get the property schema to know how to interpret the array
    let data_type = get_plan_property_data_type(plan, property)?;

    // Parse the array to get the JSON bytes
    parse_ffi_array(ffi_array, &data_type)
}

/// Parse an FFI array containing JSON and deserialize to the target type.
fn parse_ffi_array<T: DeserializeOwned>(
    ffi_array: arrow_array::ffi::FFI_ArrowArray,
    data_type: &DataType,
) -> Result<T> {
    let bytes = parse_ffi_array_to_bytes(ffi_array, data_type)?;
    serde_json::from_slice::<T>(&bytes)
        .map_err(|e| sedona_internal_datafusion_err!("Failed to deserialize property: {}", e))
}

/// Parse an FFI array and return the raw bytes.
fn parse_ffi_array_to_bytes(
    ffi_array: arrow_array::ffi::FFI_ArrowArray,
    data_type: &DataType,
) -> Result<Vec<u8>> {
    let data = unsafe { arrow_array::ffi::from_ffi_and_data_type(ffi_array, data_type.clone())? };
    let array = arrow_array::make_array(data);

    // Handle different array types
    match data_type {
        DataType::Binary => {
            let binary_array = array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!("Expected binary array from get_property")
                })?;
            Ok(binary_array.value(0).to_vec())
        }
        DataType::LargeBinary => {
            let binary_array = array
                .as_any()
                .downcast_ref::<arrow_array::LargeBinaryArray>()
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!("Expected large binary array from get_property")
                })?;
            Ok(binary_array.value(0).to_vec())
        }
        DataType::Utf8 => {
            let string_array = array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!("Expected string array from get_property")
                })?;
            Ok(string_array.value(0).as_bytes().to_vec())
        }
        DataType::LargeUtf8 => {
            let string_array = array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!("Expected large string array from get_property")
                })?;
            Ok(string_array.value(0).as_bytes().to_vec())
        }
        _ => sedona_internal_err!("Unsupported data type for property: {:?}", data_type),
    }
}

/// Get a string property from a [SedonaCExecutionPlan].
///
/// This is a convenience wrapper around [get_plan_property] for string values.
pub fn get_plan_string_property(plan: &SedonaCExecutionPlan, property: &str) -> Result<String> {
    let Some(get_property) = plan.get_property else {
        return sedona_internal_err!("SedonaCExecutionPlan does not have get_property");
    };

    let property_cstr = CString::new(property)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid property name: {}", e))?;

    let mut ffi_args = SedonaCExecutionPlanArgs {
        args: std::ptr::null(),
        args_len: 0,
        exec_plans: std::ptr::null(),
        num_exec_plans: 0,
        exprs: std::ptr::null(),
        num_exprs: 0,
        reserved: null_mut(),
    };

    let mut ffi_array = arrow_array::ffi::FFI_ArrowArray::empty();
    let mut err = SedonaCError::default();

    let code = unsafe {
        get_property(
            plan,
            property_cstr.as_ptr(),
            &mut ffi_args,
            &mut ffi_array,
            &mut err,
        )
    };

    if code != ERRNO_OK {
        return sedona_internal_err!("Failed to get '{}': {}", property, err);
    }

    // Get the property schema to know how to interpret the array
    let data_type = get_plan_property_data_type(plan, property)?;

    let bytes = parse_ffi_array_to_bytes(ffi_array, &data_type)?;
    String::from_utf8(bytes)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid UTF-8 in '{}': {}", property, e))
}

/// Get the schema for a property from a [SedonaCTableProvider].
///
/// Returns the DataType describing the property's data type.
/// If `get_property_schema` is not implemented, defaults to Binary.
fn get_table_provider_property_data_type(
    provider: &SedonaCTableProvider,
    property: &str,
) -> Result<DataType> {
    let Some(get_property_schema) = provider.get_property_schema else {
        // Default to Binary if get_property_schema is not implemented
        return Ok(DataType::Binary);
    };

    let property_cstr = CString::new(property)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid property name: {}", e))?;

    let mut ffi_schema = FFI_ArrowSchema::empty();
    let mut err = SedonaCError::default();

    let code =
        unsafe { get_property_schema(provider, property_cstr.as_ptr(), &mut ffi_schema, &mut err) };

    if code != ERRNO_OK {
        return sedona_internal_err!("Failed to get property schema for '{}': {}", property, err);
    }

    // Try to convert the FFI schema to a Field
    let field = Field::try_from(&ffi_schema).map_err(|e| {
        sedona_internal_datafusion_err!("Failed to parse property schema for '{}': {}", property, e)
    })?;
    Ok(field.data_type().clone())
}

/// Get a string property from a [SedonaCTableProvider].
pub fn get_table_provider_string_property(
    provider: &SedonaCTableProvider,
    property: &str,
) -> Result<String> {
    let Some(get_property) = provider.get_property else {
        return sedona_internal_err!("SedonaCTableProvider does not have get_property");
    };

    let property_cstr = CString::new(property)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid property name: {}", e))?;

    let mut ffi_args = SedonaCExecutionPlanArgs {
        args: std::ptr::null(),
        args_len: 0,
        exec_plans: std::ptr::null(),
        num_exec_plans: 0,
        exprs: std::ptr::null(),
        num_exprs: 0,
        reserved: null_mut(),
    };

    let mut ffi_array = arrow_array::ffi::FFI_ArrowArray::empty();
    let mut err = SedonaCError::default();

    let code = unsafe {
        get_property(
            provider,
            property_cstr.as_ptr(),
            &mut ffi_args,
            &mut ffi_array,
            &mut err,
        )
    };

    if code != ERRNO_OK {
        return sedona_internal_err!("Failed to get '{}': {}", property, err);
    }

    // Get the property schema to know how to interpret the array
    let data_type = get_table_provider_property_data_type(provider, property)?;

    let bytes = parse_ffi_array_to_bytes(ffi_array, &data_type)?;
    String::from_utf8(bytes)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid UTF-8 in '{}': {}", property, e))
}

/// A cancellation check callback for FFI operations.
///
/// Returns `true` if the operation should be cancelled, `false` to continue.
/// This allows Python, R, ADBC, and other runtimes to integrate their own
/// cancellation mechanisms.
pub type CancelChecker = Box<dyn Fn() -> bool + Send + Sync>;

/// A RecordBatchReader that lazily polls a SendableRecordBatchStream.
///
/// This allows exporting a DataFusion stream over FFI without collecting all
/// batches upfront. Used when exporting execution plans across FFI boundaries.
pub struct StreamingRecordBatchReader {
    schema: SchemaRef,
    stream: SendableRecordBatchStream,
    runtime: tokio::runtime::Handle,
    cancel_checker: Option<CancelChecker>,
    cancelled: bool,
}

impl StreamingRecordBatchReader {
    /// Create a new StreamingRecordBatchReader from a SendableRecordBatchStream.
    pub fn new(stream: SendableRecordBatchStream, runtime: tokio::runtime::Handle) -> Self {
        Self {
            schema: stream.schema(),
            stream,
            runtime,
            cancel_checker: None,
            cancelled: false,
        }
    }

    /// Create a new StreamingRecordBatchReader with a cancellation checker.
    ///
    /// The cancellation checker is called before each batch is fetched. If it
    /// returns `true`, iteration stops with a cancellation error on the next
    /// call and `None` on subsequent calls.
    pub fn with_cancel_checker(
        stream: SendableRecordBatchStream,
        runtime: tokio::runtime::Handle,
        cancel_checker: CancelChecker,
    ) -> Self {
        Self {
            schema: stream.schema(),
            stream,
            runtime,
            cancel_checker: Some(cancel_checker),
            cancelled: false,
        }
    }
}

impl Iterator for StreamingRecordBatchReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        // If already cancelled, return None to stop iteration
        if self.cancelled {
            return None;
        }

        // Check for cancellation before fetching the next batch
        if let Some(ref checker) = self.cancel_checker {
            if checker() {
                self.cancelled = true;
                return Some(Err(ArrowError::ExternalError(Box::new(
                    std::io::Error::new(std::io::ErrorKind::Interrupted, "Operation cancelled"),
                ))));
            }
        }

        let stream = &mut self.stream;
        let runtime = &self.runtime;

        std::thread::scope(|s| {
            s.spawn(|| {
                runtime.block_on(async {
                    match stream.next().await {
                        Some(Ok(batch)) => Some(Ok(batch)),
                        Some(Err(e)) => Some(Err(ArrowError::ExternalError(Box::new(e)))),
                        None => None,
                    }
                })
            })
            .join()
            .unwrap_or_else(|_| {
                Some(Err(ArrowError::InvalidArgumentError(
                    "Iterator thread panicked".to_string(),
                )))
            })
        })
    }
}

impl RecordBatchReader for StreamingRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Convert an FFI ArrowArrayStream into a SendableRecordBatchStream.
///
/// This is the inverse of StreamingRecordBatchReader - it takes an FFI stream
/// and converts it back to a DataFusion stream. Used when importing execution
/// plans from across FFI boundaries.
///
/// # Safety
///
/// The caller must ensure that the FFI stream pointer is valid and properly
/// initialized.
pub unsafe fn ffi_stream_to_sendable(
    ffi_stream: &mut FFI_ArrowArrayStream,
) -> Result<SendableRecordBatchStream> {
    ffi_stream_to_sendable_with_cancel(ffi_stream, None)
}

/// Convert an FFI ArrowArrayStream into a SendableRecordBatchStream with cancellation support.
///
/// The cancellation checker is called before each batch is read. If it returns
/// `true`, the stream yields a cancellation error.
///
/// # Safety
///
/// The caller must ensure that the FFI stream pointer is valid and properly
/// initialized.
pub unsafe fn ffi_stream_to_sendable_with_cancel(
    ffi_stream: &mut FFI_ArrowArrayStream,
    cancel_checker: Option<CancelChecker>,
) -> Result<SendableRecordBatchStream> {
    let reader = arrow_array::ffi_stream::ArrowArrayStreamReader::from_raw(ffi_stream)?;

    let schema = reader.schema();
    let stream = futures::stream::iter(reader).map(move |result| {
        // Check for cancellation before yielding each batch
        if let Some(ref checker) = cancel_checker {
            if checker() {
                return exec_err!("Operation cancelled");
            }
        }
        result.map_err(|e| datafusion_common::DataFusionError::ArrowError(Box::new(e), None))
    });

    Ok(Box::pin(
        datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::{Field, Schema};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// Create a slow stream that yields batches with a configurable delay.
    fn create_slow_stream(
        num_batches: usize,
        delay_ms: u64,
    ) -> (SchemaRef, SendableRecordBatchStream) {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let schema_clone = schema.clone();

        let stream = futures::stream::iter(0..num_batches).then(move |i| {
            let schema = schema_clone.clone();
            async move {
                if delay_ms > 0 {
                    // Use std::thread::sleep because tokio::time::sleep requires
                    // the runtime to poll, which can deadlock with block_on
                    std::thread::sleep(Duration::from_millis(delay_ms));
                }
                let array = Int32Array::from(vec![i as i32]);
                Ok(RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap())
            }
        });

        (
            schema.clone(),
            Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream),
            ),
        )
    }

    #[tokio::test]
    async fn test_streaming_reader_basic() {
        let runtime = tokio::runtime::Handle::current();
        let (_schema, stream) = create_slow_stream(5, 10);

        let reader = StreamingRecordBatchReader::new(stream, runtime);
        let batches: Vec<_> = reader.collect();

        assert_eq!(batches.len(), 5);
        for (i, batch) in batches.iter().enumerate() {
            let batch = batch.as_ref().unwrap();
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            assert_eq!(array.value(0), i as i32);
        }
    }

    #[tokio::test]
    async fn test_streaming_reader_cancel() {
        let runtime = tokio::runtime::Handle::current();
        let (_schema, stream) = create_slow_stream(10, 50);

        // Cancel after reading 3 batches
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let cancel_checker: CancelChecker = Box::new(move || {
            let count = counter_clone.fetch_add(1, Ordering::SeqCst);
            count >= 3
        });

        let reader =
            StreamingRecordBatchReader::with_cancel_checker(stream, runtime, cancel_checker);
        let batches: Vec<_> = reader.collect();

        // Should have 3 successful batches + 1 cancellation error
        assert_eq!(batches.len(), 4);

        // First 3 should be Ok
        for (i, batch) in batches.iter().enumerate().take(3) {
            assert!(batch.is_ok(), "batch {} should be Ok", i);
        }

        // Last one should be a cancellation error
        let last = batches.last().unwrap();
        assert!(last.is_err());
        let err = last.as_ref().unwrap_err();
        assert!(
            err.to_string().contains("cancelled"),
            "error should mention cancellation: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_ffi_stream_to_sendable_basic() {
        use arrow_array::ffi_stream::FFI_ArrowArrayStream;

        let runtime = tokio::runtime::Handle::current();
        let (_schema, stream) = create_slow_stream(5, 10);

        // Export to FFI stream
        let reader = StreamingRecordBatchReader::new(stream, runtime.clone());
        let mut ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));

        // Import back
        let imported = unsafe { ffi_stream_to_sendable(&mut ffi_stream).unwrap() };

        // Collect results
        let batches: Vec<_> = imported.collect::<Vec<_>>().await;

        assert_eq!(batches.len(), 5);
        for (i, batch) in batches.iter().enumerate() {
            let batch = batch.as_ref().unwrap();
            let array = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            assert_eq!(array.value(0), i as i32);
        }
    }

    #[tokio::test]
    async fn test_ffi_stream_to_sendable_cancel() {
        use arrow_array::ffi_stream::FFI_ArrowArrayStream;

        let runtime = tokio::runtime::Handle::current();
        let (_schema, stream) = create_slow_stream(10, 50);

        // Export to FFI stream (no cancellation on export side)
        let reader = StreamingRecordBatchReader::new(stream, runtime.clone());
        let mut ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));

        // Cancel after reading 3 batches on import side
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();
        let cancel_checker: CancelChecker =
            Box::new(move || cancelled_clone.load(Ordering::SeqCst));

        let imported = unsafe {
            ffi_stream_to_sendable_with_cancel(&mut ffi_stream, Some(cancel_checker)).unwrap()
        };

        // Collect with cancellation after 3 batches, stop on first error
        let mut batches = Vec::new();
        let mut stream = imported;
        while let Some(result) = stream.next().await {
            let is_err = result.is_err();
            batches.push(result);
            if batches.len() == 3 {
                cancelled.store(true, Ordering::SeqCst);
            }
            if is_err {
                break; // Stop on first error
            }
        }

        // Should have 3 successful batches + 1 cancellation error
        assert_eq!(batches.len(), 4);

        // First 3 should be Ok
        for (i, batch) in batches.iter().enumerate().take(3) {
            assert!(batch.is_ok(), "batch {} should be Ok", i);
        }

        // Last one should be a cancellation error
        let last = batches.last().unwrap();
        assert!(last.is_err());
        let err = last.as_ref().unwrap_err();
        assert!(
            err.to_string().contains("cancelled"),
            "error should mention cancellation: {}",
            err
        );
    }
}
