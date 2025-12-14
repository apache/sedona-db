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

use arrow_array::{
    ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema},
    make_array, ArrayRef,
};
use arrow_schema::{ArrowError, Field};
use datafusion_common::{plan_err, DataFusionError, Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::datatypes::SedonaType;
use std::{
    ffi::{c_char, c_int, c_void, CStr, CString},
    fmt::Debug,
    iter::zip,
    ptr::{null, null_mut, swap_nonoverlapping},
    str::FromStr,
};

use crate::extension::{ffi_arrow_schema_is_valid, SedonaCScalarKernel, SedonaCScalarKernelImpl};

/// Wrapper around a [SedonaCScalarKernel] that implements [SedonaScalarKernel]
///
/// This is the means by which a kernel implementation may be imported from a
/// C implementation.
pub struct ImportedScalarKernel {
    inner: SedonaCScalarKernel,
    function_name: Option<String>,
}

impl Debug for ImportedScalarKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedScalarKernel")
            .field("inner", &"<SedonaCScalarKernel>")
            .finish()
    }
}

impl TryFrom<SedonaCScalarKernel> for ImportedScalarKernel {
    type Error = DataFusionError;

    fn try_from(value: SedonaCScalarKernel) -> Result<Self> {
        match (&value.function_name, &value.new_impl, &value.release) {
            (Some(function_name), Some(_), Some(_)) => {
                let name_ptr = unsafe { function_name(&value) };
                let name = if name_ptr.is_null() {
                    None
                } else {
                    Some(
                        unsafe { CStr::from_ptr(name_ptr) }
                            .to_string_lossy()
                            .into_owned(),
                    )
                };

                Ok(Self {
                    inner: value,
                    function_name: name,
                })
            }
            _ => sedona_internal_err!("Can't import released or uninitialized SedonaCScalarKernel"),
        }
    }
}

impl ImportedScalarKernel {
    pub fn function_name(&self) -> Option<&str> {
        self.function_name.as_deref()
    }
}

impl SedonaScalarKernel for ImportedScalarKernel {
    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        let mut inner_impl = CScalarKernelImplWrapper::try_new(&self.inner)?;
        inner_impl.init(args, scalar_args)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        num_rows: usize,
    ) -> Result<ColumnarValue> {
        let arg_scalars = args
            .iter()
            .map(|arg| {
                if let ColumnarValue::Scalar(scalar) = arg {
                    Some(scalar)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut inner_impl = CScalarKernelImplWrapper::try_new(&self.inner)?;
        inner_impl.init(arg_types, &arg_scalars)?;
        let result_array = inner_impl.execute(args, return_type, num_rows)?;
        for arg in args {
            if let ColumnarValue::Array(_) = arg {
                return Ok(ColumnarValue::Array(result_array));
            }
        }

        if result_array.len() != 1 {
            sedona_internal_err!(
                "Expected scalar result but got result with length {}",
                result_array.len()
            )
        } else {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result_array,
                0,
            )?))
        }
    }

    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
        sedona_internal_err!(
            "Should not be called because return_type_from_args_and_scalars() is implemented"
        )
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("Should not be called because invoke_batch_from_args() is implemented")
    }
}

/// Wrapper class handling the verbose details of preparing and executing FFI calls
/// for the [SedonaCScalarKernelImpl]
struct CScalarKernelImplWrapper {
    inner: SedonaCScalarKernelImpl,
}

impl CScalarKernelImplWrapper {
    fn try_new(factory: &SedonaCScalarKernel) -> Result<Self> {
        if let Some(init) = factory.new_impl {
            let mut inner = SedonaCScalarKernelImpl::default();
            unsafe { init(factory, &mut inner) };
            Ok(Self { inner })
        } else {
            sedona_internal_err!("SedonaCScalarKernel is not valid")
        }
    }

    fn init(
        &mut self,
        arg_types: &[SedonaType],
        arg_scalars: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        if arg_types.len() != arg_scalars.len() {
            return sedona_internal_err!("field/scalar lengths must be identical");
        }

        // Convert arg_types to Vec<Field>
        let arg_fields = arg_types
            .iter()
            .map(|sedona_type| sedona_type.to_storage_field("", true))
            .collect::<Result<Vec<_>>>()?;

        // Convert arg types to Vec<FFI_ArrowSchema>
        let ffi_fields = arg_fields
            .iter()
            .map(FFI_ArrowSchema::try_from)
            .collect::<Result<Vec<_>, ArrowError>>()?;

        // Convert arg types to Vec<*const FFI_ArrowSchema>
        let ffi_field_ptrs = ffi_fields
            .iter()
            .map(|ffi_field| ffi_field as *const FFI_ArrowSchema)
            .collect::<Vec<_>>();

        // Convert arg_scalars to Vec<Option<FFI_ArrowArray>>
        let mut ffi_scalars = arg_scalars
            .iter()
            .map(|maybe_scalar| {
                if let Some(scalar) = maybe_scalar {
                    let array = scalar.to_array()?;
                    Ok(Some(FFI_ArrowArray::new(&array.to_data())))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Convert arg_scalars to Vec<*mut FFI_ArrowArray>
        let ffi_scalar_ptrs = ffi_scalars
            .iter_mut()
            .map(|maybe_ffi_scalar| match maybe_ffi_scalar {
                Some(ffi_scalar) => ffi_scalar as *mut FFI_ArrowArray,
                None => null_mut(),
            })
            .collect::<Vec<_>>();

        // Call the FFI implementation of init
        if let Some(init) = self.inner.init {
            let mut ffi_out = FFI_ArrowSchema::empty();
            let code = unsafe {
                init(
                    &mut self.inner,
                    ffi_field_ptrs.as_ptr(),
                    ffi_scalar_ptrs.as_ptr(),
                    arg_types.len() as i64,
                    &mut ffi_out,
                )
            };

            // On success, convert the output to SedonaType. If the implementation
            // returned a "released" schema, this is the equivalent to our return_type()
            // returning None (for "this kernel doesn't apply").
            // On error, query the FFI implementation for the last error string.
            if code == 0 {
                if ffi_arrow_schema_is_valid(&ffi_out) {
                    let field = Field::try_from(&ffi_out)?;
                    Ok(Some(SedonaType::from_storage_field(&field)?))
                } else {
                    Ok(None)
                }
            } else {
                plan_err!(
                    "SedonaCScalarKernelImpl::init failed: {}",
                    self.last_error(code)
                )
            }
        } else {
            sedona_internal_err!("Invalid SedonaCScalarKernelImpl")
        }
    }

    fn execute(
        &mut self,
        args: &[ColumnarValue],
        return_type: &SedonaType,
        num_rows: usize,
    ) -> Result<ArrayRef> {
        // Convert args to Vec<ArrayRef>
        let arg_arrays = args
            .iter()
            .map(|arg| match arg {
                ColumnarValue::Array(array) => Ok(array.clone()),
                ColumnarValue::Scalar(scalar_value) => scalar_value.to_array(),
            })
            .collect::<Result<Vec<_>>>()?;

        // Convert args to Vec<FFI_ArrowArray>
        let mut ffi_args = arg_arrays
            .iter()
            .map(|arg| FFI_ArrowArray::new(&arg.to_data()))
            .collect::<Vec<_>>();
        let ffi_arg_ptrs = ffi_args
            .iter_mut()
            .map(|arg| arg as *mut FFI_ArrowArray)
            .collect::<Vec<_>>();

        // Call the FFI implementation of execute()
        if let Some(execute) = self.inner.execute {
            let mut ffi_out = FFI_ArrowArray::empty();
            let code = unsafe {
                execute(
                    &mut self.inner,
                    ffi_arg_ptrs.as_ptr(),
                    args.len() as i64,
                    num_rows as i64,
                    &mut ffi_out,
                )
            };

            // On success, convert the result to an ArrayRef.
            // On error, query the FFI implementation for the last error string.
            if code == 0 {
                let data = unsafe {
                    arrow_array::ffi::from_ffi_and_data_type(
                        ffi_out,
                        return_type.storage_type().clone(),
                    )?
                };
                Ok(arrow_array::make_array(data))
            } else {
                plan_err!(
                    "SedonaCScalarKernelImpl::execute failed: {}",
                    self.last_error(code)
                )
            }
        } else {
            sedona_internal_err!("Invalid SedonaCScalarKernelImpl")
        }
    }

    /// Helper to get the last error from the FFI implementation as a Rust String
    fn last_error(&mut self, code: c_int) -> String {
        if let Some(get_last_error) = self.inner.get_last_error {
            let c_err = unsafe { get_last_error(&mut self.inner) };
            if c_err.is_null() {
                format!("({code})")
            } else {
                unsafe { CStr::from_ptr(c_err) }
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "Invalid SedonaCScalarKernelImpl".to_string()
        }
    }
}

/// Wrapper around a [ScalarKernelRef] that may be used to export an existing
/// kernel across an FFI boundary using the [SedonaCScalarKernel]
pub struct ExportedScalarKernel {
    inner: ScalarKernelRef,
    function_name: Option<CString>,
}

impl From<ScalarKernelRef> for ExportedScalarKernel {
    fn from(value: ScalarKernelRef) -> Self {
        ExportedScalarKernel {
            inner: value,
            function_name: None,
        }
    }
}

impl From<ExportedScalarKernel> for SedonaCScalarKernel {
    fn from(value: ExportedScalarKernel) -> Self {
        let box_value = Box::new(value);
        Self {
            function_name: Some(c_factory_function_name),
            new_impl: Some(c_factory_new_impl),
            release: Some(c_factory_release),
            private_data: Box::leak(box_value) as *mut ExportedScalarKernel as *mut c_void,
        }
    }
}

impl ExportedScalarKernel {
    /// Add a function name to this exported kernel
    ///
    /// This ensures the kernel will be registered with the appropriate function
    /// when passed across a boundary.
    pub fn with_function_name(self, function_name: impl AsRef<str>) -> Self {
        Self {
            inner: self.inner,
            function_name: Some(CString::from_str(function_name.as_ref()).unwrap()),
        }
    }

    fn new_impl(&self) -> ExportedScalarKernelImpl {
        ExportedScalarKernelImpl::new(self.inner.clone())
    }
}

/// C callable wrapper to expose [ExportedScalarKernel::function_name]
unsafe extern "C" fn c_factory_function_name(self_: *const SedonaCScalarKernel) -> *const c_char {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedScalarKernel)
        .as_ref()
        .unwrap();
    if let Some(function_name) = &private_data.function_name {
        function_name.as_ptr()
    } else {
        null()
    }
}

/// C callable wrapper around [ExportedScalarKernel::new_impl]
unsafe extern "C" fn c_factory_new_impl(
    self_: *const SedonaCScalarKernel,
    out: *mut SedonaCScalarKernelImpl,
) {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedScalarKernel)
        .as_ref()
        .unwrap();
    *out = SedonaCScalarKernelImpl::from(private_data.new_impl())
}

/// C Callable wrapper called when this value is dropped via FFI
unsafe extern "C" fn c_factory_release(self_: *mut SedonaCScalarKernel) {
    assert!(!self_.is_null());
    let self_ref = self_.as_mut().unwrap();

    assert!(!self_ref.private_data.is_null());
    let boxed = Box::from_raw(self_ref.private_data as *mut ExportedScalarKernel);
    drop(boxed);

    self_ref.private_data = null_mut();
    self_ref.release = None;
}

/// Rust-backed implementation of [SedonaCScalarKernelImpl]
struct ExportedScalarKernelImpl {
    inner: ScalarKernelRef,
    last_arg_types: Option<Vec<SedonaType>>,
    last_return_type: Option<SedonaType>,
    last_error: CString,
}

impl From<ExportedScalarKernelImpl> for SedonaCScalarKernelImpl {
    fn from(value: ExportedScalarKernelImpl) -> Self {
        let box_value = Box::new(value);
        Self {
            init: Some(c_kernel_init),
            execute: Some(c_kernel_execute),
            get_last_error: Some(c_kernel_last_error),
            release: Some(c_kernel_release),
            private_data: Box::leak(box_value) as *mut ExportedScalarKernelImpl as *mut c_void,
        }
    }
}

impl ExportedScalarKernelImpl {
    fn new(kernel: ScalarKernelRef) -> Self {
        Self {
            inner: kernel,
            last_arg_types: None,
            last_return_type: None,
            last_error: CString::default(),
        }
    }

    fn init(
        &mut self,
        ffi_types: &[*const FFI_ArrowSchema],
        ffi_scalar_args: &[*mut FFI_ArrowArray],
    ) -> Result<Option<FFI_ArrowSchema>> {
        // Convert the input types to Vec<Field>
        let arg_fields = ffi_types
            .iter()
            .map(|ptr| {
                if let Some(ffi_schema) = unsafe { ptr.as_ref() } {
                    Field::try_from(ffi_schema)
                } else {
                    Err(ArrowError::CDataInterface(
                        "FFI_ArrowSchema is NULL".to_string(),
                    ))
                }
            })
            .collect::<Result<Vec<_>, ArrowError>>()?;

        // Convert the input types to Vec<SedonaType>
        let args = arg_fields
            .iter()
            .map(SedonaType::from_storage_field)
            .collect::<Result<Vec<_>>>()?;

        // Convert the scalar arguments to Vec<Option<ArrayRef>>
        let arg_arrays = zip(ffi_scalar_args, &args)
            .map(|(ptr, arg)| {
                if ptr.is_null() {
                    Ok(None)
                } else {
                    let owned_ffi_array = unsafe { FFI_ArrowArray::from_raw(*ptr) };
                    let data = unsafe {
                        from_ffi_and_data_type(owned_ffi_array, arg.storage_type().clone())?
                    };
                    Ok(Some(make_array(data)))
                }
            })
            .collect::<Result<Vec<_>, ArrowError>>()?;

        // Convert the scalar arguments to Vec<Option<ScalarValue>>
        let scalar_args = arg_arrays
            .iter()
            .map(|maybe_array| {
                if let Some(array) = maybe_array {
                    Ok(Some(ScalarValue::try_from_array(array, 0)?))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Convert the scalar arguments to Vec<Option<&ScalarValue>>
        let scalar_arg_refs = scalar_args
            .iter()
            .map(|arg| arg.as_ref())
            .collect::<Vec<_>>();

        // Call the implementation
        let maybe_return_type = self
            .inner
            .return_type_from_args_and_scalars(&args, &scalar_arg_refs)?;

        // Convert the result to FFI_ArrowSchema (if not None)
        let return_ffi_schema = if let Some(return_type) = &maybe_return_type {
            let return_field = return_type.to_storage_field("", true)?;
            let return_ffi_schema = FFI_ArrowSchema::try_from(&return_field)?;
            Some(return_ffi_schema)
        } else {
            None
        };

        // Save the argument types and return type for following calls to execute()
        self.last_arg_types.replace(args);
        self.last_return_type = maybe_return_type;

        Ok(return_ffi_schema)
    }

    fn execute(&self, ffi_args: &[*mut FFI_ArrowArray], num_rows: i64) -> Result<FFI_ArrowArray> {
        match (&self.last_arg_types, &self.last_return_type) {
            (Some(arg_types), Some(return_type)) => {
                // Resolve args as Vec<ArrayRef>
                let arg_arrays = zip(ffi_args, arg_types)
                    .map(|(ptr, arg)| {
                        let owned_ffi_array = unsafe { FFI_ArrowArray::from_raw(*ptr) };
                        let data = unsafe {
                            from_ffi_and_data_type(owned_ffi_array, arg.storage_type().clone())?
                        };
                        Ok(make_array(data))
                    })
                    .collect::<Result<Vec<_>, ArrowError>>()?;

                // Resolve args as Vec<ColumnarValue>
                let args = arg_arrays
                    .into_iter()
                    .map(|array| {
                        if array.len() as i64 == num_rows {
                            Ok(ColumnarValue::Array(array))
                        } else {
                            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                                &array, 0,
                            )?))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Call the implementation
                let result_value = self.inner.invoke_batch_from_args(
                    arg_types,
                    &args,
                    return_type,
                    num_rows as usize,
                )?;

                // Convert the result to an ArrayRef
                let result_array = match result_value {
                    ColumnarValue::Array(array) => array,
                    ColumnarValue::Scalar(scalar_value) => scalar_value.to_array()?,
                };

                // Convert the result to a FFI_ArrowArray
                let result_ffi_array = FFI_ArrowArray::new(&result_array.to_data());
                Ok(result_ffi_array)
            }
            _ => {
                sedona_internal_err!("Call to ExportedScalarKernel::execute() before init()")
            }
        }
    }
}

/// C callable wrapper around [ExportedScalarKernelImpl::init]
unsafe extern "C" fn c_kernel_init(
    self_: *mut SedonaCScalarKernelImpl,
    arg_types: *const *const FFI_ArrowSchema,
    scalar_args: *const *mut FFI_ArrowArray,
    n_args: i64,
    out: *mut FFI_ArrowSchema,
) -> c_int {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedScalarKernelImpl)
        .as_mut()
        .unwrap();

    let ffi_types = std::slice::from_raw_parts(arg_types, n_args as usize);
    let ffi_scalar_args = std::slice::from_raw_parts(scalar_args, n_args as usize);

    match private_data.init(ffi_types, ffi_scalar_args) {
        Ok(Some(mut return_ffi_schema)) => {
            swap_nonoverlapping(&mut return_ffi_schema as *mut _, out, 1);
            0
        }
        Ok(None) => {
            *out = FFI_ArrowSchema::empty();
            0
        }
        Err(err) => {
            private_data.last_error =
                CString::from_str(&err.message()).unwrap_or(CString::default());
            libc::EINVAL
        }
    }
}

/// C callable wrapper around [ExportedScalarKernelImpl::execute]
unsafe extern "C" fn c_kernel_execute(
    self_: *mut SedonaCScalarKernelImpl,
    args: *const *mut FFI_ArrowArray,
    n_args: i64,
    n_rows: i64,
    out: *mut FFI_ArrowArray,
) -> c_int {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedScalarKernelImpl)
        .as_mut()
        .unwrap();

    let ffi_args = std::slice::from_raw_parts(args, n_args as usize);
    match private_data.execute(ffi_args, n_rows) {
        Ok(mut ffi_array) => {
            swap_nonoverlapping(&mut ffi_array as *mut _, out, 1);
            0
        }
        Err(err) => {
            private_data.last_error =
                CString::from_str(&err.message()).unwrap_or(CString::default());
            libc::EINVAL
        }
    }
}

/// C Callable wrapper to retrieve the last error string
unsafe extern "C" fn c_kernel_last_error(self_: *mut SedonaCScalarKernelImpl) -> *const c_char {
    assert!(!self_.is_null());
    let self_ref = self_.as_ref().unwrap();

    assert!(!self_ref.private_data.is_null());
    let private_data = (self_ref.private_data as *mut ExportedScalarKernelImpl)
        .as_ref()
        .unwrap();
    private_data.last_error.as_ptr()
}

/// C Callable wrapper called when this value is dropped via FFI
unsafe extern "C" fn c_kernel_release(self_: *mut SedonaCScalarKernelImpl) {
    assert!(!self_.is_null());
    let self_ref = self_.as_mut().unwrap();

    assert!(!self_ref.private_data.is_null());
    let boxed = Box::from_raw(self_ref.private_data as *mut ExportedScalarKernelImpl);
    drop(boxed);

    self_ref.private_data = null_mut();
    self_ref.release = None;
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_schema::DataType;
    use datafusion_common::exec_err;
    use datafusion_expr::Volatility;
    use sedona_expr::scalar_udf::{SedonaScalarUDF, SimpleSedonaScalarKernel};
    use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};
    use sedona_testing::{create::create_array, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn ffi_roundtrip() {
        let kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
            Arc::new(|_, args| Ok(args[0].clone())),
        );

        let array_value = create_array(&[Some("POINT (0 1)"), None], &WKB_GEOMETRY);

        let udf_native = SedonaScalarUDF::new(
            "simple_udf",
            vec![kernel.clone()],
            Volatility::Immutable,
            None,
        );

        let tester = ScalarUdfTester::new(udf_native.into(), vec![WKB_GEOMETRY]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester.invoke_scalar("POINT (0 1)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 1)");

        assert_eq!(
            &tester.invoke_array(array_value.clone()).unwrap(),
            &array_value
        );

        let exported_kernel = ExportedScalarKernel::from(kernel.clone());
        let ffi_kernel = SedonaCScalarKernel::from(exported_kernel);
        let imported_kernel = ImportedScalarKernel::try_from(ffi_kernel).unwrap();

        let udf_from_ffi = SedonaScalarUDF::new(
            "simple_udf_from_ffi",
            vec![Arc::new(imported_kernel)],
            Volatility::Immutable,
            None,
        );

        let ffi_tester = ScalarUdfTester::new(udf_from_ffi.clone().into(), vec![WKB_GEOMETRY]);
        ffi_tester.assert_return_type(WKB_GEOMETRY);

        let result = ffi_tester.invoke_scalar("POINT (0 1)").unwrap();
        ffi_tester.assert_scalar_result_equals(result, "POINT (0 1)");

        assert_eq!(
            &ffi_tester.invoke_array(array_value.clone()).unwrap(),
            &array_value
        );

        // Check the case of a kernel that does not apply to input arguments
        let ffi_tester = ScalarUdfTester::new(udf_from_ffi.clone().into(), vec![]);
        let err = ffi_tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "simple_udf_from_ffi([]): No kernel matching arguments"
        );
    }

    #[test]
    fn named_kernel() {
        let kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
            Arc::new(|_, args| Ok(args[0].clone())),
        );

        // Without intervening, we have a None name
        let exported_kernel = ExportedScalarKernel::from(kernel.clone());
        let ffi_kernel = SedonaCScalarKernel::from(exported_kernel);
        let imported_kernel = ImportedScalarKernel::try_from(ffi_kernel).unwrap();
        assert!(imported_kernel.function_name().is_none());

        // If we set a function name, it should be roundtripped
        let exported_kernel =
            ExportedScalarKernel::from(kernel.clone()).with_function_name("foofy");
        let ffi_kernel = SedonaCScalarKernel::from(exported_kernel);
        let imported_kernel = ImportedScalarKernel::try_from(ffi_kernel).unwrap();
        assert_eq!(imported_kernel.function_name(), Some("foofy"));
    }

    #[test]
    fn invoke_batch_from_scalar() {
        let kernel = Arc::new(ReturnTypeFromScalars {}) as ScalarKernelRef;
        let exported_kernel = ExportedScalarKernel::from(kernel.clone());
        let ffi_kernel = SedonaCScalarKernel::from(exported_kernel);
        let imported_kernel = ImportedScalarKernel::try_from(ffi_kernel).unwrap();

        let udf_from_ffi = SedonaScalarUDF::new(
            "simple_udf_from_ffi",
            vec![Arc::new(imported_kernel)],
            Volatility::Immutable,
            None,
        );

        let ffi_tester = ScalarUdfTester::new(
            udf_from_ffi.clone().into(),
            vec![SedonaType::Arrow(DataType::Utf8)],
        );
        let return_type = ffi_tester.return_type_with_scalar(Some("foofy")).unwrap();
        assert_eq!(return_type, SedonaType::Arrow(DataType::Utf8));
    }

    #[derive(Debug)]
    struct ReturnTypeFromScalars {}

    impl SedonaScalarKernel for ReturnTypeFromScalars {
        fn return_type_from_args_and_scalars(
            &self,
            _args: &[SedonaType],
            scalar_args: &[Option<&ScalarValue>],
        ) -> Result<Option<SedonaType>> {
            if let Some(arg0) = scalar_args[0] {
                Ok(Some(SedonaType::Arrow(arg0.data_type())))
            } else {
                Ok(None)
            }
        }

        fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
            unreachable!()
        }

        fn invoke_batch(
            &self,
            _arg_types: &[SedonaType],
            _args: &[ColumnarValue],
        ) -> Result<ColumnarValue> {
            unreachable!()
        }
    }

    #[test]
    fn erroring_invoke_batch() {
        let kernel = SimpleSedonaScalarKernel::new_ref(
            ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
            Arc::new(|_, _args| exec_err!("this invoke_batch() always errors")),
        );

        let exported_kernel = ExportedScalarKernel::from(kernel.clone());
        let ffi_kernel = SedonaCScalarKernel::from(exported_kernel);
        let imported_kernel = ImportedScalarKernel::try_from(ffi_kernel).unwrap();

        let udf_from_ffi = SedonaScalarUDF::new(
            "simple_udf_from_ffi",
            vec![Arc::new(imported_kernel)],
            Volatility::Immutable,
            None,
        );

        let ffi_tester = ScalarUdfTester::new(udf_from_ffi.clone().into(), vec![WKB_GEOMETRY]);
        ffi_tester.assert_return_type(WKB_GEOMETRY);

        let err = ffi_tester.invoke_scalar("POINT (0 1)").unwrap_err();
        assert_eq!(
            err.message(),
            "SedonaCScalarKernelImpl::execute failed: this invoke_batch() always errors"
        );
    }

    #[test]
    fn erroring_return_type() {
        let kernel = Arc::new(ErroringReturnType {}) as ScalarKernelRef;
        let exported_kernel = ExportedScalarKernel::from(kernel.clone());
        let ffi_kernel = SedonaCScalarKernel::from(exported_kernel);
        let imported_kernel = ImportedScalarKernel::try_from(ffi_kernel).unwrap();

        let udf_from_ffi = SedonaScalarUDF::new(
            "simple_udf_from_ffi",
            vec![Arc::new(imported_kernel)],
            Volatility::Immutable,
            None,
        );

        let ffi_tester = ScalarUdfTester::new(udf_from_ffi.clone().into(), vec![WKB_GEOMETRY]);
        let err = ffi_tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "SedonaCScalarKernelImpl::init failed: this implementation of return_type always errors"
        );
    }

    #[derive(Debug)]
    struct ErroringReturnType {}

    impl SedonaScalarKernel for ErroringReturnType {
        fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
            plan_err!("this implementation of return_type always errors")
        }

        fn invoke_batch(
            &self,
            _arg_types: &[SedonaType],
            _args: &[ColumnarValue],
        ) -> Result<ColumnarValue> {
            unreachable!()
        }
    }
}
