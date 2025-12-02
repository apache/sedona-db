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
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    ArrayRef,
};
use arrow_schema::{ArrowError, Field};
use datafusion_common::{plan_err, Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::SedonaScalarKernel;
use sedona_schema::datatypes::SedonaType;
use std::{
    ffi::{c_int, CStr},
    fmt::Debug,
    ptr::null_mut,
};

use crate::extension::{SedonaCScalarUdf, SedonaCScalarUdfFactory};

pub struct ExtensionSedonaScalarKernel {
    inner: SedonaCScalarUdfFactory,
}

impl Debug for ExtensionSedonaScalarKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtensionSedonaScalarKernel")
            .field("inner", &"<SedonaCScalarUdfFactory>")
            .finish()
    }
}

impl SedonaScalarKernel for ExtensionSedonaScalarKernel {
    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        let mut inner_impl = CScalarUdfWrapper::try_new(&self.inner)?;
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

        let mut inner_impl = CScalarUdfWrapper::try_new(&self.inner)?;
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

struct CScalarUdfWrapper {
    inner: SedonaCScalarUdf,
}

impl CScalarUdfWrapper {
    fn try_new(factory: &SedonaCScalarUdfFactory) -> Result<Self> {
        if let Some(init) = factory.new_scalar_udf_impl {
            let mut inner = SedonaCScalarUdf::default();
            unsafe { init(factory, &mut inner) };
            Ok(Self { inner })
        } else {
            sedona_internal_err!("SedonaCScalarUdfFactory is not valid")
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

        let arg_fields = arg_types
            .iter()
            .map(|sedona_type| sedona_type.to_storage_field("", true))
            .collect::<Result<Vec<_>>>()?;
        let ffi_fields = arg_fields
            .iter()
            .map(FFI_ArrowSchema::try_from)
            .collect::<Result<Vec<_>, ArrowError>>()?;
        let ffi_field_ptrs = ffi_fields
            .iter()
            .map(|ffi_field| ffi_field as *const FFI_ArrowSchema)
            .collect::<Vec<_>>();

        let mut ffi_scalars = arg_scalars
            .iter()
            .map(|maybe_scalar| -> Result<Option<FFI_ArrowArray>> {
                if let Some(scalar) = maybe_scalar {
                    let array = scalar.to_array()?;
                    Ok(Some(FFI_ArrowArray::new(&array.to_data())))
                } else {
                    Ok(None)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let mut ffi_scalar_ptrs = ffi_scalars
            .iter_mut()
            .map(|maybe_ffi_scalar| match maybe_ffi_scalar {
                Some(ffi_scalar) => ffi_scalar as *mut FFI_ArrowArray,
                None => null_mut(),
            })
            .collect::<Vec<_>>();

        if let Some(init) = self.inner.init {
            let mut ffi_out = FFI_ArrowSchema::empty();
            let code = unsafe {
                init(
                    &mut self.inner,
                    ffi_field_ptrs.as_ptr(),
                    ffi_scalar_ptrs.as_mut_ptr(),
                    arg_types.len() as i64,
                    &mut ffi_out,
                )
            };
            if code == 0 {
                match Field::try_from(&ffi_out) {
                    Ok(field) => Ok(Some(SedonaType::from_storage_field(&field)?)),
                    Err(_) => Ok(None),
                }
            } else {
                plan_err!("SedonaCScalarUdf::init failed: {}", self.last_error(code))
            }
        } else {
            sedona_internal_err!("Invalid SedonaCScalarUdf")
        }
    }

    fn execute(
        &mut self,
        args: &[ColumnarValue],
        return_type: &SedonaType,
        num_rows: usize,
    ) -> Result<ArrayRef> {
        let arg_arrays = args
            .iter()
            .map(|arg| match arg {
                ColumnarValue::Array(array) => Ok(array.clone()),
                ColumnarValue::Scalar(scalar_value) => scalar_value.to_array(),
            })
            .collect::<Result<Vec<_>>>()?;
        let mut ffi_args = arg_arrays
            .iter()
            .map(|arg| FFI_ArrowArray::new(&arg.to_data()))
            .collect::<Vec<_>>();

        if let Some(execute) = self.inner.execute {
            let mut ffi_out = FFI_ArrowArray::empty();
            let code = unsafe {
                execute(
                    &mut self.inner,
                    &mut ffi_args.as_mut_ptr(),
                    args.len() as i64,
                    num_rows as i64,
                    &mut ffi_out,
                )
            };

            if code == 0 {
                let data = unsafe {
                    arrow_array::ffi::from_ffi_and_data_type(
                        ffi_out,
                        return_type.storage_type().clone(),
                    )?
                };
                Ok(arrow_array::make_array(data))
            } else {
                plan_err!("SedonaCScalarUdf::init failed: {}", self.last_error(code))
            }
        } else {
            sedona_internal_err!("Invalid SedonaCScalarUdf")
        }
    }

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
            "Invalid SedonaCScalarUdf".to_string()
        }
    }
}
