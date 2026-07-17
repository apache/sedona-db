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

use std::{iter::zip, sync::Arc};

use arrow_array::ArrayRef;
use arrow_schema::FieldRef;
use datafusion_common::{config::ConfigOptions, Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, Expr, Literal, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
};
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::SedonaType;
use sedona_testing_base::create::create_scalar;

pub(crate) struct ScalarUdfTester {
    udf: ScalarUDF,
    arg_types: Vec<SedonaType>,
    config_options: Arc<ConfigOptions>,
}

impl ScalarUdfTester {
    pub(crate) fn new(udf: ScalarUDF, arg_types: Vec<SedonaType>) -> Self {
        Self {
            udf,
            arg_types,
            config_options: Arc::new(ConfigOptions::default()),
        }
    }

    pub(crate) fn assert_return_type(&self, data_type: impl TryInto<SedonaType>) {
        let expected = match data_type.try_into() {
            Ok(t) => t,
            Err(_) => panic!("Failed to convert to SedonaType"),
        };
        assert_eq!(self.return_type().unwrap(), expected)
    }

    pub(crate) fn return_type(&self) -> Result<SedonaType> {
        let scalar_arguments = vec![None; self.arg_types.len()];
        self.return_type_with_scalars_inner(&scalar_arguments)
    }

    pub(crate) fn invoke_scalar(&self, arg: impl Literal) -> Result<ScalarValue> {
        let scalar_arg = Self::scalar_lit(arg, &self.arg_types[0])?;
        let return_type = self
            .return_type_with_scalars_inner(&[Some(scalar_arg.clone())])
            .ok();

        let args = vec![ColumnarValue::Scalar(scalar_arg)];
        if let ColumnarValue::Scalar(scalar) = self.invoke_with_return_type(args, return_type)? {
            Ok(scalar)
        } else {
            sedona_internal_err!("Expected scalar result from scalar invoke")
        }
    }

    pub(crate) fn invoke_scalar_scalar<T0: Literal, T1: Literal>(
        &self,
        arg0: T0,
        arg1: T1,
    ) -> Result<ScalarValue> {
        let scalar_arg0 = Self::scalar_lit(arg0, &self.arg_types[0])?;
        let scalar_arg1 = Self::scalar_lit(arg1, &self.arg_types[1])?;
        let return_type = self
            .return_type_with_scalars_inner(&[Some(scalar_arg0.clone()), Some(scalar_arg1.clone())])
            .ok();

        let args = vec![
            ColumnarValue::Scalar(scalar_arg0),
            ColumnarValue::Scalar(scalar_arg1),
        ];
        if let ColumnarValue::Scalar(scalar) = self.invoke_with_return_type(args, return_type)? {
            Ok(scalar)
        } else {
            sedona_internal_err!("Expected scalar result from binary scalar invoke")
        }
    }

    pub(crate) fn invoke_array_scalar(
        &self,
        array: ArrayRef,
        arg: impl Literal,
    ) -> Result<ArrayRef> {
        let mut args =
            vec![ColumnarValue::Array(array).cast_to(self.arg_types[0].storage_type(), None)?];
        args.push(Self::scalar_arg(arg, &self.arg_types[1])?);

        if let ColumnarValue::Array(array) = self.invoke(args)? {
            Ok(array)
        } else {
            sedona_internal_err!("Expected array result from array/scalar invoke")
        }
    }

    pub(crate) fn invoke_arrays(&self, arrays: Vec<ArrayRef>) -> Result<ArrayRef> {
        let args = zip(arrays, &self.arg_types)
            .map(|(array, sedona_type)| {
                ColumnarValue::Array(array).cast_to(sedona_type.storage_type(), None)
            })
            .collect::<Result<_>>()?;

        if let ColumnarValue::Array(array) = self.invoke(args)? {
            Ok(array)
        } else {
            sedona_internal_err!("Expected array result from array invoke")
        }
    }

    fn invoke(&self, args: Vec<ColumnarValue>) -> Result<ColumnarValue> {
        let scalar_args = args
            .iter()
            .map(|arg| match arg {
                ColumnarValue::Array(_) => None,
                ColumnarValue::Scalar(scalar_value) => Some(scalar_value.clone()),
            })
            .collect::<Vec<_>>();

        let return_type = self.return_type_with_scalars_inner(&scalar_args)?;
        self.invoke_with_return_type(args, Some(return_type))
    }

    fn invoke_with_return_type(
        &self,
        args: Vec<ColumnarValue>,
        return_type: Option<SedonaType>,
    ) -> Result<ColumnarValue> {
        assert_eq!(args.len(), self.arg_types.len(), "Unexpected arg length");

        let mut number_rows = 1;
        for arg in &args {
            if let ColumnarValue::Array(array) = arg {
                number_rows = array.len();
                break;
            }
        }

        let return_type = match return_type {
            Some(return_type) => return_type,
            None => self.return_type()?,
        };

        let args = ScalarFunctionArgs {
            args,
            arg_fields: self.arg_fields(),
            number_rows,
            return_field: return_type.to_storage_field("", true)?.into(),
            config_options: Arc::clone(&self.config_options),
        };

        self.udf.invoke_with_args(args)
    }

    fn return_type_with_scalars_inner(
        &self,
        scalar_arguments: &[Option<ScalarValue>],
    ) -> Result<SedonaType> {
        let arg_fields = self
            .arg_types
            .iter()
            .map(|sedona_type| sedona_type.to_storage_field("", true).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;

        let scalar_arguments_ref: Vec<Option<&ScalarValue>> =
            scalar_arguments.iter().map(|x| x.as_ref()).collect();
        let args = ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_arguments_ref,
        };
        let return_field = self.udf.return_field_from_args(args)?;
        SedonaType::from_storage_field(&return_field)
    }

    fn scalar_arg(arg: impl Literal, sedona_type: &SedonaType) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Scalar(Self::scalar_lit(arg, sedona_type)?))
    }

    fn scalar_lit(arg: impl Literal, sedona_type: &SedonaType) -> Result<ScalarValue> {
        if let Expr::Literal(scalar, _) = arg.lit() {
            let is_geometry_or_geography = match sedona_type {
                SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _) => true,
                SedonaType::Arrow(_) if sedona_type.is_item_crs() => true,
                _ => false,
            };

            if is_geometry_or_geography {
                if let ScalarValue::Utf8(expected_wkt) = scalar {
                    Ok(create_scalar(expected_wkt.as_deref(), sedona_type))
                } else if &scalar.data_type() == sedona_type.storage_type() {
                    Ok(scalar)
                } else if scalar.is_null() {
                    Ok(create_scalar(None, sedona_type))
                } else {
                    sedona_internal_err!("Can't interpret scalar {scalar} as type {sedona_type}")
                }
            } else {
                scalar.cast_to(sedona_type.storage_type())
            }
        } else {
            sedona_internal_err!("Can't use test scalar invoke where .lit() returns non-literal")
        }
    }

    fn arg_fields(&self) -> Vec<FieldRef> {
        self.arg_types
            .iter()
            .map(|data_type| data_type.to_storage_field("", false).map(Arc::new))
            .collect::<Result<Vec<_>>>()
            .unwrap()
    }
}
