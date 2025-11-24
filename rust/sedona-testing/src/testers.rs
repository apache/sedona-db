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

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{FieldRef, Schema};
use datafusion_common::{config::ConfigOptions, Result, ScalarValue};
use datafusion_expr::{
    function::{AccumulatorArgs, StateFieldsArgs},
    Accumulator, AggregateUDF, ColumnarValue, Expr, Literal, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDF,
};
use datafusion_physical_expr::{expressions::Column, PhysicalExpr};
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::SedonaType;

use crate::{
    compare::assert_scalar_equal,
    create::{create_array, create_scalar},
};

/// Low-level tester for aggregate functions
///
/// This struct provides a means by which to run a simple check of an
/// aggregate implementation by approximating one method DataFusion
/// might use to perform the aggregation. Whereas DataFusion may arrange
/// various calls to accumulate, state, and update_batch to optimize
/// for different cases, this tester is always created by aggregating
/// states that were in turn created from accumulating one batch.
///
/// This is not a replacement for testing at a higher level using
/// DataFusion's actual aggregate implementation but provides
/// a useful mechanism to ensure all the pieces of an accumulator
/// are plugged in.
pub struct AggregateUdfTester {
    udf: AggregateUDF,
    arg_types: Vec<SedonaType>,
}

impl AggregateUdfTester {
    /// Create a new tester
    pub fn new(udf: AggregateUDF, arg_types: Vec<SedonaType>) -> Self {
        Self { udf, arg_types }
    }

    /// Compute the return type
    pub fn return_type(&self) -> Result<SedonaType> {
        let arg_fields = self
            .arg_types
            .iter()
            .map(|arg_type| arg_type.to_storage_field("", true).map(Arc::new))
            .collect::<Result<Vec<_>>>()?;

        let out_field = self.udf.return_field(&arg_fields)?;
        SedonaType::from_storage_field(&out_field)
    }

    /// Perform a simple aggregation using WKT as geometry input
    pub fn aggregate_wkt(&self, batches: Vec<Vec<Option<&str>>>) -> Result<ScalarValue> {
        let batches_array = batches
            .into_iter()
            .map(|batch| create_array(&batch, &self.arg_types[0]))
            .collect::<Vec<_>>();
        self.aggregate(&batches_array)
    }

    /// Perform a simple aggregation
    ///
    /// Each batch in batches is accumulated with its own accumulator
    /// and serialized into its own state, after which the states are accumulated
    /// in batches of one. This has the effect of testing all the pieces of
    /// an aggregator in a somewhat configurable/predictable way.
    pub fn aggregate(&self, batches: &Vec<ArrayRef>) -> Result<ScalarValue> {
        let state_schema = Arc::new(Schema::new(self.state_fields()?));
        let mut state_accumulator = self.new_accumulator()?;

        for batch in batches {
            let mut batch_accumulator = self.new_accumulator()?;
            batch_accumulator.update_batch(std::slice::from_ref(batch))?;
            let state_batch_of_one = RecordBatch::try_new(
                state_schema.clone(),
                batch_accumulator
                    .state()?
                    .into_iter()
                    .map(|v| v.to_array())
                    .collect::<Result<Vec<_>>>()?,
            )?;
            state_accumulator.merge_batch(state_batch_of_one.columns())?;
        }

        state_accumulator.evaluate()
    }

    fn new_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        let mock_schema = Schema::new(self.arg_fields());
        let exprs = (0..self.arg_types.len())
            .map(|i| -> Arc<dyn PhysicalExpr> { Arc::new(Column::new("col", i)) })
            .collect::<Vec<_>>();
        let accumulator_args = AccumulatorArgs {
            return_field: self.udf.return_field(mock_schema.fields())?,
            schema: &mock_schema,
            ignore_nulls: true,
            order_bys: &[],
            is_reversed: false,
            name: "",
            is_distinct: false,
            exprs: &exprs,
        };

        self.udf.accumulator(accumulator_args)
    }

    fn state_fields(&self) -> Result<Vec<FieldRef>> {
        let state_field_args = StateFieldsArgs {
            name: "",
            input_fields: &self.arg_fields(),
            return_field: self.udf.return_field(&self.arg_fields())?,
            ordering_fields: &[],
            is_distinct: false,
        };
        self.udf.state_fields(state_field_args)
    }

    fn arg_fields(&self) -> Vec<FieldRef> {
        self.arg_types
            .iter()
            .map(|sedona_type| sedona_type.to_storage_field("", true).map(Arc::new))
            .collect::<Result<Vec<_>>>()
            .unwrap()
    }
}

/// Low-level tester for scalar functions
///
/// This struct provides a means by which to run a simple check of an
/// scalar UDF implementation by simulating how DataFusion might call it.
///
/// This is not a replacement for testing at a higher level using DataFusion's
/// actual implementation but provides a useful mechanism to ensure all the
/// pieces of an scalar UDF are plugged in.
///
/// Note that arguments are always cast to the values passed [Self::new]:
/// to test different combinations of argument types, use a new tester.
pub struct ScalarUdfTester {
    udf: ScalarUDF,
    arg_types: Vec<SedonaType>,
}

impl ScalarUdfTester {
    /// Create a new tester
    pub fn new(udf: ScalarUDF, arg_types: Vec<SedonaType>) -> Self {
        Self { udf, arg_types }
    }

    /// Assert the return type of the function for the argument types used
    /// to construct this tester
    ///
    /// Both [SedonaType] or [DataType] objects can be used as the expected
    /// data type.
    pub fn assert_return_type(&self, data_type: impl TryInto<SedonaType>) {
        let expected = match data_type.try_into() {
            Ok(t) => t,
            Err(_) => panic!("Failed to convert to SedonaType"),
        };
        assert_eq!(self.return_type().unwrap(), expected)
    }

    /// Assert the result of invoking this function
    ///
    /// Both actual and expected are interpreted according to the calculated
    /// return type (notably, WKT is interpreted as geometry or geography output).
    pub fn assert_scalar_result_equals(&self, actual: impl Literal, expected: impl Literal) {
        self.assert_scalar_result_equals_inner(actual, expected, None);
    }

    /// Assert the result of invoking this function with the return type specified
    ///
    /// This is for UDFs implementing `SedonaScalarKernel::return_type_from_args_and_scalars()`.
    pub fn assert_scalar_result_equals_with_return_type(
        &self,
        actual: impl Literal,
        expected: impl Literal,
        return_type: SedonaType,
    ) {
        self.assert_scalar_result_equals_inner(actual, expected, Some(return_type));
    }

    fn assert_scalar_result_equals_inner(
        &self,
        actual: impl Literal,
        expected: impl Literal,
        return_type: Option<SedonaType>,
    ) {
        let return_type = return_type.unwrap_or_else(|| self.return_type().unwrap());
        let actual = Self::scalar_lit(actual, &return_type).unwrap();
        let expected = Self::scalar_lit(expected, &return_type).unwrap();
        assert_scalar_equal(&actual, &expected);
    }

    /// Compute the return type
    pub fn return_type(&self) -> Result<SedonaType> {
        let scalar_arguments = vec![None; self.arg_types.len()];
        self.return_type_with_scalars_inner(&scalar_arguments)
    }

    /// Compute the return type from one scalar argument
    ///
    /// This is for UDFs implementing `SedonaScalarKernel::return_type_from_args_and_scalars()`.
    pub fn return_type_with_scalar(&self, arg0: Option<impl Literal>) -> Result<SedonaType> {
        let scalar_arguments = vec![arg0
            .map(|x| Self::scalar_lit(x, &self.arg_types[0]))
            .transpose()?];
        self.return_type_with_scalars_inner(&scalar_arguments)
    }

    /// Compute the return type from two scalar arguments
    ///
    /// This is for UDFs implementing `SedonaScalarKernel::return_type_from_args_and_scalars()`.
    pub fn return_type_with_scalar_scalar(
        &self,
        arg0: Option<impl Literal>,
        arg1: Option<impl Literal>,
    ) -> Result<SedonaType> {
        let scalar_arguments = vec![
            arg0.map(|x| Self::scalar_lit(x, &self.arg_types[0]))
                .transpose()?,
            arg1.map(|x| Self::scalar_lit(x, &self.arg_types[1]))
                .transpose()?,
        ];
        self.return_type_with_scalars_inner(&scalar_arguments)
    }

    /// Compute the return type from three scalar arguments
    ///
    /// This is for UDFs implementing `SedonaScalarKernel::return_type_from_args_and_scalars()`.
    pub fn return_type_with_scalar_scalar_scalar(
        &self,
        arg0: Option<impl Literal>,
        arg1: Option<impl Literal>,
        arg2: Option<impl Literal>,
    ) -> Result<SedonaType> {
        let scalar_arguments = vec![
            arg0.map(|x| Self::scalar_lit(x, &self.arg_types[0]))
                .transpose()?,
            arg1.map(|x| Self::scalar_lit(x, &self.arg_types[1]))
                .transpose()?,
            arg2.map(|x| Self::scalar_lit(x, &self.arg_types[2]))
                .transpose()?,
        ];
        self.return_type_with_scalars_inner(&scalar_arguments)
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

    /// Invoke this function with a scalar
    pub fn invoke_scalar(&self, arg: impl Literal) -> Result<ScalarValue> {
        let scalar_arg = Self::scalar_lit(arg, &self.arg_types[0])?;

        // Some UDF calculate the return type from the input scalar arguments, so try it first.
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

    /// Invoke this function with a geometry scalar
    pub fn invoke_wkb_scalar(&self, wkt_value: Option<&str>) -> Result<ScalarValue> {
        self.invoke_scalar(create_scalar(wkt_value, &self.arg_types[0]))
    }

    /// Invoke this function with two scalars
    pub fn invoke_scalar_scalar<T0: Literal, T1: Literal>(
        &self,
        arg0: T0,
        arg1: T1,
    ) -> Result<ScalarValue> {
        let scalar_arg0 = Self::scalar_lit(arg0, &self.arg_types[0])?;
        let scalar_arg1 = Self::scalar_lit(arg1, &self.arg_types[1])?;

        // Some UDF calculate the return type from the input scalar arguments, so try it first.
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

    /// Invoke this function with three scalars
    pub fn invoke_scalar_scalar_scalar<T0: Literal, T1: Literal, T2: Literal>(
        &self,
        arg0: T0,
        arg1: T1,
        arg2: T2,
    ) -> Result<ScalarValue> {
        let scalar_arg0 = Self::scalar_lit(arg0, &self.arg_types[0])?;
        let scalar_arg1 = Self::scalar_lit(arg1, &self.arg_types[1])?;
        let scalar_arg2 = Self::scalar_lit(arg2, &self.arg_types[2])?;

        // Some UDF calculate the return type from the input scalar arguments, so try it first.
        let return_type = self
            .return_type_with_scalars_inner(&[
                Some(scalar_arg0.clone()),
                Some(scalar_arg1.clone()),
                Some(scalar_arg2.clone()),
            ])
            .ok();

        let args = vec![
            ColumnarValue::Scalar(scalar_arg0),
            ColumnarValue::Scalar(scalar_arg1),
            ColumnarValue::Scalar(scalar_arg2),
        ];
        if let ColumnarValue::Scalar(scalar) = self.invoke_with_return_type(args, return_type)? {
            Ok(scalar)
        } else {
            sedona_internal_err!("Expected scalar result from binary scalar invoke")
        }
    }

    /// Invoke this function with a geometry array
    pub fn invoke_wkb_array(&self, wkb_values: Vec<Option<&str>>) -> Result<ArrayRef> {
        self.invoke_array(create_array(&wkb_values, &self.arg_types[0]))
    }

    /// Invoke this function with a geometry array and a scalar
    pub fn invoke_wkb_array_scalar(
        &self,
        wkb_values: Vec<Option<&str>>,
        arg: impl Literal,
    ) -> Result<ArrayRef> {
        let wkb_array = create_array(&wkb_values, &self.arg_types[0]);
        self.invoke_arrays_scalar(vec![wkb_array], arg)
    }

    /// Invoke this function with an array
    pub fn invoke_array(&self, array: ArrayRef) -> Result<ArrayRef> {
        self.invoke_arrays(vec![array])
    }

    /// Invoke a binary function with an array and a scalar
    pub fn invoke_array_scalar(&self, array: ArrayRef, arg: impl Literal) -> Result<ArrayRef> {
        self.invoke_arrays_scalar(vec![array], arg)
    }

    /// Invoke a binary function with an array, and two scalars
    pub fn invoke_array_scalar_scalar(
        &self,
        array: ArrayRef,
        arg0: impl Literal,
        arg1: impl Literal,
    ) -> Result<ArrayRef> {
        self.invoke_arrays_scalar_scalar(vec![array], arg0, arg1)
    }

    /// Invoke a binary function with a scalar and an array
    pub fn invoke_scalar_array(&self, arg: impl Literal, array: ArrayRef) -> Result<ArrayRef> {
        self.invoke_scalar_arrays(arg, vec![array])
    }

    /// Invoke a binary function with two arrays
    pub fn invoke_array_array(&self, array0: ArrayRef, array1: ArrayRef) -> Result<ArrayRef> {
        self.invoke_arrays(vec![array0, array1])
    }

    /// Invoke a binary function with two arrays and a scalar
    pub fn invoke_array_array_scalar(
        &self,
        array0: ArrayRef,
        array1: ArrayRef,
        arg: impl Literal,
    ) -> Result<ArrayRef> {
        self.invoke_arrays_scalar(vec![array0, array1], arg)
    }

    fn invoke_scalar_arrays(&self, arg: impl Literal, arrays: Vec<ArrayRef>) -> Result<ArrayRef> {
        let mut args = zip(arrays, &self.arg_types)
            .map(|(array, sedona_type)| {
                ColumnarValue::Array(array).cast_to(sedona_type.storage_type(), None)
            })
            .collect::<Result<Vec<_>>>()?;
        let index = args.len();
        args.insert(0, Self::scalar_arg(arg, &self.arg_types[index])?);

        if let ColumnarValue::Array(array) = self.invoke(args)? {
            Ok(array)
        } else {
            sedona_internal_err!("Expected array result from scalar/array invoke")
        }
    }

    fn invoke_arrays_scalar(&self, arrays: Vec<ArrayRef>, arg: impl Literal) -> Result<ArrayRef> {
        let mut args = zip(arrays, &self.arg_types)
            .map(|(array, sedona_type)| {
                ColumnarValue::Array(array).cast_to(sedona_type.storage_type(), None)
            })
            .collect::<Result<Vec<_>>>()?;
        let index = args.len();
        args.push(Self::scalar_arg(arg, &self.arg_types[index])?);

        if let ColumnarValue::Array(array) = self.invoke(args)? {
            Ok(array)
        } else {
            sedona_internal_err!("Expected array result from array/scalar invoke")
        }
    }

    fn invoke_arrays_scalar_scalar(
        &self,
        arrays: Vec<ArrayRef>,
        arg0: impl Literal,
        arg1: impl Literal,
    ) -> Result<ArrayRef> {
        let mut args = zip(arrays, &self.arg_types)
            .map(|(array, sedona_type)| {
                ColumnarValue::Array(array).cast_to(sedona_type.storage_type(), None)
            })
            .collect::<Result<Vec<_>>>()?;
        let index = args.len();
        args.push(Self::scalar_arg(arg0, &self.arg_types[index])?);
        args.push(Self::scalar_arg(arg1, &self.arg_types[index + 1])?);

        if let ColumnarValue::Array(array) = self.invoke(args)? {
            Ok(array)
        } else {
            sedona_internal_err!("Expected array result from array/scalar invoke")
        }
    }

    // Invoke a function with a set of arrays
    pub fn invoke_arrays(&self, arrays: Vec<ArrayRef>) -> Result<ArrayRef> {
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

    pub fn invoke(&self, args: Vec<ColumnarValue>) -> Result<ColumnarValue> {
        self.invoke_with_return_type(args, None)
    }
    pub fn invoke_with_return_type(
        &self,
        args: Vec<ColumnarValue>,
        return_type: Option<SedonaType>,
    ) -> Result<ColumnarValue> {
        assert_eq!(args.len(), self.arg_types.len(), "Unexpected arg length");

        let mut number_rows = 1;
        for arg in &args {
            match arg {
                ColumnarValue::Array(array) => {
                    number_rows = array.len();
                    break;
                }
                _ => continue,
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
            // TODO: Consider piping actual ConfigOptions for more realistic testing
            // See: https://github.com/apache/sedona-db/issues/248
            config_options: Arc::new(ConfigOptions::default()),
        };

        self.udf.invoke_with_args(args)
    }

    fn scalar_arg(arg: impl Literal, sedona_type: &SedonaType) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Scalar(Self::scalar_lit(arg, sedona_type)?))
    }

    fn scalar_lit(arg: impl Literal, sedona_type: &SedonaType) -> Result<ScalarValue> {
        if let Expr::Literal(scalar, _) = arg.lit() {
            if matches!(
                sedona_type,
                SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _)
            ) {
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
