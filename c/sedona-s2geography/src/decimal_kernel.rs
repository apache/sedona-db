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

use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue, config::ConfigOptions};
use datafusion_expr::ColumnarValue;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::datatypes::SedonaType;

/// A decimal-only overload of a kernel with a floating-point argument.
///
/// Arrow handles decimal scaling and conversion; the inner kernel remains
/// responsible for matching every other argument and propagating the output CRS.
#[derive(Debug)]
pub(crate) struct DecimalToFloat64Kernel {
    inner: ScalarKernelRef,
    argument: usize,
}

impl DecimalToFloat64Kernel {
    pub(crate) fn new(inner: ScalarKernelRef, argument: usize) -> Self {
        Self { inner, argument }
    }

    fn cast_arg_types(&self, args: &[SedonaType]) -> Option<Vec<SedonaType>> {
        match args.get(self.argument)? {
            SedonaType::Arrow(
                DataType::Decimal32(_, _)
                | DataType::Decimal64(_, _)
                | DataType::Decimal128(_, _)
                | DataType::Decimal256(_, _),
            ) => {
                let mut args = args.to_vec();
                args[self.argument] = SedonaType::Arrow(DataType::Float64);
                Some(args)
            }
            _ => None,
        }
    }
}

impl SedonaScalarKernel for DecimalToFloat64Kernel {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.return_type_from_args_and_scalars(args, &vec![None; args.len()])
    }

    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        let Some(args) = self.cast_arg_types(args) else {
            return Ok(None);
        };
        let scalar = scalar_args
            .get(self.argument)
            .copied()
            .flatten()
            .map(|value| value.cast_to(&DataType::Float64))
            .transpose()?;
        let mut scalar_args = scalar_args.to_vec();
        if let Some(arg) = scalar_args.get_mut(self.argument) {
            *arg = scalar.as_ref();
        }
        self.inner
            .return_type_from_args_and_scalars(&args, &scalar_args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("Should not be called (use invoke_batch_from_args())")
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let Some(arg_types) = self.cast_arg_types(arg_types) else {
            return sedona_internal_err!("Expected a decimal argument for decimal overload");
        };
        let mut args = args.to_vec();
        args[self.argument] = args[self.argument].cast_to(&DataType::Float64, None)?;
        self.inner
            .invoke_batch_from_args(&arg_types, &args, return_type, num_rows, config_options)
    }
}
