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

use arrow_array::ArrayRef;
use arrow_schema::{DataType, FieldRef, UnionFields};
use datafusion_common::{config::ConfigOptions, datatype::DataTypeExt, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;

/// SD_SimplifyStorage() scalar UDF implementation
///
/// This function is invoked to strip dictionary, run-end-encoded, or dictionary
/// types from storage if needed (or return the input otherwise). This is to support
/// integration with other libraries like GDAL that haven't yet supported these
/// types.
pub fn sd_simplifystorage_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "sd_simplifystorage",
        vec![Arc::new(SDSimplifyStorage {})],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct SDSimplifyStorage {}

impl SedonaScalarKernel for SDSimplifyStorage {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let field = args[0].to_storage_field("", true)?;
        let new_field = simplify_field(field.into());
        Ok(Some(SedonaType::from_storage_field(&new_field)?))
    }

    fn invoke_batch_from_args(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let target = Arc::new(return_type.to_storage_field("", true)?);
        match &args[0] {
            ColumnarValue::Array(array) => {
                Ok(ColumnarValue::Array(simplify_array(array, &target)?))
            }
            ColumnarValue::Scalar(scalar_value) => {
                let array = simplify_array(&scalar_value.to_array()?, &target)?;
                Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                    &array, 0,
                )?))
            }
        }
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("Unexpected call to invoke_batch()")
    }
}

fn simplify_field(field: FieldRef) -> FieldRef {
    let new_type = match field.data_type() {
        DataType::BinaryView => DataType::Binary,
        DataType::Utf8View => DataType::Utf8,
        DataType::Dictionary(_key_type, value_type) => {
            simplify_field(value_type.clone().into_nullable_field_ref())
                .data_type()
                .clone()
        }
        DataType::RunEndEncoded(_run_ends, values) => {
            simplify_field(values.clone()).data_type().clone()
        }
        DataType::ListView(field) | DataType::List(field) => {
            DataType::List(simplify_field(field.clone()))
        }
        DataType::LargeListView(field) | DataType::LargeList(field) => {
            DataType::LargeList(simplify_field(field.clone()))
        }
        DataType::FixedSizeList(field, list_size) => {
            DataType::FixedSizeList(simplify_field(field.clone()), *list_size)
        }
        DataType::Struct(fields) => {
            DataType::Struct(fields.into_iter().cloned().map(simplify_field).collect())
        }
        DataType::Union(union_fields, union_mode) => {
            let new_fields = union_fields
                .iter()
                .map(|(_, field)| simplify_field(field.clone()))
                .collect::<Vec<_>>();
            let new_ids = union_fields.iter().map(|(idx, _)| idx).collect::<Vec<_>>();
            let new_union_fields = UnionFields::new(new_ids, new_fields);
            DataType::Union(new_union_fields, *union_mode)
        }
        DataType::Map(field, is_ordered) => {
            DataType::Map(simplify_field(field.clone()), *is_ordered)
        }
        _ => field.data_type().clone(),
    };

    let new_nullable = if let DataType::RunEndEncoded(_, values) = field.data_type() {
        field.is_nullable() || values.is_nullable()
    } else {
        field.is_nullable()
    };

    field
        .as_ref()
        .clone()
        .with_data_type(new_type)
        .with_nullable(new_nullable)
        .into()
}

fn simplify_array(array: &ArrayRef, target: &FieldRef) -> Result<ArrayRef> {
    Ok(datafusion_common::arrow::compute::cast(
        array,
        target.data_type(),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = sd_simplifystorage_udf().into();
        assert_eq!(udf.name(), "sd_simplifystorage");
    }

    #[rstest]
    fn simplify_identity(
        // All these types don't need to be simplified
        #[values(
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::LargeUtf8),
            SedonaType::Arrow(DataType::Binary),
            SedonaType::Arrow(DataType::LargeBinary),
            SedonaType::Arrow(DataType::Struct(vec![Field::new("foofy", DataType::Utf8, false)].into())),
            SedonaType::Arrow(DataType::new_list(DataType::Utf8, true)),
            SedonaType::Arrow(DataType::List(WKB_GEOMETRY.to_storage_field("item", true).unwrap().into())),
            WKB_GEOMETRY,
        )]
        sedona_type: SedonaType,
    ) {
        let udf = sd_simplifystorage_udf();
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type.clone());
    }

    #[rstest]
    fn simplify_actually(
        // All these types don't need to be simplified
        #[values(
            (SedonaType::Arrow(DataType::Utf8View), SedonaType::Arrow(DataType::Utf8)),
            (WKB_VIEW_GEOMETRY, WKB_GEOMETRY)
        )]
        sedona_type: (SedonaType, SedonaType),
    ) {
        let (initial_type, simplified_type) = sedona_type;
        let udf = sd_simplifystorage_udf();
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![initial_type.clone()]);
        tester.assert_return_type(simplified_type.clone());
    }
}
