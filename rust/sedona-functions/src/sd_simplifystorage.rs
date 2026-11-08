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

use arrow_array::{
    Array, ArrayRef, GenericBinaryArray, OffsetSizeTrait, builder::GenericBinaryBuilder,
    cast::AsArray,
};
use arrow_schema::{DataType, FieldRef, UnionFields};
use datafusion_common::{
    Result, ScalarValue, config::ConfigOptions, datatype::DataTypeExt, exec_datafusion_err,
};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;

/// SD_SimplifyStorage() scalar UDF implementation
///
/// This function is invoked to strip strip view, dictionary, or run-end encoded
/// types from storage if needed (or return the input otherwise). This is to support
/// integration with other libraries like GDAL that haven't yet supported these
/// storage encodings.
///
/// For geometry and geography columns, items encoded as EWKB (i.e., with
/// PostGIS-style Z, M, or SRID flags in the geometry type code) are also rewritten
/// as ISO WKB. Some consumers (e.g., GDAL when writing via pyogrio) silently write
/// NULL geometries when given EWKB with an embedded SRID.
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
        let new_field = simplify_field(field.into())?;
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
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(simplify_array(
                array,
                &target,
                return_type,
            )?)),
            ColumnarValue::Scalar(scalar_value) => {
                let array = simplify_array(&scalar_value.to_array()?, &target, return_type)?;
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

fn simplify_field(field: FieldRef) -> Result<FieldRef> {
    let new_type = match field.data_type() {
        DataType::BinaryView => DataType::Binary,
        DataType::Utf8View => DataType::Utf8,
        DataType::Dictionary(_key_type, value_type) => {
            simplify_field(value_type.clone().into_nullable_field_ref())?
                .data_type()
                .clone()
        }
        DataType::RunEndEncoded(_run_ends, values) => {
            simplify_field(values.clone())?.data_type().clone()
        }
        DataType::ListView(field) | DataType::List(field) => {
            DataType::List(simplify_field(field.clone())?)
        }
        DataType::LargeListView(field) | DataType::LargeList(field) => {
            DataType::LargeList(simplify_field(field.clone())?)
        }
        DataType::FixedSizeList(field, list_size) => {
            DataType::FixedSizeList(simplify_field(field.clone())?, *list_size)
        }
        DataType::Struct(fields) => DataType::Struct(
            fields
                .into_iter()
                .cloned()
                .map(simplify_field)
                .collect::<Result<_>>()?,
        ),
        DataType::Union(union_fields, union_mode) => {
            let new_fields = union_fields
                .iter()
                .map(|(_, field)| simplify_field(field.clone()))
                .collect::<Result<Vec<_>>>()?;
            let new_ids = union_fields.iter().map(|(idx, _)| idx).collect::<Vec<_>>();
            let new_union_fields = UnionFields::try_new(new_ids, new_fields)?;
            DataType::Union(new_union_fields, *union_mode)
        }
        DataType::Map(field, is_ordered) => {
            DataType::Map(simplify_field(field.clone())?, *is_ordered)
        }
        _ => field.data_type().clone(),
    };

    let new_nullable = if let DataType::RunEndEncoded(_, values) = field.data_type() {
        field.is_nullable() || values.is_nullable()
    } else {
        field.is_nullable()
    };

    Ok(field
        .as_ref()
        .clone()
        .with_data_type(new_type)
        .with_nullable(new_nullable)
        .into())
}

fn simplify_array(
    array: &ArrayRef,
    target: &FieldRef,
    return_type: &SedonaType,
) -> Result<ArrayRef> {
    let array = datafusion_common::arrow::compute::cast(array, target.data_type())?;
    if !matches!(
        return_type,
        SedonaType::Wkb(_, _) | SedonaType::WkbLarge(_, _)
    ) {
        return Ok(array);
    }

    let normalized = match array.data_type() {
        DataType::Binary => normalize_ewkb(array.as_binary::<i32>())?,
        DataType::LargeBinary => normalize_ewkb(array.as_binary::<i64>())?,
        _ => None,
    };

    Ok(normalized.unwrap_or(array))
}

/// EWKB flags in the geometry type code (Z, M, and SRID)
const EWKB_FLAGS: u32 = 0x80000000 | 0x40000000 | 0x20000000;

/// Check whether the type code of a WKB item contains EWKB flags
fn has_ewkb_flags(buf: &[u8]) -> bool {
    let Some(code_bytes) = buf.get(1..5) else {
        return false;
    };
    let code_bytes: [u8; 4] = code_bytes.try_into().unwrap();
    let code = match buf[0] {
        0x00 => u32::from_be_bytes(code_bytes),
        0x01 => u32::from_le_bytes(code_bytes),
        _ => return false,
    };

    code & EWKB_FLAGS != 0
}

/// Rewrite any EWKB items as ISO WKB, returning None if no item needed rewriting
fn normalize_ewkb<O: OffsetSizeTrait>(array: &GenericBinaryArray<O>) -> Result<Option<ArrayRef>> {
    if !array.iter().flatten().any(has_ewkb_flags) {
        return Ok(None);
    }

    let mut builder =
        GenericBinaryBuilder::<O>::with_capacity(array.len(), array.value_data().len());
    for item in array.iter() {
        match item {
            Some(buf) if has_ewkb_flags(buf) => {
                let geom = wkb::reader::read_wkb(buf).map_err(|e| exec_datafusion_err!("{e}"))?;
                wkb::writer::write_geometry(&mut builder, &geom, &Default::default())
                    .map_err(|e| exec_datafusion_err!("{e}"))?;
                builder.append_value([]);
            }
            Some(buf) => builder.append_value(buf),
            None => builder.append_null(),
        }
    }

    Ok(Some(Arc::new(builder.finish())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::BinaryArray;
    use arrow_schema::{DataType, Field};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_LARGE_GEOMETRY, WKB_VIEW_GEOGRAPHY,
        WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{
        compare::assert_array_equal,
        create::create_array,
        fixtures::{
            MULTIPOINT_WITH_SRID_4326_EWKB, POINT_WITH_SRID_4326_EWKB, POINT_ZM_WITH_SRID_4326_EWKB,
        },
        testers::ScalarUdfTester,
    };

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = sd_simplifystorage_udf().into();
        assert_eq!(udf.name(), "sd_simplifystorage");
    }

    #[test]
    fn simplify_identity() {
        let udf = sd_simplifystorage_udf();
        let types_that_dont_need_simplification = [
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::LargeUtf8),
            SedonaType::Arrow(DataType::Binary),
            SedonaType::Arrow(DataType::LargeBinary),
            SedonaType::Arrow(DataType::Struct(
                vec![Field::new("foofy", DataType::Utf8, false)].into(),
            )),
            SedonaType::Arrow(DataType::new_list(DataType::Utf8, true)),
            SedonaType::Arrow(DataType::List(
                WKB_GEOMETRY.to_storage_field("item", true).unwrap().into(),
            )),
            WKB_GEOMETRY,
        ];

        for sedona_type in types_that_dont_need_simplification {
            let tester = ScalarUdfTester::new(udf.clone().into(), vec![sedona_type.clone()]);
            tester.assert_return_type(sedona_type.clone());

            let initial_scalar = ScalarValue::Null
                .cast_to(sedona_type.storage_type())
                .unwrap();
            let scalar_result = tester.invoke_scalar(initial_scalar.clone()).unwrap();
            assert_eq!(scalar_result, initial_scalar);

            let initial_array = initial_scalar.to_array_of_size(10).unwrap();
            let array_result = tester.invoke_array(initial_array.clone()).unwrap();
            assert_eq!(&array_result, &initial_array);
        }
    }

    #[test]
    fn simplify_actually() {
        let udf = sd_simplifystorage_udf();

        let cases = [
            // Check primitive types that need to be simplified
            (
                SedonaType::Arrow(DataType::Utf8View),
                SedonaType::Arrow(DataType::Utf8),
            ),
            (
                SedonaType::Arrow(DataType::BinaryView),
                SedonaType::Arrow(DataType::Binary),
            ),
            // Check nested types that need to be recursively simplified
            (
                SedonaType::Arrow(DataType::new_list(DataType::Utf8View, true)),
                SedonaType::Arrow(DataType::new_list(DataType::Utf8, true)),
            ),
            (
                SedonaType::Arrow(DataType::new_large_list(DataType::Utf8View, true)),
                SedonaType::Arrow(DataType::new_large_list(DataType::Utf8, true)),
            ),
            (
                SedonaType::Arrow(DataType::new_fixed_size_list(DataType::Utf8View, 2, true)),
                SedonaType::Arrow(DataType::new_fixed_size_list(DataType::Utf8, 2, true)),
            ),
            (
                SedonaType::Arrow(DataType::Struct(
                    vec![Field::new("foofy", DataType::Utf8View, false)].into(),
                )),
                SedonaType::Arrow(DataType::Struct(
                    vec![Field::new("foofy", DataType::Utf8, false)].into(),
                )),
            ),
            // Check dictionary types
            (
                SedonaType::Arrow(DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Utf8View),
                )),
                SedonaType::Arrow(DataType::Utf8),
            ),
            // Check run-end encoded
            (
                SedonaType::Arrow(DataType::RunEndEncoded(
                    DataType::Int32.into_nullable_field_ref(),
                    DataType::Utf8View.into_nullable_field_ref(),
                )),
                SedonaType::Arrow(DataType::Utf8),
            ),
            // Check complex complex nested types that need to be recursively simplified
            (
                SedonaType::Arrow(DataType::ListView(
                    Field::new("item", DataType::Utf8, true).into(),
                )),
                SedonaType::Arrow(DataType::new_list(DataType::Utf8, true)),
            ),
            (
                SedonaType::Arrow(DataType::LargeListView(
                    Field::new("item", DataType::Utf8, true).into(),
                )),
                SedonaType::Arrow(DataType::new_large_list(DataType::Utf8, true)),
            ),
            // Extension type metadata should be propagated
            (WKB_VIEW_GEOMETRY, WKB_GEOMETRY),
            (
                SedonaType::Arrow(DataType::List(
                    WKB_VIEW_GEOMETRY
                        .to_storage_field("item", true)
                        .unwrap()
                        .into(),
                )),
                SedonaType::Arrow(DataType::List(
                    WKB_GEOMETRY.to_storage_field("item", true).unwrap().into(),
                )),
            ),
        ];

        for (initial_type, simplified_type) in cases {
            let tester = ScalarUdfTester::new(udf.clone().into(), vec![initial_type.clone()]);
            let return_type = tester.return_type().unwrap();
            assert_eq!(
                return_type, simplified_type,
                "expected {initial_type:?} to simplify to {simplified_type:?} but got {return_type:?}"
            );

            // A few types aren't well supported by Arrow/DataFusion internals which make it
            // difficult to create test data.
            if !matches!(
                initial_type,
                SedonaType::Arrow(DataType::RunEndEncoded(_, _))
                    | SedonaType::Arrow(DataType::ListView(_))
                    | SedonaType::Arrow(DataType::LargeListView(_))
            ) {
                let initial_scalar = ScalarValue::Null
                    .cast_to(initial_type.storage_type())
                    .unwrap();
                let expected_scalar = ScalarValue::Null
                    .cast_to(simplified_type.storage_type())
                    .unwrap();

                let scalar_result = tester.invoke_scalar(initial_scalar.clone()).unwrap();
                assert_eq!(scalar_result, expected_scalar);

                let initial_array = initial_scalar.to_array_of_size(10).unwrap();
                let expected_array = expected_scalar.to_array_of_size(10).unwrap();
                let array_result = tester.invoke_array(initial_array).unwrap();
                assert_eq!(&array_result, &expected_array);
            }
        }
    }

    #[rstest]
    #[case(WKB_GEOMETRY, WKB_GEOMETRY)]
    #[case(WKB_LARGE_GEOMETRY, WKB_LARGE_GEOMETRY)]
    #[case(WKB_VIEW_GEOMETRY, WKB_GEOMETRY)]
    #[case(WKB_GEOGRAPHY, WKB_GEOGRAPHY)]
    #[case(WKB_VIEW_GEOGRAPHY, WKB_GEOGRAPHY)]
    fn simplify_ewkb_to_iso_wkb(
        #[case] initial_type: SedonaType,
        #[case] simplified_type: SedonaType,
    ) {
        let udf = sd_simplifystorage_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![initial_type.clone()]);
        tester.assert_return_type(simplified_type.clone());

        let iso_point = create_array(&[Some("POINT (1 2)")], &WKB_GEOMETRY);
        let iso_point = iso_point.as_binary::<i32>().value(0).to_vec();
        let ewkb: BinaryArray = [
            Some(iso_point),
            None,
            Some(POINT_WITH_SRID_4326_EWKB.to_vec()),
            Some(POINT_ZM_WITH_SRID_4326_EWKB.to_vec()),
            Some(MULTIPOINT_WITH_SRID_4326_EWKB.to_vec()),
        ]
        .iter()
        .collect();
        let initial_array =
            datafusion_common::arrow::compute::cast(&ewkb, initial_type.storage_type()).unwrap();

        let expected = create_array(
            &[
                Some("POINT (1 2)"),
                None,
                Some("POINT (1 2)"),
                Some("POINT ZM (1 2 3 4)"),
                Some("MULTIPOINT ((1 2), (3 4))"),
            ],
            &simplified_type,
        );

        let result = tester.invoke_array(initial_array).unwrap();
        assert_array_equal(&result, &expected);

        // assert_array_equal() may compare geometries rather than bytes, so also check
        // that the bytes are identical to the ISO WKB
        let as_bytes = |array: &ArrayRef| {
            let array = datafusion_common::arrow::compute::cast(array, &DataType::Binary).unwrap();
            array
                .as_binary::<i32>()
                .iter()
                .map(|item| item.map(|item| item.to_vec()))
                .collect::<Vec<_>>()
        };
        assert_eq!(as_bytes(&result), as_bytes(&expected));
    }

    #[test]
    fn simplify_ewkb_non_geometry_unchanged() {
        let udf = sd_simplifystorage_udf();
        let tester =
            ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::BinaryView)]);

        let binary: BinaryArray = [Some(POINT_WITH_SRID_4326_EWKB.to_vec())].iter().collect();
        let initial_array =
            datafusion_common::arrow::compute::cast(&binary, &DataType::BinaryView).unwrap();
        let result = tester.invoke_array(initial_array).unwrap();
        assert_eq!(&result, &(Arc::new(binary) as ArrayRef));
    }

    #[test]
    fn ewkb_flags() {
        assert!(!has_ewkb_flags(&[]));
        assert!(!has_ewkb_flags(&[0x01, 0x01, 0x00]));
        assert!(!has_ewkb_flags(&[0x01, 0xe9, 0x03, 0x00, 0x00]));
        assert!(!has_ewkb_flags(&[0x00, 0x00, 0x00, 0x03, 0xe9]));
        assert!(has_ewkb_flags(&POINT_WITH_SRID_4326_EWKB));
        assert!(has_ewkb_flags(&[0x01, 0x01, 0x00, 0x00, 0x80]));
        assert!(has_ewkb_flags(&[0x00, 0x20, 0x00, 0x00, 0x01]));
    }
}
