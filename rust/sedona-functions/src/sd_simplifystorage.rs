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
    Array, BinaryViewArray, GenericBinaryArray, OffsetSizeTrait,
    builder::{BinaryBuilder, GenericBinaryBuilder, LargeBinaryBuilder},
};
use arrow_schema::{DataType, FieldRef, UnionFields};
use datafusion_common::{
    Result, ScalarValue,
    cast::{as_binary_array, as_binary_view_array, as_large_binary_array},
    config::ConfigOptions,
    datatype::DataTypeExt,
    exec_datafusion_err,
};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;

use crate::executor::{WkbBytesExecutor, visible_wkb_bytes};

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
/// NULL geometries when given EWKB with an embedded SRID. The flag check reads only
/// the type-code high byte. When no item has those flags and the storage type
/// already matches, the input is returned without copying.
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
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let target = return_type.to_storage_field("", true)?;
        let target_type = target.data_type().clone();
        if is_wkb_sedona(return_type) && is_wkb_sedona(&arg_types[0]) {
            return simplify_wkb_storage(arg_types, args, &target_type, return_type);
        }
        cast_columnar(&args[0], &target_type)
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

fn is_wkb_sedona(sedona_type: &SedonaType) -> bool {
    matches!(
        sedona_type,
        SedonaType::Wkb(_, _) | SedonaType::WkbLarge(_, _) | SedonaType::WkbView(_, _)
    )
}

/// Simplify one geometry or geography value.
///
/// EWKB is detected on the original storage, before a binary-view cast.
/// `WkbBytesExecutor` then rewrites in one pass. ISO WKB whose storage type
/// already matches `target` is returned as-is.
fn simplify_wkb_storage(
    arg_types: &[SedonaType],
    args: &[ColumnarValue],
    target: &DataType,
    return_type: &SedonaType,
) -> Result<ColumnarValue> {
    let has_ewkb = match &args[0] {
        ColumnarValue::Array(array) => array_has_ewkb(array.as_ref(), &arg_types[0])?,
        ColumnarValue::Scalar(scalar) => scalar_has_ewkb(scalar),
    };
    if has_ewkb {
        return rewrite_ewkb_as_iso(arg_types, args, return_type);
    }

    if storage_matches(&args[0], target) {
        return Ok(args[0].clone());
    }
    cast_columnar(&args[0], target)
}

fn storage_matches(value: &ColumnarValue, target: &DataType) -> bool {
    match value {
        ColumnarValue::Array(array) => array.data_type() == target,
        ColumnarValue::Scalar(scalar) => scalar.data_type() == *target,
    }
}

fn cast_columnar(value: &ColumnarValue, target: &DataType) -> Result<ColumnarValue> {
    match value {
        ColumnarValue::Array(array) => Ok(ColumnarValue::Array(
            datafusion_common::arrow::compute::cast(array, target)?,
        )),
        ColumnarValue::Scalar(scalar) => {
            let array = datafusion_common::arrow::compute::cast(&scalar.to_array()?, target)?;
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &array, 0,
            )?))
        }
    }
}

/// High byte of an EWKB geometry type code: Z (0x80), M (0x40), and SRID (0x20).
///
/// ISO type codes keep this byte at zero. geoarrow's `_any_ewkb` treats any
/// non-zero high byte as EWKB; these are the flag bits that occupy it.
const EWKB_HIGH_BYTE: u8 = 0x80 | 0x40 | 0x20;

/// Check whether a WKB header's type code contains EWKB flags.
///
/// Reads the endian byte and the type-code high byte only. Short buffers and
/// unrecognized endian markers are not EWKB.
fn buf_has_ewkb_flags(buf: &[u8]) -> bool {
    let Some(&endian) = buf.first() else {
        return false;
    };
    let high = match endian {
        // Little endian: the high byte of the u32 type code is at offset 4.
        0x01 => buf.get(4).copied(),
        // Big endian: it is the first byte of the type code.
        0x00 => buf.get(1).copied(),
        _ => return false,
    };
    matches!(high, Some(byte) if (byte & EWKB_HIGH_BYTE) != 0)
}

/// True when any item is EWKB.
///
/// Same shape as geoarrow's `_any_ewkb`: an empty column is not EWKB, a hit
/// on the first item returns immediately, and otherwise only each item's
/// type-code high byte is inspected. Nulls are skipped, so a null slot cannot
/// force a copy.
fn array_has_ewkb(array: &dyn Array, arg_type: &SedonaType) -> Result<bool> {
    match arg_type {
        SedonaType::Wkb(_, _) => Ok(binary_has_ewkb(as_binary_array(array)?)),
        SedonaType::WkbLarge(_, _) => Ok(binary_has_ewkb(as_large_binary_array(array)?)),
        SedonaType::WkbView(_, _) => Ok(view_has_ewkb(as_binary_view_array(array)?)),
        _ => sedona_internal_err!("SD_SimplifyStorage expected WKB storage, got {arg_type}"),
    }
}

fn binary_has_ewkb<O: OffsetSizeTrait>(array: &GenericBinaryArray<O>) -> bool {
    any_header_ewkb(array.len(), |index| binary_header_is_ewkb(array, index))
}

fn view_has_ewkb(array: &BinaryViewArray) -> bool {
    any_header_ewkb(array.len(), |index| {
        if array.is_null(index) {
            return false;
        }
        // `value` is a pointer into inline or buffer data. Keep only the header.
        let value = array.value(index);
        let header_len = value.len().min(5);
        buf_has_ewkb_flags(&value[..header_len])
    })
}

/// Read at most the 5-byte WKB header out of the values buffer.
///
/// Offsets are absolute into `value_data`, including for a sliced array, so
/// bytes that belong to an item outside the slice are not inspected.
fn binary_header_is_ewkb<O: OffsetSizeTrait>(array: &GenericBinaryArray<O>, index: usize) -> bool {
    if array.is_null(index) {
        return false;
    }
    let offsets = array.value_offsets();
    let start = offsets[index].as_usize();
    let end = offsets[index + 1].as_usize();
    let data = array.value_data();
    if start > end || end > data.len() {
        return false;
    }
    let header_end = end.min(start.saturating_add(5));
    buf_has_ewkb_flags(&data[start..header_end])
}

fn any_header_ewkb(len: usize, mut is_ewkb: impl FnMut(usize) -> bool) -> bool {
    if len == 0 {
        return false;
    }
    if len > 1 && is_ewkb(0) {
        return true;
    }
    let start = usize::from(len > 1);
    (start..len).any(is_ewkb)
}

fn scalar_has_ewkb(scalar: &ScalarValue) -> bool {
    let bytes = match scalar {
        ScalarValue::Binary(value)
        | ScalarValue::LargeBinary(value)
        | ScalarValue::BinaryView(value) => value.as_deref(),
        _ => None,
    };
    bytes.is_some_and(buf_has_ewkb_flags)
}

/// Rewrite EWKB items as ISO WKB in one pass over the original storage.
fn rewrite_ewkb_as_iso(
    arg_types: &[SedonaType],
    args: &[ColumnarValue],
    return_type: &SedonaType,
) -> Result<ColumnarValue> {
    let executor = WkbBytesExecutor::new(arg_types, args);
    let n = executor.num_iterations();
    let nbytes = visible_wkb_bytes(&arg_types[0], &args[0])?;
    if matches!(return_type, SedonaType::WkbLarge(_, _)) {
        let mut builder = LargeBinaryBuilder::with_capacity(n, nbytes);
        append_iso_wkb(&executor, &mut builder)?;
        executor.finish(Arc::new(builder.finish()))
    } else {
        let mut builder = BinaryBuilder::with_capacity(n, nbytes);
        append_iso_wkb(&executor, &mut builder)?;
        executor.finish(Arc::new(builder.finish()))
    }
}

fn append_iso_wkb<O: OffsetSizeTrait>(
    executor: &WkbBytesExecutor,
    builder: &mut GenericBinaryBuilder<O>,
) -> Result<()> {
    executor.execute_wkb_void(|item| {
        match item {
            Some(buf) if buf_has_ewkb_flags(buf) => {
                let geom = wkb::reader::read_wkb(buf).map_err(|e| exec_datafusion_err!("{e}"))?;
                wkb::writer::write_geometry(builder, &geom, &Default::default())
                    .map_err(|e| exec_datafusion_err!("{e}"))?;
                builder.append_value([]);
            }
            Some(buf) => builder.append_value(buf),
            None => builder.append_null(),
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, BinaryArray, cast::AsArray};
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
        assert!(!buf_has_ewkb_flags(&[]));
        assert!(!buf_has_ewkb_flags(&[0x01, 0x01, 0x00]));
        assert!(!buf_has_ewkb_flags(&[0x01, 0xe9, 0x03, 0x00, 0x00]));
        assert!(!buf_has_ewkb_flags(&[0x00, 0x00, 0x00, 0x03, 0xe9]));
        assert!(buf_has_ewkb_flags(&POINT_WITH_SRID_4326_EWKB));
        assert!(buf_has_ewkb_flags(&[0x01, 0x01, 0x00, 0x00, 0x80]));
        assert!(buf_has_ewkb_flags(&[0x00, 0x20, 0x00, 0x00, 0x01]));
        // A non-zero byte past the header is not an EWKB flag.
        assert!(!buf_has_ewkb_flags(&[0x01, 0x01, 0x00, 0x00, 0x00, 0xff]));
    }

    fn invoke_storage(arg_type: SedonaType, arg: ColumnarValue) -> Result<ColumnarValue> {
        let kernel = SDSimplifyStorage {};
        let return_type = kernel
            .return_type(std::slice::from_ref(&arg_type))?
            .expect("return type");
        let args = [arg];
        let arg_types = [arg_type];
        kernel.invoke_batch_from_args(&arg_types, &args, &return_type, 1, None)
    }

    #[test]
    fn simplify_iso_wkb_does_not_copy() {
        let iso = create_array(
            &[Some("POINT (1 2)"), None, Some("POINT (3 4)")],
            &WKB_GEOMETRY,
        );
        let ColumnarValue::Array(out) =
            invoke_storage(WKB_GEOMETRY, ColumnarValue::Array(iso.clone())).unwrap()
        else {
            panic!("expected an array");
        };
        assert!(Arc::ptr_eq(&iso, &out));

        let large = datafusion_common::arrow::compute::cast(&iso, &DataType::LargeBinary).unwrap();
        let ColumnarValue::Array(out) =
            invoke_storage(WKB_LARGE_GEOMETRY, ColumnarValue::Array(large.clone())).unwrap()
        else {
            panic!("expected an array");
        };
        assert!(Arc::ptr_eq(&large, &out));

        // EWKB that sits before a slice must not be visible to the probe.
        let iso_bytes = iso.as_binary::<i32>().value(0).to_vec();
        let mixed: BinaryArray = [
            Some(POINT_WITH_SRID_4326_EWKB.to_vec()),
            Some(iso_bytes.clone()),
            None,
            Some(iso_bytes.clone()),
        ]
        .into_iter()
        .collect();
        let mixed: ArrayRef = Arc::new(mixed);
        let sliced = mixed.slice(1, 3);
        let ColumnarValue::Array(out) =
            invoke_storage(WKB_GEOMETRY, ColumnarValue::Array(sliced.clone())).unwrap()
        else {
            panic!("expected an array");
        };
        assert!(Arc::ptr_eq(&sliced, &out));
        assert_eq!(out.as_binary::<i32>().value(0), iso_bytes.as_slice());

        // A view still has to be cast, but ISO bytes are not rewritten.
        let view = create_array(&[Some("POINT (0 1)"), None], &WKB_VIEW_GEOMETRY);
        let ColumnarValue::Array(out) =
            invoke_storage(WKB_VIEW_GEOMETRY, ColumnarValue::Array(view.clone())).unwrap()
        else {
            panic!("expected an array");
        };
        assert_eq!(out.data_type(), &DataType::Binary);
        assert!(!Arc::ptr_eq(&view, &out));
        assert_eq!(
            out.as_binary::<i32>().value(0),
            view.as_binary_view().value(0)
        );
    }

    #[test]
    fn simplify_sliced_ewkb_uses_slice_offsets() {
        let iso = create_array(&[Some("POINT (1 2)")], &WKB_GEOMETRY);
        let iso_bytes = iso.as_binary::<i32>().value(0).to_vec();
        let mixed: BinaryArray = [
            Some(iso_bytes.clone()),
            Some(POINT_WITH_SRID_4326_EWKB.to_vec()),
            None,
            Some(POINT_ZM_WITH_SRID_4326_EWKB.to_vec()),
        ]
        .into_iter()
        .collect();
        // Drop the leading ISO item so the first visible item is EWKB and the
        // probe's early return has to use slice-relative offsets.
        let mixed: ArrayRef = Arc::new(mixed);
        let sliced = mixed.slice(1, 3);
        let ColumnarValue::Array(out) =
            invoke_storage(WKB_GEOMETRY, ColumnarValue::Array(sliced)).unwrap()
        else {
            panic!("expected an array");
        };
        let expected = create_array(
            &[Some("POINT (1 2)"), None, Some("POINT ZM (1 2 3 4)")],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&out, &expected);
    }

    #[test]
    fn simplify_ewkb_scalar() {
        let scalar = ScalarValue::Binary(Some(POINT_WITH_SRID_4326_EWKB.to_vec()));
        let ColumnarValue::Scalar(ScalarValue::Binary(Some(bytes))) =
            invoke_storage(WKB_GEOMETRY, ColumnarValue::Scalar(scalar)).unwrap()
        else {
            panic!("expected a binary scalar");
        };
        let expected = create_array(&[Some("POINT (1 2)")], &WKB_GEOMETRY);
        assert_eq!(bytes.as_slice(), expected.as_binary::<i32>().value(0));

        let iso = ScalarValue::Binary(Some(expected.as_binary::<i32>().value(0).to_vec()));
        let ColumnarValue::Scalar(out) =
            invoke_storage(WKB_GEOMETRY, ColumnarValue::Scalar(iso.clone())).unwrap()
        else {
            panic!("expected a scalar");
        };
        assert_eq!(out, iso);
    }
}
