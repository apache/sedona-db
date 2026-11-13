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
use std::{sync::Arc, vec};

use arrow_array::new_null_array;
use arrow_schema::DataType;
use datafusion_common::{cast::as_int64_array, error::Result, scalar::ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_functions::executor::{WkbBytesExecutor, WkbExecutor};
use sedona_schema::{
    datatypes::{
        SedonaType, WKB_GEOGRAPHY_WGS84, WKB_GEOMETRY, WKB_VIEW_GEOGRAPHY_WGS84, WKB_VIEW_GEOMETRY,
    },
    matchers::ArgMatcher,
};

use crate::geoarrow_c::{ArrayReader, ArrayWriter, CError, Visitor, WkbReader};

/// ST_GeomFromWKT() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKT reading using geoarrow-c's WKT reader
pub fn st_geomfromwkt_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCCast::new(
        ArgMatcher::new(vec![ArgMatcher::is_string()], WKB_GEOMETRY),
        Some(WKB_GEOMETRY),
        WKB_GEOMETRY,
    ))
}

/// ST_GeogFromWKT() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKT reading using geoarrow-c's WKT reader
pub fn st_geogfromwkt_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCCast::new(
        ArgMatcher::new(vec![ArgMatcher::is_string()], WKB_GEOGRAPHY_WGS84.clone()),
        Some(WKB_GEOGRAPHY_WGS84.clone()),
        WKB_GEOGRAPHY_WGS84.clone(),
    ))
}

/// ST_GeomFromWKB() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKB validation using geoarrow-c's WKB reader
pub fn st_geomfromwkb_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCCast::new(
        ArgMatcher::new(vec![ArgMatcher::is_binary()], WKB_VIEW_GEOMETRY),
        None,
        WKB_VIEW_GEOMETRY,
    ))
}

/// ST_GeogFromWKB() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKB validation using geoarrow-c's WKB reader
pub fn st_geogfromwkb_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCCast::new(
        ArgMatcher::new(
            vec![ArgMatcher::is_binary()],
            WKB_VIEW_GEOGRAPHY_WGS84.clone(),
        ),
        None,
        WKB_VIEW_GEOGRAPHY_WGS84.clone(),
    ))
}

/// ST_AsText() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKT writing using geoarrow-c's WKT writer
pub fn st_astext_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCAsText {
        matcher: ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Utf8),
        ),
    })
}

/// ST_AsText(geometry, precision) implementation using geoarrow-c.
///
/// geoarrow-c is bundled with Ryu enabled, so coordinate formatting does not
/// depend on the platform locale or libc's `snprintf()` implementation.
pub fn st_astext_precision_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(vec![Arc::new(GeoArrowCAsText {
        matcher: ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
            ],
            SedonaType::Arrow(DataType::Utf8),
        ),
    })])
}

#[derive(Debug)]
struct GeoArrowCAsText {
    matcher: ArgMatcher,
}

impl SedonaScalarKernel for GeoArrowCAsText {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbBytesExecutor::new(arg_types, args);
        if arg_types[0] == SedonaType::Arrow(DataType::Null) {
            return executor.finish(new_null_array(&DataType::Utf8, executor.num_iterations()));
        }

        // Both overloads use the same writer, with either constant or per-row
        // precision. Keep scalar validation outside the loop, including empty batches.
        let precision = match args.get(1) {
            Some(arg) => arg.cast_to(&DataType::Int64, None)?,
            None => ColumnarValue::Scalar(ScalarValue::Int64(Some(16))),
        };
        match precision {
            ColumnarValue::Scalar(ScalarValue::Int64(Some(precision))) => {
                validate_precision(precision)?;
                write_wkt(&executor, std::iter::repeat(Some(precision)))
            }
            ColumnarValue::Scalar(ScalarValue::Int64(None)) => {
                executor.finish(new_null_array(&DataType::Utf8, executor.num_iterations()))
            }
            ColumnarValue::Array(array) => write_wkt(&executor, as_int64_array(&array)?.iter()),
            _ => unreachable!("precision was cast to Int64"),
        }
    }
}

fn validate_precision(precision: i64) -> Result<i32> {
    if !(0..=16).contains(&precision) {
        return Err(datafusion_common::DataFusionError::Execution(format!(
            "ST_AsText precision must be between 0 and 16, got {precision}"
        )));
    }

    Ok(precision as i32)
}

fn write_wkt(
    executor: &WkbBytesExecutor,
    mut precisions: impl Iterator<Item = Option<i64>>,
) -> Result<ColumnarValue> {
    let mut error = CError::new();
    let mut reader = WkbReader::try_new()?;
    let mut writer = ArrayWriter::try_new(&SedonaType::Arrow(DataType::Utf8))?;
    executor.execute_wkb_void(|wkb| {
        match (wkb, precisions.next().unwrap()) {
            (Some(wkb), Some(precision)) => {
                writer.set_precision(validate_precision(precision)?)?;
                reader.append_to(wkb, &mut writer, &mut error)?;
            }
            _ => writer.append_null()?,
        }
        Ok(())
    })?;
    executor.finish(writer.finish()?)
}

#[derive(Debug)]
struct GeoArrowCCast {
    matcher: ArgMatcher,
    writer_type: Option<SedonaType>,
    output_type: DataType,
}

impl GeoArrowCCast {
    pub fn new(
        matcher: ArgMatcher,
        writer_type: Option<SedonaType>,
        output_type: SedonaType,
    ) -> Self {
        Self {
            matcher,
            writer_type,
            output_type: output_type.storage_type().clone(),
        }
    }
}

impl SedonaScalarKernel for GeoArrowCCast {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // Let the existing generic kernels handle untyped NULL literals. The
        // GeoArrow reader needs a concrete physical input type.
        if args
            .iter()
            .any(|arg| arg == &SedonaType::Arrow(DataType::Null))
        {
            return Ok(None);
        }

        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let array_in = args[0].to_array(executor.num_iterations())?;

        let mut error = CError::new();
        let mut reader = ArrayReader::try_new(&arg_types[0])?;

        let array_out = if let Some(output_type) = &self.writer_type {
            let mut writer = ArrayWriter::try_new(output_type)?;
            let visitor = writer.visitor_mut();

            reader.visit(&array_in, visitor, &mut error)?;
            Arc::new(writer.finish()?)
        } else {
            let mut visitor = Visitor::void();

            reader.visit(&array_in, &mut visitor, &mut error)?;
            array_in
        };

        let value_out = executor.finish(array_out)?;
        if self.output_type == value_out.data_type() {
            Ok(value_out)
        } else {
            value_out.cast_to(&self.output_type, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Int64Array, StringArray};
    use arrow_schema::DataType;
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_LARGE_GEOMETRY, WKB_VIEW_GEOMETRY,
    };

    use sedona_testing::{
        create::{create_array, create_scalar_storage},
        testers::ScalarUdfTester,
    };

    use super::*;

    #[rstest]
    fn fromwkt(
        #[values(DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View)] data_type: DataType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_geomfromwkt", st_geomfromwkt_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(data_type)]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester.invoke_scalar("POINT (1 2)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");

        let utf8_array: StringArray = [Some("POINT (1 2)"), None, Some("POINT (3 4)")]
            .iter()
            .collect();

        assert_eq!(
            &tester.invoke_array(Arc::new(utf8_array)).unwrap(),
            &create_array(
                &[Some("POINT (1 2)"), None, Some("POINT (3 4)")],
                &WKB_GEOMETRY,
            )
        );
    }

    #[rstest]
    fn fromwkb(
        #[values(WKB_GEOMETRY, WKB_LARGE_GEOMETRY, WKB_VIEW_GEOMETRY)] data_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_geomfromwkb", st_geomfromwkb_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![SedonaType::Arrow(data_type.storage_type().clone())],
        );
        tester.assert_return_type(WKB_VIEW_GEOMETRY);

        let result = tester
            .invoke_scalar(create_scalar_storage(Some("POINT (1 2)"), &data_type))
            .unwrap();
        assert_eq!(result.data_type(), DataType::BinaryView);
        tester.assert_scalar_result_equals(result, "POINT (1 2)");
        let input = create_array(&[Some("POINT (1 2)"), None], &data_type);
        let result = tester.invoke_array(input).unwrap();
        assert_eq!(result.data_type(), &DataType::BinaryView);
        assert_eq!(result.null_count(), 1);
    }

    #[rstest]
    fn astext(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] data_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_astext", st_astext_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![data_type]);
        tester.assert_return_type(DataType::Utf8);

        let result = tester.invoke_scalar("POINT (1 2)").unwrap();
        assert_eq!(result, ScalarValue::Utf8(Some("POINT (1 2)".to_string())));
    }

    #[test]
    fn cast_kernels_defer_untyped_nulls() {
        let null_type = [SedonaType::Arrow(DataType::Null)];
        assert_eq!(
            st_astext_impl().return_type(&null_type).unwrap(),
            Some(SedonaType::Arrow(DataType::Utf8))
        );
        assert_eq!(st_geomfromwkt_impl().return_type(&null_type).unwrap(), None);
        assert_eq!(st_geomfromwkb_impl().return_type(&null_type).unwrap(), None);
    }

    #[test]
    fn astext_precision() {
        let udf = SedonaScalarUDF::from_impl("st_astext", st_astext_precision_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Int64)],
        );
        tester.assert_return_type(DataType::Utf8);

        let result = tester
            .invoke_scalar_scalar("POINT (1.123456789 2.5)", 6_i64)
            .unwrap();
        assert_eq!(
            result,
            ScalarValue::Utf8(Some("POINT (1.123457 2.5)".to_string()))
        );

        let result = tester
            .invoke_scalar_scalar("POINT (1.9 2.1)", 0_i64)
            .unwrap();
        assert_eq!(result, ScalarValue::Utf8(Some("POINT (2 2)".to_string())));

        let result = tester
            .invoke_scalar_scalar("POINT (1.123456789 2.5)", ScalarValue::Int64(None))
            .unwrap();
        assert_eq!(result, ScalarValue::Utf8(None));
    }

    #[test]
    fn astext_precision_array() {
        let udf = SedonaScalarUDF::from_impl("st_astext", st_astext_precision_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Int64)],
        );
        let geometry = create_array(
            &[
                Some("POINT (1.123456789 2.5)"),
                Some("POINT (1.123456789 2.5)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let precision = Arc::new(Int64Array::from(vec![Some(3), None, Some(6)]));
        let result = tester.invoke_array_array(geometry, precision).unwrap();
        let expected: StringArray = [Some("POINT (1.123 2.5)"), None, None]
            .into_iter()
            .collect();
        assert_eq!(result.as_ref(), &expected);
    }

    #[test]
    fn astext_precision_rejects_out_of_range_values() {
        let udf = SedonaScalarUDF::from_impl("st_astext", st_astext_precision_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Int64)],
        );
        let error = tester
            .invoke_scalar_scalar("POINT (1 2)", 17_i64)
            .unwrap_err();
        assert_eq!(
            error.strip_backtrace(),
            "Execution error: ST_AsText precision must be between 0 and 16, got 17"
        );
    }

    #[rstest]
    fn astext_geometry_matches_visitor(
        #[values(WKB_GEOMETRY, WKB_LARGE_GEOMETRY, WKB_VIEW_GEOMETRY)] data_type: SedonaType,
    ) {
        let values = [
            Some("POINT (99 99)"), // discarded by slicing
            None,
            Some("POINT EMPTY"),
            Some("POINT Z (1.123456789 2.5 3.75)"),
            Some("POINT M (1 2 3)"),
            Some("POINT ZM (1 2 3 4)"),
            Some("LINESTRING (0 1, 2.125 3.875, -4.5 5.5)"),
            Some("LINESTRING EMPTY"),
            Some("POLYGON ((0 0, 10 0, 10 10, 0 0), (1 1, 2 1, 2 2, 1 1))"),
            Some("POLYGON EMPTY"),
            Some("MULTIPOINT ((1 2), (3 4))"),
            Some("MULTIPOINT EMPTY"),
            Some("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"),
            Some("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)))"),
            Some(
                "GEOMETRYCOLLECTION (POINT EMPTY, GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1)))",
            ),
            Some("GEOMETRYCOLLECTION EMPTY"),
            None,
        ];
        let input = create_array(&values, &data_type).slice(1, values.len() - 1);
        let udf = SedonaScalarUDF::from_impl("st_astext", st_astext_precision_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![data_type.clone(), SedonaType::Arrow(DataType::Int64)],
        );
        for precision in [0, 3, 16] {
            let mut reader = ArrayReader::try_new(&data_type).unwrap();
            let mut writer = ArrayWriter::try_new(&SedonaType::Arrow(DataType::Utf8)).unwrap();
            writer.set_precision(precision).unwrap();
            reader
                .visit(&input, writer.visitor_mut(), &mut CError::new())
                .unwrap();
            let expected = writer.finish().unwrap();
            let actual = tester
                .invoke_array_scalar(input.clone(), precision as i64)
                .unwrap();
            assert_eq!(&actual, &expected);
        }
    }

    #[rstest]
    fn astext_precision_broadcast_and_slice(
        #[values(WKB_GEOMETRY, WKB_LARGE_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        data_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_astext", st_astext_precision_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![data_type.clone(), SedonaType::Arrow(DataType::Int64)],
        );
        let precision: Arc<dyn Array> = Arc::new(Int64Array::from(vec![
            Some(99),
            Some(0),
            None,
            Some(3),
            Some(6),
        ]));
        let precision = precision.slice(1, 4);
        let expected = StringArray::from(vec![
            Some("POINT (1 2)"),
            None,
            Some("POINT (1.123 2.5)"),
            Some("POINT (1.123457 2.5)"),
        ]);
        let result = tester
            .invoke_scalar_array("POINT (1.123456789 2.5)", precision.clone())
            .unwrap();
        assert_eq!(result.as_ref(), &expected);
        let input = create_array(&[Some("POINT (1.123456789 2.5)"); 5], &data_type).slice(1, 4);
        let result = tester.invoke_array_array(input, precision).unwrap();
        assert_eq!(result.as_ref(), &expected);
    }

    #[test]
    fn astext_endianness_and_invalid_wkb() {
        let udf = SedonaScalarUDF::from_impl("st_astext", st_astext_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);
        // A little-endian collection with a big-endian and a little-endian point.
        let mut wkb = vec![1];
        wkb.extend(7_u32.to_le_bytes());
        wkb.extend(2_u32.to_le_bytes());
        wkb.push(0);
        wkb.extend(1_u32.to_be_bytes());
        wkb.extend(1.25_f64.to_be_bytes());
        wkb.extend(2.5_f64.to_be_bytes());
        wkb.push(1);
        wkb.extend(1_u32.to_le_bytes());
        wkb.extend(3.75_f64.to_le_bytes());
        wkb.extend(4.5_f64.to_le_bytes());
        let result = tester
            .invoke_scalar(ScalarValue::Binary(Some(wkb.clone())))
            .unwrap();
        assert_eq!(
            result,
            ScalarValue::Utf8(Some(
                "GEOMETRYCOLLECTION (POINT (1.25 2.5), POINT (3.75 4.5))".into()
            ))
        );
        for length in [0, 1, 4, 8, wkb.len() - 1] {
            assert!(
                tester
                    .invoke_scalar(ScalarValue::Binary(Some(wkb[..length].to_vec())))
                    .is_err()
            );
        }
    }

    #[test]
    fn astext_precision_null_empty_and_invalid() {
        let udf = SedonaScalarUDF::from_impl("st_astext", st_astext_precision_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Int64)],
        );
        let geometry = create_array(&[None, Some("POINT (1 2)")], &WKB_GEOMETRY);
        let precision = Arc::new(Int64Array::from(vec![Some(99), None]));
        let result = tester
            .invoke_array_array(geometry.clone(), precision)
            .unwrap();
        assert_eq!(result.null_count(), 2);
        assert!(
            tester
                .invoke_array_scalar(geometry.clone(), -1_i64)
                .is_err()
        );
        assert!(
            tester
                .invoke_array_array(
                    geometry,
                    Arc::new(Int64Array::from(vec![Some(0), Some(17)]))
                )
                .is_err()
        );
        let empty = create_array(&[], &WKB_GEOMETRY);
        assert_eq!(
            tester
                .invoke_array_scalar(empty.clone(), 6_i64)
                .unwrap()
                .len(),
            0
        );
        assert!(tester.invoke_array_scalar(empty, 17_i64).is_err());

        let udf = SedonaScalarUDF::from_impl("st_astext", st_astext_precision_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                SedonaType::Arrow(DataType::Null),
                SedonaType::Arrow(DataType::Int64),
            ],
        );
        assert_eq!(
            tester
                .invoke_scalar_scalar(ScalarValue::Null, 6_i64)
                .unwrap(),
            ScalarValue::Utf8(None)
        );
    }

    #[test]
    fn errors() {
        let udf = SedonaScalarUDF::from_impl("st_geomfromwkt", st_geomfromwkt_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Utf8)]);
        let err = tester.invoke_scalar("This is not valid wkt").unwrap_err();

        assert_eq!(
            err.message(),
            "Invalid argument: Expected geometry type at byte 0"
        );
    }

    #[test]
    fn geog() {
        let udf = SedonaScalarUDF::from_impl("st_geogfromwkt", st_geogfromwkt_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Utf8)]);
        tester.assert_return_type(WKB_GEOGRAPHY_WGS84.clone());

        let result = tester.invoke_scalar("POINT (1 2)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");

        let udf = SedonaScalarUDF::from_impl("st_geogfromwkb", st_geogfromwkb_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Binary)]);
        tester.assert_return_type(WKB_VIEW_GEOGRAPHY_WGS84.clone());

        let result = tester
            .invoke_scalar(create_scalar_storage(Some("POINT (1 2)"), &WKB_GEOGRAPHY))
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");
    }
}
