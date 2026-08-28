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

use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{cast::as_float64_array, error::Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::{executor::WkbExecutor, st_setsrid::SRIDifiedKernel};

// 1 byte order + 4 geometry type + 4 ring count + 4 point count + 5 * (8-byte X + 8-byte Y) = 93 bytes.
const ENVELOPE_WKB_SIZE: usize = 93;

/// ST_MakeEnvelope() scalar UDF implementation
///
/// Constructs a rectangular polygon from minimum and maximum X/Y coordinates,
/// with an optional SRID or CRS argument.
pub fn st_makeenvelope_udf() -> SedonaScalarUDF {
    let kernel = Arc::new(STMakeEnvelope {});
    let sridified_kernel = Arc::new(SRIDifiedKernel::new(kernel.clone()));

    SedonaScalarUDF::new(
        "st_makeenvelope",
        vec![sridified_kernel, kernel],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STMakeEnvelope {}

impl SedonaScalarKernel for STMakeEnvelope {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(vec![ArgMatcher::is_numeric(); 4], WKB_GEOMETRY).match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let min_x = args[0].cast_to(&DataType::Float64, None)?;
        let min_y = args[1].cast_to(&DataType::Float64, None)?;
        let max_x = args[2].cast_to(&DataType::Float64, None)?;
        let max_y = args[3].cast_to(&DataType::Float64, None)?;

        let mut item = empty_envelope_wkb();

        if let (
            ColumnarValue::Scalar(ScalarValue::Float64(min_x)),
            ColumnarValue::Scalar(ScalarValue::Float64(min_y)),
            ColumnarValue::Scalar(ScalarValue::Float64(max_x)),
            ColumnarValue::Scalar(ScalarValue::Float64(max_y)),
        ) = (&min_x, &min_y, &max_x, &max_y)
        {
            return match (min_x, min_y, max_x, max_y) {
                (Some(min_x), Some(min_y), Some(max_x), Some(max_y)) => {
                    write_envelope_coordinates(&mut item, *min_x, *min_y, *max_x, *max_y);
                    Ok(ScalarValue::Binary(Some(item.to_vec())).into())
                }
                _ => Ok(ScalarValue::Binary(None).into()),
            };
        }

        let num_iterations = executor.num_iterations();
        let min_x_array = min_x.to_array(num_iterations)?;
        let min_y_array = min_y.to_array(num_iterations)?;
        let max_x_array = max_x.to_array(num_iterations)?;
        let max_y_array = max_y.to_array(num_iterations)?;
        let min_x_f64 = as_float64_array(&min_x_array)?;
        let min_y_f64 = as_float64_array(&min_y_array)?;
        let max_x_f64 = as_float64_array(&max_x_array)?;
        let max_y_f64 = as_float64_array(&max_y_array)?;

        let mut builder =
            BinaryBuilder::with_capacity(num_iterations, ENVELOPE_WKB_SIZE * num_iterations);

        for (((min_x, min_y), max_x), max_y) in
            zip(zip(zip(min_x_f64, min_y_f64), max_x_f64), max_y_f64)
        {
            match (min_x, min_y, max_x, max_y) {
                (Some(min_x), Some(min_y), Some(max_x), Some(max_y)) => {
                    write_envelope_coordinates(&mut item, min_x, min_y, max_x, max_y);
                    builder.append_value(item);
                }
                _ => builder.append_null(),
            }
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish())))
    }
}

fn empty_envelope_wkb() -> [u8; ENVELOPE_WKB_SIZE] {
    let mut item = [0; ENVELOPE_WKB_SIZE];
    item[0] = 1; // little endian
    item[1..5].copy_from_slice(&3_u32.to_le_bytes()); // Polygon
    item[5..9].copy_from_slice(&1_u32.to_le_bytes()); // one ring
    item[9..13].copy_from_slice(&5_u32.to_le_bytes()); // five coordinates
    item
}

fn write_envelope_coordinates(
    item: &mut [u8; ENVELOPE_WKB_SIZE],
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) {
    let coordinates = [
        (min_x, min_y),
        (min_x, max_y),
        (max_x, max_y),
        (max_x, min_y),
        (min_x, min_y),
    ];

    for (index, (x, y)) in coordinates.into_iter().enumerate() {
        let offset = 13 + index * 16;
        item[offset..offset + 8].copy_from_slice(&x.to_le_bytes());
        item[offset + 8..offset + 16].copy_from_slice(&y.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::ArrayRef;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY};
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal_wkb_geometry},
        create::{create_array, create_array_item_crs},
        testers::ScalarUdfTester,
    };

    use super::*;

    fn scalar(value: Option<f64>) -> ColumnarValue {
        ScalarValue::Float64(value).into()
    }

    fn invoke_scalar(tester: &ScalarUdfTester, bounds: [Option<f64>; 4]) -> ScalarValue {
        match tester
            .invoke(bounds.into_iter().map(scalar).collect())
            .unwrap()
        {
            ColumnarValue::Scalar(value) => value,
            _ => panic!("Expected scalar result"),
        }
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_makeenvelope_udf().into();
        assert_eq!(udf.name(), "st_makeenvelope");
    }

    #[test]
    fn udf_invoke_scalars() {
        let tester = ScalarUdfTester::new(
            st_makeenvelope_udf().into(),
            vec![SedonaType::Arrow(DataType::Float64); 4],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let result = invoke_scalar(
            &tester,
            [Some(1.234), Some(2.234), Some(3.345), Some(3.345)],
        );
        tester.assert_scalar_result_equals(
            result,
            "POLYGON ((1.234 2.234, 1.234 3.345, 3.345 3.345, 3.345 2.234, 1.234 2.234))",
        );

        let result = invoke_scalar(&tester, [Some(2.0), Some(3.0), Some(1.0), Some(0.0)]);
        tester.assert_scalar_result_equals(result, "POLYGON ((2 3, 2 0, 1 0, 1 3, 2 3))");

        let result = invoke_scalar(&tester, [Some(1.0), Some(2.0), Some(1.0), Some(2.0)]);
        tester.assert_scalar_result_equals(result, "POLYGON ((1 2, 1 2, 1 2, 1 2, 1 2))");

        let result = invoke_scalar(&tester, [Some(1.0), None, Some(3.0), Some(4.0)]);
        tester.assert_scalar_result_equals(result, ScalarValue::Null);
    }

    #[test]
    fn udf_invoke_arrays_and_broadcast_scalars() {
        let tester = ScalarUdfTester::new(
            st_makeenvelope_udf().into(),
            vec![SedonaType::Arrow(DataType::Float64); 4],
        );
        let min_x: ArrayRef = arrow_array::create_array!(Float64, [Some(0.0), Some(1.0), None]);

        let result = tester
            .invoke(vec![
                min_x.into(),
                scalar(Some(2.0)),
                scalar(Some(3.0)),
                scalar(Some(4.0)),
            ])
            .unwrap();
        let ColumnarValue::Array(result) = result else {
            panic!("Expected array result");
        };

        assert_array_equal(
            &result,
            &create_array(
                &[
                    Some("POLYGON ((0 2, 0 4, 3 4, 3 2, 0 2))"),
                    Some("POLYGON ((1 2, 1 4, 3 4, 3 2, 1 2))"),
                    None,
                ],
                &WKB_GEOMETRY,
            ),
        );
    }

    #[test]
    fn udf_invoke_with_srid_column() {
        let tester = ScalarUdfTester::new(
            st_makeenvelope_udf().into(),
            vec![
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        let result = tester
            .invoke_arrays(vec![
                arrow_array::create_array!(Float64, [Some(0.0), Some(1.0)]),
                arrow_array::create_array!(Float64, [Some(2.0), Some(3.0)]),
                arrow_array::create_array!(Float64, [Some(4.0), Some(5.0)]),
                arrow_array::create_array!(Float64, [Some(6.0), Some(7.0)]),
                arrow_array::create_array!(Int32, [Some(4326), Some(3857)]),
            ])
            .unwrap();

        assert_array_equal(
            &result,
            &create_array_item_crs(
                &[
                    Some("POLYGON ((0 2, 0 6, 4 6, 4 2, 0 2))"),
                    Some("POLYGON ((1 3, 1 7, 5 7, 5 3, 1 3))"),
                ],
                [Some("OGC:CRS84"), Some("EPSG:3857")],
                &WKB_GEOMETRY,
            ),
        );
    }

    #[test]
    fn udf_invoke_with_scalar_srid() {
        let tester = ScalarUdfTester::new(
            st_makeenvelope_udf().into(),
            vec![
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        let result = tester
            .invoke(vec![
                scalar(Some(1.0)),
                scalar(Some(2.0)),
                scalar(Some(3.0)),
                scalar(Some(4.0)),
                ScalarValue::Int32(Some(4326)).into(),
            ])
            .unwrap();
        let ColumnarValue::Scalar(result) = result else {
            panic!("Expected scalar result");
        };
        assert_scalar_equal_wkb_geometry(&result, Some("POLYGON ((1 2, 1 4, 3 4, 3 2, 1 2))"));
    }
}
