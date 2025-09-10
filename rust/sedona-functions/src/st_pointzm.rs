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
use std::{
    io::{Cursor, Write},
    sync::Arc,
    vec,
};

use arrow_array::{builder::BinaryBuilder, Array};
use arrow_schema::DataType;
use datafusion_common::cast::as_float64_array;
use datafusion_common::error::Result;
use datafusion_common::scalar::ScalarValue;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::Dimensions;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::{
    error::SedonaGeometryError,
    wkb_factory::{write_wkb_coord, write_wkb_point_header},
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::WkbExecutor;

// 1 byte for endian(1) + 4 bytes for type(4)
const WKB_HEADER_SIZE: usize = 5;

/// ST_PointZ() scalar UDF implementation
///
/// Native implementation to create Z geometries from coordinates.
pub fn st_pointz_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_pointz",
        vec![Arc::new(STGeoFromPointZm {
            out_type: WKB_GEOMETRY,
            dim: Dimensions::Xyz,
        })],
        Volatility::Immutable,
        Some(three_coord_point_doc("ST_PointZ", "Geometry", "Z")),
    )
}

/// ST_PointM() scalar UDF implementation
///
/// Native implementation to create M geometries from coordinates.
pub fn st_pointm_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_pointm",
        vec![Arc::new(STGeoFromPointZm {
            out_type: WKB_GEOMETRY,
            dim: Dimensions::Xym,
        })],
        Volatility::Immutable,
        Some(three_coord_point_doc("ST_PointM", "Geometry", "M")),
    )
}

/// ST_PointZM() scalar UDF implementation
///
/// Native implementation to create ZM geometries from coordinates.
pub fn st_pointzm_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_pointzm",
        vec![Arc::new(STGeoFromPointZm {
            out_type: WKB_GEOMETRY,
            dim: Dimensions::Xyzm,
        })],
        Volatility::Immutable,
        Some(xyzm_point_doc("ST_PointZM", "Geometry")),
    )
}

fn three_coord_point_doc(name: &str, out_type_name: &str, third_dim: &str) -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!(
            "Construct a Point {} from X, Y and {}",
            out_type_name.to_lowercase(),
            third_dim
        ),
        format!("{name} (x: Double, y: Double, z: Double)"),
    )
    .with_argument("x", "double: X value")
    .with_argument("y", "double: Y value")
    .with_argument(
        third_dim.to_lowercase(),
        format!("double: {third_dim} value"),
    )
    .with_sql_example(format!("{name}(-64.36, 45.09, 100.0)"))
    .build()
}

fn xyzm_point_doc(name: &str, out_type_name: &str) -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!(
            "Construct a Point {} from X, Y, Z and M",
            out_type_name.to_lowercase()
        ),
        format!("{name} (x: Double, y: Double, z: Double)"),
    )
    .with_argument("x", "double: X value")
    .with_argument("y", "double: Y value")
    .with_argument("z", "double: Z value")
    .with_argument("m", "double: M value")
    .with_sql_example(format!("{name}(-64.36, 45.09, 100.0, 50.0)"))
    .build()
}

#[derive(Debug)]
struct STGeoFromPointZm {
    out_type: SedonaType,
    dim: Dimensions,
}

impl SedonaScalarKernel for STGeoFromPointZm {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let num_coords = self.dim.size();
        let expected_args = vec![ArgMatcher::is_numeric(); num_coords];
        let matcher = ArgMatcher::new(expected_args, self.out_type.clone());
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let num_coords = self.dim.size();
        let executor = WkbExecutor::new(arg_types, args);

        // Cast all arguments to Float64
        let coord_values: Result<Vec<_>> = args
            .iter()
            .map(|arg| arg.cast_to(&DataType::Float64, None))
            .collect();
        let coord_values = coord_values?;

        // Check if all arguments are scalars
        let all_scalars = coord_values
            .iter()
            .all(|v| matches!(v, ColumnarValue::Scalar(_)));

        if all_scalars {
            let scalar_coords: Result<Vec<_>> = coord_values
                .iter()
                .map(|v| match v {
                    ColumnarValue::Scalar(ScalarValue::Float64(val)) => Ok(*val),
                    _ => Err(datafusion_common::DataFusionError::Internal(
                        "Expected Float64 scalar".to_string(),
                    )),
                })
                .collect();
            let scalar_coords = scalar_coords?;

            // Check if any coordinate is null
            if scalar_coords.iter().any(|coord| coord.is_none()) {
                return Ok(ScalarValue::Binary(None).into());
            }

            // Populate WKB with coordinates
            let coord_values: Vec<f64> = scalar_coords.into_iter().map(|c| c.unwrap()).collect();
            let mut buffer = Vec::new();
            let mut cursor = Cursor::new(&mut buffer);
            write_wkb_pointzm(&mut cursor, &coord_values, self.dim)
                .map_err(|err| -> DataFusionError { DataFusionError::External(Box::new(err)) })?;
            return Ok(ScalarValue::Binary(Some(buffer)).into());
        }

        // Handle array case
        let coord_arrays: Result<Vec<_>> = coord_values
            .iter()
            .map(|v| v.to_array(executor.num_iterations()))
            .collect();
        let coord_arrays = coord_arrays?;

        let coord_f64_arrays: Result<Vec<_>> = coord_arrays
            .iter()
            .map(|array| as_float64_array(array))
            .collect();
        let coord_f64_arrays = coord_f64_arrays?;

        // Calculate WKB item size based on coordinates: endian(1) + type(4) + coords(8 each)
        let wkb_size = WKB_HEADER_SIZE + (num_coords * 8);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            wkb_size * executor.num_iterations(),
        );

        for i in 0..executor.num_iterations() {
            let num_dimensions = self.dim.size();
            let arrays = (0..num_dimensions)
                .map(|j| coord_f64_arrays[j])
                .collect::<Vec<_>>();
            let any_null = arrays.iter().any(|&v| v.is_null(i));
            let values = arrays.iter().map(|v| v.value(i)).collect::<Vec<_>>();
            if !any_null {
                write_wkb_pointzm(&mut builder, &values, self.dim).map_err(|_| {
                    datafusion_common::DataFusionError::Internal(
                        "Failed to write WKB point header".to_string(),
                    )
                })?;
                builder.append_value([]);
            } else {
                builder.append_null();
            }
        }

        let new_array = builder.finish();
        Ok(ColumnarValue::Array(Arc::new(new_array)))
    }
}

fn write_wkb_pointzm(
    buf: &mut impl Write,
    coords: &[f64],
    dim: Dimensions,
) -> Result<(), SedonaGeometryError> {
    let values = coords;
    write_wkb_point_header(buf, dim)?;
    match dim.size() {
        3 => {
            let coord = (values[0], values[1], values[2]);
            write_wkb_coord(buf, coord)
        }
        4 => {
            let coord = (values[0], values[1], values[2], values[3]);
            write_wkb_coord(buf, coord)
        }
        _ => Err(SedonaGeometryError::Invalid(
            "Unsupported number of dimensions".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;

    use sedona_testing::{create::create_array, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let pointz: ScalarUDF = st_pointz_udf().into();
        assert_eq!(pointz.name(), "st_pointz");
        assert!(pointz.documentation().is_some());

        let pointm: ScalarUDF = st_pointm_udf().into();
        assert_eq!(pointm.name(), "st_pointm");
        assert!(pointm.documentation().is_some());

        let pointzm: ScalarUDF = st_pointzm_udf().into();
        assert_eq!(pointzm.name(), "st_pointzm");
        assert!(pointzm.documentation().is_some());
    }

    #[rstest]
    #[case(DataType::Float64, DataType::Float64)]
    #[case(DataType::Float32, DataType::Float64)]
    #[case(DataType::Float64, DataType::Float32)]
    #[case(DataType::Float32, DataType::Float32)]
    fn udf_invoke(#[case] lhs_type: DataType, #[case] rhs_type: DataType) {
        // Just test one of the UDFs
        // We have other functions to ensure the logic works for all Z, M, and ZM
        let udf = st_pointz_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                SedonaType::Arrow(lhs_type.clone()),
                SedonaType::Arrow(rhs_type.clone()),
                SedonaType::Arrow(lhs_type.clone()),
            ],
        );

        let array_1 =
            ColumnarValue::Array(create_array!(Float64, [Some(1.0), Some(2.0), None, None]))
                .cast_to(&lhs_type, None)
                .unwrap()
                .to_array(4)
                .unwrap();
        let array_2 =
            ColumnarValue::Array(create_array!(Float64, [Some(5.0), None, Some(7.0), None]))
                .cast_to(&rhs_type, None)
                .unwrap()
                .to_array(4)
                .unwrap();

        let array_3 = ColumnarValue::Array(create_array!(
            Float64,
            [Some(3.0), Some(3.0), Some(3.0), None]
        ))
        .cast_to(&lhs_type, None)
        .unwrap()
        .to_array(4)
        .unwrap();

        // Check scalar
        let result = tester.invoke_scalar_scalar_scalar(1.0, 2.0, 3.0).unwrap();
        tester.assert_scalar_result_equals(result, "POINT Z (1 2 3)");

        // Check scalar nulls
        let result = tester
            .invoke_scalar_scalar_scalar(ScalarValue::Null, 2.0, 3.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        let result = tester
            .invoke_scalar_scalar_scalar(1.0, ScalarValue::Null, 3.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        let result = tester
            .invoke_scalar_scalar_scalar(1.0, 2.0, ScalarValue::Null)
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        // Check arrays
        let result = tester
            .invoke_arrays(vec![array_1.clone(), array_2.clone(), array_3.clone()])
            .unwrap();
        assert_eq!(
            &result,
            &create_array(&[Some("POINT Z (1 5 3)"), None, None, None], &WKB_GEOMETRY)
        );
    }

    #[test]
    fn test_pointz() {
        let udf = st_pointz_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );

        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester.invoke_scalar_scalar_scalar(1.0, 2.0, 3.0).unwrap();
        tester.assert_scalar_result_equals(result, "POINT Z (1 2 3)");

        // Test array case
        let array_1: ArrayRef = create_array!(Float64, [Some(1.0), Some(2.0), None, None]);
        let array_2: ArrayRef = create_array!(Float64, [Some(5.0), None, Some(7.0), None]);
        let array_3: ArrayRef = create_array!(Float64, [Some(3.0), Some(3.0), Some(3.0), None]);

        let result = tester
            .invoke_arrays(vec![array_1.clone(), array_2.clone(), array_3.clone()])
            .unwrap();
        assert_eq!(
            &result,
            &create_array(&[Some("POINT Z (1 5 3)"), None, None, None], &WKB_GEOMETRY)
        );
    }

    #[test]
    fn test_pointm() {
        let udf = st_pointm_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );

        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester.invoke_scalar_scalar_scalar(1.0, 2.0, 3.0).unwrap();
        tester.assert_scalar_result_equals(result, "POINT M (1 2 3)");

        // Test array case
        let array_1: ArrayRef = create_array!(Float64, [Some(1.0), Some(2.0), None, None]);
        let array_2: ArrayRef = create_array!(Float64, [Some(5.0), None, Some(7.0), None]);
        let array_3: ArrayRef = create_array!(Float64, [Some(3.0), Some(3.0), Some(3.0), None]);

        let result = tester
            .invoke_arrays(vec![array_1.clone(), array_2.clone(), array_3.clone()])
            .unwrap();
        assert_eq!(
            &result,
            &create_array(&[Some("POINT M (1 5 3)"), None, None, None], &WKB_GEOMETRY)
        );
    }

    #[test]
    fn test_pointzm() {
        let udf = st_pointzm_udf();

        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );

        tester.assert_return_type(WKB_GEOMETRY);

        // Test array case (hard to test the scalar case compactly and it reuses code paths above)
        let array_1: ArrayRef = create_array!(Float64, [Some(1.0), Some(2.0), None, None]);
        let array_2: ArrayRef = create_array!(Float64, [Some(5.0), None, Some(7.0), None]);
        let array_3: ArrayRef = create_array!(Float64, [Some(3.0), Some(3.0), Some(3.0), None]);
        let array_4: ArrayRef = create_array!(Float64, [Some(9.0), Some(4.0), Some(4.0), None]);

        let result = tester
            .invoke_arrays(vec![
                array_1.clone(),
                array_2.clone(),
                array_3.clone(),
                array_4.clone(),
            ])
            .unwrap();
        assert_eq!(
            &result,
            &create_array(
                &[Some("POINT ZM (1 5 3 9)"), None, None, None],
                &WKB_GEOMETRY
            )
        );
    }
}
