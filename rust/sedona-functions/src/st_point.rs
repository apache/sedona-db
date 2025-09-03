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

use arrow_array::{builder::BinaryBuilder, Array};
use arrow_schema::DataType;
use datafusion_common::cast::as_float64_array;
use datafusion_common::error::Result;
use datafusion_common::scalar::ScalarValue;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{ArgMatcher, SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY};

use crate::executor::WkbExecutor;

// WKB geometry type constants following ISO standards
const WKB_POINT: u32 = 1;
const WKB_POINT_Z: u32 = 1001;
const WKB_POINT_M: u32 = 2001;
const WKB_POINT_ZM: u32 = 3001;
const WKB_HEADER_SIZE: usize = 5;

/// ST_Point() scalar UDF implementation
///
/// Native implementation to create geometries from coordinates.
/// See [`st_geogpoint_udf`] for the corresponding geography constructor.
pub fn st_point_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_point",
        vec![Arc::new(STGeoFromPoint {
            out_type: WKB_GEOMETRY,
            wkb_type: WKB_POINT,
            num_coords: 2,
        })],
        Volatility::Immutable,
        Some(doc(
            "ST_Point",
            "Geometry",
            &["x", "y"],
            "ST_Point(-64.36, 45.09)",
        )),
    )
}

/// ST_GeogPoint() scalar UDF implementation
///
/// Native implementation to create geographies from coordinates.
pub fn st_geogpoint_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geogpoint",
        vec![Arc::new(STGeoFromPoint {
            out_type: WKB_GEOGRAPHY,
            wkb_type: WKB_POINT,
            num_coords: 2,
        })],
        Volatility::Immutable,
        Some(doc(
            "st_geogpoint",
            "Geography",
            &["x", "y"],
            "st_geogpoint(-64.36, 45.09)",
        )),
    )
}

/// ST_PointZ() scalar UDF implementation
///
/// Native implementation to create Z geometries from coordinates.
pub fn st_pointz_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_pointz",
        vec![Arc::new(STGeoFromPoint {
            out_type: WKB_GEOMETRY,
            wkb_type: WKB_POINT_Z,
            num_coords: 3,
        })],
        Volatility::Immutable,
        Some(doc(
            "ST_PointZ",
            "Geometry",
            &["x", "y", "z"],
            "ST_PointZ(-64.36, 45.09, 100.0)",
        )),
    )
}

/// ST_PointM() scalar UDF implementation
///
/// Native implementation to create M geometries from coordinates.
pub fn st_pointm_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_pointm",
        vec![Arc::new(STGeoFromPoint {
            out_type: WKB_GEOMETRY,
            wkb_type: WKB_POINT_M,
            num_coords: 3,
        })],
        Volatility::Immutable,
        Some(doc(
            "ST_PointM",
            "Geometry",
            &["x", "y", "m"],
            "ST_PointM(-64.36, 45.09, 50.0)",
        )),
    )
}

/// ST_PointZM() scalar UDF implementation
///
/// Native implementation to create ZM geometries from coordinates.
pub fn st_pointzm_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_pointzm",
        vec![Arc::new(STGeoFromPoint {
            out_type: WKB_GEOMETRY,
            wkb_type: WKB_POINT_ZM,
            num_coords: 4,
        })],
        Volatility::Immutable,
        Some(doc(
            "ST_PointZM",
            "Geometry",
            &["x", "y", "z", "m"],
            "ST_PointZM(-64.36, 45.09, 100.0, 50.0)",
        )),
    )
}

fn doc(name: &str, out_type_name: &str, params: &[&str], example: &str) -> Documentation {
    let description = match params.len() {
        2 => format!(
            "Construct a Point {} from X and Y",
            out_type_name.to_lowercase()
        ),
        3 if params[2] == "z" => format!(
            "Construct a Point {} from X, Y and Z",
            out_type_name.to_lowercase()
        ),
        3 if params[2] == "m" => format!(
            "Construct a Point {} from X, Y and M",
            out_type_name.to_lowercase()
        ),
        4 => format!(
            "Construct a Point {} from X, Y, Z and M",
            out_type_name.to_lowercase()
        ),
        _ => unreachable!(),
    };

    let signature = format!(
        "{name} ({})",
        params
            .iter()
            .map(|p| format!("{}: Double", p))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut builder = Documentation::builder(DOC_SECTION_OTHER, description, signature)
        .with_argument("x", "double: X coordinate")
        .with_argument("y", "double: Y coordinate");

    if params.len() >= 3 {
        if params[2] == "z" {
            builder = builder.with_argument("z", "double: Z coordinate");
        } else if params[2] == "m" {
            builder = builder.with_argument("m", "double: M coordinate");
        }
    }

    if params.len() == 4 {
        builder = builder
            .with_argument("z", "double: Z coordinate")
            .with_argument("m", "double: M coordinate");
    }

    builder.with_sql_example(example).build()
}

#[derive(Debug)]
struct STGeoFromPoint {
    out_type: SedonaType,
    wkb_type: u32,
    num_coords: usize,
}

impl SedonaScalarKernel for STGeoFromPoint {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let expected_args = vec![ArgMatcher::is_numeric(); self.num_coords];
        let matcher = ArgMatcher::new(expected_args, self.out_type.clone());
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);

        // Cast all arguments to Float64
        let coord_values: Result<Vec<_>> = args
            .iter()
            .map(|arg| arg.cast_to(&DataType::Float64, None))
            .collect();
        let coord_values = coord_values?;

        // Calculate WKB item size based on coordinates: endian(1) + type(4) + coords(8 each)
        let wkb_size = WKB_HEADER_SIZE + (self.num_coords * 8);
        let mut item = vec![0u8; wkb_size];
        // Little endian
        item[0] = 0x01;
        // Geometry type
        item[1..WKB_HEADER_SIZE].copy_from_slice(&self.wkb_type.to_le_bytes());

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
            populate_wkb_item(&mut item, &coord_values);
            return Ok(ScalarValue::Binary(Some(item)).into());
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

        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            wkb_size * executor.num_iterations(),
        );

        for i in 0..executor.num_iterations() {
            let mut coords = Vec::with_capacity(self.num_coords);
            let mut has_null = false;

            for array in &coord_f64_arrays {
                if array.is_null(i) {
                    has_null = true;
                    break;
                } else {
                    coords.push(array.value(i));
                }
            }

            if has_null {
                builder.append_null();
            } else {
                populate_wkb_item(&mut item, &coords);
                builder.append_value(&item);
            }
        }

        let new_array = builder.finish();
        Ok(ColumnarValue::Array(Arc::new(new_array)))
    }
}

fn populate_wkb_item(item: &mut [u8], coords: &[f64]) {
    for (i, coord) in coords.iter().enumerate() {
        let start_idx = WKB_HEADER_SIZE + (i * 8);
        let end_idx = start_idx + 8;
        item[start_idx..end_idx].copy_from_slice(&coord.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::create_array;
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_testing::{
        compare::assert_value_equal,
        create::{create_array_value, create_scalar_value},
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let geom_from_point: ScalarUDF = st_point_udf().into();
        assert_eq!(geom_from_point.name(), "st_point");
        assert!(geom_from_point.documentation().is_some());

        let pointz: ScalarUDF = st_pointz_udf().into();
        assert_eq!(pointz.name(), "st_pointz");
        assert!(pointz.documentation().is_some());

        let pointm: ScalarUDF = st_pointm_udf().into();
        assert_eq!(pointm.name(), "st_pointm");
        assert!(pointm.documentation().is_some());

        let pointzm: ScalarUDF = st_pointzm_udf().into();
        assert_eq!(pointzm.name(), "st_pointzm");
        assert!(pointzm.documentation().is_some());

        let geog_from_point: ScalarUDF = st_geogpoint_udf().into();
        assert_eq!(geog_from_point.name(), "st_geogpoint");
        assert!(geog_from_point.documentation().is_some());
    }

    #[rstest]
    #[case(DataType::Float64, DataType::Float64)]
    #[case(DataType::Float32, DataType::Float64)]
    #[case(DataType::Float64, DataType::Float32)]
    #[case(DataType::Float32, DataType::Float32)]
    fn udf_invoke(#[case] lhs_type: DataType, #[case] rhs_type: DataType) {
        let udf = st_point_udf();

        let lhs_scalar_null = ScalarValue::Float64(None).cast_to(&lhs_type).unwrap();
        let lhs_scalar = ScalarValue::Float64(Some(1.0)).cast_to(&lhs_type).unwrap();
        let rhs_scalar_null = ScalarValue::Float64(None).cast_to(&rhs_type).unwrap();
        let rhs_scalar = ScalarValue::Float64(Some(2.0)).cast_to(&rhs_type).unwrap();
        let lhs_array =
            ColumnarValue::Array(create_array!(Float64, [Some(1.0), Some(2.0), None, None]))
                .cast_to(&lhs_type, None)
                .unwrap();
        let rhs_array =
            ColumnarValue::Array(create_array!(Float64, [Some(5.0), None, Some(7.0), None]))
                .cast_to(&rhs_type, None)
                .unwrap();

        // Check scalar
        assert_value_equal(
            &udf.invoke_batch(&[lhs_scalar.clone().into(), rhs_scalar.clone().into()], 3)
                .unwrap(),
            &create_scalar_value(Some("POINT (1 2)"), &WKB_GEOMETRY),
        );

        // Check scalar null combinations
        assert_value_equal(
            &udf.invoke_batch(
                &[lhs_scalar.clone().into(), rhs_scalar_null.clone().into()],
                1,
            )
            .unwrap(),
            &create_scalar_value(None, &WKB_GEOMETRY),
        );

        assert_value_equal(
            &udf.invoke_batch(
                &[lhs_scalar_null.clone().into(), rhs_scalar.clone().into()],
                1,
            )
            .unwrap(),
            &create_scalar_value(None, &WKB_GEOMETRY),
        );

        assert_value_equal(
            &udf.invoke_batch(
                &[
                    lhs_scalar_null.clone().into(),
                    rhs_scalar_null.clone().into(),
                ],
                1,
            )
            .unwrap(),
            &create_scalar_value(None, &WKB_GEOMETRY),
        );

        // Check array
        assert_value_equal(
            &udf.invoke_batch(&[lhs_array.clone(), rhs_array.clone()], 4)
                .unwrap(),
            &create_array_value(&[Some("POINT (1 5)"), None, None, None], &WKB_GEOMETRY),
        );

        // Check array/scalar combinations
        assert_value_equal(
            &udf.invoke_batch(&[lhs_array.clone(), rhs_scalar.clone().into()], 4)
                .unwrap(),
            &create_array_value(
                &[Some("POINT (1 2)"), Some("POINT (2 2)"), None, None],
                &WKB_GEOMETRY,
            ),
        );

        assert_value_equal(
            &udf.invoke_batch(&[lhs_scalar.clone().into(), rhs_array], 4)
                .unwrap(),
            &create_array_value(
                &[Some("POINT (1 5)"), None, Some("POINT (1 7)"), None],
                &WKB_GEOMETRY,
            ),
        );
    }

    #[test]
    fn geog() {
        let udf = st_geogpoint_udf();

        assert_value_equal(
            &udf.invoke_batch(
                &[
                    ScalarValue::Float64(Some(1.0)).into(),
                    ScalarValue::Float64(Some(2.0)).into(),
                ],
                1,
            )
            .unwrap(),
            &create_scalar_value(Some("POINT (1 2)"), &WKB_GEOGRAPHY),
        );
    }

    #[test]
    fn test_pointz() {
        let udf = st_pointz_udf();

        // Test scalar case
        assert_value_equal(
            &udf.invoke_batch(
                &[
                    ScalarValue::Float64(Some(1.0)).into(),
                    ScalarValue::Float64(Some(2.0)).into(),
                    ScalarValue::Float64(Some(3.0)).into(),
                ],
                1,
            )
            .unwrap(),
            &create_scalar_value(Some("POINT Z (1 2 3)"), &WKB_GEOMETRY),
        );

        // Test array and null cases
        // Even if xy are valid, result is null if z is null
        let x_array =
            ColumnarValue::Array(create_array!(Float64, [Some(1.0), Some(2.0), None, None]))
                .cast_to(&DataType::Float64, None)
                .unwrap();

        let y_array = ColumnarValue::Array(create_array!(
            Float64,
            [Some(5.0), Some(1.0), Some(7.0), None]
        ))
        .cast_to(&DataType::Float64, None)
        .unwrap();

        let z_array =
            ColumnarValue::Array(create_array!(Float64, [Some(10.0), None, Some(12.0), None]))
                .cast_to(&DataType::Float64, None)
                .unwrap();

        assert_value_equal(
            &udf.invoke_batch(&[x_array.clone(), y_array.clone(), z_array.clone()], 1)
                .unwrap(),
            &create_array_value(&[Some("POINT Z (1 5 10)"), None, None, None], &WKB_GEOMETRY),
        );
    }

    #[test]
    fn test_pointm() {
        let udf = st_pointm_udf();

        // Test scalar case
        assert_value_equal(
            &udf.invoke_batch(
                &[
                    ScalarValue::Float64(Some(1.0)).into(),
                    ScalarValue::Float64(Some(2.0)).into(),
                    ScalarValue::Float64(Some(4.0)).into(),
                ],
                1,
            )
            .unwrap(),
            &create_scalar_value(Some("POINT M (1 2 4)"), &WKB_GEOMETRY),
        );

        // Test array and null cases
        // Even if xy are valid, result is null if z is null
        let x_array =
            ColumnarValue::Array(create_array!(Float64, [Some(1.0), Some(2.0), None, None]))
                .cast_to(&DataType::Float64, None)
                .unwrap();

        let y_array = ColumnarValue::Array(create_array!(
            Float64,
            [Some(5.0), Some(1.0), Some(7.0), None]
        ))
        .cast_to(&DataType::Float64, None)
        .unwrap();

        let m_array =
            ColumnarValue::Array(create_array!(Float64, [Some(10.0), None, Some(12.0), None]))
                .cast_to(&DataType::Float64, None)
                .unwrap();

        assert_value_equal(
            &udf.invoke_batch(&[x_array.clone(), y_array.clone(), m_array.clone()], 1)
                .unwrap(),
            &create_array_value(&[Some("POINT M (1 5 10)"), None, None, None], &WKB_GEOMETRY),
        );
    }

    #[test]
    fn test_pointzm() {
        let udf = st_pointzm_udf();

        // Test scalar case
        assert_value_equal(
            &udf.invoke_batch(
                &[
                    ScalarValue::Float64(Some(1.0)).into(),
                    ScalarValue::Float64(Some(2.0)).into(),
                    ScalarValue::Float64(Some(3.0)).into(),
                    ScalarValue::Float64(Some(4.0)).into(),
                ],
                1,
            )
            .unwrap(),
            &create_scalar_value(Some("POINT ZM (1 2 3 4)"), &WKB_GEOMETRY),
        );

        // Even if xy are valid, result is null if z or m is null
        // Test array and null cases
        // Even if xy are valid, result is null if z is null
        let x_array = ColumnarValue::Array(create_array!(
            Float64,
            [Some(1.0), Some(2.0), None, Some(1.0)]
        ))
        .cast_to(&DataType::Float64, None)
        .unwrap();

        let y_array = ColumnarValue::Array(create_array!(
            Float64,
            [Some(5.0), Some(1.0), Some(7.0), Some(2.0)]
        ))
        .cast_to(&DataType::Float64, None)
        .unwrap();

        let z_array = ColumnarValue::Array(create_array!(
            Float64,
            [Some(20.0), Some(1.0), Some(7.0), None]
        ))
        .cast_to(&DataType::Float64, None)
        .unwrap();

        let m_array = ColumnarValue::Array(create_array!(
            Float64,
            [Some(10.0), None, Some(12.0), Some(4.0)]
        ))
        .cast_to(&DataType::Float64, None)
        .unwrap();

        assert_value_equal(
            &udf.invoke_batch(
                &[
                    x_array.clone(),
                    y_array.clone(),
                    z_array.clone(),
                    m_array.clone(),
                ],
                1,
            )
            .unwrap(),
            &create_array_value(
                &[Some("POINT ZM (1 5 20 10)"), None, None, None],
                &WKB_GEOMETRY,
            ),
        );
    }
}
