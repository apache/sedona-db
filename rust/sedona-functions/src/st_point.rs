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
use std::{iter::zip, sync::Arc, vec};

use arrow_array::builder::BinaryBuilder;
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

/// ST_Point() scalar UDF implementation
///
/// Native implementation to create geometries from coordinates.
/// See [`st_geogpoint_udf`] for the corresponding geography constructor.
pub fn st_point_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_point",
        vec![Arc::new(STGeoFromPoint {
            out_type: WKB_GEOMETRY,
        })],
        Volatility::Immutable,
        Some(doc("ST_Point", "Geometry")),
    )
}

/// ST_GeogPoint() scalar UDF implementation
///
/// Native implementation to create geometries from coordinates.
/// See [`st_geogpoint_udf`] for the corresponding geography constructor.
pub fn st_geogpoint_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geogpoint",
        vec![Arc::new(STGeoFromPoint {
            out_type: WKB_GEOGRAPHY,
        })],
        Volatility::Immutable,
        Some(doc("st_geogpoint", "Geography")),
    )
}

fn doc(name: &str, out_type_name: &str) -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!(
            "Construct a Point {} from X and Y",
            out_type_name.to_lowercase()
        ),
        format!("{name} (x: Double, y: Double)"),
    )
    .with_argument("x", "double: X value")
    .with_argument("y", "double: Y value")
    .with_sql_example(format!("{name}(-64.36, 45.09)"))
    .build()
}

#[derive(Debug)]
struct STGeoFromPoint {
    out_type: SedonaType,
}

impl SedonaScalarKernel for STGeoFromPoint {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_numeric(), ArgMatcher::is_numeric()],
            self.out_type.clone(),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);

        let x = &args[0].cast_to(&DataType::Float64, None)?;
        let y = &args[1].cast_to(&DataType::Float64, None)?;

        let mut item: [u8; 21] = [0x00; 21];
        item[0] = 0x01;
        item[1] = 0x01;

        // Handle the Scalar case to ensure that Scalar + Scalar -> Scalar
        if let (
            ColumnarValue::Scalar(ScalarValue::Float64(x_float)),
            ColumnarValue::Scalar(ScalarValue::Float64(y_float)),
        ) = (x, y)
        {
            if let (Some(x), Some(y)) = (x_float, y_float) {
                populate_wkb_item(&mut item, x, y);
                return Ok(ScalarValue::Binary(Some(item.to_vec())).into());
            } else {
                return Ok(ScalarValue::Binary(None).into());
            }
        }

        // Ensure both sides are arrays before iterating
        let x_array = x.to_array(executor.num_iterations())?;
        let y_array = y.to_array(executor.num_iterations())?;
        let x_f64 = as_float64_array(&x_array)?;
        let y_f64 = as_float64_array(&y_array)?;

        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            item.len() * executor.num_iterations(),
        );

        for (x_elem, y_elem) in zip(x_f64, y_f64) {
            match (x_elem, y_elem) {
                (Some(x), Some(y)) => {
                    populate_wkb_item(&mut item, &x, &y);
                    builder.append_value(item);
                }
                _ => {
                    builder.append_null();
                }
            }
        }

        let new_array = builder.finish();
        Ok(ColumnarValue::Array(Arc::new(new_array)))
    }
}

fn populate_wkb_item(item: &mut [u8], x: &f64, y: &f64) {
    item[5..13].copy_from_slice(&x.to_le_bytes());
    item[13..21].copy_from_slice(&y.to_le_bytes());
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
}
