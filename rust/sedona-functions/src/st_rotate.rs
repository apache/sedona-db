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
use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{transform::transform, wkb_factory::WKB_MIN_PROBABLE_BYTES};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::sync::Arc;

use crate::{
    executor::WkbExecutor,
    st_affine_helpers::{self, RotateAxis},
};

/// ST_Rotate() scalar UDF
///
/// Native implementation for rotate transformation
pub fn st_rotate_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_rotate",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STRotate {
            axis: RotateAxis::Z,
        })]),
        Volatility::Immutable,
        Some(st_rotate_doc("")),
    )
}

/// ST_RotateX() scalar UDF
///
/// Native implementation for rotate transformation
pub fn st_rotate_x_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_rotatex",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STRotate {
            axis: RotateAxis::X,
        })]),
        Volatility::Immutable,
        Some(st_rotate_doc("X")),
    )
}

/// ST_RotateY() scalar UDF
///
/// Native implementation for rotate transformation
pub fn st_rotate_y_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_rotatey",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STRotate {
            axis: RotateAxis::Y,
        })]),
        Volatility::Immutable,
        Some(st_rotate_doc("Y")),
    )
}

fn st_rotate_doc(axis: &str) -> Documentation {
    let suffix = match axis {
        "Z" => "",
        _ => axis,
    };
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!("Rotates a geometry by a specified angle in radians counter-clockwise around the {axis}-axis "),
        format!("ST_Rotate{suffix} (geom: Geometry, rot: Double)"),
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("rot", "angle (in radians)")
    .with_sql_example(
        format!("SELECT ST_Rotate{suffix}(ST_GeomFromText('POLYGON Z ((1 0 1, 1 1 1, 2 2 2, 1 0 1))'), radians(45))"),
    )
    .build()
}

#[derive(Debug)]
struct STRotate {
    axis: RotateAxis,
}

impl SedonaScalarKernel for STRotate {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let angle = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;

        let mut affine_iter = st_affine_helpers::DAffineIterator::from_angle(&angle, self.axis)?;

        executor.execute_wkb_void(|maybe_wkb| {
            let maybe_mat = affine_iter.next().unwrap();
            match (maybe_wkb, maybe_mat) {
                (Some(wkb), Some(mat)) => {
                    transform(&wkb, &mat, &mut builder)
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Array;
    use datafusion_common::ScalarValue;
    use datafusion_expr::{ColumnarValue, ScalarUDF};
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, create::create_scalar,
        testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_rotate_udf: ScalarUDF = st_rotate_udf().into();
        assert_eq!(st_rotate_udf.name(), "st_rotate");
        assert!(st_rotate_udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_rotate_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT M (1 2 3)"),
            ],
            &sedona_type,
        );
        let expected_identity = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT M (1 2 3)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_identity = tester
            .invoke_arrays(prepare_args(points.clone(), &[Some(0.0_f64.to_radians())]))
            .unwrap();
        assert_array_equal(&result_identity, &expected_identity);

        // Don't test the rotated results here since it's hard to match with the exact number.
    }

    fn prepare_args(wkt: Arc<dyn Array>, mat: &[Option<f64>]) -> Vec<Arc<dyn Array>> {
        let n = wkt.len();
        let mut args: Vec<Arc<dyn Array>> = mat
            .iter()
            .map(|a| {
                let values = vec![*a; n];
                Arc::new(arrow_array::Float64Array::from(values)) as Arc<dyn Array>
            })
            .collect();
        args.insert(0, wkt);
        args
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_rotate_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(sedona_type.clone());

        let geom = create_scalar(Some("POINT (1 2)"), &sedona_type);
        let args = vec![
            ColumnarValue::Scalar(geom),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(0.0))),
        ];

        let result = tester.invoke(args).unwrap();
        if let ColumnarValue::Scalar(scalar) = result {
            tester.assert_scalar_result_equals(scalar, "POINT (1 2)");
        } else {
            panic!("Expected scalar result from item CRS rotate invoke");
        }
    }
}
