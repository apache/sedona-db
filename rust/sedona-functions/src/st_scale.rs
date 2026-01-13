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
use arrow_array::{builder::BinaryBuilder, Array};
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
    st_affine_helpers::{self},
};

/// ST_Scale() scalar UDF
///
/// Native implementation for scale transformation
pub fn st_scale_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_scale",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STScale { is_3d: true }),
            Arc::new(STScale { is_3d: false }),
        ]),
        Volatility::Immutable,
        Some(st_scale_doc()),
    )
}

fn st_scale_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Scales the geometry to a new size by multiplying the ordinates with the corresponding scaling factors",
        "ST_Scale (geom: Geometry, scaleX: Double, scaleY: Double, scaleZ: Double)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("scaleX", "scaling factor for X")
    .with_argument("scaleY", "scaling factor for Y")
    .with_argument("scaleZ", "scaling factor for Z")
    .with_sql_example(
        "SELECT ST_Scale(ST_GeomFromText('POLYGON Z ((1 0 1, 1 1 1, 2 2 2, 1 0 1))'), 1, 2, 3)",
    )
    .build()
}

#[derive(Debug)]
struct STScale {
    is_3d: bool,
}

impl SedonaScalarKernel for STScale {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let arg_matchers = if self.is_3d {
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
            ]
        } else {
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
            ]
        };

        let matcher = ArgMatcher::new(arg_matchers, WKB_GEOMETRY);

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

        let array_args = args[1..]
            .iter()
            .map(|arg| {
                arg.cast_to(&DataType::Float64, None)?
                    .to_array(executor.num_iterations())
            })
            .collect::<Result<Vec<Arc<dyn Array>>>>()?;

        let mut affine_iter = if self.is_3d {
            st_affine_helpers::DAffineIterator::from_scale_3d(&array_args)?
        } else {
            st_affine_helpers::DAffineIterator::from_scale_2d(&array_args)?
        };

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
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_scale_udf: ScalarUDF = st_scale_udf().into();
        assert_eq!(st_scale_udf.name(), "st_scale");
        assert!(st_scale_udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester_2d = ScalarUdfTester::new(
            st_scale_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester_2d.assert_return_type(WKB_GEOMETRY);

        let points_2d = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT M (1 2 3)"),
            ],
            &sedona_type,
        );

        let expected_identity_2d = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT M (1 2 3)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_identity_2d = tester_2d
            .invoke_arrays(prepare_args(points_2d.clone(), &[Some(1.0), Some(1.0)]))
            .unwrap();
        assert_array_equal(&result_identity_2d, &expected_identity_2d);

        let expected_scale_2d = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (10 40)"),
                Some("POINT M (10 40 3)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_scale_2d = tester_2d
            .invoke_arrays(prepare_args(points_2d, &[Some(10.0), Some(20.0)]))
            .unwrap();
        assert_array_equal(&result_scale_2d, &expected_scale_2d);

        let points_3d = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT Z (1 2 3)"),
                Some("POINT ZM (1 2 3 4)"),
            ],
            &sedona_type,
        );

        let expected_scale_2d_on_3d = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT Z (2 6 3)"),
                Some("POINT ZM (2 6 3 4)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_scale_2d_on_3d = tester_2d
            .invoke_arrays(prepare_args(points_3d.clone(), &[Some(2.0), Some(3.0)]))
            .unwrap();
        assert_array_equal(&result_scale_2d_on_3d, &expected_scale_2d_on_3d);

        let tester_3d = ScalarUdfTester::new(
            st_scale_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester_3d.assert_return_type(WKB_GEOMETRY);

        let expected_identity_3d = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT Z (1 2 3)"),
                Some("POINT ZM (1 2 3 4)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_identity_3d = tester_3d
            .invoke_arrays(prepare_args(
                points_3d.clone(),
                &[Some(1.0), Some(1.0), Some(1.0)],
            ))
            .unwrap();
        assert_array_equal(&result_identity_3d, &expected_identity_3d);

        let expected_scale_3d = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT Z (2 6 12)"),
                Some("POINT ZM (2 6 12 4)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_scale_3d = tester_3d
            .invoke_arrays(prepare_args(points_3d, &[Some(2.0), Some(3.0), Some(4.0)]))
            .unwrap();
        assert_array_equal(&result_scale_3d, &expected_scale_3d);

        let points_2d_for_3d = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT M (1 2 3)"),
            ],
            &sedona_type,
        );

        let expected_scale_3d_on_2d = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (2 6)"),
                Some("POINT M (2 6 3)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_scale_3d_on_2d = tester_3d
            .invoke_arrays(prepare_args(
                points_2d_for_3d,
                &[Some(2.0), Some(3.0), Some(4.0)],
            ))
            .unwrap();
        assert_array_equal(&result_scale_3d_on_2d, &expected_scale_3d_on_2d);
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
            st_scale_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester.assert_return_type(sedona_type.clone());

        let result = tester.invoke_scalar_scalar_scalar("POINT (1 2)", 2, 3).unwrap();
        tester.assert_scalar_result_equals("POINT (2 6)"); 
    }
}
