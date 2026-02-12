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

use arrow_array::{builder::BinaryBuilder, Array, Float64Array};
use arrow_schema::DataType;
use datafusion_common::{cast::as_float64_array, error::Result, DataFusionError};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::Dimensions;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{
    error::SedonaGeometryError,
    transform::{transform, CrsTransform},
    wkb_factory::WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::WkbExecutor;

// *** 2D *************************

/// ST_Force2D() scalar UDF
pub fn st_force2d_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_force2d",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STForce2D {
                is_geography: false,
            }),
            Arc::new(STForce2D { is_geography: true }),
        ]),
        Volatility::Immutable,
        Some(st_force2d_doc()),
    )
}

fn st_force2d_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Forces the geometry into a 2-dimensional model",
        "ST_Force2D (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_Force2D(ST_GeomFromWKT('POINT Z (1 2 3)'))")
    .build()
}

#[derive(Debug)]
struct STForce2D {
    is_geography: bool,
}

impl SedonaScalarKernel for STForce2D {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = if self.is_geography {
            ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY)
        } else {
            ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY)
        };

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

        let trans = Force2DTransform {};
        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    transform(wkb, &trans, &mut builder)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    builder.append_value([]);
                }
                _ => {
                    builder.append_null();
                }
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct Force2DTransform {}

impl CrsTransform for Force2DTransform {
    fn output_dim(&self) -> Option<geo_traits::Dimensions> {
        Some(geo_traits::Dimensions::Xy)
    }

    fn transform_coord(
        &self,
        _coord: &mut (f64, f64),
    ) -> std::result::Result<(), SedonaGeometryError> {
        Ok(())
    }
}

// *** 3D *************************

/// ST_Force3D() scalar UDF
pub fn st_force3d_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_force3d",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STForce3D {
                is_geography: false,
            }),
            Arc::new(STForce3D { is_geography: true }),
        ]),
        Volatility::Immutable,
        Some(st_force3d_doc()),
    )
}

fn st_force3d_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Forces the geometry into a 3-dimensional model.",
        "ST_Force3D (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("z", "numeric: default Z value")
    .with_sql_example("SELECT ST_Force3D(ST_GeomFromWKT('POINT (1 2)'))")
    .build()
}

#[derive(Debug)]
struct STForce3D {
    is_geography: bool,
}

impl SedonaScalarKernel for STForce3D {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = if self.is_geography {
            ArgMatcher::new(
                vec![
                    ArgMatcher::is_geography(),
                    ArgMatcher::optional(ArgMatcher::is_numeric()),
                ],
                WKB_GEOGRAPHY,
            )
        } else {
            ArgMatcher::new(
                vec![
                    ArgMatcher::is_geometry(),
                    ArgMatcher::optional(ArgMatcher::is_numeric()),
                ],
                WKB_GEOMETRY,
            )
        };

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

        let z_array = match args.get(1) {
            Some(arg) => arg
                .cast_to(&DataType::Float64, None)?
                .to_array(executor.num_iterations())?,
            None => Arc::new(Float64Array::from(vec![0.0; executor.num_iterations()])),
        };
        let z_array = as_float64_array(&z_array)?;

        let mut i = 0usize;
        executor.execute_wkb_void(|maybe_wkb| {
            match (maybe_wkb, z_array.is_null(i)) {
                (Some(wkb), false) => {
                    let trans = Force3DTransform {
                        z: z_array.value(i),
                    };
                    transform(wkb, &trans, &mut builder)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    builder.append_value([]);
                }
                _ => {
                    builder.append_null();
                }
            }
            i += 1;

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct Force3DTransform {
    z: f64,
}

impl CrsTransform for Force3DTransform {
    fn output_dim(&self) -> Option<geo_traits::Dimensions> {
        Some(geo_traits::Dimensions::Xyz)
    }

    fn transform_coord(
        &self,
        _coord: &mut (f64, f64),
    ) -> std::result::Result<(), SedonaGeometryError> {
        unreachable!()
    }
    fn transform_coord_3d(
        &self,
        coord: &mut (f64, f64, f64),
        input_dims: Dimensions,
    ) -> Result<(), SedonaGeometryError> {
        // If the input doesn't have Z coordinate, fill with the default value
        if matches!(
            input_dims,
            Dimensions::Xy | Dimensions::Xym | Dimensions::Unknown(_)
        ) {
            coord.2 = self.z
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::create_array;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_force3d: ScalarUDF = st_force3d_udf().into();
        assert_eq!(st_force3d.name(), "st_force3d");
        assert!(st_force3d.documentation().is_some());
    }

    #[rstest]
    fn udf_3d_without_z(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_force3d_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT Z EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT Z (3 4 5)"),
                Some("POINT ZM (8 9 10 11)"),
            ],
            &sedona_type,
        );

        let expected = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT Z EMPTY"),
                Some("POINT Z (1 2 0)"),
                Some("POINT Z (3 4 5)"),
                Some("POINT Z (8 9 10)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points]).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_3d_with_z(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_force3d_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT Z (3 4 5)"),
                Some("POINT (6 7)"),
                Some("POINT ZM (8 9 10 11)"),
            ],
            &sedona_type,
        );
        let z = create_array!(
            Float64,
            [Some(9.0), Some(9.0), Some(9.0), Some(9.0), None, Some(9.0)]
        );

        let expected = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT Z (1 2 9)"),
                Some("POINT Z (3 4 5)"),
                None,
                Some("POINT Z (8 9 10)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points, z]).unwrap();
        assert_array_equal(&result, &expected);
    }
}
