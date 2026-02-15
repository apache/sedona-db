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

use arrow_array::{builder::BinaryBuilder, Float64Array};
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

fn optional_numeric_arg_as_f64(
    args: &[ColumnarValue],
    index: usize,
    len: usize,
) -> Result<Float64Array> {
    let array = match args.get(index) {
        Some(arg) => arg.cast_to(&DataType::Float64, None)?.to_array(len)?,
        None => Arc::new(Float64Array::from(vec![0.0; len])),
    };

    Ok(as_float64_array(&array)?.clone())
}

fn build_return_matcher(is_geography: bool, num_optional_numeric_args: usize) -> ArgMatcher {
    let mut matchers = vec![if is_geography {
        ArgMatcher::is_geography()
    } else {
        ArgMatcher::is_geometry()
    }];
    for _ in 0..num_optional_numeric_args {
        matchers.push(ArgMatcher::optional(ArgMatcher::is_numeric()));
    }
    let output_type = if is_geography {
        WKB_GEOGRAPHY
    } else {
        WKB_GEOMETRY
    };
    ArgMatcher::new(matchers, output_type)
}

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
        build_return_matcher(self.is_geography, 0).match_args(args)
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
        build_return_matcher(self.is_geography, 1).match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let num_rows = executor.num_iterations();
        let mut builder = BinaryBuilder::with_capacity(num_rows, WKB_MIN_PROBABLE_BYTES * num_rows);

        let z_array = optional_numeric_arg_as_f64(args, 1, num_rows)?;
        let mut z_iter = z_array.iter();
        executor.execute_wkb_void(|maybe_wkb| {
            match (maybe_wkb, z_iter.next().unwrap()) {
                (Some(wkb), Some(z)) => {
                    let trans = Force3DTransform { z };
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
        Err(SedonaGeometryError::Invalid(
            "Unexpected call to transform_coord()".to_string(),
        ))
    }
    fn transform_coord_xyz(
        &self,
        coord: &mut (f64, f64, f64),
        input_dims: Dimensions,
    ) -> Result<(), SedonaGeometryError> {
        if matches!(
            input_dims,
            Dimensions::Xy | Dimensions::Xym | Dimensions::Unknown(_)
        ) {
            coord.2 = self.z
        }
        Ok(())
    }
}

// *** 3DM *************************

/// ST_Force3DM() scalar UDF
pub fn st_force3dm_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_force3dm",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STForce3DM {
                is_geography: false,
            }),
            Arc::new(STForce3DM { is_geography: true }),
        ]),
        Volatility::Immutable,
        Some(st_force3dm_doc()),
    )
}

fn st_force3dm_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Forces the geometry into a 3DM-dimensional model.",
        "ST_Force3DM (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("m", "numeric: default M value")
    .with_sql_example("SELECT ST_Force3DM(ST_GeomFromWKT('POINT (1 2)'))")
    .build()
}

#[derive(Debug)]
struct STForce3DM {
    is_geography: bool,
}

impl SedonaScalarKernel for STForce3DM {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        build_return_matcher(self.is_geography, 1).match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let num_rows = executor.num_iterations();
        let mut builder = BinaryBuilder::with_capacity(num_rows, WKB_MIN_PROBABLE_BYTES * num_rows);

        let m_array = optional_numeric_arg_as_f64(args, 1, num_rows)?;
        let mut m_iter = m_array.iter();
        executor.execute_wkb_void(|maybe_wkb| {
            match (maybe_wkb, m_iter.next().unwrap()) {
                (Some(wkb), Some(m)) => {
                    let trans = Force3DMTransform { m };
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
struct Force3DMTransform {
    m: f64,
}

impl CrsTransform for Force3DMTransform {
    fn output_dim(&self) -> Option<geo_traits::Dimensions> {
        Some(geo_traits::Dimensions::Xym)
    }

    fn transform_coord(
        &self,
        _coord: &mut (f64, f64),
    ) -> std::result::Result<(), SedonaGeometryError> {
        Err(SedonaGeometryError::Invalid(
            "Unexpected call to transform_coord()".to_string(),
        ))
    }

    fn transform_coord_xym(
        &self,
        coord: &mut (f64, f64, f64),
        input_dims: Dimensions,
    ) -> Result<(), SedonaGeometryError> {
        if matches!(
            input_dims,
            Dimensions::Xy | Dimensions::Xyz | Dimensions::Unknown(_)
        ) {
            coord.2 = self.m;
        }
        Ok(())
    }
}

// *** 4D *************************

/// ST_Force4D() scalar UDF
pub fn st_force4d_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_force4d",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STForce4D {
                is_geography: false,
            }),
            Arc::new(STForce4D { is_geography: true }),
        ]),
        Volatility::Immutable,
        Some(st_force4d_doc()),
    )
}

fn st_force4d_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Forces the geometry into a 4-dimensional model.",
        "ST_Force4D (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("z", "numeric: default Z value")
    .with_argument("m", "numeric: default M value")
    .with_sql_example("SELECT ST_Force4D(ST_GeomFromWKT('POINT (1 2)'))")
    .build()
}

#[derive(Debug)]
struct STForce4D {
    is_geography: bool,
}

impl SedonaScalarKernel for STForce4D {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        build_return_matcher(self.is_geography, 2).match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let num_rows = executor.num_iterations();
        let mut builder = BinaryBuilder::with_capacity(num_rows, WKB_MIN_PROBABLE_BYTES * num_rows);

        let z_array = optional_numeric_arg_as_f64(args, 1, num_rows)?;
        let m_array = optional_numeric_arg_as_f64(args, 2, num_rows)?;
        let mut z_iter = z_array.iter();
        let mut m_iter = m_array.iter();
        executor.execute_wkb_void(|maybe_wkb| {
            match (maybe_wkb, z_iter.next().unwrap(), m_iter.next().unwrap()) {
                (Some(wkb), Some(z), Some(m)) => {
                    let trans = Force4DTransform { z, m };
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
struct Force4DTransform {
    z: f64,
    m: f64,
}

impl CrsTransform for Force4DTransform {
    fn output_dim(&self) -> Option<geo_traits::Dimensions> {
        Some(geo_traits::Dimensions::Xyzm)
    }

    fn transform_coord(
        &self,
        _coord: &mut (f64, f64),
    ) -> std::result::Result<(), SedonaGeometryError> {
        Err(SedonaGeometryError::Invalid(
            "Unexpected call to transform_coord()".to_string(),
        ))
    }

    fn transform_coord_xyzm(
        &self,
        coord: &mut (f64, f64, f64, f64),
        input_dims: Dimensions,
    ) -> Result<(), SedonaGeometryError> {
        match input_dims {
            Dimensions::Xy | Dimensions::Unknown(_) => {
                coord.2 = self.z;
                coord.3 = self.m;
            }
            Dimensions::Xyz => {
                coord.3 = self.m;
            }
            Dimensions::Xym => {
                coord.2 = self.z;
            }
            Dimensions::Xyzm => {}
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
        let st_force2d: ScalarUDF = st_force2d_udf().into();
        assert_eq!(st_force2d.name(), "st_force2d");
        assert!(st_force2d.documentation().is_some());

        let st_force3d: ScalarUDF = st_force3d_udf().into();
        assert_eq!(st_force3d.name(), "st_force3d");
        assert!(st_force3d.documentation().is_some());

        let st_force3dm: ScalarUDF = st_force3dm_udf().into();
        assert_eq!(st_force3dm.name(), "st_force3dm");
        assert!(st_force3dm.documentation().is_some());

        let st_force4d: ScalarUDF = st_force4d_udf().into();
        assert_eq!(st_force4d.name(), "st_force4d");
        assert!(st_force4d.documentation().is_some());
    }

    #[rstest]
    fn udf_2d(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_force2d_udf().into(), vec![sedona_type.clone()]);
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
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT (3 4)"),
                Some("POINT (8 9)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points]).unwrap();
        assert_array_equal(&result, &expected);
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
                Some("POINT M (6 7 8)"),
                Some("POINT ZM (9 10 11 12)"),
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
                Some("POINT Z (6 7 0)"),
                Some("POINT Z (9 10 11)"),
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
                Some("POINT (1 2)"),
                Some("POINT Z (3 4 5)"),
                Some("POINT M (6 7 8)"),
                Some("POINT ZM (9 10 11 12)"),
            ],
            &sedona_type,
        );
        let z = create_array!(
            Float64,
            [
                Some(9.0),
                Some(9.0),
                Some(9.0),
                None,
                Some(9.0),
                Some(9.0),
                Some(9.0)
            ]
        );

        let expected = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT Z (1 2 9)"),
                None,
                Some("POINT Z (3 4 5)"),
                Some("POINT Z (6 7 9)"),
                Some("POINT Z (9 10 11)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points, z]).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_3dm_without_m(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_force3dm_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT Z (3 4 5)"),
                Some("POINT M (6 7 8)"),
                Some("POINT ZM (9 10 11 12)"),
            ],
            &sedona_type,
        );

        let expected = create_array(
            &[
                None,
                Some("POINT M EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT M (1 2 0)"),
                Some("POINT M (3 4 0)"),
                Some("POINT M (6 7 8)"),
                Some("POINT M (9 10 12)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points]).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_3dm_with_m(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_force3dm_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT (1 2)"),
                Some("POINT Z (3 4 5)"),
                Some("POINT M (6 7 8)"),
                Some("POINT ZM (9 10 11 12)"),
            ],
            &sedona_type,
        );
        let m = create_array!(
            Float64,
            [
                Some(9.0),
                Some(9.0),
                Some(9.0),
                None,
                Some(9.0),
                Some(9.0),
                Some(9.0)
            ]
        );

        let expected = create_array(
            &[
                None,
                Some("POINT M EMPTY"),
                Some("POINT M (1 2 9)"),
                None,
                Some("POINT M (3 4 9)"),
                Some("POINT M (6 7 8)"),
                Some("POINT M (9 10 12)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points, m]).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_4d_without_defaults(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_force4d_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT Z EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT Z (3 4 5)"),
                Some("POINT M (6 7 8)"),
                Some("POINT ZM (9 10 11 12)"),
            ],
            &sedona_type,
        );

        let expected = create_array(
            &[
                None,
                Some("POINT ZM EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT ZM (1 2 0 0)"),
                Some("POINT ZM (3 4 5 0)"),
                Some("POINT ZM (6 7 0 8)"),
                Some("POINT ZM (9 10 11 12)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points]).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_4d_with_defaults(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_force4d_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT (1 2)"),
                Some("POINT Z (3 4 5)"),
                Some("POINT M (6 7 8)"),
                Some("POINT ZM (9 10 11 12)"),
            ],
            &sedona_type,
        );
        let z = create_array!(
            Float64,
            [
                Some(8.0),
                Some(8.0),
                Some(8.0),
                None,
                Some(8.0),
                Some(8.0),
                Some(8.0)
            ]
        );
        let m = create_array!(
            Float64,
            [
                Some(9.0),
                Some(9.0),
                Some(9.0),
                Some(9.0),
                Some(9.0),
                Some(9.0),
                Some(9.0)
            ]
        );

        let expected = create_array(
            &[
                None,
                Some("POINT ZM EMPTY"),
                Some("POINT ZM (1 2 8 9)"),
                None,
                Some("POINT ZM (3 4 5 9)"),
                Some("POINT ZM (6 7 8 8)"),
                Some("POINT ZM (9 10 11 12)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points, z, m]).unwrap();
        assert_array_equal(&result, &expected);
    }
}
