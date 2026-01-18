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
use arrow_array::{builder::BinaryBuilder, types::Float64Type, Array, PrimitiveArray};
use arrow_schema::DataType;
use datafusion_common::{cast::as_float64_array, error::Result, DataFusionError};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};

use sedona_common::sedona_internal_err;
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
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::sync::Arc;

use crate::executor::WkbExecutor;

/// ST_Translate() scalar UDF
pub fn st_translate_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_translate",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STTranslate { is_3d: true }),
            Arc::new(STTranslate { is_3d: false }),
        ]),
        Volatility::Immutable,
        Some(st_translate_doc()),
    )
}

fn st_translate_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Update coordinates of geom by a fixed offset",
        "ST_Translate (geom: Geometry, deltax: numeric, deltay: numeric)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("deltax", "numeric: X value difference")
    .with_argument("deltay", "numeric: Y value difference")
    .with_sql_example("SELECT ST_Translate(ST_GeomFromWKT('LINESTRING(0 1, 2 3, 4 5)'), 2.0, 3.0)")
    .build()
}

#[derive(Debug)]
struct STTranslate {
    is_3d: bool,
}

impl SedonaScalarKernel for STTranslate {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = if self.is_3d {
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
        let matcher = ArgMatcher::new(matchers, WKB_GEOMETRY);

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
            .collect::<Result<Vec<Arc<dyn arrow_array::Array>>>>()?;

        let deltax_array = as_float64_array(&array_args[0])?;
        let deltay_array = as_float64_array(&array_args[1])?;

        let mut deltas = if self.is_3d {
            if args.len() != 4 {
                return sedona_internal_err!("Invalid number of arguments are passed");
            }

            let deltaz_array = as_float64_array(&array_args[2])?;
            Deltas::new(deltax_array, deltay_array, Some(deltaz_array))
        } else {
            if args.len() != 3 {
                return sedona_internal_err!("Invalid number of arguments are passed");
            }

            Deltas::new(deltax_array, deltay_array, None)
        };

        executor.execute_wkb_void(|maybe_wkb| {
            match (maybe_wkb, deltas.next().unwrap()) {
                (Some(wkb), Some((deltax, deltay, deltaz))) => {
                    let trans = Translate {
                        deltax,
                        deltay,
                        deltaz,
                    };
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
struct Deltas<'a> {
    index: usize,
    x: &'a PrimitiveArray<Float64Type>,
    y: &'a PrimitiveArray<Float64Type>,
    z: Option<&'a PrimitiveArray<Float64Type>>,
    no_null: bool,
}

impl<'a> Deltas<'a> {
    fn new(
        x: &'a PrimitiveArray<Float64Type>,
        y: &'a PrimitiveArray<Float64Type>,
        z: Option<&'a PrimitiveArray<Float64Type>>,
    ) -> Self {
        let no_null = x.null_count() == 0
            && y.null_count() == 0
            && match z {
                Some(z) => z.null_count() == 0,
                None => true,
            };

        Self {
            index: 0,
            x,
            y,
            z,
            no_null,
        }
    }
    fn is_null(&self, i: usize) -> bool {
        if self.no_null {
            return false;
        }

        self.x.is_null(i)
            || self.y.is_null(i)
            || match self.z {
                Some(z) => z.is_null(i),
                None => false,
            }
    }
}

impl<'a> Iterator for Deltas<'a> {
    type Item = Option<(f64, f64, f64)>;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        self.index += 1;

        if self.is_null(i) {
            return Some(None);
        }

        let x = self.x.value(i);
        let y = self.y.value(i);
        let z = match self.z {
            Some(z) => z.value(i),
            None => 0.0,
        };
        Some(Some((x, y, z)))
    }
}

#[derive(Debug)]
struct Translate {
    deltax: f64,
    deltay: f64,
    deltaz: f64,
}

impl CrsTransform for Translate {
    fn transform_coord(&self, coord: &mut (f64, f64)) -> Result<(), SedonaGeometryError> {
        coord.0 += self.deltax;
        coord.1 += self.deltay;
        Ok(())
    }

    fn transform_coord_3d(
        &self,
        coord: &mut (f64, f64, f64),
    ) -> std::result::Result<(), SedonaGeometryError> {
        coord.0 += self.deltax;
        coord.1 += self.deltay;
        coord.2 += self.deltaz;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::create_array;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_translate_udf: ScalarUDF = st_translate_udf().into();
        assert_eq!(st_translate_udf.name(), "st_translate");
        assert!(st_translate_udf.documentation().is_some());
    }

    #[rstest]
    fn udf_2d(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_translate_udf().into(),
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
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT Z EMPTY"),
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("POINT Z (4 5 6)"),
            ],
            &sedona_type,
        );

        let dx = create_array!(
            Float64,
            [
                Some(1.0),
                None,
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0)
            ]
        );
        let dy = create_array!(
            Float64,
            [
                Some(2.0),
                Some(2.0),
                None,
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(2.0)
            ]
        );

        let expected = create_array(
            &[
                None,
                None,
                None,
                Some("POINT EMPTY"),
                Some("POINT Z EMPTY"),
                Some("POINT (1 3)"),
                Some("POINT (3 5)"),
                Some("POINT Z (5 7 6)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points, dx, dy]).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_3d(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_translate_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT Z EMPTY"),
                Some("POINT (0 1)"),
                Some("POINT Z (4 5 6)"),
            ],
            &sedona_type,
        );

        let dx = create_array!(
            Float64,
            [
                Some(1.0),
                None,
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0)
            ]
        );
        let dy = create_array!(
            Float64,
            [
                Some(2.0),
                Some(2.0),
                None,
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(2.0)
            ]
        );
        let dz = create_array!(
            Float64,
            [
                Some(3.0),
                Some(3.0),
                Some(3.0),
                None,
                Some(3.0),
                Some(3.0),
                Some(3.0),
                Some(3.0)
            ]
        );

        let expected = create_array(
            &[
                None,
                None,
                None,
                None,
                Some("POINT EMPTY"),
                Some("POINT Z EMPTY"),
                Some("POINT (1 3)"),
                Some("POINT Z (5 7 9)"),
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_arrays(vec![points, dx, dy, dz]).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_translate_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester.assert_return_type(sedona_type);

        let result = tester
            .invoke_scalar_scalar_scalar(
                "POINT (0 1)",
                ScalarValue::Float64(Some(1.0)),
                ScalarValue::Float64(Some(2.0)),
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 3)");
    }
}
