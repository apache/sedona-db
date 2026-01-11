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
use datafusion_common::{error::Result, exec_err, DataFusionError};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{
    CoordTrait, GeometryCollectionTrait as _, GeometryTrait, LineStringTrait,
    MultiLineStringTrait as _, MultiPointTrait as _, MultiPolygonTrait as _, PointTrait,
    PolygonTrait as _,
};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::wkb_factory::{
    write_wkb_coord, write_wkb_empty_point, write_wkb_geometrycollection_header,
    write_wkb_linestring_header, write_wkb_multilinestring_header, write_wkb_multipoint_header,
    write_wkb_multipolygon_header, write_wkb_point_header, write_wkb_polygon_header,
    write_wkb_polygon_ring_header, WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{io::Write, sync::Arc};

use crate::{executor::WkbExecutor, st_affine_helpers};

/// ST_Affine() scalar UDF
///
/// Native implementation for affine transformation
pub fn st_affine_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_affine",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STAffine { is_3d: true }),
            Arc::new(STAffine { is_3d: false }),
        ]),
        Volatility::Immutable,
        Some(st_affine_doc()),
    )
}

fn st_affine_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Apply an affine transformation to the given geometry.",
        "ST_Affine (geom: Geometry, a: Double, b: Double, c: Double, d: Double, e: Double, f: Double, g: Double, h: Double, i: Double, xOff: Double, yOff: Double, zOff: Double)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("a", "a component of the affine matrix")
    .with_argument("b", "a component of the affine matrix")
    .with_argument("c", "a component of the affine matrix")
    .with_argument("d", "a component of the affine matrix")
    .with_argument("e", "a component of the affine matrix")
    .with_argument("f", "a component of the affine matrix")
    .with_argument("g", "a component of the affine matrix")
    .with_argument("h", "a component of the affine matrix")
    .with_argument("i", "a component of the affine matrix")
    .with_argument("xOff", "X offset")
    .with_argument("yOff", "Y offset")
    .with_argument("zOff", "Z offset")
    .with_sql_example("SELECT ST_Affine(ST_GeomFromText('POLYGON Z ((1 0 1, 1 1 1, 2 2 2, 1 0 1))'), 1, 2, 4, 1, 1, 2, 3, 2, 5, 4, 8, 3)")
    .build()
}

#[derive(Debug)]
struct STAffine {
    is_3d: bool,
}

impl SedonaScalarKernel for STAffine {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let arg_matchers = if self.is_3d {
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(), // a
                ArgMatcher::is_numeric(), // b
                ArgMatcher::is_numeric(), // c
                ArgMatcher::is_numeric(), // d
                ArgMatcher::is_numeric(), // e
                ArgMatcher::is_numeric(), // f
                ArgMatcher::is_numeric(), // g
                ArgMatcher::is_numeric(), // h
                ArgMatcher::is_numeric(), // i
                ArgMatcher::is_numeric(), // xOff
                ArgMatcher::is_numeric(), // yOff
                ArgMatcher::is_numeric(), // zOff
            ]
        } else {
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(), // a
                ArgMatcher::is_numeric(), // b
                ArgMatcher::is_numeric(), // d
                ArgMatcher::is_numeric(), // e
                ArgMatcher::is_numeric(), // xOff
                ArgMatcher::is_numeric(), // yOff
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
            st_affine_helpers::DAffineIterator::new_3d(&array_args)?
        } else {
            st_affine_helpers::DAffineIterator::new_2d(&array_args)?
        };

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    let mat = affine_iter.next().unwrap();
                    invoke_scalar(&wkb, &mut builder, &mat, &wkb.dim())?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    geom: &impl GeometryTrait<T = f64>,
    writer: &mut impl Write,
    mat: &st_affine_helpers::DAffine,
    dim: &geo_traits::Dimensions,
) -> Result<()> {
    let dims = geom.dim();
    match geom.as_type() {
        geo_traits::GeometryType::Point(pt) => {
            if pt.coord().is_some() {
                write_wkb_point_header(writer, dims)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                write_transformed_coord(writer, pt.coord().unwrap(), mat, dim)?;
            } else {
                write_wkb_empty_point(writer, dims)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            }
        }

        geo_traits::GeometryType::MultiPoint(multi_point) => {
            write_wkb_multipoint_header(writer, dims, multi_point.points().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for pt in multi_point.points() {
                invoke_scalar(&pt, writer, mat, dim)?;
            }
        }

        geo_traits::GeometryType::LineString(ls) => {
            write_wkb_linestring_header(writer, dims, ls.coords().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            write_transformed_coords(writer, ls.coords(), mat, dim)?;
        }

        geo_traits::GeometryType::Polygon(pgn) => {
            let num_rings = pgn.interiors().count() + pgn.exterior().is_some() as usize;
            write_wkb_polygon_header(writer, dims, num_rings)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            if let Some(exterior) = pgn.exterior() {
                write_transformed_ring(writer, exterior, mat, dim)?;
            }

            for interior in pgn.interiors() {
                write_transformed_ring(writer, interior, mat, dim)?;
            }
        }

        geo_traits::GeometryType::MultiLineString(mls) => {
            write_wkb_multilinestring_header(writer, dims, mls.line_strings().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for ls in mls.line_strings() {
                invoke_scalar(&ls, writer, mat, dim)?;
            }
        }

        geo_traits::GeometryType::MultiPolygon(mpgn) => {
            write_wkb_multipolygon_header(writer, dims, mpgn.polygons().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for pgn in mpgn.polygons() {
                invoke_scalar(&pgn, writer, mat, dim)?;
            }
        }

        geo_traits::GeometryType::GeometryCollection(gcn) => {
            write_wkb_geometrycollection_header(writer, dims, gcn.geometries().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for geom in gcn.geometries() {
                invoke_scalar(&geom, writer, mat, dim)?;
            }
        }

        _ => {
            return Err(DataFusionError::Execution(
                "Unsupported geometry type for reversal operation".to_string(),
            ));
        }
    }
    Ok(())
}

fn write_transformed_ring(
    writer: &mut impl Write,
    ring: impl LineStringTrait<T = f64>,
    affine: &st_affine_helpers::DAffine,
    dim: &geo_traits::Dimensions,
) -> Result<()> {
    write_wkb_polygon_ring_header(writer, ring.coords().count())
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    write_transformed_coords(writer, ring.coords(), affine, dim)
}

fn write_transformed_coords<I>(
    writer: &mut impl Write,
    coords: I,
    affine: &st_affine_helpers::DAffine,
    dim: &geo_traits::Dimensions,
) -> Result<()>
where
    I: DoubleEndedIterator,
    I::Item: CoordTrait<T = f64>,
{
    coords
        .into_iter()
        .try_for_each(|coord| write_transformed_coord(writer, coord, affine, dim))
}

fn write_transformed_coord<C>(
    writer: &mut impl Write,
    coord: C,
    affine: &st_affine_helpers::DAffine,
    dim: &geo_traits::Dimensions,
) -> Result<()>
where
    C: CoordTrait<T = f64>,
{
    match dim {
        geo_traits::Dimensions::Xy => {
            let transformed = affine.transform_point2(coord.x(), coord.y());
            write_wkb_coord(writer, transformed)
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        }
        geo_traits::Dimensions::Xym => {
            let transformed = affine.transform_point2(coord.x(), coord.y());
            // Preserve m value
            let m = coord.nth(2).unwrap();
            write_wkb_coord(writer, (transformed.0, transformed.1, m))
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        }
        geo_traits::Dimensions::Xyz => {
            let transformed = affine.transform_point3(coord.x(), coord.y(), coord.nth(2).unwrap());
            write_wkb_coord(writer, transformed)
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        }
        geo_traits::Dimensions::Xyzm => {
            let transformed = affine.transform_point3(coord.x(), coord.y(), coord.nth(2).unwrap());
            // Preserve m value
            let m = coord.nth(3).unwrap();
            write_wkb_coord(writer, (transformed.0, transformed.1, transformed.2, m))
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        }
        geo_traits::Dimensions::Unknown(_) => {
            exec_err!("A geometry with unknown dimension cannot be transformed")
        }
    }
}

#[cfg(test)]
mod tests {
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
        let st_affine_udf: ScalarUDF = st_affine_udf().into();
        assert_eq!(st_affine_udf.name(), "st_affine");
        assert!(st_affine_udf.documentation().is_some());
    }

    #[rstest]
    fn udf_2d(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester_2d = ScalarUdfTester::new(
            st_affine_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester_2d.assert_return_type(WKB_GEOMETRY);

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

        // identity transformation

        #[rustfmt::skip]
        let m_identity = &[
            Some(1.0), Some(0.0),
            Some(0.0), Some(1.0),
            Some(0.0), Some(0.0),
        ];

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

        let result_identity = tester_2d
            .invoke_arrays(prepare_args(points.clone(), m_identity))
            .unwrap();
        assert_array_equal(&result_identity, &expected_identity);

        // scale transformation

        #[rustfmt::skip]
        let m_scale = &[
            Some(10.0), Some(0.0),
            Some(0.0), Some(10.0),
            Some(0.0), Some(0.0),
        ];

        let expected_scale = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (10 20)"),
                Some("POINT M (10 20 3)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_scale = tester_2d
            .invoke_arrays(prepare_args(points.clone(), m_scale))
            .unwrap();
        assert_array_equal(&result_scale, &expected_scale);

        // 2D matrix with 3D input (z/m preserved)
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

        let expected_scale_3d = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT Z (10 20 3)"),
                Some("POINT ZM (10 20 3 4)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_scale_3d = tester_2d
            .invoke_arrays(prepare_args(points_3d, m_scale))
            .unwrap();
        assert_array_equal(&result_scale_3d, &expected_scale_3d);
    }

    #[rstest]
    fn udf_3d(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester_3d = ScalarUdfTester::new(
            st_affine_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester_3d.assert_return_type(WKB_GEOMETRY);

        let points = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT Z (1 2 3)"),
                Some("POINT ZM (1 2 3 4)"),
            ],
            &sedona_type,
        );

        // identity matrix
        #[rustfmt::skip]
        let m_identity = &[
            Some(1.0), Some(0.0), Some(0.0),
            Some(0.0), Some(1.0), Some(0.0),
            Some(0.0), Some(0.0), Some(1.0),
            Some(0.0), Some(0.0), Some(0.0),
        ];

        let expected_identity = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT Z (1 2 3)"),
                Some("POINT ZM (1 2 3 4)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_identity = tester_3d
            .invoke_arrays(prepare_args(points.clone(), m_identity))
            .unwrap();
        assert_array_equal(&result_identity, &expected_identity);

        // scale transformation
        #[rustfmt::skip]
        let m_scale = &[
            Some(10.0), Some(0.0), Some(0.0),
            Some(0.0), Some(10.0), Some(0.0),
            Some(0.0), Some(0.0), Some(10.0),
            Some(0.0), Some(0.0), Some(0.0),
        ];

        let expected_scale = create_array(
            &[
                None,
                Some("POINT Z EMPTY"),
                Some("POINT ZM EMPTY"),
                Some("POINT Z (10 20 30)"),
                Some("POINT ZM (10 20 30 4)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_scale = tester_3d
            .invoke_arrays(prepare_args(points, m_scale))
            .unwrap();
        assert_array_equal(&result_scale, &expected_scale);

        // 3D matrix with 2D input (z translation ignored, m preserved)
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

        #[rustfmt::skip]
        let m_translate = &[
            Some(1.0), Some(0.0), Some(0.0),
            Some(0.0), Some(1.0), Some(0.0),
            Some(0.0), Some(0.0), Some(1.0),
            Some(10.0), Some(20.0), Some(30.0),
        ];

        let expected_translate = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("POINT M EMPTY"),
                Some("POINT (11 22)"),
                Some("POINT M (11 22 3)"),
            ],
            &WKB_GEOMETRY,
        );

        let result_translate = tester_3d
            .invoke_arrays(prepare_args(points_2d, m_translate))
            .unwrap();
        assert_array_equal(&result_translate, &expected_translate);
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
            st_affine_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester.assert_return_type(sedona_type.clone());

        let geom = create_scalar(Some("POINT (1 2)"), &sedona_type);
        let args = vec![
            ColumnarValue::Scalar(geom),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(2.0))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(0.0))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(0.0))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(2.0))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(3.0))),
        ];

        let result = tester.invoke(args).unwrap();
        if let ColumnarValue::Scalar(scalar) = result {
            tester.assert_scalar_result_equals(scalar, "POINT (3 7)");
        } else {
            panic!("Expected scalar result from item CRS affine invoke");
        }
    }
}
