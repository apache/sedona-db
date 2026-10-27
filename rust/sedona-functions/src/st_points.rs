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
use arrow_array::builder::{BinaryBuilder, UInt64Builder};
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{
    CoordTrait, Dimensions, GeometryCollectionTrait, GeometryTrait, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use sedona_common::sedona_internal_err;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{
    error::SedonaGeometryError,
    wkb_factory::{
        WKB_MIN_PROBABLE_BYTES, write_wkb_coord, write_wkb_multipoint_header, write_wkb_point_header,
    },
    wkb_header::any_part_dimensions,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{io::Write, sync::Arc};

use crate::executor::WkbExecutor;

/// ST_Points() scalar UDF
///
/// Native implementation to get all the points of a geometry as MULTIPOINT
pub fn st_points_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_points",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STPoints {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
            }),
            Arc::new(STPoints {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY),
            }),
        ]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STPoints {
    matcher: ArgMatcher,
}

impl SedonaScalarKernel for STPoints {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
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

        executor.execute_wkb_void(|maybe_wkb| {
            if let Some(wkb) = maybe_wkb {
                // We need to know the number of points before actually writing the points.
                let n_points = count_wkb_points_recursively(wkb);
                // Union of input header and every part's dimensions so mixed Z/M
                // collections write a consistent MultiPoint (missing ordinates as NaN).
                let dims = output_dimensions(wkb)?;

                if write_wkb_multipoint_header(&mut builder, dims, n_points).is_err() {
                    return sedona_internal_err!("Failed to write WKB point header");
                };

                if write_wkb_points_recursively(&mut builder, wkb, dims).is_err() {
                    return sedona_internal_err!("Failed to write WKB point header");
                };

                builder.append_value([]);
            } else {
                builder.append_null();
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// ST_NPoints() scalar UDF
///
/// Native implementation to count all the points of a geometry
pub fn st_npoints_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_npoints",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STNPoints)]),
        Volatility::Immutable,
    )
    .with_aliases(vec!["st_numpoints".to_string()])
}

#[derive(Debug)]
struct STNPoints;

impl SedonaScalarKernel for STNPoints {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::UInt64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = UInt64Builder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_wkb| {
            if let Some(wkb) = maybe_wkb {
                builder.append_value(count_wkb_points_recursively(wkb) as u64);
            } else {
                builder.append_null();
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn count_wkb_points_recursively<'a>(wkb: &'a wkb::reader::Wkb<'a>) -> usize {
    match wkb.as_type() {
        geo_traits::GeometryType::Point(point) => {
            if point.is_empty() {
                0
            } else {
                1
            }
        }
        geo_traits::GeometryType::LineString(line_string) => line_string.num_coords(),
        geo_traits::GeometryType::Polygon(polygon) => {
            let mut n = match polygon.exterior() {
                Some(ring) => ring.num_coords(),
                None => 0,
            };
            n += polygon.interiors().map(|r| r.num_coords()).sum::<usize>();

            n
        }
        geo_traits::GeometryType::MultiPoint(multi_point) => {
            multi_point.points().filter(|p| !p.is_empty()).count()
        }
        geo_traits::GeometryType::MultiLineString(multi_line_string) => multi_line_string
            .line_strings()
            .map(|l| l.num_coords())
            .sum(),
        geo_traits::GeometryType::MultiPolygon(multi_polygon) => {
            let mut n = 0;
            for polygon in multi_polygon.polygons() {
                n += match polygon.exterior() {
                    Some(ring) => ring.num_coords(),
                    None => 0,
                };
                n += polygon.interiors().map(|r| r.num_coords()).sum::<usize>();
            }
            n
        }
        geo_traits::GeometryType::GeometryCollection(geometry_collection) => {
            let mut n = 0;
            for geometry in geometry_collection.geometries() {
                n += count_wkb_points_recursively(geometry);
            }
            n
        }
        _ => 0,
    }
}

/// Union of the geometry's own dimensions and every nested part's dimensions.
fn output_dimensions(wkb: &wkb::reader::Wkb<'_>) -> datafusion_common::Result<Dimensions> {
    let has_z = matches!(wkb.dim(), Dimensions::Xyz | Dimensions::Xyzm)
        || any_part_dimensions(wkb.buf(), |d| matches!(d, Dimensions::Xyz | Dimensions::Xyzm))
            .map_err(|e| datafusion_common::DataFusionError::External(Box::new(e)))?;
    let has_m = matches!(wkb.dim(), Dimensions::Xym | Dimensions::Xyzm)
        || any_part_dimensions(wkb.buf(), |d| matches!(d, Dimensions::Xym | Dimensions::Xyzm))
            .map_err(|e| datafusion_common::DataFusionError::External(Box::new(e)))?;
    Ok(match (has_z, has_m) {
        (false, false) => Dimensions::Xy,
        (true, false) => Dimensions::Xyz,
        (false, true) => Dimensions::Xym,
        (true, true) => Dimensions::Xyzm,
    })
}

/// Extract Z/M for `target`, filling a missing ordinate with NaN (GEOS / Shapely style).
fn write_wkb_point_from_coord(
    buf: &mut impl Write,
    coord: impl CoordTrait<T = f64>,
    target: Dimensions,
) -> Result<(), SedonaGeometryError> {
    write_wkb_point_header(buf, target)?;
    let src = coord.dim();
    let x = coord.x();
    let y = coord.y();
    let z = match src {
        Dimensions::Xyz | Dimensions::Xyzm => Some(coord.nth_or_panic(2)),
        _ => None,
    };
    let m = match src {
        Dimensions::Xym => Some(coord.nth_or_panic(2)),
        Dimensions::Xyzm => Some(coord.nth_or_panic(3)),
        _ => None,
    };
    match target {
        Dimensions::Xy => write_wkb_coord(buf, (x, y)),
        Dimensions::Xyz => write_wkb_coord(buf, (x, y, z.unwrap_or(f64::NAN))),
        Dimensions::Xym => write_wkb_coord(buf, (x, y, m.unwrap_or(f64::NAN))),
        Dimensions::Xyzm => write_wkb_coord(
            buf,
            (x, y, z.unwrap_or(f64::NAN), m.unwrap_or(f64::NAN)),
        ),
        Dimensions::Unknown(_) => Err(SedonaGeometryError::Invalid(
            "Unsupported number of dimensions".to_string(),
        )),
    }
}

fn write_wkb_points_from_coords(
    buf: &mut impl Write,
    coords: impl Iterator<Item = impl CoordTrait<T = f64>>,
    target: Dimensions,
) -> Result<(), SedonaGeometryError> {
    for coord in coords {
        write_wkb_point_from_coord(buf, coord, target)?;
    }
    Ok(())
}

fn write_wkb_points_recursively<'a>(
    buf: &mut impl Write,
    wkb: &'a wkb::reader::Wkb<'a>,
    target: Dimensions,
) -> Result<(), SedonaGeometryError> {
    match wkb.as_type() {
        geo_traits::GeometryType::Point(point) => {
            if let Some(coord) = point.coord() {
                write_wkb_point_from_coord(buf, coord, target)?
            }
        }
        geo_traits::GeometryType::LineString(line_string) => {
            write_wkb_points_from_coords(buf, line_string.coords(), target)?;
        }
        geo_traits::GeometryType::Polygon(polygon) => {
            if let Some(ring) = polygon.exterior() {
                write_wkb_points_from_coords(buf, ring.coords(), target)?
            }
            for ring in polygon.interiors() {
                write_wkb_points_from_coords(buf, ring.coords(), target)?;
            }
        }
        geo_traits::GeometryType::MultiPoint(multi_point) => {
            for point in multi_point.points() {
                if let Some(coord) = point.coord() {
                    write_wkb_point_from_coord(buf, coord, target)?
                }
            }
        }
        geo_traits::GeometryType::MultiLineString(multi_line_string) => {
            for line_string in multi_line_string.line_strings() {
                write_wkb_points_from_coords(buf, line_string.coords(), target)?;
            }
        }
        geo_traits::GeometryType::MultiPolygon(multi_polygon) => {
            for polygon in multi_polygon.polygons() {
                if let Some(ring) = polygon.exterior() {
                    write_wkb_points_from_coords(buf, ring.coords(), target)?
                }
                for ring in polygon.interiors() {
                    write_wkb_points_from_coords(buf, ring.coords(), target)?;
                }
            }
        }
        geo_traits::GeometryType::GeometryCollection(geometry_collection) => {
            for geometry in geometry_collection.geometries() {
                write_wkb_points_recursively(buf, geometry, target)?;
            }
        }
        _ => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY_ITEM_CRS};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_points_udf: ScalarUDF = st_points_udf().into();
        assert_eq!(st_points_udf.name(), "st_points");
        assert!(st_points_udf.documentation().is_none());

        let st_npoints_udf: ScalarUDF = st_npoints_udf().into();
        assert_eq!(st_npoints_udf.name(), "st_npoints");
        assert!(st_npoints_udf.documentation().is_none());
    }

    #[test]
    fn npoints_aliases() {
        let udf: ScalarUDF = st_npoints_udf().into();
        assert!(udf.aliases().contains(&"st_numpoints".to_string()));
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_GEOGRAPHY)] sedona_type: SedonaType) {
        use arrow_array::UInt64Array;

        let tester_points = ScalarUdfTester::new(st_points_udf().into(), vec![sedona_type.clone()]);
        let tester_npoints =
            ScalarUdfTester::new(st_npoints_udf().into(), vec![sedona_type.clone()]);

        let input = create_array(
            &[
                // 2d
                Some("POINT (1 2)"),
                Some("LINESTRING (1 2, 3 4, 5 6)"),
                Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"),
                Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (1 1, 3 1, 1 3, 1 1))"),
                Some("MULTIPOINT (1 2, 3 4, 5 6, 7 8)"),
                Some("MULTILINESTRING ((1 2, 3 4), EMPTY, (5 6, 7 8))"),
                Some(
                    "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)), EMPTY, ((0 0, 5 0, 0 5, 0 0), (1 1, 3 1, 1 3, 1 1)))",
                ),
                Some("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING EMPTY, LINESTRING (3 4, 5 6))"),
                // 3d and 4d
                Some("LINESTRING Z (1 2 3, 4 5 6, 7 8 9)"),
                Some("LINESTRING M (1 2 3, 4 5 6, 7 8 9)"),
                Some("LINESTRING ZM (1 2 3 4, 5 6 7 8, 9 0 1 2)"),
                // empty
                Some("POINT EMPTY"),
                Some("LINESTRING EMPTY"),
                Some("POLYGON EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTILINESTRING EMPTY"),
                Some("MULTIPOLYGON EMPTY"),
                Some("GEOMETRYCOLLECTION EMPTY"),
                // null
                None,
            ],
            &sedona_type,
        );

        let expected_points = create_array(
            &[
                Some("MULTIPOINT (1 2)"),
                Some("MULTIPOINT (1 2, 3 4, 5 6)"),
                Some("MULTIPOINT (0 0, 10 0, 10 10, 0 10, 0 0)"),
                Some("MULTIPOINT (0 0, 10 0, 10 10, 0 10, 0 0, 1 1, 3 1, 1 3, 1 1)"),
                Some("MULTIPOINT (1 2, 3 4, 5 6, 7 8)"),
                Some("MULTIPOINT (1 2, 3 4, 5 6, 7 8)"),
                Some(
                    "MULTIPOINT (0 0, 10 0, 10 10, 0 10, 0 0, 0 0, 5 0, 0 5, 0 0, 1 1, 3 1, 1 3, 1 1)",
                ),
                Some("MULTIPOINT (1 2, 3 4, 5 6)"),
                // 3d and 4d
                Some("MULTIPOINT Z (1 2 3, 4 5 6, 7 8 9)"),
                Some("MULTIPOINT M (1 2 3, 4 5 6, 7 8 9)"),
                Some("MULTIPOINT ZM (1 2 3 4, 5 6 7 8, 9 0 1 2)"),
                // empty returns empty
                Some("MULTIPOINT EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTIPOINT EMPTY"),
                // null
                None,
            ],
            &sedona_type.clone(),
        );

        let result_points = tester_points.invoke_array(input.clone()).unwrap();
        assert_array_equal(&result_points, &expected_points);

        let expected_npoints: Arc<dyn arrow_array::Array> = Arc::new(UInt64Array::from(vec![
            Some(1),
            Some(3),
            Some(5),
            Some(9),
            Some(4),
            Some(4),
            Some(13),
            Some(3),
            // 3d and 4d
            Some(3),
            Some(3),
            Some(3),
            // empty returns 0
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            Some(0),
            // null
            None,
        ]));

        let result_points = tester_npoints.invoke_array(input.clone()).unwrap();
        assert_array_equal(&result_points, &expected_npoints);
    }

    #[rstest]
    fn udf_invoke_item_crs(
        #[values(WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_points_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type);

        let result = tester.invoke_scalar("LINESTRING (1 2, 3 4, 5 6)").unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT (1 2, 3 4, 5 6)");
    }

    #[test]
    fn mixed_dimensions_from_iso_wkb() {
        use datafusion_common::ScalarValue;
        use geo_traits::{GeometryTrait, MultiPointTrait, PointTrait};
        use wkb::reader::read_wkb;

        let tester = ScalarUdfTester::new(st_points_udf().into(), vec![WKB_GEOMETRY]);

        // ISO WKB for GEOMETRYCOLLECTION M (POINT (1 2), POINT M (3 4 5)), as in #1513.
        // Before the fix this errored with "Invalid buffer length for MultiPoint".
        let wkb: Vec<u8> = vec![
            0x01, 0xD7, 0x07, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, // GC M, 2 geoms
            0x01, 0x01, 0x00, 0x00, 0x00, // Point XY
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // 2.0
            0x01, 0xD1, 0x07, 0x00, 0x00, // Point M
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // 3.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, // 4.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x40, // 5.0
        ];
        assert_eq!(wkb.len(), 59);
        let result = tester
            .invoke_scalar(ScalarValue::Binary(Some(wkb)))
            .unwrap();
        let ScalarValue::Binary(Some(out)) = result else {
            panic!("expected binary geometry, got {result:?}");
        };
        let out_wkb = read_wkb(&out).expect("ST_Points output must be readable WKB");
        assert_eq!(out_wkb.dim(), Dimensions::Xym);
        let geo_traits::GeometryType::MultiPoint(mp) = out_wkb.as_type() else {
            panic!("expected MultiPoint");
        };
        assert_eq!(mp.num_points(), 2);
        let p0 = mp.point(0).unwrap().coord().unwrap();
        assert_eq!((p0.x(), p0.y()), (1.0, 2.0));
        assert!(p0.nth_or_panic(2).is_nan());
        let p1 = mp.point(1).unwrap().coord().unwrap();
        assert_eq!((p1.x(), p1.y(), p1.nth_or_panic(2)), (3.0, 4.0, 5.0));
    }

    #[test]
    fn mixed_dimensions_from_wkt_preserves_m() {
        use datafusion_common::ScalarValue;
        use geo_traits::{GeometryTrait, MultiPointTrait, PointTrait};
        use wkb::reader::read_wkb;

        let tester = ScalarUdfTester::new(st_points_udf().into(), vec![WKB_GEOMETRY]);
        let result = tester
            .invoke_scalar("GEOMETRYCOLLECTION (POINT (1 2), POINT M (3 4 5))")
            .unwrap();
        let ScalarValue::Binary(Some(out)) = result else {
            panic!("expected binary geometry, got {result:?}");
        };
        let out_wkb = read_wkb(&out).expect("ST_Points output must be readable WKB");
        // Regardless of how WKT encodes the collection header, the MultiPoint must
        // carry M so the second point's measure is not dropped.
        assert_eq!(out_wkb.dim(), Dimensions::Xym);
        let geo_traits::GeometryType::MultiPoint(mp) = out_wkb.as_type() else {
            panic!("expected MultiPoint");
        };
        assert_eq!(mp.num_points(), 2);
        let p1 = mp.point(1).unwrap().coord().unwrap();
        assert_eq!((p1.x(), p1.y(), p1.nth_or_panic(2)), (3.0, 4.0, 5.0));
    }
}
