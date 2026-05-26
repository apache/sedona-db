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

use std::io::Write;
use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use arrow_array::Array;
use arrow_schema::DataType;
use datafusion_common::cast::as_float64_array;
use datafusion_common::exec_datafusion_err;
use datafusion_common::Result;
use datafusion_expr::ColumnarValue;
use datafusion_expr::Volatility;
use geo_traits::{
    CoordTrait, Dimensions, GeometryCollectionTrait, GeometryTrait, LineStringTrait,
    MultiLineStringTrait, MultiPolygonTrait, PolygonTrait,
};
use sedona_expr::item_crs::ItemCrsKernel;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::error::SedonaGeometryError;
use sedona_geometry::wkb_factory::{
    write_wkb_coord_trait, write_wkb_geometrycollection_header, write_wkb_linestring_header,
    write_wkb_multilinestring_header, write_wkb_multipolygon_header, write_wkb_polygon_header,
    write_wkb_polygon_ring_header, WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::Wkb;

use crate::executor::WkbExecutor;

/// ST_Segmentize() scalar UDF for geometry
///
/// Native implementation to densify a geometry by adding intermediate points
/// along segments that exceed a maximum length. Uses Euclidean (planar) distance.
/// This matches the behavior of GEOSDensify / PostGIS ST_Segmentize for geometry.
pub fn st_segmentize_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_segmentize",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STSegmentize {
            matcher: ArgMatcher::new(
                vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
                WKB_GEOMETRY,
            ),
        })]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STSegmentize {
    matcher: ArgMatcher,
}

impl SedonaScalarKernel for STSegmentize {
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

        // Get max_segment_length as Float64 array
        let max_segment_length_array = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let max_segment_length_values = as_float64_array(&max_segment_length_array)?;

        let mut idx = 0usize;
        executor.execute_wkb_void(|maybe_wkb| {
            let max_seg_len = if max_segment_length_values.is_null(idx) {
                None
            } else {
                Some(max_segment_length_values.value(idx))
            };
            idx += 1;

            match (maybe_wkb, max_seg_len) {
                (Some(wkb), Some(max_len)) if max_len > 0.0 => {
                    segmentize_wkb(&wkb, max_len, &mut builder)
                        .map_err(|e| exec_datafusion_err!("Segmentize error: {e}"))?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn segmentize_wkb(
    geom: &Wkb,
    max_segment_length: f64,
    writer: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    let dims = geom.dim();
    match geom.as_type() {
        // Points don't need segmentization, copy buffer directly
        geo_traits::GeometryType::Point(_) => writer.write_all(geom.buf())?,
        geo_traits::GeometryType::LineString(ls) => {
            segmentize_linestring_wkb(ls, dims, max_segment_length, writer)?;
        }
        geo_traits::GeometryType::Polygon(pgn) => {
            segmentize_polygon_wkb(pgn, dims, max_segment_length, writer)?;
        }
        geo_traits::GeometryType::MultiPoint(_) => writer.write_all(geom.buf())?,
        geo_traits::GeometryType::MultiLineString(mls) => {
            write_wkb_multilinestring_header(writer, dims, mls.line_strings().count())?;
            for ls in mls.line_strings() {
                segmentize_linestring_wkb(ls, dims, max_segment_length, writer)?;
            }
        }
        geo_traits::GeometryType::MultiPolygon(mpgn) => {
            write_wkb_multipolygon_header(writer, dims, mpgn.polygons().count())?;
            for pgn in mpgn.polygons() {
                segmentize_polygon_wkb(pgn, dims, max_segment_length, writer)?;
            }
        }
        geo_traits::GeometryType::GeometryCollection(gc) => {
            write_wkb_geometrycollection_header(writer, dims, gc.geometries().count())?;
            for child in gc.geometries() {
                segmentize_wkb(child, max_segment_length, writer)?;
            }
        }
        _ => {
            return Err(SedonaGeometryError::Invalid(
                "unknown geometry type".to_string(),
            ));
        }
    }
    Ok(())
}

fn segmentize_linestring_wkb(
    ls: &wkb::reader::LineString,
    dims: Dimensions,
    max_segment_length: f64,
    writer: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    let coords: Vec<_> = ls.coords().collect();
    let output_count = count_segmentized_coords(&coords, max_segment_length);
    write_wkb_linestring_header(writer, dims, output_count)?;
    write_segmentized_coords(writer, dims, &coords, max_segment_length)
}

fn segmentize_polygon_wkb(
    pgn: &wkb::reader::Polygon,
    dims: Dimensions,
    max_segment_length: f64,
    writer: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    let num_rings = pgn.num_interiors() + pgn.exterior().is_some() as usize;
    write_wkb_polygon_header(writer, dims, num_rings)?;

    if let Some(exterior) = pgn.exterior() {
        segmentize_linearring_wkb(exterior, dims, max_segment_length, writer)?;
    }

    for interior in pgn.interiors() {
        segmentize_linearring_wkb(interior, dims, max_segment_length, writer)?;
    }
    Ok(())
}

fn segmentize_linearring_wkb(
    ring: &wkb::reader::LinearRing,
    dims: Dimensions,
    max_segment_length: f64,
    writer: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    let coords: Vec<_> = ring.coords().collect();
    let output_count = count_segmentized_coords(&coords, max_segment_length);
    write_wkb_polygon_ring_header(writer, output_count)?;
    write_segmentized_coords(writer, dims, &coords, max_segment_length)
}

/// Count the number of coordinates that will be output after segmentization
fn count_segmentized_coords<C: CoordTrait<T = f64>>(
    coords: &[C],
    max_segment_length: f64,
) -> usize {
    if coords.is_empty() {
        return 0;
    }

    let mut count = 1usize; // First coordinate

    for i in 1..coords.len() {
        let num_segments = calc_num_segments(&coords[i - 1], &coords[i], max_segment_length);
        count += num_segments;
    }

    count
}

/// Write segmentized coordinates to the output
fn write_segmentized_coords<C: CoordTrait<T = f64>>(
    writer: &mut impl Write,
    dims: Dimensions,
    coords: &[C],
    max_segment_length: f64,
) -> Result<(), SedonaGeometryError> {
    if coords.is_empty() {
        return Ok(());
    }

    // Write first coordinate
    write_wkb_coord_trait(writer, &coords[0])?;

    // Write remaining segments
    for i in 1..coords.len() {
        write_interpolated_coords(writer, dims, &coords[i - 1], &coords[i], max_segment_length)?;
    }

    Ok(())
}

/// Calculate the number of segments needed for a single edge
fn calc_num_segments<C: CoordTrait<T = f64>>(c1: &C, c2: &C, max_segment_length: f64) -> usize {
    let dx = c2.x() - c1.x();
    let dy = c2.y() - c1.y();
    let distance = (dx * dx + dy * dy).sqrt();

    if distance <= max_segment_length {
        1
    } else {
        (distance / max_segment_length).ceil() as usize
    }
}

/// Write interpolated coordinates between two points (excluding start, including end)
fn write_interpolated_coords<C: CoordTrait<T = f64>>(
    writer: &mut impl Write,
    dims: Dimensions,
    c1: &C,
    c2: &C,
    max_segment_length: f64,
) -> Result<(), SedonaGeometryError> {
    let num_segments = calc_num_segments(c1, c2, max_segment_length);

    if num_segments == 1 {
        // No subdivision needed, just write end coordinate
        write_wkb_coord_trait(writer, c2)?;
    } else {
        // Interpolate intermediate points
        let x1 = c1.x();
        let y1 = c1.y();
        let x2 = c2.x();
        let y2 = c2.y();

        // Get Z and M values if present
        let (z1, z2) = match dims {
            Dimensions::Xyz | Dimensions::Xyzm => {
                (Some(c1.nth_or_panic(2)), Some(c2.nth_or_panic(2)))
            }
            _ => (None, None),
        };

        let (m1, m2) = match dims {
            Dimensions::Xym => (Some(c1.nth_or_panic(2)), Some(c2.nth_or_panic(2))),
            Dimensions::Xyzm => (Some(c1.nth_or_panic(3)), Some(c2.nth_or_panic(3))),
            _ => (None, None),
        };

        for i in 1..=num_segments {
            let t = i as f64 / num_segments as f64;
            let x = x1 + t * (x2 - x1);
            let y = y1 + t * (y2 - y1);

            match dims {
                Dimensions::Xy => {
                    writer.write_all(&x.to_le_bytes())?;
                    writer.write_all(&y.to_le_bytes())?;
                }
                Dimensions::Xyz => {
                    let z = z1.unwrap() + t * (z2.unwrap() - z1.unwrap());
                    writer.write_all(&x.to_le_bytes())?;
                    writer.write_all(&y.to_le_bytes())?;
                    writer.write_all(&z.to_le_bytes())?;
                }
                Dimensions::Xym => {
                    let m = m1.unwrap() + t * (m2.unwrap() - m1.unwrap());
                    writer.write_all(&x.to_le_bytes())?;
                    writer.write_all(&y.to_le_bytes())?;
                    writer.write_all(&m.to_le_bytes())?;
                }
                Dimensions::Xyzm => {
                    let z = z1.unwrap() + t * (z2.unwrap() - z1.unwrap());
                    let m = m1.unwrap() + t * (m2.unwrap() - m1.unwrap());
                    writer.write_all(&x.to_le_bytes())?;
                    writer.write_all(&y.to_le_bytes())?;
                    writer.write_all(&z.to_le_bytes())?;
                    writer.write_all(&m.to_le_bytes())?;
                }
                _ => {
                    return Err(SedonaGeometryError::Invalid(
                        "Unsupported dimension for segmentize".to_string(),
                    ));
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    fn prepare_args(
        geom_array: Arc<dyn arrow_array::Array>,
        max_segment_lengths: &[Option<f64>],
    ) -> Vec<Arc<dyn arrow_array::Array>> {
        let n = geom_array.len();
        let values: Vec<Option<f64>> = max_segment_lengths
            .iter()
            .cycle()
            .take(n)
            .copied()
            .collect();
        let max_segment_length_array = arrow_array::Float64Array::from(values);
        vec![geom_array, Arc::new(max_segment_length_array)]
    }

    #[test]
    fn udf_metadata() {
        let udf: datafusion_expr::ScalarUDF = st_segmentize_udf().into();
        assert_eq!(udf.name(), "st_segmentize");
    }

    #[rstest]
    fn test_null_handling(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let geoms = create_array(&[None, Some("POINT (0 0)"), None], &sedona_type);
        let result = tester
            .invoke_arrays(prepare_args(geoms, &[Some(1.0), None, None]))
            .unwrap();

        let expected = create_array(&[None, None, None], &WKB_GEOMETRY);
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn test_point_no_change(
        #[values(
            "POINT EMPTY",
            "POINT Z EMPTY",
            "POINT M EMPTY",
            "POINT ZM EMPTY",
            "POINT (0 1)",
            "POINT Z (0 1 100)",
            "POINT M (0 1 100)",
            "POINT ZM (0 1 100 200)"
        )]
        wkt: &str,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester.invoke_scalar_scalar(wkt, 1e9).unwrap();
        tester.assert_scalar_result_equals(result, wkt);
    }

    #[rstest]
    fn test_linestring_no_change(
        #[values(
            "LINESTRING EMPTY",
            "LINESTRING Z EMPTY",
            "LINESTRING M EMPTY",
            "LINESTRING ZM EMPTY",
            "LINESTRING (0 0, 1 0)",
            "LINESTRING Z (0 0 100, 1 0 200)",
            "LINESTRING M (0 0 100, 1 0 200)",
            "LINESTRING ZM (0 0 100 10, 1 0 200 20)"
        )]
        wkt: &str,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester.invoke_scalar_scalar(wkt, 1e9).unwrap();
        tester.assert_scalar_result_equals(result, wkt);
    }

    #[rstest]
    fn test_polygon_no_change(
        #[values(
            "POLYGON EMPTY",
            "POLYGON Z EMPTY",
            "POLYGON M EMPTY",
            "POLYGON ZM EMPTY",
            "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
            "POLYGON Z ((0 0 100, 1 0 100, 1 1 100, 0 1 100, 0 0 100))",
            "POLYGON M ((0 0 100, 1 0 100, 1 1 100, 0 1 100, 0 0 100))",
            "POLYGON ZM ((0 0 100 10, 1 0 100 10, 1 1 100 10, 0 1 100 10, 0 0 100 10))"
        )]
        wkt: &str,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester.invoke_scalar_scalar(wkt, 1e9).unwrap();
        tester.assert_scalar_result_equals(result, wkt);
    }

    #[rstest]
    fn test_multipoint_no_change(
        #[values(
            "MULTIPOINT EMPTY",
            "MULTIPOINT Z EMPTY",
            "MULTIPOINT M EMPTY",
            "MULTIPOINT ZM EMPTY",
            "MULTIPOINT ((0 0), (1 1))",
            "MULTIPOINT Z ((0 0 100), (1 1 200))",
            "MULTIPOINT M ((0 0 100), (1 1 200))",
            "MULTIPOINT ZM ((0 0 100 10), (1 1 200 20))"
        )]
        wkt: &str,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester.invoke_scalar_scalar(wkt, 1e9).unwrap();
        tester.assert_scalar_result_equals(result, wkt);
    }

    #[rstest]
    fn test_multilinestring_no_change(
        #[values(
            "MULTILINESTRING EMPTY",
            "MULTILINESTRING Z EMPTY",
            "MULTILINESTRING M EMPTY",
            "MULTILINESTRING ZM EMPTY",
            "MULTILINESTRING ((0 0, 1 0), (2 2, 3 3))",
            "MULTILINESTRING Z ((0 0 100, 1 0 200), (2 2 100, 3 3 200))",
            "MULTILINESTRING M ((0 0 100, 1 0 200), (2 2 100, 3 3 200))",
            "MULTILINESTRING ZM ((0 0 100 10, 1 0 200 20), (2 2 100 10, 3 3 200 20))"
        )]
        wkt: &str,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester.invoke_scalar_scalar(wkt, 1e9).unwrap();
        tester.assert_scalar_result_equals(result, wkt);
    }

    #[rstest]
    fn test_multipolygon_no_change(
        #[values(
            "MULTIPOLYGON EMPTY",
            "MULTIPOLYGON Z EMPTY",
            "MULTIPOLYGON M EMPTY",
            "MULTIPOLYGON ZM EMPTY",
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "MULTIPOLYGON Z (((0 0 100, 1 0 100, 1 1 100, 0 1 100, 0 0 100)))",
            "MULTIPOLYGON M (((0 0 100, 1 0 100, 1 1 100, 0 1 100, 0 0 100)))",
            "MULTIPOLYGON ZM (((0 0 100 10, 1 0 100 10, 1 1 100 10, 0 1 100 10, 0 0 100 10)))"
        )]
        wkt: &str,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester.invoke_scalar_scalar(wkt, 1e9).unwrap();
        tester.assert_scalar_result_equals(result, wkt);
    }

    #[rstest]
    fn test_geometrycollection_no_change(
        #[values(
            "GEOMETRYCOLLECTION EMPTY",
            "GEOMETRYCOLLECTION Z EMPTY",
            "GEOMETRYCOLLECTION M EMPTY",
            "GEOMETRYCOLLECTION ZM EMPTY",
            "GEOMETRYCOLLECTION (POINT (0 1))",
            "GEOMETRYCOLLECTION Z (POINT Z (0 1 100))",
            "GEOMETRYCOLLECTION M (POINT M (0 1 100))",
            "GEOMETRYCOLLECTION ZM (POINT ZM (0 1 100 10))"
        )]
        wkt: &str,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester.invoke_scalar_scalar(wkt, 1e9).unwrap();
        tester.assert_scalar_result_equals(result, wkt);
    }

    #[rstest]
    fn test_linestring_split(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // A line from (0,0) to (0,2) has length 2
        // With max_segment_length of 1.1, it should be split into 2 segments
        let geoms = create_array(&[Some("LINESTRING (0 0, 0 2)")], &sedona_type);
        let result = tester
            .invoke_arrays(prepare_args(geoms, &[Some(1.1)]))
            .unwrap();

        let expected = create_array(&[Some("LINESTRING (0 0, 0 1, 0 2)")], &WKB_GEOMETRY);
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn test_linestring_split_multiple(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // A line from (0,0) to (0,4) has length 4
        // With max_segment_length of 1.1, it should be split into 4 segments
        let geoms = create_array(&[Some("LINESTRING (0 0, 0 4)")], &sedona_type);
        let result = tester
            .invoke_arrays(prepare_args(geoms, &[Some(1.1)]))
            .unwrap();

        let expected = create_array(
            &[Some("LINESTRING (0 0, 0 1, 0 2, 0 3, 0 4)")],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn test_linestring_z_interpolation(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // Z values should be linearly interpolated
        let geoms = create_array(&[Some("LINESTRING Z (0 0 100, 0 2 200)")], &sedona_type);
        let result = tester
            .invoke_arrays(prepare_args(geoms, &[Some(1.1)]))
            .unwrap();

        let expected = create_array(
            &[Some("LINESTRING Z (0 0 100, 0 1 150, 0 2 200)")],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn test_linestring_m_interpolation(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // M values should be linearly interpolated
        let geoms = create_array(&[Some("LINESTRING M (0 0 0, 0 2 100)")], &sedona_type);
        let result = tester
            .invoke_arrays(prepare_args(geoms, &[Some(1.1)]))
            .unwrap();

        let expected = create_array(
            &[Some("LINESTRING M (0 0 0, 0 1 50, 0 2 100)")],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn test_linestring_zm_interpolation(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // Both Z and M values should be linearly interpolated
        let geoms = create_array(
            &[Some("LINESTRING ZM (0 0 100 0, 0 2 200 100)")],
            &sedona_type,
        );
        let result = tester
            .invoke_arrays(prepare_args(geoms, &[Some(1.1)]))
            .unwrap();

        let expected = create_array(
            &[Some("LINESTRING ZM (0 0 100 0, 0 1 150 50, 0 2 200 100)")],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn test_polygon_split(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // 2x2 square, max segment 1.1 -> each edge split into 2
        let geoms = create_array(&[Some("POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))")], &sedona_type);
        let result = tester
            .invoke_arrays(prepare_args(geoms, &[Some(1.1)]))
            .unwrap();

        let expected = create_array(
            &[Some(
                "POLYGON ((0 0, 0 1, 0 2, 1 2, 2 2, 2 1, 2 0, 1 0, 0 0))",
            )],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn test_item_crs_preserved(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_segmentize_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // Item CRS should be preserved in the output type
        tester.assert_return_type(WKB_GEOMETRY_ITEM_CRS.clone());
    }
}
