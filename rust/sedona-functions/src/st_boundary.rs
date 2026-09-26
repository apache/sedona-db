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

use std::{io::Write, sync::Arc};

use arrow_array::builder::BinaryBuilder;
use datafusion_common::{error::Result, exec_datafusion_err};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{
    CoordTrait, GeometryCollectionTrait, GeometryTrait, LineStringTrait, MultiLineStringTrait,
    MultiPolygonTrait, PolygonTrait,
};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{
    error::SedonaGeometryError,
    wkb_factory::{
        write_wkb_geometrycollection_header, write_wkb_linestring_header,
        write_wkb_multilinestring_header, write_wkb_multipoint_header, write_wkb_point_header,
        WKB_MIN_PROBABLE_BYTES,
    },
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::{
    reader::{Coord, LinearRing, Wkb},
    Endianness,
};

use crate::executor::WkbExecutor;

/// ST_Boundary() scalar UDF implementation using geo-traits over WKB
pub fn st_boundary_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_boundary",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STBoundary {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
            }),
            Arc::new(STBoundary {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY),
            }),
        ]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STBoundary {
    matcher: ArgMatcher,
}

impl SedonaScalarKernel for STBoundary {
    fn return_type(&self, args: &[SedonaType]) -> datafusion_common::Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> datafusion_common::Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    invoke_scalar(wkb, &mut builder)
                        .map_err(|e| exec_datafusion_err!("ST_Boundary error: {e}"))?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geom: &Wkb, writer: &mut impl Write) -> Result<(), SedonaGeometryError> {
    write_boundary(geom, boundary_shape(geom), writer)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryShape {
    EmptyCollection,
    Point,
    MultiPoint(usize),
    Line,
    MultiLine(usize),
    Collection {
        empty_collections: usize,
        points: usize,
        lines: usize,
    },
}

impl BoundaryShape {
    fn num_points(self) -> usize {
        match self {
            Self::Point => 1,
            Self::MultiPoint(count) | Self::Collection { points: count, .. } => count,
            _ => 0,
        }
    }

    fn num_lines(self) -> usize {
        match self {
            Self::Line => 1,
            Self::MultiLine(count) | Self::Collection { lines: count, .. } => count,
            _ => 0,
        }
    }
}

fn boundary_shape(geom: &Wkb) -> BoundaryShape {
    match geom.as_type() {
        geo_traits::GeometryType::Point(_) | geo_traits::GeometryType::MultiPoint(_) => {
            BoundaryShape::EmptyCollection
        }
        geo_traits::GeometryType::LineString(line) => BoundaryShape::MultiPoint(
            usize::from(line.num_coords() > 1 && !coords_equal(line, 0, line.num_coords() - 1)) * 2,
        ),
        geo_traits::GeometryType::MultiLineString(lines) => {
            BoundaryShape::MultiPoint(multiline_boundary_count(lines))
        }
        geo_traits::GeometryType::Polygon(polygon) => match polygon_ring_count(polygon) {
            1 => BoundaryShape::Line,
            count => BoundaryShape::MultiLine(count),
        },
        geo_traits::GeometryType::MultiPolygon(polygons) => {
            BoundaryShape::MultiLine(polygons.polygons().map(polygon_ring_count).sum())
        }
        geo_traits::GeometryType::GeometryCollection(collection) => {
            geometry_collection_boundary_shape(collection)
        }
        _ => unreachable!("WKB only supports simple feature geometry types"),
    }
}

fn geometry_collection_boundary_shape(
    collection: &wkb::reader::GeometryCollection<'_>,
) -> BoundaryShape {
    let mut empty_collections = 0;
    let mut points = 0;
    let mut lines = 0;

    for geom in collection.geometries() {
        let child_shape = boundary_shape(geom);
        if child_shape == BoundaryShape::EmptyCollection {
            empty_collections += 1;
        } else {
            points += child_shape.num_points();
            lines += child_shape.num_lines();
        }
    }

    let component_count = empty_collections + usize::from(points > 0) + usize::from(lines > 0);
    match component_count {
        0 => BoundaryShape::EmptyCollection,
        1 if empty_collections == 1 => BoundaryShape::EmptyCollection,
        1 if points == 1 => BoundaryShape::Point,
        1 if points > 1 => BoundaryShape::MultiPoint(points),
        1 if lines == 1 => BoundaryShape::Line,
        1 => BoundaryShape::MultiLine(lines),
        _ => BoundaryShape::Collection {
            empty_collections,
            points,
            lines,
        },
    }
}

fn multiline_boundary_count(lines: &wkb::reader::MultiLineString<'_>) -> usize {
    multiline_boundary_coords(lines).count()
}

fn multiline_boundary_coords<'a>(
    lines: &'a wkb::reader::MultiLineString<'a>,
) -> impl Iterator<Item = Coord<'a>> + 'a {
    endpoints(lines).enumerate().filter_map(|(index, coord)| {
        let key = coord_key(&coord);
        if endpoints(lines)
            .take(index)
            .any(|candidate| coord_key(&candidate) == key)
        {
            return None;
        }

        let occurrences = endpoints(lines)
            .filter(|candidate| coord_key(candidate) == key)
            .count();
        (occurrences % 2 == 1).then_some(coord)
    })
}

fn endpoints<'a>(
    lines: &'a wkb::reader::MultiLineString<'a>,
) -> impl Iterator<Item = Coord<'a>> + 'a {
    lines.line_strings().flat_map(|line| {
        if line.num_coords() == 0 {
            [None, None]
        } else {
            [line.coord(0), line.coord(line.num_coords() - 1)]
        }
        .into_iter()
        .flatten()
    })
}

fn coords_equal(line: &wkb::reader::LineString<'_>, lhs: usize, rhs: usize) -> bool {
    coord_key(&line.coord(lhs).unwrap()) == coord_key(&line.coord(rhs).unwrap())
}

fn coord_key(coord: &impl CoordTrait<T = f64>) -> (u64, u64) {
    // Normalize signed zero because -0 and +0 are the same topological position.
    let x = if coord.x() == 0.0 { 0.0 } else { coord.x() };
    let y = if coord.y() == 0.0 { 0.0 } else { coord.y() };
    (x.to_bits(), y.to_bits())
}

fn polygon_ring_count(polygon: &wkb::reader::Polygon<'_>) -> usize {
    polygon.num_interiors() + usize::from(polygon.exterior().is_some())
}

fn write_boundary(
    geom: &Wkb,
    shape: BoundaryShape,
    writer: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    match shape {
        BoundaryShape::EmptyCollection => {
            write_wkb_geometrycollection_header(writer, geom.dim(), 0)
        }
        BoundaryShape::Point => write_boundary_points(geom, writer),
        BoundaryShape::MultiPoint(count) => {
            write_wkb_multipoint_header(writer, geom.dim(), count)?;
            write_boundary_points(geom, writer)
        }
        BoundaryShape::Line => write_boundary_lines(geom, writer),
        BoundaryShape::MultiLine(count) => {
            write_wkb_multilinestring_header(writer, geom.dim(), count)?;
            write_boundary_lines(geom, writer)
        }
        BoundaryShape::Collection {
            empty_collections,
            points,
            lines,
        } => {
            let component_count =
                empty_collections + usize::from(points > 0) + usize::from(lines > 0);
            write_wkb_geometrycollection_header(writer, geom.dim(), component_count)?;

            if let geo_traits::GeometryType::GeometryCollection(collection) = geom.as_type() {
                for child in collection.geometries() {
                    if boundary_shape(child) == BoundaryShape::EmptyCollection {
                        write_wkb_geometrycollection_header(writer, child.dim(), 0)?;
                    }
                }
            }

            match points {
                0 => {}
                1 => write_boundary_points(geom, writer)?,
                _ => {
                    write_wkb_multipoint_header(writer, geom.dim(), points)?;
                    write_boundary_points(geom, writer)?;
                }
            }
            match lines {
                0 => {}
                1 => write_boundary_lines(geom, writer)?,
                _ => {
                    write_wkb_multilinestring_header(writer, geom.dim(), lines)?;
                    write_boundary_lines(geom, writer)?;
                }
            }
            Ok(())
        }
    }
}

fn write_boundary_points(geom: &Wkb, writer: &mut impl Write) -> Result<(), SedonaGeometryError> {
    match geom.as_type() {
        geo_traits::GeometryType::LineString(line) => {
            if line.num_coords() > 1 && !coords_equal(line, 0, line.num_coords() - 1) {
                write_point(writer, &line.coord(0).unwrap())?;
                write_point(writer, &line.coord(line.num_coords() - 1).unwrap())?;
            }
        }
        geo_traits::GeometryType::MultiLineString(lines) => {
            for coord in multiline_boundary_coords(lines) {
                write_point(writer, &coord)?;
            }
        }
        geo_traits::GeometryType::GeometryCollection(collection) => {
            for child in collection.geometries() {
                if boundary_shape(child) != BoundaryShape::EmptyCollection {
                    write_boundary_points(child, writer)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn write_boundary_lines(geom: &Wkb, writer: &mut impl Write) -> Result<(), SedonaGeometryError> {
    match geom.as_type() {
        geo_traits::GeometryType::Polygon(polygon) => write_polygon_rings(polygon, writer)?,
        geo_traits::GeometryType::MultiPolygon(polygons) => {
            for polygon in polygons.polygons() {
                write_polygon_rings(polygon, writer)?;
            }
        }
        geo_traits::GeometryType::GeometryCollection(collection) => {
            for child in collection.geometries() {
                if boundary_shape(child) != BoundaryShape::EmptyCollection {
                    write_boundary_lines(child, writer)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn write_polygon_rings(
    polygon: &wkb::reader::Polygon<'_>,
    writer: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    if let Some(exterior) = polygon.exterior() {
        write_ring(writer, exterior)?;
    }
    for interior in polygon.interiors() {
        write_ring(writer, interior)?;
    }
    Ok(())
}

fn write_point(writer: &mut impl Write, coord: &Coord<'_>) -> Result<(), SedonaGeometryError> {
    write_wkb_point_header(writer, coord.dim())?;
    write_coord(writer, coord)
}

fn write_ring(writer: &mut impl Write, ring: &LinearRing<'_>) -> Result<(), SedonaGeometryError> {
    write_wkb_linestring_header(writer, ring.dim(), ring.num_coords())?;
    write_coords(writer, ring.coords_slice(), ring.byte_order())
}

fn write_coord(writer: &mut impl Write, coord: &Coord<'_>) -> Result<(), SedonaGeometryError> {
    write_coords(writer, coord.coord_slice(), coord.byte_order())
}

fn write_coords(
    writer: &mut impl Write,
    coords: &[u8],
    byte_order: Endianness,
) -> Result<(), SedonaGeometryError> {
    if matches!(byte_order, Endianness::LittleEndian) {
        writer.write_all(coords)?;
    } else {
        for ordinate in coords.as_chunks::<{ size_of::<f64>() }>().0 {
            let mut little_endian = [0; size_of::<f64>()];
            little_endian.copy_from_slice(ordinate);
            little_endian.reverse();
            writer.write_all(&little_endian)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY_ITEM_CRS};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_boundary_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type.clone());

        let result = tester
            .invoke_scalar(
                "GEOMETRYCOLLECTION(LINESTRING(1 1,2 2),GEOMETRYCOLLECTION(POLYGON((3 3,4 4,5 5,3 3)),GEOMETRYCOLLECTION(LINESTRING(6 6,7 7),POLYGON((8 8,9 9,10 10,8 8)))))",
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION(MULTIPOINT((1 1),(2 2),(6 6),(7 7)),MULTILINESTRING((3 3,4 4,5 5,3 3),(8 8,9 9,10 10,8 8)))");

        let result = tester
            .invoke_scalar("LINESTRING(100 150,50 60, 70 80, 160 170)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT((100 150),(160 170))");

        let result = tester
            .invoke_scalar(
                "POLYGON (( 10 130, 50 190, 110 190, 140 150, 150 80, 100 10, 20 40, 10 130 ), ( 70 40, 100 50, 120 80, 80 110, 50 90, 70 40 ))"
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTILINESTRING((10 130,50 190,110 190,140 150,150 80,100 10,20 40,10 130), (70 40,100 50,120 80,80 110,50 90,70 40))"
        );

        let result = tester
            .invoke_scalar("MULTILINESTRING ((10 10, 20 20), (30 30, 40 40, 30 30))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT (10 10, 20 20)");

        // Endpoints shared by an even number of components are not in the
        // boundary (the OGC mod-2 boundary node rule).
        let result = tester
            .invoke_scalar("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT (0 0, 2 2)");

        let result = tester.invoke_scalar("GEOMETRYCOLLECTION(MULTIPOINT(-2 3, -2 2), LINESTRING(5 5, 10 10), POLYGON((-7 4.2, -7.1 5, -7.1 4.3, -7 4.2)))").unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(GEOMETRYCOLLECTION EMPTY, MULTIPOINT(5 5, 10 10), LINESTRING(-7 4.2, -7.1 5, -7.1 4.3, -7 4.2))"
        );

        let result = tester.invoke_scalar("POINT (10 20)").unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");

        let result = tester
            .invoke_scalar("MULTIPOINT (5 5, 10 10, 15 15)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");

        let result = tester
            .invoke_scalar("LINESTRING (0 0, 1 1, 0 1, 0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT EMPTY");

        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0,0 10,10 10,10 0,0 0)");

        let result = tester
            .invoke_scalar(
                "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((10 10, 10 11, 11 11, 11 10, 10 10)))",
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTILINESTRING((0 0,0 1,1 1,1 0,0 0),(10 10,10 11,11 11,11 10,10 10))",
        );

        let result = tester
            .invoke_scalar("GEOMETRYCOLLECTION(POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)), GEOMETRYCOLLECTION(LINESTRING(10 10, 10 20)))")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(MULTIPOINT((10 10),(10 20)), LINESTRING(0 0,0 1,1 1,1 0,0 0))",
        );

        let result = tester.invoke_scalar("GEOMETRYCOLLECTION EMPTY").unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");

        let result = tester.invoke_scalar("POLYGON EMPTY").unwrap();
        tester.assert_scalar_result_equals(result, "MULTILINESTRING EMPTY");

        let result = tester.invoke_scalar("MULTIPOLYGON EMPTY").unwrap();
        tester.assert_scalar_result_equals(result, "MULTILINESTRING EMPTY");

        let result = tester
            .invoke_scalar("LINESTRING Z (0 0 1, 1 1 2, 0 0 3)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT Z EMPTY");
    }

    #[rstest]
    fn udf_invoke_item_crs(
        #[values(WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_boundary_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type);

        let result = tester.invoke_scalar("POINT (1 3)").unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");
    }
}
