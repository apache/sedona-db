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
use std::cell::RefCell;

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use geo_traits::*;
use geos::GResult;
use wkb::{reader::*, Endianness};

/// A factory for converting WKB to GEOS geometries.
///
/// This factory uses a scratch buffer to store intermediate coordinate data.
/// The scratch buffer is reused for each conversion, which reduces memory allocation
/// overhead.
pub struct GEOSWkbFactory {
    scratch: RefCell<Vec<f64>>,
}

impl Default for GEOSWkbFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl GEOSWkbFactory {
    /// Create a new GEOSWkbFactory.
    pub fn new() -> Self {
        Self {
            scratch: RefCell::new(Vec::new()),
        }
    }

    /// Create a GEOS geometry from a WKB.
    pub fn create(&self, wkb: &Wkb) -> GResult<geos::Geometry> {
        let scratch = &mut self.scratch.borrow_mut();
        geometry_to_geos(scratch, wkb)
    }
}

fn geometry_to_geos(scratch: &mut Vec<f64>, wkb: &Wkb) -> GResult<geos::Geometry> {
    let geom = wkb.as_type();
    match geom {
        geo_traits::GeometryType::Point(p) => point_to_geos(scratch, p),
        geo_traits::GeometryType::LineString(ls) => line_string_to_geos(scratch, ls),
        geo_traits::GeometryType::Polygon(poly) => polygon_to_geos(scratch, poly),
        geo_traits::GeometryType::MultiPoint(mp) => multi_point_to_geos(scratch, mp),
        geo_traits::GeometryType::MultiLineString(mls) => multi_line_string_to_geos(scratch, mls),
        geo_traits::GeometryType::MultiPolygon(mpoly) => multi_polygon_to_geos(scratch, mpoly),
        geo_traits::GeometryType::GeometryCollection(gc) => {
            geometry_collection_to_geos(scratch, gc)
        }
        _ => Err(geos::Error::ConversionError(
            "Unsupported geometry type".to_string(),
        )),
    }
}

fn point_to_geos(scratch: &mut Vec<f64>, p: &Point) -> GResult<geos::Geometry> {
    if p.is_empty() {
        geos::Geometry::create_empty_point()
    } else {
        let coord_seq = create_coord_sequence_from_raw_parts(
            p.coord_slice(),
            p.dimension(),
            p.byte_order(),
            1,
            scratch,
        )?;
        let point = geos::Geometry::create_point(coord_seq)?;
        Ok(point)
    }
}

fn line_string_to_geos(scratch: &mut Vec<f64>, ls: &LineString) -> GResult<geos::Geometry> {
    let num_points = ls.num_coords();
    if num_points == 0 {
        geos::Geometry::create_empty_line_string()
    } else {
        let coord_seq = create_coord_sequence_from_raw_parts(
            ls.coords_slice(),
            ls.dimension(),
            ls.byte_order(),
            num_points,
            scratch,
        )?;
        geos::Geometry::create_line_string(coord_seq)
    }
}

fn polygon_to_geos(scratch: &mut Vec<f64>, poly: &Polygon) -> GResult<geos::Geometry> {
    // Create exterior ring
    let exterior = if let Some(ring) = poly.exterior() {
        let coord_seq = create_coord_sequence_from_raw_parts(
            ring.coords_slice(),
            ring.dimension(),
            ring.byte_order(),
            ring.num_coords(),
            scratch,
        )?;
        geos::Geometry::create_linear_ring(coord_seq)?
    } else {
        return geos::Geometry::create_empty_polygon();
    };

    // Create interior rings
    let num_interiors = poly.num_interiors();
    let mut interior_rings = Vec::with_capacity(num_interiors);
    for i in 0..num_interiors {
        let ring = poly.interior(i).unwrap();
        let coord_seq = create_coord_sequence_from_raw_parts(
            ring.coords_slice(),
            ring.dimension(),
            ring.byte_order(),
            ring.num_coords(),
            scratch,
        )?;
        let interior_ring = geos::Geometry::create_linear_ring(coord_seq)?;
        interior_rings.push(interior_ring);
    }

    geos::Geometry::create_polygon(exterior, interior_rings)
}

fn multi_point_to_geos(scratch: &mut Vec<f64>, mp: &MultiPoint) -> GResult<geos::Geometry> {
    let num_points = mp.num_points();
    if num_points == 0 {
        // Create an empty multi-point by creating a geometry collection with no geometries
        geos::Geometry::create_empty_collection(geos::GeometryTypes::MultiPoint)
    } else {
        let mut points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let point = unsafe { mp.point_unchecked(i) };
            let geos_point = point_to_geos(scratch, &point)?;
            points.push(geos_point);
        }
        geos::Geometry::create_multipoint(points)
    }
}

fn multi_line_string_to_geos(
    scratch: &mut Vec<f64>,
    mls: &MultiLineString,
) -> GResult<geos::Geometry> {
    let num_line_strings = mls.num_line_strings();
    if num_line_strings == 0 {
        geos::Geometry::create_empty_collection(geos::GeometryTypes::MultiLineString)
    } else {
        let mut line_strings = Vec::with_capacity(num_line_strings);
        for i in 0..num_line_strings {
            let ls = unsafe { mls.line_string_unchecked(i) };
            let geos_line_string = line_string_to_geos(scratch, ls)?;
            line_strings.push(geos_line_string);
        }
        geos::Geometry::create_multiline_string(line_strings)
    }
}

fn multi_polygon_to_geos(scratch: &mut Vec<f64>, mpoly: &MultiPolygon) -> GResult<geos::Geometry> {
    let num_polygons = mpoly.num_polygons();
    if num_polygons == 0 {
        geos::Geometry::create_empty_collection(geos::GeometryTypes::MultiPolygon)
    } else {
        let mut polygons = Vec::with_capacity(num_polygons);
        for i in 0..num_polygons {
            let poly = unsafe { mpoly.polygon_unchecked(i) };
            let geos_polygon = polygon_to_geos(scratch, poly)?;
            polygons.push(geos_polygon);
        }
        geos::Geometry::create_multipolygon(polygons)
    }
}

fn geometry_collection_to_geos(
    scratch: &mut Vec<f64>,
    gc: &GeometryCollection,
) -> GResult<geos::Geometry> {
    if gc.num_geometries() == 0 {
        geos::Geometry::create_empty_collection(geos::GeometryTypes::GeometryCollection)
    } else {
        let num_geometries = gc.num_geometries();
        let mut geometries = Vec::with_capacity(num_geometries);
        for i in 0..num_geometries {
            let geom = gc.geometry(i).unwrap();
            let geos_geom = geometry_to_geos(scratch, geom)?;
            geometries.push(geos_geom);
        }
        geos::Geometry::create_geometry_collection(geometries)
    }
}

const NATIVE_ENDIANNESS: Endianness = if cfg!(target_endian = "big") {
    Endianness::BigEndian
} else {
    Endianness::LittleEndian
};

fn create_coord_sequence_from_raw_parts(
    buf: &[u8],
    dim: Dimension,
    byte_order: Endianness,
    num_coords: usize,
    scratch: &mut Vec<f64>,
) -> GResult<geos::CoordSeq> {
    let (has_z, has_m, dim_size) = match dim {
        Dimension::Xy => (false, false, 2),
        Dimension::Xyz => (true, false, 3),
        Dimension::Xym => (false, true, 3),
        Dimension::Xyzm => (true, true, 4),
    };
    let num_ordinates = dim_size * num_coords;

    // If the byte order matches native endianness, we can potentially use zero-copy
    if byte_order == NATIVE_ENDIANNESS {
        let ptr = buf.as_ptr();

        // On platforms with unaligned memory access support, we can construct the coord seq
        // directly from the raw parts without copying to the scratch buffer.
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            let coords_f64 =
                unsafe { &*core::ptr::slice_from_raw_parts(ptr as *const f64, num_ordinates) };
            geos::CoordSeq::new_from_buffer(coords_f64, num_coords, has_z, has_m)
        }

        // On platforms without unaligned memory access support, we need to copy the data to the
        // scratch buffer to make sure the data is aligned.
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            unsafe {
                scratch.clear();
                scratch.reserve(num_ordinates);
                scratch.set_len(num_ordinates);
                std::ptr::copy_nonoverlapping(
                    ptr,
                    scratch.as_mut_ptr() as *mut u8,
                    num_ordinates * std::mem::size_of::<f64>(),
                );
                geos::CoordSeq::new_from_buffer(scratch.as_slice(), num_coords, has_z, has_m)
            }
        }
    } else {
        // Need to convert byte order
        match byte_order {
            Endianness::BigEndian => {
                save_f64_to_scratch::<BigEndian>(scratch, buf, num_ordinates);
            }
            Endianness::LittleEndian => {
                save_f64_to_scratch::<LittleEndian>(scratch, buf, num_ordinates);
            }
        }
        geos::CoordSeq::new_from_buffer(scratch.as_slice(), num_coords, has_z, has_m)
    }
}

fn save_f64_to_scratch<B: ByteOrder>(scratch: &mut Vec<f64>, buf: &[u8], num_ordinates: usize) {
    scratch.clear();
    scratch.reserve(num_ordinates);
    // Safety: we have already reserved the capacity, so we can set the length safely.
    // Justification: rewriting the loop to not use Vec::push makes it many times faster,
    // since it eliminates several memory loads and stores for vector's length and capacity,
    // and it enables the compiler to generate vectorized code.
    #[allow(clippy::uninit_vec)]
    unsafe {
        scratch.set_len(num_ordinates);
    }
    assert!(num_ordinates * 8 <= buf.len());
    for (i, tgt) in scratch.iter_mut().enumerate().take(num_ordinates) {
        let offset = i * 8;
        let value = B::read_f64(&buf[offset..]);
        *tgt = value;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use geo_types::{
        line_string, point, polygon, Geometry, GeometryCollection, LineString, MultiLineString,
        MultiPoint, MultiPolygon, Point, Polygon,
    };
    use geos::Geom;
    use wkb::{
        writer::{
            write_geometry_collection, write_line_string, write_multi_line_string,
            write_multi_point, write_multi_polygon, write_point, write_polygon, WriteOptions,
        },
        Endianness,
    };

    pub(super) fn point_2d() -> Point {
        point!(
            x: 0., y: 1.
        )
    }

    pub(super) fn linestring_2d() -> LineString {
        line_string![
            (x: 0., y: 1.),
            (x: 1., y: 2.)
        ]
    }

    pub(super) fn polygon_2d() -> Polygon {
        polygon![
            (x: -111., y: 45.),
            (x: -111., y: 41.),
            (x: -104., y: 41.),
            (x: -104., y: 45.),
        ]
    }

    pub(super) fn polygon_2d_with_interior() -> Polygon {
        polygon!(
            exterior: [
                (x: -111., y: 45.),
                (x: -111., y: 41.),
                (x: -104., y: 41.),
                (x: -104., y: 45.),
            ],
            interiors: [
                [
                    (x: -110., y: 44.),
                    (x: -110., y: 42.),
                    (x: -105., y: 42.),
                    (x: -105., y: 44.),
                ],
            ],
        )
    }

    pub(super) fn multi_point_2d() -> MultiPoint {
        MultiPoint::new(vec![
            point!(
                x: 0., y: 1.
            ),
            point!(
                x: 1., y: 2.
            ),
        ])
    }

    pub(super) fn multi_line_string_2d() -> MultiLineString {
        MultiLineString::new(vec![
            line_string![
                (x: -111., y: 45.),
                (x: -111., y: 41.),
                (x: -104., y: 41.),
                (x: -104., y: 45.),
            ],
            line_string![
                (x: -110., y: 44.),
                (x: -110., y: 42.),
                (x: -105., y: 42.),
                (x: -105., y: 44.),
            ],
        ])
    }

    pub(super) fn multi_polygon_2d() -> MultiPolygon {
        MultiPolygon::new(vec![
            polygon![
                (x: -111., y: 45.),
                (x: -111., y: 41.),
                (x: -104., y: 41.),
                (x: -104., y: 45.),
            ],
            polygon!(
                exterior: [
                    (x: -111., y: 45.),
                    (x: -111., y: 41.),
                    (x: -104., y: 41.),
                    (x: -104., y: 45.),
                ],
                interiors: [
                    [
                        (x: -110., y: 44.),
                        (x: -110., y: 42.),
                        (x: -105., y: 42.),
                        (x: -105., y: 44.),
                    ],
                ],
            ),
        ])
    }

    pub(super) fn geometry_collection_2d() -> GeometryCollection {
        GeometryCollection::new_from(vec![
            Geometry::Point(point_2d()),
            Geometry::LineString(linestring_2d()),
            Geometry::Polygon(polygon_2d()),
            Geometry::Polygon(polygon_2d_with_interior()),
            Geometry::MultiPoint(multi_point_2d()),
            Geometry::MultiLineString(multi_line_string_2d()),
            Geometry::MultiPolygon(multi_polygon_2d()),
        ])
    }

    fn test_geometry_conversion(geo_geom: &Geometry, endianness: Endianness) {
        // Convert geo geometry to WKB
        let mut buf = Vec::new();
        let write_options = WriteOptions { endianness };
        match geo_geom {
            Geometry::Point(p) => write_point(&mut buf, p, &write_options).unwrap(),
            Geometry::LineString(ls) => write_line_string(&mut buf, ls, &write_options).unwrap(),
            Geometry::Polygon(p) => write_polygon(&mut buf, p, &write_options).unwrap(),
            Geometry::MultiPoint(mp) => write_multi_point(&mut buf, mp, &write_options).unwrap(),
            Geometry::MultiLineString(mls) => {
                write_multi_line_string(&mut buf, mls, &write_options).unwrap()
            }
            Geometry::MultiPolygon(mp) => {
                write_multi_polygon(&mut buf, mp, &write_options).unwrap()
            }
            Geometry::GeometryCollection(gc) => {
                write_geometry_collection(&mut buf, gc, &write_options).unwrap()
            }
            Geometry::Line(_) => panic!("Line geometry not supported in tests"),
            Geometry::Rect(_) => panic!("Rect geometry not supported in tests"),
            Geometry::Triangle(_) => panic!("Triangle geometry not supported in tests"),
        }

        // Read WKB back
        let wkb = wkb::reader::read_wkb(&buf).unwrap();

        // Convert to GEOS using our ToGeos converter
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        // Convert back to geo for comparison
        let geo_from_geos: Geometry = geos_geom.try_into().unwrap();

        // Compare the geometries
        assert_eq!(*geo_geom, geo_from_geos);
    }

    #[test]
    fn test_point_conversion() {
        let point = point_2d();
        let geo_geom = Geometry::Point(point);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_empty_point_conversion() {
        // Create an empty point by writing NaN coordinates
        let mut buf = Vec::new();
        buf.push(1); // Little endian
        buf.extend_from_slice(&1u32.to_le_bytes()); // Point type
        buf.extend_from_slice(&f64::NAN.to_le_bytes()); // x = NaN
        buf.extend_from_slice(&f64::NAN.to_le_bytes()); // y = NaN

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        assert!(geos_geom.is_empty().unwrap());
    }

    #[test]
    fn test_line_string_conversion() {
        let line_string = linestring_2d();
        let geo_geom = Geometry::LineString(line_string);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_empty_line_string_conversion() {
        let mut buf = Vec::new();
        write_line_string(
            &mut buf,
            &LineString::new(vec![]),
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        assert!(geos_geom.is_empty().unwrap());
    }

    #[test]
    fn test_polygon_conversion() {
        let polygon = polygon_2d();
        let geo_geom = Geometry::Polygon(polygon);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_polygon_with_interior_conversion() {
        let polygon = polygon_2d_with_interior();
        let geo_geom = Geometry::Polygon(polygon);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_multi_point_conversion() {
        let multi_point = multi_point_2d();
        let geo_geom = Geometry::MultiPoint(multi_point);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_empty_multi_point_conversion() {
        let mut buf = Vec::new();
        write_multi_point(
            &mut buf,
            &MultiPoint::new(vec![]),
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        assert!(geos_geom.is_empty().unwrap());
    }

    #[test]
    fn test_multi_line_string_conversion() {
        let multi_line_string = multi_line_string_2d();
        let geo_geom = Geometry::MultiLineString(multi_line_string);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_empty_multi_line_string_conversion() {
        let mut buf = Vec::new();
        write_multi_line_string(
            &mut buf,
            &MultiLineString::new(vec![]),
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        assert!(geos_geom.is_empty().unwrap());
    }

    #[test]
    fn test_multi_polygon_conversion() {
        let multi_polygon = multi_polygon_2d();
        let geo_geom = Geometry::MultiPolygon(multi_polygon);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_empty_multi_polygon_conversion() {
        let mut buf = Vec::new();
        write_multi_polygon(
            &mut buf,
            &MultiPolygon::new(vec![]),
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        assert!(geos_geom.is_empty().unwrap());
    }

    #[test]
    fn test_geometry_collection_conversion() {
        let geometry_collection = geometry_collection_2d();
        let geo_geom = Geometry::GeometryCollection(geometry_collection);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_empty_geometry_collection_conversion() {
        let mut buf = Vec::new();
        write_geometry_collection(
            &mut buf,
            &GeometryCollection::new_from(vec![]),
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        assert!(geos_geom.is_empty().unwrap());
    }

    #[test]
    fn test_nested_geometry_collection() {
        // Create a geometry collection containing other geometry collections
        let inner_gc1 = GeometryCollection::new_from(vec![
            Geometry::Point(point_2d()),
            Geometry::LineString(linestring_2d()),
        ]);

        let inner_gc2 = GeometryCollection::new_from(vec![
            Geometry::Polygon(polygon_2d()),
            Geometry::MultiPoint(multi_point_2d()),
        ]);

        let outer_gc = GeometryCollection::new_from(vec![
            Geometry::GeometryCollection(inner_gc1),
            Geometry::GeometryCollection(inner_gc2),
            Geometry::MultiLineString(multi_line_string_2d()),
        ]);

        let geo_geom = Geometry::GeometryCollection(outer_gc);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_coordinate_precision() {
        // Test with high precision coordinates
        let high_precision_point = Point::new(123.456789012345, -98.765432109876);
        let geo_geom = Geometry::Point(high_precision_point);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_large_coordinates() {
        // Test with very large coordinate values
        let large_point = Point::new(1e10, -1e10);
        let geo_geom = Geometry::Point(large_point);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_negative_coordinates() {
        // Test with negative coordinates
        let negative_point = Point::new(-180.0, -90.0);
        let geo_geom = Geometry::Point(negative_point);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_zero_coordinates() {
        // Test with zero coordinates
        let zero_point = Point::new(0.0, 0.0);
        let geo_geom = Geometry::Point(zero_point);

        test_geometry_conversion(&geo_geom, Endianness::LittleEndian);
        test_geometry_conversion(&geo_geom, Endianness::BigEndian);
    }

    #[test]
    fn test_endianness_handling() {
        let factory = GEOSWkbFactory::new();
        // Test that both endianness variants work correctly
        let point = point_2d();
        let geo_geom = Geometry::Point(point);

        // Test little endian
        let mut buf_le = Vec::new();
        write_point(
            &mut buf_le,
            &point_2d(),
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        let wkb_le = read_wkb(&buf_le).unwrap();
        let geos_geom_le = factory.create(&wkb_le).unwrap();
        let geo_from_geos_le: Geometry = geos_geom_le.try_into().unwrap();

        // Test big endian
        let mut buf_be = Vec::new();
        write_point(
            &mut buf_be,
            &point_2d(),
            &WriteOptions {
                endianness: Endianness::BigEndian,
            },
        )
        .unwrap();
        let wkb_be = read_wkb(&buf_be).unwrap();
        let geos_geom_be = factory.create(&wkb_be).unwrap();
        let geo_from_geos_be: Geometry = geos_geom_be.try_into().unwrap();

        // Both should produce the same result
        assert_eq!(geo_from_geos_le, geo_from_geos_be);
        assert_eq!(geo_geom, geo_from_geos_le);
    }

    #[test]
    fn test_xyz_dimension_handling() {
        // Test XYZ dimension handling by manually creating WKB with XYZ coordinates
        let mut buf = Vec::new();

        // Write WKB header for LineString XYZ (type 1002)
        buf.push(1); // Little endian
        buf.extend_from_slice(&1002u32.to_le_bytes()); // LineString XYZ
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 points

        // Write XYZ coordinates: (0.0, 1.0, 10.0), (1.0, 2.0, 20.0)
        buf.extend_from_slice(&0.0f64.to_le_bytes());
        buf.extend_from_slice(&1.0f64.to_le_bytes());
        buf.extend_from_slice(&10.0f64.to_le_bytes());
        buf.extend_from_slice(&1.0f64.to_le_bytes());
        buf.extend_from_slice(&2.0f64.to_le_bytes());
        buf.extend_from_slice(&20.0f64.to_le_bytes());

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        // Verify the geometry was created successfully
        assert!(!geos_geom.is_empty().unwrap());

        // Verify coordinates by checking the WKT representation
        let wkt = geos_geom.to_wkt().unwrap();
        // Expected WKT for LineString with XYZ coordinates (0.0, 1.0, 10.0), (1.0, 2.0, 20.0)
        let expected_wkt = "LINESTRING Z (0 1 10, 1 2 20)";
        assert_eq!(wkt, expected_wkt);
    }

    #[test]
    fn test_xym_dimension_handling() {
        // Test XYM dimension handling by manually creating WKB with XYM coordinates
        let mut buf = Vec::new();

        // Write WKB header for LineString XYM (type 2002)
        buf.push(1); // Little endian
        buf.extend_from_slice(&2002u32.to_le_bytes()); // LineString XYM
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 points

        // Write XYM coordinates: (0.0, 1.0, 100.0), (1.0, 2.0, 200.0)
        buf.extend_from_slice(&0.0f64.to_le_bytes());
        buf.extend_from_slice(&1.0f64.to_le_bytes());
        buf.extend_from_slice(&100.0f64.to_le_bytes());
        buf.extend_from_slice(&1.0f64.to_le_bytes());
        buf.extend_from_slice(&2.0f64.to_le_bytes());
        buf.extend_from_slice(&200.0f64.to_le_bytes());

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        // Verify the geometry was created successfully
        assert!(!geos_geom.is_empty().unwrap());

        // Verify coordinates by checking the WKT representation
        let wkt = geos_geom.to_wkt().unwrap();
        // Expected WKT for LineString with XYM coordinates (0.0, 1.0, 100.0), (1.0, 2.0, 200.0)
        let expected_wkt = "LINESTRING M (0 1 100, 1 2 200)";
        assert_eq!(wkt, expected_wkt);
    }

    #[test]
    fn test_xyzm_dimension_handling() {
        // Test XYZM dimension handling by manually creating WKB with XYZM coordinates
        let mut buf = Vec::new();

        // Write WKB header for LineString XYZM (type 3002)
        buf.push(1); // Little endian
        buf.extend_from_slice(&3002u32.to_le_bytes()); // LineString XYZM
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 points

        // Write XYZM coordinates: (0.0, 1.0, 10.0, 100.0), (1.0, 2.0, 20.0, 200.0)
        buf.extend_from_slice(&0.0f64.to_le_bytes());
        buf.extend_from_slice(&1.0f64.to_le_bytes());
        buf.extend_from_slice(&10.0f64.to_le_bytes());
        buf.extend_from_slice(&100.0f64.to_le_bytes());
        buf.extend_from_slice(&1.0f64.to_le_bytes());
        buf.extend_from_slice(&2.0f64.to_le_bytes());
        buf.extend_from_slice(&20.0f64.to_le_bytes());
        buf.extend_from_slice(&200.0f64.to_le_bytes());

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        // Verify the geometry was created successfully
        assert!(!geos_geom.is_empty().unwrap());

        // Verify coordinates by checking the WKT representation
        let wkt = geos_geom.to_wkt().unwrap();
        // Expected WKT for LineString with XYZM coordinates (0.0, 1.0, 10.0, 100.0), (1.0, 2.0, 20.0, 200.0)
        let expected_wkt = "LINESTRING ZM (0 1 10 100, 1 2 20 200)";
        assert_eq!(wkt, expected_wkt);
    }

    #[test]
    fn test_big_endian_xyz_dimension_handling() {
        // Test XYZ dimension handling with big endian byte order
        let mut buf = Vec::new();

        // Write WKB header for LineString XYZ (type 1002) in big endian
        buf.push(0); // Big endian
        buf.extend_from_slice(&1002u32.to_be_bytes()); // LineString XYZ
        buf.extend_from_slice(&2u32.to_be_bytes()); // 2 points

        // Write XYZ coordinates in big endian: (0.0, 1.0, 10.0), (1.0, 2.0, 20.0)
        buf.extend_from_slice(&0.0f64.to_be_bytes());
        buf.extend_from_slice(&1.0f64.to_be_bytes());
        buf.extend_from_slice(&10.0f64.to_be_bytes());
        buf.extend_from_slice(&1.0f64.to_be_bytes());
        buf.extend_from_slice(&2.0f64.to_be_bytes());
        buf.extend_from_slice(&20.0f64.to_be_bytes());

        let wkb = read_wkb(&buf).unwrap();
        let geos_geom = GEOSWkbFactory::new().create(&wkb).unwrap();

        // Verify the geometry was created successfully
        assert!(!geos_geom.is_empty().unwrap());

        // Verify coordinates by checking the WKT representation
        let wkt = geos_geom.to_wkt().unwrap();
        // Expected WKT for LineString with XYZ coordinates (0.0, 1.0, 10.0), (1.0, 2.0, 20.0)
        let expected_wkt = "LINESTRING Z (0 1 10, 1 2 20)";
        assert_eq!(wkt, expected_wkt);
    }

    /// Represents a single WKB test case, holding the expected geometry type, Dimension,
    /// the raw WKB bytes, and the WKT string.
    /// This is the direct Rust equivalent of your C++ `WKBTestCase` struct, with WKT added.
    #[derive(Debug, PartialEq, Clone)]
    pub struct WkbTestCase {
        pub dimension: Dimension,
        pub wkb_bytes: Vec<u8>,
        pub wkt_string: String, // Added WKT field
    }

    // You can then define your test cases as a `Vec<WkbTestCase>`
    pub fn get_wkb_test_cases() -> Vec<WkbTestCase> {
        vec![
            // POINT EMPTY
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
                ],
                wkt_string: "POINT EMPTY".to_string(),
            },
            // POINT (30 10)
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                ],
                wkt_string: "POINT (30 10)".to_string(),
            },
            // POINT Z (30 10 40)
            WkbTestCase {
                dimension: Dimension::Xyz,
                wkb_bytes: vec![
                    0x01, 0xe9, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x44, 0x40,
                ],
                wkt_string: "POINT Z (30 10 40)".to_string(),
            },
            // POINT M (30 10 300)
            WkbTestCase {
                dimension: Dimension::Xym,
                wkb_bytes: vec![
                    0x01, 0xd1, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0xc0, 0x72, 0x40,
                ],
                wkt_string: "POINT M (30 10 300)".to_string(),
            },
            // POINT ZM (30 10 40 300)
            WkbTestCase {
                dimension: Dimension::Xyzm,
                wkb_bytes: vec![
                    0x01, 0xb9, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40,
                ],
                wkt_string: "POINT ZM (30 10 40 300)".to_string(),
            },
            // POINT (30 10) (big endian)
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x00, 0x00, 0x00, 0x00, 0x01, 0x40, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x40, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                ],
                wkt_string: "POINT (30 10)".to_string(), // WKT is endian-agnostic
            },
            // LINESTRING EMPTY
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![0x01, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                wkt_string: "LINESTRING EMPTY".to_string(),
            },
            // LINESTRING (30 10, 10 30, 40 40)
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x01, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x44, 0x40,
                ],
                wkt_string: "LINESTRING (30 10, 10 30, 40 40)".to_string(),
            },
            // LINESTRING Z (30 10 40, 10 30 40, 40 40 80)
            WkbTestCase {
                dimension: Dimension::Xyz,
                wkb_bytes: vec![
                    0x01, 0xea, 0x03, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x54, 0x40
                ],
                wkt_string: "LINESTRING Z (30 10 40, 10 30 40, 40 40 80)".to_string(),
            },
            // LINESTRING M (30 10 300, 10 30 300, 40 40 1600)
            WkbTestCase {
                dimension: Dimension::Xym,
                wkb_bytes: vec![
                    0x01, 0xd2, 0x07, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x99, 0x40,
                ],
                wkt_string: "LINESTRING M (30 10 300, 10 30 300, 40 40 1600)".to_string(),
            },
            // LINESTRING ZM (30 10 40 300, 10 30 40 300, 40 40 80 1600)
            WkbTestCase {
                dimension: Dimension::Xyzm,
                wkb_bytes: vec![
                    0x01, 0xba, 0x0b, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x99, 0x40,
                ],
                wkt_string: "LINESTRING ZM (30 10 40 300, 10 30 40 300, 40 40 80 1600)".to_string(),
            },
            // LINESTRING (30 10, 10 30, 40 40) (big endian)
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x40, 0x3e, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x40, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
                    0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x3e, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x40, 0x44, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x44, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00,
                ],
                wkt_string: "LINESTRING (30 10, 10 30, 40 40)".to_string(),
            },
            // POLYGON EMPTY
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![0x01, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                wkt_string: "POLYGON EMPTY".to_string(),
            },
            // POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x24, 0x40,
                ],
                wkt_string: "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))".to_string(),
            },
            // POLYGON Z ((30 10 40, 40 40 80, 20 40 60, 10 20 30, 30 10 40))
            WkbTestCase {
                dimension: Dimension::Xyz,
                wkb_bytes: vec![
                    0x01, 0xeb, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x4e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x44, 0x40,
                ],
                wkt_string: "POLYGON Z ((30 10 40, 40 40 80, 20 40 60, 10 20 30, 30 10 40))".to_string(),
            },
            // POLYGON M ((30 10 300, 40 40 1600, 20 40 800, 10 20 200, 30 10 300))
            WkbTestCase {
                dimension: Dimension::Xym,
                wkb_bytes: vec![
                    0x01, 0xd3, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x89, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x69, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0xc0, 0x72, 0x40,
                ],
                wkt_string: "POLYGON M ((30 10 300, 40 40 1600, 20 40 800, 10 20 200, 30 10 300))".to_string(),
            },
            // POLYGON ZM ((30 10 40 300, 40 40 80 1600, 20 40 60 800, 10 20 30 200, 30
            // 10 40 300))
            WkbTestCase {
                dimension: Dimension::Xyzm,
                wkb_bytes: vec![
                    0x01, 0xbb, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x54, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4e, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x89, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x69, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0xc0, 0x72, 0x40,
                ],
                wkt_string: "POLYGON ZM ((30 10 40 300, 40 40 80 1600, 20 40 60 800, 10 20 30 200, 30 10 40 300))".to_string(),
            },
            // MULTIPOINT EMPTY
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                wkt_string: "MULTIPOINT EMPTY".to_string(),
            },
            // MULTIPOINT ((30 10))
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x01, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40,
                ],
                wkt_string: "MULTIPOINT ((30 10))".to_string(),
            },
            // MULTIPOINT Z ((30 10 40))
            WkbTestCase {
                dimension: Dimension::Xyz,
                wkb_bytes: vec![
                    0x01, 0xec, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xe9, 0x03, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40,
                ],
                wkt_string: "MULTIPOINT Z ((30 10 40))".to_string(),
            },
            // MULTIPOINT M ((30 10 300))
            WkbTestCase {
                dimension: Dimension::Xym,
                wkb_bytes: vec![
                    0x01, 0xd4, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xd1, 0x07, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40,
                ],
                wkt_string: "MULTIPOINT M ((30 10 300))".to_string(),
            },
            // MULTIPOINT ZM ((30 10 40 300))
            WkbTestCase {
                dimension: Dimension::Xyzm,
                wkb_bytes: vec![
                    0x01, 0xbc, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xb9, 0x0b, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40,
                ],
                wkt_string: "MULTIPOINT ZM ((30 10 40 300))".to_string(),
            },
            // MULTILINESTRING EMPTY
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![0x01, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                wkt_string: "MULTILINESTRING EMPTY".to_string(),
            },
            // MULTILINESTRING ((30 10, 10 30, 40 40))
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x01, 0x05, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x00,
                    0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
                    0x40,
                ],
                wkt_string: "MULTILINESTRING ((30 10, 10 30, 40 40))".to_string(),
            },
            // MULTILINESTRING Z ((30 10 40, 10 30 40, 40 40 80))
            WkbTestCase {
                dimension: Dimension::Xyz,
                wkb_bytes: vec![
                    0x01, 0xed, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xea, 0x03, 0x00,
                    0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0x40,
                ],
                wkt_string: "MULTILINESTRING Z ((30 10 40, 10 30 40, 40 40 80))".to_string(),
            },
            // MULTILINESTRING M ((30 10 300, 10 30 300, 40 40 1600))
            WkbTestCase {
                dimension: Dimension::Xym,
                wkb_bytes: vec![
                    0x01, 0xd5, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xd2, 0x07, 0x00,
                    0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40,
                ],
                wkt_string: "MULTILINESTRING M ((30 10 300, 10 30 300, 40 40 1600))".to_string(),
            },
            // MULTILINESTRING ZM ((30 10 40 300, 10 30 40 300, 40 40 80 1600))
            WkbTestCase {
                dimension: Dimension::Xyzm,
                wkb_bytes: vec![
                    0x01, 0xbd, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xba, 0x0b, 0x00,
                0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e,
                0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00,
                0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x54, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40
                ],
                wkt_string: "MULTILINESTRING ZM ((30 10 40 300, 10 30 40 300, 40 40 80 1600))".to_string(),
            },
            // MULTIPOLYGON EMPTY
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![0x01, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                wkt_string: "MULTIPOLYGON EMPTY".to_string(),
            },
            // MULTIPOLYGON (((30 10, 40 40, 20 40, 10 20, 30 10)))
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x01, 0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x03, 0x00, 0x00,
                    0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40,
                ],
                wkt_string: "MULTIPOLYGON (((30 10, 40 40, 20 40, 10 20, 30 10)))".to_string(),
            },
            // MULTIPOLYGON Z (((30 10 40, 40 40 80, 20 40 60, 10 20 30, 30 10 40)))
            WkbTestCase {
                dimension: Dimension::Xyz,
                wkb_bytes: vec![
                    0x01, 0xee, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xeb, 0x03, 0x00,
                    0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x54, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x4e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40,
                ],
                wkt_string: "MULTIPOLYGON Z (((30 10 40, 40 40 80, 20 40 60, 10 20 30, 30 10 40)))".to_string(),
            },
            // MULTIPOLYGON M (((30 10 300, 40 40 1600, 20 40 800, 10 20 200, 30 10 300)))
            WkbTestCase {
                dimension: Dimension::Xym,
                wkb_bytes: vec![
                    0x01, 0xd6, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xd3, 0x07, 0x00,
                    0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x99, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x89, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x69,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40,
                ],
                wkt_string: "MULTIPOLYGON M (((30 10 300, 40 40 1600, 20 40 800, 10 20 200, 30 10 300)))".to_string(),
            },
            // MULTIPOLYGON ZM (((30 10 40 300, 40 40 80 1600, 20 40 60 800, 10 20 30 200, 30
            // 10 40 300)))
            WkbTestCase {
                dimension: Dimension::Xyzm,
                wkb_bytes: vec![
                    0x01, 0xbe, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xbb, 0x0b, 0x00,
                    0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0,
                    0x72, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x99, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x4e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x89,
                    0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x34, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x69, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x3e, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x44, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40,
                ],
                wkt_string: "MULTIPOLYGON ZM (((30 10 40 300, 40 40 80 1600, 20 40 60 800, 10 20 30 200, 30 10 40 300)))".to_string(),
            },
            // GEOMETRYCOLLECTION EMPTY
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![0x01, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                wkt_string: "GEOMETRYCOLLECTION EMPTY".to_string(),
            },
            // GEOMETRYCOLLECTION (POINT (30 10))
            WkbTestCase {
                dimension: Dimension::Xy,
                wkb_bytes: vec![
                    0x01, 0x07, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40,
                ],
                wkt_string: "GEOMETRYCOLLECTION (POINT (30 10))".to_string(),
            },
            // GEOMETRYCOLLECTION Z (POINT Z (30 10 40))
            WkbTestCase {
                dimension: Dimension::Xyz,
                wkb_bytes: vec![
                    0x01, 0xef, 0x03, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xe9, 0x03, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40,
                ],
                wkt_string: "GEOMETRYCOLLECTION Z (POINT Z (30 10 40))".to_string(),
            },
            // GEOMETRYCOLLECTION M (POINT M (30 10 300))
            WkbTestCase {
                dimension: Dimension::Xym,
                wkb_bytes: vec![
                    0x01, 0xd7, 0x07, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xd1, 0x07, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40,
                ],
                wkt_string: "GEOMETRYCOLLECTION M (POINT M (30 10 300))".to_string(),
            },
            // GEOMETRYCOLLECTION ZM (POINT ZM (30 10 40 300))
            WkbTestCase {
                dimension: Dimension::Xyzm,
                wkb_bytes: vec![
                    0x01, 0xbf, 0x0b, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0xb9, 0x0b, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3e, 0x40, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x24, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x44, 0x40, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0xc0, 0x72, 0x40,
                ],
                wkt_string: "GEOMETRYCOLLECTION ZM (POINT ZM (30 10 40 300))".to_string(),
            },
        ]
    }

    #[test]
    fn test_using_comprehensive_cases() {
        let factory = GEOSWkbFactory::new();
        let test_cases = get_wkb_test_cases();
        for test_case in test_cases {
            let wkb = read_wkb(&test_case.wkb_bytes).unwrap();
            let geos_geom = factory.create(&wkb).unwrap();
            let wkt_from_geos = geos_geom.to_wkt().unwrap();
            assert_eq!(
                wkt_from_geos, test_case.wkt_string,
                "Failed for test case {}",
                test_case.wkt_string
            );
        }
    }
}
