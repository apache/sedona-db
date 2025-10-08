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
use std::marker::PhantomData;

use crate::*;
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use geo_types::{Coord as GeoCoord, Line};
use wkb::reader::{
    Coord, Dimension, GeometryCollection, LineString, LinearRing, MultiLineString, MultiPoint,
    MultiPolygon, Point, Polygon, Wkb,
};
use wkb::Endianness;

// ┌──────────────────────────────────────────────────────────┐
// │ Coord                                                    │
// └──────────────────────────────────────────────────────────┘
impl CoordTraitExt for Coord<'_> {
    #[inline]
    fn geo_coord(&self) -> geo_types::Coord<f64> {
        let coord_slice = self.coord_slice();
        unsafe {
            let x_bytes = std::slice::from_raw_parts(coord_slice.as_ptr(), 8);
            let y_bytes = std::slice::from_raw_parts(coord_slice.as_ptr().add(8), 8);
            match self.byte_order() {
                Endianness::BigEndian => {
                    let x = BigEndian::read_f64(x_bytes);
                    let y = BigEndian::read_f64(y_bytes);
                    geo_types::Coord { x, y }
                }
                Endianness::LittleEndian => {
                    let x = LittleEndian::read_f64(x_bytes);
                    let y = LittleEndian::read_f64(y_bytes);
                    geo_types::Coord { x, y }
                }
            }
        }
    }
}

impl GeoTraitExtWithTypeTag for Coord<'_> {
    type Tag = CoordTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ Point                                                    │
// └──────────────────────────────────────────────────────────┘

impl PointTraitExt for Point<'_> {
    forward_point_trait_ext_funcs!();
}

impl GeoTraitExtWithTypeTag for Point<'_> {
    type Tag = PointTag;
}

impl PointTraitExt for &Point<'_> {
    forward_point_trait_ext_funcs!();
}

impl GeoTraitExtWithTypeTag for &Point<'_> {
    type Tag = PointTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ LineString                                               │
// └──────────────────────────────────────────────────────────┘

impl LineStringTraitExt for LineString<'_> {
    forward_line_string_trait_ext_funcs!();

    #[inline(always)]
    fn lines(&'_ self) -> impl ExactSizeIterator<Item = Line<f64>> + '_ {
        let buf = self.coords_slice();
        let dim_size = dimension_size(self.dimension());
        let num_coords = self.num_coords();
        match self.byte_order() {
            Endianness::LittleEndian => {
                EndianLineIter::LE(LineIter::new(buf, num_coords, dim_size))
            }
            Endianness::BigEndian => EndianLineIter::BE(LineIter::new(buf, num_coords, dim_size)),
        }
    }

    #[inline(always)]
    fn coord_iter(&self) -> impl Iterator<Item = GeoCoord<f64>> {
        let buf = self.coords_slice();
        let dim_size = dimension_size(self.dimension());
        let num_coords = self.num_coords();
        match self.byte_order() {
            Endianness::LittleEndian => {
                EndianCoordIter::LE(CoordIter::new(buf, num_coords, dim_size))
            }
            Endianness::BigEndian => EndianCoordIter::BE(CoordIter::new(buf, num_coords, dim_size)),
        }
    }
}

impl GeoTraitExtWithTypeTag for LineString<'_> {
    type Tag = LineStringTag;
}

impl LineStringTraitExt for &LineString<'_> {
    forward_line_string_trait_ext_funcs!();

    #[inline(always)]
    fn lines(&'_ self) -> impl ExactSizeIterator<Item = Line<f64>> + '_ {
        (*self).lines()
    }

    #[inline(always)]
    fn coord_iter(&self) -> impl Iterator<Item = GeoCoord<f64>> {
        (*self).coord_iter()
    }
}

impl GeoTraitExtWithTypeTag for &LineString<'_> {
    type Tag = LineStringTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ LinearRing                                               │
// └──────────────────────────────────────────────────────────┘

impl LineStringTraitExt for LinearRing<'_> {
    forward_line_string_trait_ext_funcs!();

    #[inline(always)]
    fn lines(&'_ self) -> impl ExactSizeIterator<Item = Line<f64>> + '_ {
        let buf = self.coords_slice();
        let dim_size = dimension_size(self.dimension());
        let num_coords = self.num_coords();
        match self.byte_order() {
            Endianness::LittleEndian => {
                EndianLineIter::LE(LineIter::new(buf, num_coords, dim_size))
            }
            Endianness::BigEndian => EndianLineIter::BE(LineIter::new(buf, num_coords, dim_size)),
        }
    }

    #[inline(always)]
    fn coord_iter(&self) -> impl Iterator<Item = GeoCoord<f64>> {
        let buf = self.coords_slice();
        let dim_size = dimension_size(self.dimension());
        let num_coords = self.num_coords();
        match self.byte_order() {
            Endianness::LittleEndian => {
                EndianCoordIter::LE(CoordIter::new(buf, num_coords, dim_size))
            }
            Endianness::BigEndian => EndianCoordIter::BE(CoordIter::new(buf, num_coords, dim_size)),
        }
    }
}

impl GeoTraitExtWithTypeTag for LinearRing<'_> {
    type Tag = LineStringTag;
}

impl LineStringTraitExt for &LinearRing<'_> {
    forward_line_string_trait_ext_funcs!();

    #[inline(always)]
    fn lines(&'_ self) -> impl ExactSizeIterator<Item = Line<f64>> + '_ {
        (*self).lines()
    }

    #[inline(always)]
    fn coord_iter(&self) -> impl Iterator<Item = GeoCoord<f64>> {
        (*self).coord_iter()
    }
}

impl GeoTraitExtWithTypeTag for &LinearRing<'_> {
    type Tag = LineStringTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ Polygon                                                  │
// └──────────────────────────────────────────────────────────┘

impl PolygonTraitExt for Polygon<'_> {
    forward_polygon_trait_ext_funcs!();
}

impl GeoTraitExtWithTypeTag for Polygon<'_> {
    type Tag = PolygonTag;
}

impl PolygonTraitExt for &Polygon<'_> {
    forward_polygon_trait_ext_funcs!();
}

impl GeoTraitExtWithTypeTag for &Polygon<'_> {
    type Tag = PolygonTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ MultiPoint                                               │
// └──────────────────────────────────────────────────────────┘

impl MultiPointTraitExt for MultiPoint<'_> {
    forward_multi_point_trait_ext_funcs!();
}

impl GeoTraitExtWithTypeTag for MultiPoint<'_> {
    type Tag = MultiPointTag;
}

impl<'a, 'b> MultiPointTraitExt for &'b MultiPoint<'a>
where
    'a: 'b,
{
    forward_multi_point_trait_ext_funcs!();
}

impl<'a, 'b> GeoTraitExtWithTypeTag for &'b MultiPoint<'a>
where
    'a: 'b,
{
    type Tag = MultiPointTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ MultiLineString                                          │
// └──────────────────────────────────────────────────────────┘

impl MultiLineStringTraitExt for MultiLineString<'_> {
    forward_multi_line_string_trait_ext_funcs!();
}

impl GeoTraitExtWithTypeTag for MultiLineString<'_> {
    type Tag = MultiLineStringTag;
}

impl<'a, 'b> MultiLineStringTraitExt for &'b MultiLineString<'a>
where
    'a: 'b,
{
    forward_multi_line_string_trait_ext_funcs!();
}

impl<'a, 'b> GeoTraitExtWithTypeTag for &'b MultiLineString<'a>
where
    'a: 'b,
{
    type Tag = MultiLineStringTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ MultiPolygon                                             │
// └──────────────────────────────────────────────────────────┘

impl MultiPolygonTraitExt for MultiPolygon<'_> {
    forward_multi_polygon_trait_ext_funcs!();
}

impl GeoTraitExtWithTypeTag for MultiPolygon<'_> {
    type Tag = MultiPolygonTag;
}

impl<'a, 'b> MultiPolygonTraitExt for &'b MultiPolygon<'a>
where
    'a: 'b,
{
    forward_multi_polygon_trait_ext_funcs!();
}

impl<'a, 'b> GeoTraitExtWithTypeTag for &'b MultiPolygon<'a>
where
    'a: 'b,
{
    type Tag = MultiPolygonTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ GeometryCollection                                       │
// └──────────────────────────────────────────────────────────┘

impl GeometryCollectionTraitExt for GeometryCollection<'_> {
    forward_geometry_collection_trait_ext_funcs!();
}

impl GeoTraitExtWithTypeTag for GeometryCollection<'_> {
    type Tag = GeometryCollectionTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ Wkb/Geometry                                             │
// └──────────────────────────────────────────────────────────┘

impl<'a> GeometryTraitExt for Wkb<'a> {
    forward_geometry_trait_ext_funcs!(f64);

    type InnerGeometryRef<'b>
        = &'b Wkb<'a>
    where
        Self: 'b;

    #[inline]
    fn geometry_ext(&self, i: usize) -> Option<Self::InnerGeometryRef<'_>> {
        let GeometryType::GeometryCollection(gc) = self.as_type() else {
            return None;
        };
        gc.geometry(i)
    }

    #[inline]
    unsafe fn geometry_unchecked_ext(&self, i: usize) -> Self::InnerGeometryRef<'_> {
        let GeometryType::GeometryCollection(gc) = self.as_type() else {
            panic!("Called geometry_unchecked_ext on a non-GeometryCollection geometry");
        };
        gc.geometry_unchecked(i)
    }

    #[inline]
    fn geometries_ext(&self) -> impl Iterator<Item = Self::InnerGeometryRef<'_>> {
        let GeometryType::GeometryCollection(gc) = self.as_type() else {
            panic!("Called geometries_ext on a non-GeometryCollection geometry");
        };
        gc.geometries()
    }
}

impl<'a, 'b> GeometryTraitExt for &'b Wkb<'a>
where
    'a: 'b,
{
    forward_geometry_trait_ext_funcs!(f64);

    type InnerGeometryRef<'c>
        = &'b Wkb<'a>
    where
        Self: 'c;

    #[inline]
    fn geometry_ext(&self, i: usize) -> Option<Self::InnerGeometryRef<'_>> {
        (*self).geometry_ext(i)
    }

    #[inline]
    unsafe fn geometry_unchecked_ext(&self, i: usize) -> Self::InnerGeometryRef<'_> {
        (*self).geometry_unchecked_ext(i)
    }

    #[inline]
    fn geometries_ext(&self) -> impl Iterator<Item = Self::InnerGeometryRef<'_>> {
        (*self).geometries_ext()
    }
}

impl GeoTraitExtWithTypeTag for Wkb<'_> {
    type Tag = GeometryTag;
}

impl<'a, 'b> GeoTraitExtWithTypeTag for &'b Wkb<'a>
where
    'a: 'b,
{
    type Tag = GeometryTag;
}

// ┌──────────────────────────────────────────────────────────┐
// │ Iterators                                                │
// └──────────────────────────────────────────────────────────┘

/// Iterator over coordinates in a WKB buffer using a compile-time endianness.
pub struct CoordIter<'a, B: ByteOrder> {
    buf: &'a [u8],
    current_offset: usize,
    remaining: usize,
    dim_size: usize,
    _marker: PhantomData<B>,
}

impl<'a, B: ByteOrder> CoordIter<'a, B> {
    #[inline]
    /// Creates a new coordinate iterator over the provided buffer.
    pub fn new(buf: &'a [u8], num_coords: usize, dim_size: usize) -> Self {
        Self {
            buf,
            current_offset: 0,
            remaining: num_coords,
            dim_size,
            _marker: PhantomData,
        }
    }
}

impl<B: ByteOrder> Iterator for CoordIter<'_, B> {
    type Item = GeoCoord<f64>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // SAFETY: We're reading raw memory from the buffer at calculated offsets.
        // This assumes the buffer contains valid data and offsets are within bounds.
        let coord = unsafe {
            let x_bytes = std::slice::from_raw_parts(self.buf.as_ptr().add(self.current_offset), 8);
            let y_bytes =
                std::slice::from_raw_parts(self.buf.as_ptr().add(self.current_offset + 8), 8);
            let x = B::read_f64(x_bytes);
            let y = B::read_f64(y_bytes);
            GeoCoord { x, y }
        };

        self.current_offset += self.dim_size * 8;
        self.remaining -= 1;
        Some(coord)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<B: ByteOrder> ExactSizeIterator for CoordIter<'_, B> {}

/// Iterator over line segments derived from sequential coordinates in a WKB buffer.
pub struct LineIter<'a, B: ByteOrder> {
    coord_iter: CoordIter<'a, B>,
    prev_coord: Option<GeoCoord<f64>>,
}

impl<'a, B: ByteOrder> LineIter<'a, B> {
    #[inline]
    /// Creates a new line iterator over the provided buffer.
    pub fn new(buf: &'a [u8], num_coords: usize, dim_size: usize) -> Self {
        Self {
            coord_iter: CoordIter::new(buf, num_coords, dim_size),
            prev_coord: None,
        }
    }
}

impl<B: ByteOrder> Iterator for LineIter<'_, B> {
    type Item = Line<f64>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let current_coord = self.coord_iter.next()?;

        match self.prev_coord {
            Some(prev_coord) => {
                let line = Line::new(prev_coord, current_coord);
                self.prev_coord = Some(current_coord);
                Some(line)
            }
            None => {
                // Grab the next coordinate to form the first line segment
                let next_coord = self.coord_iter.next()?;
                let line = Line::new(current_coord, next_coord);
                self.prev_coord = Some(next_coord);
                Some(line)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min, max) = self.coord_iter.size_hint();
        (min.saturating_sub(1), max.map(|m| m.saturating_sub(1)))
    }
}

impl<B: ByteOrder> ExactSizeIterator for LineIter<'_, B> {}

/// Wrapper around [`CoordIter`] that selects the concrete endianness at runtime.
///
/// The dispatch in the iterator methods is static and can be inlined by the
/// compiler, so callers do not pay the cost of dynamic allocation.
pub enum EndianCoordIter<'a> {
    LE(CoordIter<'a, LittleEndian>),
    BE(CoordIter<'a, BigEndian>),
}

impl Iterator for EndianCoordIter<'_> {
    type Item = GeoCoord<f64>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // We rely on compiler optimization to hoist the match out of the loop, so that
        // there's no performance overhead of checking the endianness inside the loop.
        match self {
            EndianCoordIter::LE(iter) => iter.next(),
            EndianCoordIter::BE(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            EndianCoordIter::LE(iter) => iter.size_hint(),
            EndianCoordIter::BE(iter) => iter.size_hint(),
        }
    }
}

/// Wrapper around [`LineIter`] that selects the concrete endianness at runtime.
pub enum EndianLineIter<'a> {
    LE(LineIter<'a, LittleEndian>),
    BE(LineIter<'a, BigEndian>),
}

impl Iterator for EndianLineIter<'_> {
    type Item = Line<f64>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            EndianLineIter::LE(iter) => iter.next(),
            EndianLineIter::BE(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            EndianLineIter::LE(iter) => iter.size_hint(),
            EndianLineIter::BE(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for EndianLineIter<'_> {}

// ┌──────────────────────────────────────────────────────────┐
// │ Utils.                                                   │
// └──────────────────────────────────────────────────────────┘
/// Returns the dimensionality (number of ordinates) represented by a WKB [`Dimension`].
fn dimension_size(dim: Dimension) -> usize {
    match dim {
        Dimension::Xy => 2,
        Dimension::Xyz | Dimension::Xym => 3,
        Dimension::Xyzm => 4,
    }
}
