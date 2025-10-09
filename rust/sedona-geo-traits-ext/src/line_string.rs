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
// Extend LineStringTrait traits for the `geo-traits` crate

use geo_traits::{GeometryTrait, LineStringTrait, UnimplementedLineString};
use geo_types::{Coord, CoordNum, Line, LineString, Triangle};

use crate::{CoordTraitExt, GeoTraitExtWithTypeTag, LineStringTag};

/// Additional convenience methods for [`LineStringTrait`] implementers that mirror `geo-types`.
pub trait LineStringTraitExt:
    LineStringTrait + GeoTraitExtWithTypeTag<Tag = LineStringTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    type CoordTypeExt<'a>: 'a + CoordTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Returns the coordinate at the provided index.
    fn coord_ext(&self, i: usize) -> Option<Self::CoordTypeExt<'_>>;

    /// Returns a coordinate by index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `i` is a valid index less than the number of coordinates.
    /// Otherwise, this function may cause undefined behavior.
    unsafe fn coord_unchecked_ext(&self, i: usize) -> Self::CoordTypeExt<'_>;

    /// Returns an iterator over all coordinates as extension trait instances.
    fn coords_ext(&self) -> impl Iterator<Item = Self::CoordTypeExt<'_>>;

    /// Returns a coordinate by index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `i` is a valid index less than the number of coordinates.
    /// Otherwise, this function may cause undefined behavior.
    #[inline]
    unsafe fn geo_coord_unchecked(&self, i: usize) -> Coord<Self::T> {
        self.coord_unchecked_ext(i).geo_coord()
    }

    /// Return an iterator yielding one [`Line`] for each line segment
    /// in the [`LineString`][`geo_types::LineString`].
    #[inline]
    fn lines(&'_ self) -> impl ExactSizeIterator<Item = Line<<Self as GeometryTrait>::T>> + '_ {
        let num_coords = self.num_coords();
        (0..num_coords.saturating_sub(1)).map(|i| unsafe {
            let coord1 = self.geo_coord_unchecked(i);
            let coord2 = self.geo_coord_unchecked(i + 1);
            Line::new(coord1, coord2)
        })
    }

    /// Return an iterator yielding one [`Line`] for each line segment in the [`LineString`][`geo_types::LineString`],
    /// starting from the **end** point of the LineString, working towards the start.
    ///
    /// Note: This is like [`Self::lines`], but the sequence **and** the orientation of
    /// segments are reversed.
    #[inline]
    fn rev_lines(&'_ self) -> impl ExactSizeIterator<Item = Line<<Self as GeometryTrait>::T>> + '_ {
        let num_coords = self.num_coords();
        (1..num_coords).rev().map(|i| unsafe {
            let coord1 = self.geo_coord_unchecked(i);
            let coord2 = self.geo_coord_unchecked(i - 1);
            Line::new(coord2, coord1)
        })
    }

    /// An iterator which yields the coordinates of a [`LineString`][`geo_types::LineString`] as [Triangle]s
    #[inline]
    fn triangles(
        &'_ self,
    ) -> impl ExactSizeIterator<Item = Triangle<<Self as GeometryTrait>::T>> + '_ {
        let num_coords = self.num_coords();
        let end = num_coords.saturating_sub(2);
        (0..end).map(|i| unsafe {
            let coord1 = self.geo_coord_unchecked(i);
            let coord2 = self.geo_coord_unchecked(i + 1);
            let coord3 = self.geo_coord_unchecked(i + 2);
            Triangle::new(coord1, coord2, coord3)
        })
    }

    /// Returns an iterator yielding the coordinates of this line string as [`geo_types::Coord`] values.
    #[inline]
    fn coord_iter(&self) -> impl Iterator<Item = Coord<<Self as GeometryTrait>::T>> {
        self.coords_ext().map(|c| c.geo_coord())
    }

    #[inline]
    /// Returns true when the line string is closed (its first and last coordinates are equal).
    fn is_closed(&self) -> bool {
        let num_coords = self.num_coords();
        if num_coords <= 1 {
            true
        } else {
            let (first, last) = unsafe {
                (
                    self.geo_coord_unchecked(0),
                    self.geo_coord_unchecked(num_coords - 1),
                )
            };
            first == last
        }
    }
}

#[macro_export]
/// Forwards [`LineStringTraitExt`] methods to an underlying [`LineStringTrait`] implementation.
macro_rules! forward_line_string_trait_ext_funcs {
    () => {
        type CoordTypeExt<'__l_inner>
            = <Self as LineStringTrait>::CoordType<'__l_inner>
        where
            Self: '__l_inner;

        #[inline]
        fn coord_ext(&self, i: usize) -> Option<Self::CoordTypeExt<'_>> {
            <Self as LineStringTrait>::coord(self, i)
        }

        #[inline]
        unsafe fn coord_unchecked_ext(&self, i: usize) -> Self::CoordTypeExt<'_> {
            <Self as LineStringTrait>::coord_unchecked(self, i)
        }

        #[inline]
        fn coords_ext(&self) -> impl Iterator<Item = Self::CoordTypeExt<'_>> {
            <Self as LineStringTrait>::coords(self)
        }
    };
}

impl<T> LineStringTraitExt for LineString<T>
where
    T: CoordNum,
{
    forward_line_string_trait_ext_funcs!();

    unsafe fn geo_coord_unchecked(&self, i: usize) -> Coord<Self::T> {
        *self.0.get_unchecked(i)
    }

    // Delegate to the `geo-types` implementation for less performance overhead
    fn lines(&'_ self) -> impl ExactSizeIterator<Item = Line<<Self as GeometryTrait>::T>> + '_ {
        self.lines()
    }

    fn rev_lines(&'_ self) -> impl ExactSizeIterator<Item = Line<<Self as GeometryTrait>::T>> + '_ {
        self.rev_lines()
    }

    fn triangles(
        &'_ self,
    ) -> impl ExactSizeIterator<Item = Triangle<<Self as GeometryTrait>::T>> + '_ {
        self.triangles()
    }

    fn is_closed(&self) -> bool {
        self.is_closed()
    }

    fn coord_iter(&self) -> impl Iterator<Item = Coord<<Self as GeometryTrait>::T>> {
        self.0.iter().copied()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for LineString<T> {
    type Tag = LineStringTag;
}

impl<T> LineStringTraitExt for &LineString<T>
where
    T: CoordNum,
{
    forward_line_string_trait_ext_funcs!();

    unsafe fn geo_coord_unchecked(&self, i: usize) -> Coord<Self::T> {
        *self.0.get_unchecked(i)
    }

    // Delegate to the `geo-types` implementation for less performance overhead
    fn lines(&'_ self) -> impl ExactSizeIterator<Item = Line<<Self as GeometryTrait>::T>> + '_ {
        (*self).lines()
    }

    fn rev_lines(&'_ self) -> impl ExactSizeIterator<Item = Line<<Self as GeometryTrait>::T>> + '_ {
        (*self).rev_lines()
    }

    fn triangles(
        &'_ self,
    ) -> impl ExactSizeIterator<Item = Triangle<<Self as GeometryTrait>::T>> + '_ {
        (*self).triangles()
    }

    fn is_closed(&self) -> bool {
        (*self).is_closed()
    }

    fn coord_iter(&self) -> impl Iterator<Item = Coord<<Self as GeometryTrait>::T>> {
        self.0.iter().copied()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &LineString<T> {
    type Tag = LineStringTag;
}

impl<T> LineStringTraitExt for UnimplementedLineString<T>
where
    T: CoordNum,
{
    forward_line_string_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedLineString<T> {
    type Tag = LineStringTag;
}
