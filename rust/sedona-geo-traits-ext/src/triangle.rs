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
// Extend TriangleTrait traits for the `geo-traits` crate

use geo_traits::{GeometryTrait, TriangleTrait, UnimplementedTriangle};
use geo_types::{polygon, Coord, CoordNum, Line, Polygon, Triangle};

use crate::{CoordTraitExt, GeoTraitExtWithTypeTag, TriangleTag};

/// Extension trait that augments [`geo_traits::TriangleTrait`] with convenient
/// coordinate accessors and adapters.
pub trait TriangleTraitExt: TriangleTrait + GeoTraitExtWithTypeTag<Tag = TriangleTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    /// Extension-aware coordinate type returned from triangle accessors.
    type CoordTypeExt<'a>: 'a + CoordTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Returns the first vertex with the extension trait applied.
    fn first_ext(&self) -> Self::CoordTypeExt<'_>;
    /// Returns the second vertex with the extension trait applied.
    fn second_ext(&self) -> Self::CoordTypeExt<'_>;
    /// Returns the third vertex with the extension trait applied.
    fn third_ext(&self) -> Self::CoordTypeExt<'_>;
    /// Returns all three vertices as extension-aware coordinates.
    fn coords_ext(&self) -> [Self::CoordTypeExt<'_>; 3];

    #[inline]
    /// Returns the first vertex as a `geo-types::Coord`.
    fn first_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.first_ext().geo_coord()
    }

    #[inline]
    /// Returns the second vertex as a `geo-types::Coord`.
    fn second_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.second_ext().geo_coord()
    }

    #[inline]
    /// Returns the third vertex as a `geo-types::Coord`.
    fn third_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.third_ext().geo_coord()
    }

    #[inline]
    /// Returns the triangle vertices as an array of coordinates.
    fn to_array(&self) -> [Coord<<Self as GeometryTrait>::T>; 3] {
        [self.first_coord(), self.second_coord(), self.third_coord()]
    }

    #[inline]
    /// Returns the three edges as line segments in traversal order.
    fn to_lines(&self) -> [Line<<Self as GeometryTrait>::T>; 3] {
        [
            Line::new(self.first_coord(), self.second_coord()),
            Line::new(self.second_coord(), self.third_coord()),
            Line::new(self.third_coord(), self.first_coord()),
        ]
    }

    #[inline]
    /// Converts the triangle into a polygon whose shell walks the triangle vertices.
    fn to_polygon(&self) -> Polygon<<Self as GeometryTrait>::T> {
        polygon![
            self.first_coord(),
            self.second_coord(),
            self.third_coord(),
            self.first_coord(),
        ]
    }

    #[inline]
    /// Iterates over the triangle vertices as coordinates.
    fn coord_iter(&self) -> impl Iterator<Item = Coord<<Self as GeometryTrait>::T>> {
        [self.first_coord(), self.second_coord(), self.third_coord()].into_iter()
    }
}

#[macro_export]
/// Forwards [`TriangleTraitExt`] methods to the underlying
/// [`geo_traits::TriangleTrait`] implementation while returning extension trait
/// wrappers.
macro_rules! forward_triangle_trait_ext_funcs {
    () => {
        type CoordTypeExt<'__l_inner>
            = <Self as TriangleTrait>::CoordType<'__l_inner>
        where
            Self: '__l_inner;

        #[inline]
        fn first_ext(&self) -> Self::CoordTypeExt<'_> {
            <Self as TriangleTrait>::first(self)
        }

        #[inline]
        fn second_ext(&self) -> Self::CoordTypeExt<'_> {
            <Self as TriangleTrait>::second(self)
        }

        #[inline]
        fn third_ext(&self) -> Self::CoordTypeExt<'_> {
            <Self as TriangleTrait>::third(self)
        }

        #[inline]
        fn coords_ext(&self) -> [Self::CoordTypeExt<'_>; 3] {
            [self.first_ext(), self.second_ext(), self.third_ext()]
        }
    };
}

impl<T> TriangleTraitExt for Triangle<T>
where
    T: CoordNum,
{
    forward_triangle_trait_ext_funcs!();

    fn first_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.0
    }

    fn second_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.1
    }

    fn third_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.2
    }

    fn to_array(&self) -> [Coord<<Self as GeometryTrait>::T>; 3] {
        self.to_array()
    }

    fn to_lines(&self) -> [Line<<Self as GeometryTrait>::T>; 3] {
        self.to_lines()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for Triangle<T> {
    type Tag = TriangleTag;
}

impl<T> TriangleTraitExt for &Triangle<T>
where
    T: CoordNum,
{
    forward_triangle_trait_ext_funcs!();

    fn first_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.0
    }

    fn second_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.1
    }

    fn third_coord(&self) -> Coord<<Self as GeometryTrait>::T> {
        self.2
    }

    fn to_array(&self) -> [Coord<<Self as GeometryTrait>::T>; 3] {
        (*self).to_array()
    }

    fn to_lines(&self) -> [Line<<Self as GeometryTrait>::T>; 3] {
        (*self).to_lines()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &Triangle<T> {
    type Tag = TriangleTag;
}

impl<T> TriangleTraitExt for UnimplementedTriangle<T>
where
    T: CoordNum,
{
    forward_triangle_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedTriangle<T> {
    type Tag = TriangleTag;
}
