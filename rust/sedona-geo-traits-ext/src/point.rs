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
// Extend PointTrait traits for the `geo-traits` crate

use geo_traits::{CoordTrait, GeometryTrait, PointTrait, UnimplementedPoint};
use geo_types::{Coord, CoordNum, Point};

use crate::{CoordTraitExt, GeoTraitExtWithTypeTag, PointTag};

/// Extension methods that expose `geo-types` conveniences for [`PointTrait`] implementers.
pub trait PointTraitExt: PointTrait + GeoTraitExtWithTypeTag<Tag = PointTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    type CoordTypeExt<'a>: 'a + CoordTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Returns the underlying coordinate view for this point, if available.
    fn coord_ext(&self) -> Option<Self::CoordTypeExt<'_>>;

    #[inline]
    /// Converts the point into a concrete [`geo_types::Point`].
    fn geo_point(&self) -> Option<Point<<Self as GeometryTrait>::T>> {
        self.coord_ext()
            .map(|coord| Point::new(coord.x(), coord.y()))
    }

    #[inline]
    /// Converts the point into a concrete [`geo_types::Coord`].
    fn geo_coord(&self) -> Option<Coord<<Self as GeometryTrait>::T>> {
        self.coord_ext().map(|coord| coord.geo_coord())
    }
}

#[macro_export]
/// Forwards [`PointTraitExt`] methods to the wrapped [`PointTrait`] implementation.
macro_rules! forward_point_trait_ext_funcs {
    () => {
        type CoordTypeExt<'__l_inner>
            = <Self as PointTrait>::CoordType<'__l_inner>
        where
            Self: '__l_inner;

        #[inline]
        fn coord_ext(&self) -> Option<Self::CoordTypeExt<'_>> {
            <Self as PointTrait>::coord(self)
        }
    };
}

impl<T> PointTraitExt for Point<T>
where
    T: CoordNum,
{
    forward_point_trait_ext_funcs!();

    fn geo_point(&self) -> Option<Point<T>> {
        Some(*self)
    }

    fn geo_coord(&self) -> Option<Coord<T>> {
        Some(self.0)
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for Point<T> {
    type Tag = PointTag;
}

impl<T> PointTraitExt for &Point<T>
where
    T: CoordNum,
{
    forward_point_trait_ext_funcs!();

    fn geo_point(&self) -> Option<Point<T>> {
        Some(**self)
    }

    fn geo_coord(&self) -> Option<Coord<T>> {
        Some(self.0)
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &Point<T> {
    type Tag = PointTag;
}

impl<T> PointTraitExt for UnimplementedPoint<T>
where
    T: CoordNum,
{
    forward_point_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedPoint<T> {
    type Tag = PointTag;
}
