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
// Extend MultiPointTrait traits for the `geo-traits` crate

use geo_traits::{GeometryTrait, MultiPointTrait, UnimplementedMultiPoint};
use geo_types::{Coord, CoordNum, MultiPoint};

use crate::{CoordTraitExt, GeoTraitExtWithTypeTag, MultiPointTag, PointTraitExt};

/// Extension trait that augments [`geo_traits::MultiPointTrait`] with richer
/// ergonomics and accessors.
///
/// The trait keeps parity with the APIs provided by `geo-types::MultiPoint`
/// while still working with trait objects that only implement
/// [`geo_traits::MultiPointTrait`]. It also wires the geometry up with a
/// [`MultiPointTag`](crate::MultiPointTag) so the type can participate in the
/// shared `GeoTraitExtWithTypeTag` machinery.
pub trait MultiPointTraitExt:
    MultiPointTrait + GeoTraitExtWithTypeTag<Tag = MultiPointTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    /// Extension-aware point type returned from accessors on this multi point.
    type PointTypeExt<'a>: 'a + PointTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Returns the point at index `i`, wrapped in the extension trait.
    ///
    /// This mirrors [`geo_traits::MultiPointTrait::point`] but guarantees the
    /// returned point implements [`PointTraitExt`].
    fn point_ext(&self, i: usize) -> Option<Self::PointTypeExt<'_>>;

    /// Returns a point by index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `i` is a valid index less than the number of points.
    /// Otherwise, this function may cause undefined behavior.
    unsafe fn point_unchecked_ext(&self, i: usize) -> Self::PointTypeExt<'_>;

    /// Returns a coordinate by index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `i` is a valid index less than the number of points.
    /// Otherwise, this function may cause undefined behavior.
    /// Returns the coordinate at index `i` without bounds checking.
    ///
    /// This helper is primarily used by iterator adapters that need direct
    /// coordinate access while still honoring the [`PointTraitExt`] abstraction.
    ///
    /// # Safety
    /// The caller must ensure that `i` is a valid index less than the number of points.
    /// Otherwise, this function may cause undefined behavior.
    #[inline]
    unsafe fn geo_coord_unchecked(&self, i: usize) -> Option<Coord<<Self as GeometryTrait>::T>> {
        let point = unsafe { self.point_unchecked_ext(i) };
        point.coord_ext().map(|c| c.geo_coord())
    }

    /// Returns an iterator over all points, each wrapped in [`PointTraitExt`].
    fn points_ext(&self) -> impl DoubleEndedIterator<Item = Self::PointTypeExt<'_>>;

    /// Iterates over the coordinates contained in this multi point.
    ///
    /// For trait-based implementations this is derived from
    /// [`points_ext`](Self::points_ext), while concrete `geo-types::MultiPoint`
    /// instances provide a specialized iterator that avoids intermediate
    /// allocations.
    #[inline]
    fn coord_iter(&self) -> impl DoubleEndedIterator<Item = Coord<<Self as GeometryTrait>::T>> {
        self.points_ext().flat_map(|p| p.geo_coord())
    }
}

#[macro_export]
/// Forwards [`MultiPointTraitExt`] methods to the underlying
/// [`geo_traits::MultiPointTrait`] implementation while maintaining the
/// extension trait wrappers.
macro_rules! forward_multi_point_trait_ext_funcs {
    () => {
        type PointTypeExt<'__l_inner>
            = <Self as MultiPointTrait>::InnerPointType<'__l_inner>
        where
            Self: '__l_inner;

        #[inline]
        fn point_ext(&self, i: usize) -> Option<Self::PointTypeExt<'_>> {
            <Self as MultiPointTrait>::point(self, i)
        }

        #[inline]
        unsafe fn point_unchecked_ext(&self, i: usize) -> Self::PointTypeExt<'_> {
            <Self as MultiPointTrait>::point_unchecked(self, i)
        }

        #[inline]
        fn points_ext(&self) -> impl DoubleEndedIterator<Item = Self::PointTypeExt<'_>> {
            <Self as MultiPointTrait>::points(self)
        }
    };
}

impl<T> MultiPointTraitExt for MultiPoint<T>
where
    T: CoordNum,
{
    forward_multi_point_trait_ext_funcs!();

    /// Specialized coordinate accessor for `geo_types::MultiPoint`.
    unsafe fn geo_coord_unchecked(&self, i: usize) -> Option<Coord<T>> {
        Some(self.0.get_unchecked(i).0)
    }

    // Specialized implementation for geo_types::MultiPoint to reduce performance overhead
    fn coord_iter(&self) -> impl DoubleEndedIterator<Item = Coord<<Self as GeometryTrait>::T>> {
        self.0.iter().map(|p| p.0)
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for MultiPoint<T> {
    type Tag = MultiPointTag;
}

impl<T> MultiPointTraitExt for &MultiPoint<T>
where
    T: CoordNum,
{
    forward_multi_point_trait_ext_funcs!();

    /// Specialized coordinate accessor for `&geo_types::MultiPoint`.
    unsafe fn geo_coord_unchecked(&self, i: usize) -> Option<Coord<T>> {
        Some(self.0.get_unchecked(i).0)
    }

    // Specialized implementation for geo_types::MultiPoint to reduce performance overhead
    fn coord_iter(&self) -> impl DoubleEndedIterator<Item = Coord<<Self as GeometryTrait>::T>> {
        self.0.iter().map(|p| p.0)
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &MultiPoint<T> {
    type Tag = MultiPointTag;
}

impl<T> MultiPointTraitExt for UnimplementedMultiPoint<T>
where
    T: CoordNum,
{
    forward_multi_point_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedMultiPoint<T> {
    type Tag = MultiPointTag;
}
