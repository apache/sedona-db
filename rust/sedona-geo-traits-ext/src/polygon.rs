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
// Extend PolygonTrait traits for the `geo-traits` crate

use geo_traits::{GeometryTrait, PolygonTrait, UnimplementedPolygon};
use geo_types::{CoordNum, Polygon};

use crate::{GeoTraitExtWithTypeTag, LineStringTraitExt, PolygonTag};

/// Extension trait that augments [`geo_traits::PolygonTrait`] with
/// extension-aware accessors over exterior and interior rings.
///
/// Implementations are able to return ring references that also implement
/// [`LineStringTraitExt`], bringing parity with `geo-types::Polygon` while
/// remaining ergonomic through trait objects.
pub trait PolygonTraitExt: PolygonTrait + GeoTraitExtWithTypeTag<Tag = PolygonTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    /// Type of ring returned from accessor methods, wrapped with
    /// [`LineStringTraitExt`].
    type RingTypeExt<'a>: 'a + LineStringTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Returns the exterior ring with the extension trait applied.
    fn exterior_ext(&self) -> Option<Self::RingTypeExt<'_>>;

    /// Iterates over each interior ring with the extension trait applied.
    fn interiors_ext(
        &self,
    ) -> impl DoubleEndedIterator + ExactSizeIterator<Item = Self::RingTypeExt<'_>>;

    /// Returns the `i`th interior ring, if present, wrapped with the extension trait.
    fn interior_ext(&self, i: usize) -> Option<Self::RingTypeExt<'_>>;

    /// Returns an interior ring by index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `i` is a valid index less than the number of interior rings.
    /// Otherwise, this function may cause undefined behavior.
    unsafe fn interior_unchecked_ext(&self, i: usize) -> Self::RingTypeExt<'_>;
}

#[macro_export]
/// Forwards [`PolygonTraitExt`] methods to the underlying
/// [`geo_traits::PolygonTrait`] implementation while preserving the extension
/// trait wrappers.
macro_rules! forward_polygon_trait_ext_funcs {
    () => {
        type RingTypeExt<'__l_inner>
            = <Self as PolygonTrait>::RingType<'__l_inner>
        where
            Self: '__l_inner;

        #[inline]
        fn exterior_ext(&self) -> Option<Self::RingTypeExt<'_>> {
            <Self as PolygonTrait>::exterior(self)
        }

        #[inline]
        fn interiors_ext(
            &self,
        ) -> impl DoubleEndedIterator + ExactSizeIterator<Item = Self::RingTypeExt<'_>> {
            <Self as PolygonTrait>::interiors(self)
        }

        #[inline]
        fn interior_ext(&self, i: usize) -> Option<Self::RingTypeExt<'_>> {
            <Self as PolygonTrait>::interior(self, i)
        }

        #[inline]
        unsafe fn interior_unchecked_ext(&self, i: usize) -> Self::RingTypeExt<'_> {
            <Self as PolygonTrait>::interior_unchecked(self, i)
        }
    };
}

impl<T> PolygonTraitExt for Polygon<T>
where
    T: CoordNum,
{
    forward_polygon_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for Polygon<T> {
    type Tag = PolygonTag;
}

impl<T> PolygonTraitExt for &Polygon<T>
where
    T: CoordNum,
{
    forward_polygon_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &Polygon<T> {
    type Tag = PolygonTag;
}

impl<T> PolygonTraitExt for UnimplementedPolygon<T>
where
    T: CoordNum,
{
    forward_polygon_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedPolygon<T> {
    type Tag = PolygonTag;
}
