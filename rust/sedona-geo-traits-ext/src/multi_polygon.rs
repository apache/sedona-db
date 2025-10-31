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
// Extend MultiPolygonTrait traits for the `geo-traits` crate

use geo_traits::{GeometryTrait, MultiPolygonTrait, UnimplementedMultiPolygon};
use geo_types::{CoordNum, MultiPolygon};

use crate::{GeoTraitExtWithTypeTag, MultiPolygonTag, PolygonTraitExt};

/// Extension trait that enriches [`geo_traits::MultiPolygonTrait`] with Sedona
/// conveniences.
///
/// Implementations can expose polygon members through the
/// [`PolygonTraitExt`] abstraction, ensuring consistent access to exterior and
/// interior rings regardless of the backing geometry type.
pub trait MultiPolygonTraitExt:
    MultiPolygonTrait + GeoTraitExtWithTypeTag<Tag = MultiPolygonTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    /// Extension-aware polygon type yielded by accessor methods.
    type PolygonTypeExt<'a>: 'a + PolygonTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Returns the polygon at index `i`, wrapped with [`PolygonTraitExt`].
    fn polygon_ext(&self, i: usize) -> Option<Self::PolygonTypeExt<'_>>;

    /// Returns a polygon by index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that `i` is a valid index less than the number of polygons.
    /// Otherwise, this function may cause undefined behavior.
    unsafe fn polygon_unchecked_ext(&self, i: usize) -> Self::PolygonTypeExt<'_>;

    /// Iterates over all polygon members, each wrapped with the extension trait.
    fn polygons_ext(&self) -> impl Iterator<Item = Self::PolygonTypeExt<'_>>;
}

#[macro_export]
/// Forwards [`MultiPolygonTraitExt`] methods to the underlying
/// [`geo_traits::MultiPolygonTrait`] implementation while preserving the
/// extension trait wrappers.
macro_rules! forward_multi_polygon_trait_ext_funcs {
    () => {
        type PolygonTypeExt<'__l_inner>
            = <Self as MultiPolygonTrait>::InnerPolygonType<'__l_inner>
        where
            Self: '__l_inner;

        #[inline]
        fn polygon_ext(&self, i: usize) -> Option<Self::PolygonTypeExt<'_>> {
            <Self as MultiPolygonTrait>::polygon(self, i)
        }

        #[inline]
        unsafe fn polygon_unchecked_ext(&self, i: usize) -> Self::PolygonTypeExt<'_> {
            <Self as MultiPolygonTrait>::polygon_unchecked(self, i)
        }

        #[inline]
        fn polygons_ext(&self) -> impl Iterator<Item = Self::PolygonTypeExt<'_>> {
            <Self as MultiPolygonTrait>::polygons(self)
        }
    };
}

impl<T> MultiPolygonTraitExt for MultiPolygon<T>
where
    T: CoordNum,
{
    type PolygonTypeExt<'a>
        = <Self as MultiPolygonTrait>::InnerPolygonType<'a>
    where
        Self: 'a;

    #[inline]
    fn polygon_ext(&self, i: usize) -> Option<Self::PolygonTypeExt<'_>> {
        self.0.get(i)
    }

    #[inline]
    unsafe fn polygon_unchecked_ext(&self, i: usize) -> Self::PolygonTypeExt<'_> {
        self.0.get_unchecked(i)
    }

    #[inline]
    fn polygons_ext(&self) -> impl Iterator<Item = Self::PolygonTypeExt<'_>> {
        self.0.iter()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for MultiPolygon<T> {
    type Tag = MultiPolygonTag;
}

impl<T> MultiPolygonTraitExt for &MultiPolygon<T>
where
    T: CoordNum,
{
    type PolygonTypeExt<'a>
        = <Self as MultiPolygonTrait>::InnerPolygonType<'a>
    where
        Self: 'a;

    #[inline]
    fn polygon_ext(&self, i: usize) -> Option<Self::PolygonTypeExt<'_>> {
        self.0.get(i)
    }

    #[inline]
    unsafe fn polygon_unchecked_ext(&self, i: usize) -> Self::PolygonTypeExt<'_> {
        self.0.get_unchecked(i)
    }

    #[inline]
    fn polygons_ext(&self) -> impl Iterator<Item = Self::PolygonTypeExt<'_>> {
        self.0.iter()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &MultiPolygon<T> {
    type Tag = MultiPolygonTag;
}

impl<T> MultiPolygonTraitExt for UnimplementedMultiPolygon<T>
where
    T: CoordNum,
{
    forward_multi_polygon_trait_ext_funcs!();
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedMultiPolygon<T> {
    type Tag = MultiPolygonTag;
}
