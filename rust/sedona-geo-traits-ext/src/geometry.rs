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
// Extend GeometryTrait traits for the `geo-traits` crate

use core::{borrow::Borrow, panic};

use geo_traits::*;
use geo_types::*;

use crate::*;

#[allow(clippy::type_complexity)]
/// Extension trait that augments [`geo_traits::GeometryTrait`] with Sedona's
/// additional helpers and type tagging support.
///
/// The trait adds accessors that mirror the behavior of `geo-types::Geometry`
/// while keeping the code ergonomic when working through trait objects.
/// Implementations must also opt into [`GeoTraitExtWithTypeTag`] so geometries
/// can be introspected using [`GeometryTag`].
pub trait GeometryTraitExt: GeometryTrait + GeoTraitExtWithTypeTag<Tag = GeometryTag>
where
    <Self as GeometryTrait>::T: CoordNum,
{
    /// Extension-aware point type exposed by this geometry.
    type PointTypeExt<'a>: 'a + PointTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Extension-aware line string type exposed by this geometry.
    type LineStringTypeExt<'a>: 'a + LineStringTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Extension-aware polygon type exposed by this geometry.
    type PolygonTypeExt<'a>: 'a + PolygonTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Extension-aware multi point type exposed by this geometry.
    type MultiPointTypeExt<'a>: 'a + MultiPointTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Extension-aware multi line string type exposed by this geometry.
    type MultiLineStringTypeExt<'a>: 'a + MultiLineStringTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Extension-aware multi polygon type exposed by this geometry.
    type MultiPolygonTypeExt<'a>: 'a + MultiPolygonTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Extension-aware triangle type exposed by this geometry.
    type TriangleTypeExt<'a>: 'a + TriangleTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Extension-aware rectangle type exposed by this geometry.
    type RectTypeExt<'a>: 'a + RectTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    /// Extension-aware line type exposed by this geometry.
    type LineTypeExt<'a>: 'a + LineTraitExt<T = <Self as GeometryTrait>::T>
    where
        Self: 'a;

    // Note that we don't have a GeometryCollectionTypeExt here, because it would introduce recursive GATs
    // such as G::GeometryCollectionTypeExt::GeometryTypeExt::GeometryCollectionTypeExt::... and easily
    // trigger a Rust compiler bug: https://github.com/rust-lang/rust/issues/128887 and https://github.com/rust-lang/rust/issues/131960.
    // See also https://github.com/geoarrow/geoarrow-rs/issues/1339.
    //
    // Although this could be worked around by not implementing generic functions using trait-based approach and use
    // function-based approach instead, see https://github.com/geoarrow/geoarrow-rs/pull/956 and https://github.com/georust/wkb/pull/77,
    // we are not certain if there will be other issues caused by recursive GATs in the future. So we decided to completely get rid
    // of recursive GATs.

    /// Reference type yielded when iterating over nested geometries inside a collection.
    type InnerGeometryRef<'a>: 'a + Borrow<Self>
    where
        Self: 'a;

    /// Returns true if this geometry is a GeometryCollection
    #[inline]
    fn is_collection(&self) -> bool {
        matches!(self.as_type(), GeometryType::GeometryCollection(_))
    }

    /// Returns the number of geometries inside this GeometryCollection
    #[inline]
    fn num_geometries_ext(&self) -> usize {
        let GeometryType::GeometryCollection(gc) = self.as_type() else {
            panic!("Not a GeometryCollection");
        };
        gc.num_geometries()
    }

    /// Cast this geometry to a [`GeometryTypeExt`] enum, which allows for downcasting to a specific
    /// type. This does not work when the geometry is a GeometryCollection. Please use `is_collection`
    /// to check if the geometry is NOT a GeometryCollection first before calling this method.
    fn as_type_ext(
        &self,
    ) -> GeometryTypeExt<
        '_,
        Self::PointTypeExt<'_>,
        Self::LineStringTypeExt<'_>,
        Self::PolygonTypeExt<'_>,
        Self::MultiPointTypeExt<'_>,
        Self::MultiLineStringTypeExt<'_>,
        Self::MultiPolygonTypeExt<'_>,
        Self::RectTypeExt<'_>,
        Self::TriangleTypeExt<'_>,
        Self::LineTypeExt<'_>,
    >;

    /// Returns a geometry by index, or None if the index is out of bounds. This method only works with
    /// GeometryCollection. Please use `is_collection` to check if the geometry is a GeometryCollection first before
    /// calling this method.
    fn geometry_ext(&self, i: usize) -> Option<Self::InnerGeometryRef<'_>>;

    /// Returns a geometry by index without bounds checking. This method only works with GeometryCollection.
    /// Please use `is_collection` to check if the geometry is a GeometryCollection first before calling this method.
    ///
    /// # Safety
    /// The caller must ensure that `i` is a valid index less than the number of geometries.
    /// Otherwise, this function may cause undefined behavior.
    unsafe fn geometry_unchecked_ext(&self, i: usize) -> Self::InnerGeometryRef<'_>;

    /// Returns an iterator over the geometries in this GeometryCollection. This method only works with
    /// GeometryCollection. Please use `is_collection` to check if the geometry is a GeometryCollection first before
    /// calling this method.
    fn geometries_ext(&self) -> impl Iterator<Item = Self::InnerGeometryRef<'_>>;
}

#[derive(Debug)]
/// Borrowed view into a concrete geometry type implementing the extension traits.
pub enum GeometryTypeExt<'a, P, LS, Y, MP, ML, MY, R, TT, L>
where
    P: PointTraitExt,
    LS: LineStringTraitExt,
    Y: PolygonTraitExt,
    MP: MultiPointTraitExt,
    ML: MultiLineStringTraitExt,
    MY: MultiPolygonTraitExt,
    R: RectTraitExt,
    TT: TriangleTraitExt,
    L: LineTraitExt,
    <P as GeometryTrait>::T: CoordNum,
    <LS as GeometryTrait>::T: CoordNum,
    <Y as GeometryTrait>::T: CoordNum,
    <MP as GeometryTrait>::T: CoordNum,
    <ML as GeometryTrait>::T: CoordNum,
    <MY as GeometryTrait>::T: CoordNum,
    <R as GeometryTrait>::T: CoordNum,
    <TT as GeometryTrait>::T: CoordNum,
    <L as GeometryTrait>::T: CoordNum,
{
    Point(&'a P),
    LineString(&'a LS),
    Polygon(&'a Y),
    MultiPoint(&'a MP),
    MultiLineString(&'a ML),
    MultiPolygon(&'a MY),
    Rect(&'a R),
    Triangle(&'a TT),
    Line(&'a L),
}

#[macro_export]
/// Forwards [`GeometryTraitExt`] associated types and methods to the
/// underlying [`geo_traits::GeometryTrait`] implementation while retaining the
/// extension trait wrappers.
macro_rules! forward_geometry_trait_ext_funcs {
    ($t:ty) => {
        type PointTypeExt<'__g_inner>
            = <Self as GeometryTrait>::PointType<'__g_inner>
        where
            Self: '__g_inner;

        type LineStringTypeExt<'__g_inner>
            = <Self as GeometryTrait>::LineStringType<'__g_inner>
        where
            Self: '__g_inner;

        type PolygonTypeExt<'__g_inner>
            = <Self as GeometryTrait>::PolygonType<'__g_inner>
        where
            Self: '__g_inner;

        type MultiPointTypeExt<'__g_inner>
            = <Self as GeometryTrait>::MultiPointType<'__g_inner>
        where
            Self: '__g_inner;

        type MultiLineStringTypeExt<'__g_inner>
            = <Self as GeometryTrait>::MultiLineStringType<'__g_inner>
        where
            Self: '__g_inner;

        type MultiPolygonTypeExt<'__g_inner>
            = <Self as GeometryTrait>::MultiPolygonType<'__g_inner>
        where
            Self: '__g_inner;

        type RectTypeExt<'__g_inner>
            = <Self as GeometryTrait>::RectType<'__g_inner>
        where
            Self: '__g_inner;

        type TriangleTypeExt<'__g_inner>
            = <Self as GeometryTrait>::TriangleType<'__g_inner>
        where
            Self: '__g_inner;

        type LineTypeExt<'__g_inner>
            = <Self as GeometryTrait>::LineType<'__g_inner>
        where
            Self: '__g_inner;

        fn as_type_ext(
            &self,
        ) -> GeometryTypeExt<
            '_,
            Self::PointTypeExt<'_>,
            Self::LineStringTypeExt<'_>,
            Self::PolygonTypeExt<'_>,
            Self::MultiPointTypeExt<'_>,
            Self::MultiLineStringTypeExt<'_>,
            Self::MultiPolygonTypeExt<'_>,
            Self::RectTypeExt<'_>,
            Self::TriangleTypeExt<'_>,
            Self::LineTypeExt<'_>,
        > {
            match self.as_type() {
                GeometryType::Point(p) => GeometryTypeExt::Point(p),
                GeometryType::LineString(ls) => GeometryTypeExt::LineString(ls),
                GeometryType::Polygon(p) => GeometryTypeExt::Polygon(p),
                GeometryType::MultiPoint(mp) => GeometryTypeExt::MultiPoint(mp),
                GeometryType::MultiLineString(mls) => GeometryTypeExt::MultiLineString(mls),
                GeometryType::MultiPolygon(mp) => GeometryTypeExt::MultiPolygon(mp),
                GeometryType::GeometryCollection(_) => {
                    panic!("GeometryCollection is not supported in GeometryTraitExt::as_type_ext")
                }
                GeometryType::Rect(r) => GeometryTypeExt::Rect(r),
                GeometryType::Triangle(t) => GeometryTypeExt::Triangle(t),
                GeometryType::Line(l) => GeometryTypeExt::Line(l),
            }
        }
    };
}

impl<T> GeometryTraitExt for Geometry<T>
where
    T: CoordNum,
{
    forward_geometry_trait_ext_funcs!(T);

    type InnerGeometryRef<'a>
        = &'a Geometry<T>
    where
        Self: 'a;

    fn geometry_ext(&self, i: usize) -> Option<&Geometry<T>> {
        let GeometryType::GeometryCollection(gc) = self.as_type() else {
            panic!("Not a GeometryCollection");
        };
        gc.geometry(i)
    }

    unsafe fn geometry_unchecked_ext(&self, i: usize) -> &Geometry<T> {
        let GeometryType::GeometryCollection(gc) = self.as_type() else {
            panic!("Not a GeometryCollection");
        };
        gc.geometry_unchecked(i)
    }

    fn geometries_ext(&self) -> impl Iterator<Item = &Geometry<T>> {
        let GeometryType::GeometryCollection(gc) = self.as_type() else {
            panic!("Not a GeometryCollection");
        };
        gc.geometries()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for Geometry<T> {
    type Tag = GeometryTag;
}

impl<'a, T> GeometryTraitExt for &'a Geometry<T>
where
    T: CoordNum,
{
    forward_geometry_trait_ext_funcs!(T);

    type InnerGeometryRef<'b>
        = &'a Geometry<T>
    where
        Self: 'b;

    fn geometry_ext(&self, i: usize) -> Option<&'a Geometry<T>> {
        let g = *self;
        g.geometry_ext(i)
    }

    unsafe fn geometry_unchecked_ext(&self, i: usize) -> &'a Geometry<T> {
        let g = *self;
        g.geometry_unchecked_ext(i)
    }

    fn geometries_ext(&self) -> impl Iterator<Item = &'a Geometry<T>> {
        let g = *self;
        g.geometries_ext()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for &Geometry<T> {
    type Tag = GeometryTag;
}

impl<T> GeometryTraitExt for UnimplementedGeometry<T>
where
    T: CoordNum,
{
    forward_geometry_trait_ext_funcs!(T);

    type InnerGeometryRef<'a>
        = &'a UnimplementedGeometry<T>
    where
        Self: 'a;

    fn geometry_ext(&self, _i: usize) -> Option<Self::InnerGeometryRef<'_>> {
        unimplemented!()
    }

    unsafe fn geometry_unchecked_ext(&self, _i: usize) -> Self::InnerGeometryRef<'_> {
        unimplemented!()
    }

    fn geometries_ext(&self) -> impl Iterator<Item = Self::InnerGeometryRef<'_>> {
        unimplemented!();

        // For making the type checker happy
        #[allow(unreachable_code)]
        core::iter::empty()
    }
}

impl<T: CoordNum> GeoTraitExtWithTypeTag for UnimplementedGeometry<T> {
    type Tag = GeometryTag;
}
