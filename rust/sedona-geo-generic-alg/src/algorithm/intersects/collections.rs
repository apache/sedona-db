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
//! Intersects implementations for Geometry and GeometryCollection (generic)
//!
//! Ported (and contains copied code) from `geo::algorithm::intersects::collections`:
//! <https://github.com/georust/geo/blob/f2326a3dd1fa9ff39d3e65618eb7ca2bacad2c0c/geo/src/algorithm/intersects/collections.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
use core::borrow::Borrow;
use sedona_geo_traits_ext::*;

use super::has_disjoint_bboxes;
use super::IntersectsTrait;
use crate::GeoNum;

macro_rules! impl_intersects_geometry {
    ($rhs_type:ident, $rhs_tag:ident) => {
        impl<T, LHS, RHS> IntersectsTrait<GeometryTag, $rhs_tag, RHS> for LHS
        where
            T: GeoNum,
            LHS: GeometryTraitExt<T = T>,
            RHS: $rhs_type<T = T>,
        {
            fn intersects_trait(&self, rhs: &RHS) -> bool {
                if self.is_collection() {
                    self.geometries_ext()
                        .any(|lhs_inner| lhs_inner.borrow().intersects_trait(rhs))
                } else {
                    match self.as_type_ext() {
                        GeometryTypeExt::Point(g) => g.intersects_trait(rhs),
                        GeometryTypeExt::Line(g) => g.intersects_trait(rhs),
                        GeometryTypeExt::LineString(g) => g.intersects_trait(rhs),
                        GeometryTypeExt::Polygon(g) => g.intersects_trait(rhs),
                        GeometryTypeExt::MultiPoint(g) => g.intersects_trait(rhs),
                        GeometryTypeExt::MultiLineString(g) => g.intersects_trait(rhs),
                        GeometryTypeExt::MultiPolygon(g) => g.intersects_trait(rhs),
                        GeometryTypeExt::Rect(g) => g.intersects_trait(rhs),
                        GeometryTypeExt::Triangle(g) => g.intersects_trait(rhs),
                    }
                }
            }
        }
    };
}

impl_intersects_geometry!(CoordTraitExt, CoordTag);
impl_intersects_geometry!(PointTraitExt, PointTag);
impl_intersects_geometry!(LineStringTraitExt, LineStringTag);
impl_intersects_geometry!(PolygonTraitExt, PolygonTag);
impl_intersects_geometry!(MultiPointTraitExt, MultiPointTag);
impl_intersects_geometry!(MultiLineStringTraitExt, MultiLineStringTag);
impl_intersects_geometry!(MultiPolygonTraitExt, MultiPolygonTag);
impl_intersects_geometry!(GeometryTraitExt, GeometryTag);
impl_intersects_geometry!(GeometryCollectionTraitExt, GeometryCollectionTag);
impl_intersects_geometry!(LineTraitExt, LineTag);
impl_intersects_geometry!(RectTraitExt, RectTag);
impl_intersects_geometry!(TriangleTraitExt, TriangleTag);

symmetric_intersects_trait_impl!(
    GeoNum,
    CoordTraitExt,
    CoordTag,
    GeometryTraitExt,
    GeometryTag
);
symmetric_intersects_trait_impl!(GeoNum, LineTraitExt, LineTag, GeometryTraitExt, GeometryTag);
symmetric_intersects_trait_impl!(GeoNum, RectTraitExt, RectTag, GeometryTraitExt, GeometryTag);
symmetric_intersects_trait_impl!(
    GeoNum,
    TriangleTraitExt,
    TriangleTag,
    GeometryTraitExt,
    GeometryTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    PolygonTraitExt,
    PolygonTag,
    GeometryTraitExt,
    GeometryTag
);

// Generate implementations for GeometryCollection by delegating to the Geometry implementation
macro_rules! impl_intersects_geometry_collection_from_geometry {
    ($rhs_type:ident, $rhs_tag:ident) => {
        impl<T, LHS, RHS> IntersectsTrait<GeometryCollectionTag, $rhs_tag, RHS> for LHS
        where
            T: GeoNum,
            LHS: GeometryCollectionTraitExt<T = T>,
            RHS: $rhs_type<T = T>,
        {
            fn intersects_trait(&self, rhs: &RHS) -> bool {
                if has_disjoint_bboxes(self, rhs) {
                    return false;
                }
                self.geometries_ext().any(|geom| geom.intersects_trait(rhs))
            }
        }
    };
}

impl_intersects_geometry_collection_from_geometry!(CoordTraitExt, CoordTag);
impl_intersects_geometry_collection_from_geometry!(PointTraitExt, PointTag);
impl_intersects_geometry_collection_from_geometry!(LineStringTraitExt, LineStringTag);
impl_intersects_geometry_collection_from_geometry!(PolygonTraitExt, PolygonTag);
impl_intersects_geometry_collection_from_geometry!(MultiPointTraitExt, MultiPointTag);
impl_intersects_geometry_collection_from_geometry!(MultiLineStringTraitExt, MultiLineStringTag);
impl_intersects_geometry_collection_from_geometry!(MultiPolygonTraitExt, MultiPolygonTag);
impl_intersects_geometry_collection_from_geometry!(GeometryTraitExt, GeometryTag);
impl_intersects_geometry_collection_from_geometry!(
    GeometryCollectionTraitExt,
    GeometryCollectionTag
);
impl_intersects_geometry_collection_from_geometry!(LineTraitExt, LineTag);
impl_intersects_geometry_collection_from_geometry!(RectTraitExt, RectTag);
impl_intersects_geometry_collection_from_geometry!(TriangleTraitExt, TriangleTag);

symmetric_intersects_trait_impl!(
    GeoNum,
    CoordTraitExt,
    CoordTag,
    GeometryCollectionTraitExt,
    GeometryCollectionTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    LineTraitExt,
    LineTag,
    GeometryCollectionTraitExt,
    GeometryCollectionTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    RectTraitExt,
    RectTag,
    GeometryCollectionTraitExt,
    GeometryCollectionTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    TriangleTraitExt,
    TriangleTag,
    GeometryCollectionTraitExt,
    GeometryCollectionTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    PolygonTraitExt,
    PolygonTag,
    GeometryCollectionTraitExt,
    GeometryCollectionTag
);
