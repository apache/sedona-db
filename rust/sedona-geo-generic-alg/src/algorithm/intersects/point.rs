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
//! Intersects implementations for Point and MultiPoint (generic)
//!
//! Ported (and contains copied code) from `geo::algorithm::intersects::point`:
//! <https://github.com/georust/geo/blob/f2326a3dd1fa9ff39d3e65618eb7ca2bacad2c0c/geo/src/algorithm/intersects/point.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
use sedona_geo_traits_ext::*;

use super::IntersectsTrait;
use crate::*;

// Generate implementations for Point<T> by delegating to Coord<T>
macro_rules! impl_intersects_point_from_coord {
    ($num_type:ident, $rhs_type:ident, $rhs_tag:ident) => {
        impl<T, LHS, RHS> IntersectsTrait<PointTag, $rhs_tag, RHS> for LHS
        where
            T: $num_type,
            LHS: PointTraitExt<T = T>,
            RHS: $rhs_type<T = T>,
        {
            fn intersects_trait(&self, rhs: &RHS) -> bool {
                self.coord_ext().is_some_and(|c| c.intersects_trait(rhs))
            }
        }
    };
}

impl_intersects_point_from_coord!(CoordNum, CoordTraitExt, CoordTag);
impl_intersects_point_from_coord!(CoordNum, PointTraitExt, PointTag);
impl_intersects_point_from_coord!(GeoNum, LineStringTraitExt, LineStringTag);
impl_intersects_point_from_coord!(GeoNum, PolygonTraitExt, PolygonTag);
impl_intersects_point_from_coord!(CoordNum, MultiPointTraitExt, MultiPointTag);
impl_intersects_point_from_coord!(GeoNum, MultiLineStringTraitExt, MultiLineStringTag);
impl_intersects_point_from_coord!(GeoNum, MultiPolygonTraitExt, MultiPolygonTag);
impl_intersects_point_from_coord!(GeoNum, GeometryTraitExt, GeometryTag);
impl_intersects_point_from_coord!(GeoNum, GeometryCollectionTraitExt, GeometryCollectionTag);
impl_intersects_point_from_coord!(GeoNum, LineTraitExt, LineTag);
impl_intersects_point_from_coord!(CoordNum, RectTraitExt, RectTag);
impl_intersects_point_from_coord!(GeoNum, TriangleTraitExt, TriangleTag);

// Generate implementations for MultiPoint<T> by delegating to Point<T>
macro_rules! impl_intersects_multipoint_from_point {
    ($num_type:ident, $rhs_type:ident, $rhs_tag:ident) => {
        impl<T, LHS, RHS> IntersectsTrait<MultiPointTag, $rhs_tag, RHS> for LHS
        where
            T: $num_type,
            LHS: MultiPointTraitExt<T = T>,
            RHS: $rhs_type<T = T>,
        {
            fn intersects_trait(&self, rhs: &RHS) -> bool {
                self.points_ext().any(|p| p.intersects_trait(rhs))
            }
        }
    };
}

impl_intersects_multipoint_from_point!(CoordNum, CoordTraitExt, CoordTag);
impl_intersects_multipoint_from_point!(CoordNum, PointTraitExt, PointTag);
impl_intersects_multipoint_from_point!(GeoNum, LineStringTraitExt, LineStringTag);
impl_intersects_multipoint_from_point!(GeoNum, PolygonTraitExt, PolygonTag);
impl_intersects_multipoint_from_point!(CoordNum, MultiPointTraitExt, MultiPointTag);
impl_intersects_multipoint_from_point!(GeoNum, MultiLineStringTraitExt, MultiLineStringTag);
impl_intersects_multipoint_from_point!(GeoNum, MultiPolygonTraitExt, MultiPolygonTag);
impl_intersects_multipoint_from_point!(GeoNum, GeometryTraitExt, GeometryTag);
impl_intersects_multipoint_from_point!(GeoNum, GeometryCollectionTraitExt, GeometryCollectionTag);
impl_intersects_multipoint_from_point!(GeoNum, LineTraitExt, LineTag);
impl_intersects_multipoint_from_point!(CoordNum, RectTraitExt, RectTag);
impl_intersects_multipoint_from_point!(GeoNum, TriangleTraitExt, TriangleTag);

symmetric_intersects_trait_impl!(
    CoordNum,
    CoordTraitExt,
    CoordTag,
    MultiPointTraitExt,
    MultiPointTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    LineTraitExt,
    LineTag,
    MultiPointTraitExt,
    MultiPointTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    TriangleTraitExt,
    TriangleTag,
    MultiPointTraitExt,
    MultiPointTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    PolygonTraitExt,
    PolygonTag,
    MultiPointTraitExt,
    MultiPointTag
);
