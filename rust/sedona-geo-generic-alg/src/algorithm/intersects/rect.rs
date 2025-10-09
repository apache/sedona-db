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
//! Intersects implementations for Rect (generic)
//!
//! Ported (and contains copied code) from `geo::algorithm::intersects::rect`:
//! <https://github.com/georust/geo/blob/5d667f844716a3d0a17aa60bc0a58528cb5808c3/geo/src/algorithm/intersects/rect.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
use geo_traits::CoordTrait;
use sedona_geo_traits_ext::*;

use super::IntersectsTrait;
use crate::*;

impl<T, LHS, RHS> IntersectsTrait<RectTag, CoordTag, RHS> for LHS
where
    T: CoordNum,
    LHS: RectTraitExt<T = T>,
    RHS: CoordTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        let lhs_x = rhs.x();
        let lhs_y = rhs.y();

        lhs_x >= self.min().x()
            && lhs_y >= self.min().y()
            && lhs_x <= self.max().x()
            && lhs_y <= self.max().y()
    }
}

symmetric_intersects_trait_impl!(CoordNum, CoordTraitExt, CoordTag, RectTraitExt, RectTag);
symmetric_intersects_trait_impl!(CoordNum, RectTraitExt, RectTag, PointTraitExt, PointTag);
symmetric_intersects_trait_impl!(
    CoordNum,
    RectTraitExt,
    RectTag,
    MultiPointTraitExt,
    MultiPointTag
);

impl<T, LHS, RHS> IntersectsTrait<RectTag, RectTag, RHS> for LHS
where
    T: CoordNum,
    LHS: RectTraitExt<T = T>,
    RHS: RectTraitExt<T = T>,
{
    fn intersects_trait(&self, other: &RHS) -> bool {
        if self.max().x() < other.min().x() {
            return false;
        }

        if self.max().y() < other.min().y() {
            return false;
        }

        if self.min().x() > other.max().x() {
            return false;
        }

        if self.min().y() > other.max().y() {
            return false;
        }

        true
    }
}

// Same logic as polygon x line, but avoid an allocation.
impl<T, LHS, RHS> IntersectsTrait<RectTag, LineTag, RHS> for LHS
where
    T: GeoNum,
    LHS: RectTraitExt<T = T>,
    RHS: LineTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        let lt = self.min_coord();
        let rb = self.max_coord();
        let lb = Coord::from((lt.x, rb.y));
        let rt = Coord::from((rb.x, lt.y));

        // If either rhs.{start,end} lies inside Rect, then true
        self.intersects_trait(&rhs.start_ext())
            || self.intersects_trait(&rhs.end_ext())
            || Line::new(lt, rt).intersects_trait(rhs)
            || Line::new(rt, rb).intersects_trait(rhs)
            || Line::new(lb, rb).intersects_trait(rhs)
            || Line::new(lt, lb).intersects_trait(rhs)
    }
}

symmetric_intersects_trait_impl!(GeoNum, LineTraitExt, LineTag, RectTraitExt, RectTag);

impl<T, LHS, RHS> IntersectsTrait<RectTag, TriangleTag, RHS> for LHS
where
    T: GeoNum,
    LHS: RectTraitExt<T = T>,
    RHS: TriangleTraitExt<T = T>,
{
    fn intersects_trait(&self, other: &RHS) -> bool {
        self.intersects_trait(&other.to_polygon())
    }
}

symmetric_intersects_trait_impl!(GeoNum, TriangleTraitExt, TriangleTag, RectTraitExt, RectTag);
