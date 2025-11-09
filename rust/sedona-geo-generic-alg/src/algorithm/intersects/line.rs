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
//! Intersects implementations for Line (generic)
//!
//! Ported (and contains copied code) from `geo::algorithm::intersects::line`:
//! <https://github.com/georust/geo/blob/f2326a3dd1fa9ff39d3e65618eb7ca2bacad2c0c/geo/src/algorithm/intersects/line.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
use sedona_geo_traits_ext::*;

use super::{point_in_rect, IntersectsTrait};
use crate::*;

impl<T, LHS, RHS> IntersectsTrait<LineTag, CoordTag, RHS> for LHS
where
    T: GeoNum,
    LHS: LineTraitExt<T = T>,
    RHS: CoordTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        let start = self.start_coord();
        let end = self.end_coord();
        let rhs = rhs.geo_coord();

        // First we check if the point is collinear with the line.
        T::Ker::orient2d(start, end, rhs) == Orientation::Collinear
        // In addition, the point must have _both_ coordinates
        // within the start and end bounds.
            && point_in_rect(rhs, start, end)
    }
}

symmetric_intersects_trait_impl!(GeoNum, CoordTraitExt, CoordTag, LineTraitExt, LineTag);
symmetric_intersects_trait_impl!(GeoNum, LineTraitExt, LineTag, PointTraitExt, PointTag);

impl<T, LHS, RHS> IntersectsTrait<LineTag, LineTag, RHS> for LHS
where
    T: GeoNum,
    LHS: LineTraitExt<T = T>,
    RHS: LineTraitExt<T = T>,
{
    fn intersects_trait(&self, line: &RHS) -> bool {
        let start_ext = self.start_ext();
        let self_start = self.start_coord();
        let self_end = self.end_coord();
        let line_start = line.start_coord();
        let line_end = line.end_coord();

        // Special case: self is equiv. to a point.
        if self_start == self_end {
            return line.intersects_trait(&start_ext);
        }

        // Precondition: start and end are distinct.

        // Check if orientation of rhs.{start,end} are different
        // with respect to self.{start,end}.
        let check_1_1 = T::Ker::orient2d(self_start, self_end, line_start);
        let check_1_2 = T::Ker::orient2d(self_start, self_end, line_end);

        if check_1_1 != check_1_2 {
            // Since the checks are different,
            // rhs.{start,end} are distinct, and rhs is not
            // collinear with self. Thus, there is exactly
            // one point on the infinite extensions of rhs,
            // that is collinear with self.

            // By continuity, this point is not on the
            // exterior of rhs. Now, check the same with
            // self, rhs swapped.

            let check_2_1 = T::Ker::orient2d(line_start, line_end, self_start);
            let check_2_2 = T::Ker::orient2d(line_start, line_end, self_end);

            // By similar argument, there is (exactly) one
            // point on self, collinear with rhs. Thus,
            // those two have to be same, and lies (interior
            // or boundary, but not exterior) on both lines.
            check_2_1 != check_2_2
        } else if check_1_1 == Orientation::Collinear {
            // Special case: collinear line segments.

            // Equivalent to 4 point-line intersection
            // checks, but removes the calls to the kernel
            // predicates.
            point_in_rect(line_start, self_start, self_end)
                || point_in_rect(line_end, self_start, self_end)
                || point_in_rect(self_end, line_start, line_end)
                || point_in_rect(self_end, line_start, line_end)
        } else {
            false
        }
    }
}

impl<T, LHS, RHS> IntersectsTrait<LineTag, TriangleTag, RHS> for LHS
where
    T: GeoNum,
    LHS: LineTraitExt<T = T>,
    RHS: TriangleTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        self.intersects_trait(&rhs.to_polygon())
    }
}

symmetric_intersects_trait_impl!(GeoNum, TriangleTraitExt, TriangleTag, LineTraitExt, LineTag);
