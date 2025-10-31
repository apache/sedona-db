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
//! Intersects implementations for Coord and Point (generic)
//!
//! Ported (and contains copied code) from `geo::algorithm::intersects::coordinate`:
//! <https://github.com/georust/geo/blob/f2326a3dd1fa9ff39d3e65618eb7ca2bacad2c0c/geo/src/algorithm/intersects/coordinate.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
use sedona_geo_traits_ext::{CoordTag, CoordTraitExt, PointTag, PointTraitExt};

use super::IntersectsTrait;
use crate::*;

impl<T, LHS, RHS> IntersectsTrait<CoordTag, CoordTag, RHS> for LHS
where
    T: CoordNum,
    LHS: CoordTraitExt<T = T>,
    RHS: CoordTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        self.geo_coord() == rhs.geo_coord()
    }
}

// The other side of this is handled via a blanket impl.
impl<T, LHS, RHS> IntersectsTrait<CoordTag, PointTag, RHS> for LHS
where
    T: CoordNum,
    LHS: CoordTraitExt<T = T>,
    RHS: PointTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        rhs.geo_coord().is_some_and(|c| self.geo_coord() == c)
    }
}
