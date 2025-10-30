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
//! <https://github.com/georust/geo/blob/f2326a3dd1fa9ff39d3e65618eb7ca2bacad2c0c/geo/src/algorithm/intersects/rect.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
use geo_traits::{CoordTrait, LineStringTrait};
use sedona_geo_traits_ext::*;

use super::IntersectsTrait;
use crate::{intersects::has_disjoint_bboxes, *};

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

impl<T, LHS, RHS> IntersectsTrait<RectTag, PolygonTag, RHS> for LHS
where
    T: GeoNum,
    LHS: RectTraitExt<T = T>,
    RHS: PolygonTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        // simplified logic based on Polygon intersects Polygon

        if has_disjoint_bboxes(self, rhs) {
            return false;
        }

        // empty polygon cannot intersect with rectangle
        let Some(exterior) = rhs.exterior_ext() else {
            return false;
        };
        if exterior.num_coords() == 0 {
            return false;
        }

        // if any of the polygon's corners intersect the rectangle
        let first_coord = unsafe { exterior.geo_coord_unchecked(0) };
        if self.intersects(&first_coord) {
            return true;
        }

        // or any point of the rectangle intersects the polygon
        if self.min_coord().intersects(rhs) {
            return true;
        }

        let rect_lines = self.to_lines();

        // or any of the polygon's lines intersect the rectangle's lines
        if exterior.lines().any(|rhs_line| {
            rect_lines
                .iter()
                .any(|self_line| self_line.intersects(&rhs_line))
        }) {
            return true;
        }
        rhs.interiors_ext().any(|interior| {
            interior.lines().any(|rhs_line| {
                rect_lines
                    .iter()
                    .any(|self_line| self_line.intersects(&rhs_line))
            })
        })
    }
}

symmetric_intersects_trait_impl!(GeoNum, PolygonTraitExt, PolygonTag, RectTraitExt, RectTag);

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
        let lb = self.min_coord();
        let rt = self.max_coord();
        let lt = Coord::from((lb.x, rt.y));
        let rb = Coord::from((rt.x, lb.y));

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

#[cfg(test)]
mod test_triangle {
    use geo::Convert;

    use super::*;

    #[test]
    fn test_disjoint() {
        let rect: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let triangle = Triangle::from([(0., 11.), (1., 11.), (1., 12.)]);
        assert!(!rect.intersects(&triangle));
    }

    #[test]
    fn test_partial() {
        let rect: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let triangle = Triangle::from([(1., 1.), (1., 2.), (2., 1.)]);
        assert!(rect.intersects(&triangle));
    }

    #[test]
    fn test_triangle_inside_rect() {
        let rect: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let triangle = Triangle::from([(1., 1.), (1., 2.), (2., 1.)]);
        assert!(rect.intersects(&triangle));
    }

    #[test]
    fn test_rect_inside_triangle() {
        let rect: Rect<f64> = Rect::new((1, 1), (2, 2)).convert();
        let triangle = Triangle::from([(0., 10.), (10., 0.), (0., 0.)]);
        assert!(rect.intersects(&triangle));
    }
}

#[cfg(test)]
mod test_polygon {
    use geo::{Convert, CoordsIter};

    use super::*;

    #[test]
    fn test_disjoint() {
        let rect: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let polygon: Polygon<f64> = Rect::new((11, 11), (12, 12)).to_polygon().convert();
        assert!(!rect.intersects(&polygon));
    }

    #[test]
    fn test_partial() {
        let rect: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let polygon: Polygon<f64> = Rect::new((9, 9), (12, 12)).to_polygon().convert();
        assert!(rect.intersects(&polygon));
    }

    #[test]
    fn test_rect_inside_polygon() {
        let rect: Rect<f64> = Rect::new((1, 1), (2, 2)).convert();
        let polygon: Polygon<f64> = Rect::new((0, 0), (10, 10)).to_polygon().convert();
        assert!(rect.intersects(&polygon));
    }

    #[test]
    fn test_polygon_inside_rect() {
        let rect: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let polygon: Polygon<f64> = Rect::new((1, 1), (2, 2)).to_polygon().convert();
        assert!(rect.intersects(&polygon));
    }

    // Hole related tests

    #[test]
    fn test_rect_inside_polygon_hole() {
        let bound: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let hole = Rect::new((1, 1), (9, 9)).convert();
        let rect = Rect::new((4, 4), (6, 6)).convert();
        let polygon = Polygon::new(
            bound.exterior_coords_iter().collect(),
            vec![hole.exterior_coords_iter().collect()],
        );

        assert!(!rect.intersects(&polygon));
    }

    #[test]
    fn test_rect_equals_polygon_hole() {
        let bound: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let rect: Rect = Rect::new((4, 4), (6, 6)).convert();
        let polygon = Polygon::new(
            bound.exterior_coords_iter().collect(),
            vec![rect.exterior_coords_iter().collect()],
        );

        assert!(rect.intersects(&polygon));
    }
}
