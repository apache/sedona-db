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
//! Intersects implementations for Triangle (generic)
//!
//! Ported (and contains copied code) from `geo::algorithm::intersects::triangle`:
//! <https://github.com/georust/geo/blob/f2326a3dd1fa9ff39d3e65618eb7ca2bacad2c0c/geo/src/algorithm/intersects/triangle.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
use super::IntersectsTrait;
use crate::{intersects::has_disjoint_bboxes, *};
use geo_traits::LineStringTrait;
use sedona_geo_traits_ext::*;

impl<T, LHS, RHS> IntersectsTrait<TriangleTag, CoordTag, RHS> for LHS
where
    T: GeoNum,
    LHS: TriangleTraitExt<T = T>,
    RHS: CoordTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        let rhs = rhs.geo_coord();

        let mut orientations = self
            .to_lines()
            .map(|l| T::Ker::orient2d(l.start, l.end, rhs));

        orientations.sort();

        !orientations
            .windows(2)
            .any(|win| win[0] != win[1] && win[1] != Orientation::Collinear)

        // // neglecting robust predicates, hence faster
        // let p0x = self.0.x.to_f64().unwrap();
        // let p0y = self.0.y.to_f64().unwrap();
        // let p1x = self.1.x.to_f64().unwrap();
        // let p1y = self.1.y.to_f64().unwrap();
        // let p2x = self.2.x.to_f64().unwrap();
        // let p2y = self.2.y.to_f64().unwrap();

        // let px = rhs.x.to_f64().unwrap();
        // let py = rhs.y.to_f64().unwrap();

        // let s = (p0x - p2x) * (py - p2y) - (p0y - p2y) * (px - p2x);
        // let t = (p1x - p0x) * (py - p0y) - (p1y - p0y) * (px - p0x);

        // if (s < 0.) != (t < 0.) && s != 0. && t != 0. {
        //     return false;
        // }

        // let d = (p2x - p1x) * (py - p1y) - (p2y - p1y) * (px - p1x);
        // d == 0. || (d < 0.) == (s + t <= 0.)
    }
}

symmetric_intersects_trait_impl!(
    GeoNum,
    CoordTraitExt,
    CoordTag,
    TriangleTraitExt,
    TriangleTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    TriangleTraitExt,
    TriangleTag,
    PointTraitExt,
    PointTag
);

impl<T, LHS, RHS> IntersectsTrait<TriangleTag, TriangleTag, RHS> for LHS
where
    T: GeoNum,
    LHS: TriangleTraitExt<T = T>,
    RHS: TriangleTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        self.to_polygon().intersects_trait(&rhs.to_polygon())
    }
}

impl<T, LHS, RHS> IntersectsTrait<TriangleTag, PolygonTag, RHS> for LHS
where
    T: GeoNum,
    LHS: TriangleTraitExt<T = T>,
    RHS: PolygonTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        // simplified logic based on Polygon intersects Polygon

        if has_disjoint_bboxes(self, rhs) {
            return false;
        }

        // empty polygon cannot intersect with triangle
        let Some(exterior) = rhs.exterior_ext() else {
            return false;
        };
        if exterior.num_coords() == 0 {
            return false;
        }

        // if any of the polygon's corners intersect the triangle
        let first_coord = unsafe { exterior.geo_coord_unchecked(0) };
        if self.intersects(&first_coord) {
            return true;
        }

        // or any point of the triangle intersects the polygon
        if self.first_coord().intersects(rhs) {
            return true;
        }

        let rect_lines = self.to_lines();

        // or any of the polygon's lines intersect the triangle's lines
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

symmetric_intersects_trait_impl!(
    GeoNum,
    PolygonTraitExt,
    PolygonTag,
    TriangleTraitExt,
    TriangleTag
);

impl<T, LHS, RHS> IntersectsTrait<TriangleTag, RectTag, RHS> for LHS
where
    T: GeoNum,
    LHS: TriangleTraitExt<T = T>,
    RHS: RectTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        // simplified logic based on Polygon intersects Polygon

        if has_disjoint_bboxes(self, rhs) {
            return false;
        }

        // if any of the rectangle's corners intersect the triangle
        self.intersects(&rhs.min_coord())

        // or some corner of the triangle intersects the rectangle
        || self.first_coord().intersects(rhs)

        // or any of the triangle's lines intersect the rectangle's lines
        || rhs.to_lines().iter().any(|rhs_line| {
            self.to_lines().iter().any(|self_line| self_line.intersects(&rhs_line))
        })
    }
}

symmetric_intersects_trait_impl!(GeoNum, RectTraitExt, RectTag, TriangleTraitExt, TriangleTag);

#[cfg(test)]
mod test_polygon {
    use geo::{Convert, CoordsIter};

    use super::*;

    #[test]
    fn test_disjoint() {
        let triangle = Triangle::from([(0., 0.), (10., 0.), (10., 10.)]);
        let polygon: Polygon<f64> = Rect::new((11, 11), (12, 12)).to_polygon().convert();
        assert!(!triangle.intersects(&polygon));
    }

    #[test]
    fn test_partial() {
        let triangle = Triangle::from([(0., 0.), (10., 0.), (10., 10.)]);
        let polygon: Polygon<f64> = Rect::new((9, 9), (12, 12)).to_polygon().convert();
        assert!(triangle.intersects(&polygon));
    }

    #[test]
    fn test_triangle_inside_polygon() {
        let triangle = Triangle::from([(1., 1.), (2., 1.), (2., 2.)]);
        let polygon: Polygon<f64> = Rect::new((0, 0), (10, 10)).to_polygon().convert();
        assert!(triangle.intersects(&polygon));
    }

    #[test]
    fn test_polygon_inside_triangle() {
        let triangle = Triangle::from([(0., 0.), (10., 0.), (10., 10.)]);
        let polygon: Polygon<f64> = Rect::new((1, 1), (2, 2)).to_polygon().convert();
        assert!(triangle.intersects(&polygon));
    }

    // Hole related tests

    #[test]
    fn test_rect_inside_polygon_hole() {
        let bound: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let hole = Rect::new((1, 1), (9, 9)).convert();
        let triangle = Triangle::from([(4., 4.), (4., 6.), (6., 6.)]);
        let polygon = Polygon::new(
            bound.exterior_coords_iter().collect(),
            vec![hole.exterior_coords_iter().collect()],
        );

        assert!(!triangle.intersects(&polygon));
    }

    #[test]
    fn test_triangle_equals_polygon_hole() {
        let bound: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        let triangle = Triangle::from([(4., 4.), (4., 6.), (6., 6.)]);
        let polygon = Polygon::new(
            bound.exterior_coords_iter().collect(),
            vec![triangle.exterior_coords_iter().collect()],
        );

        assert!(triangle.intersects(&polygon));
    }
}
