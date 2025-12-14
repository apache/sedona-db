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
//! Intersects implementations for LineString and MultiLineString (generic)
//!
//! Ported (and contains copied code) from `geo::algorithm::intersects::line_string`:
//! <https://github.com/georust/geo/blob/f2326a3dd1fa9ff39d3e65618eb7ca2bacad2c0c/geo/src/algorithm/intersects/line_string.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
use sedona_geo_traits_ext::*;

use super::{has_disjoint_bboxes, IntersectsTrait};
use crate::*;

// Generate implementations for LineString<T> by delegating to Line<T>
macro_rules! impl_intersects_line_string_from_line {
    ($rhs_type:ident, $rhs_tag:ident) => {
        impl<T, LHS, RHS> IntersectsTrait<LineStringTag, $rhs_tag, RHS> for LHS
        where
            T: GeoNum,
            LHS: LineStringTraitExt<T = T>,
            RHS: $rhs_type<T = T>,
        {
            fn intersects_trait(&self, rhs: &RHS) -> bool {
                if has_disjoint_bboxes(self, rhs) {
                    return false;
                }
                self.lines().any(|l| l.intersects_trait(rhs))
            }
        }
    };
}

impl_intersects_line_string_from_line!(CoordTraitExt, CoordTag);
impl_intersects_line_string_from_line!(PointTraitExt, PointTag);
impl_intersects_line_string_from_line!(LineStringTraitExt, LineStringTag);
impl_intersects_line_string_from_line!(MultiPointTraitExt, MultiPointTag);
impl_intersects_line_string_from_line!(MultiLineStringTraitExt, MultiLineStringTag);
impl_intersects_line_string_from_line!(GeometryTraitExt, GeometryTag);
impl_intersects_line_string_from_line!(GeometryCollectionTraitExt, GeometryCollectionTag);
impl_intersects_line_string_from_line!(LineTraitExt, LineTag);

impl<T, LHS, RHS> IntersectsTrait<LineStringTag, PolygonTag, RHS> for LHS
where
    T: GeoNum,
    LHS: LineStringTraitExt<T = T>,
    RHS: PolygonTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        if self.num_coords() == 0 {
            return false;
        }
        if let Some(exterior) = rhs.exterior_ext() {
            if has_disjoint_bboxes(self, rhs) {
                return false;
            }

            // if no lines intersections, then linestring is either disjoint or within the polygon
            // therefore sufficient to check any one point
            let first_coord = unsafe { self.geo_coord_unchecked(0) };
            first_coord.intersects(rhs)
                || self.lines().any(|l| {
                    exterior.lines().any(|other| l.intersects(&other))
                        || rhs
                            .interiors_ext()
                            .any(|interior| interior.lines().any(|other| l.intersects(&other)))
                })
        } else {
            false
        }
    }
}

impl<T, LHS, RHS> IntersectsTrait<LineStringTag, MultiPolygonTag, RHS> for LHS
where
    T: GeoNum,
    LHS: LineStringTraitExt<T = T>,
    RHS: MultiPolygonTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        if self.num_coords() == 0 {
            return false;
        }
        if has_disjoint_bboxes(self, rhs) {
            return false;
        }
        // splitting into `LineString intersects Polygon`
        rhs.polygons_ext().any(|poly| self.intersects(&poly))
    }
}

impl<T, LHS, RHS> IntersectsTrait<LineStringTag, RectTag, RHS> for LHS
where
    T: GeoNum,
    LHS: LineStringTraitExt<T = T>,
    RHS: RectTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        if self.num_coords() == 0 {
            return false;
        }

        let first_coord = unsafe { self.geo_coord_unchecked(0) };
        first_coord.intersects(rhs)
            || self
                .lines()
                .any(|l| rhs.to_lines().iter().any(|other| l.intersects(&other)))
    }
}

impl<T, LHS, RHS> IntersectsTrait<LineStringTag, TriangleTag, RHS> for LHS
where
    T: GeoNum,
    LHS: LineStringTraitExt<T = T>,
    RHS: TriangleTraitExt<T = T>,
{
    fn intersects_trait(&self, rhs: &RHS) -> bool {
        if self.num_coords() == 0 {
            return false;
        }

        let first_coord = unsafe { self.geo_coord_unchecked(0) };
        first_coord.intersects(rhs)
            || self
                .lines()
                .any(|l| rhs.to_lines().iter().any(|other| l.intersects(&other)))
    }
}

symmetric_intersects_trait_impl!(
    GeoNum,
    CoordTraitExt,
    CoordTag,
    LineStringTraitExt,
    LineStringTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    LineTraitExt,
    LineTag,
    LineStringTraitExt,
    LineStringTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    RectTraitExt,
    RectTag,
    LineStringTraitExt,
    LineStringTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    TriangleTraitExt,
    TriangleTag,
    LineStringTraitExt,
    LineStringTag
);

// Generate implementations for MultiLineString<T> by delegating to LineString<T>
macro_rules! impl_intersects_multi_line_string_from_line_string {
    ($rhs_type:ident, $rhs_tag:ident) => {
        impl<T, LHS, RHS> IntersectsTrait<MultiLineStringTag, $rhs_tag, RHS> for LHS
        where
            T: GeoNum,
            LHS: MultiLineStringTraitExt<T = T>,
            RHS: $rhs_type<T = T>,
        {
            fn intersects_trait(&self, rhs: &RHS) -> bool {
                if has_disjoint_bboxes(self, rhs) {
                    return false;
                }
                self.line_strings_ext().any(|ls| ls.intersects_trait(rhs))
            }
        }
    };
}

impl_intersects_multi_line_string_from_line_string!(CoordTraitExt, CoordTag);
impl_intersects_multi_line_string_from_line_string!(PointTraitExt, PointTag);
impl_intersects_multi_line_string_from_line_string!(LineStringTraitExt, LineStringTag);
impl_intersects_multi_line_string_from_line_string!(PolygonTraitExt, PolygonTag);
impl_intersects_multi_line_string_from_line_string!(MultiPointTraitExt, MultiPointTag);
impl_intersects_multi_line_string_from_line_string!(MultiLineStringTraitExt, MultiLineStringTag);
impl_intersects_multi_line_string_from_line_string!(MultiPolygonTraitExt, MultiPolygonTag);
impl_intersects_multi_line_string_from_line_string!(GeometryTraitExt, GeometryTag);
impl_intersects_multi_line_string_from_line_string!(
    GeometryCollectionTraitExt,
    GeometryCollectionTag
);
impl_intersects_multi_line_string_from_line_string!(LineTraitExt, LineTag);
impl_intersects_multi_line_string_from_line_string!(RectTraitExt, RectTag);
impl_intersects_multi_line_string_from_line_string!(TriangleTraitExt, TriangleTag);

symmetric_intersects_trait_impl!(
    GeoNum,
    CoordTraitExt,
    CoordTag,
    MultiLineStringTraitExt,
    MultiLineStringTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    LineTraitExt,
    LineTag,
    MultiLineStringTraitExt,
    MultiLineStringTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    RectTraitExt,
    RectTag,
    MultiLineStringTraitExt,
    MultiLineStringTag
);
symmetric_intersects_trait_impl!(
    GeoNum,
    TriangleTraitExt,
    TriangleTag,
    MultiLineStringTraitExt,
    MultiLineStringTag
);

#[cfg(test)]
mod test {
    use geo::{Convert, CoordsIter};

    use super::*;
    use crate::wkt;

    #[test]
    fn test_linestring_inside_polygon() {
        let ls: LineString<f64> = wkt! {LINESTRING(1 1, 2 2)}.convert();
        let poly: Polygon<f64> = Rect::new((0, 0), (10, 10)).to_polygon().convert();
        assert!(ls.intersects(&poly));
    }

    #[test]
    fn test_linestring_partial_polygon() {
        let ls: LineString<f64> = wkt! {LINESTRING(-1 -1, 2 2)}.convert();
        let poly: Polygon<f64> = Rect::new((0, 0), (10, 10)).to_polygon().convert();
        assert!(ls.intersects(&poly));
    }
    #[test]
    fn test_linestring_disjoint_polygon() {
        let ls: LineString<f64> = wkt! {LINESTRING(-1 -1, -2 -2)}.convert();
        let poly: Polygon<f64> = Rect::new((0, 0), (10, 10)).to_polygon().convert();
        assert!(!ls.intersects(&poly));
    }
    #[test]
    fn test_linestring_in_polygon_hole() {
        let ls: LineString<f64> = wkt! {LINESTRING(4 4, 6 6)}.convert();
        let bound = Rect::new((0, 0), (10, 10)).convert();
        let hole = Rect::new((1, 1), (9, 9)).convert();
        let poly = Polygon::new(
            bound.exterior_coords_iter().collect(),
            vec![hole.exterior_coords_iter().collect()],
        );

        assert!(!ls.intersects(&poly));
    }

    // ls_rect
    #[test]
    fn test_linestring_inside_rect() {
        let ls: LineString<f64> = wkt! {LINESTRING(1 1, 2 2)}.convert();
        let poly: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        assert!(ls.intersects(&poly));
    }
    #[test]
    fn test_linestring_partial_rect() {
        let ls: LineString<f64> = wkt! {LINESTRING(-1 -1, 2 2)}.convert();
        let poly: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        assert!(ls.intersects(&poly));
    }
    #[test]
    fn test_linestring_disjoint_rect() {
        let ls: LineString<f64> = wkt! {LINESTRING(-1 -1, -2 -2)}.convert();
        let poly: Rect<f64> = Rect::new((0, 0), (10, 10)).convert();
        assert!(!ls.intersects(&poly));
    }

    // ls_triangle
    #[test]
    fn test_linestring_inside_triangle() {
        let ls: LineString<f64> = wkt! {LINESTRING(5 5, 5 4)}.convert();
        let poly: Triangle<f64> = wkt! {TRIANGLE(0 0, 10 0, 5 10)}.convert();
        assert!(ls.intersects(&poly));
    }
    #[test]
    fn test_linestring_partial_triangle() {
        let ls: LineString<f64> = wkt! {LINESTRING(5 5, 5 -4)}.convert();
        let poly: Triangle<f64> = wkt! {TRIANGLE(0 0, 10 0, 5 10)}.convert();
        assert!(ls.intersects(&poly));
    }
    #[test]
    fn test_linestring_disjoint_triangle() {
        let ls: LineString<f64> = wkt! {LINESTRING(5 -5, 5 -4)}.convert();
        let poly: Triangle<f64> = wkt! {TRIANGLE(0 0, 10 0, 5 10)}.convert();
        assert!(!ls.intersects(&poly));
    }
}
