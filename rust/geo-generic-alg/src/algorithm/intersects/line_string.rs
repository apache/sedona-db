use geo_traits_ext::*;

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
impl_intersects_line_string_from_line!(PolygonTraitExt, PolygonTag);
impl_intersects_line_string_from_line!(MultiPointTraitExt, MultiPointTag);
impl_intersects_line_string_from_line!(MultiLineStringTraitExt, MultiLineStringTag);
impl_intersects_line_string_from_line!(MultiPolygonTraitExt, MultiPolygonTag);
impl_intersects_line_string_from_line!(GeometryTraitExt, GeometryTag);
impl_intersects_line_string_from_line!(GeometryCollectionTraitExt, GeometryCollectionTag);
impl_intersects_line_string_from_line!(LineTraitExt, LineTag);
impl_intersects_line_string_from_line!(RectTraitExt, RectTag);
impl_intersects_line_string_from_line!(TriangleTraitExt, TriangleTag);

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
