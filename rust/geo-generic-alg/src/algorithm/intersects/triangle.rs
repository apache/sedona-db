use super::IntersectsTrait;
use crate::*;
use geo_traits_ext::*;

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
