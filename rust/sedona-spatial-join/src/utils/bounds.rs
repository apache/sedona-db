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

use float_next_after::NextAfter;
use sedona_geometry::{
    bounding_box::BoundingBox,
    interval::{Interval, IntervalTrait, WraparoundInterval},
};

/// A float32 bounding box with wraparound support
///
/// This struct is conceptually similar to the Rect<f32> but explicitly supports
/// wraparound x intervals to ensure raw xmin and xmax values are not misused.
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds2D {
    x: (f32, f32),
    y: (f32, f32),
}

impl Bounds2D {
    /// Create a new empty rectangle
    pub fn empty() -> Self {
        Self::new(WraparoundInterval::empty(), Interval::empty())
    }

    /// Returns true if this rectangle is empty
    pub fn is_empty(&self) -> bool {
        self.x().is_empty() || self.y().is_empty()
    }

    /// Create a new Bounds2D from f64 intervals
    ///
    /// This constructor ensures that the resulting f32 bounds are enlarged to contain
    /// the original intervals.
    pub fn new(x: impl Into<WraparoundInterval>, y: impl Into<Interval>) -> Self {
        let x: WraparoundInterval = x.into();
        let y: Interval = y.into();

        // For wraparound intervals we need to make the min and max slightly closer to
        // each other rather than slightly farther away.
        let x_float = if x.is_wraparound() {
            let swapped_neg_x_float = f64_interval_to_f32(-x.hi(), -x.lo());
            (-swapped_neg_x_float.1, -swapped_neg_x_float.0)
        } else {
            f64_interval_to_f32(x.lo(), x.hi())
        };

        Self::new_from_raw(x_float, f64_interval_to_f32(y.lo(), y.hi()))
    }

    /// Create a new Bounds2D from raw f32 intervals
    pub fn new_from_raw(x: (f32, f32), y: (f32, f32)) -> Self {
        Self { x, y }
    }

    /// Return true if the x interval of this box wraps around or false otherwise
    pub fn is_wraparound(&self) -> bool {
        !self.is_empty() && self.x.0 > self.x.1
    }

    /// Split this Bounds2D into two bounds where neither wraps around
    ///
    /// Note that one or both of the outputs may be empty or contain infinite
    /// bounds.
    pub fn split(&self, wraparound: &Interval) -> (Bounds2D, Bounds2D) {
        if self.is_wraparound() {
            let (x_left, x_right) = self.x().split();
            let y = self.y();
            (
                Self::new(
                    x_left.intersection(wraparound).unwrap_or(Interval::full()),
                    y,
                ),
                Self::new(
                    x_right.intersection(wraparound).unwrap_or(Interval::full()),
                    y,
                ),
            )
        } else {
            (self.clone(), Self::empty())
        }
    }

    /// Decompose this object into raw x and y intervals
    ///
    /// Note that the x interval may wrap such that xmin > xmax.
    pub fn into_inner(self) -> ((f32, f32), (f32, f32)) {
        (self.x, self.y)
    }

    /// Return the x [WraparoundInterval] for these bounds
    pub fn x(&self) -> WraparoundInterval {
        WraparoundInterval::new(self.x.0 as f64, self.x.1 as f64)
    }

    /// Return the y [Interval] for these bounds
    pub fn y(&self) -> Interval {
        Interval::new(self.y.0 as f64, self.y.1 as f64)
    }

    /// Returns `true` if this bounds intersects with another (including touching edges).
    pub fn intersects(&self, other: &Bounds2D) -> bool {
        self.x().intersects_interval(&other.x()) && self.y().intersects_interval(&other.y())
    }

    /// Returns `true` if this bounds intersects the given point (including touching edges)
    pub fn intersects_point(&self, point: (f64, f64)) -> bool {
        self.x().intersects_value(point.0) && self.y().intersects_value(point.1)
    }
}

impl From<&Bounds2D> for BoundingBox {
    fn from(val: &Bounds2D) -> Self {
        BoundingBox::xy(val.x(), val.y())
    }
}

fn f64_interval_to_f32(min_x: f64, max_x: f64) -> (f32, f32) {
    let mut new_min_x = min_x as f32;
    let mut new_max_x = max_x as f32;

    if (new_min_x as f64) > min_x {
        new_min_x = new_min_x.next_after(f32::NEG_INFINITY);
    }
    if (new_max_x as f64) < max_x {
        new_max_x = new_max_x.next_after(f32::INFINITY);
    }

    debug_assert!((new_min_x as f64) <= min_x);
    debug_assert!((new_max_x as f64) >= max_x);

    (new_min_x, new_max_x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds2d_simple() {
        let bounds = Bounds2D::new((0.0, 100.0), (0.0, 100.0));
        let ((min_x, max_x), (min_y, max_y)) = bounds.into_inner();

        assert_eq!(min_x, 0.0f32);
        assert_eq!(min_y, 0.0f32);
        assert!(max_x >= 100.0f32);
        assert!(max_y >= 100.0f32);
    }

    #[test]
    fn test_bounds2d_negative() {
        let bounds = Bounds2D::new((-50.0, 50.0), (-50.0, 50.0));
        let ((min_x, max_x), (min_y, max_y)) = bounds.into_inner();

        assert_eq!(min_x, -50.0f32);
        assert_eq!(min_y, -50.0f32);
        assert!(max_x >= 50.0f32);
        assert!(max_y >= 50.0f32);
    }

    #[test]
    fn test_bounds2d_preserves_bounds() {
        let bounds = Bounds2D::new((10.5, 20.7), (30.3, 40.9));
        let ((min_x, max_x), (min_y, max_y)) = bounds.into_inner();

        // Min bounds should be <= the original values (rounded down if needed)
        assert!(min_x <= 10.5f32);
        assert!(min_y <= 30.3f32);

        // Max bounds should be >= the original values (next representable f32)
        assert!(max_x >= 20.7f32);
        assert!(max_y >= 40.9f32);

        // The inclusive original bounds should be strictly less than the exclusive bounds
        let max_x_inclusive = 20.7f32;
        let max_y_inclusive = 40.9f32;
        assert!(max_x >= max_x_inclusive);
        assert!(max_y >= max_y_inclusive);
    }

    #[test]
    fn test_bounds2d_contains_point() {
        let bounds = Bounds2D::new((0.0, 10.0), (0.0, 10.0));
        assert!(bounds.intersects_point((5.0, 5.0)));
        assert!(bounds.intersects_point((0.0, 0.0)));
        assert!(!bounds.intersects_point((15.0, 5.0)));
    }

    #[test]
    fn test_bounds2d_preserves_bounds_wraparound() {
        // Wraparound interval from 170.5 to -170.3 (crossing antimeridian)
        let x_interval = WraparoundInterval::new(170.5, -170.3);
        let y_interval = Interval::new(30.3, 40.9);
        let bounds = Bounds2D::new(x_interval, y_interval);
        let ((min_x, max_x), (min_y, max_y)) = bounds.clone().into_inner();

        // For wraparound, min_x > max_x
        assert!(min_x > max_x, "wraparound should have min_x > max_x");

        // Y bounds should be enlarged to contain the original interval
        assert!(min_y <= 30.3f32);
        assert!(max_y >= 40.9f32);

        // The f32 bounds should properly represent the original interval
        // Verify by checking that the converted-back interval contains points from the original
        let converted_x = bounds.x();
        assert!(
            converted_x.intersects_value(170.5),
            "converted interval should contain original lo"
        );
        assert!(
            converted_x.intersects_value(-170.3),
            "converted interval should contain original hi"
        );
        assert!(
            converted_x.intersects_value(180.0),
            "converted interval should contain points in the wraparound region"
        );
        assert!(
            converted_x.intersects_value(-180.0),
            "converted interval should contain points in the wraparound region"
        );
    }

    #[test]
    fn test_bounds2d_intersects_and_area_wraparound() {
        // Wraparound box crossing the antimeridian
        let a = Bounds2D::new(WraparoundInterval::new(170.0, -170.0), (0.0, 10.0));

        // Regular box on the positive side - should intersect
        let b = Bounds2D::new((175.0, 180.0), (5.0, 15.0));
        assert!(a.intersects(&b));

        // Regular box on the negative side - should intersect
        let c = Bounds2D::new((-180.0, -175.0), (5.0, 15.0));
        assert!(a.intersects(&c));

        // Regular box in the middle (not crossing) - should NOT intersect
        let d = Bounds2D::new((0.0, 10.0), (0.0, 10.0));
        assert!(!a.intersects(&d));

        // Two wraparound boxes that overlap
        let e = Bounds2D::new(WraparoundInterval::new(175.0, -175.0), (5.0, 15.0));
        assert!(a.intersects(&e));
    }

    #[test]
    fn test_bounds2d_contains_point_wraparound() {
        // Wraparound box from 170 to -170 (covering 170 to 180 and -180 to -170)
        let bounds = Bounds2D::new(WraparoundInterval::new(170.0, -170.0), (0.0, 10.0));

        // Point inside on the positive side
        assert!(bounds.intersects_point((175.0, 5.0)));
        // Point inside on the negative side
        assert!(bounds.intersects_point((-175.0, 5.0)));
        // Point at the boundary
        assert!(bounds.intersects_point((170.0, 0.0)));
        // Point outside (in the gap)
        assert!(!bounds.intersects_point((0.0, 5.0)));
        // Point outside (wrong y)
        assert!(!bounds.intersects_point((175.0, 15.0)));
    }
}
