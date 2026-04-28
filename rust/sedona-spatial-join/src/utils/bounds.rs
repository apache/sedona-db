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
use geo::{Coord, Rect};
use sedona_geometry::interval::{Interval, IntervalTrait, WraparoundInterval};

#[derive(Debug, Clone, PartialEq)]
pub struct Bounds2D {
    x: (f32, f32),
    y: (f32, f32),
}

impl Bounds2D {
    pub fn empty() -> Self {
        Self::new(WraparoundInterval::empty(), Interval::empty())
    }

    pub fn is_empty(&self) -> bool {
        self.x().is_empty() && self.y().is_empty()
    }

    pub fn new(x: impl Into<WraparoundInterval>, y: impl Into<Interval>) -> Self {
        let x: WraparoundInterval = x.into();
        let y: Interval = y.into();

        let x_float = if x.is_wraparound() {
            let swapped_neg_x_float = f64_interval_to_f32(-x.hi(), -x.lo());
            (-swapped_neg_x_float.1, -swapped_neg_x_float.0)
        } else {
            f64_interval_to_f32(x.lo(), x.hi())
        };

        Self::new_from_raw(x_float, f64_interval_to_f32(y.lo(), y.hi()))
    }

    pub fn new_from_raw(x: (f32, f32), y: (f32, f32)) -> Self {
        Self { x, y }
    }

    pub fn is_wraparound(&self) -> bool {
        !self.is_empty() && self.x.0 > self.x.1
    }

    pub fn split(&self) -> (Bounds2D, Bounds2D) {
        if self.is_wraparound() {
            let x_left = (-f32::INFINITY, self.x.1);
            let x_right = (self.x.0, f32::INFINITY);
            (
                Self {
                    x: x_left,
                    y: self.y,
                },
                Self {
                    x: x_right,
                    y: self.y,
                },
            )
        } else {
            (self.clone(), Self::empty())
        }
    }

    pub fn into_inner(self) -> ((f32, f32), (f32, f32)) {
        (self.x, self.y)
    }

    /// Create a Bounds2D from a geo::Rect<f32>
    pub fn from_geo_rect(rect: &Rect<f32>) -> Self {
        let min = rect.min();
        let max = rect.max();
        // No need for next_after since we're already in f32
        Self {
            x: (min.x, max.x),
            y: (min.y, max.y),
        }
    }

    pub fn x(&self) -> WraparoundInterval {
        WraparoundInterval::new(self.x.0 as f64, self.x.1 as f64)
    }

    pub fn y(&self) -> Interval {
        Interval::new(self.y.0 as f64, self.y.1 as f64)
    }

    /// Convert to a tuple of (min_x, min_y, max_x, max_y) as f32 values
    ///
    /// Returns None if the bounds are empty.
    pub fn to_f32_tuple(&self) -> Option<(f32, f32, f32, f32)> {
        if self.is_empty() {
            None
        } else {
            Some((self.x.0, self.y.0, self.x.1, self.y.1))
        }
    }

    /// Convert to a geo::Rect<f32>
    ///
    /// Returns None if the bounds are empty.
    pub fn to_geo_rect(&self) -> Option<Rect<f32>> {
        self.to_f32_tuple().map(|(min_x, min_y, max_x, max_y)| {
            Rect::new(Coord { x: min_x, y: min_y }, Coord { x: max_x, y: max_y })
        })
    }

    /// Returns `true` if this bounds intersects with another (including touching edges).
    pub fn intersects(&self, other: &Bounds2D) -> bool {
        self.x.0 <= other.x.1
            && self.x.1 >= other.x.0
            && self.y.0 <= other.y.1
            && self.y.1 >= other.y.0
    }

    /// Returns the intersection area between this bounds and another.
    pub fn intersection_area(&self, other: &Bounds2D) -> f32 {
        if !self.intersects(other) {
            return 0.0;
        }

        let min_x = self.x.0.max(other.x.0);
        let min_y = self.y.0.max(other.y.0);
        let max_x = self.x.1.min(other.x.1);
        let max_y = self.y.1.min(other.y.1);

        (max_x - min_x).max(0.0) * (max_y - min_y).max(0.0)
    }

    /// Returns `true` if this bounds contains the given point.
    pub fn contains_point(&self, point: &Coord<f32>) -> bool {
        point.x >= self.x.0 && point.x <= self.x.1 && point.y >= self.y.0 && point.y <= self.y.1
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
        let (min_x, min_y, max_x, max_y) =
            bounds.to_f32_tuple().expect("bounds should not be empty");

        assert_eq!(min_x, 0.0f32);
        assert_eq!(min_y, 0.0f32);
        assert!(max_x >= 100.0f32);
        assert!(max_y >= 100.0f32);
    }

    #[test]
    fn test_bounds2d_negative() {
        let bounds = Bounds2D::new((-50.0, 50.0), (-50.0, 50.0));
        let (min_x, min_y, max_x, max_y) =
            bounds.to_f32_tuple().expect("bounds should not be empty");

        assert_eq!(min_x, -50.0f32);
        assert_eq!(min_y, -50.0f32);
        assert!(max_x >= 50.0f32);
        assert!(max_y >= 50.0f32);
    }

    #[test]
    fn test_bounds2d_preserves_bounds() {
        let bounds = Bounds2D::new((10.5, 20.7), (30.3, 40.9));
        let (min_x, min_y, max_x, max_y) =
            bounds.to_f32_tuple().expect("bounds should not be empty");

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
    fn test_bounds2d_large_values() {
        let bounds = Bounds2D::new((-180.0, 180.0), (-90.0, 90.0));
        let (min_x, min_y, max_x, max_y) =
            bounds.to_f32_tuple().expect("bounds should not be empty");

        assert_eq!(min_x, -180.0f32);
        assert_eq!(min_y, -90.0f32);
        assert!(max_x >= 180.0f32);
        assert!(max_y >= 90.0f32);
    }

    #[test]
    fn test_bounds2d_intersects_and_area() {
        let a = Bounds2D::new((0.0, 10.0), (0.0, 10.0));
        let b = Bounds2D::new((5.0, 15.0), (5.0, 15.0));
        assert!(a.intersects(&b));
        assert!(a.intersection_area(&b) > 0.0);

        let c = Bounds2D::new((20.0, 30.0), (20.0, 30.0));
        assert!(!a.intersects(&c));
        assert_eq!(a.intersection_area(&c), 0.0);
    }

    #[test]
    fn test_bounds2d_empty() {
        let bounds = Bounds2D::new(Interval::empty(), Interval::empty());
        assert!(bounds.to_f32_tuple().is_none());
        assert!(bounds.to_geo_rect().is_none());
    }

    #[test]
    fn test_bounds2d_contains_point() {
        let bounds = Bounds2D::new((0.0, 10.0), (0.0, 10.0));
        assert!(bounds.contains_point(&Coord { x: 5.0, y: 5.0 }));
        assert!(bounds.contains_point(&Coord { x: 0.0, y: 0.0 }));
        assert!(!bounds.contains_point(&Coord { x: 15.0, y: 5.0 }));
    }

    #[test]
    fn test_bounds2d_to_geo_rect_roundtrip() {
        let bounds = Bounds2D::new((10.0, 20.0), (30.0, 40.0));
        let rect = bounds.to_geo_rect().expect("bounds should not be empty");
        let roundtrip = Bounds2D::from_geo_rect(&rect);

        // from_geo_rect doesn't apply next_after, so values should be close
        assert!((roundtrip.x.0 - bounds.x.0).abs() < 1e-5);
        assert!((roundtrip.x.1 - bounds.x.1).abs() < 1e-5);
        assert!((roundtrip.y.0 - bounds.y.0).abs() < 1e-5);
        assert!((roundtrip.y.1 - bounds.y.1).abs() < 1e-5);
    }
}
