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

//! Utility functions for spatial partitioning.

use datafusion_common::{DataFusionError, Result};
use geo::{Coord, CoordNum, Rect};
use geo_index::rtree::util::f64_box_to_f32;
use sedona_geometry::bounding_box::BoundingBox;
use sedona_geometry::interval::IntervalTrait;

/// Convert a BoundingBox to f32 coordinates
///
/// # Arguments
/// * `bbox` - The BoundingBox to convert
///
/// # Returns
/// A tuple of (min_x, min_y, max_x, max_y) as f32 values
///
/// # Errors
/// Returns an error if the bounding box has wraparound coordinates (e.g., crossing the anti-meridian)
pub(crate) fn bbox_to_f32_rect(bbox: &BoundingBox) -> Result<Option<(f32, f32, f32, f32)>> {
    // Check for wraparound coordinates
    if bbox.x().is_wraparound() {
        return Err(DataFusionError::Execution(
            "BoundingBox has wraparound coordinates, which is not supported yet".to_string(),
        ));
    }

    let min_x = bbox.x().lo();
    let min_y = bbox.y().lo();
    let max_x = bbox.x().hi();
    let max_y = bbox.y().hi();

    if min_x <= max_x && min_y <= max_y {
        Ok(Some(f64_box_to_f32(min_x, min_y, max_x, max_y)))
    } else {
        Ok(None)
    }
}

/// Convert a [`BoundingBox`] into a [`Rect<f32>`] with the same adjusted bounds as
/// [`bbox_to_f32_rect`].
pub(crate) fn bbox_to_geo_rect(bbox: &BoundingBox) -> Result<Option<Rect<f32>>> {
    if let Some((min_x, min_y, max_x, max_y)) = bbox_to_f32_rect(bbox)? {
        Ok(Some(make_rect(min_x, min_y, max_x, max_y)))
    } else {
        Ok(None)
    }
}

/// Creates a `Rect` from four coordinate values representing the bounding box.
///
/// This is a convenience function that constructs a `geo::Rect` from individual
/// coordinate components.
pub(crate) fn make_rect<T: CoordNum>(xmin: T, ymin: T, xmax: T, ymax: T) -> Rect<T> {
    Rect::new(Coord { x: xmin, y: ymin }, Coord { x: xmax, y: ymax })
}

/// Returns `true` if two rectangles intersect (including touching edges).
pub(crate) fn rects_intersect(a: &Rect<f32>, b: &Rect<f32>) -> bool {
    let (a_min, a_max) = (a.min(), a.max());
    let (b_min, b_max) = (b.min(), b.max());

    a_min.x <= b_max.x && a_max.x >= b_min.x && a_min.y <= b_max.y && a_max.y >= b_min.y
}

/// Returns the intersection area between two rectangles.
pub(crate) fn rect_intersection_area(a: &Rect<f32>, b: &Rect<f32>) -> f32 {
    if !rects_intersect(a, b) {
        return 0.0;
    }

    let (a_min, a_max) = (a.min(), a.max());
    let (b_min, b_max) = (b.min(), b.max());

    let min_x = a_min.x.max(b_min.x);
    let min_y = a_min.y.max(b_min.y);
    let max_x = a_max.x.min(b_max.x);
    let max_y = a_max.y.min(b_max.y);

    (max_x - min_x).max(0.0) * (max_y - min_y).max(0.0)
}

/// Returns `true` if the rectangle contains the given point.
pub(crate) fn rect_contains_point(rect: &Rect<f32>, point: &Coord<f32>) -> bool {
    let min = rect.min();
    let max = rect.max();
    point.x >= min.x && point.x <= max.x && point.y >= min.y && point.y <= max.y
}

#[cfg(test)]
mod tests {
    use sedona_geometry::interval::Interval;

    use super::*;

    #[test]
    fn test_bbox_to_f32_rect_simple() {
        let bbox = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let (min_x, min_y, max_x, max_y) = bbox_to_f32_rect(&bbox)
            .expect("bbox_to_f32_rect failed")
            .expect("bbox should not be empty");

        assert_eq!(min_x, 0.0f32);
        assert_eq!(min_y, 0.0f32);
        assert!(max_x >= 100.0f32);
        assert!(max_y >= 100.0f32);
    }

    #[test]
    fn test_bbox_to_f32_rect_negative() {
        let bbox = BoundingBox::xy((-50.0, 50.0), (-50.0, 50.0));
        let (min_x, min_y, max_x, max_y) = bbox_to_f32_rect(&bbox)
            .expect("bbox_to_f32_rect failed")
            .expect("bbox should not be empty");

        assert_eq!(min_x, -50.0f32);
        assert_eq!(min_y, -50.0f32);
        assert!(max_x >= 50.0f32);
        assert!(max_y >= 50.0f32);
    }

    #[test]
    fn test_bbox_to_f32_rect_preserves_bounds() {
        let bbox = BoundingBox::xy((10.5, 20.7), (30.3, 40.9));
        let (min_x, min_y, max_x, max_y) = bbox_to_f32_rect(&bbox)
            .expect("bbox_to_f32_rect failed")
            .expect("bbox should not be empty");

        // Min bounds should be <= the original values (rounded down if needed)
        assert!(min_x <= 10.5f32);
        assert!(min_y <= 30.3f32);

        // Max bounds should be > the original values (next representable f32)
        assert!(max_x >= 20.7f32);
        assert!(max_y >= 40.9f32);

        // The inclusive original bounds should be strictly less than the exclusive bounds
        let max_x_inclusive = 20.7f32;
        let max_y_inclusive = 40.9f32;
        assert!(max_x >= max_x_inclusive);
        assert!(max_y >= max_y_inclusive);
    }

    #[test]
    fn test_bbox_to_f32_rect_large_values() {
        let bbox = BoundingBox::xy((-180.0, 180.0), (-90.0, 90.0));
        let (min_x, min_y, max_x, max_y) = bbox_to_f32_rect(&bbox)
            .expect("bbox_to_f32_rect failed")
            .expect("bbox should not be empty");

        assert_eq!(min_x, -180.0f32);
        assert_eq!(min_y, -90.0f32);
        assert!(max_x >= 180.0f32);
        assert!(max_y >= 90.0f32);
    }

    #[test]
    fn test_bbox_to_f32_rect_wraparound_rejected() {
        // Create a bbox with wraparound (crossing anti-meridian)
        use sedona_geometry::interval::WraparoundInterval;
        let bbox = BoundingBox::xy(WraparoundInterval::new(170.0, -170.0), (0.0, 50.0));
        let result = bbox_to_f32_rect(&bbox);
        assert!(result.is_err());
    }

    #[test]
    fn test_rect_helpers() {
        let a = make_rect(0.0_f32, 0.0_f32, 10.0_f32, 10.0_f32);
        let b = make_rect(5.0_f32, 5.0_f32, 15.0_f32, 15.0_f32);
        assert!(rects_intersect(&a, &b));
        assert!(rect_intersection_area(&a, &b) > 0.0);

        let c = make_rect(20.0_f32, 20.0_f32, 30.0_f32, 30.0_f32);
        assert!(!rects_intersect(&a, &c));
        assert_eq!(rect_intersection_area(&a, &c), 0.0);
    }

    #[test]
    fn test_bbox_to_f32_rect_empty() {
        let bbox = BoundingBox::xy(Interval::empty(), Interval::empty());
        assert!(bbox_to_f32_rect(&bbox).unwrap().is_none());
        assert!(bbox_to_geo_rect(&bbox).unwrap().is_none());
    }
}
