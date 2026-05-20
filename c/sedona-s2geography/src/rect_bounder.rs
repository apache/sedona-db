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

use std::ptr;

use sedona_geometry::bounds::WkbBounder2D;
use sedona_geometry::error::SedonaGeometryError;
use sedona_geometry::interval::Interval;
use sedona_geometry::interval::IntervalTrait;
use sedona_geometry::interval::WraparoundInterval;
use sedona_geometry::wkb_header::WkbHeader;

use crate::geography::Geography;
use crate::geography::GeographyFactory;
use crate::s2geog_call;
use crate::s2geog_check;
use crate::s2geography_c_bindgen::*;
use crate::utils::S2GeogCError;

/// High level generic geography bounder implementation
///
/// This bounder implements [WkbBounder2D] for use in generic algorithms that need
/// rectangle bounds. This bounder is composed of two components: exact bounds
/// (represented by intervals) and the inexact component (represented by a
/// [RectBounder]). The inexact component is used to ingest new geometries and
/// the exact component is used to ingest precalculated bounds. This ensures that
/// the precalculated bounds are not expanded more than once (the [RectBounder]
/// expands bounds slightly to account for numerical errors when calculating
/// the bounds of geodesics).
#[derive(Debug, Default)]
pub struct WkbGeographyBounder {
    exact_x: WraparoundInterval,
    exact_y: Interval,
    inner: RectBounder,
    factory: GeographyFactory,
    geog: Geography<'static>,
}

impl WkbBounder2D for WkbGeographyBounder {
    fn update_bounds(
        &mut self,
        x: WraparoundInterval,
        y: Interval,
    ) -> Result<(), SedonaGeometryError> {
        self.exact_x = self.exact_x.merge_interval(&x);
        self.exact_y = self.exact_y.merge_interval(&y);
        Ok(())
    }

    fn update_wkb_bytes(&mut self, wkb_value: &[u8]) -> Result<(), SedonaGeometryError> {
        // Special-case the point because the rect bounder will expand the bounds slightly
        // and involves a roundtrip through a 3D point vector that makes this slightly
        // more faithful to the input.
        let header = WkbHeader::try_new(wkb_value)?;
        if header.geometry_type_id()? == sedona_geometry::types::GeometryTypeId::Point {
            let (x, y) = header.first_xy();
            self.exact_x = self.exact_x.merge_value(x);
            self.exact_y = self.exact_y.merge_value(y);
            return Ok(());
        }

        self.factory
            .init_from_wkb(wkb_value, &mut self.geog)
            .map_err(|e| SedonaGeometryError::External(Box::new(e)))?;
        self.inner
            .bound(&self.geog)
            .map_err(|e| SedonaGeometryError::External(Box::new(e)))?;

        Ok(())
    }

    fn expand_by_distance(
        &mut self,
        distance: f64,
        radius: Option<f64>,
    ) -> Result<(), SedonaGeometryError> {
        if distance > 0.0 {
            // Union the exact bounds with the inner implementation
            self.inner.update_rect(
                self.exact_x.lo(),
                self.exact_y.lo(),
                self.exact_x.hi(),
                self.exact_y.hi(),
            );

            // If we have an explicit radius, pass it on to s2geography. Otherwise, expand using the default
            if let Some(radius) = radius {
                self.inner.expand_by_distance_with_radius(distance, radius);
            } else {
                self.inner.expand_by_distance(distance);
            }
        }

        Ok(())
    }

    fn finish(&self) -> (WraparoundInterval, Interval) {
        let (mut x, mut y) = (self.exact_x, self.exact_y);

        let maybe_result = self.inner.finish();
        debug_assert!(maybe_result.is_ok());
        if let Some((xmin, ymin, xmax, ymax)) = self.inner.finish().unwrap_or_default() {
            x = x.merge_interval(&(xmin, xmax).into());
            y = y.merge_interval(&(ymin, ymax).into());
        }

        (x, y)
    }

    fn mem_used(&self) -> usize {
        // The RectBounder is roughly four additional doubles; the factory is roughly 64 bytes
        // since we don't use its internal coordinate storage. This may be slightly larger
        // (up to geometry with the largest number of nodes seen).
        size_of::<WkbGeographyBounder>() + self.geog.mem_used() + 4 * size_of::<f64>() + 64
    }
}

/// Safe wrapper around S2GeogRectBounder for computing bounding rectangles
///
/// This struct accumulates bounds from multiple geographies and can compute
/// the minimum bounding rectangle that contains all of them.
#[derive(Debug)]
pub struct RectBounder {
    ptr: *mut S2GeogRectBounder,
}

impl RectBounder {
    /// Create a new rect bounder
    pub fn new() -> Self {
        let mut ptr: *mut S2GeogRectBounder = ptr::null_mut();
        unsafe { s2geog_check!(S2GeogRectBounderCreate(&mut ptr)) }.unwrap();
        Self { ptr }
    }

    /// Clear the bounder, resetting it to an empty state
    pub fn clear(&mut self) {
        unsafe {
            S2GeogRectBounderClear(self.ptr);
        }
    }

    /// Add a geography to the bounding computation
    pub fn bound(&mut self, geog: &Geography) -> Result<(), S2GeogCError> {
        unsafe { s2geog_call!(S2GeogRectBounderBound(self.ptr, geog.as_ptr())) }
    }

    /// Perform the minimum expansion required to satisfy a distance expansion
    pub fn expand_by_distance(&mut self, distance_meters: f64) {
        unsafe { S2GeogRectBounderExpandByDistance(self.ptr, distance_meters) }
    }

    /// Perform the minimum expansion required to satisfy a distance expansion
    /// with a custom sphere radius
    ///
    /// This is useful when working with non-Earth spheres or custom projections.
    pub fn expand_by_distance_with_radius(&mut self, distance_meters: f64, radius: f64) {
        unsafe { S2GeogRectBounderExpandByDistanceWithRadius(self.ptr, distance_meters, radius) }
    }

    /// Update bounds with a rectangle specified by corners
    ///
    /// Coordinates are in degrees: x is longitude, y is latitude.
    pub fn update_rect(&mut self, x_lo: f64, y_lo: f64, x_hi: f64, y_hi: f64) {
        unsafe { S2GeogRectBounderUpdateRect(self.ptr, x_lo, y_lo, x_hi, y_hi) }
    }

    /// Check if the bounder is empty (no geometries or only empty geometries
    /// have been added)
    pub fn is_empty(&self) -> bool {
        unsafe { S2GeogRectBounderIsEmpty(self.ptr) != 0 }
    }

    /// Finish the bounding computation and return the bounding rectangle
    ///
    /// Returns `(xmin, ymin, xmax, ymax)` which represent the west, south, east, and
    /// north bounds of the geography. The xmin may be greater than xmax for the case
    /// where the geography wraps around the antimeridian.
    ///
    /// Returns `None` if the bounder is empty.
    pub fn finish(&self) -> Result<Option<(f64, f64, f64, f64)>, S2GeogCError> {
        if self.is_empty() {
            return Ok(None);
        }

        let mut lo = S2GeogVertex {
            v: [0.0, 0.0, 0.0, 0.0],
        };
        let mut hi = S2GeogVertex {
            v: [0.0, 0.0, 0.0, 0.0],
        };

        unsafe {
            s2geog_call!(S2GeogRectBounderFinish(self.ptr, &mut lo, &mut hi))?;
        }

        Ok(Some((lo.v[0], lo.v[1], hi.v[0], hi.v[1])))
    }
}

impl Default for RectBounder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RectBounder {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                S2GeogRectBounderDestroy(self.ptr);
            }
        }
    }
}

// Safety: RectBounder contains only a pointer to C++ data that is thread-safe
// when accessed through its const methods
unsafe impl Send for RectBounder {}

// Safety: RectBounder owns its C++ object exclusively and doesn't share state
// with other instances (mut methods ensure unique ownership)
unsafe impl Sync for RectBounder {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geography::GeographyFactory;

    #[test]
    fn test_rect_bounder_empty() {
        let bounder = RectBounder::new();
        assert!(bounder.is_empty());
        assert!(bounder.finish().unwrap().is_none());
    }

    #[test]
    fn test_rect_bounder_multiple_points() {
        let mut factory = GeographyFactory::new();

        let mut bounder = RectBounder::new();
        bounder
            .bound(&factory.from_wkt("POINT (179 0)").unwrap())
            .unwrap();
        bounder
            .bound(&factory.from_wkt("POINT (-179 20)").unwrap())
            .unwrap();

        assert!(!bounder.is_empty());
        let result = bounder.finish().unwrap();
        assert!(result.is_some());
        let (lo_lng, lo_lat, hi_lng, hi_lat) = result.unwrap();

        // Wraparound bounding box crossing the antimeridian
        // lo_lng > hi_lng indicates the interval wraps around
        assert!(lo_lng > 178.9, "lo_lng should be near 179: {lo_lng}");
        assert!(hi_lng < -178.9, "hi_lng should be near -179: {hi_lng}");
        assert!(lo_lat <= 0.0, "lo_lat should be <= 0: {lo_lat}");
        assert!(hi_lat >= 20.0, "hi_lat should be >= 20: {hi_lat}");

        bounder.expand_by_distance(100_000.0); // 100km
        let expanded = bounder.finish().unwrap().unwrap();
        assert!(expanded.0 < lo_lng);
        assert!(expanded.1 < lo_lat);
        assert!(expanded.2 > hi_lng);
        assert!(expanded.3 > hi_lat);

        bounder.clear();
        assert!(bounder.is_empty());
    }

    #[test]
    fn test_rect_bounder_expand_by_distance_with_radius() {
        let mut factory = GeographyFactory::new();

        // Test with half Earth's radius - should result in twice the angular expansion
        let mut bounder = RectBounder::new();
        bounder
            .bound(&factory.from_wkt("POINT (10 20)").unwrap())
            .unwrap();

        let half_earth_radius = 6371000.0 / 2.0; // Half of Earth's radius in meters
        bounder.expand_by_distance_with_radius(1000.0, half_earth_radius);

        let result = bounder.finish().unwrap().unwrap();
        let (lo_lng, lo_lat, hi_lng, hi_lat) = result;

        // With half the radius, 1km expands more (~0.018 degrees vs ~0.009 degrees)
        assert!(lo_lng < 10.0 - 0.016);
        assert!(hi_lng > 10.0 + 0.016);
        assert!(lo_lat < 20.0 - 0.016);
        assert!(hi_lat > 20.0 + 0.016);
    }

    #[test]
    fn test_rect_bounder_update_rect() {
        let mut bounder = RectBounder::new();

        // Update with a rectangle specified by corners
        bounder.update_rect(-10.0, -20.0, 30.0, 40.0);

        assert!(!bounder.is_empty());
        let result = bounder.finish().unwrap().unwrap();
        let (lo_lng, lo_lat, hi_lng, hi_lat) = result;

        // Bounds should match the input rectangle exactly (close to f64 precision)
        // due to roundtripping through radians
        assert!((lo_lng - (-10.0)).abs() < f64::EPSILON);
        assert!((lo_lat - (-20.0)).abs() < f64::EPSILON);
        assert!((hi_lng - 30.0).abs() < 1.0e-14);
        assert!((hi_lat - 40.0).abs() < f64::EPSILON);

        // Test wraparound case (crossing antimeridian: x_lo > x_hi)
        bounder.clear();
        bounder.update_rect(170.0, -10.0, -170.0, 10.0);

        assert!(!bounder.is_empty());
        let result = bounder.finish().unwrap().unwrap();
        let (lo_lng, lo_lat, hi_lng, hi_lat) = result;

        // Wraparound: lo_lng > hi_lng
        assert!((lo_lng - 170.0).abs() < f64::EPSILON);
        assert!((lo_lat - (-10.0)).abs() < f64::EPSILON);
        assert!((hi_lng - (-170.0)).abs() < f64::EPSILON);
        assert!((hi_lat - 10.0).abs() < f64::EPSILON);
    }

    // Helper to create WKB point bytes (little-endian)
    fn make_wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut wkb = Vec::with_capacity(21);
        wkb.push(0x01); // little-endian
        wkb.extend_from_slice(&1u32.to_le_bytes()); // Point type
        wkb.extend_from_slice(&x.to_le_bytes());
        wkb.extend_from_slice(&y.to_le_bytes());
        wkb
    }

    // Helper to create WKB linestring bytes (little-endian)
    fn make_wkb_linestring(coords: &[(f64, f64)]) -> Vec<u8> {
        let mut wkb = Vec::new();
        wkb.push(0x01); // little-endian
        wkb.extend_from_slice(&2u32.to_le_bytes()); // LineString type
        wkb.extend_from_slice(&(coords.len() as u32).to_le_bytes());
        for (x, y) in coords {
            wkb.extend_from_slice(&x.to_le_bytes());
            wkb.extend_from_slice(&y.to_le_bytes());
        }
        wkb
    }

    #[test]
    fn test_wkb_geography_bounder_empty() {
        let bounder = WkbGeographyBounder::default();
        let (x, y) = bounder.finish();
        assert!(x.is_empty());
        assert!(y.is_empty());
    }

    #[test]
    fn test_wkb_geography_bounder_multiple_points() {
        let mut bounder = WkbGeographyBounder::default();

        // Wraparound scenario: points at 179 and -179 longitude
        bounder
            .update_wkb_bytes(&make_wkb_point(179.0, 0.0))
            .unwrap();
        bounder
            .update_wkb_bytes(&make_wkb_point(-179.0, 20.0))
            .unwrap();

        let (x, y) = bounder.finish();

        // Wraparound bounding box crossing the antimeridian
        // lo > hi indicates the interval wraps around
        assert!(x.lo() > 178.9, "x.lo() should be near 179: {}", x.lo());
        assert!(x.hi() < -178.9, "x.hi() should be near -179: {}", x.hi());
        assert!(y.lo() <= 0.0, "y.lo() should be <= 0: {}", y.lo());
        assert!(y.hi() >= 20.0, "y.hi() should be >= 20: {}", y.hi());
    }

    #[test]
    fn test_wkb_geography_bounder_expand_by_distance() {
        let mut bounder = WkbGeographyBounder::default();

        // Wraparound scenario
        bounder
            .update_wkb_bytes(&make_wkb_point(179.0, 0.0))
            .unwrap();
        bounder
            .update_wkb_bytes(&make_wkb_point(-179.0, 20.0))
            .unwrap();

        let (x_before, y_before) = bounder.finish();

        bounder.expand_by_distance(100_000.0, None).unwrap(); // 100km
        let (x_after, y_after) = bounder.finish();

        // Expanded bounds should be larger (further from center)
        // For wraparound: lo should increase (move east), hi should decrease (move west)
        assert!(
            x_after.lo() < x_before.lo(),
            "x.lo() should expand: {} < {}",
            x_after.lo(),
            x_before.lo()
        );
        assert!(
            x_after.hi() > x_before.hi(),
            "x.hi() should expand: {} > {}",
            x_after.hi(),
            x_before.hi()
        );
        assert!(
            y_after.lo() < y_before.lo(),
            "y.lo() should expand: {} < {}",
            y_after.lo(),
            y_before.lo()
        );
        assert!(
            y_after.hi() > y_before.hi(),
            "y.hi() should expand: {} > {}",
            y_after.hi(),
            y_before.hi()
        );
    }

    #[test]
    fn test_wkb_geography_bounder_expand_by_distance_with_radius() {
        let mut bounder = WkbGeographyBounder::default();

        bounder
            .update_wkb_bytes(&make_wkb_point(10.0, 20.0))
            .unwrap();

        let half_earth_radius = 6371000.0 / 2.0; // Half of Earth's radius in meters
        bounder
            .expand_by_distance(1000.0, Some(half_earth_radius))
            .unwrap();

        let (x, y) = bounder.finish();

        // With half the radius, 1km expands more (~0.018 degrees vs ~0.009 degrees)
        assert!(x.lo() < 10.0 - 0.016);
        assert!(x.hi() > 10.0 + 0.016);
        assert!(y.lo() < 20.0 - 0.016);
        assert!(y.hi() > 20.0 + 0.016);
    }

    #[test]
    fn test_wkb_geography_bounder_update_bounds() {
        let mut bounder = WkbGeographyBounder::default();

        // Update with precalculated bounds
        bounder
            .update_bounds((-10.0, 30.0).into(), (-20.0, 40.0).into())
            .unwrap();

        let (x, y) = bounder.finish();

        // Bounds should match exactly
        assert_eq!(x.lo(), -10.0);
        assert_eq!(x.hi(), 30.0);
        assert_eq!(y.lo(), -20.0);
        assert_eq!(y.hi(), 40.0);

        // Test wraparound case
        let mut bounder = WkbGeographyBounder::default();
        bounder
            .update_bounds((170.0, -170.0).into(), (-10.0, 10.0).into())
            .unwrap();

        let (x, y) = bounder.finish();

        // Wraparound: lo > hi
        assert_eq!(x.lo(), 170.0);
        assert_eq!(x.hi(), -170.0);
        assert_eq!(y.lo(), -10.0);
        assert_eq!(y.hi(), 10.0);
    }

    #[test]
    fn test_wkb_geography_bounder_linestring() {
        let mut bounder = WkbGeographyBounder::default();

        // LineString that doesn't get special-cased like Point does
        let linestring = make_wkb_linestring(&[(10.0, 20.0), (30.0, 40.0)]);
        bounder.update_wkb_bytes(&linestring).unwrap();

        let (x, y) = bounder.finish();

        // Bounds should encompass the linestring (with some tolerance for geodesic calculation)
        assert!(x.lo() <= 10.1, "x.lo() should be <= 10.0: {}", x.lo());
        assert!(x.hi() >= 29.9, "x.hi() should be >= 30.0: {}", x.hi());
        assert!(y.lo() <= 20.1, "y.lo() should be <= 20.0: {}", y.lo());
        assert!(y.hi() >= 39.9, "y.hi() should be >= 40.0: {}", y.hi());
    }

    #[test]
    fn test_wkb_geography_bounder_mem_used() {
        let bounder = WkbGeographyBounder::default();
        let mem = bounder.mem_used();
        // Should be at least the size of the struct itself
        assert!(mem >= size_of::<WkbGeographyBounder>());
    }
}
