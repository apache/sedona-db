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
    ) -> Result<(), sedona_geometry::error::SedonaGeometryError> {
        self.exact_x = self.exact_x.merge_interval(&x);
        self.exact_y = self.exact_y.merge_interval(&y);
        Ok(())
    }

    fn update_wkb_bytes(
        &mut self,
        wkb_value: &[u8],
    ) -> Result<(), sedona_geometry::error::SedonaGeometryError> {
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
            .map_err(|e| sedona_geometry::error::SedonaGeometryError::External(Box::new(e)))?;
        self.inner
            .bound(&self.geog)
            .map_err(|e| sedona_geometry::error::SedonaGeometryError::External(Box::new(e)))?;

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
            .bound(&factory.from_wkt("POINT (0 0)").unwrap())
            .unwrap();
        bounder
            .bound(&factory.from_wkt("POINT (10 20)").unwrap())
            .unwrap();

        assert!(!bounder.is_empty());
        let result = bounder.finish().unwrap();
        assert!(result.is_some());
        let (lo_lng, lo_lat, hi_lng, hi_lat) = result.unwrap();

        // Bounding box should encompass both points
        assert!(lo_lng <= 0.0);
        assert!(lo_lat <= 0.0);
        assert!(hi_lng >= 10.0);
        assert!(hi_lat >= 20.0);

        bounder.expand_by_distance(100_000.0); // 100km
        let expanded = bounder.finish().unwrap().unwrap();
        assert!(expanded.0 < lo_lng);
        assert!(expanded.1 < lo_lat);
        assert!(expanded.2 > hi_lng);
        assert!(expanded.3 > hi_lat);

        bounder.clear();
        assert!(bounder.is_empty());
    }
}
