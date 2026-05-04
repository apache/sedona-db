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

use crate::geography::Geography;
use crate::s2geog_call;
use crate::s2geog_check;
use crate::s2geography_c_bindgen::*;
use crate::utils::S2GeogCError;

/// Safe wrapper around S2GeogRectBounder for computing bounding rectangles
///
/// This struct accumulates bounds from multiple geographies and can compute
/// the minimum bounding rectangle that contains all of them.
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
