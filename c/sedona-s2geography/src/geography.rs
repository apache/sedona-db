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

use crate::s2geog_call;
use crate::s2geog_check;
use crate::s2geography_c_bindgen::*;
use crate::utils::S2GeogCError;

/// Safe wrapper around an S2Geog geography object
pub struct Geography {
    ptr: *mut S2Geog,
}

impl Geography {
    /// Create a new empty geography
    pub fn new() -> Self {
        let mut ptr: *mut S2Geog = ptr::null_mut();
        unsafe { s2geog_check!(S2GeogCreate(&mut ptr)) }.unwrap();
        Self { ptr }
    }

    /// Force building the internal shape index for this geography
    ///
    /// Most operations attempt to avoid building the internal ShapeIndex
    /// because doing so can be slow; however, for repeated calls to the same
    /// function (e.g., predicates), it can be faster to force the creation
    /// of an index.
    pub fn prepare(&mut self) -> Result<(), S2GeogCError> {
        unsafe { s2geog_call!(S2GeogForcePrepare(self.ptr)) }
    }
}

impl Default for Geography {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Geography {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                S2GeogDestroy(self.ptr);
            }
        }
    }
}

// Safety: Geography contains only a pointer to C++ data that is thread-safe
// when accessed through const methods
unsafe impl Send for Geography {}

/// Factory for creating Geography objects from various formats
pub struct GeographyFactory {
    ptr: *mut S2GeogFactory,
}

impl GeographyFactory {
    /// Create a new geography factory
    pub fn new() -> Self {
        let mut ptr: *mut S2GeogFactory = ptr::null_mut();
        unsafe { s2geog_check!(S2GeogFactoryCreate(&mut ptr)) }.unwrap();
        Self { ptr }
    }

    /// Create a geography from WKB bytes
    pub fn from_wkb(&mut self, wkb: &[u8]) -> Result<Geography, S2GeogCError> {
        let mut geog = Geography::new();
        self.init_from_wkb(wkb, &mut geog)?;
        Ok(geog)
    }

    /// Create a geography from WKT string
    pub fn from_wkt(&mut self, wkt: &str) -> Result<Geography, S2GeogCError> {
        let mut geog = Geography::new();
        self.init_from_wkt(wkt, &mut geog)?;
        Ok(geog)
    }

    fn init_from_wkb(&mut self, wkb: &[u8], geog: &mut Geography) -> Result<(), S2GeogCError> {
        unsafe {
            s2geog_call!(S2GeogFactoryInitFromWkbNonOwning(
                self.ptr,
                wkb.as_ptr(),
                wkb.len(),
                geog.ptr,
            ))
        }
    }

    fn init_from_wkt(&mut self, wkt: &str, geog: &mut Geography) -> Result<(), S2GeogCError> {
        unsafe {
            s2geog_call!(S2GeogFactoryInitFromWkt(
                self.ptr,
                wkt.as_ptr() as *const _,
                wkt.len(),
                geog.ptr,
            ))
        }
    }
}

impl Default for GeographyFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GeographyFactory {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                S2GeogFactoryDestroy(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geography_from_wkt() {
        let mut factory = GeographyFactory::new();
        let mut geog = factory.from_wkt("POINT (0 1)").unwrap();
        geog.prepare().unwrap();
    }

    #[test]
    fn test_geography_from_wkb() {
        let wkb_bytes = sedona_testing::create::make_wkb("POINT (0 1)");
        let mut factory = GeographyFactory::new();
        let mut geog = factory.from_wkb(&wkb_bytes).unwrap();
        geog.prepare().unwrap();
    }
}
