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

use std::marker::PhantomData;
use std::ptr;

use crate::s2geog_call;
use crate::s2geog_check;
use crate::s2geography_c_bindgen::*;
use crate::utils::S2GeogCError;

/// Safe wrapper around an unbound S2Geog geography object
///
/// The Geography is a wrapper around a traditional geometry optimized for
/// a spherical interpretation of edges using the S2 geometry library. The
/// geography caches some internal scratch space and may be reused when
/// looping over many geographies derived from the same lifetime.
///
/// Some Geography objects are tied to the lifetime of their input (e.g.,
/// when reading WKB); however, others own their coordinates (e.g., when
/// reading WKT).
///
/// Geographies are thread safe. The internal index is built on the fly when
/// required in a thread safe way (although it may result in faster evaluation
/// to force a build early in a situation where it is known that a lot of
/// evaluations are about to happen).
#[derive(Debug)]
pub struct Geography<'a> {
    ptr: *mut S2Geog,
    _marker: PhantomData<&'a [u8]>,
}

impl<'a> Geography<'a> {
    /// Create a new empty geography
    pub fn new() -> Self {
        let mut ptr: *mut S2Geog = ptr::null_mut();
        unsafe { s2geog_check!(S2GeogCreate(&mut ptr)) }.unwrap();
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Force build an index on this geography
    ///
    /// This operation forces an index build, which may be expensive but can result in
    /// faster evaluation when passed to some operations (e.g., predicates).
    pub fn prepare(&mut self) -> Result<(), S2GeogCError> {
        unsafe { s2geog_call!(S2GeogForcePrepare(self.ptr)) }
    }

    /// Estimate the memory used by this geography
    pub fn mem_used(&self) -> usize {
        unsafe { S2GeogMemUsed(self.ptr) }
    }

    pub(crate) fn as_ptr(&self) -> *const S2Geog {
        self.ptr
    }
}

impl<'a> Default for Geography<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Drop for Geography<'a> {
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
unsafe impl<'a> Send for Geography<'a> {}
unsafe impl<'a> Sync for Geography<'a> {}

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

    /// Create a bound geography from WKB bytes
    ///
    /// The returned geography is bound to the lifetime of the WKB buffer,
    /// ensuring the buffer is not dropped while the geography is in use.
    /// This is necessary because the underlying C++ implementation
    /// references the original WKB data.
    pub fn from_wkb<'a>(&mut self, wkb: &'a [u8]) -> Result<Geography<'a>, S2GeogCError> {
        let mut geog = Geography::new();
        self.init_from_wkb(wkb, &mut geog)?;
        Ok(geog)
    }

    /// Create a geography from WKT string
    pub fn from_wkt(&mut self, wkt: &str) -> Result<Geography<'static>, S2GeogCError> {
        let mut geog = Geography::new();
        self.init_from_wkt(wkt, &mut geog)?;
        Ok(geog)
    }

    /// Initialize an existing Geography with a new WKB buffer
    ///
    /// This is the function that should be used in the event of looping over
    /// WKBs from the same arrow array.
    pub fn init_from_wkb(&mut self, wkb: &[u8], geog: &mut Geography) -> Result<(), S2GeogCError> {
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
        assert!(geog.mem_used() >= 150);
        geog.prepare().unwrap();
    }

    #[test]
    fn test_geography_from_wkb() {
        let wkb_bytes = sedona_testing::create::make_wkb("POINT (0 1)");
        let mut factory = GeographyFactory::new();
        let mut geog = factory.from_wkb(&wkb_bytes).unwrap();
        assert!(geog.mem_used() >= 150);
        geog.prepare().unwrap();
    }
}
