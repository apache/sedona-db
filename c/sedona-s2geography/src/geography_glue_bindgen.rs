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

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct ArrowSchema {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ArrowArray {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SedonaGeographyArrowUdf {
    pub init: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaGeographyArrowUdf,
            arg_schema: *mut ArrowSchema,
            options: *const c_char,
            out: *mut ArrowSchema,
        ) -> c_int,
    >,
    pub execute: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaGeographyArrowUdf,
            args: *mut *mut ArrowArray,
            n_args: i64,
            out: *mut ArrowArray,
        ) -> c_int,
    >,
    pub get_last_error:
        Option<unsafe extern "C" fn(self_: *mut SedonaGeographyArrowUdf) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaGeographyArrowUdf)>,
    pub private_data: *mut c_void,
}

macro_rules! declare_s2_c_udfs {
    ($($name:ident),*) => {
        $(
            paste::item! {
                pub fn [<SedonaGeographyInitUdf $name>](out: *mut SedonaGeographyArrowUdf);
            }
        )*
    }
}

unsafe extern "C" {
    pub fn SedonaGeographyGlueNanoarrowVersion() -> *const c_char;
    pub fn SedonaGeographyGlueGeoArrowVersion() -> *const c_char;
    pub fn SedonaGeographyGlueOpenSSLVersion() -> *const c_char;
    pub fn SedonaGeographyGlueS2GeometryVersion() -> *const c_char;
    pub fn SedonaGeographyGlueAbseilVersion() -> *const c_char;
    pub fn SedonaGeographyGlueTestLinkage() -> f64;

    declare_s2_c_udfs!(
        Area,
        Centroid,
        ClosestPoint,
        Contains,
        ConvexHull,
        Difference,
        Distance,
        Equals,
        Intersection,
        Intersects,
        Length,
        LineInterpolatePoint,
        LineLocatePoint,
        MaxDistance,
        Perimeter,
        ShortestLine,
        SymDifference,
        Union
    );
}
