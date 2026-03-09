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

unsafe extern "C" {
    pub fn SedonaGeographyGlueOpenSSLVersion() -> *const c_char;
    pub fn SedonaGeographyGlueS2GeometryVersion() -> *const c_char;
    pub fn SedonaGeographyGlueAbseilVersion() -> *const c_char;
    pub fn SedonaGeographyGlueTestLinkage() -> f64;
    pub fn SedonaGeographyGlueLngLatToCellId(lng: f64, lat: f64) -> u64;
    pub fn SedonaGeographyGlueNumKernels() -> usize;
    pub fn SedonaGeographyGlueInitKernels(
        kernels_array: *mut c_void,
        kernels_size_bytes: usize,
    ) -> c_int;
}
