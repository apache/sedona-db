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
use std::ffi::CStr;

mod error;
mod geoarrow_c;
mod geoarrow_c_bindgen;
mod kernels;
pub mod register;

pub fn geoarrow_c_version() -> String {
    let char_ptr = unsafe { geoarrow_c_bindgen::SedonaDBGeoArrowVersion() };
    let c_str = unsafe { CStr::from_ptr(char_ptr).to_str().unwrap() };
    c_str.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions() {
        assert_eq!(geoarrow_c_version(), "0.2.0-SNAPSHOT")
    }
}
