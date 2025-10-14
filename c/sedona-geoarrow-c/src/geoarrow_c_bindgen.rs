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
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
pub struct ArrowArray {
    _private: [u8; 0],
}

#[repr(C)]
pub struct GeoArrowCoordView {
    _private: [u8; 0],
}

#[cfg(target_env = "msvc")]
pub type enum_t = std::os::raw::c_int;

#[cfg(not(target_env = "msvc"))]
pub type enum_t = std::os::raw::c_uint;

pub type GeoArrowGeometryType = enum_t;
pub type GeoArrowDimensions = enum_t;
pub type GeoArrowType = enum_t;

pub const GeoArrowType_GEOARROW_TYPE_WKB: GeoArrowType = 100001;
pub const GeoArrowType_GEOARROW_TYPE_WKT: GeoArrowType = 100003;
pub const GeoArrowType_GEOARROW_TYPE_WKB_VIEW: GeoArrowType = 100005;
pub const GeoArrowType_GEOARROW_TYPE_WKT_VIEW: GeoArrowType = 100006;

pub type GeoArrowErrorCode = c_int;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowError {
    pub message: [c_char; 1024usize],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowVisitor {
    pub feat_start: Option<unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> c_int>,
    pub null_feat: Option<unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> c_int>,
    pub geom_start: Option<
        unsafe extern "C" fn(
            v: *mut GeoArrowVisitor,
            geometry_type: GeoArrowGeometryType,
            dimensions: GeoArrowDimensions,
        ) -> c_int,
    >,
    pub ring_start: Option<unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> c_int>,
    pub coords: Option<
        unsafe extern "C" fn(v: *mut GeoArrowVisitor, coords: *const GeoArrowCoordView) -> c_int,
    >,
    pub ring_end: Option<unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> c_int>,
    pub geom_end: Option<unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> c_int>,
    pub feat_end: Option<unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> c_int>,
    pub private_data: *mut c_void,
    pub error: *mut GeoArrowError,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowArrayReader {
    pub private_data: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowArrayWriter {
    pub private_data: *mut c_void,
}

unsafe extern "C" {
    pub fn SedonaDBGeoArrowVersion() -> *const c_char;

    pub fn SedonaDBGeoArrowVisitorInitVoid(v: *mut GeoArrowVisitor);

    pub fn SedonaDBGeoArrowArrayReaderInitFromType(
        reader: *mut GeoArrowArrayReader,
        type_: GeoArrowType,
    ) -> GeoArrowErrorCode;

    pub fn SedonaDBGeoArrowArrayReaderSetArray(
        reader: *mut GeoArrowArrayReader,
        array: *const ArrowArray,
        error: *mut GeoArrowError,
    ) -> GeoArrowErrorCode;

    pub fn SedonaDBGeoArrowArrayReaderVisit(
        reader: *mut GeoArrowArrayReader,
        offset: i64,
        length: i64,
        v: *mut GeoArrowVisitor,
    ) -> GeoArrowErrorCode;

    pub fn SedonaDBGeoArrowArrayReaderReset(reader: *mut GeoArrowArrayReader);

    pub fn SedonaDBGeoArrowArrayWriterInitFromType(
        writer: *mut GeoArrowArrayWriter,
        type_: GeoArrowType,
    ) -> GeoArrowErrorCode;

    pub fn SedonaDBGeoArrowArrayWriterInitVisitor(
        writer: *mut GeoArrowArrayWriter,
        v: *mut GeoArrowVisitor,
    ) -> GeoArrowErrorCode;

    pub fn SedonaDBGeoArrowArrayWriterFinish(
        writer: *mut GeoArrowArrayWriter,
        array: *mut ArrowArray,
        error: *mut GeoArrowError,
    ) -> GeoArrowErrorCode;

    pub fn SedonaDBGeoArrowArrayWriterReset(writer: *mut GeoArrowArrayWriter);
}
