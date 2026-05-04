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

pub type S2GeogErrorCode = c_int;

pub const S2GEOGRAPHY_OK: S2GeogErrorCode = 0;

#[repr(C)]
pub struct S2GeogError {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct S2GeogVertex {
    pub v: [f64; 4],
}

#[repr(C)]
pub struct S2Geog {
    _private: [u8; 0],
}

#[repr(C)]
pub struct S2GeogFactory {
    _private: [u8; 0],
}

#[repr(C)]
pub struct S2GeogRectBounder {
    _private: [u8; 0],
}

pub const S2GEOGRAPHY_KERNEL_FORMAT_SEDONA_UDF: c_int = 1;

#[repr(C)]
pub struct S2GeogOp {
    _private: [u8; 0],
}

pub const S2GEOGRAPHY_OP_INTERSECTS: c_int = 1;
pub const S2GEOGRAPHY_OP_CONTAINS: c_int = 2;
pub const S2GEOGRAPHY_OP_WITHIN: c_int = 3;
pub const S2GEOGRAPHY_OP_EQUALS: c_int = 4;
pub const S2GEOGRAPHY_OP_DISTANCE_WITHIN: c_int = 5;
pub const S2GEOGRAPHY_OUTPUT_TYPE_BOOL: c_int = 1;

unsafe extern "C" {
    pub fn S2GeogErrorCreate(err: *mut *mut S2GeogError) -> S2GeogErrorCode;
    pub fn S2GeogErrorGetMessage(err: *const S2GeogError) -> *const c_char;
    pub fn S2GeogErrorDestroy(err: *mut S2GeogError);

    pub fn S2GeogLngLatToCellId(vertex: *const S2GeogVertex) -> u64;

    pub fn S2GeogCreate(geog: *mut *mut S2Geog) -> S2GeogErrorCode;
    pub fn S2GeogForcePrepare(geog: *mut S2Geog, err: *mut S2GeogError) -> S2GeogErrorCode;
    pub fn S2GeogMemUsed(geog: *mut S2Geog) -> usize;
    pub fn S2GeogDestroy(geog: *mut S2Geog);

    pub fn S2GeogFactoryCreate(geog_factory: *mut *mut S2GeogFactory) -> S2GeogErrorCode;
    pub fn S2GeogFactoryInitFromWkbNonOwning(
        geog_factory: *mut S2GeogFactory,
        buf: *const u8,
        buf_size: usize,
        out: *mut S2Geog,
        err: *mut S2GeogError,
    ) -> S2GeogErrorCode;
    pub fn S2GeogFactoryInitFromWkt(
        geog_factory: *mut S2GeogFactory,
        buf: *const c_char,
        buf_size: usize,
        out: *mut S2Geog,
        err: *mut S2GeogError,
    ) -> S2GeogErrorCode;
    pub fn S2GeogFactoryDestroy(geog_factory: *mut S2GeogFactory);

    pub fn S2GeogRectBounderCreate(rect_bounder: *mut *mut S2GeogRectBounder) -> S2GeogErrorCode;
    pub fn S2GeogRectBounderClear(rect_bounder: *mut S2GeogRectBounder);
    pub fn S2GeogRectBounderBound(
        rect_bounder: *mut S2GeogRectBounder,
        geog: *const S2Geog,
        err: *mut S2GeogError,
    ) -> S2GeogErrorCode;
    pub fn S2GeogRectBounderExpandByDistance(
        rect_bounder: *mut S2GeogRectBounder,
        distance_meters: f64,
    );
    pub fn S2GeogRectBounderIsEmpty(rect_bounder: *mut S2GeogRectBounder) -> u8;
    pub fn S2GeogRectBounderFinish(
        rect_bounder: *mut S2GeogRectBounder,
        lo: *mut S2GeogVertex,
        hi: *mut S2GeogVertex,
        err: *mut S2GeogError,
    ) -> S2GeogErrorCode;
    pub fn S2GeogRectBounderDestroy(rect_bounder: *mut S2GeogRectBounder);

    pub fn S2GeogNumKernels() -> usize;
    pub fn S2GeogInitKernels(
        kernels_array: *mut c_void,
        kernels_array_size_bytes: usize,
        format: c_int,
    ) -> S2GeogErrorCode;

    pub fn S2GeogOpCreate(op: *mut *mut S2GeogOp, op_id: c_int) -> S2GeogErrorCode;
    pub fn S2GeogOpName(op: *const S2GeogOp) -> *const c_char;
    pub fn S2GeogOpOutputType(op: *const S2GeogOp) -> c_int;
    pub fn S2GeogOpEvalGeogGeog(
        op: *mut S2GeogOp,
        arg0: *const S2Geog,
        arg1: *const S2Geog,
        err: *mut S2GeogError,
    ) -> S2GeogErrorCode;
    pub fn S2GeogOpEvalGeogGeogDouble(
        op: *mut S2GeogOp,
        arg0: *const S2Geog,
        arg1: *const S2Geog,
        arg2: f64,
        err: *mut S2GeogError,
    ) -> S2GeogErrorCode;
    pub fn S2GeogOpGetInt(op: *mut S2GeogOp) -> i64;
    pub fn S2GeogOpDestroy(op: *mut S2GeogOp);

    pub fn S2GeogNanoarrowVersion() -> *const c_char;
    pub fn S2GeogGeoArrowVersion() -> *const c_char;
    pub fn S2GeogOpenSSLVersion() -> *const c_char;
    pub fn S2GeogS2GeometryVersion() -> *const c_char;
    pub fn S2GeogAbseilVersion() -> *const c_char;
}
