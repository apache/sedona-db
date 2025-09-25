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
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct pj_ctx {
    _unused: [u8; 0],
}
pub type PJ_CONTEXT = pj_ctx;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PJ_AREA {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PJconsts {
    _unused: [u8; 0],
}
pub type PJ = PJconsts;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PJ_XYZT {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub t: f64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union PJ_COORD {
    pub v: [f64; 4usize],
    pub xyzt: PJ_XYZT,
}

pub const PJ_DIRECTION_PJ_FWD: PJ_DIRECTION = 1;
pub const PJ_DIRECTION_PJ_IDENT: PJ_DIRECTION = 0;
pub const PJ_DIRECTION_PJ_INV: PJ_DIRECTION = -1;
pub type PJ_DIRECTION = ::std::os::raw::c_int;

pub const PJ_LOG_LEVEL_PJ_LOG_NONE: PJ_LOG_LEVEL = 0;
pub const PJ_LOG_LEVEL_PJ_LOG_ERROR: PJ_LOG_LEVEL = 1;
pub const PJ_LOG_LEVEL_PJ_LOG_DEBUG: PJ_LOG_LEVEL = 2;
pub const PJ_LOG_LEVEL_PJ_LOG_TRACE: PJ_LOG_LEVEL = 3;
pub const PJ_LOG_LEVEL_PJ_LOG_TELL: PJ_LOG_LEVEL = 4;
pub type PJ_LOG_LEVEL = ::std::os::raw::c_uint;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct PJ_INFO {
    pub major: ::std::os::raw::c_int,
    pub minor: ::std::os::raw::c_int,
    pub patch: ::std::os::raw::c_int,
    pub release: *const ::std::os::raw::c_char,
    pub version: *const ::std::os::raw::c_char,
    pub searchpath: *const ::std::os::raw::c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ProjApi {
    pub proj_area_create: ::std::option::Option<unsafe extern "C" fn() -> *mut PJ_AREA>,
    pub proj_area_destroy: ::std::option::Option<unsafe extern "C" fn(area: *mut PJ_AREA)>,
    pub proj_area_set_bbox: ::std::option::Option<
        unsafe extern "C" fn(
            area: *mut PJ_AREA,
            west_lon_degree: f64,
            south_lat_degree: f64,
            east_lon_degree: f64,
            north_lat_degree: f64,
        ) -> ::std::os::raw::c_int,
    >,
    pub proj_context_create: ::std::option::Option<unsafe extern "C" fn() -> *mut PJ_CONTEXT>,
    pub proj_context_destroy: ::std::option::Option<unsafe extern "C" fn(ctx: *mut PJ_CONTEXT)>,
    pub proj_context_errno:
        ::std::option::Option<unsafe extern "C" fn(ctx: *mut PJ_CONTEXT) -> ::std::os::raw::c_int>,
    pub proj_context_errno_string: ::std::option::Option<
        unsafe extern "C" fn(
            ctx: *mut PJ_CONTEXT,
            err: ::std::os::raw::c_int,
        ) -> *const ::std::os::raw::c_char,
    >,
    pub proj_context_set_database_path: ::std::option::Option<
        unsafe extern "C" fn(
            ctx: *mut PJ_CONTEXT,
            dbPath: *const ::std::os::raw::c_char,
            auxDbPaths: *const *const ::std::os::raw::c_char,
            options: *const *const ::std::os::raw::c_char,
        ) -> ::std::os::raw::c_int,
    >,
    pub proj_context_set_search_paths: ::std::option::Option<
        unsafe extern "C" fn(
            ctx: *mut PJ_CONTEXT,
            count_paths: ::std::os::raw::c_int,
            paths: *const *const ::std::os::raw::c_char,
        ),
    >,
    pub proj_create: ::std::option::Option<
        unsafe extern "C" fn(
            ctx: *mut PJ_CONTEXT,
            definition: *const ::std::os::raw::c_char,
        ) -> *mut PJ,
    >,
    pub proj_create_crs_to_crs_from_pj: ::std::option::Option<
        unsafe extern "C" fn(
            ctx: *mut PJ_CONTEXT,
            source_crs: *mut PJ,
            target_crs: *mut PJ,
            area: *mut PJ_AREA,
            options: *const *const ::std::os::raw::c_char,
        ) -> *mut PJ,
    >,
    pub proj_cs_get_axis_count: ::std::option::Option<
        unsafe extern "C" fn(ctx: *mut PJ_CONTEXT, cs: *const PJ) -> ::std::os::raw::c_int,
    >,
    pub proj_destroy: ::std::option::Option<unsafe extern "C" fn(P: *mut PJ)>,
    pub proj_errno:
        ::std::option::Option<unsafe extern "C" fn(P: *const PJ) -> ::std::os::raw::c_int>,
    pub proj_errno_reset: ::std::option::Option<unsafe extern "C" fn(P: *mut PJ)>,
    pub proj_info: ::std::option::Option<unsafe extern "C" fn() -> PJ_INFO>,
    pub proj_log_level: ::std::option::Option<
        unsafe extern "C" fn(ctx: *mut PJ_CONTEXT, level: PJ_LOG_LEVEL) -> PJ_LOG_LEVEL,
    >,
    pub proj_normalize_for_visualization: ::std::option::Option<
        unsafe extern "C" fn(ctx: *mut PJ_CONTEXT, obj: *const PJ) -> *mut PJ,
    >,
    pub proj_trans: ::std::option::Option<
        unsafe extern "C" fn(P: *mut PJ, direction: PJ_DIRECTION, coord: PJ_COORD) -> PJ_COORD,
    >,
    pub proj_trans_array: ::std::option::Option<
        unsafe extern "C" fn(
            P: *mut PJ,
            direction: PJ_DIRECTION,
            n: usize,
            coord: *mut PJ_COORD,
        ) -> PJ_COORD,
    >,
    pub release: ::std::option::Option<unsafe extern "C" fn(arg1: *mut ProjApi)>,
    pub private_data: *mut ::std::os::raw::c_void,
}

unsafe extern "C" {
    pub fn proj_dyn_api_init(
        api: *mut ProjApi,
        shared_object_path: *const ::std::os::raw::c_char,
        err_msg: *mut ::std::os::raw::c_char,
        len: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
