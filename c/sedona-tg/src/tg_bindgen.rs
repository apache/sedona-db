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

use std::os::raw::{c_char, c_void};

#[cfg(target_env = "msvc")]
use std::os::raw::c_int;

#[cfg(target_env = "msvc")]
pub type tg_index = c_int;

#[cfg(not(target_env = "msvc"))]
use std::os::raw::c_uint;

#[cfg(not(target_env = "msvc"))]
pub type tg_index = c_uint;

pub const tg_index_TG_NONE: tg_index = 1;
pub const tg_index_TG_DEFAULT: tg_index = 0;
pub const tg_index_TG_NATURAL: tg_index = 2;
pub const tg_index_TG_YSTRIPES: tg_index = 3;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tg_geom {
    _unused: [u8; 0],
}

unsafe extern "C" {
    pub fn tg_geom_free(geom: *mut tg_geom);
    pub fn tg_parse_wktn_ix(wkt: *const c_char, len: usize, ix: tg_index) -> *mut tg_geom;
    pub fn tg_parse_wkb_ix(wkb: *const u8, len: usize, ix: tg_index) -> *mut tg_geom;
    pub fn tg_geom_error(geom: *const tg_geom) -> *const c_char;
    pub fn tg_geom_wkb(geom: *const tg_geom, dst: *mut u8, n: usize) -> usize;
    pub fn tg_geom_wkt(geom: *const tg_geom, dst: *mut c_char, n: usize) -> usize;
    pub fn tg_geom_memsize(geom: *const tg_geom) -> usize;
    pub fn tg_geom_equals(a: *const tg_geom, b: *const tg_geom) -> bool;
    pub fn tg_geom_intersects(a: *const tg_geom, b: *const tg_geom) -> bool;
    pub fn tg_geom_disjoint(a: *const tg_geom, b: *const tg_geom) -> bool;
    pub fn tg_geom_contains(a: *const tg_geom, b: *const tg_geom) -> bool;
    pub fn tg_geom_within(a: *const tg_geom, b: *const tg_geom) -> bool;
    pub fn tg_geom_covers(a: *const tg_geom, b: *const tg_geom) -> bool;
    pub fn tg_geom_coveredby(a: *const tg_geom, b: *const tg_geom) -> bool;
    pub fn tg_geom_touches(a: *const tg_geom, b: *const tg_geom) -> bool;
    pub fn tg_env_set_allocator(
        malloc: Option<unsafe extern "C" fn(arg1: usize) -> *mut c_void>,
        realloc: Option<unsafe extern "C" fn(arg1: *mut c_void, arg2: usize) -> *mut c_void>,
        free: Option<unsafe extern "C" fn(arg1: *mut c_void)>,
    );
}
