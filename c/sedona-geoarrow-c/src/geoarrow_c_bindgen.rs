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
#![allow(dead_code)]

#[repr(C)]
pub struct ArrowSchema {
    // Contents deliberately omitted
    _private: [u8; 0],
}

#[repr(C)]
pub struct ArrowArray {
    // Contents deliberately omitted
    _private: [u8; 0],
}

#[repr(C)]
pub struct GeoArrowCoordView {
    // Contents deliberately omitted
    _private: [u8; 0],
}

pub type GeoArrowGeometryType = ::std::os::raw::c_uint;

pub type GeoArrowDimensions = ::std::os::raw::c_uint;

pub type GeoArrowErrorCode = ::std::os::raw::c_int;

pub type GeoArrowType = ::std::os::raw::c_uint;

pub const GeoArrowType_GEOARROW_TYPE_UNINITIALIZED: GeoArrowType = 0;
pub const GeoArrowType_GEOARROW_TYPE_WKB: GeoArrowType = 100001;
pub const GeoArrowType_GEOARROW_TYPE_LARGE_WKB: GeoArrowType = 100002;
pub const GeoArrowType_GEOARROW_TYPE_WKT: GeoArrowType = 100003;
pub const GeoArrowType_GEOARROW_TYPE_LARGE_WKT: GeoArrowType = 100004;
pub const GeoArrowType_GEOARROW_TYPE_WKB_VIEW: GeoArrowType = 100005;
pub const GeoArrowType_GEOARROW_TYPE_WKT_VIEW: GeoArrowType = 100006;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowError {
    pub message: [::std::os::raw::c_char; 1024usize],
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of GeoArrowError"][::std::mem::size_of::<GeoArrowError>() - 1024usize];
    ["Alignment of GeoArrowError"][::std::mem::align_of::<GeoArrowError>() - 1usize];
    ["Offset of field: GeoArrowError::message"]
        [::std::mem::offset_of!(GeoArrowError, message) - 0usize];
};
#[doc = " \\brief A read-only view of a string\n \\ingroup geoarrow-utility"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowStringView {
    #[doc = " \\brief Pointer to the beginning of the string. May be NULL if size_bytes is 0.\n there is no requirement that the string is null-terminated."]
    pub data: *const ::std::os::raw::c_char,
    #[doc = " \\brief The size of the string in bytes"]
    pub size_bytes: i64,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of GeoArrowStringView"][::std::mem::size_of::<GeoArrowStringView>() - 16usize];
    ["Alignment of GeoArrowStringView"][::std::mem::align_of::<GeoArrowStringView>() - 8usize];
    ["Offset of field: GeoArrowStringView::data"]
        [::std::mem::offset_of!(GeoArrowStringView, data) - 0usize];
    ["Offset of field: GeoArrowStringView::size_bytes"]
        [::std::mem::offset_of!(GeoArrowStringView, size_bytes) - 8usize];
};
#[doc = " \\brief A read-only view of a buffer\n \\ingroup geoarrow-utility"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowBufferView {
    #[doc = " \\brief Pointer to the beginning of the string. May be NULL if size_bytes is 0."]
    pub data: *const u8,
    #[doc = " \\brief The size of the buffer in bytes"]
    pub size_bytes: i64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowVisitor {
    #[doc = " \\brief Called when starting to iterate over a new feature"]
    pub feat_start: ::std::option::Option<
        unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> ::std::os::raw::c_int,
    >,
    #[doc = " \\brief Called after feat_start for a null_feature"]
    pub null_feat: ::std::option::Option<
        unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> ::std::os::raw::c_int,
    >,
    #[doc = " \\brief Called after feat_start for a new geometry\n\n Every non-null feature will have at least one call to geom_start.\n Collections (including multi-geometry types) will have nested calls to geom_start."]
    pub geom_start: ::std::option::Option<
        unsafe extern "C" fn(
            v: *mut GeoArrowVisitor,
            geometry_type: GeoArrowGeometryType,
            dimensions: GeoArrowDimensions,
        ) -> ::std::os::raw::c_int,
    >,
    #[doc = " \\brief For polygon geometries, called after geom_start at the beginning of a ring"]
    pub ring_start: ::std::option::Option<
        unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> ::std::os::raw::c_int,
    >,
    #[doc = " \\brief Called when a sequence of coordinates is encountered\n\n This callback may be called more than once (i.e., readers are free to chunk\n coordinates however they see fit). The GeoArrowCoordView may represent\n either interleaved of struct coordinates depending on the reader implementation."]
    pub coords: ::std::option::Option<
        unsafe extern "C" fn(
            v: *mut GeoArrowVisitor,
            coords: *const GeoArrowCoordView,
        ) -> ::std::os::raw::c_int,
    >,
    #[doc = " \\brief For polygon geometries, called at the end of a ring\n\n Every call to ring_start must have a matching call to ring_end"]
    pub ring_end: ::std::option::Option<
        unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> ::std::os::raw::c_int,
    >,
    #[doc = " \\brief Called at the end of a geometry\n\n Every call to geom_start must have a matching call to geom_end."]
    pub geom_end: ::std::option::Option<
        unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> ::std::os::raw::c_int,
    >,
    #[doc = " \\brief Called at the end of a feature, including null features\n\n Every call to feat_start must have a matching call to feat_end."]
    pub feat_end: ::std::option::Option<
        unsafe extern "C" fn(v: *mut GeoArrowVisitor) -> ::std::os::raw::c_int,
    >,
    #[doc = " \\brief Opaque visitor-specific data"]
    pub private_data: *mut ::std::os::raw::c_void,
    #[doc = " \\brief The error into which the reader and/or visitor can place a detailed\n message.\n\n When a visitor is initializing callbacks and private_data it should take care\n to not change the value of error. This value can be NULL."]
    pub error: *mut GeoArrowError,
}

unsafe extern "C" {
    #[doc = " \\brief Initialize a GeoArrowVisitor with a visitor that does nothing"]
    pub fn SedonaDBGeoArrowVisitorInitVoid(v: *mut GeoArrowVisitor);
}

unsafe extern "C" {
    #[doc = " \\brief Return a version string in the form \"major.minor.patch\""]
    pub fn SedonaDBGeoArrowVersion() -> *const ::std::os::raw::c_char;
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowArrayReader {
    pub private_data: *mut ::std::os::raw::c_void,
}

unsafe extern "C" {
    #[doc = " \\brief Initialize a GeoArrowArrayReader from a GeoArrowType\n\n If GEOARROW_OK is returned, the caller is responsible for calling\n GeoArrowArrayReaderReset()."]
    pub fn SedonaDBGeoArrowArrayReaderInitFromType(
        reader: *mut GeoArrowArrayReader,
        type_: GeoArrowType,
    ) -> GeoArrowErrorCode;
}

unsafe extern "C" {
    #[doc = " \\brief Initialize a GeoArrowArrayReader from an ArrowSchema\n\n If GEOARROW_OK is returned, the caller is responsible for calling\n GeoArrowArrayReaderReset()."]
    pub fn SedonaDBGeoArrowArrayReaderInitFromSchema(
        reader: *mut GeoArrowArrayReader,
        schema: *const ArrowSchema,
        error: *mut GeoArrowError,
    ) -> GeoArrowErrorCode;
}
unsafe extern "C" {
    #[doc = " \\brief Set a GeoArrowArray to read"]
    pub fn SedonaDBGeoArrowArrayReaderSetArray(
        reader: *mut GeoArrowArrayReader,
        array: *const ArrowArray,
        error: *mut GeoArrowError,
    ) -> GeoArrowErrorCode;
}
unsafe extern "C" {
    #[doc = " \\brief Visit a GeoArrowArray\n\n The caller must have initialized the GeoArrowVisitor with the appropriate\n writer before calling this function."]
    pub fn SedonaDBGeoArrowArrayReaderVisit(
        reader: *mut GeoArrowArrayReader,
        offset: i64,
        length: i64,
        v: *mut GeoArrowVisitor,
    ) -> GeoArrowErrorCode;
}

unsafe extern "C" {
    #[doc = " \\brief Free resources held by a GeoArrowArrayReader"]
    pub fn SedonaDBGeoArrowArrayReaderReset(reader: *mut GeoArrowArrayReader);
}
#[doc = " \\brief Generc GeoArrow array writer"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GeoArrowArrayWriter {
    pub private_data: *mut ::std::os::raw::c_void,
}

unsafe extern "C" {
    #[doc = " \\brief Initialize the memory of a GeoArrowArrayWriter from a GeoArrowType\n\n If GEOARROW_OK is returned, the caller is responsible for calling\n GeoArrowWKTWriterReset()."]
    pub fn SedonaDBGeoArrowArrayWriterInitFromType(
        writer: *mut GeoArrowArrayWriter,
        type_: GeoArrowType,
    ) -> GeoArrowErrorCode;
}

unsafe extern "C" {
    #[doc = " \\brief Initialize the memory of a GeoArrowArrayWriter from an ArrowSchema\n\n If GEOARROW_OK is returned, the caller is responsible for calling\n GeoArrowWKTWriterReset()."]
    pub fn SedonaDBGeoArrowArrayWriterInitFromSchema(
        writer: *mut GeoArrowArrayWriter,
        schema: *const ArrowSchema,
    ) -> GeoArrowErrorCode;
}
unsafe extern "C" {
    #[doc = " \\brief Set the precision to use for array writers writing to WKT\n\n Returns EINVAL for precision values that are not valid or if the writer\n is not writing to WKT. Must be called before GeoArrowArrayWriterInitVisitor().\n The default precision value is 16. See GeoArrowWKTWriter for details."]
    pub fn SedonaDBGeoArrowArrayWriterSetPrecision(
        writer: *mut GeoArrowArrayWriter,
        precision: ::std::os::raw::c_int,
    ) -> GeoArrowErrorCode;
}
unsafe extern "C" {
    #[doc = " \\brief Set the MULTIPOINT output mode when writing to WKT\n\n Returns EINVAL if the writer is not writing to WKT. Must be called before\n GeoArrowArrayWriterInitVisitor(). The default value is 1. See GeoArrowWKTWriter for\n details."]
    pub fn SedonaDBGeoArrowArrayWriterSetFlatMultipoint(
        writer: *mut GeoArrowArrayWriter,
        flat_multipoint: ::std::os::raw::c_int,
    ) -> GeoArrowErrorCode;
}
unsafe extern "C" {
    #[doc = " \\brief Populate a GeoArrowVisitor pointing to this writer"]
    pub fn SedonaDBGeoArrowArrayWriterInitVisitor(
        writer: *mut GeoArrowArrayWriter,
        v: *mut GeoArrowVisitor,
    ) -> GeoArrowErrorCode;
}
unsafe extern "C" {
    #[doc = " \\brief Finish an ArrowArray containing elements from the visited input\n\n This function can be called more than once to support multiple batches."]
    pub fn SedonaDBGeoArrowArrayWriterFinish(
        writer: *mut GeoArrowArrayWriter,
        array: *mut ArrowArray,
        error: *mut GeoArrowError,
    ) -> GeoArrowErrorCode;
}
unsafe extern "C" {
    #[doc = " \\brief Free resources held by a GeoArrowArrayWriter"]
    pub fn SedonaDBGeoArrowArrayWriterReset(writer: *mut GeoArrowArrayWriter);
}
