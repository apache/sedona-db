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
use std::{ffi::CStr, fmt::Display, mem::transmute, ptr};

use arrow_array::{ffi::FFI_ArrowArray, make_array, ArrayRef};
use arrow_schema::DataType;
use sedona_schema::datatypes::SedonaType;

use crate::{error::GeoArrowCError, geoarrow_c_bindgen::*};

/// Rust wrapper around the [GeoArrowArrayReader]
///
/// The reader is a generalized reader that can visit all GeoArrow arrays
/// using an appropriate [Visitor] (e.g., the one provided by the
/// [ArrayWriter]).
pub struct ArrayReader {
    inner: GeoArrowArrayReader,
}

impl Drop for ArrayReader {
    fn drop(&mut self) {
        if !self.ptr_mut().is_null() {
            unsafe { SedonaDBGeoArrowArrayReaderReset(self.ptr_mut()) };
        }
    }
}

impl ArrayReader {
    /// Create an [ArrayReader] that reads a [SedonaType], if possible
    pub fn try_new(sedona_type: &SedonaType) -> Result<Self, GeoArrowCError> {
        let type_id = geoarrow_type_id(sedona_type)?;

        let inner = GeoArrowArrayReader {
            private_data: ptr::null_mut(),
        };

        let mut out = Self { inner };
        unsafe { SedonaDBGeoArrowArrayReaderInitFromType(out.ptr_mut(), type_id) };
        Ok(out)
    }

    /// Visit an [ArrayRef of the appropriate type]
    ///
    /// This can be called more than once for any array of the appropriate type.
    pub fn visit(
        &mut self,
        array: &ArrayRef,
        visitor: &mut Visitor,
        error: &mut CError,
    ) -> Result<(), GeoArrowCError> {
        let (ffi_array, _) = arrow_array::ffi::to_ffi(&array.to_data())?;
        let array_len = array.len().try_into()?;

        unsafe {
            error.reset();
            let ffi_array_ptr: *const ArrowArray = transmute(&ffi_array as *const FFI_ArrowArray);
            let code =
                SedonaDBGeoArrowArrayReaderSetArray(self.ptr_mut(), ffi_array_ptr, error.ptr_mut());
            error.msg_not_ok(code)?;

            visitor.set_error(error);
            let code =
                SedonaDBGeoArrowArrayReaderVisit(self.ptr_mut(), 0, array_len, visitor.ptr_mut());
            visitor.reset_error();

            error.msg_not_ok(code)?;
        };

        Ok(())
    }

    fn ptr_mut(&mut self) -> *mut GeoArrowArrayReader {
        &mut self.inner
    }
}

/// Rust wrapper around the [GeoArrowArrayWriter]
///
/// This writer can create arrays of most types except large/view variants of
/// the WKT and WKB types.
pub struct ArrayWriter {
    inner: GeoArrowArrayWriter,
    visitor: Visitor,
    data_type: DataType,
}

impl Drop for ArrayWriter {
    fn drop(&mut self) {
        if !self.ptr_mut().is_null() {
            unsafe { SedonaDBGeoArrowArrayWriterReset(self.ptr_mut()) };
        }
    }
}

impl ArrayWriter {
    /// Create an [ArrayWriter] that writes a [SedonaType], if possible
    pub fn try_new(sedona_type: &SedonaType) -> Result<Self, GeoArrowCError> {
        let type_id = geoarrow_type_id(sedona_type)?;

        let inner = GeoArrowArrayWriter {
            private_data: ptr::null_mut(),
        };

        let mut out = Self {
            inner,
            visitor: Visitor::new(),
            data_type: arrow_storage_type(type_id)?,
        };
        unsafe { SedonaDBGeoArrowArrayWriterInitFromType(out.ptr_mut(), type_id) };
        Ok(out)
    }

    fn ptr_mut(&mut self) -> *mut GeoArrowArrayWriter {
        &mut self.inner
    }

    /// Create a reference to the internal visitor pointing to this writer
    pub fn visitor_mut(&mut self) -> &mut Visitor {
        unsafe { SedonaDBGeoArrowArrayWriterInitVisitor(self.ptr_mut(), self.visitor.ptr_mut()) };
        &mut self.visitor
    }

    /// Finish the current state of this visitor
    ///
    /// This can be called more than once between visits.
    pub fn finish(&mut self) -> Result<ArrayRef, GeoArrowCError> {
        let mut out = arrow_array::ffi::FFI_ArrowArray::empty();
        unsafe {
            let array_ptr: *mut ArrowArray =
                transmute(&mut out as *mut arrow_array::ffi::FFI_ArrowArray);
            let code =
                SedonaDBGeoArrowArrayWriterFinish(self.ptr_mut(), array_ptr, ptr::null_mut());
            if code != 0 {
                return Err(GeoArrowCError::Code(code));
            }
        }

        let array_data =
            unsafe { arrow_array::ffi::from_ffi_and_data_type(out, self.data_type.clone())? };

        Ok(make_array(array_data))
    }
}

/// Wrapper around the [GeoArrowError]
///
/// Most GeoArrow operations accept a pointer to the [GeoArrowError] to communicate
/// a string error if necessary. This is a fixed 1024 bytes and can be a performance
/// issue if allocated in a loop.
pub struct CError {
    inner: GeoArrowError,
}

impl CError {
    /// Create a new error
    pub fn new() -> Self {
        Self {
            inner: GeoArrowError { message: [0; 1024] },
        }
    }

    fn reset(&mut self) {
        self.inner.message[0] = 0;
    }

    fn ptr_mut(&mut self) -> *mut GeoArrowError {
        &mut self.inner
    }

    /// Check code and return a Rust error if non-OK that includes the message in this object
    pub fn msg_not_ok(&mut self, code: i32) -> Result<(), GeoArrowCError> {
        if code != 0 {
            Err(GeoArrowCError::Message(code, self.to_string()))
        } else {
            Ok(())
        }
    }
}

impl Display for CError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            write!(
                f,
                "{}",
                CStr::from_ptr(self.inner.message.as_ptr())
                    .to_str()
                    .unwrap()
            )
        }
    }
}

impl Default for CError {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper around the [GeoArrowVisitor]
///
/// The Visitor is an intermediary between array readers and writers/collectors
/// to enable more generic code to be written. This is typically managed by the
/// writer but can be useful for testing.
pub struct Visitor {
    inner: GeoArrowVisitor,
}

impl Visitor {
    /// Create a new visitor wrapper
    pub fn new() -> Self {
        Self {
            inner: GeoArrowVisitor {
                feat_start: None,
                null_feat: None,
                geom_start: None,
                ring_start: None,
                coords: None,
                ring_end: None,
                geom_end: None,
                feat_end: None,
                private_data: ptr::null_mut(),
                error: ptr::null_mut(),
            },
        }
    }

    /// Create a new visitor wrapper that wraps a visitor that does nothing
    pub fn void() -> Self {
        let mut visitor = Self::new();
        unsafe { SedonaDBGeoArrowVisitorInitVoid(visitor.ptr_mut()) };
        visitor
    }

    fn ptr_mut(&mut self) -> *mut GeoArrowVisitor {
        &mut self.inner
    }

    fn set_error(&mut self, error: &mut CError) {
        self.inner.error = error.ptr_mut();
    }

    fn reset_error(&mut self) {
        self.inner.error = ptr::null_mut();
    }
}

impl Default for Visitor {
    fn default() -> Self {
        Self::new()
    }
}

impl From<&str> for GeoArrowStringView {
    fn from(value: &str) -> Self {
        Self {
            data: value.as_ptr() as *const _,
            size_bytes: value.len() as i64,
        }
    }
}

/// Convert a [SedonaType] to a GeoArrow type identifier
fn geoarrow_type_id(sedona_type: &SedonaType) -> Result<GeoArrowType, GeoArrowCError> {
    let type_id = match sedona_type {
        SedonaType::Wkb(_, _) => GeoArrowType_GEOARROW_TYPE_WKB,
        SedonaType::WkbView(_, _) => GeoArrowType_GEOARROW_TYPE_WKB_VIEW,
        SedonaType::Arrow(data_type) => match data_type {
            DataType::Binary => GeoArrowType_GEOARROW_TYPE_WKB,
            DataType::BinaryView => GeoArrowType_GEOARROW_TYPE_WKB_VIEW,
            DataType::Utf8 => GeoArrowType_GEOARROW_TYPE_WKT,
            DataType::Utf8View => GeoArrowType_GEOARROW_TYPE_WKT_VIEW,
            _ => {
                return Err(GeoArrowCError::Invalid(format!(
                    "Can't guess GeoArrow type from {sedona_type:?}"
                )));
            }
        },
    };

    Ok(type_id)
}

/// Convert to add a GeoArrow type identifier to a [SedonaType]
#[allow(non_upper_case_globals)]
fn arrow_storage_type(type_id: GeoArrowType) -> Result<DataType, GeoArrowCError> {
    Ok(match type_id {
        GeoArrowType_GEOARROW_TYPE_WKB => DataType::Binary,
        GeoArrowType_GEOARROW_TYPE_WKB_VIEW => DataType::BinaryView,
        GeoArrowType_GEOARROW_TYPE_WKT => DataType::Utf8,
        GeoArrowType_GEOARROW_TYPE_WKT_VIEW => DataType::Utf8View,
        _ => {
            return Err(GeoArrowCError::Invalid(format!(
                "Can't guess Arrow type from GeoArrowType with ID {type_id:?}"
            )))
        }
    })
}

#[cfg(test)]
mod test {
    use datafusion_common::cast::as_string_array;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::create::create_array_storage;

    use super::*;

    #[test]
    fn array_reader() {
        let mut error = CError::new();
        let mut reader = ArrayReader::try_new(&WKB_GEOMETRY).unwrap();
        let mut visitor = Visitor::void();
        let wkb_array = create_array_storage(&[Some("POINT (0 1)"), None], &WKB_GEOMETRY);
        reader.visit(&wkb_array, &mut visitor, &mut error).unwrap();
    }

    #[test]
    fn visit_error() {
        let mut error = CError::new();
        let mut reader = ArrayReader::try_new(&SedonaType::Arrow(DataType::Utf8)).unwrap();
        let mut visitor = Visitor::void();
        let wkt_array: ArrayRef = arrow_array::create_array!(Utf8, [Some("NOT WKT")]);
        let err = reader
            .visit(&wkt_array, &mut visitor, &mut error)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid argument: Expected geometry type at byte 0"
        );
    }

    #[test]
    fn array_writer() {
        let mut error = CError::new();
        let mut reader = ArrayReader::try_new(&WKB_GEOMETRY).unwrap();
        let mut writer = ArrayWriter::try_new(&SedonaType::Arrow(DataType::Utf8)).unwrap();
        let wkb_array = create_array_storage(&[Some("POINT (0 1)"), None], &WKB_GEOMETRY);
        reader
            .visit(&wkb_array, writer.visitor_mut(), &mut error)
            .unwrap();

        let wkt_array = writer.finish().unwrap();
        let values_out = as_string_array(&wkt_array)
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(values_out, vec![Some("POINT (0 1)"), None]);
    }

    #[test]
    fn type_to_geoarrow() {
        assert_eq!(
            geoarrow_type_id(&WKB_GEOMETRY).unwrap(),
            GeoArrowType_GEOARROW_TYPE_WKB
        );
        assert_eq!(
            geoarrow_type_id(&WKB_VIEW_GEOMETRY).unwrap(),
            GeoArrowType_GEOARROW_TYPE_WKB_VIEW
        );
        assert_eq!(
            geoarrow_type_id(&SedonaType::Arrow(DataType::Binary)).unwrap(),
            GeoArrowType_GEOARROW_TYPE_WKB
        );
        assert_eq!(
            geoarrow_type_id(&SedonaType::Arrow(DataType::BinaryView)).unwrap(),
            GeoArrowType_GEOARROW_TYPE_WKB_VIEW
        );
        assert_eq!(
            geoarrow_type_id(&SedonaType::Arrow(DataType::Utf8)).unwrap(),
            GeoArrowType_GEOARROW_TYPE_WKT
        );
        assert_eq!(
            geoarrow_type_id(&SedonaType::Arrow(DataType::Utf8View)).unwrap(),
            GeoArrowType_GEOARROW_TYPE_WKT_VIEW
        );

        let err = geoarrow_type_id(&SedonaType::Arrow(DataType::Null)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Can't guess GeoArrow type from Arrow(Null)"
        );
    }

    #[test]
    fn type_from_geoarrow() {
        assert_eq!(
            arrow_storage_type(GeoArrowType_GEOARROW_TYPE_WKB).unwrap(),
            DataType::Binary
        );
        assert_eq!(
            arrow_storage_type(GeoArrowType_GEOARROW_TYPE_WKB_VIEW).unwrap(),
            DataType::BinaryView
        );
        assert_eq!(
            arrow_storage_type(GeoArrowType_GEOARROW_TYPE_WKT).unwrap(),
            DataType::Utf8
        );
        assert_eq!(
            arrow_storage_type(GeoArrowType_GEOARROW_TYPE_WKT_VIEW).unwrap(),
            DataType::Utf8View
        );

        let err = arrow_storage_type(GeoArrowType_GEOARROW_TYPE_LINESTRING).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Can't guess Arrow type from GeoArrowType with ID 2"
        );
    }
}
