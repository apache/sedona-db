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
use std::collections::HashMap;

use arrow_array::{
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    ArrayRef,
};
use arrow_schema::{ArrowError, Fields, Schema};

use crate::{error::S2GeographyError, geography_glue_bindgen::*};

/// Wrapper for scalar UDFs exposed by s2geography::arrow_udf
///
/// Provides a minimal wrapper around the C callables that define
/// an scalar UDF as exposed by the s2geography library.
///
/// These are designed to be sufficiently cheap to initialize that
/// they can be constructed on the stack, initialized, and executed
/// for a single batch.
#[derive(Debug)]
pub struct S2ScalarUDF {
    inner: SedonaGeographyArrowUdf,
}

impl Drop for S2ScalarUDF {
    fn drop(&mut self) {
        if let Some(releaser) = self.inner.release {
            unsafe { releaser(&mut self.inner) }
        }
    }
}

// We have a bunch of these, so we use a macro to declare them. We could
// use a string or enum to retrieve them as well if this becomes unwieldy.
macro_rules! define_s2_udfs {
    ($($name:ident),*) => {
        $(
            pub fn $name() -> S2ScalarUDF {
                let mut out = Self::allocate();
                unsafe { paste::paste!([<SedonaGeographyInitUdf $name>])(&mut out.inner) }
                out
            }
        )*
    }
}

#[allow(non_snake_case)]
impl S2ScalarUDF {
    define_s2_udfs![
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
    ];

    /// Initialize the UDF instance with argument types and options
    ///
    /// This must be called before calling execute().
    pub fn init(
        &mut self,
        arg_types: Fields,
        options: Option<HashMap<String, String>>,
    ) -> Result<FFI_ArrowSchema, S2GeographyError> {
        if options.is_some() {
            // See FFI_ArrowSchema::with_metadata() for implementation. This
            // is to pass options like the radius to use for length/perimeter/area
            // calculations that are currently hard-coded to the WGS84 mean radius.
            return Err(S2GeographyError::Internal(
                "scalar UDF options not yet implemented".to_string(),
            ));
        }

        let arg_schema = Schema::new(arg_types);
        let mut ffi_arg_schema = FFI_ArrowSchema::try_from(arg_schema)?;
        let ffi_options = [0x00, 0x00, 0x00, 0x00];
        let mut ffi_field_out = FFI_ArrowSchema::empty();

        unsafe {
            let ffi_arg_schema: *mut ArrowSchema =
                &mut ffi_arg_schema as *mut FFI_ArrowSchema as *mut ArrowSchema;
            let ffi_out: *mut ArrowSchema =
                &mut ffi_field_out as *mut FFI_ArrowSchema as *mut ArrowSchema;
            let errc = self.inner.init.unwrap()(
                &mut self.inner,
                ffi_arg_schema,
                ffi_options.as_ptr(),
                ffi_out,
            );

            let last_err = self.last_error();
            if errc != 0 && last_err.is_empty() {
                Err(S2GeographyError::Code(errc))
            } else if errc != 0 {
                Err(S2GeographyError::Message(errc, last_err))
            } else {
                Ok(ffi_field_out)
            }
        }
    }

    /// Execute a batch
    ///
    /// The resulting FFI_ArrowArray requires the FFI_ArrowSchema returned by
    /// init() to be transformed into an [ArrayRef].
    pub fn execute(&mut self, args: &[ArrayRef]) -> Result<FFI_ArrowArray, S2GeographyError> {
        let mut args_ffi = args
            .iter()
            .map(|arg| arrow_array::ffi::to_ffi(&arg.to_data()))
            .collect::<Result<Vec<_>, ArrowError>>()?;
        let arg_ptrs = args_ffi
            .iter_mut()
            .map(|arg| &mut arg.0 as *mut FFI_ArrowArray)
            .collect::<Vec<_>>();
        let mut ffi_array_out = FFI_ArrowArray::empty();

        unsafe {
            let arg_ptrs: *mut *mut ArrowArray = arg_ptrs.as_ptr() as *mut *mut ArrowArray;
            let ffi_out: *mut ArrowArray =
                &mut ffi_array_out as *mut FFI_ArrowArray as *mut ArrowArray;
            let errc = self.inner.execute.unwrap()(
                &mut self.inner,
                arg_ptrs,
                args_ffi.len() as i64,
                ffi_out,
            );

            let last_err = self.last_error();
            if errc != 0 && last_err.is_empty() {
                Err(S2GeographyError::Code(errc))
            } else if errc != 0 {
                Err(S2GeographyError::Message(errc, last_err))
            } else {
                Ok(ffi_array_out)
            }
        }
    }

    fn last_error(&mut self) -> String {
        let c_str = unsafe {
            let raw_c_str = self.inner.get_last_error.unwrap()(&mut self.inner);
            std::ffi::CStr::from_ptr(raw_c_str)
        };

        c_str.to_string_lossy().into_owned()
    }

    fn allocate() -> Self {
        Self {
            inner: SedonaGeographyArrowUdf {
                init: None,
                execute: None,
                get_last_error: None,
                release: None,
                private_data: std::ptr::null_mut(),
            },
        }
    }
}

/// Dependency versions for underlying libraries
pub struct Versions {}

impl Versions {
    /// Return the statically linked nanoarrow version as a string
    pub fn nanoarrow() -> String {
        unsafe {
            let raw_c_str = SedonaGeographyGlueNanoarrowVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Return the statically linked geoarrow version as a string
    pub fn geoarrow() -> String {
        unsafe {
            let raw_c_str = SedonaGeographyGlueGeoArrowVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Return the statically linked s2 version as a string
    pub fn s2geometry() -> String {
        unsafe {
            let raw_c_str = SedonaGeographyGlueS2GeometryVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Return the linked Abseil version as a string
    ///
    /// Depending on build-time settings, this may have been statically
    /// or dynamically linked.
    pub fn abseil() -> String {
        unsafe {
            let raw_c_str = SedonaGeographyGlueAbseilVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Return the linked OpenSSL version as a string
    ///
    /// Depending on build-time settings, this may have been statically
    /// or dynamically linked.
    pub fn openssl() -> String {
        unsafe {
            let raw_c_str = SedonaGeographyGlueOpenSSLVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// A simple function that performs a non-trivial operation
    ///
    /// This is needed as a smoke check to ensure required libraries are linked.
    pub fn test_linkage() -> f64 {
        unsafe { SedonaGeographyGlueTestLinkage() }
    }
}

#[cfg(test)]
mod test {
    use arrow_array::ffi;
    use arrow_schema::DataType;
    use sedona_schema::datatypes::WKB_GEOGRAPHY;
    use sedona_testing::create::create_array_storage;

    use super::*;

    #[test]
    fn scalar_udf() {
        let mut udf = S2ScalarUDF::Length();

        let out_field_ffi = udf
            .init(
                vec![WKB_GEOGRAPHY.to_storage_field("", true).unwrap()].into(),
                None,
            )
            .unwrap();

        let out_data_type = DataType::try_from(&out_field_ffi).unwrap();
        assert_eq!(out_data_type, DataType::Float64);

        let in_array = create_array_storage(
            &[
                Some("POINT (0 1)"),
                Some("LINESTRING (0 0, 0 1)"),
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                None,
            ],
            &WKB_GEOGRAPHY,
        );

        let out_array_ffi = udf.execute(&[in_array]).unwrap();
        let out_array_data = unsafe { ffi::from_ffi(out_array_ffi, &out_field_ffi).unwrap() };
        let out_array = arrow_array::make_array(out_array_data);

        let expected: ArrayRef = arrow_array::create_array!(
            Float64,
            [Some(0.0), Some(111195.10117748393), Some(0.0), None]
        );
        assert_eq!(&out_array, &expected);
    }

    #[test]
    fn scalar_udf_errors() {
        let mut udf = S2ScalarUDF::Length();
        let err = udf.init(Fields::empty(), None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid argument: Expected one argument in unary s2geography UDF"
        );

        let err = udf.execute(&[]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid argument: Expected one argument/one argument type in in unary s2geography UDF"
        );
    }

    #[test]
    fn test_versions() {
        assert_eq!(Versions::nanoarrow(), "0.7.0-SNAPSHOT");
        assert_eq!(Versions::geoarrow(), "0.2.0-SNAPSHOT");
        assert_eq!(Versions::s2geometry(), "0.11.1");
        assert!(Versions::abseil().starts_with("20"));
        assert!(Versions::openssl().contains("."));
        assert!(Versions::test_linkage() > 0.0);
    }
}
