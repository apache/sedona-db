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
use std::{sync::Arc, vec};

use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ArgMatcher, ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_schema::datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY};

use crate::geoarrow_c::{ArrayReader, ArrayWriter, CError, Visitor};

/// ST_GeomFromWKT() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKT reading using geoarrow-c's WKT reader
pub fn st_geomfromwkt_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCCast::new(
        ArgMatcher::new(vec![ArgMatcher::is_string()], WKB_GEOMETRY),
        Some(WKB_GEOMETRY),
        WKB_GEOMETRY,
    ))
}

/// ST_GeogFromWKT() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKT reading using geoarrow-c's WKT reader
pub fn st_geogfromwkt_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCCast::new(
        ArgMatcher::new(vec![ArgMatcher::is_string()], WKB_GEOGRAPHY),
        Some(WKB_GEOGRAPHY),
        WKB_GEOGRAPHY,
    ))
}

/// ST_GeomFromWKB() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKB validation using geoarrow-c's WKB reader
pub fn st_geomfromwkb_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCCast::new(
        ArgMatcher::new(vec![ArgMatcher::is_binary()], WKB_GEOMETRY),
        None,
        WKB_GEOMETRY,
    ))
}

/// ST_GeogFromWKB() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKB validation using geoarrow-c's WKB reader
pub fn st_geogfromwkb_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCCast::new(
        ArgMatcher::new(vec![ArgMatcher::is_binary()], WKB_GEOGRAPHY),
        None,
        WKB_GEOGRAPHY,
    ))
}

/// ST_AsText() scalar UDF implementation using geoarrow-c
///
/// An implementation of WKT writing using geoarrow-c's WKT writer
pub fn st_astext_impl() -> ScalarKernelRef {
    Arc::new(GeoArrowCCast::new(
        ArgMatcher::new(vec![ArgMatcher::is_geometry_or_geography()], STRING),
        Some(STRING),
        STRING,
    ))
}

#[derive(Debug)]
struct GeoArrowCCast {
    matcher: ArgMatcher,
    writer_type: Option<SedonaType>,
    output_type: DataType,
}

impl GeoArrowCCast {
    pub fn new(
        matcher: ArgMatcher,
        writer_type: Option<SedonaType>,
        output_type: SedonaType,
    ) -> Self {
        Self {
            matcher,
            writer_type,
            output_type: output_type.storage_type().clone(),
        }
    }
}

impl SedonaScalarKernel for GeoArrowCCast {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let array_in = args[0].to_array(executor.num_iterations())?;

        let mut error = CError::new();
        let mut reader = ArrayReader::try_new(&arg_types[0])?;

        let array_out = if let Some(output_type) = &self.writer_type {
            let mut writer = ArrayWriter::try_new(output_type)?;
            let visitor = writer.visitor_mut();

            reader.visit(&array_in, visitor, &mut error)?;
            Arc::new(writer.finish()?)
        } else {
            let mut visitor = Visitor::void();

            reader.visit(&array_in, &mut visitor, &mut error)?;
            array_in
        };

        let value_out = executor.finish(array_out)?;
        if self.output_type == value_out.data_type() {
            Ok(value_out)
        } else {
            value_out.cast_to(&self.output_type, None)
        }
    }
}

const STRING: SedonaType = SedonaType::Arrow(DataType::Utf8);

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use arrow_schema::DataType;
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_functions::register::default_function_set;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};

    use sedona_testing::{
        compare::assert_value_equal,
        create::{create_array_value, create_scalar_storage, create_scalar_value},
    };

    use super::*;

    #[rstest]
    fn fromwkt(#[values(DataType::Utf8, DataType::Utf8View)] data_type: DataType) {
        let mut function_set = default_function_set();
        let udf = function_set.scalar_udf_mut("st_geomfromwkt").unwrap();
        udf.add_kernel(st_geomfromwkt_impl());

        assert_value_equal(
            &udf.invoke_batch(
                &[
                    ColumnarValue::Scalar(ScalarValue::Utf8(Some("POINT (1 2)".to_string())))
                        .cast_to(&data_type, None)
                        .unwrap(),
                ],
                1,
            )
            .unwrap(),
            &create_scalar_value(Some("POINT (1 2)"), &WKB_GEOMETRY),
        );

        let utf8_array: StringArray = [Some("POINT (1 2)"), None, Some("POINT (3 4)")]
            .iter()
            .collect();
        let utf8_value = ColumnarValue::Array(Arc::new(utf8_array))
            .cast_to(&data_type, None)
            .unwrap();
        assert_value_equal(
            &udf.invoke_batch(&[utf8_value], 1).unwrap(),
            &create_array_value(
                &[Some("POINT (1 2)"), None, Some("POINT (3 4)")],
                &WKB_GEOMETRY,
            ),
        );
    }

    #[rstest]
    fn fromwkb(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] data_type: SedonaType) {
        let mut function_set = default_function_set();
        let udf = function_set.scalar_udf_mut("st_geomfromwkb").unwrap();
        udf.add_kernel(st_geomfromwkb_impl());

        assert_value_equal(
            &udf.invoke_batch(
                &[create_scalar_storage(Some("POINT (1 2)"), &data_type).into()],
                1,
            )
            .unwrap(),
            &create_scalar_value(Some("POINT (1 2)"), &WKB_GEOMETRY),
        );
    }

    #[rstest]
    fn astext(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] data_type: SedonaType) {
        let mut function_set = default_function_set();
        let udf = function_set.scalar_udf_mut("st_astext").unwrap();
        udf.add_kernel(st_astext_impl());

        assert_value_equal(
            &udf.invoke_batch(&[create_scalar_value(Some("POINT (1 2)"), &data_type)], 1)
                .unwrap(),
            &ScalarValue::Utf8(Some("POINT (1 2)".to_string())).into(),
        );
    }

    #[test]
    fn errors() {
        let mut function_set = default_function_set();
        let udf = function_set.scalar_udf_mut("st_geomfromwkt").unwrap();
        udf.add_kernel(st_geomfromwkt_impl());

        let err = udf
            .invoke_batch(
                &[ScalarValue::Utf8(Some("this is not valid wkt".to_string())).into()],
                1,
            )
            .unwrap_err();

        assert_eq!(
            err.message(),
            "Invalid argument: Expected geometry type at byte 0"
        );
    }

    #[test]
    fn geog() {
        let mut function_set = default_function_set();
        let udf = function_set.scalar_udf_mut("st_geogfromwkt").unwrap();
        udf.add_kernel(st_geogfromwkt_impl());

        assert_value_equal(
            &udf.invoke_batch(
                &[ScalarValue::Utf8(Some("POINT (1 2)".to_string())).into()],
                1,
            )
            .unwrap(),
            &create_scalar_value(Some("POINT (1 2)"), &WKB_GEOGRAPHY),
        );

        let udf = function_set.scalar_udf_mut("st_geogfromwkb").unwrap();
        udf.add_kernel(st_geogfromwkb_impl());
        assert_value_equal(
            &udf.invoke_batch(
                &[create_scalar_storage(Some("POINT (1 2)"), &WKB_GEOGRAPHY).into()],
                1,
            )
            .unwrap(),
            &create_scalar_value(Some("POINT (1 2)"), &WKB_GEOGRAPHY),
        );
    }
}
