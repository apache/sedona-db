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
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

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
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Utf8),
        ),
        Some(SedonaType::Arrow(DataType::Utf8)),
        SedonaType::Arrow(DataType::Utf8),
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

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use arrow_schema::DataType;
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};

    use sedona_testing::{create::create_scalar_storage, testers::ScalarUdfTester};

    use super::*;

    #[rstest]
    fn fromwkt(#[values(DataType::Utf8, DataType::Utf8View)] data_type: DataType) {
        use sedona_testing::create::create_array;

        let udf = SedonaScalarUDF::from_kernel("st_geomfromwkt", st_geomfromwkt_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(data_type)]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester.invoke_scalar("POINT (1 2)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");

        let utf8_array: StringArray = [Some("POINT (1 2)"), None, Some("POINT (3 4)")]
            .iter()
            .collect();

        assert_eq!(
            &tester.invoke_array(Arc::new(utf8_array)).unwrap(),
            &create_array(
                &[Some("POINT (1 2)"), None, Some("POINT (3 4)")],
                &WKB_GEOMETRY,
            )
        );
    }

    #[rstest]
    fn fromwkb(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] data_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_geomfromwkb", st_geomfromwkb_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![SedonaType::Arrow(data_type.storage_type().clone())],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar(create_scalar_storage(Some("POINT (1 2)"), &data_type))
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");
    }

    #[rstest]
    fn astext(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] data_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_astext", st_astext_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![data_type]);
        tester.assert_return_type(DataType::Utf8);

        let result = tester.invoke_scalar("POINT (1 2)").unwrap();
        assert_eq!(result, ScalarValue::Utf8(Some("POINT (1 2)".to_string())));
    }

    #[test]
    fn errors() {
        let udf = SedonaScalarUDF::from_kernel("st_geomfromwkt", st_geomfromwkt_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Utf8)]);
        let err = tester.invoke_scalar("This is not valid wkt").unwrap_err();

        assert_eq!(
            err.message(),
            "Invalid argument: Expected geometry type at byte 0"
        );
    }

    #[test]
    fn geog() {
        let udf = SedonaScalarUDF::from_kernel("st_geogfromwkt", st_geogfromwkt_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Utf8)]);
        tester.assert_return_type(WKB_GEOGRAPHY);

        let result = tester.invoke_scalar("POINT (1 2)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");

        let udf = SedonaScalarUDF::from_kernel("st_geogfromwkb", st_geogfromwkb_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Binary)]);
        tester.assert_return_type(WKB_GEOGRAPHY);

        let result = tester
            .invoke_scalar(create_scalar_storage(Some("POINT (1 2)"), &WKB_GEOGRAPHY))
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");
    }
}
