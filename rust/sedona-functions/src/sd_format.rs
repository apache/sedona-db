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

use crate::executor::WkbExecutor;
use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::{
    error::{DataFusionError, Result},
    ScalarValue,
};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{ArgMatcher, SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;

/// SD_Format() scalar UDF implementation
///
/// This function is invoked to obtain a proxy array with human-readable
/// output. For most arrays, this just returns the array (which will be
/// formatted using its storage type by the Arrow formatter).
pub fn sd_format_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "sd_format",
        vec![Arc::new(SDFormatDefault {}), Arc::new(SDFormatGeometry {})],
        Volatility::Immutable,
        Some(sd_format_doc()),
    )
}

fn sd_format_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return a version of value suitable for formatting/display with
         the options provided. This is used to inject custom behaviour for a
         SedonaType specifically for formatting values.",
        "SD_Format (value: Any, [options: String])",
    )
    .with_argument("value", "Any: Any input value")
    .with_argument(
        "options",
        "
    String: JSON-encoded options. The following options are currently supported:

    - width_hint (numeric): The approximate width of the output. The value provided will
      typically be an overestimate and the value may be further abrevidated by
      the renderer. This value is purely a hint and may be ignored.",
    )
    .with_sql_example("SELECT SD_Format(ST_Point(1.0, 2.0, '{}'))")
    .build()
}

/// Default implementation that returns its input (i.e., by default, just
/// do whatever DataFusion would have done with the value and ignore any
/// options that were provided)
#[derive(Debug)]
struct SDFormatDefault {}

impl SedonaScalarKernel for SDFormatDefault {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        Ok(Some(args[0].clone()))
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        Ok(args[0].clone())
    }
}

/// Implementation format geometry or geography
///
/// This is very similar to ST_AsText except it respects the width_hint by
/// stopping the render for each item when too many characters have been written.
#[derive(Debug)]
struct SDFormatGeometry {}

impl SedonaScalarKernel for SDFormatGeometry {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_optional(ArgMatcher::is_string()),
            ],
            SedonaType::Arrow(DataType::Utf8),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let mut maybe_width_hint: Option<usize> = None;
        if args.len() >= 2 {
            if let ColumnarValue::Scalar(ScalarValue::Utf8(Some(options_value))) =
                args[1].cast_to(&DataType::Utf8, None)?
            {
                let options: serde_json::Value = options_value
                    .parse()
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                if let Some(width_hint_value) = options.get("width_hint") {
                    if let Some(width_hint_i64) = width_hint_value.as_i64() {
                        maybe_width_hint = Some(
                            width_hint_i64
                                .try_into()
                                .map_err(|e| DataFusionError::External(Box::new(e)))?,
                        );
                    }
                }
            }
        }

        let executor = WkbExecutor::new(&arg_types[0..1], &args[0..1]);

        let min_output_size = match maybe_width_hint {
            Some(width_hint) => executor.num_iterations() * width_hint,
            None => executor.num_iterations() * 25,
        };

        // Initialize an output builder of the appropriate type
        let mut builder = StringBuilder::with_capacity(executor.num_iterations(), min_output_size);

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    let mut builder_wrapper = LimitedSizeOutput::new(
                        &mut builder,
                        maybe_width_hint.unwrap_or(usize::MAX),
                    );

                    // We ignore this error on purpose: we raised it on purpose to prevent
                    // the WKT writer from writing too many characters
                    #[allow(unused_must_use)]
                    wkt::to_wkt::write_geometry(&mut builder_wrapper, &item);

                    builder.append_value("");
                }
                None => builder.append_null(),
            };

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

struct LimitedSizeOutput<'a, T> {
    inner: &'a mut T,
    current_item_size: usize,
    max_item_size: usize,
}

impl<'a, T> LimitedSizeOutput<'a, T> {
    pub fn new(inner: &'a mut T, max_item_size: usize) -> Self {
        Self {
            inner,
            current_item_size: 0,
            max_item_size,
        }
    }
}

impl<'a, T: std::fmt::Write> std::fmt::Write for LimitedSizeOutput<'a, T> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.inner.write_str(s)?;
        self.current_item_size += s.len();
        if self.current_item_size > self.max_item_size {
            Err(std::fmt::Error)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, StringArray};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{create::create_array, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = sd_format_udf().into();
        assert_eq!(udf.name(), "sd_format");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) {
        use arrow_array::ArrayRef;

        let udf = sd_format_udf();
        let unary_tester = ScalarUdfTester::new(udf.clone().into(), vec![sedona_type.clone()]);
        let binary_tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Utf8)],
        );

        // With omitted, Null, or invalid options, the output should be identical
        let wkt_values = vec![Some("POINT(1 2)"), None, Some("LINESTRING(3 5,7 8)")];
        let wkt_array = create_array(&wkt_values, &sedona_type);
        let expected_array: ArrayRef = Arc::new(wkt_values.iter().collect::<StringArray>());

        assert_eq!(
            &unary_tester.invoke_wkb_array(wkt_values.clone()).unwrap(),
            &expected_array
        );
        assert_eq!(
            &binary_tester
                .invoke_array_scalar(wkt_array.clone(), "{}")
                .unwrap(),
            &expected_array
        );
        assert_eq!(
            &binary_tester
                .invoke_array_scalar(wkt_array.clone(), ScalarValue::Null)
                .unwrap(),
            &expected_array
        );

        // Invalid options should error
        let err = binary_tester
            .invoke_array_scalar(wkt_array.clone(), r#"{"width_hint": -1}"#)
            .unwrap_err();
        assert_eq!(
            err.message(),
            "out of range integral type conversion attempted"
        );

        // For a very small width hint, we should get truncated values
        let expected_array: ArrayRef =
            create_array!(Utf8, [Some("POINT"), None, Some("LINESTRING")]);
        assert_eq!(
            &binary_tester
                .invoke_array_scalar(wkt_array.clone(), r#"{"width_hint": 3}"#)
                .unwrap(),
            &expected_array
        );
    }
}
