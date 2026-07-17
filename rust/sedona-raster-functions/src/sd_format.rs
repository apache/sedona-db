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
use std::{fmt::Write, sync::Arc};

use arrow_array::{builder::StringBuilder, cast::AsArray};
use arrow_schema::DataType;
use datafusion_common::{
    error::{DataFusionError, Result},
    internal_err, ScalarValue,
};
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_raster::{array::RasterStructArray, display::RasterDisplay};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

pub fn sd_format_raster_kernel() -> ScalarKernelRef {
    Arc::new(SDFormatRaster {})
}

#[derive(Debug)]
struct SDFormatRaster {}

impl SedonaScalarKernel for SDFormatRaster {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::optional(ArgMatcher::is_string()),
            ],
            SedonaType::Arrow(DataType::Utf8),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let maybe_width_hint = format_width_hint(args)?;
        raster_value_to_formatted_value(&args[0], maybe_width_hint)
    }
}

fn format_width_hint(args: &[ColumnarValue]) -> Result<Option<usize>> {
    if args.len() < 2 {
        return Ok(None);
    }

    if let ColumnarValue::Scalar(ScalarValue::Utf8(Some(options_value))) =
        args[1].cast_to(&DataType::Utf8, None)?
    {
        let options: serde_json::Value = options_value
            .parse()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        if let Some(width_hint_value) = options.get("width_hint") {
            if let Some(width_hint_i64) = width_hint_value.as_i64() {
                return Ok(Some(
                    width_hint_i64
                        .try_into()
                        .map_err(|e| DataFusionError::External(Box::new(e)))?,
                ));
            }
        }
    }

    Ok(None)
}

fn raster_value_to_formatted_value(
    columnar_value: &ColumnarValue,
    maybe_width_hint: Option<usize>,
) -> Result<ColumnarValue> {
    match columnar_value {
        ColumnarValue::Array(array) => {
            let struct_array = array.as_struct();
            let raster_array = RasterStructArray::try_new(struct_array)?;
            let min_output_size = match maybe_width_hint {
                Some(width_hint) => raster_array.len() * width_hint,
                None => raster_array.len() * 48,
            };
            let mut builder =
                StringBuilder::with_capacity(raster_array.len(), min_output_size.max(1));

            for i in 0..raster_array.len() {
                if raster_array.is_null(i) {
                    builder.append_null();
                    continue;
                }

                let raster = raster_array.get(i)?;
                let mut limited_output =
                    LimitedSizeOutput::new(&mut builder, maybe_width_hint.unwrap_or(usize::MAX));
                let _ = write!(limited_output, "{}", RasterDisplay(&raster));
                builder.append_value("");
            }

            Ok(ColumnarValue::Array(Arc::new(builder.finish())))
        }
        ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
            let formatted = raster_value_to_formatted_value(
                &ColumnarValue::Array(Arc::new(struct_array.as_ref().clone())),
                maybe_width_hint,
            )?;
            if let ColumnarValue::Array(array) = formatted {
                Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                    &array, 0,
                )?))
            } else {
                internal_err!("Expected array formatted value for raster scalar")
            }
        }
        _ => internal_err!("Unsupported raster columnar value"),
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
    use std::sync::Arc;

    use arrow_array::Array;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{SedonaType, RASTER};
    use sedona_testing::{rasters::generate_test_rasters, testers::ScalarUdfTester};

    use super::*;

    fn sd_format_raster_udf() -> SedonaScalarUDF {
        SedonaScalarUDF::from_impl("sd_format", sd_format_raster_kernel())
    }

    #[test]
    fn sd_format_formats_raster_columns() {
        let udf = sd_format_raster_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER]);

        let raster_array = generate_test_rasters(3, Some(1)).unwrap();
        let result = tester.invoke_array(Arc::new(raster_array.clone())).unwrap();
        let formatted = result.as_string::<i32>();

        assert_eq!(formatted.value(0), "[1x2/1] @ [1 1.6 1.1 2] / OGC:CRS84");
        assert!(formatted.is_null(1));
        assert_eq!(
            formatted.value(2),
            "[3x4/1] @ [3 2.4 3.84 4.24] skew=(0.06, 0.08) / OGC:CRS84"
        );
    }

    #[test]
    fn sd_format_formats_raster_columns_with_null() {
        let udf = sd_format_raster_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER]);

        let raster_array = generate_test_rasters(3, Some(1)).unwrap();
        let result = tester.invoke_array(Arc::new(raster_array)).unwrap();
        let formatted = result.as_string::<i32>();

        assert!(formatted.value(0).starts_with("[1x2/"));
        assert!(formatted.is_null(1));
        assert!(formatted.value(2).starts_with("[3x4/"));
    }

    #[test]
    fn sd_format_formats_raster_columns_with_width_hint() {
        let udf = sd_format_raster_udf();
        let tester =
            ScalarUdfTester::new(udf.into(), vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let raster_array = generate_test_rasters(2, None).unwrap();
        let result = tester
            .invoke_array_scalar(Arc::new(raster_array), r#"{"width_hint": 10}"#)
            .unwrap();
        let formatted = result.as_string::<i32>();

        let full_output = "[1x2/1] @ [1 1.6 1.1 2] / OGC:CRS84";
        assert!(formatted.value(0).starts_with("["));
        assert!(formatted.value(0).len() < full_output.len());
    }
}
