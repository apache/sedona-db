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

use crate::executor::RasterExecutor;
use arrow_array::builder::StringBuilder;
use arrow_array::cast::AsArray;
use arrow_array::Array;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_GeoReference() scalar UDF implementation
///
/// Returns the georeference metadata of raster as a string in GDAL or ESRI format
pub fn rs_georeference_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_georeference",
        vec![
            Arc::new(RsGeoReferenceOneArg {}),
            Arc::new(RsGeoReferenceTwoArg {}),
        ],
        Volatility::Immutable,
        Some(rs_georeference_doc()),
    )
}

fn rs_georeference_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the georeference metadata of raster as a string in GDAL or ESRI format as commonly seen in a world file. Default is GDAL if not specified. Both formats output six lines: scalex, skewy, skewx, scaley, upperleftx, upperlefty. In GDAL format the upper-left coordinates refer to the corner of the upper-left pixel, while in ESRI format they are shifted to the center of the upper-left pixel.".to_string(),
        "RS_GeoReference(raster: Raster, format: String = 'GDAL')".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("format", "String: Output format, either 'GDAL' (default) or 'ESRI'. GDAL reports the upper-left corner of the upper-left pixel; ESRI shifts the coordinates to the center of the upper-left pixel.")
    .with_sql_example("SELECT RS_GeoReference(RS_Example())".to_string())
    .build()
}

/// Format type for GeoReference output as commonly seen in a
/// [world file](https://en.wikipedia.org/wiki/World_file).
///
/// Both formats output six lines: scalex, skewy, skewx, scaley, upperleftx, upperlefty.
/// The difference is how the upper-left coordinate is reported:
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeoReferenceFormat {
    /// GDAL format: upperleftx and upperlefty are the coordinates of the upper-left corner
    /// of the upper-left pixel.
    Gdal,
    /// ESRI format: upperleftx and upperlefty are shifted to the center of the upper-left
    /// pixel, i.e. `upperleftx + scalex * 0.5` and `upperlefty + scaley * 0.5`.
    Esri,
}

impl GeoReferenceFormat {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "GDAL" => Ok(GeoReferenceFormat::Gdal),
            "ESRI" => Ok(GeoReferenceFormat::Esri),
            _ => Err(DataFusionError::Execution(format!(
                "Invalid GeoReference format '{}'. Supported formats are 'GDAL' and 'ESRI'.",
                s
            ))),
        }
    }
}

/// Estimated bytes per georeference string for StringBuilder preallocation.
/// Output is 6 lines of `{:.10}` formatted f64 values separated by newlines.
/// Each value is at most ~20 bytes (e.g. "-12345678.1234567890"), giving
/// 6 * 20 + 5 newlines = 125 bytes.
const PREALLOC_BYTES_PER_GEOREF: usize = 125;

/// One-argument kernel: RS_GeoReference(raster) - uses GDAL format by default
#[derive(Debug)]
struct RsGeoReferenceOneArg {}

impl SedonaScalarKernel for RsGeoReferenceOneArg {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Utf8),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);

        let preallocate_bytes = PREALLOC_BYTES_PER_GEOREF * executor.num_iterations();
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), preallocate_bytes);

        executor.execute_raster_void(|_i, raster_opt| {
            format_georeference(raster_opt, GeoReferenceFormat::Gdal, &mut builder)
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Two-argument kernel: RS_GeoReference(raster, format)
#[derive(Debug)]
struct RsGeoReferenceTwoArg {}

impl SedonaScalarKernel for RsGeoReferenceTwoArg {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_string()],
            SedonaType::Arrow(DataType::Utf8),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);

        // Expand the format parameter to an array
        let format_array = args[1].clone().into_array(executor.num_iterations())?;
        let format_array = format_array.as_string::<i32>();

        let preallocate_bytes = PREALLOC_BYTES_PER_GEOREF * executor.num_iterations();
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), preallocate_bytes);

        executor.execute_raster_void(|i, raster_opt| {
            if format_array.is_null(i) {
                builder.append_null();
                return Ok(());
            }
            let format = GeoReferenceFormat::from_str(format_array.value(i))?;
            format_georeference(raster_opt, format, &mut builder)
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Format the georeference metadata for a raster
fn format_georeference(
    raster_opt: Option<&sedona_raster::array::RasterRefImpl<'_>>,
    format: GeoReferenceFormat,
    builder: &mut StringBuilder,
) -> Result<()> {
    match raster_opt {
        None => builder.append_null(),
        Some(raster) => {
            let metadata = raster.metadata();
            let scale_x = metadata.scale_x();
            let scale_y = metadata.scale_y();
            let skew_x = metadata.skew_x();
            let skew_y = metadata.skew_y();
            let upper_left_x = metadata.upper_left_x();
            let upper_left_y = metadata.upper_left_y();

            let georeference = match format {
                GeoReferenceFormat::Gdal => {
                    format!(
                        "{:.10}\n{:.10}\n{:.10}\n{:.10}\n{:.10}\n{:.10}",
                        scale_x, skew_y, skew_x, scale_y, upper_left_x, upper_left_y
                    )
                }
                GeoReferenceFormat::Esri => {
                    let esri_upper_left_x = upper_left_x + scale_x * 0.5;
                    let esri_upper_left_y = upper_left_y + scale_y * 0.5;
                    format!(
                        "{:.10}\n{:.10}\n{:.10}\n{:.10}\n{:.10}\n{:.10}",
                        scale_x, skew_y, skew_x, scale_y, esri_upper_left_x, esri_upper_left_y
                    )
                }
            };

            builder.append_value(georeference);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, StringArray};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_georeference_udf().into();
        assert_eq!(udf.name(), "rs_georeference");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn udf_georeference_gdal_default() {
        let udf: ScalarUDF = rs_georeference_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        tester.assert_return_type(DataType::Utf8);

        // Test with rasters (one-arg, default GDAL)
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let result = tester.invoke_array(Arc::new(rasters.clone())).unwrap();

        let expected: Arc<dyn Array> = Arc::new(StringArray::from(vec![
            Some("0.1000000000\n0.0000000000\n0.0000000000\n-0.2000000000\n1.0000000000\n2.0000000000"),
            None,
            Some("0.2000000000\n0.0800000000\n0.0600000000\n-0.4000000000\n3.0000000000\n4.0000000000"),
        ]));
        assert_array_equal(&result, &expected);

        // Test with explicit "GDAL" or "gdal" (two-arg)
        for format in ["GDAL", "gdal"] {
            let udf: ScalarUDF = rs_georeference_udf().into();
            let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);
            let result = tester
                .invoke_array_scalar(Arc::new(rasters.clone()), format)
                .unwrap();
            assert_array_equal(&result, &expected);
        }
    }

    #[test]
    fn udf_georeference_esri() {
        let udf: ScalarUDF = rs_georeference_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let expected: Arc<dyn Array> = Arc::new(StringArray::from(vec![
            Some("0.1000000000\n0.0000000000\n0.0000000000\n-0.2000000000\n1.0500000000\n1.9000000000"),
            None,
            Some("0.2000000000\n0.0800000000\n0.0600000000\n-0.4000000000\n3.1000000000\n3.8000000000"),
        ]));

        for format in ["ESRI", "esri"] {
            let rasters = generate_test_rasters(3, Some(1)).unwrap();
            let result = tester
                .invoke_array_scalar(Arc::new(rasters), format)
                .unwrap();
            assert_array_equal(&result, &expected);
        }
    }

    #[test]
    fn udf_georeference_null_scalar() {
        let udf: ScalarUDF = rs_georeference_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        // Test with null scalar
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));
    }

    #[test]
    fn udf_georeference_with_array_format() {
        let udf: ScalarUDF = rs_georeference_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = generate_test_rasters(4, Some(1)).unwrap();
        let formats = Arc::new(StringArray::from(vec![
            Some("GDAL"), // explicit GDAL
            Some("ESRI"), // won't matter since raster 1 is null
            None,         // null format -> NULL output
            Some("ESRI"), // explicit ESRI
        ]));

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), formats])
            .unwrap();
        let expected: Arc<dyn Array> = Arc::new(StringArray::from(vec![
                // explicit GDAL
                Some("0.1000000000\n0.0000000000\n0.0000000000\n-0.2000000000\n1.0000000000\n2.0000000000"),
                // null raster
                None,
                // null format -> NULL output
                None,
                // explicit ESRI
                Some("0.3000000000\n0.1200000000\n0.0900000000\n-0.6000000000\n4.1500000000\n4.7000000000"),
        ]));
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn udf_georeference_invalid_format() {
        let udf: ScalarUDF = rs_georeference_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let result = tester.invoke_array_scalar(Arc::new(rasters), "INVALID");

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Invalid GeoReference format"),
            "Expected error about invalid format, got: {}",
            err_msg
        );
    }
}
