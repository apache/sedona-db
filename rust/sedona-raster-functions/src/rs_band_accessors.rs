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

use std::sync::Arc;
use std::vec;

use crate::executor::RasterExecutor;
use arrow_array::builder::{BooleanBuilder, Float64Builder, StringBuilder};
use arrow_array::{cast::AsArray, types::Int32Type, Array};
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_common::{exec_err, DataFusionError};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::raster::{BandDataType, StorageType};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a BandDataType to the Java-compatible pixel type name.
fn band_data_type_name(dt: &BandDataType) -> &'static str {
    match dt {
        BandDataType::UInt8 => "UNSIGNED_8BITS",
        BandDataType::UInt16 => "UNSIGNED_16BITS",
        BandDataType::Int16 => "SIGNED_16BITS",
        BandDataType::Int32 => "SIGNED_32BITS",
        BandDataType::Float32 => "REAL_32BITS",
        BandDataType::Float64 => "REAL_64BITS",
        // Extra types present in Rust but not in the Java Sedona
        BandDataType::UInt32 => "UNSIGNED_32BITS",
        BandDataType::UInt64 => "UNSIGNED_64BITS",
        BandDataType::Int64 => "SIGNED_64BITS",
        BandDataType::Int8 => "SIGNED_8BITS",
    }
}

/// Byte size of a single pixel for the given band data type.
fn data_type_byte_size(data_type: &BandDataType) -> usize {
    match data_type {
        BandDataType::UInt8 | BandDataType::Int8 => 1,
        BandDataType::UInt16 | BandDataType::Int16 => 2,
        BandDataType::UInt32 | BandDataType::Int32 | BandDataType::Float32 => 4,
        BandDataType::UInt64 | BandDataType::Int64 | BandDataType::Float64 => 8,
    }
}

/// Convert raw nodata bytes to f64 for the given band data type.
fn bytes_to_f64(bytes: &[u8], band_type: &BandDataType) -> Result<f64> {
    macro_rules! read_le_f64 {
        ($t:ty, $n:expr) => {{
            let arr: [u8; $n] = bytes.try_into().map_err(|_| {
                DataFusionError::Execution(format!(
                    "Invalid byte slice length for type {}, expected: {}, actual: {}",
                    stringify!($t),
                    $n,
                    bytes.len()
                ))
            })?;
            Ok(<$t>::from_le_bytes(arr) as f64)
        }};
    }

    match band_type {
        BandDataType::UInt8 => {
            if bytes.len() != 1 {
                return Err(DataFusionError::Execution(format!(
                    "Invalid byte length for UInt8: expected 1, got {}",
                    bytes.len()
                )));
            }
            Ok(bytes[0] as f64)
        }
        BandDataType::Int8 => {
            if bytes.len() != 1 {
                return Err(DataFusionError::Execution(format!(
                    "Invalid byte length for Int8: expected 1, got {}",
                    bytes.len()
                )));
            }
            Ok(bytes[0] as i8 as f64)
        }
        BandDataType::UInt16 => read_le_f64!(u16, 2),
        BandDataType::Int16 => read_le_f64!(i16, 2),
        BandDataType::UInt32 => read_le_f64!(u32, 4),
        BandDataType::Int32 => read_le_f64!(i32, 4),
        BandDataType::UInt64 => read_le_f64!(u64, 8),
        BandDataType::Int64 => read_le_f64!(i64, 8),
        BandDataType::Float32 => read_le_f64!(f32, 4),
        BandDataType::Float64 => read_le_f64!(f64, 8),
    }
}

/// Validate and return a (1-based) band index, erroring if out of range.
fn validate_band_index(band_index: i32, num_bands: usize) -> Result<()> {
    if band_index < 1 || band_index as usize > num_bands {
        return exec_err!(
            "Provided band index {} is not in the range [1, {}]",
            band_index,
            num_bands
        );
    }
    Ok(())
}

// ===========================================================================
// RS_BandPixelType
// ===========================================================================

/// RS_BandPixelType() scalar UDF implementation
///
/// Returns the pixel data type of the specified band as a string.
/// Accepts an optional band_index parameter (1-based, default is 1).
pub fn rs_bandpixeltype_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_bandpixeltype",
        vec![
            Arc::new(RsBandPixelType {}),
            Arc::new(RsBandPixelTypeWithBand {}),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsBandPixelType {}

impl SedonaScalarKernel for RsBandPixelType {
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
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), executor.num_iterations() * 20);

        executor
            .execute_raster_void(|_i, raster_opt| get_pixel_type(raster_opt, 1, &mut builder))?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct RsBandPixelTypeWithBand {}

impl SedonaScalarKernel for RsBandPixelTypeWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
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
        let band_index_array = args[1].clone().into_array(executor.num_iterations())?;
        let band_index_array = band_index_array.as_primitive::<Int32Type>();

        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), executor.num_iterations() * 20);

        executor.execute_raster_void(|i, raster_opt| {
            let band_index = if band_index_array.is_null(i) {
                1
            } else {
                band_index_array.value(i)
            };
            get_pixel_type(raster_opt, band_index, &mut builder)
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn get_pixel_type(
    raster_opt: Option<&sedona_raster::array::RasterRefImpl<'_>>,
    band_index: i32,
    builder: &mut StringBuilder,
) -> Result<()> {
    match raster_opt {
        None => {
            builder.append_null();
            Ok(())
        }
        Some(raster) => {
            let num_bands = raster.bands().len();
            validate_band_index(band_index, num_bands)?;
            let band = raster.bands().band(band_index as usize)?;
            let dt = band.metadata().data_type()?;
            builder.append_value(band_data_type_name(&dt));
            Ok(())
        }
    }
}

// ===========================================================================
// RS_BandNoDataValue
// ===========================================================================

/// RS_BandNoDataValue() scalar UDF implementation
///
/// Returns the nodata value of the specified band as a Float64.
/// Returns null if the band has no nodata value defined.
/// Accepts an optional band_index parameter (1-based, default is 1).
pub fn rs_bandnodatavalue_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_bandnodatavalue",
        vec![
            Arc::new(RsBandNoDataValue {}),
            Arc::new(RsBandNoDataValueWithBand {}),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsBandNoDataValue {}

impl SedonaScalarKernel for RsBandNoDataValue {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Float64),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        executor
            .execute_raster_void(|_i, raster_opt| get_nodata_value(raster_opt, 1, &mut builder))?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct RsBandNoDataValueWithBand {}

impl SedonaScalarKernel for RsBandNoDataValueWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            SedonaType::Arrow(DataType::Float64),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let band_index_array = args[1].clone().into_array(executor.num_iterations())?;
        let band_index_array = band_index_array.as_primitive::<Int32Type>();

        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|i, raster_opt| {
            let band_index = if band_index_array.is_null(i) {
                1
            } else {
                band_index_array.value(i)
            };
            get_nodata_value(raster_opt, band_index, &mut builder)
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn get_nodata_value(
    raster_opt: Option<&sedona_raster::array::RasterRefImpl<'_>>,
    band_index: i32,
    builder: &mut Float64Builder,
) -> Result<()> {
    match raster_opt {
        None => {
            builder.append_null();
            Ok(())
        }
        Some(raster) => {
            let num_bands = raster.bands().len();
            validate_band_index(band_index, num_bands)?;
            let band = raster.bands().band(band_index as usize)?;
            let band_meta = band.metadata();
            match band_meta.nodata_value() {
                None => {
                    builder.append_null();
                }
                Some(nodata_bytes) => {
                    let dt = band_meta.data_type()?;
                    let val = bytes_to_f64(nodata_bytes, &dt)?;
                    builder.append_value(val);
                }
            }
            Ok(())
        }
    }
}

// ===========================================================================
// RS_BandIsNoData
// ===========================================================================

/// RS_BandIsNoData() scalar UDF implementation
///
/// Returns true if all pixels in the specified band equal the nodata value.
/// Returns false if the band has no nodata value defined.
/// Accepts an optional band_index parameter (1-based, default is 1).
pub fn rs_bandisnodata_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_bandisnodata",
        vec![
            Arc::new(RsBandIsNoData {}),
            Arc::new(RsBandIsNoDataWithBand {}),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsBandIsNoData {}

impl SedonaScalarKernel for RsBandIsNoData {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Boolean),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        executor
            .execute_raster_void(|_i, raster_opt| get_is_nodata(raster_opt, 1, &mut builder))?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct RsBandIsNoDataWithBand {}

impl SedonaScalarKernel for RsBandIsNoDataWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            SedonaType::Arrow(DataType::Boolean),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let band_index_array = args[1].clone().into_array(executor.num_iterations())?;
        let band_index_array = band_index_array.as_primitive::<Int32Type>();

        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|i, raster_opt| {
            let band_index = if band_index_array.is_null(i) {
                1
            } else {
                band_index_array.value(i)
            };
            get_is_nodata(raster_opt, band_index, &mut builder)
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn get_is_nodata(
    raster_opt: Option<&sedona_raster::array::RasterRefImpl<'_>>,
    band_index: i32,
    builder: &mut BooleanBuilder,
) -> Result<()> {
    match raster_opt {
        None => {
            builder.append_null();
            Ok(())
        }
        Some(raster) => {
            let num_bands = raster.bands().len();
            validate_band_index(band_index, num_bands)?;
            let band = raster.bands().band(band_index as usize)?;
            let band_meta = band.metadata();

            // If no nodata value defined, return false
            let nodata_bytes = match band_meta.nodata_value() {
                None => {
                    builder.append_value(false);
                    return Ok(());
                }
                Some(b) => b,
            };

            // OutDbRef bands don't have inline pixel data
            if band_meta.storage_type()? == StorageType::OutDbRef {
                return exec_err!("RS_BandIsNoData does not support out-db raster bands");
            }

            let dt = band_meta.data_type()?;
            let pixel_size = data_type_byte_size(&dt);
            let data = band.data();

            // Check every pixel against the nodata bytes
            let all_nodata = data
                .chunks_exact(pixel_size)
                .all(|chunk| chunk == nodata_bytes);

            builder.append_value(all_nodata);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BooleanArray, Float64Array, Int32Array, StringArray, StructArray};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::{BandMetadata, RasterMetadata};
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    /// Build a single-row raster StructArray with custom metadata and band metadata.
    fn build_custom_raster(
        meta: &RasterMetadata,
        band_meta: &BandMetadata,
        data: &[u8],
        crs: Option<&str>,
    ) -> StructArray {
        let mut builder = RasterBuilder::new(1);
        builder.start_raster(meta, crs).expect("start raster");
        builder
            .start_band(BandMetadata {
                datatype: band_meta.datatype,
                nodata_value: band_meta.nodata_value.clone(),
                storage_type: band_meta.storage_type,
                outdb_url: band_meta.outdb_url.clone(),
                outdb_band_id: band_meta.outdb_band_id,
            })
            .expect("start band");
        builder.band_data_writer().append_value(data);
        builder.finish_band().expect("finish band");
        builder.finish_raster().expect("finish raster");
        builder.finish().expect("finish")
    }

    // -----------------------------------------------------------------------
    // RS_BandPixelType tests
    // -----------------------------------------------------------------------

    #[test]
    fn udf_bandpixeltype_metadata() {
        let udf: ScalarUDF = rs_bandpixeltype_udf().into();
        assert_eq!(udf.name(), "rs_bandpixeltype");
    }

    #[test]
    fn udf_bandpixeltype_default_band() {
        let udf: ScalarUDF = rs_bandpixeltype_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        tester.assert_return_type(DataType::Utf8);

        // generate_test_rasters creates UInt16 bands
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let string_array = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected StringArray");

        assert_eq!(string_array.value(0), "UNSIGNED_16BITS");
        assert!(string_array.is_null(1));
        assert_eq!(string_array.value(2), "UNSIGNED_16BITS");
    }

    #[test]
    fn udf_bandpixeltype_with_band() {
        let udf: ScalarUDF = rs_bandpixeltype_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let band_indices = Int32Array::from(vec![1, 1, 1]);
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(band_indices)])
            .unwrap();
        let string_array = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected StringArray");

        assert_eq!(string_array.value(0), "UNSIGNED_16BITS");
        assert!(string_array.is_null(1));
        assert_eq!(string_array.value(2), "UNSIGNED_16BITS");
    }

    #[test]
    fn udf_bandpixeltype_invalid_band_errors() {
        let udf: ScalarUDF = rs_bandpixeltype_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let rasters = generate_test_rasters(1, None).unwrap();
        let band_indices = Int32Array::from(vec![5]); // out of range
        let result = tester.invoke_arrays(vec![Arc::new(rasters), Arc::new(band_indices)]);
        assert!(result.is_err());
    }

    #[test]
    fn udf_bandpixeltype_null_scalar() {
        let udf: ScalarUDF = rs_bandpixeltype_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));
    }

    // -----------------------------------------------------------------------
    // RS_BandNoDataValue tests
    // -----------------------------------------------------------------------

    #[test]
    fn udf_bandnodatavalue_metadata() {
        let udf: ScalarUDF = rs_bandnodatavalue_udf().into();
        assert_eq!(udf.name(), "rs_bandnodatavalue");
    }

    #[test]
    fn udf_bandnodatavalue_default_band() {
        let udf: ScalarUDF = rs_bandnodatavalue_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        tester.assert_return_type(DataType::Float64);

        // generate_test_rasters creates bands with nodata = [0, 0] (UInt16 = 0)
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(Float64Array::from(vec![Some(0.0), None, Some(0.0)]));
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn udf_bandnodatavalue_no_nodata() {
        // Create a raster without nodata
        let meta = RasterMetadata {
            width: 2,
            height: 2,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
        let band_meta = BandMetadata {
            datatype: BandDataType::UInt8,
            nodata_value: None,
            storage_type: StorageType::InDb,
            outdb_url: None,
            outdb_band_id: None,
        };
        let data = vec![1u8, 2, 3, 4];
        let rasters = build_custom_raster(&meta, &band_meta, &data, Some("OGC:CRS84"));

        let udf: ScalarUDF = rs_bandnodatavalue_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let float_array = result
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Expected Float64Array");
        assert!(float_array.is_null(0));
    }

    #[test]
    fn udf_bandnodatavalue_null_scalar() {
        let udf: ScalarUDF = rs_bandnodatavalue_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Float64(None));
    }

    // -----------------------------------------------------------------------
    // RS_BandIsNoData tests
    // -----------------------------------------------------------------------

    #[test]
    fn udf_bandisnodata_metadata() {
        let udf: ScalarUDF = rs_bandisnodata_udf().into();
        assert_eq!(udf.name(), "rs_bandisnodata");
    }

    #[test]
    fn udf_bandisnodata_not_all_nodata() {
        let udf: ScalarUDF = rs_bandisnodata_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        tester.assert_return_type(DataType::Boolean);

        // generate_test_rasters: pixel data is 0,1,2,... and nodata=0
        // So the first pixel matches nodata but subsequent don't -> false
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(BooleanArray::from(vec![Some(false), None, Some(false)]));
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn udf_bandisnodata_all_nodata() {
        // Create a raster where all pixels equal nodata
        let meta = RasterMetadata {
            width: 2,
            height: 2,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
        let nodata_bytes = 255u8.to_le_bytes().to_vec();
        let band_meta = BandMetadata {
            datatype: BandDataType::UInt8,
            nodata_value: Some(nodata_bytes),
            storage_type: StorageType::InDb,
            outdb_url: None,
            outdb_band_id: None,
        };
        // All pixels are 255 (same as nodata)
        let data = vec![255u8, 255, 255, 255];
        let rasters = build_custom_raster(&meta, &band_meta, &data, Some("OGC:CRS84"));

        let udf: ScalarUDF = rs_bandisnodata_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let bool_array = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Expected BooleanArray");
        assert!(bool_array.value(0));
    }

    #[test]
    fn udf_bandisnodata_no_nodata_defined() {
        // No nodata defined -> return false
        let meta = RasterMetadata {
            width: 2,
            height: 2,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
        let band_meta = BandMetadata {
            datatype: BandDataType::UInt8,
            nodata_value: None,
            storage_type: StorageType::InDb,
            outdb_url: None,
            outdb_band_id: None,
        };
        let data = vec![0u8, 0, 0, 0];
        let rasters = build_custom_raster(&meta, &band_meta, &data, Some("OGC:CRS84"));

        let udf: ScalarUDF = rs_bandisnodata_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let bool_array = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Expected BooleanArray");
        assert!(!bool_array.value(0));
    }

    #[test]
    fn udf_bandisnodata_null_scalar() {
        let udf: ScalarUDF = rs_bandisnodata_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Boolean(None));
    }
}
