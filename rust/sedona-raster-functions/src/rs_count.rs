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
use arrow_array::builder::Int64Builder;
use arrow_array::{cast::AsArray, types::Int32Type, Array, BooleanArray};
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_common::exec_err;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::raster::{BandDataType, StorageType};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// Byte size of a single pixel for the given band data type.
fn data_type_byte_size(data_type: &BandDataType) -> usize {
    match data_type {
        BandDataType::UInt8 | BandDataType::Int8 => 1,
        BandDataType::UInt16 | BandDataType::Int16 => 2,
        BandDataType::UInt32 | BandDataType::Int32 | BandDataType::Float32 => 4,
        BandDataType::UInt64 | BandDataType::Int64 | BandDataType::Float64 => 8,
    }
}

/// RS_Count() scalar UDF implementation
///
/// Returns the count of pixels in the specified band.
/// When excludeNoData is true (default), pixels equal to the nodata value are excluded.
/// When excludeNoData is false, returns width * height.
/// Accepts optional band_index (1-based, default 1) and excludeNoData (default true) parameters.
pub fn rs_count_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_count",
        vec![
            Arc::new(RsCount {}),
            Arc::new(RsCountWithBand {}),
            Arc::new(RsCountWithBandAndExclude {}),
        ],
        Volatility::Immutable,
    )
}

// ---------------------------------------------------------------------------
// 1-arg kernel: RS_Count(raster)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RsCount {}

impl SedonaScalarKernel for RsCount {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Int64),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = Int64Builder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|_i, raster_opt| {
            count_pixels(raster_opt, 1, true, &mut builder)
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

// ---------------------------------------------------------------------------
// 2-arg kernel: RS_Count(raster, band)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RsCountWithBand {}

impl SedonaScalarKernel for RsCountWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            SedonaType::Arrow(DataType::Int64),
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

        let mut builder = Int64Builder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|i, raster_opt| {
            let band_index = if band_index_array.is_null(i) {
                1
            } else {
                band_index_array.value(i)
            };
            count_pixels(raster_opt, band_index, true, &mut builder)
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

// ---------------------------------------------------------------------------
// 3-arg kernel: RS_Count(raster, band, excludeNoData)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RsCountWithBandAndExclude {}

impl SedonaScalarKernel for RsCountWithBandAndExclude {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_boolean(),
            ],
            SedonaType::Arrow(DataType::Int64),
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
        let exclude_array = args[2].clone().into_array(executor.num_iterations())?;
        let exclude_array = exclude_array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("Expected BooleanArray for excludeNoData");

        let mut builder = Int64Builder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|i, raster_opt| {
            let band_index = if band_index_array.is_null(i) {
                1
            } else {
                band_index_array.value(i)
            };
            let exclude_nodata = if exclude_array.is_null(i) {
                true
            } else {
                exclude_array.value(i)
            };
            count_pixels(raster_opt, band_index, exclude_nodata, &mut builder)
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn count_pixels(
    raster_opt: Option<&sedona_raster::array::RasterRefImpl<'_>>,
    band_index: i32,
    exclude_nodata: bool,
    builder: &mut Int64Builder,
) -> Result<()> {
    match raster_opt {
        None => {
            builder.append_null();
            Ok(())
        }
        Some(raster) => {
            let num_bands = raster.bands().len();
            if band_index < 1 || band_index as usize > num_bands {
                return exec_err!(
                    "Provided band index {} is not in the range [1, {}]",
                    band_index,
                    num_bands
                );
            }

            let total_pixels = raster.metadata().width() as i64 * raster.metadata().height() as i64;

            let band = raster.bands().band(band_index as usize)?;
            let band_meta = band.metadata();
            let nodata_bytes = band_meta.nodata_value();

            // If not excluding nodata, or no nodata value defined, return total pixel count
            if !exclude_nodata || nodata_bytes.is_none() {
                builder.append_value(total_pixels);
                return Ok(());
            }

            let nodata_bytes = nodata_bytes.unwrap();

            // OutDbRef bands don't have inline pixel data
            if band_meta.storage_type()? == StorageType::OutDbRef {
                return exec_err!(
                    "RS_Count with excludeNoData=true does not support out-db raster bands"
                );
            }

            let dt = band_meta.data_type()?;
            let pixel_size = data_type_byte_size(&dt);
            let data = band.data();

            let nodata_count = data
                .chunks_exact(pixel_size)
                .filter(|chunk| *chunk == nodata_bytes)
                .count() as i64;

            builder.append_value(total_pixels - nodata_count);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BooleanArray, Int32Array, Int64Array};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    /// Build a single-row raster with one band and no nodata value.
    fn build_no_nodata_raster(
        width: usize,
        height: usize,
        data_type: BandDataType,
        band_bytes: &[u8],
        crs: Option<&str>,
    ) -> arrow_array::StructArray {
        use sedona_raster::builder::RasterBuilder;
        use sedona_raster::traits::{BandMetadata, RasterMetadata};
        let mut builder = RasterBuilder::new(1);
        let metadata = RasterMetadata {
            width: width as u64,
            height: height as u64,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
        builder.start_raster(&metadata, crs).expect("start raster");
        builder
            .start_band(BandMetadata {
                datatype: data_type,
                nodata_value: None,
                storage_type: StorageType::InDb,
                outdb_url: None,
                outdb_band_id: None,
            })
            .expect("start band");
        builder.band_data_writer().append_value(band_bytes);
        builder.finish_band().expect("finish band");
        builder.finish_raster().expect("finish raster");
        builder.finish().expect("finish")
    }

    #[test]
    fn udf_count_metadata() {
        let udf: ScalarUDF = rs_count_udf().into();
        assert_eq!(udf.name(), "rs_count");
    }

    #[test]
    fn udf_count_default() {
        let udf: ScalarUDF = rs_count_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        tester.assert_return_type(DataType::Int64);

        // generate_test_rasters: raster 0 is 1x2=2 pixels, data=[0,0,1,0] (u16)
        // nodata=0 -> pixel 0 is nodata, pixel 1 is not -> count=1
        // raster 1: null
        // raster 2: 3x4=12 pixels, data=[0..11] as u16
        // nodata=0 -> pixel 0 is nodata -> count=11
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(Int64Array::from(vec![Some(1), None, Some(11)]));
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn udf_count_exclude_false() {
        let udf: ScalarUDF = rs_count_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Boolean),
            ],
        );

        // With excludeNoData=false, should return total pixel count
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let bands = Int32Array::from(vec![1, 1, 1]);
        let exclude = BooleanArray::from(vec![false, false, false]);
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(Int64Array::from(vec![Some(2), None, Some(12)]));

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(bands), Arc::new(exclude)])
            .unwrap();
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn udf_count_no_nodata_defined() {
        // When no nodata is defined, excludeNoData=true should still return total pixels
        let data = vec![0u8; 9];
        let rasters = build_no_nodata_raster(3, 3, BandDataType::UInt8, &data, Some("OGC:CRS84"));

        let udf: ScalarUDF = rs_count_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let int_array = result
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Expected Int64Array");
        assert_eq!(int_array.value(0), 9);
    }

    #[test]
    fn udf_count_null_scalar() {
        let udf: ScalarUDF = rs_count_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Int64(None));
    }

    #[test]
    fn udf_count_invalid_band_errors() {
        let udf: ScalarUDF = rs_count_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let rasters = generate_test_rasters(1, None).unwrap();
        let bands = Int32Array::from(vec![5]); // out of range
        let result = tester.invoke_arrays(vec![Arc::new(rasters), Arc::new(bands)]);
        assert!(result.is_err());
    }
}
