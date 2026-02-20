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
use arrow_array::builder::{Float64Builder, StringBuilder};
use arrow_array::{cast::AsArray, types::Int32Type};
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

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
        let mut band_index_iter = band_index_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let band_index = band_index_iter.next().unwrap().unwrap_or(1);
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
            if band_index < 1 || band_index > num_bands as i32 {
                builder.append_null();
                return Ok(());
            }
            let band = raster.bands().band(band_index as usize)?;
            let dt = band.metadata().data_type()?;
            builder.append_value(dt.pixel_type_name());
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
        let mut band_index_iter = band_index_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let band_index = band_index_iter.next().unwrap().unwrap_or(1);
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
            if band_index < 1 || band_index > num_bands as i32 {
                builder.append_null();
                return Ok(());
            }
            let band = raster.bands().band(band_index as usize)?;
            let band_meta = band.metadata();
            match band_meta.nodata_value_as_f64()? {
                None => builder.append_null(),
                Some(val) => builder.append_value(val),
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Float64Array, Int32Array, StringArray, StructArray};
    use datafusion_expr::ScalarUDF;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::{BandMetadata, RasterMetadata};
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::{BandDataType, StorageType};
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
    fn udf_bandpixeltype_non_existing_band() {
        let udf: ScalarUDF = rs_bandpixeltype_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let rasters = generate_test_rasters(1, None).unwrap();
        let band_indices = Int32Array::from(vec![5]); // out of range
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(band_indices)])
            .unwrap();
        assert!(result.is_null(0));
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
    fn udf_bandnodatavalue_non_existing_band() {
        let udf: ScalarUDF = rs_bandnodatavalue_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);
        tester.assert_return_type(DataType::Float64);

        // generate_test_rasters creates bands with nodata = [0, 0] (UInt16 = 0)
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let bands = Int32Array::from(vec![0, 1, 2]); // out of range band index
        let result = tester
            .invoke_array_array(Arc::new(rasters), Arc::new(bands))
            .unwrap();
        assert!(result.is_null(0));
        assert!(result.is_null(1));
        assert!(result.is_null(2));
    }
}
