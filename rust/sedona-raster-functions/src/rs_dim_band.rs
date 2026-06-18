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

use arrow_schema::DataType;
use datafusion_common::cast::as_string_view_array;
use datafusion_common::error::Result;
use datafusion_common::{arrow_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_datafusion_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::executor::RasterExecutor;
use crate::rs_ensure_loaded::{NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY};
use crate::rs_slice::{extract_slice, require_any_band_has_dim};

/// RS_DimToBand(raster, dim_name) -> Raster
///
/// Expands each band that has the named dimension into multiple bands
/// (one per index along that dimension), removing that dimension from each.
/// Bands that do not have the named dimension are passed through unchanged.
/// Spatial dimensions cannot be expanded.
pub fn rs_dimtoband_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_dimtoband",
        vec![Arc::new(RsDimToBand {})],
        Volatility::Immutable,
    )
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsDimToBand {}

impl SedonaScalarKernel for RsDimToBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_string()],
            SedonaType::Raster,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);

        let dim_name_array = args[1].clone().cast_to(&DataType::Utf8View, None)?;
        let dim_name_array = dim_name_array.into_array(executor.num_iterations())?;
        let dim_name_array = as_string_view_array(&dim_name_array)?;

        let mut new_builder = RasterBuilder::new(executor.num_iterations());
        let mut dim_name_iter = dim_name_array.iter();

        executor.execute_raster_void(|_i, raster_opt| {
            let dim_name = dim_name_iter.next().unwrap();

            match (raster_opt, dim_name) {
                (None, _) | (_, None) => {
                    new_builder.append_null()?;
                    Ok(())
                }
                (Some(raster), Some(name)) => {
                    if name == raster.x_dim() || name == raster.y_dim() {
                        return exec_err!("RS_DimToBand: cannot expand spatial dimension '{name}'");
                    }

                    require_any_band_has_dim(raster, name, "RS_DimToBand")?;

                    let t: [f64; 6] = raster.transform().try_into().map_err(|_| {
                        sedona_internal_datafusion_err!("raster transform is not 6 elements")
                    })?;
                    let spatial_dims = raster.spatial_dims();
                    new_builder.start_raster_nd(
                        &t,
                        &spatial_dims,
                        raster.spatial_shape(),
                        raster.crs(),
                    )?;

                    for band_idx in 0..raster.num_bands() {
                        let band = raster
                            .band(band_idx)
                            .map_err(|e| arrow_datafusion_err!(e))?;

                        let maybe_dim_idx = band.dim_index(name);
                        match maybe_dim_idx {
                            None => {
                                // Band doesn't have this dimension -- pass through
                                let dim_names = band.dim_names();
                                let band_name = raster.band_name(band_idx);
                                new_builder.start_band_nd(
                                    band_name,
                                    &dim_names,
                                    band.shape(),
                                    band.data_type(),
                                    band.nodata(),
                                    None,
                                    None,
                                )?;
                                let ndb = band.nd_buffer()?;
                                let data = ndb.as_contiguous()?;
                                new_builder.band_data_writer().append_value(data);
                                new_builder.finish_band()?;
                            }
                            Some(dim_idx) => {
                                let dim_size = band.shape()[dim_idx];
                                let new_dim_names: Vec<&str> = band
                                    .dim_names()
                                    .into_iter()
                                    .enumerate()
                                    .filter(|&(i, _)| i != dim_idx)
                                    .map(|(_, n)| n)
                                    .collect();
                                let new_shape: Vec<i64> = band
                                    .shape()
                                    .iter()
                                    .enumerate()
                                    .filter(|&(i, _)| i != dim_idx)
                                    .map(|(_, &s)| s)
                                    .collect();

                                let orig_band_name = raster.band_name(band_idx);

                                for idx in 0..dim_size {
                                    let sliced_data =
                                        extract_slice(band.as_ref(), dim_idx, idx, 1)?;

                                    let new_band_name =
                                        orig_band_name.map(|n| format!("{n}_{name}_{idx}"));
                                    new_builder.start_band_nd(
                                        new_band_name.as_deref(),
                                        &new_dim_names,
                                        &new_shape,
                                        band.data_type(),
                                        band.nodata(),
                                        None,
                                        None,
                                    )?;
                                    new_builder.band_data_writer().append_value(&sliced_data);
                                    new_builder.finish_band()?;
                                }
                            }
                        }
                    }

                    new_builder.finish_raster()?;
                    Ok(())
                }
            }
        })?;

        executor.finish(Arc::new(new_builder.finish()?))
    }
}

/// RS_BandToDim(raster, dim_name) -> Raster
///
/// Merges all bands into a single band by prepending a new dimension with
/// the given name. All bands must have identical dim_names, shape, and
/// data_type. The data from each band is concatenated in order.
pub fn rs_bandtodim_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_bandtodim",
        vec![Arc::new(RsBandToDim {})],
        Volatility::Immutable,
    )
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsBandToDim {}

impl SedonaScalarKernel for RsBandToDim {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_string()],
            SedonaType::Raster,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);

        let dim_name_array = args[1].clone().cast_to(&DataType::Utf8View, None)?;
        let dim_name_array = dim_name_array.into_array(executor.num_iterations())?;
        let dim_name_array = as_string_view_array(&dim_name_array)?;

        let mut new_builder = RasterBuilder::new(executor.num_iterations());
        let mut dim_name_iter = dim_name_array.iter();

        executor.execute_raster_void(|_i, raster_opt| {
            let dim_name = dim_name_iter.next().unwrap();

            match (raster_opt, dim_name) {
                (None, _) | (_, None) => {
                    new_builder.append_null()?;
                    Ok(())
                }
                (Some(raster), Some(name)) => {
                    let num_bands = raster.num_bands();
                    if num_bands == 0 {
                        return exec_err!("RS_BandToDim: raster has no bands");
                    }

                    let band0 = raster.band(0).map_err(|e| arrow_datafusion_err!(e))?;
                    let ref_dim_names = band0.dim_names();
                    let ref_shape = band0.shape().to_vec();
                    let ref_data_type = band0.data_type();
                    let ref_nodata: Option<Vec<u8>> = band0.nodata().map(|n| n.to_vec());

                    // The new dim name must not collide with an existing dim
                    // on the input bands; otherwise the output's dim_names
                    // would have duplicates and the round-trip through
                    // RS_DimToBand can't recover the original raster.
                    if ref_dim_names.contains(&name) {
                        return exec_err!(
                            "RS_BandToDim: dimension '{name}' already exists on the input bands; \
                             pick a different name"
                        );
                    }

                    for i in 1..num_bands {
                        let band = raster.band(i).map_err(|e| arrow_datafusion_err!(e))?;
                        if band.dim_names() != ref_dim_names {
                            return exec_err!(
                                "RS_BandToDim: band {i} has different dim_names than band 0"
                            );
                        }
                        if band.shape() != ref_shape.as_slice() {
                            return exec_err!(
                                "RS_BandToDim: band {i} has different shape than band 0"
                            );
                        }
                        if band.data_type() != ref_data_type {
                            return exec_err!(
                                "RS_BandToDim: band {i} has different data_type than band 0"
                            );
                        }
                        // nodata must agree too — two bands with different
                        // sentinel values can't collapse into one output band
                        // without losing information about which sentinel
                        // applies where.
                        let band_nodata: Option<Vec<u8>> = band.nodata().map(|n| n.to_vec());
                        if band_nodata != ref_nodata {
                            return exec_err!(
                                "RS_BandToDim: band {i} has different nodata value than band 0"
                            );
                        }
                    }

                    let mut new_dim_names: Vec<&str> = Vec::with_capacity(ref_dim_names.len() + 1);
                    new_dim_names.push(name);
                    new_dim_names.extend(ref_dim_names.iter());

                    let mut new_shape: Vec<i64> = Vec::with_capacity(ref_shape.len() + 1);
                    new_shape.push(num_bands as i64);
                    new_shape.extend_from_slice(&ref_shape);

                    let mut concat_data = Vec::new();
                    for i in 0..num_bands {
                        let band = raster.band(i).map_err(|e| arrow_datafusion_err!(e))?;
                        let ndb = band.nd_buffer()?;
                        let data = ndb.as_contiguous()?;
                        concat_data.extend_from_slice(data);
                    }

                    let nodata = ref_nodata.as_deref();

                    let t: [f64; 6] = raster.transform().try_into().map_err(|_| {
                        sedona_internal_datafusion_err!("raster transform is not 6 elements")
                    })?;
                    let spatial_dims = raster.spatial_dims();
                    new_builder.start_raster_nd(
                        &t,
                        &spatial_dims,
                        raster.spatial_shape(),
                        raster.crs(),
                    )?;
                    new_builder.start_band_nd(
                        None,
                        &new_dim_names,
                        &new_shape,
                        ref_data_type,
                        nodata,
                        None,
                        None,
                    )?;
                    new_builder.band_data_writer().append_value(&concat_data);
                    new_builder.finish_band()?;
                    new_builder.finish_raster()?;

                    Ok(())
                }
            }
        })?;

        executor.finish(Arc::new(new_builder.finish()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StructArray;
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::raster_spec::{assert_rasters_equal, RasterSpec};
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    /// Build a single-row 3D raster with 1 band, shape [time, y, x],
    /// and sequential UInt8 data.
    fn build_3d_raster_sequential(time: u64, height: u64, width: u64) -> StructArray {
        RasterSpec::nd(
            &["time", "y", "x"],
            &[time as i64, height as i64, width as i64],
        )
        .crs(None)
        .band(BandDataType::UInt8)
        .name("temp")
        .build()
    }

    /// Build a single-row 2D raster with N bands, each [y, x], with
    /// sequential data starting at `band_idx * y * x`.
    fn build_multi_band_2d(num_bands: usize, height: u64, width: u64) -> StructArray {
        let pixels = (height * width) as usize;
        let mut spec = RasterSpec::d2(width as i64, height as i64).crs(None);
        for b in 0..num_bands {
            let offset = b * pixels;
            let data: Vec<u8> = (offset..offset + pixels).map(|i| i as u8).collect();
            spec = spec.band_values(&data);
        }
        spec.build()
    }

    #[test]
    fn dimtoband_3d_to_bands() {
        let udf: ScalarUDF = rs_dimtoband_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        // 1 band with shape [time=3, y=2, x=2], sequential data 0..12
        let rasters = build_3d_raster_sequential(3, 2, 2);
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), "time")
            .unwrap();

        // Pixel values are asserted in the Python end-to-end tests; this checks
        // the band-name derivation, which needs a named input band.
        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::try_new(result_struct).unwrap();
        let raster = raster_array.get(0).unwrap();
        assert_eq!(raster.band_name(0), Some("temp_time_0"));
        assert_eq!(raster.band_name(1), Some("temp_time_1"));
        assert_eq!(raster.band_name(2), Some("temp_time_2"));
    }

    #[test]
    fn dimtoband_null_raster() {
        let udf: ScalarUDF = rs_dimtoband_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = generate_test_rasters(1, Some(0)).unwrap();
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), "time")
            .unwrap();

        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::try_new(result_struct).unwrap();
        assert!(raster_array.is_null(0));
    }

    #[test]
    fn bandtodim_bands_to_3d() {
        let udf: ScalarUDF = rs_bandtodim_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        // 3 bands, each [y=2, x=2], sequential data
        let rasters = build_multi_band_2d(3, 2, 2);
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), "newdim")
            .unwrap();

        // 1 band [newdim=3, y=2, x=2] concatenating the 3 input bands' data.
        let expected = RasterSpec::nd(&["newdim", "y", "x"], &[3, 2, 2])
            .crs(None)
            .band_values(&(0u8..12).collect::<Vec<u8>>());
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn bandtodim_mismatched_shapes_error() {
        // Build a raster whose bands disagree along a non-spatial axis.
        // Both bands satisfy the raster's spatial extent (y=2, x=2) so the
        // builder accepts them, but their time-axis sizes differ — that is
        // what RS_BandToDim must reject.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[2, 2], None)
            .unwrap();

        builder
            .start_band_nd(
                None,
                &["time", "y", "x"],
                &[3, 2, 2],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value([0u8; 12]);
        builder.finish_band().unwrap();

        builder
            .start_band_nd(
                None,
                &["time", "y", "x"],
                &[4, 2, 2],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value([0u8; 16]);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        let rasters = builder.finish().unwrap();

        let udf: ScalarUDF = rs_bandtodim_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        // Use a new dim name ("batch") that doesn't collide with the bands'
        // existing dim names so the dim-collision check passes and we reach
        // the shape-mismatch check.
        let result = tester.invoke_array_scalar(Arc::new(rasters), "batch");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("different shape"),
            "Unexpected error: {err_msg}"
        );
    }

    #[test]
    fn bandtodim_mismatched_nodata_error() {
        // Two bands match on dim_names/shape/data_type but disagree on
        // nodata sentinel. RS_BandToDim must reject — collapsing them into
        // one output band would silently inherit band 0's nodata and lose
        // information about band 1's sentinel.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[2, 2], None)
            .unwrap();
        builder
            .start_band_nd(
                None,
                &["y", "x"],
                &[2, 2],
                BandDataType::UInt8,
                Some(&[0u8]),
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value([0u8; 4]);
        builder.finish_band().unwrap();
        builder
            .start_band_nd(
                None,
                &["y", "x"],
                &[2, 2],
                BandDataType::UInt8,
                Some(&[255u8]),
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value([0u8; 4]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let rasters = builder.finish().unwrap();

        let udf: ScalarUDF = rs_bandtodim_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);
        let result = tester.invoke_array_scalar(Arc::new(rasters), "stack");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("different nodata"), "Unexpected error: {err}");
    }

    #[test]
    fn bandtodim_new_dim_collides_with_existing_dim_error() {
        // RS_BandToDim with a new dim name that already exists on the input
        // bands would build dim_names with duplicates (e.g. ["y", "y", "x"])
        // and break the RS_DimToBand round-trip. Must reject up front.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[2, 2], None)
            .unwrap();
        for _ in 0..2 {
            builder
                .start_band_nd(
                    None,
                    &["y", "x"],
                    &[2, 2],
                    BandDataType::UInt8,
                    None,
                    None,
                    None,
                )
                .unwrap();
            builder.band_data_writer().append_value([0u8; 4]);
            builder.finish_band().unwrap();
        }
        builder.finish_raster().unwrap();
        let rasters = builder.finish().unwrap();

        let udf: ScalarUDF = rs_bandtodim_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);
        // Try to create a new dim named "y" — collides with the existing y axis.
        let result = tester.invoke_array_scalar(Arc::new(rasters), "y");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("dimension 'y' already exists"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn bandtodim_null_raster() {
        let udf: ScalarUDF = rs_bandtodim_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = generate_test_rasters(1, Some(0)).unwrap();
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), "time")
            .unwrap();

        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::try_new(result_struct).unwrap();
        assert!(raster_array.is_null(0));
    }
}
