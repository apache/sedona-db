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
use datafusion_common::cast::{as_int64_array, as_string_array};
use datafusion_common::error::Result;
use datafusion_common::{arrow_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_datafusion_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandRef, RasterRef};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::executor::RasterExecutor;
use crate::rs_ensure_loaded::{NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY};

/// RS_Slice(raster, dim_name, index) -> Raster
///
/// Slices each band along the named dimension at the given index, removing
/// that dimension from the output. Spatial dimensions cannot be sliced.
pub fn rs_slice_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_slice",
        vec![Arc::new(RsSlice {})],
        Volatility::Immutable,
    )
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsSlice {}

impl SedonaScalarKernel for RsSlice {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_integer(),
            ],
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

        let dim_name_array = args[1].clone().cast_to(&DataType::Utf8, None)?;
        let dim_name_array = dim_name_array.into_array(executor.num_iterations())?;
        let dim_name_array = as_string_array(&dim_name_array)?;

        let index_array = args[2].clone().cast_to(&DataType::Int64, None)?;
        let index_array = index_array.into_array(executor.num_iterations())?;
        let index_array = as_int64_array(&index_array)?;

        let mut new_builder = RasterBuilder::new(executor.num_iterations());
        let mut dim_name_iter = dim_name_array.iter();
        let mut index_iter = index_array.iter();

        executor.execute_raster_void(|_i, raster_opt| {
            let dim_name = dim_name_iter.next().unwrap();
            let index = index_iter.next().unwrap();

            match (raster_opt, dim_name, index) {
                (None, _, _) | (_, None, _) | (_, _, None) => {
                    new_builder.append_null()?;
                    Ok(())
                }
                (Some(raster), Some(name), Some(idx)) => {
                    if idx < 0 {
                        return exec_err!(
                            "RS_Slice: index must be non-negative, got {idx}"
                        );
                    }
                    validate_not_spatial(raster, name, "RS_Slice")?;

                    let t: [f64; 6] = raster.transform().try_into().map_err(|_| {
                        sedona_internal_datafusion_err!("raster transform is not 6 elements")
                    })?;
                    let spatial_dims = raster.spatial_dims();
                    new_builder.start_raster_nd(&t, &spatial_dims, raster.spatial_shape(), raster.crs())?;

                    require_any_band_has_dim(raster, name, "RS_Slice")?;

                    for band_idx in 0..raster.num_bands() {
                        let band = raster
                            .band(band_idx)
                            .map_err(|e| arrow_datafusion_err!(e))?;

                        // Pass-through: bands that don't carry the named
                        // dimension are emitted unchanged. Same convention as
                        // RS_DimToBand, and matches xarray's `isel` — variables
                        // without the indexed dim are left alone.
                        let Some(dim_idx) = band.dim_index(name) else {
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
                            continue;
                        };

                        let shape = band.shape();
                        if idx >= shape[dim_idx] {
                            return exec_err!(
                                "RS_Slice: index {idx} out of range for dimension '{name}' with size {}",
                                shape[dim_idx]
                            );
                        }

                        let new_dim_names: Vec<&str> = band
                            .dim_names()
                            .into_iter()
                            .enumerate()
                            .filter(|&(i, _)| i != dim_idx)
                            .map(|(_, n)| n)
                            .collect();
                        let new_shape: Vec<i64> = shape
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| i != dim_idx)
                            .map(|(_, &s)| s)
                            .collect();

                        let sliced_data =
                            extract_slice(band.as_ref(), dim_idx, idx, 1)?;

                        let band_name = raster.band_name(band_idx);
                        new_builder.start_band_nd(
                            band_name,
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

                    new_builder.finish_raster()?;
                    Ok(())
                }
            }
        })?;

        executor.finish(Arc::new(new_builder.finish()?))
    }
}

/// RS_SliceRange(raster, dim_name, start, end) -> Raster
///
/// Narrows each band along the named dimension to the half-open range
/// `[start, end)`, keeping the dimension in the output with reduced size.
pub fn rs_slicerange_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_slicerange",
        vec![Arc::new(RsSliceRange {})],
        Volatility::Immutable,
    )
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsSliceRange {}

impl SedonaScalarKernel for RsSliceRange {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
            ],
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

        let dim_name_array = args[1].clone().cast_to(&DataType::Utf8, None)?;
        let dim_name_array = dim_name_array.into_array(executor.num_iterations())?;
        let dim_name_array = as_string_array(&dim_name_array)?;

        let start_array = args[2].clone().cast_to(&DataType::Int64, None)?;
        let start_array = start_array.into_array(executor.num_iterations())?;
        let start_array = as_int64_array(&start_array)?;

        let end_array = args[3].clone().cast_to(&DataType::Int64, None)?;
        let end_array = end_array.into_array(executor.num_iterations())?;
        let end_array = as_int64_array(&end_array)?;

        let mut new_builder = RasterBuilder::new(executor.num_iterations());
        let mut dim_name_iter = dim_name_array.iter();
        let mut start_iter = start_array.iter();
        let mut end_iter = end_array.iter();

        executor.execute_raster_void(|_i, raster_opt| {
            let dim_name = dim_name_iter.next().unwrap();
            let start = start_iter.next().unwrap();
            let end = end_iter.next().unwrap();

            match (raster_opt, dim_name, start, end) {
                (None, _, _, _) | (_, None, _, _) | (_, _, None, _) | (_, _, _, None) => {
                    new_builder.append_null()?;
                    Ok(())
                }
                (Some(raster), Some(name), Some(start_val), Some(end_val)) => {
                    if start_val < 0 {
                        return exec_err!(
                            "RS_SliceRange: start must be non-negative, got {start_val}"
                        );
                    }
                    if end_val < 0 {
                        return exec_err!(
                            "RS_SliceRange: end must be non-negative, got {end_val}"
                        );
                    }
                    validate_not_spatial(raster, name, "RS_SliceRange")?;

                    if start_val >= end_val {
                        return exec_err!(
                            "RS_SliceRange: start ({start_val}) must be less than end ({end_val})"
                        );
                    }

                    let t: [f64; 6] = raster.transform().try_into().map_err(|_| {
                        sedona_internal_datafusion_err!("raster transform is not 6 elements")
                    })?;
                    let spatial_dims = raster.spatial_dims();
                    new_builder.start_raster_nd(&t, &spatial_dims, raster.spatial_shape(), raster.crs())?;

                    require_any_band_has_dim(raster, name, "RS_SliceRange")?;

                    for band_idx in 0..raster.num_bands() {
                        let band = raster
                            .band(band_idx)
                            .map_err(|e| arrow_datafusion_err!(e))?;

                        // Pass-through: bands that don't carry the named
                        // dimension are emitted unchanged. Same convention as
                        // RS_Slice and RS_DimToBand.
                        let Some(dim_idx) = band.dim_index(name) else {
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
                            continue;
                        };

                        let shape = band.shape();
                        if end_val > shape[dim_idx] {
                            return exec_err!(
                                "RS_SliceRange: end ({end_val}) out of range for dimension '{name}' with size {}",
                                shape[dim_idx]
                            );
                        }

                        let range_len = end_val - start_val;
                        let dim_names = band.dim_names();
                        let mut new_shape: Vec<i64> = shape.to_vec();
                        new_shape[dim_idx] = range_len;

                        let sliced_data =
                            extract_slice(band.as_ref(), dim_idx, start_val, range_len)?;

                        let band_name = raster.band_name(band_idx);
                        new_builder.start_band_nd(
                            band_name,
                            &dim_names,
                            &new_shape,
                            band.data_type(),
                            band.nodata(),
                            None,
                            None,
                        )?;
                        new_builder.band_data_writer().append_value(&sliced_data);
                        new_builder.finish_band()?;
                    }

                    new_builder.finish_raster()?;
                    Ok(())
                }
            }
        })?;

        executor.finish(Arc::new(new_builder.finish()?))
    }
}

/// Validate that the dimension name is not a spatial dimension.
/// Verify that at least one band in `raster` carries the named dimension.
///
/// Used as a pre-flight by the manipulation functions (`RS_Slice`,
/// `RS_SliceRange`, `RS_DimToBand`) so that a typo'd dimension name
/// surfaces as a clean error rather than silently passing every band
/// through unchanged.
pub(crate) fn require_any_band_has_dim(
    raster: &dyn RasterRef,
    name: &str,
    func_name: &str,
) -> Result<()> {
    let any_band_has_dim = (0..raster.num_bands()).any(|i| {
        raster
            .band(i)
            .ok()
            .and_then(|b| b.dim_index(name))
            .is_some()
    });
    if !any_band_has_dim {
        return exec_err!("{func_name}: no band has dimension '{name}'");
    }
    Ok(())
}

pub(crate) fn validate_not_spatial(
    raster: &dyn RasterRef,
    dim_name: &str,
    func_name: &str,
) -> Result<()> {
    if dim_name == raster.x_dim() || dim_name == raster.y_dim() {
        return exec_err!("{func_name}: cannot slice spatial dimension '{dim_name}'");
    }
    Ok(())
}

/// Extract a slice of data from a band along a given dimension.
///
/// For `count == 1`, this extracts a single index (used by RS_Slice).
/// For `count > 1`, this extracts a contiguous range `[start, start+count)`
/// (used by RS_SliceRange).
///
/// The algorithm works on C-order (row-major) layout:
/// - `outer_count`: product of shape dimensions before `dim_idx`
/// - `inner_size`: product of shape dimensions after `dim_idx` * elem_size
/// - `stride`: `shape[dim_idx] * inner_size` (bytes between outer elements)
///
/// For each outer element, we copy `count * inner_size` bytes starting at
/// `start * inner_size` within that stride.
pub(crate) fn extract_slice(
    band: &dyn BandRef,
    dim_idx: usize,
    start: i64,
    count: i64,
) -> Result<Vec<u8>> {
    let shape = band.shape();
    let elem_size = band.data_type().byte_size() as i64;
    let ndb = band.nd_buffer()?;
    let data = ndb.as_contiguous()?;

    let outer_count: i64 = shape[..dim_idx].iter().product();
    let inner_size: i64 = shape[dim_idx + 1..].iter().product::<i64>() * elem_size;
    let stride = shape[dim_idx] * inner_size;
    let copy_size = (count * inner_size) as usize;
    let offset_within_stride = start * inner_size;

    let total_output = (outer_count as usize) * copy_size;
    let mut output = Vec::with_capacity(total_output);

    for outer in 0..outer_count {
        let src_start = (outer * stride + offset_within_stride) as usize;
        output.extend_from_slice(&data[src_start..src_start + copy_size]);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StructArray;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::raster_spec::{assert_rasters_equal, RasterSpec};
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    /// Build a single-row raster with two bands of different
    /// dimensionality, used to exercise pass-through behavior.
    /// Band 0: 2D `[y=2, x=3]`, UInt8 sequential 0..6.
    /// Band 1: 3D `[time=3, y=2, x=3]`, UInt8 sequential 0..18.
    fn build_mixed_dim_raster() -> StructArray {
        RasterSpec::nd(&["time", "y", "x"], &[3, 2, 3])
            .crs(None)
            .band_nd(&["y", "x"], &[2, 3], BandDataType::UInt8)
            .band(BandDataType::UInt8)
            .build()
    }

    #[test]
    fn slice_passes_through_bands_without_dim() {
        // RS_Slice on a heterogeneous raster: bands that don't carry the
        // named dim are emitted unchanged; bands that do are sliced.
        // Matches xarray's `ds.isel({dim: i})`.
        let udf: ScalarUDF = rs_slice_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
            ],
        );

        let rasters = build_mixed_dim_raster();
        let result = tester
            .invoke_array_scalar_scalar(Arc::new(rasters), "time", 1_i64)
            .unwrap();

        // Band 0 (no `time`) passes through unchanged: 2-D [2, 3], bytes 0..6.
        // Band 1 sliced to time=1: 2-D [2, 3], bytes 6..12.
        let expected = RasterSpec::nd(&["time", "y", "x"], &[3, 2, 3])
            .crs(None)
            .band_values_nd(&["y", "x"], &[2, 3], &(0u8..6).collect::<Vec<u8>>())
            .band_values_nd(&["y", "x"], &[2, 3], &(6u8..12).collect::<Vec<u8>>());
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn slicerange_passes_through_bands_without_dim() {
        // Same pass-through semantics for RS_SliceRange.
        let kernel = RsSliceRange {};
        let arg_types = vec![
            RASTER,
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Int64),
            SedonaType::Arrow(DataType::Int64),
        ];
        let rasters = build_mixed_dim_raster();
        let args = vec![
            ColumnarValue::Array(Arc::new(rasters)),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("time".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(1))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(3))),
        ];
        let result = kernel.invoke_batch(&arg_types, &args).unwrap();
        let result = match result {
            ColumnarValue::Array(arr) => arr,
            _ => panic!("Expected array result"),
        };

        // Band 0 (no `time`) passes through unchanged: 2-D [2, 3], bytes 0..6.
        // Band 1 narrowed to [1, 3): [time=2, y=2, x=3], bytes 6..18.
        let expected = RasterSpec::nd(&["time", "y", "x"], &[3, 2, 3])
            .crs(None)
            .band_values_nd(&["y", "x"], &[2, 3], &(0u8..6).collect::<Vec<u8>>())
            .band_values_nd(
                &["time", "y", "x"],
                &[2, 2, 3],
                &(6u8..18).collect::<Vec<u8>>(),
            );
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn slice_out_of_bounds_on_band_that_has_dim_in_heterogeneous_raster() {
        // Heterogeneous raster: pre-flight passes (band 1 has `time`), but
        // the index is out of range for band 1. The error must still fire
        // — pass-through doesn't swallow bounds errors.
        let udf: ScalarUDF = rs_slice_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
            ],
        );
        // build_mixed_dim_raster: band 0 has no time, band 1 has time=3.
        let rasters = build_mixed_dim_raster();
        let result = tester.invoke_array_scalar_scalar(Arc::new(rasters), "time", 5_i64);
        let err = result.unwrap_err().to_string();
        assert!(err.contains("out of range"), "Unexpected error: {err}");
    }

    #[test]
    fn slice_null_raster() {
        let udf: ScalarUDF = rs_slice_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
            ],
        );

        let rasters = generate_test_rasters(1, Some(0)).unwrap();
        let result = tester
            .invoke_array_scalar_scalar(Arc::new(rasters), "time", 0_i64)
            .unwrap();

        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::try_new(result_struct).unwrap();
        assert!(raster_array.is_null(0));
    }
}
