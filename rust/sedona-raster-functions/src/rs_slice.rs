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
use datafusion_common::exec_err;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandRef, RasterRef};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::executor::RasterExecutor;

// ===========================================================================
// RS_Slice
// ===========================================================================

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
                    let idx = idx as u64;
                    validate_not_spatial(raster, name, "RS_Slice")?;

                    let t: [f64; 6] = raster.transform().try_into().unwrap();
                    new_builder.start_raster(&t, raster.x_dim(), raster.y_dim(), raster.crs())?;

                    for band_idx in 0..raster.num_bands() {
                        let band = raster.band(band_idx).ok_or_else(|| {
                            datafusion_common::DataFusionError::Internal(format!(
                                "RS_Slice: band {band_idx} not found"
                            ))
                        })?;

                        let dim_idx = band.dim_index(name).ok_or_else(|| {
                            datafusion_common::DataFusionError::Execution(format!(
                                "RS_Slice: dimension '{name}' not found in band {band_idx}"
                            ))
                        })?;

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
                        let new_shape: Vec<u64> = shape
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| i != dim_idx)
                            .map(|(_, &s)| s)
                            .collect();

                        let sliced_data =
                            extract_slice(band.as_ref(), dim_idx, idx, 1)?;

                        let band_name = raster.band_name(band_idx);
                        let new_dim_name_refs: Vec<&str> =
                            new_dim_names.to_vec();
                        new_builder.start_band(
                            band_name,
                            &new_dim_name_refs,
                            &new_shape,
                            band.data_type(),
                            band.nodata(),
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

// ===========================================================================
// RS_SliceRange
// ===========================================================================

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
                    let start_val = start_val as u64;
                    let end_val = end_val as u64;
                    validate_not_spatial(raster, name, "RS_SliceRange")?;

                    if start_val >= end_val {
                        return exec_err!(
                            "RS_SliceRange: start ({start_val}) must be less than end ({end_val})"
                        );
                    }

                    let t: [f64; 6] = raster.transform().try_into().unwrap();
                    new_builder.start_raster(&t, raster.x_dim(), raster.y_dim(), raster.crs())?;

                    for band_idx in 0..raster.num_bands() {
                        let band = raster.band(band_idx).ok_or_else(|| {
                            datafusion_common::DataFusionError::Internal(format!(
                                "RS_SliceRange: band {band_idx} not found"
                            ))
                        })?;

                        let dim_idx = band.dim_index(name).ok_or_else(|| {
                            datafusion_common::DataFusionError::Execution(format!(
                                "RS_SliceRange: dimension '{name}' not found in band {band_idx}"
                            ))
                        })?;

                        let shape = band.shape();
                        if end_val > shape[dim_idx] {
                            return exec_err!(
                                "RS_SliceRange: end ({end_val}) out of range for dimension '{name}' with size {}",
                                shape[dim_idx]
                            );
                        }

                        let range_len = end_val - start_val;
                        let dim_names = band.dim_names();
                        let dim_name_refs: Vec<&str> = dim_names.to_vec();
                        let mut new_shape: Vec<u64> = shape.to_vec();
                        new_shape[dim_idx] = range_len;

                        let sliced_data =
                            extract_slice(band.as_ref(), dim_idx, start_val, range_len)?;

                        let band_name = raster.band_name(band_idx);
                        new_builder.start_band(
                            band_name,
                            &dim_name_refs,
                            &new_shape,
                            band.data_type(),
                            band.nodata(),
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

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Validate that the dimension name is not a spatial dimension.
fn validate_not_spatial(raster: &dyn RasterRef, dim_name: &str, func_name: &str) -> Result<()> {
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
    start: u64,
    count: u64,
) -> Result<Vec<u8>> {
    let shape = band.shape();
    let elem_size = band.data_type().byte_size() as u64;
    let data = band.contiguous_data()?;

    let outer_count: u64 = shape[..dim_idx].iter().product();
    let inner_size: u64 = shape[dim_idx + 1..].iter().product::<u64>() * elem_size;
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StructArray;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    /// Build a single-row 3D raster with shape [time, height, width] and
    /// sequential UInt8 data so we can verify slicing correctness.
    fn build_3d_raster_sequential(time: u64, height: u64, width: u64) -> StructArray {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, "x", "y", None).unwrap();
        builder
            .start_band(
                None,
                &["time", "y", "x"],
                &[time, height, width],
                BandDataType::UInt8,
                None,
                None,
            )
            .unwrap();
        let total = (time * height * width) as usize;
        let data: Vec<u8> = (0..total).map(|i| i as u8).collect();
        builder.band_data_writer().append_value(&data);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    #[test]
    fn slice_3d_on_time() {
        let udf: ScalarUDF = rs_slice_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
            ],
        );

        // shape [time=3, y=4, x=5], sequential data 0..60
        let rasters = build_3d_raster_sequential(3, 4, 5);
        let result = tester
            .invoke_array_scalar_scalar(Arc::new(rasters), "time", 1_i64)
            .unwrap();

        let result_struct = result.as_any().downcast_ref::<StructArray>().unwrap();
        let raster_array = RasterStructArray::new(result_struct);
        let raster = raster_array.get(0).unwrap();

        // Should now be 2D: [y=4, x=5]
        let band = raster.band(0).unwrap();
        assert_eq!(band.ndim(), 2);
        assert_eq!(band.dim_names(), vec!["y", "x"]);
        assert_eq!(band.shape(), &[4, 5]);

        // Data should be time slice 1: bytes 20..40 of original
        let data = band.contiguous_data().unwrap();
        let expected: Vec<u8> = (20..40).collect();
        assert_eq!(data.as_ref(), &expected[..]);
    }

    #[test]
    fn slicerange_3d_on_time() {
        let kernel = RsSliceRange {};
        let arg_types = vec![
            RASTER,
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Int64),
            SedonaType::Arrow(DataType::Int64),
        ];

        // shape [time=3, y=4, x=5], sequential data 0..60
        let rasters = build_3d_raster_sequential(3, 4, 5);
        let args = vec![
            ColumnarValue::Array(Arc::new(rasters)),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("time".to_string()))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(0))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ];
        let result = kernel.invoke_batch(&arg_types, &args).unwrap();

        let result_struct = match result {
            ColumnarValue::Array(arr) => arr,
            _ => panic!("Expected array result"),
        };
        let result_struct = result_struct
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let raster_array = RasterStructArray::new(result_struct);
        let raster = raster_array.get(0).unwrap();

        // Should still be 3D: [time=2, y=4, x=5]
        let band = raster.band(0).unwrap();
        assert_eq!(band.ndim(), 3);
        assert_eq!(band.dim_names(), vec!["time", "y", "x"]);
        assert_eq!(band.shape(), &[2, 4, 5]);

        // Data should be first 2 time slices: bytes 0..40
        let data = band.contiguous_data().unwrap();
        let expected: Vec<u8> = (0..40).collect();
        assert_eq!(data.as_ref(), &expected[..]);
    }

    #[test]
    fn slice_spatial_dim_error() {
        let udf: ScalarUDF = rs_slice_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
            ],
        );

        let rasters = build_3d_raster_sequential(3, 4, 5);

        // Try to slice "x"
        let result = tester.invoke_array_scalar_scalar(Arc::new(rasters.clone()), "x", 0_i64);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot slice spatial dimension"),
            "Unexpected error: {err_msg}"
        );

        // Try to slice "y"
        let result = tester.invoke_array_scalar_scalar(Arc::new(rasters), "y", 0_i64);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("cannot slice spatial dimension"),
            "Unexpected error: {err_msg}"
        );
    }

    #[test]
    fn slice_index_out_of_range() {
        let udf: ScalarUDF = rs_slice_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
            ],
        );

        let rasters = build_3d_raster_sequential(3, 4, 5);
        let result = tester.invoke_array_scalar_scalar(Arc::new(rasters), "time", 3_i64);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("out of range"),
            "Unexpected error: {err_msg}"
        );
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
        let raster_array = RasterStructArray::new(result_struct);
        assert!(raster_array.is_null(0));
    }
}
