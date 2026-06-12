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

use arrow_array::builder::{Int32Builder, Int64Builder, ListBuilder, StringBuilder};
use arrow_schema::DataType;
use datafusion_common::cast::{as_int32_array, as_string_array};
use datafusion_common::error::Result;
use datafusion_common::{arrow_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::executor::RasterExecutor;

/// Check that all bands agree on a value. Returns the value from band 0,
/// or an error if any band disagrees.
fn check_band_agreement<T: PartialEq + std::fmt::Debug>(
    raster: &dyn RasterRef,
    func_name: &str,
    property_name: &str,
    extractor: impl Fn(&dyn sedona_raster::traits::BandRef) -> T,
) -> Result<T> {
    if raster.num_bands() == 0 {
        return exec_err!("{func_name}: raster has no bands");
    }
    let band0 = raster.band(0).map_err(|e| arrow_datafusion_err!(e))?;
    let value = extractor(band0.as_ref());
    for i in 1..raster.num_bands() {
        if let Ok(band) = raster.band(i) {
            let other = extractor(band.as_ref());
            if other != value {
                return exec_err!(
                    "{func_name}: bands have different {property_name} — specify a band index"
                );
            }
        }
    }
    Ok(value)
}

fn list_utf8_type() -> DataType {
    DataType::List(Arc::new(arrow_schema::Field::new(
        "item",
        DataType::Utf8,
        true,
    )))
}

fn list_int64_type() -> DataType {
    DataType::List(Arc::new(arrow_schema::Field::new(
        "item",
        DataType::Int64,
        true,
    )))
}

/// RS_NumDimensions(raster [, band]) -> Int32
///
/// Returns the number of dimensions in the raster (or a specific band).
pub fn rs_numdimensions_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_numdimensions",
        vec![
            Arc::new(RsNumDimensions {}),
            Arc::new(RsNumDimensionsWithBand {}),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsNumDimensions {}

impl SedonaScalarKernel for RsNumDimensions {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Int32),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = Int32Builder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|_i, raster_opt| match raster_opt {
            None => {
                builder.append_null();
                Ok(())
            }
            Some(raster) => {
                let ndim =
                    check_band_agreement(raster, "RS_NumDimensions", "dimensionality", |b| {
                        b.ndim()
                    })?;
                builder.append_value(ndim as i32);
                Ok(())
            }
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct RsNumDimensionsWithBand {}

impl SedonaScalarKernel for RsNumDimensionsWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            SedonaType::Arrow(DataType::Int32),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let band_index_array = args[1].clone().cast_to(&DataType::Int32, None)?;
        let band_index_array = band_index_array.into_array(executor.num_iterations())?;
        let band_index_array = as_int32_array(&band_index_array)?;

        let mut builder = Int32Builder::with_capacity(executor.num_iterations());
        let mut band_index_iter = band_index_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let band_index = band_index_iter.next().unwrap().unwrap_or(1);
            match raster_opt {
                None => {
                    builder.append_null();
                    Ok(())
                }
                Some(raster) => {
                    if band_index < 1 || band_index > raster.num_bands() as i32 {
                        builder.append_null();
                        return Ok(());
                    }
                    let band = raster
                        .band((band_index - 1) as usize)
                        .map_err(|e| arrow_datafusion_err!(e))?;
                    builder.append_value(band.ndim() as i32);
                    Ok(())
                }
            }
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// RS_DimNames(raster [, band]) -> List<Utf8>
///
/// Returns the dimension names of the raster (or a specific band).
pub fn rs_dimnames_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_dimnames",
        vec![Arc::new(RsDimNames {}), Arc::new(RsDimNamesWithBand {})],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsDimNames {}

impl SedonaScalarKernel for RsDimNames {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(list_utf8_type()),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut list_builder = ListBuilder::new(StringBuilder::new());

        executor.execute_raster_void(|_i, raster_opt| match raster_opt {
            None => {
                list_builder.append_null();
                Ok(())
            }
            Some(raster) => {
                let names = check_band_agreement(raster, "RS_DimNames", "dimension names", |b| {
                    b.dim_names()
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })?;
                for name in &names {
                    list_builder.values().append_value(name);
                }
                list_builder.append(true);
                Ok(())
            }
        })?;

        executor.finish(Arc::new(list_builder.finish()))
    }
}

#[derive(Debug)]
struct RsDimNamesWithBand {}

impl SedonaScalarKernel for RsDimNamesWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            SedonaType::Arrow(list_utf8_type()),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let band_index_array = args[1].clone().cast_to(&DataType::Int32, None)?;
        let band_index_array = band_index_array.into_array(executor.num_iterations())?;
        let band_index_array = as_int32_array(&band_index_array)?;

        let mut list_builder = ListBuilder::new(StringBuilder::new());
        let mut band_index_iter = band_index_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let band_index = band_index_iter.next().unwrap().unwrap_or(1);
            match raster_opt {
                None => {
                    list_builder.append_null();
                    Ok(())
                }
                Some(raster) => {
                    if band_index < 1 || band_index > raster.num_bands() as i32 {
                        list_builder.append_null();
                        return Ok(());
                    }
                    let band = raster
                        .band((band_index - 1) as usize)
                        .map_err(|e| arrow_datafusion_err!(e))?;
                    for name in band.dim_names() {
                        list_builder.values().append_value(name);
                    }
                    list_builder.append(true);
                    Ok(())
                }
            }
        })?;

        executor.finish(Arc::new(list_builder.finish()))
    }
}

/// RS_DimSize(raster, dim_name [, band]) -> Int64 (nullable)
///
/// Returns the size of the named dimension, or null if the dimension
/// does not exist.
pub fn rs_dimsize_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_dimsize",
        vec![Arc::new(RsDimSize {}), Arc::new(RsDimSizeWithBand {})],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsDimSize {}

impl SedonaScalarKernel for RsDimSize {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_string()],
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
        let dim_name_array = args[1].clone().cast_to(&DataType::Utf8, None)?;
        let dim_name_array = dim_name_array.into_array(executor.num_iterations())?;
        let dim_name_array = as_string_array(&dim_name_array)?;

        let mut builder = Int64Builder::with_capacity(executor.num_iterations());
        let mut dim_name_iter = dim_name_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let dim_name = dim_name_iter.next().unwrap();
            match (raster_opt, dim_name) {
                (None, _) | (_, None) => {
                    builder.append_null();
                    Ok(())
                }
                (Some(raster), Some(name)) => {
                    // Pass-through for bands that don't carry the named
                    // dimension: filter to bands that have it, then require
                    // those to agree on the size. If no band has the dim at
                    // all, error explicitly — `check_band_agreement` on the
                    // raw `Option<u64>` extractor would silently return NULL
                    // and a mixed Some/None set would report a misleading
                    // "different dimension sizes" error.
                    let mut iter = (0..raster.num_bands()).filter_map(|i| {
                        let band = raster.band(i).ok()?;
                        let size = band.dim_size(name)?;
                        Some((i, size))
                    });
                    let Some((first_idx, first_size)) = iter.next() else {
                        return exec_err!("RS_DimSize: no band has dimension '{name}'");
                    };
                    for (other_idx, other_size) in iter {
                        if other_size != first_size {
                            return exec_err!(
                                "RS_DimSize: bands have different sizes for dimension '{name}' \
                                 (band {first_idx} has {first_size}, band {other_idx} has \
                                 {other_size}) — specify a band index"
                            );
                        }
                    }
                    builder.append_value(first_size);
                    Ok(())
                }
            }
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct RsDimSizeWithBand {}

impl SedonaScalarKernel for RsDimSizeWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_integer(),
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
        let dim_name_array = args[1].clone().cast_to(&DataType::Utf8, None)?;
        let dim_name_array = dim_name_array.into_array(executor.num_iterations())?;
        let dim_name_array = as_string_array(&dim_name_array)?;
        let band_index_array = args[2].clone().cast_to(&DataType::Int32, None)?;
        let band_index_array = band_index_array.into_array(executor.num_iterations())?;
        let band_index_array = as_int32_array(&band_index_array)?;

        let mut builder = Int64Builder::with_capacity(executor.num_iterations());
        let mut dim_name_iter = dim_name_array.iter();
        let mut band_index_iter = band_index_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let dim_name = dim_name_iter.next().unwrap();
            let band_index = band_index_iter.next().unwrap().unwrap_or(1);
            match (raster_opt, dim_name) {
                (None, _) | (_, None) => {
                    builder.append_null();
                    Ok(())
                }
                (Some(raster), Some(name)) => {
                    if band_index < 1 || band_index > raster.num_bands() as i32 {
                        builder.append_null();
                        return Ok(());
                    }
                    let band = raster
                        .band((band_index - 1) as usize)
                        .map_err(|e| arrow_datafusion_err!(e))?;
                    match band.dim_size(name) {
                        Some(s) => builder.append_value(s),
                        None => builder.append_null(),
                    }
                    Ok(())
                }
            }
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// RS_Shape(raster [, band]) -> List<Int64>
///
/// Returns the shape (size of each dimension) of the raster.
pub fn rs_shape_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_shape",
        vec![Arc::new(RsShape {}), Arc::new(RsShapeWithBand {})],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsShape {}

impl SedonaScalarKernel for RsShape {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(list_int64_type()),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut list_builder = ListBuilder::new(Int64Builder::new());

        executor.execute_raster_void(|_i, raster_opt| match raster_opt {
            None => {
                list_builder.append_null();
                Ok(())
            }
            Some(raster) => {
                let shape =
                    check_band_agreement(raster, "RS_Shape", "shape", |b| b.shape().to_vec())?;
                for &s in &shape {
                    list_builder.values().append_value(s);
                }
                list_builder.append(true);
                Ok(())
            }
        })?;

        executor.finish(Arc::new(list_builder.finish()))
    }
}

#[derive(Debug)]
struct RsShapeWithBand {}

impl SedonaScalarKernel for RsShapeWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            SedonaType::Arrow(list_int64_type()),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let band_index_array = args[1].clone().cast_to(&DataType::Int32, None)?;
        let band_index_array = band_index_array.into_array(executor.num_iterations())?;
        let band_index_array = as_int32_array(&band_index_array)?;

        let mut list_builder = ListBuilder::new(Int64Builder::new());
        let mut band_index_iter = band_index_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let band_index = band_index_iter.next().unwrap().unwrap_or(1);
            match raster_opt {
                None => {
                    list_builder.append_null();
                    Ok(())
                }
                Some(raster) => {
                    if band_index < 1 || band_index > raster.num_bands() as i32 {
                        list_builder.append_null();
                        return Ok(());
                    }
                    let band = raster
                        .band((band_index - 1) as usize)
                        .map_err(|e| arrow_datafusion_err!(e))?;
                    for &s in band.shape() {
                        list_builder.values().append_value(s);
                    }
                    list_builder.append(true);
                    Ok(())
                }
            }
        })?;

        executor.finish(Arc::new(list_builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int32Array, ListArray, StringArray, StructArray};
    use datafusion_expr::ScalarUDF;
    use sedona_raster::builder::RasterBuilder;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    /// Build a single-row 2D raster StructArray.
    fn build_2d_raster(width: u64, height: u64) -> StructArray {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(
                width as i64,
                height as i64,
                0.0,
                0.0,
                1.0,
                -1.0,
                0.0,
                0.0,
                None,
            )
            .unwrap();
        builder.start_band_2d(BandDataType::Float32, None).unwrap();
        let data = vec![0u8; (width * height * 4) as usize];
        builder.band_data_writer().append_value(&data);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    /// Build a single-row 3D raster StructArray with shape [time, height, width].
    fn build_3d_raster(time: u64, height: u64, width: u64) -> StructArray {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(
                &transform,
                &["x", "y"],
                &[width as i64, height as i64],
                None,
            )
            .unwrap();
        builder
            .start_band_nd(
                None,
                &["time", "y", "x"],
                &[time as i64, height as i64, width as i64],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        let data = vec![0u8; (time * height * width * 4) as usize];
        builder.band_data_writer().append_value(&data);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    /// Build a raster where two bands both carry a `time` dimension but
    /// disagree on its size. Used to exercise the "filtered bands disagree"
    /// error path in `RS_DimSize`.
    fn build_conflicting_time_size_raster() -> StructArray {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[5, 4], None)
            .unwrap();

        // Band 0: time=3
        builder
            .start_band_nd(
                None,
                &["time", "y", "x"],
                &[3, 4, 5],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8; 3 * 4 * 5 * 4]);
        builder.finish_band().unwrap();

        // Band 1: time=7 (disagrees with band 0)
        builder
            .start_band_nd(
                None,
                &["time", "y", "x"],
                &[7, 4, 5],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8; 7 * 4 * 5 * 4]);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    /// Build a raster with two bands that have different dimensionality.
    fn build_mixed_dim_raster() -> StructArray {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[5, 4], None)
            .unwrap();

        // Band 0: 2D [4, 5]
        builder
            .start_band_nd(
                None,
                &["y", "x"],
                &[4, 5],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8; 4 * 5 * 4]);
        builder.finish_band().unwrap();

        // Band 1: 3D [3, 4, 5]
        builder
            .start_band_nd(
                None,
                &["time", "y", "x"],
                &[3, 4, 5],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8; 3 * 4 * 5 * 4]);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    #[test]
    fn numdimensions_2d() {
        let udf: ScalarUDF = rs_numdimensions_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        tester.assert_return_type(DataType::Int32);

        let rasters = build_2d_raster(4, 5);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let arr = result
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        assert_eq!(arr.value(0), 2);
    }

    #[test]
    fn numdimensions_3d() {
        let udf: ScalarUDF = rs_numdimensions_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        let rasters = build_3d_raster(3, 4, 5);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let arr = result
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        assert_eq!(arr.value(0), 3);
    }

    #[test]
    fn numdimensions_with_band() {
        let udf: ScalarUDF = rs_numdimensions_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let rasters = build_3d_raster(3, 4, 5);
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), 1_i32)
            .unwrap();
        let arr = result
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Expected Int32Array");
        assert_eq!(arr.value(0), 3);
    }

    #[test]
    fn numdimensions_null_raster() {
        let udf: ScalarUDF = rs_numdimensions_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        let rasters = generate_test_rasters(1, Some(0)).unwrap();
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert!(result.is_null(0));
    }

    #[test]
    fn numdimensions_mixed_bands_error() {
        let udf: ScalarUDF = rs_numdimensions_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        let rasters = build_mixed_dim_raster();
        let result = tester.invoke_array(Arc::new(rasters));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("bands have different dimensionality"),
            "Unexpected error: {err_msg}"
        );
    }

    #[test]
    fn dimnames_2d() {
        let udf: ScalarUDF = rs_dimnames_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        tester.assert_return_type(list_utf8_type());

        let rasters = build_2d_raster(4, 5);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let list_arr = result
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray");
        let values = list_arr.value(0);
        let str_arr = values
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected StringArray");
        assert_eq!(str_arr.len(), 2);
        assert_eq!(str_arr.value(0), "y");
        assert_eq!(str_arr.value(1), "x");
    }

    #[test]
    fn dimnames_3d() {
        let udf: ScalarUDF = rs_dimnames_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        let rasters = build_3d_raster(3, 4, 5);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let list_arr = result
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray");
        let values = list_arr.value(0);
        let str_arr = values
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected StringArray");
        assert_eq!(str_arr.len(), 3);
        assert_eq!(str_arr.value(0), "time");
        assert_eq!(str_arr.value(1), "y");
        assert_eq!(str_arr.value(2), "x");
    }

    #[test]
    fn dimnames_null_raster() {
        let udf: ScalarUDF = rs_dimnames_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        let rasters = generate_test_rasters(1, Some(0)).unwrap();
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert!(result.is_null(0));
    }

    #[test]
    fn dimnames_mixed_bands_error() {
        let udf: ScalarUDF = rs_dimnames_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        let rasters = build_mixed_dim_raster();
        let result = tester.invoke_array(Arc::new(rasters));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("bands have different dimension names"),
            "Unexpected error: {err_msg}"
        );
    }

    #[test]
    fn dimsize_2d_x() {
        let udf: ScalarUDF = rs_dimsize_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = build_2d_raster(5, 4);
        let result = tester.invoke_array_scalar(Arc::new(rasters), "x").unwrap();
        let arr = result
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("Expected Int64Array");
        assert_eq!(arr.value(0), 5);
    }

    #[test]
    fn dimsize_3d_time() {
        let udf: ScalarUDF = rs_dimsize_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = build_3d_raster(3, 4, 5);
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), "time")
            .unwrap();
        let arr = result
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("Expected Int64Array");
        assert_eq!(arr.value(0), 3);
    }

    #[test]
    fn dimsize_nonexistent_errors_when_no_band_has_dim() {
        // RS_DimSize on a dim that no band carries surfaces an explicit
        // error rather than silently returning NULL. Catches typo'd dim
        // names.
        let udf: ScalarUDF = rs_dimsize_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = build_2d_raster(4, 5);
        let result = tester.invoke_array_scalar(Arc::new(rasters), "nonexistent");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("no band has dimension 'nonexistent'"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn dimsize_with_band() {
        let udf: ScalarUDF = rs_dimsize_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        let rasters = build_3d_raster(3, 4, 5);
        let result = tester
            .invoke_array_scalar_scalar(Arc::new(rasters), "time", 1_i32)
            .unwrap();
        let arr = result
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("Expected Int64Array");
        assert_eq!(arr.value(0), 3);
    }

    #[test]
    fn dimsize_null_raster() {
        let udf: ScalarUDF = rs_dimsize_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = generate_test_rasters(1, Some(0)).unwrap();
        let result = tester.invoke_array_scalar(Arc::new(rasters), "x").unwrap();
        assert!(result.is_null(0));
    }

    #[test]
    fn dimsize_errors_when_bands_with_dim_disagree_on_size() {
        // Pass-through skips bands missing the dim, but the bands that DO
        // have it must agree on the size — otherwise the answer is
        // ambiguous and the user must specify a band index.
        let udf: ScalarUDF = rs_dimsize_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        let rasters = build_conflicting_time_size_raster();
        let result = tester.invoke_array_scalar(Arc::new(rasters), "time");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("bands have different sizes for dimension 'time'"),
            "Unexpected error: {err}"
        );
    }

    #[test]
    fn dimsize_passes_through_bands_without_dim() {
        // RS_DimSize on a heterogeneous raster: bands without the named
        // dimension are skipped; the size is read from the bands that have
        // it. Mirrors RS_Slice / RS_SliceRange / RS_DimToBand pass-through
        // semantics and matches xarray's `ds.sizes`.
        let udf: ScalarUDF = rs_dimsize_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);

        // build_mixed_dim_raster: band 0 is 2D (no `time`), band 1 is 3D
        // with `time=3`. RS_DimSize(raster, 'time') reports the size from
        // band 1.
        let rasters = build_mixed_dim_raster();
        let result = tester
            .invoke_array_scalar(Arc::new(rasters), "time")
            .unwrap();
        let arr = result
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("Expected Int64Array");
        assert_eq!(arr.value(0), 3);
    }

    #[test]
    fn shape_2d() {
        let udf: ScalarUDF = rs_shape_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        tester.assert_return_type(list_int64_type());

        let rasters = build_2d_raster(5, 4);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let list_arr = result
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray");
        let values = list_arr.value(0);
        let int_arr = values
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("Expected Int64Array");
        assert_eq!(int_arr.len(), 2);
        assert_eq!(int_arr.value(0), 4); // height
        assert_eq!(int_arr.value(1), 5); // width
    }

    #[test]
    fn shape_3d() {
        let udf: ScalarUDF = rs_shape_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        let rasters = build_3d_raster(3, 4, 5);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let list_arr = result
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("Expected ListArray");
        let values = list_arr.value(0);
        let int_arr = values
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("Expected Int64Array");
        assert_eq!(int_arr.len(), 3);
        assert_eq!(int_arr.value(0), 3); // time
        assert_eq!(int_arr.value(1), 4); // height
        assert_eq!(int_arr.value(2), 5); // width
    }

    #[test]
    fn shape_null_raster() {
        let udf: ScalarUDF = rs_shape_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        let rasters = generate_test_rasters(1, Some(0)).unwrap();
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert!(result.is_null(0));
    }

    #[test]
    fn shape_mixed_bands_error() {
        let udf: ScalarUDF = rs_shape_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        let rasters = build_mixed_dim_raster();
        let result = tester.invoke_array(Arc::new(rasters));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("bands have different shape"),
            "Unexpected error: {err_msg}"
        );
    }
}
