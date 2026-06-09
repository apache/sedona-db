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

use arrow_array::builder::{Float64Builder, Int32Builder, UInt64Builder};
use arrow_array::StructArray;
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::exec_datafusion_err;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::crs::deserialize_crs;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::{configure_thread_local_options, thread_local_provider};

/// RS_MetaData() scalar UDF implementation (GDAL-backed)
pub fn rs_metadata_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_metadata",
        vec![Arc::new(RsMetaData {})],
        Volatility::Immutable,
    )
}

fn metadata_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("upperLeftX", DataType::Float64, true),
        Field::new("upperLeftY", DataType::Float64, true),
        Field::new("gridWidth", DataType::UInt64, true),
        Field::new("gridHeight", DataType::UInt64, true),
        Field::new("scaleX", DataType::Float64, true),
        Field::new("scaleY", DataType::Float64, true),
        Field::new("skewX", DataType::Float64, true),
        Field::new("skewY", DataType::Float64, true),
        Field::new("srid", DataType::Int32, true),
        Field::new("numSampleDimensions", DataType::UInt64, true),
        Field::new("tileWidth", DataType::UInt64, true),
        Field::new("tileHeight", DataType::UInt64, true),
    ])
}

#[derive(Debug)]
struct RsMetaData {}

impl SedonaScalarKernel for RsMetaData {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Struct(metadata_struct_fields())),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke_batch_from_args(arg_types, args, &SedonaType::Arrow(DataType::Null), 0, None)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let executor = sedona_raster_functions::RasterExecutor::new(arg_types, args);
        let capacity = executor.num_iterations();

        let mut upper_left_x_builder = Float64Builder::with_capacity(capacity);
        let mut upper_left_y_builder = Float64Builder::with_capacity(capacity);
        let mut grid_width_builder = UInt64Builder::with_capacity(capacity);
        let mut grid_height_builder = UInt64Builder::with_capacity(capacity);
        let mut scale_x_builder = Float64Builder::with_capacity(capacity);
        let mut scale_y_builder = Float64Builder::with_capacity(capacity);
        let mut skew_x_builder = Float64Builder::with_capacity(capacity);
        let mut skew_y_builder = Float64Builder::with_capacity(capacity);
        let mut srid_builder = Int32Builder::with_capacity(capacity);
        let mut num_bands_builder = UInt64Builder::with_capacity(capacity);
        let mut tile_width_builder = UInt64Builder::with_capacity(capacity);
        let mut tile_height_builder = UInt64Builder::with_capacity(capacity);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            let provider = thread_local_provider(gdal)
                .map_err(|e| exec_datafusion_err!("Failed to init GDAL provider: {e}"))?;

            executor.execute_raster_void(|_i, raster_opt| {
                match raster_opt {
                    None => {
                        upper_left_x_builder.append_null();
                        upper_left_y_builder.append_null();
                        grid_width_builder.append_null();
                        grid_height_builder.append_null();
                        scale_x_builder.append_null();
                        scale_y_builder.append_null();
                        skew_x_builder.append_null();
                        skew_y_builder.append_null();
                        srid_builder.append_null();
                        num_bands_builder.append_null();
                        tile_width_builder.append_null();
                        tile_height_builder.append_null();
                    }
                    Some(raster) => {
                        let metadata = raster.metadata();

                        upper_left_x_builder.append_value(metadata.upper_left_x());
                        upper_left_y_builder.append_value(metadata.upper_left_y());
                        grid_width_builder.append_value(metadata.width());
                        grid_height_builder.append_value(metadata.height());
                        scale_x_builder.append_value(metadata.scale_x());
                        scale_y_builder.append_value(metadata.scale_y());
                        skew_x_builder.append_value(metadata.skew_x());
                        skew_y_builder.append_value(metadata.skew_y());

                        let srid = match raster.crs() {
                            None => 0i32,
                            Some(crs_str) => match deserialize_crs(crs_str) {
                                Ok(Some(crs_ref)) => {
                                    crs_ref.srid().ok().flatten().map(|s| s as i32).unwrap_or(0)
                                }
                                _ => 0i32,
                            },
                        };
                        srid_builder.append_value(srid);

                        num_bands_builder.append_value(raster.bands().len() as u64);

                        let dataset = provider.raster_ref_to_gdal(raster).map_err(|e| {
                            exec_datafusion_err!("Failed to create GDAL dataset: {e}")
                        })?;
                        let band1 = dataset
                            .as_dataset()
                            .rasterband(1)
                            .map_err(|e| exec_datafusion_err!("Failed to get band 1: {e}"))?;
                        let (block_x, block_y) = band1.block_size();
                        tile_width_builder.append_value(block_x.max(1) as u64);
                        tile_height_builder.append_value(block_y.max(1) as u64);
                    }
                }
                Ok(())
            })
        })?;

        let struct_array = StructArray::new(
            metadata_struct_fields(),
            vec![
                Arc::new(upper_left_x_builder.finish()),
                Arc::new(upper_left_y_builder.finish()),
                Arc::new(grid_width_builder.finish()),
                Arc::new(grid_height_builder.finish()),
                Arc::new(scale_x_builder.finish()),
                Arc::new(scale_y_builder.finish()),
                Arc::new(skew_x_builder.finish()),
                Arc::new(skew_y_builder.finish()),
                Arc::new(srid_builder.finish()),
                Arc::new(num_bands_builder.finish()),
                Arc::new(tile_width_builder.finish()),
                Arc::new(tile_height_builder.finish()),
            ],
            None,
        );

        executor.finish(Arc::new(struct_array))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn rs_metadata_udf_docs() {
        let udf: ScalarUDF = rs_metadata_udf().into();
        assert_eq!(udf.name(), "rs_metadata");
    }

    #[test]
    fn rs_metadata_tile_dimensions_from_gdal() {
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let (raster_array, block_x, block_y) = with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();
            let provider = thread_local_provider(gdal).unwrap();
            let dataset = provider.raster_ref_to_gdal(&raster).unwrap();
            let band1 = dataset.as_dataset().rasterband(1).unwrap();
            let (block_x, block_y) = band1.block_size();
            Ok::<_, datafusion_common::DataFusionError>((raster_array, block_x, block_y))
        })
        .unwrap();

        let udf: ScalarUDF = rs_metadata_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_array(Arc::new(raster_array)).unwrap();
        let struct_array = result.as_struct();

        let tile_width = struct_array
            .column(10)
            .as_primitive::<arrow_array::types::UInt64Type>()
            .value(0);
        let tile_height = struct_array
            .column(11)
            .as_primitive::<arrow_array::types::UInt64Type>()
            .value(0);

        assert_eq!(tile_width, block_x.max(1) as u64);
        assert_eq!(tile_height, block_y.max(1) as u64);
    }

    #[test]
    fn rs_metadata_tile_dimensions_golden() {
        let test_file = sedona_testing::data::test_raster("test5.tiff").unwrap();
        let (raster_array, block_x, block_y) = with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();
            let provider = thread_local_provider(gdal).unwrap();
            let dataset = provider.raster_ref_to_gdal(&raster).unwrap();
            let band1 = dataset.as_dataset().rasterband(1).unwrap();
            let (block_x, block_y) = band1.block_size();
            Ok::<_, datafusion_common::DataFusionError>((raster_array, block_x, block_y))
        })
        .unwrap();

        let udf: ScalarUDF = rs_metadata_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_array(Arc::new(raster_array)).unwrap();
        let struct_array = result.as_struct();

        let tile_width = struct_array
            .column(10)
            .as_primitive::<arrow_array::types::UInt64Type>()
            .value(0);
        let tile_height = struct_array
            .column(11)
            .as_primitive::<arrow_array::types::UInt64Type>()
            .value(0);

        assert_eq!(tile_width, block_x.max(1) as u64);
        assert_eq!(tile_height, block_y.max(1) as u64);
    }
}
