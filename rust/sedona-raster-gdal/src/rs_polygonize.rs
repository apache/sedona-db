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

//! RS_Polygonize UDF - Convert raster band to vector polygons
//!
//! Returns a list of polygons for all connected regions of pixels with the same
//! value in the specified band.
use std::convert::TryInto;
use std::sync::Arc;

use arrow_array::builder::{BinaryBuilder, Float64Builder, ListBuilder, StructBuilder};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::cast::as_int32_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_gdal::gdal::Gdal;
use sedona_gdal::gdal_dyn_bindgen::{OGRFieldType, OGRwkbGeometryType};
use sedona_gdal::mem::MemDatasetBuilder;

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

// `dataset` removed; the provider is used instead when creating GDAL datasets.
use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::configure_thread_local_options;

/// RS_Polygonize() scalar UDF implementation
///
/// Returns a list of polygons for connected regions of pixels with the same value
pub fn rs_polygonize_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_polygonize",
        vec![Arc::new(RsPolygonize)],
        Volatility::Immutable,
    )
}

/// Kernel implementation for RS_Polygonize
#[derive(Debug)]
struct RsPolygonize;

impl SedonaScalarKernel for RsPolygonize {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            // Return type is List<Struct<geom: Binary, value: Float64>>
            SedonaType::Arrow(polygon_value_list_type()),
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
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();

        // Get the band number array
        let band_array = args[1]
            .clone()
            .cast_to(&DataType::Int32, None)?
            .into_array(num_iterations)?;
        let band_array = as_int32_array(&band_array)?;
        let mut band_iter = band_array.iter();

        // Build result as List<Struct<geom, value>>
        let struct_fields = polygon_value_struct_fields();
        let mut list_builder = ListBuilder::new(StructBuilder::from_fields(struct_fields, 16));

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            executor.execute_raster_void(|_i, raster_opt| {
                let band_opt = band_iter.next().unwrap();

                let raster = match (raster_opt, band_opt) {
                    (Some(raster), Some(_)) => raster,
                    _ => {
                        list_builder.append_null();
                        return Ok(());
                    }
                };

                let band_num: usize = band_opt.unwrap().max(1).try_into().unwrap_or(1);

                match polygonize_raster(gdal, raster, band_num) {
                    Ok(polygon_values) => {
                        let struct_builder = list_builder.values();

                        for (wkb, value) in polygon_values {
                            // Get field builders
                            let geom_builder = struct_builder
                                .field_builder::<BinaryBuilder>(0)
                                .expect("Expected BinaryBuilder for geom field");
                            geom_builder.append_value(&wkb);

                            let value_builder = struct_builder
                                .field_builder::<Float64Builder>(1)
                                .expect("Expected Float64Builder for value field");
                            value_builder.append_value(value);

                            struct_builder.append(true);
                        }
                        list_builder.append(true);
                    }
                    Err(e) => {
                        // Log error but append null
                        eprintln!("Polygonize error: {}", e);
                        list_builder.append_null();
                    }
                }

                Ok(())
            })?;

            executor.finish(Arc::new(list_builder.finish()))
        })
    }
}

/// Return type for the list of polygon/value structs
fn polygon_value_list_type() -> DataType {
    DataType::List(Arc::new(Field::new(
        "item",
        DataType::Struct(polygon_value_struct_fields()),
        true,
    )))
}

/// Struct fields for polygon/value pairs
fn polygon_value_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("geom", DataType::Binary, false),
        Field::new("value", DataType::Float64, false),
    ])
}

/// Polygonize a raster band using GDAL
fn polygonize_raster(
    gdal: &Gdal,
    raster: &RasterRefImpl<'_>,
    band_num: usize,
) -> Result<Vec<(Vec<u8>, f64)>> {
    let bands = raster.bands();
    if band_num == 0 || band_num > bands.len() {
        return exec_err!("Band {} is out of range (1-{})", band_num, bands.len());
    }

    // Create GDAL dataset from raster (thread-local provider)
    let provider = crate::gdal_dataset_provider::thread_local_provider(gdal)
        .map_err(|e| exec_datafusion_err!("Failed to init GDAL provider: {}", e))?;
    let raster_ds = provider
        .raster_ref_to_gdal(raster)
        .map_err(|e| exec_datafusion_err!("Failed to create GDAL dataset: {}", e))?;
    let gdal_dataset = raster_ds.as_dataset();

    // Get the raster band
    let raster_band = gdal_dataset
        .rasterband(band_num)
        .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_num, e))?;

    // Create memory datasource for output polygons
    let mem_ds = unsafe {
        MemDatasetBuilder::new(0, 0)
            .build(gdal)
            .map_err(|e| exec_datafusion_err!("Failed to create memory dataset: {}", e))?
    };

    // Create layer with geometry field and value field
    let spatial_ref = gdal_dataset.spatial_ref().ok();
    let layer = mem_ds
        .create_layer(sedona_gdal::dataset::LayerOptions {
            name: "polygons",
            srs: spatial_ref.as_ref(),
            ty: OGRwkbGeometryType::wkbPolygon,
            options: None,
        })
        .map_err(|e| exec_datafusion_err!("Failed to create layer: {}", e))?;

    // Add pixel value field
    let field_defn = gdal
        .create_field_defn("value", OGRFieldType::OFTReal)
        .map_err(|e| exec_datafusion_err!("Failed to create field definition: {}", e))?;
    layer
        .create_field(&field_defn)
        .map_err(|e| exec_datafusion_err!("Failed to add field to layer: {}", e))?;

    // Call GDAL Polygonize
    gdal.polygonize(
        &raster_band,
        None,
        &layer,
        0,
        &sedona_gdal::raster::PolygonizeOptions::default(),
    )
    .map_err(|e| exec_datafusion_err!("GDAL polygonize failed: {e}"))?;

    // Extract polygons from layer
    let mut polygon_values = Vec::new();

    let mut value_field_idx: Option<i32> = None;
    for feature in layer.features() {
        let geom = feature
            .geometry()
            .ok_or_else(|| exec_datafusion_err!("Polygonize output feature missing geometry"))?;
        let wkb = geom
            .wkb()
            .map_err(|e| exec_datafusion_err!("Failed to export geometry to WKB: {e}"))?;

        let idx = match value_field_idx {
            Some(idx) => idx,
            None => {
                let idx = feature
                    .field_index("value")
                    .map_err(|e| exec_datafusion_err!("Missing 'value' field: {e}"))?;
                value_field_idx = Some(idx);
                idx
            }
        };

        let value = feature.field_as_double(idx);

        polygon_values.push((wkb, value));
    }

    Ok(polygon_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::ScalarValue;
    use sedona_raster::array::RasterStructArray;

    #[test]
    fn test_polygon_value_list_type() {
        let dt = polygon_value_list_type();
        match dt {
            DataType::List(field) => {
                assert_eq!(field.name(), "item");
                match field.data_type() {
                    DataType::Struct(fields) => {
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].name(), "geom");
                        assert_eq!(fields[1].name(), "value");
                    }
                    _ => panic!("Expected Struct data type"),
                }
            }
            _ => panic!("Expected List data type"),
        }
    }

    #[test]
    fn test_polygonize_raster() {
        // Load test raster and polygonize it
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let result = with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();
            polygonize_raster(gdal, &raster, 1)
        })
        .unwrap();

        // Should return at least one polygon
        assert!(
            !result.is_empty(),
            "Polygonize should return at least one polygon"
        );

        // Each result should have a valid WKB and value
        for (wkb, value) in &result {
            // WKB should be at least 5 bytes (header)
            assert!(wkb.len() >= 5, "WKB should be at least 5 bytes");
            // Value should be finite
            assert!(value.is_finite(), "Value should be a finite number");
        }
    }

    #[test]
    fn test_polygonize_kernel_return_type() {
        use arrow_schema::DataType;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_schema::datatypes::RASTER;

        let kernel = RsPolygonize;

        let arg_types = vec![RASTER, SedonaType::Arrow(DataType::Int32)];
        let return_type = kernel.return_type(&arg_types).unwrap();
        assert!(return_type.is_some());

        // Return type should be List<Struct<geom, value>>
        match return_type.unwrap() {
            SedonaType::Arrow(DataType::List(_)) => (),
            _ => panic!("Expected List return type"),
        }
    }

    #[test]
    fn test_polygonize_invoke_batch() {
        use arrow_schema::DataType;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_schema::datatypes::RASTER;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let raster_array =
            with_gdal(|gdal| crate::utils::load_as_indb_raster(gdal, &test_file)).unwrap();

        let kernel = RsPolygonize;

        let arg_types = vec![RASTER, SedonaType::Arrow(DataType::Int32)];
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_array))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(1))), // band
        ];

        let result = kernel
            .invoke_batch_from_args(
                &arg_types,
                &args,
                &SedonaType::Arrow(DataType::Null),
                0,
                None,
            )
            .unwrap();

        // Result should be a scalar (since input was scalar)
        match result {
            ColumnarValue::Scalar(_) => (),
            _ => panic!("Expected Scalar result"),
        }
    }
}
