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

//! RS_Polygonize UDF - Convert a raster band to vector polygons.

use std::convert::TryInto;
use std::sync::Arc;

use arrow_array::builder::{BinaryBuilder, Float64Builder, ListBuilder, StructBuilder};
use arrow_array::{ArrayRef, Int32Array};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::cast::as_int32_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_gdal::driver::Driver;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::gdal_dyn_bindgen::{OGRFieldType, OGRwkbGeometryType};
use sedona_gdal::raster::polygonize::PolygonizeOptions;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::{
    configure_thread_local_options, thread_local_provider, RasterDataset,
};

pub fn rs_polygonize_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_polygonize",
        vec![Arc::new(RsPolygonize)],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsPolygonize;

impl SedonaScalarKernel for RsPolygonize {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            SedonaType::Arrow(polygon_value_list_type()),
        )
        .match_args(args)
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
        let band_array = band_argument_array(&args[1], num_iterations)?;
        let mut band_iter = band_array.iter();

        let mut list_builder = ListBuilder::new(StructBuilder::from_fields(
            polygon_value_struct_fields(),
            num_iterations,
        ));

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            let provider = thread_local_provider(gdal)
                .map_err(|e| exec_datafusion_err!("Failed to init GDAL provider: {e}"))?;
            // Note: We deliberately use "Memory" instead of "MEM" to be compatible with older GDAL versions.
            let mem_driver = gdal
                .get_driver_by_name("Memory")
                .map_err(|e| exec_datafusion_err!("Failed to get Memory driver: {e}"))?;

            executor.execute_raster_void(|_, raster_opt| {
                let band_opt = band_iter.next().expect("band iteration should match rows");

                let raster = match (raster_opt, band_opt) {
                    (Some(raster), Some(_)) => raster,
                    _ => {
                        list_builder.append_null();
                        return Ok(());
                    }
                };

                let band_num: usize = band_opt.unwrap().max(1).try_into().unwrap_or(1);

                let bands = raster.bands();
                if band_num == 0 || band_num > bands.len() {
                    return exec_err!("Band {} is out of range (1-{})", band_num, bands.len());
                }

                let raster_ds = provider
                    .raster_ref_to_gdal(raster)
                    .map_err(|e| exec_datafusion_err!("Failed to create GDAL dataset: {e}"))?;

                let polygon_values = polygonize_raster(gdal, &mem_driver, raster_ds, band_num)?;
                let struct_builder = list_builder.values();
                for (wkb, value) in polygon_values {
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
                Ok(())
            })?;

            let result: ArrayRef = Arc::new(list_builder.finish());
            executor.finish(result)
        })
    }
}

fn band_argument_array(arg: &ColumnarValue, num_iterations: usize) -> Result<Int32Array> {
    let array = arg
        .clone()
        .cast_to(&DataType::Int32, None)?
        .into_array(num_iterations)?;
    Ok(as_int32_array(&array)?.clone())
}

fn polygon_value_list_type() -> DataType {
    DataType::List(Arc::new(Field::new(
        "item",
        DataType::Struct(polygon_value_struct_fields()),
        true,
    )))
}

fn polygon_value_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("geom", DataType::Binary, false),
        Field::new("value", DataType::Float64, false),
    ])
}

/// Perform GDAL polygonize on the given raster dataset and band, returning a list of (WKB, value) tuples.
fn polygonize_raster(
    gdal: &Gdal,
    mem_driver: &Driver,
    raster_ds: RasterDataset<'_>,
    band_num: usize,
) -> Result<Vec<(Vec<u8>, f64)>> {
    let gdal_dataset = raster_ds.as_dataset();

    let raster_band = gdal_dataset
        .rasterband(band_num)
        .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_num, e))?;

    // Create a memory dataset containing one vector layer to hold the polygonized output.
    let vector_ds = mem_driver
        .create_vector_only("")
        .map_err(|e| exec_datafusion_err!("Failed to create vector dataset: {e}"))?;

    let spatial_ref = gdal_dataset.spatial_ref().ok();
    let mut layer = vector_ds
        .create_layer(sedona_gdal::dataset::LayerOptions {
            name: "polygons",
            srs: spatial_ref.as_ref(),
            ty: OGRwkbGeometryType::wkbPolygon,
            options: None,
        })
        .map_err(|e| exec_datafusion_err!("Failed to create layer: {e}"))?;

    let field_defn = gdal
        .create_field_defn("value", OGRFieldType::OFTReal)
        .map_err(|e| exec_datafusion_err!("Failed to create field definition: {e}"))?;
    layer
        .create_field(&field_defn)
        .map_err(|e| exec_datafusion_err!("Failed to add field to layer: {e}"))?;

    // Polygonize the raster band into the vector layer, using the "value" field to store pixel values.
    gdal.polygonize(&raster_band, None, &layer, 0, &PolygonizeOptions::default())
        .map_err(|e| exec_datafusion_err!("GDAL polygonize failed: {e}"))?;

    // Extract the WKB geometry and value for each feature in the layer.
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

        polygon_values.push((wkb, feature.field_as_double(idx)));
    }

    Ok(polygon_values)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::Array;
    use datafusion_common::cast::{as_list_array, as_struct_array};
    use datafusion_common::ScalarValue;
    use datafusion_expr::{ScalarUDF, ScalarUDFImpl};
    use sedona_gdal::raster::types::Buffer;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::RasterSpec;
    use sedona_testing::testers::ScalarUdfTester;
    use tempfile::tempdir;

    fn test_raster_spec() -> RasterSpec {
        RasterSpec::d2(3, 3)
            .transform([0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
            .band_values(&[1u8, 1, 0, 1, 2, 2, 0, 2, 2])
            .nodata(255u8)
    }

    fn build_polygonize_test_raster() -> arrow_array::StructArray {
        let tmp = tempdir().unwrap();
        let path = tmp.path().join("polygonize.tif");
        let path_str = path.to_string_lossy().to_string();

        with_gdal(|gdal| {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u8>(&path_str, 3, 3, 1)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
                .unwrap();
            let band = dataset.rasterband(1).unwrap();
            band.set_no_data_value(Some(255.0)).unwrap();
            let mut buffer = Buffer::new((3, 3), vec![1u8, 1, 0, 1, 2, 2, 0, 2, 2]);
            band.write((0, 0), (3, 3), &mut buffer).unwrap();
            crate::utils::dataset_to_indb_raster(&dataset)
        })
        .unwrap()
    }

    #[test]
    fn test_rs_polygonize_udf_name() {
        assert_eq!(rs_polygonize_udf().name(), "rs_polygonize");
    }

    #[test]
    fn test_polygonize_raster() {
        let result = with_gdal(|gdal| {
            let provider = thread_local_provider(gdal).unwrap();
            let mem_driver = gdal.get_driver_by_name("Memory").unwrap();
            let raster_array = build_polygonize_test_raster();
            let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
            let raster = raster_struct.get(0).unwrap();
            let raster_ds = provider.raster_ref_to_gdal(&raster).unwrap();
            polygonize_raster(gdal, &mem_driver, raster_ds, 1)
        })
        .unwrap();

        assert!(
            !result.is_empty(),
            "Polygonize should return at least one polygon"
        );
        for (wkb, value) in &result {
            assert!(wkb.len() >= 5, "WKB should be at least 5 bytes");
            assert!(value.is_finite(), "Value should be a finite number");
        }
    }

    #[test]
    fn test_polygonize_kernel_return_type() {
        let kernel = RsPolygonize;
        let arg_types = vec![RASTER, SedonaType::Arrow(DataType::Int32)];
        let return_type = kernel.return_type(&arg_types).unwrap();
        assert!(matches!(
            return_type,
            Some(SedonaType::Arrow(DataType::List(_)))
        ));
    }

    #[test]
    fn test_polygonize_invoke_scalar() {
        let udf: ScalarUDF = rs_polygonize_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let result = tester
            .invoke_scalar_scalar(test_raster_spec(), ScalarValue::Int32(Some(1)))
            .unwrap();

        match result {
            ScalarValue::List(list) => {
                assert!(!list.is_empty());
            }
            other => panic!("Expected scalar list result, got {other:?}"),
        }
    }

    #[test]
    fn test_polygonize_invoke_array_and_nulls() {
        let udf: ScalarUDF = rs_polygonize_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let band_array = Arc::new(Int32Array::from(vec![Some(1), None]));
        let raster_array = ScalarValue::Struct(Arc::new(build_polygonize_test_raster()))
            .to_array_of_size(2)
            .unwrap();
        let result = tester
            .invoke_arrays(vec![raster_array, band_array])
            .unwrap();
        let list_array = as_list_array(&result).unwrap();

        assert_eq!(list_array.len(), 2);
        assert!(!list_array.is_null(0));
        assert!(list_array.is_null(1));

        let row0 = list_array.value(0);
        let row0 = as_struct_array(&row0).unwrap();
        assert!(!row0.is_empty());
    }

    #[test]
    fn test_polygonize_invalid_band_errors() {
        let udf: ScalarUDF = rs_polygonize_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        let err = tester
            .invoke_raster_array_scalar(
                vec![Some(test_raster_spec())],
                ScalarValue::Int32(Some(99)),
            )
            .expect_err("out-of-range band should error");

        assert!(err.to_string().contains("Band 99 is out of range"));
    }
}
