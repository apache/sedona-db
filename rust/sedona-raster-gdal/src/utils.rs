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

//! Utility functions for loading raster data via GDAL.

use arrow_array::StructArray;
use datafusion_common::error::Result;
use datafusion_common::exec_datafusion_err;
use sedona_gdal::dataset::Dataset;
use sedona_gdal::spatial_ref::SpatialRef;

use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::BandMetadata;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::{gdal_to_band_data_type, RasterMetadataFromGdalGeoTransform};

/// Append a GDAL dataset as a single in-db raster to the provided [`RasterBuilder`].
pub fn append_as_indb_raster(dataset: &Dataset, builder: &mut RasterBuilder) -> Result<()> {
    let (width, height) = dataset.raster_size();

    let geotransform = dataset
        .geo_transform()
        .map_err(|e| exec_datafusion_err!("Failed to get geotransform: {}", e))?;

    let metadata = geotransform.to_raster_metadata(width, height);

    let crs = dataset
        .spatial_ref()
        .ok()
        .and_then(|sr: SpatialRef| sr.to_projjson().ok());

    builder
        .start_raster(&metadata, crs.as_deref())
        .map_err(|e| exec_datafusion_err!("Failed to start raster: {}", e))?;

    let band_count = dataset.raster_count();
    for band_idx in 1..=band_count {
        let band = dataset
            .rasterband(band_idx)
            .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

        let gdal_type = band.band_type();
        let band_data_type = gdal_to_band_data_type(gdal_type)
            .map_err(|_| exec_datafusion_err!("Unsupported band data type: {:?}", gdal_type))?;

        // Get nodata value
        let nodata_bytes = match band_data_type {
            BandDataType::UInt64 => band
                .no_data_value_u64()
                .map(|no_data| no_data.to_le_bytes().to_vec()),
            BandDataType::Int64 => band
                .no_data_value_i64()
                .map(|no_data| no_data.to_le_bytes().to_vec()),
            _ => band
                .no_data_value()
                .map(|no_data| crate::gdal_common::nodata_f64_to_bytes(no_data, &band_data_type)),
        };

        let band_metadata = BandMetadata {
            nodata_value: nodata_bytes,
            storage_type: StorageType::InDb,
            datatype: band_data_type,
            outdb_url: None,
            outdb_band_id: None,
        };

        builder
            .start_band(band_metadata)
            .map_err(|e| exec_datafusion_err!("Failed to start band: {}", e))?;

        let band_data = read_band_data(dataset, band_idx, width, height, band_data_type)?;
        builder.band_data_writer().append_value(&band_data);

        builder
            .finish_band()
            .map_err(|e| exec_datafusion_err!("Failed to finish band: {}", e))?;
    }

    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("Failed to finish raster: {}", e))?;

    Ok(())
}

/// Materialize a single GDAL dataset as an in-db raster `StructArray`.
pub fn dataset_to_indb_raster(dataset: &Dataset) -> Result<StructArray> {
    let mut builder = RasterBuilder::new(1);
    append_as_indb_raster(dataset, &mut builder)?;

    builder
        .finish()
        .map_err(|e| exec_datafusion_err!("Failed to build raster: {}", e))
}

/// Read band data as bytes from a GDAL dataset.
fn read_band_data(
    dataset: &Dataset,
    band_idx: usize,
    width: usize,
    height: usize,
    band_type: BandDataType,
) -> Result<Vec<u8>> {
    let band = dataset
        .rasterband(band_idx)
        .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

    // Read band data based on type
    macro_rules! read_as {
        ($T:ty) => {{
            let buffer = band
                .read_as::<$T>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }};
    }

    let data: Vec<u8> = match band_type {
        BandDataType::UInt8 => read_as!(u8),
        BandDataType::Int8 => read_as!(i8),
        BandDataType::UInt16 => read_as!(u16),
        BandDataType::Int16 => read_as!(i16),
        BandDataType::UInt32 => read_as!(u32),
        BandDataType::Int32 => read_as!(i32),
        BandDataType::UInt64 => read_as!(u64),
        BandDataType::Int64 => read_as!(i64),
        BandDataType::Float32 => read_as!(f32),
        BandDataType::Float64 => read_as!(f64),
    };

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::{append_as_indb_raster, dataset_to_indb_raster};

    use arrow_array::StructArray;
    use datafusion_common::exec_datafusion_err;
    use sedona_gdal::dataset::Dataset;
    use sedona_gdal::gdal::Gdal;
    use sedona_gdal::gdal_dyn_bindgen::{GDAL_OF_RASTER, GDAL_OF_READONLY};
    use sedona_gdal::raster::types::Buffer;
    use sedona_gdal::raster::types::DatasetOptions;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::raster::{BandDataType, StorageType};
    use tempfile::TempDir;

    use crate::gdal_common::with_gdal;

    fn open_dataset(gdal: &Gdal, path: &str) -> sedona_gdal::errors::Result<Dataset> {
        gdal.open_ex_with_options(
            path,
            DatasetOptions {
                open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY,
                ..Default::default()
            },
        )
    }

    fn load_as_indb_raster(gdal: &Gdal, path: &str) -> datafusion_common::Result<StructArray> {
        let dataset = open_dataset(gdal, path).map_err(crate::gdal_common::convert_gdal_err)?;
        dataset_to_indb_raster(&dataset)
    }

    fn write_uint64_tiff(gdal: &Gdal, path: &str, nodata: u64, data: Vec<u64>) {
        let driver = gdal.get_driver_by_name("GTiff").unwrap();
        let dataset = driver.create_with_band_type::<u64>(path, 2, 2, 1).unwrap();
        dataset
            .set_geo_transform(&[100.0, 2.0, 0.0, 200.0, 0.0, -2.0])
            .unwrap();
        dataset.set_projection("EPSG:4326").unwrap();
        let band = dataset.rasterband(1).unwrap();
        band.set_no_data_value_u64(Some(nodata)).unwrap();
        let mut buffer = Buffer::new((2, 2), data);
        band.write((0, 0), (2, 2), &mut buffer).unwrap();
    }

    fn write_int64_tiff(gdal: &Gdal, path: &str, nodata: i64, data: Vec<i64>) {
        let driver = gdal.get_driver_by_name("GTiff").unwrap();
        let dataset = driver.create_with_band_type::<i64>(path, 2, 2, 1).unwrap();
        dataset
            .set_geo_transform(&[10.0, 1.0, 0.0, 20.0, 0.0, -1.0])
            .unwrap();
        let band = dataset.rasterband(1).unwrap();
        band.set_no_data_value_i64(Some(nodata)).unwrap();
        let mut buffer = Buffer::new((2, 2), data);
        band.write((0, 0), (2, 2), &mut buffer).unwrap();
    }

    fn write_byte_tiff(gdal: &Gdal, path: &str) {
        let driver = gdal.get_driver_by_name("GTiff").unwrap();
        let dataset = driver.create_with_band_type::<u8>(path, 3, 2, 1).unwrap();
        dataset
            .set_geo_transform(&[1.5, 0.25, 0.0, 4.5, 0.0, -0.25])
            .unwrap();
        dataset.set_projection("EPSG:4326").unwrap();
        let band = dataset.rasterband(1).unwrap();
        band.set_no_data_value(Some(255.0)).unwrap();
        let mut buffer = Buffer::new((3, 2), vec![1u8, 2, 3, 4, 5, 6]);
        band.write((0, 0), (3, 2), &mut buffer).unwrap();
    }

    fn write_multi_band_tiff(gdal: &Gdal, path: &str) {
        let driver = gdal.get_driver_by_name("GTiff").unwrap();
        let dataset = driver.create(path, 2, 2, 2).unwrap();
        dataset
            .set_geo_transform(&[10.0, 1.0, 0.0, 20.0, 0.0, -1.0])
            .unwrap();

        let band1 = dataset.rasterband(1).unwrap();
        // GeoTIFF stores a single dataset-level nodata value, so use the same nodata
        // for both bands in this fixture to keep the assertions format-accurate.
        band1.set_no_data_value(Some(255.0)).unwrap();
        let mut buffer1 = Buffer::new((2, 2), vec![10u8, 11, 12, 13]);
        band1.write((0, 0), (2, 2), &mut buffer1).unwrap();

        let band2 = dataset.rasterband(2).unwrap();
        band2.set_no_data_value(Some(255.0)).unwrap();
        let mut buffer2 = Buffer::new((2, 2), vec![100u8, 0, 200, 0]);
        band2.write((0, 0), (2, 2), &mut buffer2).unwrap();
    }

    fn build_multi_band_mem_dataset(gdal: &Gdal) -> Dataset {
        let driver = gdal.get_driver_by_name("MEM").unwrap();
        let dataset = driver.create("", 2, 2, 2).unwrap();
        dataset
            .set_geo_transform(&[10.0, 1.0, 0.0, 20.0, 0.0, -1.0])
            .unwrap();
        dataset.set_projection("EPSG:4326").unwrap();

        let band1 = dataset.rasterband(1).unwrap();
        band1.set_no_data_value(Some(0.0)).unwrap();
        let mut buffer1 = Buffer::new((2, 2), vec![10u8, 11, 12, 13]);
        band1.write((0, 0), (2, 2), &mut buffer1).unwrap();

        let band2 = dataset.rasterband(2).unwrap();
        band2.set_no_data_value(Some(255.0)).unwrap();
        let mut buffer2 = Buffer::new((2, 2), vec![100u8, 0, 200, 0]);
        band2.write((0, 0), (2, 2), &mut buffer2).unwrap();

        dataset
    }

    #[test]
    fn dataset_to_indb_raster_reads_single_band_geotiff() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("byte.tif");
        let path_str = path.to_string_lossy().to_string();

        with_gdal(|gdal| {
            write_byte_tiff(gdal, &path_str);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();

        let raster_array = with_gdal(|gdal| load_as_indb_raster(gdal, &path_str)).unwrap();
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let band = raster.bands().band(1).unwrap();

        assert_eq!(raster.metadata().width(), 3);
        assert_eq!(raster.metadata().height(), 2);
        assert_eq!(raster.metadata().upper_left_x(), 1.5);
        assert_eq!(raster.metadata().upper_left_y(), 4.5);
        assert!(raster.crs().is_some());
        assert_eq!(band.metadata().storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band.metadata().data_type().unwrap(), BandDataType::UInt8);
        assert_eq!(band.metadata().nodata_value().unwrap(), [255u8]);
        assert_eq!(band.data(), [1u8, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn dataset_to_indb_raster_preserves_uint64_nodata_and_data() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("uint64.tif");
        let path_str = path.to_string_lossy().to_string();
        let nodata = 9_007_199_254_740_993u64;

        with_gdal(|gdal| {
            write_uint64_tiff(gdal, &path_str, nodata, vec![1, 2, 3, 4]);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();

        let raster_array = with_gdal(|gdal| load_as_indb_raster(gdal, &path_str)).unwrap();
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let band = raster.bands().band(1).unwrap();

        assert_eq!(raster.metadata().width(), 2);
        assert_eq!(raster.metadata().height(), 2);
        assert_eq!(raster.metadata().upper_left_x(), 100.0);
        assert_eq!(raster.metadata().upper_left_y(), 200.0);
        assert_eq!(band.metadata().data_type().unwrap(), BandDataType::UInt64);
        assert_eq!(
            band.metadata().nodata_value().unwrap(),
            &nodata.to_le_bytes()
        );

        let pixels: Vec<u64> = band
            .data()
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(pixels, vec![1, 2, 3, 4]);
    }

    #[test]
    fn dataset_to_indb_raster_preserves_int64_nodata_and_data() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("int64.tif");
        let path_str = path.to_string_lossy().to_string();
        let nodata = -9_007_199_254_740_993i64;

        with_gdal(|gdal| {
            write_int64_tiff(gdal, &path_str, nodata, vec![-1, -2, -3, -4]);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();

        let raster_array = with_gdal(|gdal| load_as_indb_raster(gdal, &path_str)).unwrap();
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let band = raster.bands().band(1).unwrap();

        assert_eq!(band.metadata().data_type().unwrap(), BandDataType::Int64);
        assert_eq!(
            band.metadata().nodata_value().unwrap(),
            &nodata.to_le_bytes()
        );

        let pixels: Vec<i64> = band
            .data()
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(pixels, vec![-1, -2, -3, -4]);
    }

    #[test]
    fn dataset_to_indb_raster_preserves_multi_band_data_and_nodata() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("multi.tif");
        let path_str = path.to_string_lossy().to_string();

        with_gdal(|gdal| {
            write_multi_band_tiff(gdal, &path_str);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();

        let raster_array = with_gdal(|gdal| load_as_indb_raster(gdal, &path_str)).unwrap();
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let band1 = raster.bands().band(1).unwrap();
        let band2 = raster.bands().band(2).unwrap();

        assert_eq!(raster.bands().len(), 2);
        assert_eq!(band1.metadata().storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band1.metadata().data_type().unwrap(), BandDataType::UInt8);
        assert_eq!(band1.metadata().nodata_value().unwrap(), [255u8]);
        assert_eq!(band1.data(), [10u8, 11, 12, 13]);

        assert_eq!(band2.metadata().storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band2.metadata().data_type().unwrap(), BandDataType::UInt8);
        assert_eq!(band2.metadata().nodata_value().unwrap(), [255u8]);
        assert_eq!(band2.data(), [100u8, 0, 200, 0]);
    }

    #[test]
    fn dataset_to_indb_raster_preserves_per_band_nodata_for_mem_dataset() {
        let raster_array = with_gdal(|gdal| {
            let dataset = build_multi_band_mem_dataset(gdal);
            dataset_to_indb_raster(&dataset)
        })
        .unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let band1 = raster.bands().band(1).unwrap();
        let band2 = raster.bands().band(2).unwrap();

        assert_eq!(raster.bands().len(), 2);
        assert_eq!(band1.metadata().storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band1.metadata().data_type().unwrap(), BandDataType::UInt8);
        assert_eq!(band1.metadata().nodata_value().unwrap(), [0u8]);
        assert_eq!(band1.data(), [10u8, 11, 12, 13]);

        assert_eq!(band2.metadata().storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band2.metadata().data_type().unwrap(), BandDataType::UInt8);
        assert_eq!(band2.metadata().nodata_value().unwrap(), [255u8]);
        assert_eq!(band2.data(), [100u8, 0, 200, 0]);
    }

    #[test]
    fn append_as_indb_raster_appends_multiple_rasters() {
        let temp_dir = TempDir::new().unwrap();
        let byte_path = temp_dir.path().join("byte.tif");
        let byte_path_str = byte_path.to_string_lossy().to_string();
        let multi_path = temp_dir.path().join("multi.tif");
        let multi_path_str = multi_path.to_string_lossy().to_string();

        with_gdal(|gdal| {
            write_byte_tiff(gdal, &byte_path_str);
            write_multi_band_tiff(gdal, &multi_path_str);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();

        let raster_array = with_gdal(|gdal| {
            let byte_dataset =
                open_dataset(gdal, &byte_path_str).map_err(crate::gdal_common::convert_gdal_err)?;
            let multi_dataset = open_dataset(gdal, &multi_path_str)
                .map_err(crate::gdal_common::convert_gdal_err)?;

            let mut builder = RasterBuilder::new(2);
            append_as_indb_raster(&byte_dataset, &mut builder)?;
            append_as_indb_raster(&multi_dataset, &mut builder)?;
            builder
                .finish()
                .map_err(|e| exec_datafusion_err!("Failed to build raster array: {}", e))
        })
        .unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        assert_eq!(raster_struct.len(), 2);

        let first = raster_struct.get(0).unwrap();
        assert_eq!(first.metadata().width(), 3);
        assert_eq!(first.metadata().height(), 2);
        assert_eq!(first.bands().len(), 1);

        let second = raster_struct.get(1).unwrap();
        assert_eq!(second.metadata().width(), 2);
        assert_eq!(second.metadata().height(), 2);
        assert_eq!(second.bands().len(), 2);
    }
}
