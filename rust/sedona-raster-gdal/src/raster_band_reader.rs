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

use datafusion_common::{exec_datafusion_err, exec_err, DataFusionError, Result};
use sedona_gdal::dataset::Dataset;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::raster::GdalDataType;
use sedona_gdal::raster::{GdalType, RasterBand};
use sedona_raster::traits::RasterRef;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::band_data_type_to_gdal;
use crate::gdal_dataset_provider::{thread_local_provider, RasterDataset};

pub(crate) struct RasterBandReader<'a> {
    gdal: &'a Gdal,
    raster: &'a dyn RasterRef,
    dataset: Option<RasterDataset<'a>>,
}

impl<'a> RasterBandReader<'a> {
    pub fn new(gdal: &'a Gdal, raster: &'a dyn RasterRef) -> Self {
        Self {
            gdal,
            raster,
            dataset: None,
        }
    }

    pub fn dataset(&mut self) -> Result<Option<&RasterDataset<'a>>> {
        if self.dataset.is_some() {
            return Ok(self.dataset.as_ref());
        }

        if self.raster_has_outdb()? {
            self.ensure_dataset()?;
        }

        Ok(self.dataset.as_ref())
    }

    #[allow(unused)]
    pub fn gdal_dataset(&mut self) -> Result<Option<&Dataset>> {
        Ok(self.dataset()?.map(|dataset| dataset.as_dataset()))
    }

    pub fn read_pixel_f64(&mut self, band_idx: usize, col: usize, row: usize) -> Result<f64> {
        let metadata = self.raster.metadata();
        let width = metadata.width() as usize;
        let height = metadata.height() as usize;

        if col >= width || row >= height {
            return exec_err!("Pixel coordinates out of bounds");
        }

        let pixel_idx = row * width + col;
        let (storage_type, data_type) = self.band_meta(band_idx)?;

        match storage_type {
            StorageType::InDb => {
                let band = self.band_ref(band_idx)?;
                let data = band.data();
                read_pixel_from_bytes(data, pixel_idx, &data_type)
            }
            StorageType::OutDbRef => {
                let meta = self.raster.metadata();
                let raster_w = meta.width() as usize;
                let raster_h = meta.height() as usize;

                let dataset = self.ensure_dataset()?;
                let gdal_band = dataset
                    .as_dataset()
                    .rasterband(band_idx)
                    .map_err(|e| exec_datafusion_err!("Failed to get band: {e}"))?;
                let (src_width, src_height) = gdal_band.size();
                if raster_w != src_width || raster_h != src_height {
                    return exec_err!(
                        "Out-db dataset size mismatch: raster=({raster_w},{raster_h}) dataset=({src_width},{src_height})"
                    );
                }

                let buffer = gdal_band
                    .read_as::<f64>((col as isize, row as isize), (1, 1), (1, 1), None)
                    .map_err(|e| exec_datafusion_err!("Failed to read pixel: {e}"))?;
                Ok(buffer.data()[0])
            }
        }
    }

    pub fn read_band_f64(&mut self, band_idx: usize) -> Result<Vec<f64>> {
        let metadata = self.raster.metadata();
        let width = metadata.width() as usize;
        let height = metadata.height() as usize;
        let pixel_count = width * height;

        let (storage_type, data_type) = self.band_meta(band_idx)?;

        match storage_type {
            StorageType::InDb => {
                let band = self.band_ref(band_idx)?;
                let data = band.data();
                let mut result = Vec::with_capacity(pixel_count);
                for idx in 0..pixel_count {
                    result.push(read_pixel_from_bytes(data, idx, &data_type)?);
                }
                Ok(result)
            }
            StorageType::OutDbRef => self.read_window_f64(band_idx, (0, 0), (width, height)),
        }
    }

    #[allow(dead_code)]
    pub fn read_window_f64(
        &mut self,
        band_idx: usize,
        offset: (usize, usize),
        size: (usize, usize),
    ) -> Result<Vec<f64>> {
        let (storage_type, data_type) = self.band_meta(band_idx)?;

        match storage_type {
            StorageType::InDb => {
                let metadata = self.raster.metadata();
                let width = metadata.width() as usize;
                let height = metadata.height() as usize;
                let (xoff, yoff) = offset;
                let (win_w, win_h) = size;

                if xoff + win_w > width || yoff + win_h > height {
                    return exec_err!("Window out of bounds");
                }

                let band = self.band_ref(band_idx)?;
                let data = band.data();
                let mut result = Vec::with_capacity(win_w * win_h);
                for row in 0..win_h {
                    let base = (yoff + row) * width + xoff;
                    for col in 0..win_w {
                        let idx = base + col;
                        result.push(read_pixel_from_bytes(data, idx, &data_type)?);
                    }
                }
                Ok(result)
            }
            StorageType::OutDbRef => {
                let meta = self.raster.metadata();
                let raster_w = meta.width() as usize;
                let raster_h = meta.height() as usize;

                let (xoff, yoff) = offset;
                let (win_w, win_h) = size;
                if xoff + win_w > raster_w || yoff + win_h > raster_h {
                    return exec_err!("Window out of bounds");
                }

                let dataset = self.ensure_dataset()?;
                let gdal_band = dataset
                    .as_dataset()
                    .rasterband(band_idx)
                    .map_err(|e| exec_datafusion_err!("Failed to get band: {e}"))?;
                let (src_width, src_height) = gdal_band.size();

                if raster_w != src_width || raster_h != src_height {
                    return exec_err!(
                        "Out-db dataset size mismatch: raster=({raster_w},{raster_h}) dataset=({src_width},{src_height})"
                    );
                }

                let buffer = gdal_band
                    .read_as::<f64>(
                        (xoff as isize, yoff as isize),
                        (win_w, win_h),
                        (win_w, win_h),
                        None,
                    )
                    .map_err(|e| exec_datafusion_err!("Failed to read window: {e}"))?;
                Ok(buffer.data().to_vec())
            }
        }
    }

    pub fn read_band_bytes(&mut self, band_idx: usize) -> Result<Vec<u8>> {
        let metadata = self.raster.metadata();
        let width = metadata.width() as usize;
        let height = metadata.height() as usize;
        let pixel_count = width * height;

        let (storage_type, data_type) = self.band_meta(band_idx)?;

        match storage_type {
            StorageType::InDb => {
                let band = self.band_ref(band_idx)?;
                Ok(band.data().to_vec())
            }
            StorageType::OutDbRef => {
                let dataset = self.ensure_dataset()?;
                let gdal_band = dataset
                    .as_dataset()
                    .rasterband(band_idx)
                    .map_err(|e| exec_datafusion_err!("Failed to get band: {e}"))?;
                read_band_bytes_from_gdal(&gdal_band, data_type, pixel_count)
            }
        }
    }

    #[allow(dead_code)]
    pub fn read_window_bytes(
        &mut self,
        band_idx: usize,
        offset: (usize, usize),
        size: (usize, usize),
    ) -> Result<Vec<u8>> {
        let (storage_type, data_type) = self.band_meta(band_idx)?;

        match storage_type {
            StorageType::InDb => {
                let metadata = self.raster.metadata();
                let width = metadata.width() as usize;
                let height = metadata.height() as usize;
                let (xoff, yoff) = offset;
                let (win_w, win_h) = size;

                if xoff + win_w > width || yoff + win_h > height {
                    return exec_err!("Window out of bounds");
                }

                let band = self.band_ref(band_idx)?;
                let data = band.data();
                let byte_size = band_data_type_size(&data_type);
                let mut result = Vec::with_capacity(win_w * win_h * byte_size);
                for row in 0..win_h {
                    let base = (yoff + row) * width + xoff;
                    for col in 0..win_w {
                        let idx = base + col;
                        let byte_offset = idx * byte_size;
                        result.extend_from_slice(&data[byte_offset..byte_offset + byte_size]);
                    }
                }
                Ok(result)
            }
            StorageType::OutDbRef => {
                let dataset = self.ensure_dataset()?;
                let gdal_band = dataset
                    .as_dataset()
                    .rasterband(band_idx)
                    .map_err(|e| exec_datafusion_err!("Failed to get band: {e}"))?;
                read_window_bytes_from_gdal(&gdal_band, data_type, offset, size)
            }
        }
    }

    fn ensure_dataset(&mut self) -> Result<&RasterDataset<'a>> {
        if self.dataset.is_none() {
            let provider = thread_local_provider(self.gdal)
                .map_err(|e| exec_datafusion_err!("Failed to init GDAL provider: {e}"))?;
            let dataset = provider
                .raster_ref_to_gdal(self.raster)
                .map_err(|e| exec_datafusion_err!("Failed to create GDAL dataset: {e}"))?;
            self.dataset = Some(dataset);
        }

        Ok(self.dataset.as_ref().expect("dataset should be set"))
    }

    fn raster_has_outdb(&self) -> Result<bool> {
        let bands = self.raster.bands();
        for idx in 1..=bands.len() {
            let band = bands
                .band(idx)
                .map_err(|e| exec_datafusion_err!("Failed to get band {}: {e}", idx))?;
            if band.metadata().storage_type()? == StorageType::OutDbRef {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn band_ref(&self, band_idx: usize) -> Result<Box<dyn sedona_raster::traits::BandRef + '_>> {
        let bands = self.raster.bands();
        if band_idx == 0 || band_idx > bands.len() {
            return exec_err!("Band {} is out of range (1-{})", band_idx, bands.len());
        }
        bands
            .band(band_idx)
            .map_err(|e| exec_datafusion_err!("Failed to get band: {e}"))
    }

    fn band_meta(&self, band_idx: usize) -> Result<(StorageType, BandDataType)> {
        let band = self.band_ref(band_idx)?;
        let meta = band.metadata();
        Ok((meta.storage_type()?, meta.data_type()?))
    }
}

fn band_data_type_size(data_type: &BandDataType) -> usize {
    match data_type {
        BandDataType::UInt8 => 1,
        BandDataType::Int8 => 1,
        BandDataType::UInt16 | BandDataType::Int16 => 2,
        BandDataType::UInt32 | BandDataType::Int32 | BandDataType::Float32 => 4,
        BandDataType::UInt64 | BandDataType::Int64 => 8,
        BandDataType::Float64 => 8,
    }
}

fn read_pixel_from_bytes(data: &[u8], offset: usize, data_type: &BandDataType) -> Result<f64> {
    let byte_size = band_data_type_size(data_type);
    let byte_offset = offset * byte_size;

    if byte_offset + byte_size > data.len() {
        return exec_err!("Pixel offset out of bounds");
    }

    let value = match data_type {
        BandDataType::UInt8 => data[byte_offset] as f64,
        BandDataType::Int8 => (data[byte_offset] as i8) as f64,
        BandDataType::UInt16 => {
            u16::from_le_bytes([data[byte_offset], data[byte_offset + 1]]) as f64
        }
        BandDataType::Int16 => {
            i16::from_le_bytes([data[byte_offset], data[byte_offset + 1]]) as f64
        }
        BandDataType::UInt32 => u32::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
        ]) as f64,
        BandDataType::Int32 => i32::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
        ]) as f64,
        BandDataType::UInt64 => u64::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
            data[byte_offset + 4],
            data[byte_offset + 5],
            data[byte_offset + 6],
            data[byte_offset + 7],
        ]) as f64,
        BandDataType::Int64 => i64::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
            data[byte_offset + 4],
            data[byte_offset + 5],
            data[byte_offset + 6],
            data[byte_offset + 7],
        ]) as f64,
        BandDataType::Float32 => f32::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
        ]) as f64,
        BandDataType::Float64 => f64::from_le_bytes([
            data[byte_offset],
            data[byte_offset + 1],
            data[byte_offset + 2],
            data[byte_offset + 3],
            data[byte_offset + 4],
            data[byte_offset + 5],
            data[byte_offset + 6],
            data[byte_offset + 7],
        ]),
    };

    Ok(value)
}

fn read_band_bytes_from_gdal(
    band: &RasterBand,
    data_type: BandDataType,
    pixel_count: usize,
) -> Result<Vec<u8>> {
    let gdal_type = band_data_type_to_gdal(&data_type);
    match gdal_type {
        GdalDataType::UInt8 => read_gdal_bytes::<u8>(band, pixel_count),
        GdalDataType::Int8 => read_gdal_bytes::<i8>(band, pixel_count),
        GdalDataType::UInt16 => read_gdal_bytes::<u16>(band, pixel_count),
        GdalDataType::Int16 => read_gdal_bytes::<i16>(band, pixel_count),
        GdalDataType::UInt32 => read_gdal_bytes::<u32>(band, pixel_count),
        GdalDataType::Int32 => read_gdal_bytes::<i32>(band, pixel_count),
        GdalDataType::UInt64 => read_gdal_bytes::<u64>(band, pixel_count),
        GdalDataType::Int64 => read_gdal_bytes::<i64>(band, pixel_count),
        GdalDataType::Float32 => read_gdal_bytes::<f32>(band, pixel_count),
        GdalDataType::Float64 => read_gdal_bytes::<f64>(band, pixel_count),
        _ => Err(DataFusionError::NotImplemented(
            "Unsupported GDAL data type".to_string(),
        )),
    }
}

#[allow(dead_code)]
fn read_window_bytes_from_gdal(
    band: &RasterBand,
    data_type: BandDataType,
    offset: (usize, usize),
    size: (usize, usize),
) -> Result<Vec<u8>> {
    let gdal_type = band_data_type_to_gdal(&data_type);
    match gdal_type {
        GdalDataType::UInt8 => read_gdal_window_bytes::<u8>(band, offset, size),
        GdalDataType::Int8 => read_gdal_window_bytes::<i8>(band, offset, size),
        GdalDataType::UInt16 => read_gdal_window_bytes::<u16>(band, offset, size),
        GdalDataType::Int16 => read_gdal_window_bytes::<i16>(band, offset, size),
        GdalDataType::UInt32 => read_gdal_window_bytes::<u32>(band, offset, size),
        GdalDataType::Int32 => read_gdal_window_bytes::<i32>(band, offset, size),
        GdalDataType::UInt64 => read_gdal_window_bytes::<u64>(band, offset, size),
        GdalDataType::Int64 => read_gdal_window_bytes::<i64>(band, offset, size),
        GdalDataType::Float32 => read_gdal_window_bytes::<f32>(band, offset, size),
        GdalDataType::Float64 => read_gdal_window_bytes::<f64>(band, offset, size),
        _ => Err(DataFusionError::NotImplemented(
            "Unsupported GDAL data type".to_string(),
        )),
    }
}

fn read_gdal_bytes<T: GdalType + ToLeBytes + Copy>(
    band: &RasterBand,
    pixel_count: usize,
) -> Result<Vec<u8>> {
    let (width, height) = band.size();
    let buffer = band
        .read_as::<T>((0, 0), (width, height), (width, height), None)
        .map_err(|e| exec_datafusion_err!("Failed to read band: {e}"))?;
    let values = buffer.data();
    let mut out = Vec::with_capacity(pixel_count * std::mem::size_of::<T>());
    for value in values.iter().take(pixel_count) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

#[allow(dead_code)]
fn read_gdal_window_bytes<T: GdalType + ToLeBytes + Copy>(
    band: &RasterBand,
    offset: (usize, usize),
    size: (usize, usize),
) -> Result<Vec<u8>> {
    let buffer = band
        .read_as::<T>(
            (offset.0 as isize, offset.1 as isize),
            (size.0, size.1),
            (size.0, size.1),
            None,
        )
        .map_err(|e| exec_datafusion_err!("Failed to read window: {e}"))?;
    let values = buffer.data();
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values.iter() {
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

trait ToLeBytes {
    fn to_le_bytes(&self) -> Vec<u8>;
}

impl ToLeBytes for u8 {
    fn to_le_bytes(&self) -> Vec<u8> {
        vec![*self]
    }
}

impl ToLeBytes for i8 {
    fn to_le_bytes(&self) -> Vec<u8> {
        vec![*self as u8]
    }
}

impl ToLeBytes for u16 {
    fn to_le_bytes(&self) -> Vec<u8> {
        u16::to_le_bytes(*self).to_vec()
    }
}

impl ToLeBytes for i16 {
    fn to_le_bytes(&self) -> Vec<u8> {
        i16::to_le_bytes(*self).to_vec()
    }
}

impl ToLeBytes for u32 {
    fn to_le_bytes(&self) -> Vec<u8> {
        u32::to_le_bytes(*self).to_vec()
    }
}

impl ToLeBytes for i32 {
    fn to_le_bytes(&self) -> Vec<u8> {
        i32::to_le_bytes(*self).to_vec()
    }
}

impl ToLeBytes for u64 {
    fn to_le_bytes(&self) -> Vec<u8> {
        u64::to_le_bytes(*self).to_vec()
    }
}

impl ToLeBytes for i64 {
    fn to_le_bytes(&self) -> Vec<u8> {
        i64::to_le_bytes(*self).to_vec()
    }
}

impl ToLeBytes for f32 {
    fn to_le_bytes(&self) -> Vec<u8> {
        f32::to_le_bytes(*self).to_vec()
    }
}

impl ToLeBytes for f64 {
    fn to_le_bytes(&self) -> Vec<u8> {
        f64::to_le_bytes(*self).to_vec()
    }
}
