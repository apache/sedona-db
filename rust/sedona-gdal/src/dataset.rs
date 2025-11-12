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

use arrow_schema::ArrowError;
use gdal::raster::GdalDataType;
use gdal::Dataset;
use gdal::Metadata;
use sedona_schema::raster::BandDataType;

/// Extract geotransform components from a GDAL dataset
/// Returns (upper_left_x, upper_left_y, scale_x, scale_y, skew_x, skew_y)
pub fn geotransform_components(
    dataset: &Dataset,
) -> Result<(f64, f64, f64, f64, f64, f64), ArrowError> {
    let geotransform = dataset
        .geo_transform()
        .map_err(|e| ArrowError::ParseError(format!("Failed to get geotransform: {e}")))?;
    Ok((
        geotransform[0], // Upper-left X coordinate
        geotransform[3], // Upper-left Y coordinate
        geotransform[1], // Pixel width (scale_x)
        geotransform[5], // Pixel height (scale_y, usually negative)
        geotransform[2], // X-direction skew
        geotransform[4], // Y-direction skew
    ))
}

/// Extract tile size from a GDAL dataset
/// If not provided, defaults to raster size. In future, will consider
/// defaulting to an ideal tile size instead of full raster size once we know
/// what the idea tile size should be.
pub fn tile_size(dataset: &Dataset) -> Result<(usize, usize), ArrowError> {
    let raster_width = dataset.raster_size().0;
    let raster_height = dataset.raster_size().1;

    let tile_width = match dataset.metadata_item("TILEWIDTH", "") {
        Some(val) => val.parse::<usize>().unwrap_or(raster_width),
        None => raster_width,
    };
    let tile_height = match dataset.metadata_item("TILEHEIGHT", "") {
        Some(val) => val.parse::<usize>().unwrap_or(raster_height),
        None => raster_height,
    };

    Ok((tile_width, tile_height))
}

pub fn to_banddatatype(gdal_data_type: GdalDataType) -> Result<BandDataType, ArrowError> {
    match gdal_data_type {
        GdalDataType::UInt8 => Ok(BandDataType::UInt8),
        GdalDataType::UInt16 => Ok(BandDataType::UInt16),
        GdalDataType::Int16 => Ok(BandDataType::Int16),
        GdalDataType::UInt32 => Ok(BandDataType::UInt32),
        GdalDataType::Int32 => Ok(BandDataType::Int32),
        GdalDataType::Float32 => Ok(BandDataType::Float32),
        GdalDataType::Float64 => Ok(BandDataType::Float64),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "Unsupported GDAL data type: {:?}",
            gdal_data_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gdal::DriverManager;

    #[test]
    fn test_geotransform_components() -> Result<(), ArrowError> {
        let driver = DriverManager::get_driver_by_name("MEM")
            .map_err(|e| ArrowError::ParseError(format!("Failed to get MEM driver: {e}")))?;
        let mut dataset = driver
            .create_with_band_type::<u8, _>("", 100, 100, 1)
            .map_err(|e| ArrowError::ParseError(format!("Failed to create dataset: {e}")))?;

        let (upper_left_x, upper_left_y, pixel_width, pixel_height, rotation_x, rotation_y) =
            (10.0, 20.0, 1.5, -2.5, 0.1, 0.2);

        // Add some basic georeferencing
        dataset
            .set_geo_transform(&[
                upper_left_x,
                pixel_width,
                rotation_x,
                upper_left_y,
                rotation_y,
                pixel_height,
            ])
            .map_err(|e| ArrowError::ParseError(format!("Failed to set geotransform: {e}")))?;

        let (ulx, uly, sx, sy, rx, ry) = geotransform_components(&dataset)?;
        assert_eq!(ulx, upper_left_x);
        assert_eq!(uly, upper_left_y);
        assert_eq!(sx, pixel_width);
        assert_eq!(sy, pixel_height);
        assert_eq!(rx, rotation_x);
        assert_eq!(ry, rotation_y);
        Ok(())
    }
}
