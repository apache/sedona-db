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
use arrow_array::StructArray;
use datafusion_common::Result;
use fastrand::Rng;
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_schema::crs::lnglat;
use sedona_schema::raster::BandDataType;

/// Generate a StructArray of rasters with sequentially increasing dimensions and pixel values.
/// These tiny rasters provide fast, easy and predictable test data for unit tests.
pub fn generate_test_rasters(
    count: usize,
    null_raster_index: Option<usize>,
) -> Result<StructArray> {
    let mut builder = RasterBuilder::new(count);
    let crs = lnglat().unwrap().to_crs_string();
    for i in 0..count {
        if matches!(null_raster_index, Some(index) if index == i) {
            builder.append_null()?;
            continue;
        }

        let width = i as u64 + 1;
        let height = i as u64 + 2;
        builder.start_raster_2d(
            width,
            height,
            i as f64 + 1.0,       // origin_x
            i as f64 + 2.0,       // origin_y
            i.max(1) as f64 * 0.1, // scale_x
            i.max(1) as f64 * -0.2, // scale_y
            i as f64 * 0.03,      // skew_x
            i as f64 * 0.04,      // skew_y
            Some(&crs),
        )?;
        builder.start_band_2d(BandDataType::UInt16, Some(&[0u8, 0u8]))?;

        let pixel_count = (i + 1) * (i + 2); // width * height
        let mut band_data = Vec::with_capacity(pixel_count * 2);
        for pixel_value in 0..pixel_count as u16 {
            band_data.extend_from_slice(&pixel_value.to_le_bytes());
        }

        builder.band_data_writer().append_value(&band_data);
        builder.finish_band()?;
        builder.finish_raster()?;
    }

    Ok(builder.finish()?)
}

/// Generates a set of tiled rasters arranged in a grid.
/// Each raster has 3 bands (RGB) with random pixel values.
pub fn generate_tiled_rasters(
    tile_size: (usize, usize),
    number_of_tiles: (usize, usize),
    data_type: BandDataType,
    seed: Option<u64>,
) -> Result<StructArray> {
    let mut rng = match seed {
        Some(s) => Rng::with_seed(s),
        None => Rng::new(),
    };
    let (tile_width, tile_height) = tile_size;
    let (x_tiles, y_tiles) = number_of_tiles;
    let mut raster_builder = RasterBuilder::new(x_tiles * y_tiles);
    let band_count = 3;
    let crs = lnglat().unwrap().to_crs_string();

    for tile_y in 0..y_tiles {
        for tile_x in 0..x_tiles {
            let origin_x = (tile_x * tile_width) as f64;
            let origin_y = (tile_y * tile_height) as f64;

            raster_builder.start_raster_2d(
                tile_width as u64,
                tile_height as u64,
                origin_x,
                origin_y,
                1.0,
                1.0,
                0.0,
                0.0,
                Some(&crs),
            )?;

            for _ in 0..band_count {
                let nodata_value = get_nodata_value_for_type(&data_type);
                let nodata_value_bytes = nodata_value.clone();

                raster_builder.start_band_2d(data_type, nodata_value.as_deref())?;

                let pixel_count = tile_width * tile_height;
                let corner_position =
                    get_corner_position(tile_x, tile_y, x_tiles, y_tiles, tile_width, tile_height);
                let band_data = generate_random_band_data(
                    pixel_count,
                    &data_type,
                    nodata_value_bytes.as_deref(),
                    corner_position,
                    &mut rng,
                );

                raster_builder.band_data_writer().append_value(&band_data);
                raster_builder.finish_band()?;
            }

            raster_builder.finish_raster()?;
        }
    }

    Ok(raster_builder.finish()?)
}

/// Builds a 1x1 single-band raster with a non-invertible geotransform (zero scales and skews).
pub fn build_noninvertible_raster() -> StructArray {
    let mut builder = RasterBuilder::new(1);
    let crs = lnglat().unwrap().to_crs_string();
    builder
        .start_raster_2d(1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Some(&crs))
        .expect("start raster");
    builder
        .start_band_2d(BandDataType::UInt8, None)
        .expect("start band");
    builder.band_data_writer().append_value([0u8]);
    builder.finish_band().expect("finish band");
    builder.finish_raster().expect("finish raster");
    builder.finish().expect("finish")
}

/// Builds a single-band raster from raw bytes for tests.
pub fn raster_from_single_band(
    width: usize,
    height: usize,
    data_type: BandDataType,
    band_bytes: &[u8],
    crs: Option<&str>,
) -> StructArray {
    let mut builder = RasterBuilder::new(1);
    builder
        .start_raster_2d(
            width as u64,
            height as u64,
            0.0,
            0.0,
            1.0,
            -1.0,
            0.0,
            0.0,
            crs,
        )
        .expect("start raster");
    builder
        .start_band_2d(data_type, None)
        .expect("start band");
    builder.band_data_writer().append_value(band_bytes);
    builder.finish_band().expect("finish band");
    builder.finish_raster().expect("finish raster");
    builder.finish().expect("finish")
}

/// Builds a single raster with 3 bands of different types for testing multi-band operations.
pub fn generate_multi_band_raster() -> StructArray {
    let mut builder = RasterBuilder::new(1);
    let crs = lnglat().unwrap().to_crs_string();
    builder
        .start_raster_2d(2, 2, 10.0, 20.0, 0.5, -0.5, 0.0, 0.0, Some(&crs))
        .unwrap();

    // Band 1: UInt8, nodata=255
    builder
        .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
        .unwrap();
    builder
        .band_data_writer()
        .append_value(&[1u8, 2u8, 3u8, 4u8]);
    builder.finish_band().unwrap();

    // Band 2: UInt16, nodata=0
    builder
        .start_band_2d(BandDataType::UInt16, Some(&[0u8, 0u8]))
        .unwrap();
    let band2_data: Vec<u8> = [100u16, 200u16, 300u16, 400u16]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    builder.band_data_writer().append_value(&band2_data);
    builder.finish_band().unwrap();

    // Band 3: Float32, no nodata
    builder
        .start_band_2d(BandDataType::Float32, None)
        .unwrap();
    let band3_data: Vec<u8> = [1.5f32, 2.5f32, 3.5f32, 4.5f32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    builder.band_data_writer().append_value(&band3_data);
    builder.finish_band().unwrap();

    builder.finish_raster().unwrap();
    builder.finish().unwrap()
}

fn get_corner_position(
    tile_x: usize,
    tile_y: usize,
    x_tiles: usize,
    y_tiles: usize,
    tile_width: usize,
    tile_height: usize,
) -> Option<usize> {
    if tile_x == 0 && tile_y == 0 {
        return Some(0);
    }
    if tile_x == x_tiles - 1 && tile_y == 0 {
        return Some(tile_width - 1);
    }
    if tile_x == 0 && tile_y == y_tiles - 1 {
        return Some((tile_height - 1) * tile_width);
    }
    if tile_x == x_tiles - 1 && tile_y == y_tiles - 1 {
        return Some(tile_height * tile_width - 1);
    }
    None
}

fn generate_random_band_data(
    pixel_count: usize,
    data_type: &BandDataType,
    nodata_bytes: Option<&[u8]>,
    corner_position: Option<usize>,
    rng: &mut Rng,
) -> Vec<u8> {
    macro_rules! gen_band {
        ($byte_size:expr, $rng_expr:expr) => {{
            let byte_size: usize = $byte_size;
            let mut data = Vec::with_capacity(pixel_count * byte_size);
            for _ in 0..pixel_count {
                data.extend_from_slice(&$rng_expr.to_ne_bytes());
            }
            if let (Some(nodata), Some(pos)) = (nodata_bytes, corner_position) {
                if nodata.len() >= byte_size && pos * byte_size + byte_size <= data.len() {
                    data[pos * byte_size..(pos * byte_size) + byte_size]
                        .copy_from_slice(&nodata[0..byte_size]);
                }
            }
            data
        }};
    }

    match data_type {
        BandDataType::UInt8 => gen_band!(1, rng.u8(..)),
        BandDataType::Int8 => gen_band!(1, rng.i8(..)),
        BandDataType::UInt16 => gen_band!(2, rng.u16(..)),
        BandDataType::Int16 => gen_band!(2, rng.i16(..)),
        BandDataType::UInt32 => gen_band!(4, rng.u32(..)),
        BandDataType::Int32 => gen_band!(4, rng.i32(..)),
        BandDataType::UInt64 => gen_band!(8, rng.u64(..)),
        BandDataType::Int64 => gen_band!(8, rng.i64(..)),
        BandDataType::Float32 => gen_band!(4, rng.f32()),
        BandDataType::Float64 => gen_band!(8, rng.f64()),
    }
}

fn get_nodata_value_for_type(data_type: &BandDataType) -> Option<Vec<u8>> {
    match data_type {
        BandDataType::UInt8 => Some(vec![255u8]),
        BandDataType::Int8 => Some(i8::MIN.to_ne_bytes().to_vec()),
        BandDataType::UInt16 => Some(u16::MAX.to_ne_bytes().to_vec()),
        BandDataType::Int16 => Some(i16::MIN.to_ne_bytes().to_vec()),
        BandDataType::UInt32 => Some(u32::MAX.to_ne_bytes().to_vec()),
        BandDataType::Int32 => Some(i32::MIN.to_ne_bytes().to_vec()),
        BandDataType::UInt64 => Some(u64::MAX.to_ne_bytes().to_vec()),
        BandDataType::Int64 => Some(i64::MIN.to_ne_bytes().to_vec()),
        BandDataType::Float32 => Some(f32::NAN.to_ne_bytes().to_vec()),
        BandDataType::Float64 => Some(f64::NAN.to_ne_bytes().to_vec()),
    }
}

/// Compare two RasterStructArrays for equality.
pub fn assert_raster_arrays_equal(
    raster_array1: &RasterStructArray,
    raster_array2: &RasterStructArray,
) {
    assert_eq!(
        raster_array1.len(),
        raster_array2.len(),
        "Raster array lengths do not match"
    );

    for i in 0..raster_array1.len() {
        let raster1 = raster_array1.get(i).unwrap();
        let raster2 = raster_array2.get(i).unwrap();
        assert_raster_equal(&raster1, &raster2);
    }
}

/// Compare two rasters for equality.
pub fn assert_raster_equal(raster1: &impl RasterRef, raster2: &impl RasterRef) {
    assert_eq!(
        raster1.width(),
        raster2.width(),
        "Raster widths do not match"
    );
    assert_eq!(
        raster1.height(),
        raster2.height(),
        "Raster heights do not match"
    );
    assert_eq!(
        raster1.transform(),
        raster2.transform(),
        "Raster transforms do not match"
    );
    assert_eq!(
        raster1.x_dim(),
        raster2.x_dim(),
        "Raster x_dim does not match"
    );
    assert_eq!(
        raster1.y_dim(),
        raster2.y_dim(),
        "Raster y_dim does not match"
    );
    assert_eq!(
        raster1.num_bands(),
        raster2.num_bands(),
        "Number of bands do not match"
    );

    for band_index in 0..raster1.num_bands() {
        let band1 = raster1
            .band(band_index)
            .unwrap_or_else(|| panic!("Band {band_index} missing from raster1"));
        let band2 = raster2
            .band(band_index)
            .unwrap_or_else(|| panic!("Band {band_index} missing from raster2"));

        assert_eq!(
            band1.data_type(),
            band2.data_type(),
            "Band {band_index} data types do not match"
        );
        assert_eq!(
            band1.nodata(),
            band2.nodata(),
            "Band {band_index} nodata values do not match"
        );
        assert_eq!(
            band1.dim_names(),
            band2.dim_names(),
            "Band {band_index} dim_names do not match"
        );
        assert_eq!(
            band1.shape(),
            band2.shape(),
            "Band {band_index} shapes do not match"
        );
        assert_eq!(
            band1.contiguous_data().unwrap().as_ref(),
            band2.contiguous_data().unwrap().as_ref(),
            "Band {band_index} data does not match"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;

    #[test]
    fn test_generate_test_rasters() {
        let count = 5;
        let struct_array = generate_test_rasters(count, None).unwrap();
        let raster_array = RasterStructArray::new(&struct_array);
        assert_eq!(raster_array.len(), count);

        for i in 0..count {
            let raster = raster_array.get(i).unwrap();
            assert_eq!(raster.width(), Some(i as u64 + 1));
            assert_eq!(raster.height(), Some(i as u64 + 2));

            let t = raster.transform();
            assert_eq!(t[0], i as f64 + 1.0); // origin_x
            assert_eq!(t[3], i as f64 + 2.0); // origin_y
            assert_eq!(t[1], (i.max(1) as f64) * 0.1); // scale_x
            assert_eq!(t[5], (i.max(1) as f64) * -0.2); // scale_y
            assert_eq!(t[2], (i as f64) * 0.03); // skew_x
            assert_eq!(t[4], (i as f64) * 0.04); // skew_y

            assert_eq!(raster.num_bands(), 1);
            let band = raster.band(0).unwrap();
            assert_eq!(band.data_type(), BandDataType::UInt16);
            assert_eq!(band.nodata(), Some(&[0u8, 0u8][..]));

            let band_data = band.contiguous_data().unwrap();
            let expected_pixel_count = (i + 1) * (i + 2);
            let mut actual_pixel_values = Vec::new();
            for chunk in band_data.chunks_exact(2) {
                let value = u16::from_le_bytes([chunk[0], chunk[1]]);
                actual_pixel_values.push(value);
            }
            let expected_pixel_values: Vec<u16> = (0..expected_pixel_count as u16).collect();
            assert_eq!(actual_pixel_values, expected_pixel_values);
        }
    }

    #[test]
    fn test_generate_tiled_rasters() {
        let tile_size = (64, 64);
        let number_of_tiles = (4, 4);
        let data_type = BandDataType::UInt8;
        let struct_array =
            generate_tiled_rasters(tile_size, number_of_tiles, data_type, Some(43)).unwrap();
        let raster_array = RasterStructArray::new(&struct_array);
        assert_eq!(raster_array.len(), 16);
        for i in 0..16 {
            let raster = raster_array.get(i).unwrap();
            assert_eq!(raster.width(), Some(64));
            assert_eq!(raster.height(), Some(64));
            let t = raster.transform();
            assert_eq!(t[0], ((i % 4) * 64) as f64); // origin_x
            assert_eq!(t[3], ((i / 4) * 64) as f64); // origin_y
            assert_eq!(raster.num_bands(), 3);
            for band_index in 0..3 {
                let band = raster.band(band_index).unwrap();
                assert_eq!(band.data_type(), BandDataType::UInt8);
                assert_eq!(band.contiguous_data().unwrap().len(), 64 * 64);
            }
        }
    }

    #[test]
    fn test_generate_multi_band_raster() {
        let struct_array = generate_multi_band_raster();
        let raster_array = RasterStructArray::new(&struct_array);
        assert_eq!(raster_array.len(), 1);

        let raster = raster_array.get(0).unwrap();
        assert_eq!(raster.width(), Some(2));
        assert_eq!(raster.height(), Some(2));
        assert_eq!(raster.num_bands(), 3);

        let b1 = raster.band(0).unwrap();
        assert_eq!(b1.data_type(), BandDataType::UInt8);
        assert_eq!(b1.nodata(), Some(&[255u8][..]));
        assert_eq!(b1.contiguous_data().unwrap().as_ref(), &[1u8, 2, 3, 4]);

        let b2 = raster.band(1).unwrap();
        assert_eq!(b2.data_type(), BandDataType::UInt16);
        assert_eq!(b2.nodata(), Some(&[0u8, 0][..]));

        let b3 = raster.band(2).unwrap();
        assert_eq!(b3.data_type(), BandDataType::Float32);
        assert_eq!(b3.nodata(), None);
    }

    #[test]
    fn test_raster_arrays_equal() {
        let raster_array1 = generate_test_rasters(3, None).unwrap();
        let raster_struct_array1 = RasterStructArray::new(&raster_array1);
        assert_raster_arrays_equal(&raster_struct_array1, &raster_struct_array1);
    }

    #[test]
    #[should_panic = "Raster array lengths do not match"]
    fn test_raster_arrays_not_equal() {
        let raster_array1 = generate_test_rasters(3, None).unwrap();
        let raster_struct_array1 = RasterStructArray::new(&raster_array1);
        let raster_array2 = generate_test_rasters(4, None).unwrap();
        let raster_struct_array2 = RasterStructArray::new(&raster_array2);
        assert_raster_arrays_equal(&raster_struct_array1, &raster_struct_array2);
    }

    #[test]
    fn test_raster_equal() {
        let raster_array1 =
            generate_tiled_rasters((256, 256), (1, 1), BandDataType::UInt8, Some(43)).unwrap();
        let rsa = RasterStructArray::new(&raster_array1);
        let raster1 = rsa.get(0).unwrap();
        assert_raster_equal(&raster1, &raster1);
    }

    #[test]
    #[should_panic = "Band 0 data does not match"]
    fn test_raster_different_band_data() {
        let raster_array1 =
            generate_tiled_rasters((128, 128), (1, 1), BandDataType::UInt8, Some(43)).unwrap();
        let raster_array2 =
            generate_tiled_rasters((128, 128), (1, 1), BandDataType::UInt8, Some(47)).unwrap();
        let rsa1 = RasterStructArray::new(&raster_array1);
        let rsa2 = RasterStructArray::new(&raster_array2);
        let raster1 = rsa1.get(0).unwrap();
        let raster2 = rsa2.get(0).unwrap();
        assert_raster_equal(&raster1, &raster2);
    }

    #[test]
    #[should_panic = "Raster transforms do not match"]
    fn test_raster_different_metadata() {
        let raster_array =
            generate_tiled_rasters((128, 128), (2, 1), BandDataType::UInt8, Some(43)).unwrap();
        let rsa = RasterStructArray::new(&raster_array);
        let raster1 = rsa.get(0).unwrap();
        let raster2 = rsa.get(1).unwrap();
        assert_raster_equal(&raster1, &raster2);
    }
}
