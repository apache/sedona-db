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
use arrow_buffer::Buffer;
use datafusion_common::error::Result;
use datafusion_common::exec_datafusion_err;
use sedona_gdal::dataset::Dataset;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::gdal_dyn_bindgen::{GDAL_OF_RASTER, GDAL_OF_READONLY};
use sedona_gdal::geo_transform::GeoTransform;
use sedona_gdal::raster::types::DatasetOptions;
use sedona_gdal::raster::types::ResampleAlg;
use sedona_gdal::spatial_ref::SpatialRef;

use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata};
use sedona_schema::raster::StorageType;

use crate::gdal_common::{
    band_nodata_to_bytes, gdal_to_band_data_type, normalize_outdb_source_path, GdalBandLayout,
    RasterMetadataFromGdalGeoTransform,
};

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

        let nodata_bytes = band_nodata_to_bytes(&band)?;

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

        let band_data = band
            .read_as_bytes((0, 0), (width, height), (width, height), None)
            .map_err(|e| exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e))?;
        let band_data_len = u32::try_from(band_data.len())
            .map_err(|_| exec_datafusion_err!("Band {} data too large for Arrow view", band_idx))?;
        let block = builder
            .band_data_writer()
            .append_block(Buffer::from_vec(band_data));
        builder
            .band_data_writer()
            .try_append_view(block, 0, band_data_len)
            .map_err(|e| exec_datafusion_err!("Failed to append band {} data: {}", band_idx, e))?;

        builder
            .finish_band()
            .map_err(|e| exec_datafusion_err!("Failed to finish band: {}", e))?;
    }

    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("Failed to finish raster: {}", e))?;

    Ok(())
}

/// Append a raster source path as a single out-db raster to the provided [`RasterBuilder`].
pub fn append_as_outdb_raster(gdal: &Gdal, path: &str, builder: &mut RasterBuilder) -> Result<()> {
    let gdal_path = normalize_outdb_source_path(path);
    let dataset = gdal
        .open_ex_with_options(
            &gdal_path,
            DatasetOptions {
                open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY,
                ..Default::default()
            },
        )
        .map_err(|e| {
            exec_datafusion_err!(
                "Failed to open raster file '{}' (GDAL path '{}'): {}",
                path,
                gdal_path,
                e
            )
        })?;

    let (width, height) = dataset.raster_size();
    let geotransform = dataset
        .geo_transform()
        .map_err(|e| exec_datafusion_err!("Failed to get geotransform: {}", e))?;
    let metadata = geotransform.to_raster_metadata(width, height);

    let crs = dataset
        .spatial_ref()
        .ok()
        .and_then(|sr: SpatialRef| sr.to_projjson().ok());

    builder.start_raster(&metadata, crs.as_deref())?;

    let band_count = dataset.raster_count();
    for band_idx in 1..=band_count {
        let band = dataset
            .rasterband(band_idx)
            .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

        let gdal_type = band.band_type();
        let band_data_type = gdal_to_band_data_type(gdal_type)
            .map_err(|_| exec_datafusion_err!("Unsupported band data type: {:?}", gdal_type))?;

        let nodata_bytes = band_nodata_to_bytes(&band)?;

        let band_metadata = BandMetadata {
            nodata_value: nodata_bytes,
            storage_type: StorageType::OutDbRef,
            datatype: band_data_type,
            outdb_url: Some(path.to_string()),
            outdb_band_id: Some(band_idx as u32),
        };

        builder.start_band(band_metadata)?;
        builder.band_data_writer().append_value([]);
        builder.finish_band()?;
    }

    builder.finish_raster()?;
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

/// Append a GDAL dataset as a single **N-D** in-db raster, regrouping the flat
/// GDAL band list back into N-D bands per `layout`.
///
/// The inverse of the plane-stacking in `raster_ref_to_gdal_mem`: each source
/// band owns `plane_count` consecutive GDAL bands (band-major, plane-major), so
/// this consumes them in order and concatenates their bytes (C-order, planes
/// outermost) into one band. The spatial extent and geotransform come from the
/// (possibly transformed) `dataset`; the non-spatial structure comes from
/// `layout`.
pub fn append_nd_from_dataset(
    dataset: &Dataset,
    layout: &GdalBandLayout,
    builder: &mut RasterBuilder,
) -> Result<()> {
    let (width, height) = dataset.raster_size();

    let geotransform = dataset
        .geo_transform()
        .map_err(|e| exec_datafusion_err!("Failed to get geotransform: {}", e))?;
    let metadata = geotransform.to_raster_metadata(width, height);

    let crs = dataset
        .spatial_ref()
        .ok()
        .and_then(|sr: SpatialRef| sr.to_projjson().ok());

    append_nd_from_dataset_inner(dataset, layout, builder, &metadata, crs.as_deref(), None)
}

/// The output grid a resample writes into: geotransform, pixel dimensions, CRS,
/// and the resampling algorithm.
///
/// `crs` is carried through rather than read back from the source dataset so a
/// caller can preserve the exact source CRS string instead of round-tripping it
/// through GDAL.
pub struct ResampledGrid<'a> {
    pub transform: GeoTransform,
    pub width: usize,
    pub height: usize,
    pub crs: Option<&'a str>,
    pub alg: ResampleAlg,
}

/// Append a GDAL dataset as a single **N-D** in-db raster, resampling every band
/// to `grid`'s dimensions with `grid.alg`.
///
/// The spatial analog of [`append_nd_from_dataset`]: the full source window of
/// each GDAL band is read into a `grid.width` x `grid.height` buffer using
/// GDAL's RasterIO resampling, so band count/order and the non-spatial structure
/// in `layout` are preserved and only the trailing `(y, x)` extent changes.
pub fn append_resampled_nd_from_dataset(
    dataset: &Dataset,
    layout: &GdalBandLayout,
    builder: &mut RasterBuilder,
    grid: &ResampledGrid<'_>,
) -> Result<()> {
    let metadata = grid.transform.to_raster_metadata(grid.width, grid.height);
    append_nd_from_dataset_inner(
        dataset,
        layout,
        builder,
        &metadata,
        grid.crs,
        Some(grid.alg),
    )
}

/// The output grid a same-CRS regrid writes into: geotransform, pixel
/// dimensions, CRS, and the resampling algorithm.
///
/// Unlike [`ResampledGrid`], the regrid can move the output origin off the
/// source grid and grow the extent past the source footprint (a scale change
/// that does not tile evenly, or an origin snap). The CRS is unchanged from the
/// source — this path never reprojects — and is carried through verbatim rather
/// than round-tripping through GDAL.
pub struct RegridGrid<'a> {
    pub transform: GeoTransform,
    pub width: usize,
    pub height: usize,
    pub crs: Option<&'a str>,
    pub alg: ResampleAlg,
}

/// Regrid every band of `src_dataset` onto `grid` in the **same CRS**, appending
/// the result as a single **N-D** in-db raster regrouped per `layout`.
///
/// The output grid may sit off the source grid and extend past the source
/// footprint. Each output band buffer is pre-filled with the band's nodata value
/// (zero when a band has none); only the sub-window the source actually covers
/// (output pixels whose centre falls inside the source extent) is overwritten by
/// a fractional-window RasterIO read ([`RasterBand::read_regrid_into`]), so the
/// grown/shifted border reads back as nodata. The finished owned buffers are
/// moved into the Arrow output as view blocks rather than copied.
///
/// Both the source and output grids must be north-up (no skew): a skewed output
/// covers the source as a parallelogram that a single axis-aligned RasterIO
/// cannot capture. Skewed inputs take the extent-preserving dimension path
/// (which carries the skew through unchanged) instead; a skewed scale change or
/// grid snap is an explicit unsupported case and errors.
pub fn append_regridded_nd_from_dataset(
    src_dataset: &Dataset,
    layout: &GdalBandLayout,
    builder: &mut RasterBuilder,
    grid: &RegridGrid<'_>,
) -> Result<()> {
    let out_width = grid.width;
    let out_height = grid.height;

    let src_gt = src_dataset
        .geo_transform()
        .map_err(|e| exec_datafusion_err!("Failed to get source geotransform: {}", e))?;
    let (src_width, src_height) = src_dataset.raster_size();

    // North-up requirement (see the doc comment): skew terms must be zero on both
    // grids so each axis maps independently.
    if src_gt[2] != 0.0 || src_gt[4] != 0.0 || grid.transform[2] != 0.0 || grid.transform[4] != 0.0
    {
        return Err(exec_datafusion_err!(
            "RS_Resample: a scale change or grid snap on a skewed (non-north-up) raster is not \
             supported (only an extent-preserving width/height change is)"
        ));
    }

    // The output sub-window the source covers, per axis, plus the fractional
    // source window that maps it. Absent coverage on either axis, the whole
    // output is nodata.
    let cover_x = covered_axis(
        grid.transform[0],
        grid.transform[1],
        src_gt[0],
        src_gt[1],
        src_width,
        out_width,
    );
    let cover_y = covered_axis(
        grid.transform[3],
        grid.transform[5],
        src_gt[3],
        src_gt[5],
        src_height,
        out_height,
    );

    let metadata = grid.transform.to_raster_metadata(out_width, out_height);
    builder
        .start_raster(&metadata, grid.crs)
        .map_err(|e| exec_datafusion_err!("Failed to start raster: {}", e))?;

    let total_planes: usize = layout.bands.iter().map(|b| b.plane_count).sum();
    let gdal_band_count = src_dataset.raster_count();
    if gdal_band_count != total_planes {
        return Err(exec_datafusion_err!(
            "layout expects {total_planes} GDAL bands but dataset has {gdal_band_count}"
        ));
    }

    let mut gdal_band = 1;
    for plan in &layout.bands {
        let dim_names: Vec<&str> = plan.dim_names.iter().map(String::as_str).collect();
        // shape = [non-spatial..., height, width] — spatial from the output grid.
        let mut shape = plan.nonspatial_shape.clone();
        shape.push(out_height as i64);
        shape.push(out_width as i64);

        builder
            .start_band_nd(
                plan.name.as_deref(),
                &dim_names,
                &shape,
                plan.data_type,
                plan.nodata.as_deref(),
                None,
                None,
            )
            .map_err(|e| exec_datafusion_err!("Failed to start band: {}", e))?;

        let plane_bytes = out_width * out_height * plan.data_type.byte_size();
        let total = plane_bytes.checked_mul(plan.plane_count).ok_or_else(|| {
            exec_datafusion_err!("regridded band size overflow ({out_width}x{out_height})")
        })?;
        // One owned, nodata-pre-filled buffer holds the whole band (all planes,
        // plane-major); the covered sub-window of each plane is overwritten.
        let mut band_data = filled_with_nodata(total, plan.nodata.as_deref());
        if let (Some((ox0, ow, wx0, wsize)), Some((oy0, oh, wy0, hsize))) = (cover_x, cover_y) {
            for plane in 0..plan.plane_count {
                let band = src_dataset
                    .rasterband(gdal_band)
                    .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", gdal_band, e))?;
                let plane_buf = &mut band_data[plane * plane_bytes..(plane + 1) * plane_bytes];
                band.read_regrid_into(
                    (wx0, wy0, wsize, hsize),
                    (ox0, oy0),
                    (ow, oh),
                    out_width,
                    plane_buf,
                    grid.alg,
                )
                .map_err(|e| exec_datafusion_err!("Failed to regrid band {}: {}", gdal_band, e))?;
                gdal_band += 1;
            }
        } else {
            // No coverage on some axis: the whole band stays nodata.
            gdal_band += plan.plane_count;
        }

        let band_data_len = u32::try_from(band_data.len())
            .map_err(|_| exec_datafusion_err!("Band data too large for Arrow view"))?;
        let block = builder
            .band_data_writer()
            .append_block(Buffer::from_vec(band_data));
        builder
            .band_data_writer()
            .try_append_view(block, 0, band_data_len)
            .map_err(|e| exec_datafusion_err!("Failed to append band data: {}", e))?;

        builder
            .finish_band()
            .map_err(|e| exec_datafusion_err!("Failed to finish band: {}", e))?;
    }

    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("Failed to finish raster: {}", e))?;

    Ok(())
}

/// For one north-up axis, the output pixels whose centre falls inside the source
/// extent, plus the fractional source window (offset, size in source pixels)
/// that maps that covered run. Returns `None` when no output pixel centre lands
/// inside the source on this axis.
///
/// The source-pixel coordinate of output index `c`'s centre is the line
/// `s(c) = a + b*c` (with `b = out_scale / src_scale > 0` for same-orientation
/// grids), covered where `0 <= s(c) < src_n`.
fn covered_axis(
    out_origin: f64,
    out_scale: f64,
    src_origin: f64,
    src_scale: f64,
    src_n: usize,
    out_n: usize,
) -> Option<(usize, usize, f64, f64)> {
    if src_scale == 0.0 || out_scale == 0.0 || out_n == 0 || src_n == 0 {
        return None;
    }
    let b = out_scale / src_scale;
    if b <= 0.0 {
        // Opposite orientation between source and output — impossible for a
        // same-CRS resample, which preserves each axis's sign.
        return None;
    }
    let src_n_f = src_n as f64;
    let out_n_f = out_n as f64;
    let a = (out_origin + 0.5 * out_scale - src_origin) / src_scale;

    // Covered output indices: 0 <= a + b*c < src_n.
    let first = (-a / b).ceil().max(0.0);
    let last = ((src_n_f - a) / b).ceil() - 1.0; // largest c with s(c) < src_n
    let last = last.min(out_n_f - 1.0);
    if last < first {
        return None;
    }
    let i0 = first as usize;
    let count = (last - first) as usize + 1;

    // Fractional source window mapping the covered run: the left/top edge of
    // output index `i0` in source-pixel coordinates (centre minus half a step),
    // spanning `count` output pixels.
    let win_off = a - 0.5 * b + b * first;
    let win_size = count as f64 * b;
    Some((i0, count, win_off, win_size))
}

/// Allocate a `total`-byte buffer pre-filled with the little-endian `nodata`
/// byte pattern, or zeros when a band has no nodata. `total` is a whole number
/// of pixels, so the pattern always tiles exactly.
fn filled_with_nodata(total: usize, nodata: Option<&[u8]>) -> Vec<u8> {
    match nodata {
        Some(nd) if !nd.is_empty() && total.is_multiple_of(nd.len()) => {
            let mut buf = Vec::with_capacity(total);
            while buf.len() < total {
                buf.extend_from_slice(nd);
            }
            buf
        }
        _ => vec![0u8; total],
    }
}

/// Regroup a GDAL dataset's flat band list into N-D raster bands per `layout`,
/// reading each plane at the `metadata` grid size. With `alg = None` the read is
/// native (`out` == source size, an identity materialization); with `alg =
/// Some(_)` the full source window is resampled into the (possibly different)
/// `metadata` grid. The output geotransform/spatial grid come from `metadata`
/// and the CRS from `crs`.
fn append_nd_from_dataset_inner(
    dataset: &Dataset,
    layout: &GdalBandLayout,
    builder: &mut RasterBuilder,
    metadata: &RasterMetadata,
    crs: Option<&str>,
    alg: Option<ResampleAlg>,
) -> Result<()> {
    let (src_width, src_height) = dataset.raster_size();
    let out_width = metadata.width as usize;
    let out_height = metadata.height as usize;

    builder
        .start_raster(metadata, crs)
        .map_err(|e| exec_datafusion_err!("Failed to start raster: {}", e))?;

    let total_planes: usize = layout.bands.iter().map(|b| b.plane_count).sum();
    let gdal_band_count = dataset.raster_count();
    if gdal_band_count != total_planes {
        return Err(exec_datafusion_err!(
            "layout expects {total_planes} GDAL bands but dataset has {gdal_band_count}"
        ));
    }

    let mut gdal_band = 1;
    for plan in &layout.bands {
        let dim_names: Vec<&str> = plan.dim_names.iter().map(String::as_str).collect();
        // shape = [non-spatial..., height, width] — spatial from the output grid.
        let mut shape = plan.nonspatial_shape.clone();
        shape.push(out_height as i64);
        shape.push(out_width as i64);

        builder
            .start_band_nd(
                plan.name.as_deref(),
                &dim_names,
                &shape,
                plan.data_type,
                plan.nodata.as_deref(),
                None,
                None,
            )
            .map_err(|e| exec_datafusion_err!("Failed to start band: {}", e))?;

        let mut band_data: Vec<u8> = Vec::with_capacity(
            plan.plane_count * out_width * out_height * plan.data_type.byte_size(),
        );
        for _ in 0..plan.plane_count {
            let band = dataset
                .rasterband(gdal_band)
                .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", gdal_band, e))?;
            let plane = band
                .read_as_bytes(
                    (0, 0),
                    (src_width, src_height),
                    (out_width, out_height),
                    alg,
                )
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", gdal_band, e)
                })?;
            band_data.extend_from_slice(&plane);
            gdal_band += 1;
        }

        let band_data_len = u32::try_from(band_data.len())
            .map_err(|_| exec_datafusion_err!("Band data too large for Arrow view"))?;
        let block = builder
            .band_data_writer()
            .append_block(Buffer::from_vec(band_data));
        builder
            .band_data_writer()
            .try_append_view(block, 0, band_data_len)
            .map_err(|e| exec_datafusion_err!("Failed to append band data: {}", e))?;

        builder
            .finish_band()
            .map_err(|e| exec_datafusion_err!("Failed to finish band: {}", e))?;
    }

    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("Failed to finish raster: {}", e))?;

    Ok(())
}

/// Materialize a GDAL dataset as an N-D in-db raster `StructArray`, regrouping
/// its flat band list into N-D bands per `layout`.
pub fn gdal_dataset_to_nd_raster(
    dataset: &Dataset,
    layout: &GdalBandLayout,
) -> Result<StructArray> {
    let mut builder = RasterBuilder::new(1);
    append_nd_from_dataset(dataset, layout, &mut builder)?;
    builder
        .finish()
        .map_err(|e| exec_datafusion_err!("Failed to build raster: {}", e))
}

#[cfg(test)]
mod tests {
    use super::{append_as_indb_raster, append_as_outdb_raster, dataset_to_indb_raster};

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
    use sedona_testing::data::test_raster;
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

    fn load_as_outdb_raster(gdal: &Gdal, path: &str) -> datafusion_common::Result<StructArray> {
        let mut builder = RasterBuilder::new(1);
        append_as_outdb_raster(gdal, path, &mut builder)?;
        builder.finish().map_err(Into::into)
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

    fn write_uint16_tiff(gdal: &Gdal, path: &str, nodata: u16, data: Vec<u16>) {
        let driver = gdal.get_driver_by_name("GTiff").unwrap();
        let dataset = driver.create_with_band_type::<u16>(path, 2, 2, 1).unwrap();
        dataset
            .set_geo_transform(&[0.0, 0.5, 0.0, 1.0, 0.0, -0.5])
            .unwrap();
        dataset.set_projection("EPSG:4326").unwrap();
        let band = dataset.rasterband(1).unwrap();
        band.set_no_data_value(Some(nodata as f64)).unwrap();
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
        let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
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
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap(),
            [1u8, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn append_as_outdb_raster_reads_single_band_geotiff() {
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let raster = with_gdal(|gdal| load_as_outdb_raster(gdal, &path)).unwrap();
        let raster_struct = RasterStructArray::try_new(&raster).unwrap();
        assert_eq!(raster_struct.len(), 1);

        let raster = raster_struct.get(0).unwrap();
        assert_eq!(raster.metadata().width(), 10);
        assert_eq!(raster.metadata().height(), 10);
        assert!(raster.crs().is_some());

        let band = raster.bands().band(1).unwrap();
        assert_eq!(
            band.metadata().storage_type().unwrap(),
            StorageType::OutDbRef
        );
        assert!(band.metadata().outdb_url().unwrap().contains("test4.tiff"));
    }

    #[test]
    fn append_as_outdb_raster_preserves_uint64_nodata() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("uint64.tif");
        let path_str = path.to_string_lossy().to_string();
        let nodata = 9_007_199_254_740_993u64;

        with_gdal(|gdal| {
            write_uint64_tiff(gdal, &path_str, nodata, vec![1, 2, 3, 4]);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();

        let raster = with_gdal(|gdal| load_as_outdb_raster(gdal, &path_str)).unwrap();
        let raster_struct = RasterStructArray::try_new(&raster).unwrap();
        let raster = raster_struct.get(0).unwrap();
        let band = raster.bands().band(1).unwrap();

        assert_eq!(
            band.metadata().nodata_value().unwrap(),
            nodata.to_le_bytes()
        );
    }

    #[test]
    fn append_as_outdb_raster_preserves_int64_nodata() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("int64.tif");
        let path_str = path.to_string_lossy().to_string();
        let nodata = -9_007_199_254_740_993i64;

        with_gdal(|gdal| {
            write_int64_tiff(gdal, &path_str, nodata, vec![-1, -2, -3, -4]);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();

        let raster = with_gdal(|gdal| load_as_outdb_raster(gdal, &path_str)).unwrap();
        let raster_struct = RasterStructArray::try_new(&raster).unwrap();
        let raster = raster_struct.get(0).unwrap();
        let band = raster.bands().band(1).unwrap();

        assert_eq!(
            band.metadata().nodata_value().unwrap(),
            nodata.to_le_bytes()
        );
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
        let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
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
            .nd_buffer()
            .unwrap()
            .as_contiguous()
            .unwrap()
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
        let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = raster_struct.get(0).unwrap();
        let band = raster.bands().band(1).unwrap();

        assert_eq!(band.metadata().data_type().unwrap(), BandDataType::Int64);
        assert_eq!(
            band.metadata().nodata_value().unwrap(),
            &nodata.to_le_bytes()
        );

        let pixels: Vec<i64> = band
            .nd_buffer()
            .unwrap()
            .as_contiguous()
            .unwrap()
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(pixels, vec![-1, -2, -3, -4]);
    }

    #[test]
    fn dataset_to_indb_raster_preserves_uint16_nodata_and_data() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("uint16.tif");
        let path_str = path.to_string_lossy().to_string();
        let nodata = 513u16;

        with_gdal(|gdal| {
            write_uint16_tiff(gdal, &path_str, nodata, vec![1, 256, 511, 1024]);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();

        let raster_array = with_gdal(|gdal| load_as_indb_raster(gdal, &path_str)).unwrap();
        let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = raster_struct.get(0).unwrap();
        let band = raster.bands().band(1).unwrap();

        assert_eq!(band.metadata().data_type().unwrap(), BandDataType::UInt16);
        assert_eq!(
            band.metadata().nodata_value().unwrap(),
            &nodata.to_le_bytes()
        );

        let pixels: Vec<u16> = band
            .nd_buffer()
            .unwrap()
            .as_contiguous()
            .unwrap()
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(pixels, vec![1, 256, 511, 1024]);
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
        let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = raster_struct.get(0).unwrap();
        let band1 = raster.bands().band(1).unwrap();
        let band2 = raster.bands().band(2).unwrap();

        assert_eq!(raster.bands().len(), 2);
        assert_eq!(band1.metadata().storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band1.metadata().data_type().unwrap(), BandDataType::UInt8);
        assert_eq!(band1.metadata().nodata_value().unwrap(), [255u8]);
        assert_eq!(
            band1.nd_buffer().unwrap().as_contiguous().unwrap(),
            [10u8, 11, 12, 13]
        );

        assert_eq!(band2.metadata().storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band2.metadata().data_type().unwrap(), BandDataType::UInt8);
        assert_eq!(band2.metadata().nodata_value().unwrap(), [255u8]);
        assert_eq!(
            band2.nd_buffer().unwrap().as_contiguous().unwrap(),
            [100u8, 0, 200, 0]
        );
    }

    #[test]
    fn dataset_to_indb_raster_preserves_per_band_nodata_for_mem_dataset() {
        let raster_array = with_gdal(|gdal| {
            let dataset = build_multi_band_mem_dataset(gdal);
            dataset_to_indb_raster(&dataset)
        })
        .unwrap();

        let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = raster_struct.get(0).unwrap();
        let band1 = raster.bands().band(1).unwrap();
        let band2 = raster.bands().band(2).unwrap();

        assert_eq!(raster.bands().len(), 2);
        assert_eq!(band1.metadata().storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band1.metadata().data_type().unwrap(), BandDataType::UInt8);
        assert_eq!(band1.metadata().nodata_value().unwrap(), [0u8]);
        assert_eq!(
            band1.nd_buffer().unwrap().as_contiguous().unwrap(),
            [10u8, 11, 12, 13]
        );

        assert_eq!(band2.metadata().storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band2.metadata().data_type().unwrap(), BandDataType::UInt8);
        assert_eq!(band2.metadata().nodata_value().unwrap(), [255u8]);
        assert_eq!(
            band2.nd_buffer().unwrap().as_contiguous().unwrap(),
            [100u8, 0, 200, 0]
        );
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

        let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
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
