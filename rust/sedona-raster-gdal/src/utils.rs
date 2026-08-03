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
use datafusion_common::{exec_datafusion_err, exec_err};
use sedona_gdal::dataset::Dataset;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::gdal_dyn_bindgen::{GDAL_OF_RASTER, GDAL_OF_READONLY};
use sedona_gdal::geo_transform::{GeoTransform, GeoTransformEx};
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::types::DatasetOptions;
use sedona_gdal::raster::types::ResampleAlg;
use sedona_gdal::spatial_ref::SpatialRef;

use arrow_schema::ArrowError;
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_schema::raster::BandDataType;

use crate::gdal_common::{
    band_data_type_to_gdal, band_nodata_to_bytes, convert_gdal_err, gdal_to_band_data_type,
    normalize_outdb_source_path, set_band_nodata_from_bytes, GdalBandLayout,
};

/// Append a GDAL dataset as a single in-db raster to the provided [`RasterBuilder`].
pub fn append_as_indb_raster(dataset: &Dataset, builder: &mut RasterBuilder) -> Result<()> {
    let (width, height) = dataset.raster_size();

    let geotransform = dataset
        .geo_transform()
        .map_err(|e| exec_datafusion_err!("Failed to get geotransform: {}", e))?;

    let grid = Grid::from_gdal(geotransform, width, height);

    let crs = dataset
        .spatial_ref()
        .ok()
        .and_then(|sr: SpatialRef| sr.to_projjson().ok());

    grid.start_raster_into(builder, crs.as_deref())
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

        builder
            .start_band_2d(band_data_type, nodata_bytes.as_deref())
            .map_err(|e| exec_datafusion_err!("Failed to start band: {}", e))?;

        let band_data = band
            .read_as_bytes((0, 0), (width, height), (width, height), None)
            .map_err(|e| exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e))?;
        let band_data_len = u32::try_from(band_data.len())
            .map_err(|_| exec_datafusion_err!("Band {} data too large for Arrow view", band_idx))?;
        // Hand the freshly-read allocation to Arrow as a shared data block (a
        // refcount bump, never a copy). `append_band_data_buffer` also stores
        // sub-inline-threshold bands inline, keeping the view canonical.
        builder
            .append_band_data_buffer(&Buffer::from_vec(band_data), 0, band_data_len)
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
    let grid = Grid::from_gdal(geotransform, width, height);

    let crs = dataset
        .spatial_ref()
        .ok()
        .and_then(|sr: SpatialRef| sr.to_projjson().ok());

    grid.start_raster_into(builder, crs.as_deref())?;

    let band_count = dataset.raster_count();
    for band_idx in 1..=band_count {
        let band = dataset
            .rasterband(band_idx)
            .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

        let gdal_type = band.band_type();
        let band_data_type = gdal_to_band_data_type(gdal_type)
            .map_err(|_| exec_datafusion_err!("Unsupported band data type: {:?}", gdal_type))?;

        let nodata_bytes = band_nodata_to_bytes(&band)?;

        // Out-db band: location + band selector in the `#band=N` URI; empty data.
        let outdb_uri = format!("{path}#band={band_idx}");
        builder.start_band_nd(
            None,
            &["y", "x"],
            &[height as i64, width as i64],
            band_data_type,
            nodata_bytes.as_deref(),
            Some(&outdb_uri),
            None,
        )?;
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
    let grid = Grid::from_gdal(geotransform, width, height);

    let crs = dataset
        .spatial_ref()
        .ok()
        .and_then(|sr: SpatialRef| sr.to_projjson().ok());

    append_nd_from_dataset_inner(dataset, layout, builder, &grid, crs.as_deref(), None)
}

/// A raster's affine grid: its geotransform and pixel dimensions.
///
/// This is the geometric core shared by a source raster's grid (via
/// [`Grid::from_raster`]) and every resample/warp output target, replacing the
/// several near-identical transform-plus-dimensions structs that had accumulated
/// here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub transform: GeoTransform,
    pub width: i64,
    pub height: i64,
}

/// Axis-aligned world bounding box of a [`Grid`]'s four corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Envelope {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl Grid {
    /// Build a grid from a GDAL geotransform and pixel dimensions.
    pub fn from_gdal(transform: GeoTransform, width: usize, height: usize) -> Self {
        Self {
            transform,
            width: width as i64,
            height: height as i64,
        }
    }

    /// The grid of an existing raster, read from its transform and spatial
    /// shape. Errors if the transform is not 6 elements or width/height are
    /// missing (an invariant violation).
    pub fn from_raster(raster: &dyn RasterRef) -> Result<Self, ArrowError> {
        let transform = <[f64; 6]>::try_from(raster.transform()).map_err(|_| {
            ArrowError::InvalidArgumentError("expected a 6-element geotransform".to_string())
        })?;
        Ok(Self {
            transform,
            width: raster.width()?,
            height: raster.height()?,
        })
    }

    /// Start `builder` on a 2-D raster covering this grid, stamping `crs`.
    pub fn start_raster_into(
        &self,
        builder: &mut RasterBuilder,
        crs: Option<&str>,
    ) -> Result<(), ArrowError> {
        let t = self.transform;
        builder.start_raster_2d(
            self.width,
            self.height,
            t[0],
            t[3],
            t[1],
            t[5],
            t[2],
            t[4],
            crs,
        )
    }

    /// Bounding box of the four grid corners (handles skew; reduces to
    /// `width * |scale_x|` etc. for a north-up grid). Corners are mapped with the
    /// shared [`GeoTransformEx::apply`] — the same affine used everywhere else.
    pub fn envelope(&self) -> Envelope {
        let w = self.width as f64;
        let h = self.height as f64;
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for (col, row) in [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
            let (x, y) = self.transform.apply(col, row);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        Envelope {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }
}

/// The output grid a resample or warp writes into: a [`Grid`] plus the CRS to
/// record on the output and the resampling algorithm.
///
/// `crs` is the CRS stamped on the output — the source CRS for a same-CRS
/// resample or regrid, the reprojection target for a warp that changes CRS. It
/// is carried through verbatim rather than read back from GDAL so a caller can
/// preserve the exact CRS string it was given.
pub struct OutputGrid<'a> {
    pub grid: Grid,
    pub crs: Option<&'a str>,
    pub alg: ResampleAlg,
}

/// GDAL 3.13 is the first release that resamples Mode through a double working
/// type instead of a 32-bit float.
const GDAL_3_13_0: i32 = 3_130_000;

/// Map an algorithm name (case-insensitive) to a GDAL [`ResampleAlg`]. An empty
/// string defaults to nearest neighbour (matching Sedona Spark). `func_name`
/// names the calling SQL function for the error message.
///
/// Accepts the GDAL names plus the Spark spellings (American `NearestNeighbor`
/// and `Bicubic`, which GDAL calls `Cubic`). Shared by the RasterIO-resampling
/// and warping UDFs so they accept an identical algorithm surface.
pub fn parse_resample_algorithm(name: &str, func_name: &str) -> Result<ResampleAlg> {
    if name.is_empty() {
        return Ok(ResampleAlg::NearestNeighbour);
    }
    let alg = match name.to_ascii_lowercase().as_str() {
        "nearestneighbor" | "nearestneighbour" | "nearest" | "near" => {
            ResampleAlg::NearestNeighbour
        }
        "bilinear" => ResampleAlg::Bilinear,
        "cubic" | "bicubic" => ResampleAlg::Cubic,
        "cubicspline" => ResampleAlg::CubicSpline,
        "lanczos" => ResampleAlg::Lanczos,
        "average" => ResampleAlg::Average,
        "mode" => ResampleAlg::Mode,
        _ => {
            return exec_err!(
                "{func_name}: unknown algorithm {name:?}; expected one of \
                 NearestNeighbor, Bilinear, Cubic, CubicSpline, Lanczos, Average, Mode"
            );
        }
    };
    Ok(alg)
}

/// Reject the integer dtypes GDAL's floating resample/warp working type cannot
/// represent exactly.
///
/// `routes_through_float` is true when the pixels will pass through a floating
/// working type: always for a warp/reproject, and for RasterIO resampling only
/// on a regrid or an interpolating algorithm — a plain nearest-neighbour
/// decimation is an exact native-type copy, so it passes `false` and every dtype
/// is allowed. `func_name` names the calling SQL function for the error message.
///
/// `Int64`/`UInt64` are never representable in a floating working type (a double
/// is exact only to 2^53). `Mode` additionally uses a 32-bit float working type
/// on GDAL < 3.13, which cannot represent `Int32`/`UInt32` above 2^24; since Mode
/// is a value selection (the most-common source value), silently rounding a
/// category code is especially wrong, so those are rejected on older GDAL too.
pub fn reject_lossy_resample_dtypes(
    gdal: &Gdal,
    raster: &RasterRefImpl<'_>,
    band_count: usize,
    alg: ResampleAlg,
    routes_through_float: bool,
    func_name: &str,
) -> Result<()> {
    if !routes_through_float {
        return Ok(());
    }
    let mode_uses_float32 = alg == ResampleAlg::Mode && gdal.version_num() < GDAL_3_13_0;
    for i in 0..band_count {
        match raster.band_data_type(i) {
            Some(BandDataType::Int64 | BandDataType::UInt64) => {
                return exec_err!(
                    "{func_name} does not support Int64/UInt64 rasters: GDAL routes 64-bit \
                     integer pixels through a floating working type that cannot represent them \
                     exactly; cast to Int32/Float64 first."
                );
            }
            Some(BandDataType::Int32 | BandDataType::UInt32) if mode_uses_float32 => {
                return exec_err!(
                    "{func_name} does not support Mode resampling of Int32/UInt32 rasters on GDAL \
                     {} (before 3.13): Mode routes pixels through a 32-bit float working type that \
                     cannot represent 32-bit integers above 2^24 exactly. Upgrade to GDAL >= 3.13, \
                     which resamples Mode through a double, or cast to Float64 first.",
                    gdal.version_info("RELEASE_NAME")
                );
            }
            _ => {}
        }
    }
    Ok(())
}

/// Append a GDAL dataset as a single **N-D** in-db raster, resampling every band
/// to `output`'s dimensions with `output.alg`.
///
/// The spatial analog of [`append_nd_from_dataset`]: the full source window of
/// each GDAL band is read into an `output.grid.width` x `output.grid.height`
/// buffer using GDAL's RasterIO resampling, so band count/order and the
/// non-spatial structure in `layout` are preserved and only the trailing
/// `(y, x)` extent changes.
pub fn append_resampled_nd_from_dataset(
    dataset: &Dataset,
    layout: &GdalBandLayout,
    builder: &mut RasterBuilder,
    output: &OutputGrid<'_>,
) -> Result<()> {
    append_nd_from_dataset_inner(
        dataset,
        layout,
        builder,
        &output.grid,
        output.crs,
        Some(output.alg),
    )
}

/// Warp every band of `src_dataset` into `grid`, appending the result as a
/// single **N-D** in-db raster regrouped per `layout`.
///
/// The destination is a MEM dataset whose band buffers are pre-filled with each
/// band's nodata value (zero when a band has none), so output cells the
/// (reprojected) source footprint does not cover — the extent can grow past the
/// source, or shift under a reprojection — read back as nodata rather than as an
/// accidental zero (`GDALReprojectImage` writes only covered pixels). GDAL warps
/// into those buffers; the warped bytes are then regrouped into N-D bands via
/// [`append_nd_from_dataset_inner`], carrying `output.crs` through verbatim
/// rather than round-tripping it back out of GDAL.
///
/// `warp_memory_limit_bytes` is GDAL's working-memory cache size for the warp;
/// pass `0.0` for GDAL's own default.
pub fn append_warped_nd_from_dataset(
    gdal: &Gdal,
    src_dataset: &Dataset,
    layout: &GdalBandLayout,
    builder: &mut RasterBuilder,
    output: &OutputGrid<'_>,
    warp_memory_limit_bytes: f64,
) -> Result<()> {
    let out_width = output.grid.width as usize;
    let out_height = output.grid.height as usize;

    // One owned, nodata-pre-filled buffer per source band, its planes
    // concatenated plane-major; the destination's DATAPOINTER bands point into
    // sub-ranges of these. `band_buffers` must outlive `dst_dataset`, so it is
    // declared first (locals drop in reverse order).
    let mut band_buffers: Vec<Vec<u8>> = Vec::with_capacity(layout.bands.len());
    for plan in &layout.bands {
        let plane_bytes = out_width * out_height * plan.data_type.byte_size();
        let total = plane_bytes.checked_mul(plan.plane_count).ok_or_else(|| {
            exec_datafusion_err!("warped band size overflow ({out_width}x{out_height})")
        })?;
        band_buffers.push(filled_with_nodata(total, plan.nodata.as_deref()));
    }

    let mut dst_builder = MemDatasetBuilder::new(out_width, out_height);
    for (plan, buffer) in layout.bands.iter().zip(band_buffers.iter_mut()) {
        let gdal_type = band_data_type_to_gdal(&plan.data_type);
        let plane_bytes = out_width * out_height * plan.data_type.byte_size();
        for plane in 0..plan.plane_count {
            let ptr = buffer[plane * plane_bytes..].as_mut_ptr();
            // SAFETY: each plane sub-range holds exactly `plane_bytes` valid,
            // writable bytes aligned for the band type, and `band_buffers`
            // outlives `dst_dataset`.
            unsafe {
                dst_builder = dst_builder.add_band(gdal_type, ptr);
            }
        }
    }
    // SAFETY: the DATAPOINTER buffers in `band_buffers` outlive `dst_dataset`.
    let dst_dataset = unsafe { dst_builder.build(gdal).map_err(convert_gdal_err)? };
    dst_dataset
        .set_geo_transform(&output.grid.transform)
        .map_err(convert_gdal_err)?;
    if let Some(crs) = output.crs {
        dst_dataset.set_projection(crs).map_err(convert_gdal_err)?;
    }

    // Record each band's nodata on the destination in its native type. Walk the
    // dst bands in the same band-major / plane order as the add loop above.
    // `set_band_nodata_from_bytes` uses the exact Int64/UInt64 setters rather
    // than routing through a lossy f64, so a large 64-bit nodata stays exact.
    let mut dst_band_index = 0usize;
    for plan in &layout.bands {
        for _ in 0..plan.plane_count {
            dst_band_index += 1;
            if let Some(nodata) = plan.nodata.as_deref() {
                let band = dst_dataset
                    .rasterband(dst_band_index)
                    .map_err(convert_gdal_err)?;
                set_band_nodata_from_bytes(&band, Some(nodata))?;
            }
        }
    }

    // Reproject when the CRS differs; a same-CRS warp is a pure regrid that fills
    // grown/shifted areas with the pre-filled nodata.
    gdal.reproject_image(
        src_dataset,
        &dst_dataset,
        output.alg,
        warp_memory_limit_bytes,
    )
    .map_err(convert_gdal_err)?;

    // Read the warped buffers back out (native read, no further resampling),
    // regrouping the flat GDAL bands into N-D raster bands.
    append_nd_from_dataset_inner(
        &dst_dataset,
        layout,
        builder,
        &output.grid,
        output.crs,
        None,
    )
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
    grid: &Grid,
    crs: Option<&str>,
    alg: Option<ResampleAlg>,
) -> Result<()> {
    let (src_width, src_height) = dataset.raster_size();
    let out_width = grid.width as usize;
    let out_height = grid.height as usize;

    grid.start_raster_into(builder, crs)
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

        let capacity = out_width
            .checked_mul(out_height)
            .and_then(|a| a.checked_mul(plan.data_type.byte_size()))
            .and_then(|a| a.checked_mul(plan.plane_count))
            .ok_or_else(|| {
                exec_datafusion_err!("resampled band size overflow ({out_width}x{out_height})")
            })?;
        let mut band_data: Vec<u8> = Vec::with_capacity(capacity);
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
    use sedona_schema::raster::BandDataType;
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
        let band = raster.band(0).unwrap();

        assert_eq!(raster.width().unwrap(), 3);
        assert_eq!(raster.height().unwrap(), 2);
        assert_eq!(raster.transform()[0], 1.5);
        assert_eq!(raster.transform()[3], 4.5);
        assert!(raster.crs().is_some());
        assert!(band.is_indb());
        assert_eq!(band.data_type(), BandDataType::UInt8);
        assert_eq!(band.nodata().unwrap(), [255u8]);
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
        assert_eq!(raster.width().unwrap(), 10);
        assert_eq!(raster.height().unwrap(), 10);
        assert!(raster.crs().is_some());

        let band = raster.band(0).unwrap();
        assert!(!band.is_indb());
        assert!(band.outdb_uri().unwrap().contains("test4.tiff"));
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
        let band = raster.band(0).unwrap();

        assert_eq!(band.nodata().unwrap(), nodata.to_le_bytes());
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
        let band = raster.band(0).unwrap();

        assert_eq!(band.nodata().unwrap(), nodata.to_le_bytes());
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
        let band = raster.band(0).unwrap();

        assert_eq!(raster.width().unwrap(), 2);
        assert_eq!(raster.height().unwrap(), 2);
        assert_eq!(raster.transform()[0], 100.0);
        assert_eq!(raster.transform()[3], 200.0);
        assert_eq!(band.data_type(), BandDataType::UInt64);
        assert_eq!(band.nodata().unwrap(), &nodata.to_le_bytes());

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
        let band = raster.band(0).unwrap();

        assert_eq!(band.data_type(), BandDataType::Int64);
        assert_eq!(band.nodata().unwrap(), &nodata.to_le_bytes());

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
        let band = raster.band(0).unwrap();

        assert_eq!(band.data_type(), BandDataType::UInt16);
        assert_eq!(band.nodata().unwrap(), &nodata.to_le_bytes());

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
        let band1 = raster.band(0).unwrap();
        let band2 = raster.band(1).unwrap();

        assert_eq!(raster.num_bands(), 2);
        assert!(band1.is_indb());
        assert_eq!(band1.data_type(), BandDataType::UInt8);
        assert_eq!(band1.nodata().unwrap(), [255u8]);
        assert_eq!(
            band1.nd_buffer().unwrap().as_contiguous().unwrap(),
            [10u8, 11, 12, 13]
        );

        assert!(band2.is_indb());
        assert_eq!(band2.data_type(), BandDataType::UInt8);
        assert_eq!(band2.nodata().unwrap(), [255u8]);
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
        let band1 = raster.band(0).unwrap();
        let band2 = raster.band(1).unwrap();

        assert_eq!(raster.num_bands(), 2);
        assert!(band1.is_indb());
        assert_eq!(band1.data_type(), BandDataType::UInt8);
        assert_eq!(band1.nodata().unwrap(), [0u8]);
        assert_eq!(
            band1.nd_buffer().unwrap().as_contiguous().unwrap(),
            [10u8, 11, 12, 13]
        );

        assert!(band2.is_indb());
        assert_eq!(band2.data_type(), BandDataType::UInt8);
        assert_eq!(band2.nodata().unwrap(), [255u8]);
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
        assert_eq!(first.width().unwrap(), 3);
        assert_eq!(first.height().unwrap(), 2);
        assert_eq!(first.num_bands(), 1);

        let second = raster_struct.get(1).unwrap();
        assert_eq!(second.width().unwrap(), 2);
        assert_eq!(second.height().unwrap(), 2);
        assert_eq!(second.num_bands(), 2);
    }
}
