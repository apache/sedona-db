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
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::types::DatasetOptions;
use sedona_gdal::raster::types::ResampleAlg;
use sedona_gdal::spatial_ref::SpatialRef;
use sedona_raster::geo_transform::{GeoTransform, GeoTransformEx};

use arrow_schema::ArrowError;
use sedona_common::sedona_internal_err;
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::{RasterBuilder, StartBandArgs};
use sedona_raster::traits::RasterRef;
use sedona_schema::raster::BandDataType;

use crate::gdal_common::{
    add_layout_datapointer_bands, band_nodata_to_bytes, convert_gdal_err, gdal_to_band_data_type,
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

        let band_data = band
            .read_as_bytes((0, 0), (width, height), (width, height), None)
            .map_err(|e| exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e))?;
        // Single-band native read: the 2-D `["y", "x"]` band `start_band_2d`
        // would build, shape `[height, width]`.
        append_band_from_buffer(
            builder,
            &BandHeader {
                name: None,
                dim_names: &["y", "x"],
                shape: &[height as i64, width as i64],
                data_type: band_data_type,
                nodata: nodata_bytes.as_deref(),
            },
            band_data,
        )?;
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
        builder.start_band(StartBandArgs {
            name: None,
            dim_names: &["y", "x"],
            source_shape: &[height as i64, width as i64],
            view: None,
            data_type: band_data_type,
            nodata: nodata_bytes.as_deref(),
            outdb_uri: Some(&outdb_uri),
            outdb_format: None,
        })?;
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

    // Base pointer per band: the start of its nodata-pre-filled buffer. The
    // destination DATAPOINTER bands point into plane-major sub-ranges of these,
    // which GDAL warps into.
    //
    // SAFETY: each buffer holds exactly `plan.plane_count` writable planes of
    // `out_width*out_height*byte_size` bytes and, declared before `dst_dataset`,
    // outlives it; the helper only adds bands.
    let base_ptrs: Vec<*mut u8> = band_buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
    let mut dst_builder = MemDatasetBuilder::new(out_width, out_height);
    dst_builder = unsafe {
        add_layout_datapointer_bands(dst_builder, layout, &base_ptrs, out_width * out_height)
    };
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

/// The descriptive header for one output raster band — everything
/// [`RasterBuilder::start_band`] needs except the pixel bytes. Bundled so
/// [`append_stacked_band`] takes named fields rather than a long positional list.
pub(crate) struct BandHeader<'a> {
    pub name: Option<&'a str>,
    pub dim_names: &'a [&'a str],
    /// Full band shape `[non-spatial..., height, width]`.
    pub shape: &'a [i64],
    pub data_type: BandDataType,
    pub nodata: Option<&'a [u8]>,
}

/// Move an already-assembled per-band `Vec<u8>` of in-db pixel bytes into
/// `builder` as one band: start the band from `header`, hand the owned buffer to
/// Arrow as a shared data block (a refcount bump, never a copy), and finish the
/// band. This is the shared packaging tail of every path that builds one band's
/// bytes in an owned buffer — the read-back stacker ([`append_stacked_band`]),
/// the single-band GDAL reader ([`append_as_indb_raster`]), and RS_Clip's
/// masked-crop output.
///
/// The band is in-db (its `data` buffer is authoritative), so `outdb_uri` /
/// `outdb_format` are `None`; out-db bands carry no pixel buffer and do not use
/// this path. An oversize (> u32 view limit) band is rejected with a descriptive
/// error rather than a panic.
pub(crate) fn append_band_from_buffer(
    builder: &mut RasterBuilder,
    header: &BandHeader<'_>,
    band_data: Vec<u8>,
) -> Result<()> {
    let band_data_len = u32::try_from(band_data.len()).map_err(|_| {
        exec_datafusion_err!(
            "band data of {} bytes exceeds the binary-view limit",
            band_data.len()
        )
    })?;
    builder
        .start_band(StartBandArgs {
            name: header.name,
            dim_names: header.dim_names,
            source_shape: header.shape,
            view: None,
            data_type: header.data_type,
            nodata: header.nodata,
            outdb_uri: None,
            outdb_format: None,
        })
        .map_err(|e| exec_datafusion_err!("Failed to start band: {e}"))?;
    // Hand the owned buffer to Arrow as a shared data block (a refcount bump,
    // never a copy). `append_band_data_buffer` also stores sub-inline-threshold
    // bands inline, keeping the view canonical.
    builder
        .append_band_data_buffer(&Buffer::from(band_data), 0, band_data_len)
        .map_err(|e| exec_datafusion_err!("Failed to append band data: {e}"))?;
    builder
        .finish_band()
        .map_err(|e| exec_datafusion_err!("Failed to finish band: {e}"))?;
    Ok(())
}

/// Assemble one N-D raster band and append it to `builder`. The per-plane byte
/// size and plane count are derived from `header.shape`
/// (`[non-spatial..., height, width]`) with a fully checked multiply chain, so a
/// pathological extent overflows into a descriptive error rather than wrapping,
/// and an oversize (> u32 view limit) band is rejected before allocating rather
/// than after filling. For each plane in `0..n_planes`, `fill_plane(plane, &mut
/// buf)` appends exactly that plane's `height × width ×
/// header.data_type.byte_size()` bytes to `buf`, band-major planes concatenated
/// plane-major; the accumulated buffer is then moved into the Arrow array as a
/// zero-copy view block (a refcount bump) rather than copied through the builder.
///
/// This is the read-back / output-assembly inverse of the plane *input* bridge
/// ([`add_layout_datapointer_bands`]): the per-plane op — a GDAL band read, a
/// mask/crop, a tile-window copy — is the caller's `fill_plane` closure; the
/// plane iteration, checked buffer sizing, and band packaging are shared here.
pub(crate) fn append_stacked_band<F>(
    builder: &mut RasterBuilder,
    header: &BandHeader<'_>,
    mut fill_plane: F,
) -> Result<()>
where
    F: FnMut(usize, &mut Vec<u8>) -> Result<()>,
{
    // Derive the plane byte size and plane count from the output band shape with
    // a fully checked multiply chain: the total is the capacity of the buffer the
    // DATAPOINTER-free read-back path fills, and every factor comes from
    // untrusted band dimensions, so an overflow must error rather than wrap.
    let shape = header.shape;
    let ndim = shape.len();
    if ndim < 2 {
        return sedona_internal_err!(
            "append_stacked_band requires a trailing (height, width) pair, got shape {shape:?}"
        );
    }
    let byte_size = header.data_type.byte_size();
    let out_h = usize::try_from(shape[ndim - 2])
        .map_err(|_| exec_datafusion_err!("negative band height in shape {shape:?}"))?;
    let out_w = usize::try_from(shape[ndim - 1])
        .map_err(|_| exec_datafusion_err!("negative band width in shape {shape:?}"))?;
    let plane_bytes = out_w
        .checked_mul(out_h)
        .and_then(|px| px.checked_mul(byte_size))
        .ok_or_else(|| exec_datafusion_err!("band plane extent {out_w}x{out_h} overflows"))?;
    let mut n_planes: usize = 1;
    for &dim in &shape[..ndim - 2] {
        let dim = usize::try_from(dim)
            .map_err(|_| exec_datafusion_err!("negative band dimension in shape {shape:?}"))?;
        n_planes = n_planes.checked_mul(dim).ok_or_else(|| {
            exec_datafusion_err!("band plane count overflows for shape {shape:?}")
        })?;
    }
    let capacity = plane_bytes.checked_mul(n_planes).ok_or_else(|| {
        exec_datafusion_err!("band extent {out_w}x{out_h} of {n_planes} planes overflows")
    })?;
    // Reject an oversize (> u32 view limit) band before allocating and filling
    // the buffer rather than after. `append_band_from_buffer` re-checks the
    // realized length below; this is the pre-allocation guard.
    u32::try_from(capacity).map_err(|_| {
        exec_datafusion_err!("band data of {capacity} bytes exceeds the binary-view limit")
    })?;

    let mut band_data: Vec<u8> = Vec::with_capacity(capacity);
    for plane in 0..n_planes {
        fill_plane(plane, &mut band_data)?;
    }
    if band_data.len() != capacity {
        return sedona_internal_err!(
            "stacked band produced {} bytes, expected {capacity}",
            band_data.len()
        );
    }

    append_band_from_buffer(builder, header, band_data)
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

        // This band's planes are `plan.plane_count` consecutive GDAL bands
        // starting at `gdal_band`; read each at the output grid size (native when
        // `alg` is `None`, resampled otherwise) and stack them into the N-D band.
        let gdal_band_base = gdal_band;
        append_stacked_band(
            builder,
            &BandHeader {
                name: plan.name.as_deref(),
                dim_names: &dim_names,
                shape: &shape,
                data_type: plan.data_type,
                nodata: plan.nodata.as_deref(),
            },
            |plane, out| {
                let gdal_band = gdal_band_base + plane;
                let band = dataset
                    .rasterband(gdal_band)
                    .map_err(|e| exec_datafusion_err!("Failed to get band {gdal_band}: {e}"))?;
                let plane = band
                    .read_as_bytes(
                        (0, 0),
                        (src_width, src_height),
                        (out_width, out_height),
                        alg,
                    )
                    .map_err(|e| {
                        exec_datafusion_err!("Failed to read band {gdal_band} data: {e}")
                    })?;
                out.extend_from_slice(&plane);
                Ok(())
            },
        )?;
        gdal_band += plan.plane_count;
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
    use super::{
        append_as_indb_raster, append_as_outdb_raster, append_stacked_band, dataset_to_indb_raster,
        BandHeader,
    };

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

    #[test]
    fn test_append_stacked_band_rejects_oversize_extent() {
        // A pathological plane extent whose byte product overflows `usize` must
        // return the descriptive overflow error rather than panic (debug) or wrap
        // (release). This guards the shared band-stacking path that both RS_Tile
        // and the warp/resample read-back funnel through. `fill_plane` never runs
        // because the size check fails before the buffer is allocated.
        let huge = 1i64 << 40; // (1<<40)^2 = 2^80, overflows usize
        let mut builder = RasterBuilder::new(1);
        let err = append_stacked_band(
            &mut builder,
            &BandHeader {
                name: None,
                dim_names: &["y", "x"],
                shape: &[huge, huge],
                data_type: BandDataType::UInt8,
                nodata: None,
            },
            |_plane, _out| panic!("fill_plane must not run when the extent overflows"),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("overflows"),
            "expected an overflow error, got: {err}"
        );
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
