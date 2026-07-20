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

//! Point-sampling primitives shared by the raster value functions
//! (`RS_Value`, `RS_Values`).
//!
//! These helpers turn a geometry coordinate into a pixel read against a band's
//! [`NdBuffer`](sedona_raster::traits::NdBuffer): reprojecting into the raster's
//! CRS, mapping a world coordinate to a `(col, row)` index, and decoding the one
//! pixel there. They are geometry-shape agnostic — `RS_Value` drives them with a
//! single Point, `RS_Values` with each sub-point of a MultiPoint — so the pixel
//! math and CRS handling live in exactly one place.

use std::rc::Rc;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::{exec_datafusion_err, exec_err, DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use sedona_geometry::error::SedonaGeometryError;
use sedona_geometry::transform::{visit_point_coords, CrsEngine, CrsTransform};
use sedona_raster::affine_transformation::AffineMatrix;
use sedona_raster::traits::{nodata_bytes_to_f64_lossless, NdBuffer};
use sedona_schema::crs::CrsRef;
use sedona_schema::datatypes::SedonaType;
use wkb::reader::read_wkb;

/// Materialise an integer argument as an owned `Int32` [`ArrayRef`] for the
/// batch. Callers keep the returned `ArrayRef` alive and borrow a typed
/// `&Int32Array` view from it (via `as_int32_array`) rather than cloning the
/// typed array.
pub(crate) fn int32_array_arg(arg: &ColumnarValue, num_iterations: usize) -> Result<ArrayRef> {
    arg.clone()
        .cast_to(&DataType::Int32, None)?
        .into_array(num_iterations)
}

/// Advance the optional band-number iterator one row, yielding the 1-based band
/// to sample. A missing band argument defaults to band 1; a NULL band element
/// returns `None`, which the caller propagates to a NULL result. Band 0 and
/// negative values map to 0 so [`Bands::band`](sedona_raster::traits::Bands::band)
/// rejects them as not 1-based rather than being silently coerced.
pub(crate) fn next_band(
    band_iter: &mut Option<arrow_array::iterator::ArrayIter<&arrow_array::Int32Array>>,
) -> Option<usize> {
    match band_iter.as_mut() {
        None => Some(1),
        Some(iter) => iter.next().flatten().map(|b| b.max(0) as usize),
    }
}

/// Resolve the 1-based band to sample when no band argument was given: band 1
/// for a single-band raster, otherwise an error. Sampling an unspecified band of
/// a multiband raster is ambiguous, so the caller must name the band rather than
/// silently getting band 1 (matches `RS_SetBandNoDataValue`'s 2-argument form).
/// `func` names the calling UDF for the error message.
pub(crate) fn default_band(func: &str, num_bands: usize) -> Result<usize> {
    if num_bands == 1 {
        Ok(1)
    } else {
        exec_err!(
            "{func}: raster has {num_bands} bands; specify which band to sample (the \
             2-argument form is only allowed for a single-band raster)"
        )
    }
}

/// Resolve the coordinate transform that lands a geometry in the raster's
/// CRS, or `None` when the coordinates can be sampled as-is (equal CRSes, or
/// both absent).
///
/// Errors if exactly one of the geometry / raster carries a CRS: sampling
/// across a known and an unknown CRS would silently mislocate the geometry.
///
/// Unlike the spatial predicates (`RS_Intersects` et al.), which fall back to a
/// WGS84 pivot when a direct transform between two CRSes fails, a failed
/// transform here is propagated as an error. Sampling has to land the geometry
/// in the raster's own CRS — that is the only space its affine/pixel grid is
/// defined in — so there is no neutral CRS to fall back to: a WGS84 pivot would
/// silently sample the wrong pixel rather than compare geometries in a shared
/// space.
///
/// `func` names the calling UDF for the error messages.
pub(crate) fn point_crs_transform(
    func: &str,
    geom_crs: CrsRef<'_>,
    raster_crs: CrsRef<'_>,
    engine: &dyn CrsEngine,
) -> Result<Option<Rc<dyn CrsTransform>>> {
    match (geom_crs, raster_crs) {
        (Some(geom_crs), Some(raster_crs)) => {
            if geom_crs.crs_equals(raster_crs) {
                Ok(None)
            } else {
                engine
                    .get_transform_crs_to_crs(
                        &geom_crs.to_crs_string(),
                        &raster_crs.to_crs_string(),
                        None,
                        "",
                    )
                    .map(Some)
                    .map_err(|e| exec_datafusion_err!("{func}: CRS transform error: {e}"))
            }
        }
        (None, None) => Ok(None),
        (Some(_), None) => exec_err!("{func}: geometry has a CRS but the raster does not"),
        (None, Some(_)) => exec_err!("{func}: raster has a CRS but the geometry does not"),
    }
}

/// For a **column-level** geometry CRS, resolve the raster-CRS transform once
/// for the whole batch:
/// - `Some(None)`     — CRSes match (or both absent), sample original coordinates;
/// - `Some(Some(t))`  — CRSes differ, apply `t` to every geometry;
/// - `None`           — the geometry CRS is carried per row, so resolve per row.
///
/// Errors when exactly one side carries a CRS. Hoisting this out of the per-row
/// loop avoids both a per-row `crs_equals` (whose `to_authority_code()`
/// allocates a `String` every call) and a per-row transform lookup. `func`
/// names the calling UDF for the error messages.
#[allow(clippy::type_complexity)]
pub(crate) fn column_point_crs_transform(
    func: &str,
    geom_type: &SedonaType,
    raster_crs: CrsRef<'_>,
    engine: &dyn CrsEngine,
) -> Result<Option<Option<Rc<dyn CrsTransform>>>> {
    let geom_crs = match geom_type {
        SedonaType::Wkb(_, c) | SedonaType::WkbView(_, c) => c.as_deref(),
        // A per-item CRS varies by row; the caller resolves per row.
        _ => return Ok(None),
    };
    point_crs_transform(func, geom_crs, raster_crs, engine).map(Some)
}

/// Visit each point of a Point/MultiPoint WKB as `Some((x, y))` in the
/// raster's CRS (`None` for an empty sub-point), applying `trans` to each
/// coordinate when given.
///
/// A thin [`visit_point_coords`] adapter that parses the WKB and threads
/// DataFusion errors out of the geometry-crate callback (via
/// `SedonaGeometryError::External`), so callers sample in one pass with no
/// transformed-WKB materialisation or coordinate scratch buffer.
pub(crate) fn visit_points(
    func: &str,
    wkb: &[u8],
    trans: Option<&dyn CrsTransform>,
    mut visit: impl FnMut(Option<(f64, f64)>) -> Result<()>,
) -> Result<()> {
    let geom = read_wkb(wkb).map_err(|e| exec_datafusion_err!("{func}: {e}"))?;
    visit_point_coords(&geom, trans, |xy| {
        visit(xy).map_err(|e| SedonaGeometryError::External(Box::new(e)))
    })
    .map_err(|e| match e {
        SedonaGeometryError::External(inner) => match inner.downcast::<DataFusionError>() {
            Ok(df) => *df,
            Err(other) => exec_datafusion_err!("{func}: {other}"),
        },
        other => exec_datafusion_err!("{func}: {other}"),
    })
}

/// Map a coordinate in the raster's CRS to a 0-based `(col, row)` pixel index,
/// or `None` when the coordinate has no location to sample.
///
/// A non-finite coordinate (e.g. `POINT(NaN 5)`) returns `None`: without this
/// guard a NaN would survive `inv_transform` and the saturating `f64 -> i64`
/// cast would turn it into 0 (in bounds), silently sampling pixel column 0
/// rather than yielding NULL. `func` names the calling UDF for the error
/// message.
pub(crate) fn xy_to_pixel(
    func: &str,
    affine: &AffineMatrix,
    x: f64,
    y: f64,
) -> Result<Option<(i64, i64)>> {
    if !x.is_finite() || !y.is_finite() {
        return Ok(None);
    }
    let (raster_x, raster_y) = affine
        .inv_transform(x, y)
        .map_err(|e| exec_datafusion_err!("{func}: {e}"))?;
    // Floor (not truncate toward zero) so a point just outside the top/left edge
    // maps to a negative index and is rejected as out of bounds, rather than
    // truncating to 0 and sampling an edge pixel. The `f64 -> i64` cast saturates
    // (never panics); the bounds check in `read_pixel` rejects an out-of-range
    // index as NULL.
    Ok(Some((raster_x.floor() as i64, raster_y.floor() as i64)))
}

/// Read pixel `(col, row)` from an already-resolved band buffer and nodata
/// value. Returns `None` for an out-of-bounds pixel or one that equals nodata.
/// Resolving the band once and calling this per point lets a caller sample many
/// points from one buffer without re-resolving per point.
///
/// Reads exactly one pixel by computing its byte offset from the band's strides
/// — zero-copy and O(1), no whole-band materialisation. `func` names the
/// calling UDF for the error messages.
pub(crate) fn read_pixel(
    func: &str,
    buffer: &NdBuffer,
    nodata: Option<f64>,
    col: i64,
    row: i64,
) -> Result<Option<f64>> {
    let (height, width) = (buffer.shape[0], buffer.shape[1]);
    if row < 0 || row >= height || col < 0 || col >= width {
        return Ok(None);
    }

    // Byte offset of the (row, col) pixel via the band's own strides, so the
    // read stays correct for any layout the producer hands us. Checked
    // arithmetic throughout: `row`/`col` are already in bounds, but a corrupt
    // stride or offset must surface as an error, never an i64 overflow panic.
    let size = buffer.data_type.byte_size() as i64;
    let byte_offset = row
        .checked_mul(buffer.strides[0])
        .zip(col.checked_mul(buffer.strides[1]))
        .and_then(|(r, c)| r.checked_add(c))
        .and_then(|rc| rc.checked_add(buffer.offset as i64))
        .ok_or_else(|| exec_datafusion_err!("{func}: pixel byte offset overflow"))?;
    let end_offset = byte_offset
        .checked_add(size)
        .ok_or_else(|| exec_datafusion_err!("{func}: pixel byte offset overflow"))?;
    let start = usize::try_from(byte_offset)
        .map_err(|_| exec_datafusion_err!("{func}: negative pixel byte offset"))?;
    let end = usize::try_from(end_offset)
        .map_err(|_| exec_datafusion_err!("{func}: pixel byte offset overflow"))?;
    let bytes = buffer
        .buffer
        .get(start..end)
        .ok_or_else(|| exec_datafusion_err!("{func}: pixel is out of the band's buffer bounds"))?;

    // Decode the pixel to f64. The lossless converter errors (rather than
    // silently rounding) on Int64/UInt64 values beyond f64's exact-integer
    // range (2^53) — the value functions return a Double, so such a pixel can't
    // be represented faithfully; failing loudly is preferred over a wrong value.
    let value = nodata_bytes_to_f64_lossless(bytes, &buffer.data_type)
        .map_err(|e| exec_datafusion_err!("{func}: {e}"))?;

    if let Some(nodata) = nodata {
        if value == nodata || (value.is_nan() && nodata.is_nan()) {
            return Ok(None);
        }
    }

    Ok(Some(value))
}
