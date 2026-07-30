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

//! Shared pixel-wise min/max combine logic for rasters.
//!
//! Used by both the `RS_Min_Agg`/`RS_Max_Agg` aggregate functions
//! ([crate::rs_min_max_agg]) and the `RS_Min`/`RS_Max` scalar functions
//! ([crate::rs_min_max]), so the two surfaces can't drift on what "min/max
//! of two rasters" actually means.
//!
//! All four functions operate on a single band, selected by an optional
//! 1-based `band_index` argument (default 1) — the same convention as
//! `RS_BandPixelType`/`RS_BandNoDataValue`. An out-of-range `band_index`
//! yields a null result rather than an error, also matching that
//! convention.

use arrow_array::{Array, Int32Array, StructArray};
use arrow_schema::DataType;
use datafusion_common::{cast::as_int32_array, error::Result, exec_datafusion_err, exec_err};
use datafusion_expr::ColumnarValue;
use sedona_common::sedona_internal_err;
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::{RasterBuilder, RasterOverrides};
use sedona_raster::traits::{BandOverrides, BandRef, RasterRef};
use sedona_schema::raster::BandDataType;

/// Which pixel wins a pairwise comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MinMaxOp {
    Min,
    Max,
}

impl MinMaxOp {
    /// True if `a` should be kept over `b`.
    fn a_wins<T: PartialOrd>(&self, a: T, b: T) -> bool {
        match self {
            MinMaxOp::Min => a <= b,
            MinMaxOp::Max => a >= b,
        }
    }
}

/// Whether a pixel value is NaN. Always `false` for integer types — only
/// `f32`/`f64` override it. Lets [combine_bytes]'s macro treat NaN the same
/// way regardless of the concrete pixel type.
trait MaybeNan: Copy {
    fn is_nan_value(self) -> bool {
        false
    }
}

impl MaybeNan for u8 {}
impl MaybeNan for i8 {}
impl MaybeNan for u16 {}
impl MaybeNan for i16 {}
impl MaybeNan for u32 {}
impl MaybeNan for i32 {}
impl MaybeNan for u64 {}
impl MaybeNan for i64 {}
impl MaybeNan for f32 {
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
}
impl MaybeNan for f64 {
    fn is_nan_value(self) -> bool {
        self.is_nan()
    }
}

/// Resolved 1-based band index per row: either a fixed constant (the
/// `band_index` argument was omitted, or every row of a state array is
/// already single-band) or a per-row array (an explicit `band_index`
/// argument was supplied and may vary row to row).
pub(crate) enum BandIndices {
    Fixed(i32),
    PerRow(Int32Array),
}

impl BandIndices {
    /// 1-based band index for row `i`. A null entry in the `band_index`
    /// argument defaults to 1, matching `RS_BandPixelType`'s convention.
    pub(crate) fn get(&self, i: usize) -> i32 {
        match self {
            BandIndices::Fixed(v) => *v,
            BandIndices::PerRow(arr) => {
                if arr.is_null(i) {
                    1
                } else {
                    arr.value(i)
                }
            }
        }
    }

    /// Resolve from an optional `band_index` argument. `None` (the argument
    /// was omitted) resolves to a fixed band index of 1.
    pub(crate) fn resolve(arg: Option<&ColumnarValue>, num_rows: usize) -> Result<Self> {
        let Some(arg) = arg else {
            return Ok(BandIndices::Fixed(1));
        };
        let array = arg.clone().cast_to(&DataType::Int32, None)?;
        let array = array.into_array(num_rows)?;
        Ok(BandIndices::PerRow(as_int32_array(&array)?.clone()))
    }
}

/// Extract band `band_index` (1-based) from `raster` into a fresh
/// single-band raster (zero-copy via [BandRef::copy_into]). Returns `Ok(None)`
/// if `band_index` is out of range, matching the null-on-invalid-index
/// convention used by `RS_BandPixelType`/`RS_BandNoDataValue`.
pub(crate) fn extract_single_band(
    raster: &dyn RasterRef,
    band_index: i32,
) -> Result<Option<StructArray>> {
    let num_bands = raster.num_bands();
    if band_index < 1 || band_index as usize > num_bands {
        return Ok(None);
    }
    let idx = (band_index - 1) as usize;

    let mut builder = RasterBuilder::new(1);
    builder
        .start_raster_from(raster, RasterOverrides::default())
        .map_err(|e| exec_datafusion_err!("{e}"))?;
    raster
        .band(idx)?
        .copy_into(
            &mut builder,
            BandOverrides {
                name: raster.band_name(idx),
                ..Default::default()
            },
        )
        .map_err(|e| exec_datafusion_err!("{e}"))?;
    builder
        .finish_band()
        .map_err(|e| exec_datafusion_err!("{e}"))?;
    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("{e}"))?;
    Ok(Some(
        builder.finish().map_err(|e| exec_datafusion_err!("{e}"))?,
    ))
}

/// Select `band_index` from each of `a` and `b` and append their combined
/// min/max as the next raster row in `builder`, or a null row if
/// `band_index` doesn't exist on either side.
pub(crate) fn append_combined_band(
    op: MinMaxOp,
    builder: &mut RasterBuilder,
    a: &dyn RasterRef,
    b: &dyn RasterRef,
    band_index: i32,
) -> Result<()> {
    let band_a = extract_single_band(a, band_index)?;
    let band_b = extract_single_band(b, band_index)?;
    match (band_a, band_b) {
        (Some(ra), Some(rb)) => {
            let ra_rasters =
                RasterStructArray::try_new(&ra).map_err(|e| exec_datafusion_err!("{e}"))?;
            let rb_rasters =
                RasterStructArray::try_new(&rb).map_err(|e| exec_datafusion_err!("{e}"))?;
            let ra_view = ra_rasters.get(0).map_err(|e| exec_datafusion_err!("{e}"))?;
            let rb_view = rb_rasters.get(0).map_err(|e| exec_datafusion_err!("{e}"))?;
            append_combined_single_band(op, builder, &ra_view, &rb_view)
        }
        _ => builder
            .append_null()
            .map_err(|e| exec_datafusion_err!("{e}")),
    }
}

/// Append the pixel-wise min/max of single-band rasters `a` and `b` as the
/// next raster row in `builder`. Both must have exactly one band (callers
/// produce that via [extract_single_band]) with matching dimension names,
/// shape, data type, and nodata value — mismatches are a clean execution
/// error rather than a silently wrong result.
pub(crate) fn append_combined_single_band(
    op: MinMaxOp,
    builder: &mut RasterBuilder,
    a: &dyn RasterRef,
    b: &dyn RasterRef,
) -> Result<()> {
    if a.num_bands() != 1 || b.num_bands() != 1 {
        return sedona_internal_err!(
            "append_combined_single_band: expected single-band rasters, got {} and {} bands",
            a.num_bands(),
            b.num_bands()
        );
    }

    builder
        .start_raster_from(a, RasterOverrides::default())
        .map_err(|e| exec_datafusion_err!("{e}"))?;
    let band_a = a.band(0)?;
    let band_b = b.band(0)?;
    merge_band(
        op,
        builder,
        a.band_name(0),
        band_a.as_ref(),
        band_b.as_ref(),
    )?;
    builder
        .finish_band()
        .map_err(|e| exec_datafusion_err!("{e}"))?;
    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("{e}"))
}

fn merge_band(
    op: MinMaxOp,
    builder: &mut RasterBuilder,
    band_name: Option<&str>,
    a: &dyn BandRef,
    b: &dyn BandRef,
) -> Result<()> {
    if a.data_type() != b.data_type() {
        return exec_err!(
            "cannot combine bands with different data types ({:?} vs {:?})",
            a.data_type(),
            b.data_type()
        );
    }
    if a.shape() != b.shape() || a.dim_names() != b.dim_names() {
        return exec_err!(
            "cannot combine bands with different shapes ({:?} {:?} vs {:?} {:?})",
            a.dim_names(),
            a.shape(),
            b.dim_names(),
            b.shape()
        );
    }
    if a.nodata() != b.nodata() {
        return exec_err!("cannot combine bands with different nodata values");
    }

    let nd_a = a.nd_buffer()?;
    let nd_b = b.nd_buffer()?;
    let bytes_a = nd_a.as_contiguous()?;
    let bytes_b = nd_b.as_contiguous()?;
    let combined = combine_bytes(op, a.data_type(), bytes_a, bytes_b, a.nodata())?;

    builder
        .start_band_nd(
            band_name,
            &a.dim_names(),
            a.shape(),
            a.data_type(),
            a.nodata(),
            None,
            None,
        )
        .map_err(|e| exec_datafusion_err!("{e}"))?;
    builder.band_data_writer().append_value(&combined);
    Ok(())
}

/// Combine two same-shape, same-dtype contiguous band buffers pixel-wise.
///
/// A pixel equal to `nodata` (exact byte match — nodata is a sentinel value
/// stored inline, not a separate mask) is treated as absent: the other side
/// wins outright, and a pixel where both sides are nodata stays nodata.
fn combine_bytes(
    op: MinMaxOp,
    data_type: BandDataType,
    a: &[u8],
    b: &[u8],
    nodata: Option<&[u8]>,
) -> Result<Vec<u8>> {
    if a.len() != b.len() {
        return sedona_internal_err!(
            "mismatched band byte length combining rasters: {} vs {}",
            a.len(),
            b.len()
        );
    }

    macro_rules! combine_as {
        ($t:ty) => {{
            let elem = std::mem::size_of::<$t>();
            if a.len() % elem != 0 {
                return sedona_internal_err!(
                    "band byte length {} is not a multiple of element size {} for {:?}",
                    a.len(),
                    elem,
                    data_type
                );
            }
            let mut out = Vec::with_capacity(a.len());
            for start in (0..a.len()).step_by(elem) {
                let end = start + elem;
                let a_bytes = &a[start..end];
                let b_bytes = &b[start..end];
                let a_is_nodata = nodata.is_some_and(|nd| nd == a_bytes);
                let b_is_nodata = nodata.is_some_and(|nd| nd == b_bytes);
                let winner: &[u8] = if a_is_nodata && b_is_nodata {
                    a_bytes
                } else if a_is_nodata {
                    b_bytes
                } else if b_is_nodata {
                    a_bytes
                } else {
                    let a_val = <$t>::from_le_bytes(a_bytes.try_into().unwrap());
                    let b_val = <$t>::from_le_bytes(b_bytes.try_into().unwrap());
                    // NaN (that isn't the nodata sentinel) is treated like a
                    // missing value too, matching np.fmin/np.fmax and
                    // xarray's default skipna=True reductions: the valid
                    // side wins outright regardless of which operand is
                    // NaN, and only both-NaN stays NaN. Without this,
                    // PartialOrd comparisons against NaN are always false,
                    // which would make `b` win unconditionally whenever it
                    // was NaN — silently discarding a valid `a` value.
                    if a_val.is_nan_value() && b_val.is_nan_value() {
                        a_bytes
                    } else if a_val.is_nan_value() {
                        b_bytes
                    } else if b_val.is_nan_value() {
                        a_bytes
                    } else if op.a_wins(a_val, b_val) {
                        a_bytes
                    } else {
                        b_bytes
                    }
                };
                out.extend_from_slice(winner);
            }
            out
        }};
    }

    Ok(match data_type {
        BandDataType::UInt8 => combine_as!(u8),
        BandDataType::Int8 => combine_as!(i8),
        BandDataType::UInt16 => combine_as!(u16),
        BandDataType::Int16 => combine_as!(i16),
        BandDataType::UInt32 => combine_as!(u32),
        BandDataType::Int32 => combine_as!(i32),
        BandDataType::UInt64 => combine_as!(u64),
        BandDataType::Int64 => combine_as!(i64),
        BandDataType::Float32 => combine_as!(f32),
        BandDataType::Float64 => combine_as!(f64),
    })
}
