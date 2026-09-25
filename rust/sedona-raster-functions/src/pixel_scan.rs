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

//! Whole-band pixel scans shared by the kernels that read every pixel of a
//! band rather than sampling one (see [`crate::sampling`] for that).
//!
//! [`scan_pixels`] walks a 2-D band in row-major order through its
//! [`NdBuffer`] strides, so a strided, reversed, or broadcast view is read in
//! place without being materialised. Pixels are handed out as raw bytes, so
//! [`NodataMatcher`] can test them against the band's nodata without decoding
//! and a caller decodes only the pixels whose values it needs.

use std::ops::ControlFlow;

use datafusion_common::{Result, exec_datafusion_err, exec_err};
use sedona_raster::error::RasterResultExt;
use sedona_raster::traits::{BandRef, NdBuffer};
use sedona_schema::raster::BandDataType;

/// The pixel buffer of a 2-D `(y, x)` band. Errors for a band with any other
/// dimensions, since a whole-band scan has no meaning for it. `func` names the
/// calling UDF for the error messages.
pub(crate) fn spatial_2d_buffer<'a>(func: &str, band: &'a dyn BandRef) -> Result<NdBuffer<'a>> {
    if !band.is_spatial_2d() {
        return exec_err!("{func} supports 2-D rasters only; band is not a 2-D (y, x) grid");
    }
    Ok(band.nd_buffer().context(func)?)
}

/// Visit every pixel of a 2-D band buffer in row-major order as
/// `(col, row, bytes)`, with `col` and `row` 0-based. The scan stops early when
/// `visit` returns [`ControlFlow::Break`].
///
/// The byte range the scan touches is bounds-checked once up front, so a
/// corrupt stride or offset is an error rather than a panic partway through.
pub(crate) fn scan_pixels(
    func: &str,
    buffer: &NdBuffer,
    mut visit: impl FnMut(i64, i64, &[u8]) -> ControlFlow<()>,
) -> Result<()> {
    let (Ok([height, width]), Ok([row_stride, col_stride])) = (
        <[i64; 2]>::try_from(buffer.shape.as_slice()),
        <[i64; 2]>::try_from(buffer.strides.as_slice()),
    ) else {
        return exec_err!("{func}: expected a 2-D band buffer");
    };
    if height == 0 || width == 0 {
        return Ok(());
    }

    let size = buffer.data_type.byte_size() as i64;
    let overflow = || exec_datafusion_err!("{func}: pixel byte offset overflow");
    // A stride can be negative (a reversed view), so the lowest and highest
    // bytes touched come from whichever end of each axis the stride points at.
    let row_span = row_stride.checked_mul(height - 1).ok_or_else(overflow)?;
    let col_span = col_stride.checked_mul(width - 1).ok_or_else(overflow)?;
    let offset = i64::try_from(buffer.offset).map_err(|_| overflow())?;
    let lowest = [row_span.min(0), col_span.min(0)]
        .into_iter()
        .try_fold(offset, i64::checked_add)
        .ok_or_else(overflow)?;
    let end = [row_span.max(0), col_span.max(0), size]
        .into_iter()
        .try_fold(offset, i64::checked_add)
        .ok_or_else(overflow)?;
    if lowest < 0 || end > buffer.buffer.len() as i64 {
        return exec_err!("{func}: band view reaches outside its buffer");
    }

    // Every offset below lies within [lowest, end), which was checked above.
    for row in 0..height {
        let row_start = offset + row * row_stride;
        for col in 0..width {
            let start = (row_start + col * col_stride) as usize;
            if visit(col, row, &buffer.buffer[start..start + size as usize]).is_break() {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Tests raw pixel bytes against a band's nodata value.
///
/// Integer pixels compare by bytes, which is exact at every width (a 64-bit
/// sentinel beyond 2^53 still matches, where comparing as `f64` could not even
/// represent it). Float pixels compare numerically, so `-0.0` matches a `0.0`
/// nodata and any NaN matches a NaN nodata.
pub(crate) struct NodataMatcher<'a> {
    nodata: &'a [u8],
    data_type: BandDataType,
}

impl<'a> NodataMatcher<'a> {
    /// The matcher for `band`, or `None` when the band has no nodata value.
    pub(crate) fn for_band(func: &str, band: &'a dyn BandRef) -> Result<Option<Self>> {
        let Some(nodata) = band.nodata() else {
            return Ok(None);
        };
        let data_type = band.data_type();
        if nodata.len() != data_type.byte_size() {
            return exec_err!(
                "{func}: nodata value is {} bytes, but a {data_type:?} pixel is {}",
                nodata.len(),
                data_type.byte_size()
            );
        }
        Ok(Some(Self { nodata, data_type }))
    }

    /// Whether `pixel` (one pixel's bytes, as handed out by [`scan_pixels`])
    /// holds the nodata value.
    pub(crate) fn matches(&self, pixel: &[u8]) -> bool {
        fn float_eq(pixel: f64, nodata: f64) -> bool {
            pixel == nodata || (pixel.is_nan() && nodata.is_nan())
        }
        match self.data_type {
            BandDataType::Float32 => float_eq(
                f32::from_le_bytes(pixel.try_into().unwrap()) as f64,
                f32::from_le_bytes(self.nodata.try_into().unwrap()) as f64,
            ),
            BandDataType::Float64 => float_eq(
                f64::from_le_bytes(pixel.try_into().unwrap()),
                f64::from_le_bytes(self.nodata.try_into().unwrap()),
            ),
            _ => pixel == self.nodata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
    use sedona_testing::raster_spec::RasterSpec;

    fn buffer_of(bytes: &[u8], shape: [i64; 2], strides: [i64; 2], offset: u64) -> NdBuffer<'_> {
        NdBuffer {
            buffer: bytes,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            offset,
            data_type: BandDataType::UInt8,
        }
    }

    fn scan(buffer: &NdBuffer) -> Result<Vec<(i64, i64, u8)>> {
        let mut seen = Vec::new();
        scan_pixels("test", buffer, |col, row, bytes| {
            seen.push((col, row, bytes[0]));
            ControlFlow::Continue(())
        })?;
        Ok(seen)
    }

    #[test]
    fn scans_row_major() {
        let bytes = [1, 2, 3, 4, 5, 6];
        let seen = scan(&buffer_of(&bytes, [2, 3], [3, 1], 0)).unwrap();
        assert_eq!(
            seen,
            vec![
                (0, 0, 1),
                (1, 0, 2),
                (2, 0, 3),
                (0, 1, 4),
                (1, 1, 5),
                (2, 1, 6)
            ]
        );
    }

    #[test]
    fn scans_reversed_and_strided_views_in_place() {
        // Rows reversed (negative row stride, offset at the last row) and every
        // other column (stride 2) of a 2x4 buffer.
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8];
        let seen = scan(&buffer_of(&bytes, [2, 2], [-4, 2], 4)).unwrap();
        assert_eq!(seen, vec![(0, 0, 5), (1, 0, 7), (0, 1, 1), (1, 1, 3)]);
    }

    #[test]
    fn empty_band_visits_nothing() {
        assert!(scan(&buffer_of(&[], [0, 3], [3, 1], 0)).unwrap().is_empty());
    }

    #[test]
    fn break_stops_the_scan() {
        let bytes = [1, 2, 3, 4];
        let mut visits = 0;
        scan_pixels("test", &buffer_of(&bytes, [2, 2], [2, 1], 0), |_, _, _| {
            visits += 1;
            ControlFlow::Break(())
        })
        .unwrap();
        assert_eq!(visits, 1);
    }

    #[test]
    fn out_of_bounds_view_errors() {
        let bytes = [1, 2, 3];
        let err = scan(&buffer_of(&bytes, [2, 2], [2, 1], 0)).unwrap_err();
        assert!(err.to_string().contains("outside its buffer"), "{err}");
        let err = scan(&buffer_of(&bytes, [2, 2], [-2, 1], 0)).unwrap_err();
        assert!(err.to_string().contains("outside its buffer"), "{err}");
    }

    fn matcher_matches(spec: RasterSpec, pixels: &[&[u8]]) -> Vec<bool> {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let matcher = NodataMatcher::for_band("test", band.as_ref())
            .unwrap()
            .unwrap();
        pixels.iter().map(|p| matcher.matches(p)).collect()
    }

    #[test]
    fn integer_nodata_matches_by_bytes() {
        // u64::MAX is beyond 2^53, so it could not be compared as an f64.
        let spec = RasterSpec::d2(1, 1).band_values(&[0u64]).nodata(u64::MAX);
        assert_eq!(
            matcher_matches(
                spec,
                &[&u64::MAX.to_le_bytes(), &(u64::MAX - 1).to_le_bytes()]
            ),
            vec![true, false]
        );
    }

    #[test]
    fn float_nodata_matches_numerically() {
        let spec = RasterSpec::d2(1, 1).band_values(&[0f64]).nodata(0f64);
        assert_eq!(
            matcher_matches(spec, &[&(-0f64).to_le_bytes(), &1f64.to_le_bytes()]),
            vec![true, false]
        );
        // A NaN pixel matches a NaN nodata whatever its payload.
        let spec = RasterSpec::d2(1, 1).band_values(&[0f32]).nodata(f32::NAN);
        let other_nan = f32::from_bits(f32::NAN.to_bits() | 1);
        assert_eq!(
            matcher_matches(spec, &[&other_nan.to_le_bytes(), &0f32.to_le_bytes()]),
            vec![true, false]
        );
    }

    #[test]
    fn band_without_nodata_has_no_matcher() {
        let array = RasterSpec::d2(1, 1).band_values(&[0u8]).build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        assert!(
            NodataMatcher::for_band("test", band.as_ref())
                .unwrap()
                .is_none()
        );
    }
}
