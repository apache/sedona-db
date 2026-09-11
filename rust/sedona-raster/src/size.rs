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

//! Metadata-only byte estimates for rasters.
//!
//! Every estimate here is `Σ_bands Π raw_source_shape × pixel bytes`,
//! computed from band metadata alone — the `data` column is never read.
//! That makes one formula serve both storage kinds: for an InDb band it is
//! an identity with the actual buffer length (the read boundary rejects a
//! mismatch as corruption), and for an OutDb band it is the size loading
//! it will allocate, which is what a memory budget needs to know *before*
//! anything is loaded.
//!
//! The estimate uses the **raw source shape, not the visible shape**:
//! resident bytes follow the source buffer, so a broadcast view does not
//! inflate the figure and a slice view does not shrink it. Bands that share
//! one buffer (zero-copy derivations) are each counted in full — an
//! over-estimate, which is the safe direction for a budget.

use crate::array::RasterStructArray;
use crate::error::RasterError;
use crate::traits::{BandRef, RasterRef};

/// Bytes the band's source buffer holds (InDb) or will hold once loaded
/// (OutDb): `Π raw_source_shape × data_type.byte_size()`.
pub fn estimated_band_bytes(band: &dyn BandRef) -> Result<u64, RasterError> {
    let source_shape = band.raw_source_shape();
    let byte_size = band.data_type().byte_size() as u64;
    source_shape
        .iter()
        .try_fold(byte_size, |acc, &dim| {
            u64::try_from(dim).ok().and_then(|dim| acc.checked_mul(dim))
        })
        .ok_or_else(|| {
            RasterError::Invalid(format!(
                "band byte count overflows u64 for source_shape {source_shape:?} × {:?}",
                band.data_type()
            ))
        })
}

/// Sum of [`estimated_band_bytes`] over the raster's bands.
pub fn estimated_raster_bytes(raster: &dyn RasterRef) -> Result<u64, RasterError> {
    let mut total = 0u64;
    for band_idx in 0..raster.num_bands() {
        let band = raster.band(band_idx)?;
        total = total
            .checked_add(estimated_band_bytes(band.as_ref())?)
            .ok_or_else(|| RasterError::Invalid("raster byte count overflows u64".to_string()))?;
    }
    Ok(total)
}

/// [`estimated_raster_bytes`] for every row of a raster column; null rows
/// estimate to 0.
pub fn estimated_row_bytes(rasters: &RasterStructArray<'_>) -> Result<Vec<u64>, RasterError> {
    (0..rasters.len())
        .map(|idx| {
            if rasters.is_null(idx) {
                Ok(0)
            } else {
                estimated_raster_bytes(&rasters.get(idx)?)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{RasterBuilder, StartBandArgs};
    use crate::view_entries::{ViewEntries, ViewEntry};
    use arrow_array::StructArray;
    use sedona_schema::raster::BandDataType;

    const TRANSFORM: [f64; 6] = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];

    enum Storage {
        InDb(Vec<u8>),
        OutDb,
    }

    /// One-row raster with a single `[y, x]` band of `source_shape` and
    /// `data_type`, optionally viewed.
    fn one_band(
        spatial_shape: &[i64],
        source_shape: &[i64],
        data_type: BandDataType,
        view: Option<&ViewEntries>,
        storage: Storage,
    ) -> StructArray {
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&TRANSFORM, &["y", "x"], spatial_shape, None)
            .unwrap();
        let outdb = matches!(storage, Storage::OutDb);
        b.start_band(StartBandArgs {
            view,
            outdb_uri: outdb.then_some("mock://band"),
            outdb_format: outdb.then_some("mock"),
            ..StartBandArgs::new(&["y", "x"], source_shape, data_type)
        })
        .unwrap();
        match storage {
            Storage::InDb(bytes) => b.band_data_writer().append_value(bytes),
            Storage::OutDb => b.band_data_writer().append_value([0u8; 0]),
        }
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.finish().unwrap()
    }

    fn row_bytes(array: &StructArray) -> u64 {
        let rasters = RasterStructArray::try_new(array).unwrap();
        estimated_raster_bytes(&rasters.get(0).unwrap()).unwrap()
    }

    #[test]
    fn indb_estimate_equals_buffer_length() {
        let pixels: Vec<u8> = (0..32).collect();
        let array = one_band(
            &[4, 8],
            &[4, 8],
            BandDataType::UInt8,
            None,
            Storage::InDb(pixels),
        );
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        assert_eq!(
            estimated_band_bytes(band.as_ref()).unwrap(),
            band.nd_buffer().unwrap().buffer.len() as u64
        );
        assert_eq!(row_bytes(&array), 32);
    }

    #[test]
    fn outdb_estimate_comes_from_metadata_without_loading() {
        // 512 × 512 float32 — the spatialbench tile — is 1 MiB once loaded.
        let array = one_band(
            &[512, 512],
            &[512, 512],
            BandDataType::Float32,
            None,
            Storage::OutDb,
        );
        assert_eq!(row_bytes(&array), 1024 * 1024);
    }

    #[test]
    fn estimate_uses_raw_source_shape_not_visible_shape() {
        // Source [4, 8] viewed down to row 1 (visible [1, 8]): the buffer is
        // still 32 bytes, so the estimate must not shrink to 8.
        let view = ViewEntries::new(vec![
            ViewEntry {
                source_axis: 0,
                start: 1,
                step: 1,
                steps: 1,
            },
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 8,
            },
        ]);
        let array = one_band(
            &[1, 8],
            &[4, 8],
            BandDataType::UInt8,
            Some(&view),
            Storage::OutDb,
        );
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();
        assert_eq!(raster.band(0).unwrap().shape(), &[1, 8]);
        assert_eq!(row_bytes(&array), 32);
    }

    #[test]
    fn multi_band_and_null_rows() {
        let mut b = RasterBuilder::new(3);
        // Row 0: two bands, 2×3 UInt8 + 2×3 Int16 = 6 + 12.
        b.start_raster_nd(&TRANSFORM, &["y", "x"], &[2, 3], None)
            .unwrap();
        for dt in [BandDataType::UInt8, BandDataType::Int16] {
            b.start_band(StartBandArgs {
                outdb_uri: Some("mock://band"),
                outdb_format: Some("mock"),
                ..StartBandArgs::new(&["y", "x"], &[2, 3], dt)
            })
            .unwrap();
            b.band_data_writer().append_value([0u8; 0]);
            b.finish_band().unwrap();
        }
        b.finish_raster().unwrap();
        // Row 1: null.
        b.append_null().unwrap();
        // Row 2: no bands.
        b.start_raster_nd(&TRANSFORM, &["y", "x"], &[2, 3], None)
            .unwrap();
        b.finish_raster().unwrap();
        let array = b.finish().unwrap();

        let rasters = RasterStructArray::try_new(&array).unwrap();
        assert_eq!(estimated_row_bytes(&rasters).unwrap(), vec![18, 0, 0]);
    }

    #[test]
    fn estimate_rejects_overflow_instead_of_wrapping() {
        // 2^31 × 2^31 float64 = 2^65 bytes.
        let big = 1i64 << 31;
        let array = one_band(
            &[big, big],
            &[big, big],
            BandDataType::Float64,
            None,
            Storage::OutDb,
        );
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let err = estimated_raster_bytes(&rasters.get(0).unwrap()).unwrap_err();
        assert!(err.to_string().contains("overflows"), "{err}");
    }
}
