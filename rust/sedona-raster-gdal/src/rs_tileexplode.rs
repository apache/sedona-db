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

//! RS_TileExplode marker UDF - split a raster into a grid of tiles, one tile per
//! output row.
//!
//! Mirrors Sedona Spark's `RS_TileExplode` generator: it emits one row per tile
//! with top-level columns `(x, y, tile)`, where `RS_Tile` returns a
//! `List<Struct<x, y, tile>>` cell that the caller `UNNEST`s. Its argument
//! surface matches Sedona Spark's `RS_TileExplode` positional overloads verbatim
//! (Spark parity): `RS_Tile`'s all-bands and `bandIndices`-list shapes plus the
//! scalar `bandIndex` shape that Spark's generator adds and `RS_Tile` omits.
//!
//! This UDF is a **planning marker only**. Registering it lets the SQL binder
//! accept `RS_TileExplode(rast, w, h, …)` and resolve its argument types before
//! the tile-explode analyzer rule rewrites the call into a streaming
//! `TileExplodeExec`. Its kernel therefore never executes: `invoke_batch`
//! returns an internal error, the last-resort net for a call the rule missed
//! (templated on `st_dump.rs`'s guard).

use std::sync::Arc;

use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};

use crate::rs_tile::tile_struct_fields;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster_functions::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::{ArgMatcher, TypeMatcher};

/// RS_TileExplode() marker scalar UDF.
///
/// Mirrors Sedona Spark's `RS_TileExplode` positional overloads (Spark parity):
/// the all-bands shape `RS_TileExplode(raster, width, height[, padWithNoData[,
/// noDataVal]])`, the scalar-band shape `RS_TileExplode(raster, bandIndex, width,
/// height[, padWithNoData[, noDataVal]])`, and the band-subset shape
/// `RS_TileExplode(raster, bandIndices, width, height[, padWithNoData[,
/// noDataVal]])`. The three are told apart by the argument after the raster: an
/// integer list is `bandIndices`; otherwise a leading integer is a `bandIndex`
/// when a fourth integer (`height`) follows, and `width` when a boolean
/// `padWithNoData` (or nothing) follows. Unlike `RS_Tile`, Spark's generator
/// carries the scalar-`bandIndex` overload, so it is present here but not on
/// `RS_Tile`.
///
/// Its return type is the single-tile `Struct<x, y, tile>` a lifted tile carries
/// (not `RS_Tile`'s `List<...>` wrapper) so the binder accepts the call before
/// the analyzer rule fires. The kernel never runs.
pub fn rs_tileexplode_udf() -> SedonaScalarUDF {
    let kernel = |band_arg: BandArg, arg_count: usize| {
        Arc::new(RsTileExplode {
            band_arg,
            arg_count,
        })
    };
    SedonaScalarUDF::new(
        "rs_tileexplode",
        vec![
            // RS_TileExplode(raster, width, height[, padWithNoData[, noDataVal]])
            kernel(BandArg::All, 3),
            kernel(BandArg::All, 4),
            kernel(BandArg::All, 5),
            // RS_TileExplode(raster, bandIndex, width, height[, padWithNoData[, noDataVal]])
            kernel(BandArg::Scalar, 4),
            kernel(BandArg::Scalar, 5),
            kernel(BandArg::Scalar, 6),
            // RS_TileExplode(raster, bandIndices, width, height[, padWithNoData[, noDataVal]])
            kernel(BandArg::Array, 4),
            kernel(BandArg::Array, 5),
            kernel(BandArg::Array, 6),
        ],
        Volatility::Immutable,
    )
    // Tiling reads band pixels, so the tile-explode planner materializes OutDb
    // rasters before tiling — the same contract RS_Tile carries.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

/// Which band-selector argument a signature carries after the raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandArg {
    /// No band argument — every band is tiled: `(raster, width, height, ...)`.
    All,
    /// A single 1-based band index: `(raster, bandIndex, width, height, ...)`.
    Scalar,
    /// An array of 1-based band indices: `(raster, bandIndices, width, height, ...)`.
    Array,
}

/// Marker kernel for RS_TileExplode. It resolves a return type for binding but
/// never executes.
#[derive(Debug)]
struct RsTileExplode {
    /// The band-selector shape this kernel matches.
    band_arg: BandArg,
    /// Number of arguments in the matched signature.
    arg_count: usize,
}

impl RsTileExplode {
    /// Whether this overload carries a trailing `padWithNoData` / `noDataVal`,
    /// which determines how many matchers the ladder has. The scalar- and
    /// array-band shapes both carry one extra leading band argument, so their
    /// counts are shifted by one relative to the all-bands shape.
    fn has_pad(&self) -> bool {
        match self.band_arg {
            BandArg::All => self.arg_count >= 4,
            BandArg::Scalar | BandArg::Array => self.arg_count >= 5,
        }
    }

    fn has_nodata(&self) -> bool {
        match self.band_arg {
            BandArg::All => self.arg_count >= 5,
            BandArg::Scalar | BandArg::Array => self.arg_count >= 6,
        }
    }
}

impl SedonaScalarKernel for RsTileExplode {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // Build the same matcher ladder as RS_Tile: raster, the band selector
        // (an integer list for `bandIndices`, absent for the all-bands shape),
        // width/height, then the trailing optionals this overload includes.
        let mut matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>> = vec![ArgMatcher::is_raster()];
        match self.band_arg {
            BandArg::All => {}
            BandArg::Scalar => matchers.push(ArgMatcher::is_integer()), // bandIndex
            BandArg::Array => matchers.push(ArgMatcher::is_list_of(ArgMatcher::is_integer())),
        }
        matchers.push(ArgMatcher::is_integer()); // width
        matchers.push(ArgMatcher::is_integer()); // height
        if self.has_pad() {
            matchers.push(ArgMatcher::is_boolean());
        }
        if self.has_nodata() {
            matchers.push(ArgMatcher::is_numeric());
        }
        if matchers.len() != self.arg_count {
            return sedona_internal_err!(
                "RS_TileExplode: built {} matchers for arg_count {}",
                matchers.len(),
                self.arg_count
            );
        }

        // The single-tile struct, not RS_Tile's List wrapper: the analyzer lifts
        // this call into top-level (x, y, tile) columns, one row per tile.
        let matcher = ArgMatcher::new(
            matchers,
            SedonaType::Arrow(DataType::Struct(tile_struct_fields()?)),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!(
            "rs_tileexplode kernel must be replaced by the tile-explode analyzer rule"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::Field;
    use sedona_schema::datatypes::RASTER;

    /// Assert the overload ladder tells the no-band, scalar-band, and array-band
    /// shapes apart by argument type. The scalar `bandIndex` shape is the one
    /// `RS_TileExplode` adds over `RS_Tile`, so its disambiguation (a leading
    /// integer followed by a fourth integer `height`, versus the boolean
    /// `padWithNoData` that follows `width`) is pinned here.
    #[test]
    fn overload_ladder_matches_by_argument_type() {
        let int = SedonaType::Arrow(DataType::Int32);
        let boolean = SedonaType::Arrow(DataType::Boolean);
        let double = SedonaType::Arrow(DataType::Float64);
        let int_list = SedonaType::Arrow(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int32,
            true,
        ))));
        let matches = |band_arg: BandArg, arg_count: usize, args: &[SedonaType]| {
            RsTileExplode {
                band_arg,
                arg_count,
            }
            .return_type(args)
            .unwrap()
            .is_some()
        };

        // (raster, int, int) is the all-bands shape.
        assert!(matches(
            BandArg::All,
            3,
            &[RASTER, int.clone(), int.clone()]
        ));

        // (raster, int, int, bool) is the all-bands + padWithNoData overload, told
        // apart from scalar-band by the boolean in the pad slot.
        let with_pad = [RASTER, int.clone(), int.clone(), boolean.clone()];
        assert!(matches(BandArg::All, 4, &with_pad));
        assert!(!matches(BandArg::Scalar, 4, &with_pad));

        // (raster, int, int, int) is the scalar-band shape (bandIndex, width,
        // height): a fourth integer in the pad slot is `height`, not `pad`, so it
        // matches scalar-band only.
        let scalar_band = [RASTER, int.clone(), int.clone(), int.clone()];
        assert!(matches(BandArg::Scalar, 4, &scalar_band));
        assert!(!matches(BandArg::All, 4, &scalar_band));
        assert!(!matches(BandArg::Array, 4, &scalar_band));

        // The fully-expanded scalar-band overload:
        // (raster, int, int, int, bool, double).
        let scalar_band_full = [
            RASTER,
            int.clone(),
            int.clone(),
            int.clone(),
            boolean.clone(),
            double.clone(),
        ];
        assert!(matches(BandArg::Scalar, 6, &scalar_band_full));

        // A list in the band position selects the bandIndices overload only.
        let band_indices = [RASTER, int_list.clone(), int.clone(), int.clone()];
        assert!(matches(BandArg::Array, 4, &band_indices));
        assert!(!matches(BandArg::All, 4, &band_indices));
        assert!(!matches(BandArg::Scalar, 4, &band_indices));

        // The fully-expanded bandIndices overload:
        // (raster, list, int, int, bool, double).
        let band_indices_full = [
            RASTER,
            int_list.clone(),
            int.clone(),
            int.clone(),
            boolean.clone(),
            double.clone(),
        ];
        assert!(matches(BandArg::Array, 6, &band_indices_full));
    }

    /// The return type is the single-tile `Struct<x, y, tile>`, not a `List`.
    #[test]
    fn return_type_is_single_tile_struct() {
        let kernel = RsTileExplode {
            band_arg: BandArg::All,
            arg_count: 3,
        };
        let return_type = kernel
            .return_type(&[
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ])
            .unwrap();
        let Some(SedonaType::Arrow(DataType::Struct(fields))) = return_type else {
            panic!("expected a single-tile Struct return type, got {return_type:?}");
        };
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name(), "x");
        assert_eq!(fields[1].name(), "y");
        assert_eq!(fields[2].name(), "tile");
        assert_eq!(
            SedonaType::from_storage_field(&fields[2]).unwrap(),
            RASTER,
            "tile field must be a raster"
        );
    }

    /// The marker kernel never executes: it errors as a last-resort net.
    #[test]
    fn kernel_invocation_errors() {
        let kernel = RsTileExplode {
            band_arg: BandArg::All,
            arg_count: 3,
        };
        let err = kernel.invoke_batch(&[], &[]).unwrap_err().to_string();
        assert!(
            err.contains("must be replaced by the tile-explode analyzer rule"),
            "unexpected error: {err}"
        );
    }
}
