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

//! RS_Tile UDF - split a raster into a grid of tiles.
//!
//! Mirrors Sedona Spark's `RS_Tile`: a raster is cut into a grid of
//! `tile_width` x `tile_height` tiles. Spark returns an `Array<Raster>`;
//! SedonaDB returns a scalar `List<Struct<x, y, tile>>` — one list per input
//! raster, each item additionally carrying the tile's grid position `(x, y)`
//! (tile column, tile row) — that callers `UNNEST` to get one row per tile.
//! This mirrors how [`crate::rs_polygonize`] returns `List<Struct<geom, value>>`.
//!
//! The last row/column of tiles may not divide evenly: with
//! `pad_with_nodata = true` the edge tiles are padded to the full tile size with
//! a nodata fill; otherwise the smaller edge tile is emitted as-is.

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, NullBufferBuilder, OffsetBufferBuilder};
use arrow_array::{Array, ArrayRef, ListArray, StructArray};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::cast::{as_boolean_array, as_float64_array, as_int64_array, as_list_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};

use crate::tiling::{append_tile, resolve_band_indices, tile_grid_dims, TileParams, TileWindow};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::{ArgMatcher, TypeMatcher};

/// RS_Tile() scalar UDF implementation.
///
/// Mirrors Sedona Spark's `RS_Tile` positional overloads (Spark expands the
/// trailing optionals `padWithNoData` / `noDataVal` into sub-overloads):
/// - `RS_Tile(raster, width, height[, padWithNoData[, noDataVal]])`
/// - `RS_Tile(raster, bandIndices, width, height[, padWithNoData[, noDataVal]])`
///
/// Each of the two shapes has three sub-overloads (no optional, with
/// `padWithNoData`, with both), for six kernels total. The shapes are told
/// apart by the type of the argument after the raster: an integer (`width`) for
/// the all-bands shape, or an integer list (`bandIndices`) for the band-subset
/// shape.
pub fn rs_tile_udf() -> SedonaScalarUDF {
    let kernel = |band_arg: BandArg, arg_count: usize| {
        Arc::new(RsTile {
            band_arg,
            arg_count,
        })
    };
    SedonaScalarUDF::new(
        "rs_tile",
        vec![
            // RS_Tile(raster, width, height[, padWithNoData[, noDataVal]])
            kernel(BandArg::All, 3),
            kernel(BandArg::All, 4),
            kernel(BandArg::All, 5),
            // RS_Tile(raster, bandIndices, width, height[, padWithNoData[, noDataVal]])
            kernel(BandArg::Array, 4),
            kernel(BandArg::Array, 5),
            kernel(BandArg::Array, 6),
        ],
        Volatility::Immutable,
    )
    // Reads band pixels, so the planner materializes OutDb rasters via
    // RS_EnsureLoaded before this kernel runs. The output is a list of in-db
    // tiles (not a top-level raster), so RETURNS_BYTES does not apply.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

/// Which band-selector argument a signature carries after the raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandArg {
    /// No band argument — every band is tiled: `(raster, width, height, ...)`.
    All,
    /// An array of 1-based band indices: `(raster, bandIndices, width, height, ...)`.
    Array,
}

/// The argument position of each parameter for a given signature shape.
struct ArgLayout {
    /// The band-selector argument (scalar index or list), when present.
    band: Option<usize>,
    width: usize,
    height: usize,
    /// `padWithNoData`, when present in this overload.
    pad: Option<usize>,
    /// `noDataVal`, when present in this overload.
    nodata: Option<usize>,
}

/// Kernel implementation for RS_Tile.
#[derive(Debug)]
struct RsTile {
    /// The band-selector shape this kernel matches.
    band_arg: BandArg,
    /// Number of arguments in the matched signature.
    arg_count: usize,
}

impl SedonaScalarKernel for RsTile {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // Build the matcher ladder from the shape: raster, the band selector (an
        // integer list for `bandIndices`, absent for the all-bands shape), then
        // width/height, then the trailing optionals this overload includes.
        let mut matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>> = vec![ArgMatcher::is_raster()];
        match self.band_arg {
            BandArg::All => {}
            BandArg::Array => matchers.push(ArgMatcher::is_list_of(ArgMatcher::is_integer())),
        }
        matchers.push(ArgMatcher::is_integer()); // width
        matchers.push(ArgMatcher::is_integer()); // height
        let layout = self.layout();
        if layout.pad.is_some() {
            matchers.push(ArgMatcher::is_boolean());
        }
        if layout.nodata.is_some() {
            matchers.push(ArgMatcher::is_numeric());
        }
        if matchers.len() != self.arg_count {
            return sedona_internal_err!(
                "RS_Tile: built {} matchers for arg_count {}",
                matchers.len(),
                self.arg_count
            );
        }

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(tile_list_type()?));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke_batch_from_args(arg_types, args, &SedonaType::Arrow(DataType::Null), 0, None)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let num_iterations = RasterExecutor::num_iterations_over(args);
        let layout = self.layout();

        // width / height expanded to arrays and iterated in lockstep with the
        // raster (a NULL in either yields a NULL list for that row).
        let tile_w_array = args[layout.width]
            .clone()
            .cast_to(&DataType::Int64, None)?
            .into_array(num_iterations)?;
        let tile_w_array = as_int64_array(&tile_w_array)?.clone();
        let tile_h_array = args[layout.height]
            .clone()
            .cast_to(&DataType::Int64, None)?
            .into_array(num_iterations)?;
        let tile_h_array = as_int64_array(&tile_h_array)?.clone();

        // pad_with_nodata: absent defaults to false; when present, a NULL yields
        // a NULL row (an unresolved flag, like a NULL raster).
        let pad_array = match layout.pad {
            Some(pos) => args[pos]
                .clone()
                .cast_to(&DataType::Boolean, None)?
                .into_array(num_iterations)?,
            None => ScalarValue::Boolean(Some(false)).to_array_of_size(num_iterations)?,
        };
        let pad_array = as_boolean_array(&pad_array)?.clone();

        // no_data_val: absent means "not supplied" (None); NULL is the same
        // meaningful sentinel, so it stays an Option rather than nulling the row.
        let nodata_array = match layout.nodata {
            Some(pos) => args[pos]
                .clone()
                .cast_to(&DataType::Float64, None)?
                .into_array(num_iterations)?,
            None => ScalarValue::Float64(None).to_array_of_size(num_iterations)?,
        };
        let nodata_array = as_float64_array(&nodata_array)?.clone();

        // Band selection array, materialized once for the shape that carries one.
        // Array: a list column whose (Int64) values are the 1-based indices per
        // row.
        let (band_list_array, band_list_values) = match (self.band_arg, layout.band) {
            (BandArg::Array, Some(pos)) => {
                let target = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
                let array = args[pos]
                    .clone()
                    .cast_to(&target, None)?
                    .into_array(num_iterations)?;
                let list = as_list_array(&array)?.clone();
                let values = as_int64_array(list.values())?.clone();
                (Some(list), Some(values))
            }
            _ => (None, None),
        };

        let mut x_builder = Int32Builder::new();
        let mut y_builder = Int32Builder::new();
        let mut rast_builder = RasterBuilder::new(num_iterations);
        let mut list_offsets = OffsetBufferBuilder::<i32>::new(num_iterations);
        let mut valid_list_items = NullBufferBuilder::new(num_iterations);
        // Reused per-row scratch for the resolved 1-based band indices, so the
        // Array shape does not allocate a fresh Vec every row.
        let mut band_scratch: Vec<i64> = Vec::new();

        let exec_arg_types = [arg_types[0].clone()];
        let exec_args = [args[0].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        executor.execute_raster_void(|i, raster_opt| {
            let tile_width = (!tile_w_array.is_null(i)).then(|| tile_w_array.value(i));
            let tile_height = (!tile_h_array.is_null(i)).then(|| tile_h_array.value(i));
            let pad = (!pad_array.is_null(i)).then(|| pad_array.value(i));

            // Resolve the row's selected 1-based band indices into `band_scratch`.
            // `None` selects every band; a NULL band argument (a NULL list)
            // yields a NULL row.
            let bands: Option<&[i64]> = match self.band_arg {
                BandArg::All => None,
                BandArg::Array => {
                    let (Some(list), Some(values)) =
                        (band_list_array.as_ref(), band_list_values.as_ref())
                    else {
                        return sedona_internal_err!(
                            "RS_Tile: band list arrays missing for the Array overload"
                        );
                    };
                    if list.is_null(i) {
                        valid_list_items.append_null();
                        list_offsets.push_length(0);
                        return Ok(());
                    }
                    let offsets = list.value_offsets();
                    let (start, end) = (offsets[i] as usize, offsets[i + 1] as usize);
                    band_scratch.clear();
                    for j in start..end {
                        if values.is_null(j) {
                            return exec_err!("RS_Tile: band index must not be null");
                        }
                        band_scratch.push(values.value(j));
                    }
                    Some(band_scratch.as_slice())
                }
            };

            let (Some(raster), Some(tile_width), Some(tile_height), Some(pad_with_nodata)) =
                (raster_opt, tile_width, tile_height, pad)
            else {
                valid_list_items.append_null();
                list_offsets.push_length(0);
                return Ok(());
            };

            let params = TileParams {
                bands,
                pad_with_nodata,
                nodata: (!nodata_array.is_null(i)).then(|| nodata_array.value(i)),
            };

            let num_tiles = explode_raster_tiles(
                raster,
                tile_width,
                tile_height,
                &params,
                &mut rast_builder,
                &mut x_builder,
                &mut y_builder,
            )?;

            valid_list_items.append_non_null();
            list_offsets.push_length(num_tiles);
            Ok(())
        })?;

        let list_array = assemble_tile_list(
            x_builder,
            y_builder,
            rast_builder,
            list_offsets,
            valid_list_items,
        )?;

        RasterExecutor::finish_over(args, Arc::new(list_array))
    }
}

impl RsTile {
    /// The argument positions for this kernel's signature shape.
    fn layout(&self) -> ArgLayout {
        match self.band_arg {
            BandArg::All => ArgLayout {
                band: None,
                width: 1,
                height: 2,
                pad: (self.arg_count >= 4).then_some(3),
                nodata: (self.arg_count >= 5).then_some(4),
            },
            BandArg::Array => ArgLayout {
                band: Some(1),
                width: 2,
                height: 3,
                pad: (self.arg_count >= 5).then_some(4),
                nodata: (self.arg_count >= 6).then_some(5),
            },
        }
    }
}

/// The `List<Struct<x, y, tile>>` element struct fields, mirroring Spark's
/// `(x int, y int, tile raster)` generator output.
fn tile_struct_fields() -> Result<Fields> {
    let tile_field = RASTER
        .to_storage_field("tile", false)
        .map_err(|e| exec_datafusion_err!("RS_Tile: {e}"))?;
    Ok(Fields::from(vec![
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Int32, false),
        tile_field,
    ]))
}

/// The full `List<Struct<x, y, tile>>` return type.
fn tile_list_type() -> Result<DataType> {
    Ok(DataType::List(Arc::new(Field::new(
        "item",
        DataType::Struct(tile_struct_fields()?),
        true,
    ))))
}

/// Split one raster into a grid of tiles, appending each tile's grid position to
/// `x_builder`/`y_builder` and its raster to `rast_builder`. Returns the number
/// of tiles produced (`num_tile_x * num_tile_y`).
///
/// Tiles are emitted row-major, matching Spark: `num_tile_x =
/// ceil(width / tile_width)`, `num_tile_y = ceil(height / tile_height)`, and the
/// tile at grid position `(x, y)` covers source pixels
/// `[x * tile_width, ..)` x `[y * tile_height, ..)`.
fn explode_raster_tiles(
    raster: &RasterRefImpl<'_>,
    tile_width: i64,
    tile_height: i64,
    params: &TileParams<'_>,
    rast_builder: &mut RasterBuilder,
    x_builder: &mut Int32Builder,
    y_builder: &mut Int32Builder,
) -> Result<usize> {
    // noDataVal is a padding-only knob, so reject it up front when not padding
    // (a caller mistake). The remaining request validation and grid sizing live
    // in the reusable tiling core.
    if params.nodata.is_some() && !params.pad_with_nodata {
        return exec_err!("RS_Tile: nodata is only meaningful with pad_with_nodata = true");
    }

    let width = raster.width()?;
    let height = raster.height()?;
    let (num_tile_x, num_tile_y) = tile_grid_dims(width, height, tile_width, tile_height)?;
    // width/height are non-negative and tile sizes are >= 1 (validated above).
    let width = width as usize;
    let height = height as usize;
    let tile_w = tile_width as usize;
    let tile_h = tile_height as usize;

    let band_indices = resolve_band_indices(params.bands, raster.num_bands())?;

    for tile_y in 0..num_tile_y {
        for tile_x in 0..num_tile_x {
            let window = TileWindow::new(
                tile_x,
                tile_y,
                tile_w,
                tile_h,
                width,
                height,
                params.pad_with_nodata,
            );
            append_tile(raster, &band_indices, &window, params, rast_builder)?;
            x_builder.append_value(tile_x as i32);
            y_builder.append_value(tile_y as i32);
        }
    }

    Ok(num_tile_x * num_tile_y)
}

/// Assemble the per-row `x`/`y`/`tile` builders and list offsets into the
/// `List<Struct<x, y, tile>>` output.
fn assemble_tile_list(
    mut x_builder: Int32Builder,
    mut y_builder: Int32Builder,
    rast_builder: RasterBuilder,
    list_offsets: OffsetBufferBuilder<i32>,
    mut valid_list_items: NullBufferBuilder,
) -> Result<ListArray> {
    let x_array: ArrayRef = Arc::new(x_builder.finish());
    let y_array: ArrayRef = Arc::new(y_builder.finish());
    let tile_array: ArrayRef = Arc::new(
        rast_builder
            .finish()
            .map_err(|e| exec_datafusion_err!("RS_Tile: failed to build tiles: {e}"))?,
    );

    let fields = tile_struct_fields()?;
    let element_struct =
        StructArray::try_new(fields.clone(), vec![x_array, y_array, tile_array], None)
            .map_err(|e| exec_datafusion_err!("RS_Tile: failed to build tile struct: {e}"))?;

    let list_field = Arc::new(Field::new("item", DataType::Struct(fields), true));
    ListArray::try_new(
        list_field,
        list_offsets.finish(),
        Arc::new(element_struct),
        valid_list_items.finish(),
    )
    .map_err(|e| exec_datafusion_err!("RS_Tile: failed to build tile list: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion_common::cast::{as_int32_array, as_list_array, as_struct_array};
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{assert_rasters_equal, raster_array, RasterSpec};
    use sedona_testing::testers::ScalarUdfTester;

    /// The optional tiling parameters for the core-tiling helper tests.
    fn params(bands: Option<&[i64]>, pad_with_nodata: bool, nodata: Option<f64>) -> TileParams<'_> {
        TileParams {
            bands,
            pad_with_nodata,
            nodata,
        }
    }

    /// The registered RS_Tile UDF, used to exercise overload dispatch
    /// end to end (the matcher ladder plus the positional argument reading).
    fn udf() -> ScalarUDF {
        crate::register::default_function_set()
            .scalar_udf("rs_tile")
            .expect("rs_tile is registered")
            .clone()
            .into()
    }

    /// A scalar `ScalarValue::List` of 1-based band indices, as Spark's
    /// `array(...)` produces for the `bandIndices` overload.
    fn band_index_list(indices: &[i32]) -> ColumnarValue {
        let values: Vec<ScalarValue> = indices
            .iter()
            .map(|&i| ScalarValue::Int32(Some(i)))
            .collect();
        ColumnarValue::Scalar(ScalarValue::List(ScalarValue::new_list_nullable(
            &values,
            &DataType::Int32,
        )))
    }

    /// The `tile` column of a single-row (scalar) List result.
    fn scalar_tiles(result: &ColumnarValue) -> ArrayRef {
        let ColumnarValue::Scalar(ScalarValue::List(list)) = result else {
            panic!("expected a scalar List result, got {result:?}");
        };
        let element = as_struct_array(list.values()).unwrap();
        element.column(2).clone()
    }

    /// A 5x3 EPSG-less raster, origin (0, 3), north-up 1x1 pixels, one UInt8
    /// band with values 1..=15 (row-major). The odd extent makes both the last
    /// column (width 5 vs tile 2) and the last row (height 3 vs tile 2) partial,
    /// exercising the edge-tile paths. Expected tile pixels below come from the
    /// numpy reference in the PR description.
    fn source_5x3() -> RasterSpec {
        RasterSpec::d2(5, 3)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    }

    /// A north-up UInt8 tile: origin (`ulx`, `uly`), 1x1 pixels, no nodata.
    fn tile(width: i64, height: i64, ulx: f64, uly: f64, values: &[u8]) -> RasterSpec {
        RasterSpec::d2(width, height)
            .crs(None)
            .transform([ulx, 1.0, 0.0, uly, 0.0, -1.0])
            .band_values(values)
    }

    #[test]
    fn return_type_is_list_of_tile_structs() {
        let kernel = RsTile {
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
        let Some(SedonaType::Arrow(DataType::List(field))) = return_type else {
            panic!("expected List return type, got {return_type:?}");
        };
        // The list element carries the (x, y, tile) struct with the raster
        // extension field.
        let DataType::Struct(fields) = field.data_type() else {
            panic!("expected Struct list item");
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

    #[test]
    fn udf_over_array_packages_list_and_nulls() {
        // End-to-end through the kernel: a two-row raster column (one raster,
        // one NULL) with scalar tile sizes. The output is a List row per input:
        // 6 tiles for row 0, a NULL list for row 1.
        let kernel = RsTile {
            band_arg: BandArg::All,
            arg_count: 3,
        };
        let raster_col = raster_array(vec![Some(source_5x3()), None]);
        let result = kernel
            .invoke_batch(
                &[
                    RASTER,
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Arrow(DataType::Int32),
                ],
                &[
                    ColumnarValue::Array(Arc::new(raster_col)),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                ],
            )
            .unwrap();

        let ColumnarValue::Array(array) = result else {
            panic!("expected an array result");
        };
        let list = as_list_array(&array).unwrap();
        assert_eq!(list.len(), 2);
        assert!(!list.is_null(0));
        assert_eq!(list.value_length(0), 6, "row 0 should have 6 tiles");
        assert!(list.is_null(1), "row 1 (null raster) should be a null list");
        assert_eq!(list.value_length(1), 0);

        // The flattened element struct holds the 6 tiles of row 0.
        let element = as_struct_array(list.values()).unwrap();
        let xs = as_int32_array(element.column(0)).unwrap();
        let ys = as_int32_array(element.column(1)).unwrap();
        assert_eq!(
            (0..6)
                .map(|i| (xs.value(i), ys.value(i)))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        );
        let tiles: ArrayRef = element.column(2).clone();
        assert_rasters_equal(
            &tiles,
            &[
                Some(tile(2, 2, 0.0, 3.0, &[1, 2, 6, 7])),
                Some(tile(2, 2, 2.0, 3.0, &[3, 4, 8, 9])),
                Some(tile(1, 2, 4.0, 3.0, &[5, 10])),
                Some(tile(2, 1, 0.0, 1.0, &[11, 12])),
                Some(tile(2, 1, 2.0, 1.0, &[13, 14])),
                Some(tile(1, 1, 4.0, 1.0, &[15])),
            ],
        );
    }

    /// A 2x2, three-band UInt8 raster (band 1 = 1..=4, band 2 = 10..=40,
    /// band 3 = 100..=103), so band selection and ordering are observable.
    fn three_band_2x2() -> RasterSpec {
        RasterSpec::d2(2, 2)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .band_values(&[100u8, 101, 102, 103])
    }

    #[test]
    fn overload_ladder_matches_by_argument_type() {
        // The two shapes are told apart by the argument after the raster: an
        // integer (width) for the all-bands shape, or an integer list
        // (bandIndices) for the band-subset shape. There is no single-`bandIndex`
        // (scalar integer) shape, so `(raster, int, int, int)` matches nothing.
        let int = SedonaType::Arrow(DataType::Int32);
        let boolean = SedonaType::Arrow(DataType::Boolean);
        let double = SedonaType::Arrow(DataType::Float64);
        let int_list = SedonaType::Arrow(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int32,
            true,
        ))));
        let matches = |band_arg: BandArg, arg_count: usize, args: &[SedonaType]| {
            RsTile {
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

        // (raster, int, int, bool) is the all-bands + padWithNoData overload.
        let with_pad = [RASTER, int.clone(), int.clone(), boolean.clone()];
        assert!(matches(BandArg::All, 4, &with_pad));

        // A list in the band position selects the bandIndices overload only.
        let band_indices = [RASTER, int_list.clone(), int.clone(), int.clone()];
        assert!(matches(BandArg::Array, 4, &band_indices));
        assert!(!matches(BandArg::All, 4, &band_indices));

        // The single-`bandIndex` shape was dropped: an integer band selector
        // followed by an integer height, i.e. (raster, int, int, int), is not a
        // valid signature for either shape.
        let scalar_band = [RASTER, int.clone(), int.clone(), int.clone()];
        assert!(!matches(BandArg::All, 4, &scalar_band));
        assert!(!matches(BandArg::Array, 4, &scalar_band));

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

    #[test]
    fn bandindices_array_overload_selects_and_orders_bands() {
        // RS_Tile(raster, bandIndices, width, height): the array overload
        // keeps exactly the listed bands, in the listed order (3 then 1).
        let tester = ScalarUdfTester::new(
            udf(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Int32,
                    true,
                )))),
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
        );
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(three_band_2x2().scalar()),
                band_index_list(&[3, 1]),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))), // width
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))), // height
            ])
            .unwrap();
        assert_rasters_equal(
            &scalar_tiles(&result),
            &[Some(
                RasterSpec::d2(2, 2)
                    .crs(None)
                    .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                    .band_values(&[100u8, 101, 102, 103])
                    .band_values(&[1u8, 2, 3, 4]),
            )],
        );
    }

    #[test]
    fn nodataval_supplies_nodata_for_padding_when_band_has_none() {
        // RS_Tile(raster, width, height, padWithNoData, noDataVal): the
        // band has no nodata, so noDataVal (7) supplies the pad fill and becomes
        // the tile's recorded nodata. A 3x1 raster tiled 2x1 pads the edge tile.
        let source = RasterSpec::d2(3, 1)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3]);
        let tester = ScalarUdfTester::new(
            udf(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(source.scalar()),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))), // width
                ColumnarValue::Scalar(ScalarValue::Int32(Some(1))), // height
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))), // padWithNoData
                ColumnarValue::Scalar(ScalarValue::Float64(Some(7.0))), // noDataVal
            ])
            .unwrap();
        assert_rasters_equal(
            &scalar_tiles(&result),
            &[
                // Interior tile: no padding, so noDataVal is not applied and it
                // keeps the source nodata (none).
                Some(tile(2, 1, 0.0, 1.0, &[1, 2])),
                // Edge tile padded to width 2 with the supplied nodata 7.
                Some(tile(2, 1, 2.0, 1.0, &[3, 7]).nodata(7u8)),
            ],
        );
    }

    #[test]
    fn null_band_indices_yields_null_row() {
        // A NULL bandIndices argument (Array overload) casts to a null list and
        // propagates to a NULL list row rather than erroring on the cast or the
        // offset lookup.
        let kernel = RsTile {
            band_arg: BandArg::Array,
            arg_count: 4,
        };
        let result = kernel
            .invoke_batch(
                &[
                    RASTER,
                    SedonaType::Arrow(DataType::Null),
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Arrow(DataType::Int32),
                ],
                &[
                    ColumnarValue::Scalar(source_5x3().scalar()),
                    ColumnarValue::Scalar(ScalarValue::Null),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                ],
            )
            .unwrap();
        let ColumnarValue::Scalar(ScalarValue::List(list)) = result else {
            panic!("expected a scalar list result, got {result:?}");
        };
        assert!(list.is_null(0), "a NULL bandIndices yields a NULL list row");
    }

    #[test]
    fn nodata_without_padding_errors() {
        // noDataVal supplied with padWithNoData = false is a caller mistake.
        let err = explode_one_error(&source_5x3(), 2, 2, params(None, false, Some(5.0)));
        assert!(
            err.contains("only meaningful with pad_with_nodata"),
            "unexpected error: {err}"
        );
    }

    /// Drive `explode_raster_tiles` and return the error string (helper for the
    /// error test exercising the driver's up-front request validation).
    fn explode_one_error(
        spec: &RasterSpec,
        tile_width: i64,
        tile_height: i64,
        tile_params: TileParams<'_>,
    ) -> String {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();
        explode_raster_tiles(
            &raster,
            tile_width,
            tile_height,
            &tile_params,
            &mut RasterBuilder::new(1),
            &mut Int32Builder::new(),
            &mut Int32Builder::new(),
        )
        .unwrap_err()
        .to_string()
    }
}
