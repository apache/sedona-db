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

//! Analyzer rule that rewrites a top-level `RS_TileExplode(...)` projection
//! column into `UNNEST(RS_Tile(...))` plus a struct-flattening projection.
//!
//! `SELECT RS_TileExplode(rast, w, h) FROM t` parses to a `Projection` whose
//! single output column is the marker call's `Struct<x, y, tile>`. This rule
//! rewrites that into an `Unnest` of `RS_Tile(rast, w, h)`'s
//! `List<Struct<x, y, tile>>` output, wrapped in a `Projection` that lifts the
//! resulting struct into **top-level** `(x, y, tile)` columns (alongside any
//! pass-through sibling columns) — mirroring how a `SELECT id, UNNEST(arr)` plan
//! is a `Projection` over the row-multiplying `Unnest` node. `RS_Tile` shares the
//! tiling core with `RS_TileExplode` and already carries the tile grid
//! coordinates, so the two surfaces stay byte-identical by construction.
//!
//! This rewrite **changes the plan's output schema** (a one-column projection
//! becomes `(…siblings…, x, y, tile)`), which DataFusion forbids an
//! `OptimizerRule` from doing — every optimizer rule is checked against
//! `logically_equivalent_names_and_types`. The `Analyzer` has no per-rule schema
//! check (it is where `TypeCoercion` changes column types), so the schema-changing
//! rewrite belongs here.
//!
//! The rule is applied in two places, and is **idempotent** — once the marker is
//! lifted there is nothing left to match, so a second pass is a no-op:
//! - **Eagerly** on the unoptimized plan the SQL front-end builds (see
//!   `sedona::exec::create_plan_from_sql`), so the returned `DataFrame`'s
//!   `schema()` already reports the top-level `(x, y, tile)` columns before any
//!   execution — matching Sedona Spark's plan-time-honest generator schema. The
//!   native `Unnest`/`Projection` the rewrite produces has a schema DataFusion
//!   computes at plan-build time.
//! - As a registered `SessionStateBuilder::with_analyzer_rule` (appended *after*
//!   `TypeCoercion`), a safety net for plans built through other paths and so the
//!   optimize/execute path is unchanged.

use std::sync::Arc;

use arrow_schema::DataType;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{plan_err, Column, DFSchema, Result, UnnestOptions};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::expr_schema::ExprSchemable;
use datafusion_expr::{col, Expr, LogicalPlan, LogicalPlanBuilder, Projection, ScalarUDF};
use datafusion_functions::core::expr_ext::FieldAccessor;
use datafusion_functions_nested::expr_fn::make_array;
use datafusion_optimizer::analyzer::AnalyzerRule;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};

/// The registered name of the `RS_TileExplode` marker UDF.
const TILE_EXPLODE_NAME: &str = "rs_tileexplode";

/// Internal name for the `RS_Tile` list column the rewrite unnests. Deliberately
/// unlikely to collide with a user column so the outer projection can reference
/// the unnested struct unambiguously.
const TILES_ALIAS: &str = "__sedona_tiles";

/// The positions of the `RS_TileExplode` arguments within the marker call,
/// mirroring `RS_Tile`'s overload layout.
///
/// [`build_tile_explode_plan`] reads the tile parameters out of the argument
/// expressions through these positions when it assembles the `RS_Tile` call, so
/// the argument-order knowledge lives in one place (resolved via
/// [`infer_tile_arg_layout`]).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct TileArgLayout {
    /// Position of the raster argument (typically a column reference).
    pub raster: usize,
    /// Position of the `bandIndices` list (or scalar `bandIndex`) argument, when
    /// the band overload carries one.
    pub band: Option<usize>,
    /// Position of the `width` argument.
    pub width: usize,
    /// Position of the `height` argument.
    pub height: usize,
    /// Position of the `padWithNoData` argument, when this overload carries one.
    pub pad: Option<usize>,
    /// Position of the `noDataVal` argument, when this overload carries one.
    pub nodata: Option<usize>,
}

/// Analyzer rule that expands a top-level `RS_TileExplode(...)` projection column
/// into `UNNEST(RS_Tile(...))` + a struct-flattening projection and enforces the
/// generator's placement rules.
#[derive(Debug)]
pub struct TileExplodeAnalyzerRule {
    /// The `RS_Tile` UDF the rewrite calls to produce the tile list. Threaded in
    /// so this (dependency-light) crate does not depend on the GDAL-backed UDF
    /// crate that defines it.
    rs_tile: Arc<ScalarUDF>,
}

impl TileExplodeAnalyzerRule {
    pub fn new(rs_tile: Arc<ScalarUDF>) -> Self {
        Self { rs_tile }
    }

    /// Rewrite a `Projection` that carries a single top-level `RS_TileExplode(...)`
    /// column into `UNNEST(RS_Tile(...))` + a struct-flattening projection. Leaves
    /// every other plan node unchanged.
    fn lift_tile_explode_projection(&self, plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Projection(projection) = &plan else {
            return Ok(Transformed::no(plan));
        };

        // Locate the single top-level `RS_TileExplode` column, rejecting a nested
        // call (illegal placement) or a second generator (only one is allowed).
        let mut explode_index: Option<usize> = None;
        for (index, expr) in projection.expr.iter().enumerate() {
            if as_tile_explode(expr).is_some() {
                if explode_index.is_some() {
                    return plan_err!("RS_TileExplode may appear at most once in a SELECT list");
                }
                explode_index = Some(index);
            } else if contains_tile_explode(expr)? {
                return plan_err!(
                    "RS_TileExplode must be a single top-level column of a SELECT; it cannot be nested inside another expression"
                );
            }
        }
        let Some(explode_index) = explode_index else {
            return Ok(Transformed::no(plan));
        };

        // Re-bind by value so the projection's fields can be moved into the rewrite.
        let LogicalPlan::Projection(Projection { expr, input, .. }) = plan else {
            return sedona_internal_err!("TileExplode: expected a Projection after matching one");
        };

        let call = as_tile_explode(&expr[explode_index]).ok_or_else(|| {
            sedona_internal_datafusion_err!("TileExplode: the marker column vanished during lift")
        })?;
        let args = call.args.clone();

        // The pass-through siblings are the original projection columns other than
        // the explode column, in order; the shared builder appends (x, y, tile),
        // dropping the raster and any other unselected input column.
        let siblings: Vec<Expr> = expr
            .into_iter()
            .enumerate()
            .filter(|(index, _)| *index != explode_index)
            .map(|(_, sibling)| sibling)
            .collect();

        let lifted =
            build_tile_explode_plan(input.as_ref().clone(), args, siblings, &self.rs_tile)?;
        Ok(Transformed::yes(lifted))
    }
}

impl AnalyzerRule for TileExplodeAnalyzerRule {
    fn name(&self) -> &str {
        "sedona.tile_explode"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        let plan = plan
            .transform_down(|node| self.lift_tile_explode_projection(node))?
            .data;

        // Any `RS_TileExplode` call still present is in an illegal place: a
        // `WHERE`/`HAVING`/aggregate/`GROUP BY` expression, nested inside another
        // expression, or a second generator in the same `SELECT`. The legal
        // top-level projection column was removed by the lift above; only illegal
        // placements survive, so a surviving marker is a clear plan-time error.
        reject_surviving_tile_explode(&plan)?;
        Ok(plan)
    }
}

/// Build the `UNNEST(RS_Tile(...))` + struct-flattening projection plan that
/// replaces an `RS_TileExplode(...)` generator column.
///
/// `args` are the `RS_TileExplode` argument expressions (raster, `[band]`, width,
/// height, `[pad, [nodata]]`); `siblings` are the output columns carried through
/// ahead of the appended `(x, y, tile)`, in order; `rs_tile` is the `RS_Tile` UDF
/// used to produce the tile list. Shared by the SQL analyzer rewrite and the
/// Python `DataFrame.tile_explode` surface so both resolve the argument layout
/// and shape the plan the same way.
pub fn build_tile_explode_plan(
    input: LogicalPlan,
    args: Vec<Expr>,
    siblings: Vec<Expr>,
    rs_tile: &Arc<ScalarUDF>,
) -> Result<LogicalPlan> {
    let layout = infer_tile_arg_layout(&args, input.schema())?;

    // Assemble the `RS_Tile` arguments from the resolved layout. `RS_Tile` shares
    // the tiling core with `RS_TileExplode`, so its `List<Struct<x, y, tile>>`
    // carries exactly the tiles (with their grid coordinates) the generator emits
    // one per row.
    let mut rs_tile_args: Vec<Expr> = vec![args[layout.raster].clone()];
    if let Some(band) = layout.band {
        if second_is_band_list(&args, input.schema())? {
            // Already a `bandIndices` list — matches `RS_Tile`'s list overload.
            rs_tile_args.push(args[band].clone());
        } else {
            // `RS_Tile` carries only a `bandIndices` *list* overload, so a scalar
            // `bandIndex` is wrapped in a single-element list (identical tiles).
            rs_tile_args.push(make_array(vec![args[band].clone()]));
        }
    }
    rs_tile_args.push(args[layout.width].clone());
    rs_tile_args.push(args[layout.height].clone());
    if let Some(pad) = layout.pad {
        rs_tile_args.push(args[pad].clone());
    }
    if let Some(nodata) = layout.nodata {
        rs_tile_args.push(args[nodata].clone());
    }

    let tiles = Expr::ScalarFunction(ScalarFunction::new_udf(rs_tile.clone(), rs_tile_args))
        .alias(TILES_ALIAS);

    // Inner projection: every input column (qualifiers preserved) so the
    // pass-through siblings survive the unnest, followed by the tile list.
    let mut inner: Vec<Expr> = input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    inner.push(tiles);

    // Unnest the tile list with `preserve_nulls = false`, so a NULL or empty tile
    // list (e.g. from a NULL raster) contributes zero rows — matching Sedona
    // Spark's generator. The list column becomes a `Struct<x, y, tile>` column of
    // the same name.
    let unnest_options = UnnestOptions::new().with_preserve_nulls(false);

    // Outer projection: the pass-through siblings, then the tile struct flattened
    // into top-level (x, y, tile).
    let mut outer = siblings;
    let tiles_column = col(TILES_ALIAS);
    outer.push(tiles_column.clone().field("x").alias("x"));
    outer.push(tiles_column.clone().field("y").alias("y"));
    outer.push(tiles_column.field("tile").alias("tile"));

    LogicalPlanBuilder::from(input)
        .project(inner)?
        .unnest_column_with_options(Column::from_name(TILES_ALIAS), unnest_options)?
        .project(outer)?
        .build()
}

/// If `expr` is (an alias of) an `RS_TileExplode(...)` call, return the call.
fn as_tile_explode(expr: &Expr) -> Option<&ScalarFunction> {
    match expr {
        Expr::ScalarFunction(func) if func.func.name() == TILE_EXPLODE_NAME => Some(func),
        Expr::Alias(alias) => as_tile_explode(&alias.expr),
        _ => None,
    }
}

/// True if `expr` contains an `RS_TileExplode(...)` call anywhere within it.
fn contains_tile_explode(expr: &Expr) -> Result<bool> {
    expr.exists(|e| {
        Ok(matches!(e, Expr::ScalarFunction(func) if func.func.name() == TILE_EXPLODE_NAME))
    })
}

/// Map the `RS_TileExplode` argument expressions to their [`TileArgLayout`],
/// mirroring Sedona Spark's `RS_TileExplode` positional overloads: the no-band
/// shape `(raster, width, height, …)`, the scalar-band shape `(raster, bandIndex,
/// width, height, …)`, and the array-band shape `(raster, bandIndices, width,
/// height, …)`.
///
/// The band-carrying shapes shift `width`/`height` and the trailing optionals
/// one position to the right. The three are told apart by the argument after the
/// raster and its neighbors:
/// - a list in the second position is `bandIndices` (array-band);
/// - otherwise a leading integer is a scalar `bandIndex` when the pad slot
///   (position 3) holds an integer (`height`) rather than the boolean
///   `padWithNoData`, and is `width` when it holds a boolean or nothing.
///
/// A 3-argument call is always no-band; a 6-argument non-list call is always
/// scalar-band (no-band tops out at 5 arguments).
///
/// Exposed so a caller assembling the tile-explode plan directly (the Python
/// `DataFrame.tile_explode` surface) resolves the layout through this one shared
/// mapping rather than re-encoding the argument positions.
pub fn infer_tile_arg_layout(args: &[Expr], input_schema: &DFSchema) -> Result<TileArgLayout> {
    let count = args.len();
    if !(3..=6).contains(&count) {
        return plan_err!("RS_TileExplode expects between 3 and 6 arguments, got {count}");
    }

    // The scalar-band and array-band shapes both carry a band argument at
    // position 1 with the same downstream layout: `build_tile_explode_plan` wraps
    // a scalar `bandIndex` into a single-element `bandIndices` list, so `RS_Tile`
    // always resolves a list.
    let has_band = second_is_band_list(args, input_schema)?
        || second_is_scalar_band(args, count, input_schema)?;

    // Required positions: raster, [band], width, height. `base` is the index of
    // the first optional (padWithNoData), which the band shapes shift by one.
    let (band, width, height, base) = if has_band {
        (Some(1), 2, 3, 4)
    } else {
        (None, 1, 2, 3)
    };
    if count < base {
        return plan_err!(
            "RS_TileExplode band overload needs at least {base} arguments, got {count}"
        );
    }
    // Optional trailing positions: padWithNoData, then noDataVal.
    let pad = (count > base).then_some(base);
    let nodata = (count > base + 1).then_some(base + 1);

    Ok(TileArgLayout {
        raster: 0,
        band,
        width,
        height,
        pad,
        nodata,
    })
}

/// Whether the argument after the raster is a `bandIndices` list (the array-band
/// shape).
fn second_is_band_list(args: &[Expr], input_schema: &DFSchema) -> Result<bool> {
    Ok(matches!(
        args[1].get_type(input_schema)?,
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
    ))
}

/// Whether a non-list `(raster, Int, Int, …)` argument list carries a leading
/// scalar `bandIndex` (the scalar-band shape) rather than starting with
/// `width`/`height` (the no-band shape). A 3-argument call is always no-band; a
/// 6-argument call is always scalar-band (no-band tops out at 5); at 4 or 5
/// arguments an integer in the pad slot (position 3) is `height`, so the leading
/// integer is a `bandIndex`, whereas a boolean there is the no-band shape's
/// `padWithNoData` flag.
fn second_is_scalar_band(args: &[Expr], count: usize, input_schema: &DFSchema) -> Result<bool> {
    Ok(match count {
        3 => false,
        6 => true,
        _ => args[3].get_type(input_schema)?.is_integer(),
    })
}

/// Error if any `RS_TileExplode(...)` call survives in the analyzed plan (an
/// illegal placement the lift could not expand).
fn reject_surviving_tile_explode(plan: &LogicalPlan) -> Result<()> {
    let mut found = false;
    plan.apply_with_subqueries(|node| {
        node.apply_expressions(|expr| {
            if contains_tile_explode(expr)? {
                found = true;
                Ok(TreeNodeRecursion::Stop)
            } else {
                Ok(TreeNodeRecursion::Continue)
            }
        })?;
        Ok(if found {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;

    if found {
        return plan_err!(
            "RS_TileExplode may only appear as a single top-level column of a SELECT"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion_common::ScalarValue;
    use datafusion_expr::lit;

    /// An integer-list literal in the `bandIndices` position selects the
    /// band-subset overload.
    fn band_list() -> Expr {
        Expr::Literal(
            ScalarValue::List(ScalarValue::new_list_nullable(
                &[ScalarValue::Int32(Some(1))],
                &DataType::Int32,
            )),
            None,
        )
    }

    fn layout(args: &[Expr]) -> TileArgLayout {
        infer_tile_arg_layout(args, &DFSchema::empty()).unwrap()
    }

    #[test]
    fn all_bands_overloads_place_width_after_the_raster() {
        // (raster, width, height)
        assert_eq!(
            layout(&[col("r"), lit(2), lit(2)]),
            TileArgLayout {
                raster: 0,
                band: None,
                width: 1,
                height: 2,
                pad: None,
                nodata: None,
            }
        );
        // (raster, width, height, padWithNoData)
        assert_eq!(
            layout(&[col("r"), lit(2), lit(2), lit(true)]),
            TileArgLayout {
                raster: 0,
                band: None,
                width: 1,
                height: 2,
                pad: Some(3),
                nodata: None,
            }
        );
        // (raster, width, height, padWithNoData, noDataVal)
        assert_eq!(
            layout(&[col("r"), lit(2), lit(2), lit(true), lit(0.0)]),
            TileArgLayout {
                raster: 0,
                band: None,
                width: 1,
                height: 2,
                pad: Some(3),
                nodata: Some(4),
            }
        );
    }

    #[test]
    fn scalar_band_overloads_shift_positions_by_the_band_index() {
        // A scalar bandIndex (a leading integer followed by width/height) shifts
        // width/height and the trailing optionals by one, exactly like the
        // band-list shape. The scalar-band shape is distinguished from no-band by
        // an integer (height) rather than a boolean (pad) in position 3.
        // (raster, bandIndex, width, height)
        assert_eq!(
            layout(&[col("r"), lit(1), lit(2), lit(2)]),
            TileArgLayout {
                raster: 0,
                band: Some(1),
                width: 2,
                height: 3,
                pad: None,
                nodata: None,
            }
        );
        // (raster, bandIndex, width, height, padWithNoData)
        assert_eq!(
            layout(&[col("r"), lit(1), lit(2), lit(2), lit(true)]),
            TileArgLayout {
                raster: 0,
                band: Some(1),
                width: 2,
                height: 3,
                pad: Some(4),
                nodata: None,
            }
        );
        // (raster, bandIndex, width, height, padWithNoData, noDataVal)
        assert_eq!(
            layout(&[col("r"), lit(1), lit(2), lit(2), lit(true), lit(0.0)]),
            TileArgLayout {
                raster: 0,
                band: Some(1),
                width: 2,
                height: 3,
                pad: Some(4),
                nodata: Some(5),
            }
        );
    }

    #[test]
    fn band_subset_overloads_shift_positions_by_the_band_list() {
        // A list in the second position (bandIndices) shifts width/height and the
        // trailing optionals by one, even at the 4- and 5-argument counts the two
        // shapes share.
        // (raster, bandIndices, width, height)
        assert_eq!(
            layout(&[col("r"), band_list(), lit(2), lit(2)]),
            TileArgLayout {
                raster: 0,
                band: Some(1),
                width: 2,
                height: 3,
                pad: None,
                nodata: None,
            }
        );
        // (raster, bandIndices, width, height, padWithNoData)
        assert_eq!(
            layout(&[col("r"), band_list(), lit(2), lit(2), lit(true)]),
            TileArgLayout {
                raster: 0,
                band: Some(1),
                width: 2,
                height: 3,
                pad: Some(4),
                nodata: None,
            }
        );
        // (raster, bandIndices, width, height, padWithNoData, noDataVal)
        assert_eq!(
            layout(&[col("r"), band_list(), lit(2), lit(2), lit(true), lit(0.0)]),
            TileArgLayout {
                raster: 0,
                band: Some(1),
                width: 2,
                height: 3,
                pad: Some(4),
                nodata: Some(5),
            }
        );
    }

    #[test]
    fn rejects_out_of_range_argument_count() {
        let err = infer_tile_arg_layout(&[col("r"), lit(2)], &DFSchema::empty())
            .unwrap_err()
            .to_string();
        assert!(err.contains("between 3 and 6"), "unexpected error: {err}");
    }
}
