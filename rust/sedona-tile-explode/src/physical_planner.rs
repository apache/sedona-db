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

//! [`DefaultTileExplodePhysicalPlanner`] — resolves a [`TileExplodePlanNode`]'s
//! argument expressions into plan-time-constant tile parameters and builds a
//! [`TileExplodeExec`].

use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::DataType;
use datafusion::execution::session_state::SessionState;
use datafusion_common::cast::{as_int64_array, as_list_array};
use datafusion_common::{plan_err, Result, ScalarValue};
use datafusion_expr::Expr;
use datafusion_physical_plan::ExecutionPlan;

use sedona_query_planner::tile_explode_node::{TileArgLayout, TileExplodePlanNode};
use sedona_query_planner::tile_explode_physical_planner::TileExplodePhysicalPlanner;

use crate::exec::{TileExplodeExec, TileExplodeExecArgs};

/// The default tile-explode planner: it resolves the node's literal tile
/// arguments and produces a streaming [`TileExplodeExec`].
#[derive(Debug, Default)]
pub struct DefaultTileExplodePhysicalPlanner {}

impl DefaultTileExplodePhysicalPlanner {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TileExplodePhysicalPlanner for DefaultTileExplodePhysicalPlanner {
    fn plan_tile_explode(
        &self,
        node: &TileExplodePlanNode,
        physical_input: Arc<dyn ExecutionPlan>,
        _session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        // OutDb pixel loading is not wired through here yet: the tile-explode
        // analyzer rule removes the `RS_TileExplode` marker before the
        // `EnsureLoaded` optimizer rule can wrap the raster, so an OutDb input
        // reaches `TileExplodeExec` unloaded. Rather than reading empty/garbage
        // bytes, the exec rejects an unloaded OutDb raster with a clear runtime
        // error (see `exec::ensure_bands_loaded`). Materializing the bands here
        // via the async loader `session_state` exposes is tracked follow-up work.
        let args = resolve_exec_args(&node.args, &node.layout, &physical_input)?;
        let exec = TileExplodeExec::try_new(physical_input, args)?;
        Ok(Some(Arc::new(exec)))
    }
}

/// Resolve the node's argument expressions into plan-time constants.
fn resolve_exec_args(
    args: &[Expr],
    layout: &TileArgLayout,
    physical_input: &Arc<dyn ExecutionPlan>,
) -> Result<TileExplodeExecArgs> {
    let raster_column = resolve_raster_column(arg_at(args, layout.raster)?, physical_input)?;
    let tile_width = literal_i64(arg_at(args, layout.width)?, "width")?;
    let tile_height = literal_i64(arg_at(args, layout.height)?, "height")?;
    let pad_with_nodata = match layout.pad {
        Some(pos) => literal_bool(arg_at(args, pos)?, "padWithNoData")?,
        None => false,
    };
    let nodata = match layout.nodata {
        Some(pos) => literal_opt_f64(arg_at(args, pos)?, "noDataVal")?,
        None => None,
    };
    let bands = match layout.band {
        Some(pos) => literal_band_indices(arg_at(args, pos)?)?,
        None => None,
    };

    Ok(TileExplodeExecArgs {
        raster_column,
        tile_width,
        tile_height,
        pad_with_nodata,
        nodata,
        bands,
    })
}

fn arg_at(args: &[Expr], pos: usize) -> Result<&Expr> {
    args.get(pos).ok_or_else(|| {
        datafusion_common::DataFusionError::Plan(format!(
            "TileExplode: argument position {pos} is out of range for {} arguments",
            args.len()
        ))
    })
}

/// Resolve the raster argument (a column reference) to its index in the physical
/// input schema.
fn resolve_raster_column(expr: &Expr, physical_input: &Arc<dyn ExecutionPlan>) -> Result<usize> {
    let Expr::Column(column) = expr else {
        return plan_err!("TileExplode: the raster argument must be a column, got {expr}");
    };
    physical_input
        .schema()
        .index_of(column.name())
        .map_err(|_| {
            datafusion_common::DataFusionError::Plan(format!(
                "TileExplode: raster column '{}' not found in the input",
                column.name()
            ))
        })
}

fn literal(expr: &Expr, name: &str) -> Result<ScalarValue> {
    match expr {
        Expr::Literal(scalar, _) => Ok(scalar.clone()),
        // Constant folding an argument like `ARRAY[1]` leaves a literal wrapped in
        // a name-preserving alias (`List([1]) AS make_array(1)`); peel it.
        Expr::Alias(alias) => literal(&alias.expr, name),
        other => plan_err!("TileExplode: {name} must be a constant, got {other}"),
    }
}

fn literal_i64(expr: &Expr, name: &str) -> Result<i64> {
    match literal(expr, name)?.cast_to(&DataType::Int64)? {
        ScalarValue::Int64(Some(value)) => Ok(value),
        other => plan_err!("TileExplode: {name} must be a non-null integer, got {other:?}"),
    }
}

fn literal_bool(expr: &Expr, name: &str) -> Result<bool> {
    match literal(expr, name)?.cast_to(&DataType::Boolean)? {
        ScalarValue::Boolean(Some(value)) => Ok(value),
        other => plan_err!("TileExplode: {name} must be a non-null boolean, got {other:?}"),
    }
}

fn literal_opt_f64(expr: &Expr, name: &str) -> Result<Option<f64>> {
    match literal(expr, name)?.cast_to(&DataType::Float64)? {
        ScalarValue::Float64(value) => Ok(value),
        other => plan_err!("TileExplode: {name} must be numeric, got {other:?}"),
    }
}

/// Resolve a constant band selector into 1-based band indices. The selector is
/// either a `bandIndices` list (the array-band overload) or a scalar `bandIndex`
/// integer (the scalar-band overload); a scalar maps to a single-element list so
/// the tiling core sees the same shape. A NULL literal selects all bands (kept as
/// `None`); an empty list is kept (the tiling core maps it to "all bands",
/// matching RS_Tile).
fn literal_band_indices(expr: &Expr) -> Result<Option<Vec<i64>>> {
    let scalar = literal(expr, "bandIndices")?;
    if scalar.is_null() {
        return Ok(None);
    }
    // The scalar-band overload passes a bare integer `bandIndex`; wrap it in a
    // single-element band list, which produces the same tiles as `[bandIndex]`.
    if !matches!(
        scalar.data_type(),
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _)
    ) {
        return match scalar.cast_to(&DataType::Int64)? {
            ScalarValue::Int64(Some(band)) => Ok(Some(vec![band])),
            other => {
                plan_err!("TileExplode: bandIndex must be a non-null integer, got {other:?}")
            }
        };
    }
    let array = scalar.to_array()?;
    let list = as_list_array(&array)?;
    if list.is_null(0) {
        return Ok(None);
    }
    let values = list.value(0);
    let values = arrow::compute::cast(&values, &DataType::Int64)?;
    let ints = as_int64_array(&values)?;
    let mut out = Vec::with_capacity(ints.len());
    for i in 0..ints.len() {
        if ints.is_null(i) {
            return plan_err!("TileExplode: bandIndices must not contain a null");
        }
        out.push(ints.value(i));
    }
    Ok(Some(out))
}
