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

//! Logical extension node for the `RS_TileExplode` streaming tile generator.

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use arrow_schema::{DataType, Field};
use datafusion_common::{DFSchema, DFSchemaRef, Result};
use datafusion_expr::logical_plan::UserDefinedLogicalNodeCore;
use datafusion_expr::{Expr, LogicalPlan};
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::RASTER;

/// The positions of the `RS_TileExplode` arguments within
/// [`TileExplodePlanNode::args`], mirroring `RS_Tile`'s overload layout.
///
/// The physical planner reads the tile parameters out of the node's argument
/// expressions through these positions, so the node stays self-describing
/// without depending on the (GDAL-backed) UDF crate that resolves the overload.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct TileArgLayout {
    /// Position of the raster argument (typically a column reference).
    pub raster: usize,
    /// Position of the `bandIndices` list argument, when the band-subset shape
    /// carries one.
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

/// The three columns `RS_TileExplode` appends to its input: the tile grid
/// position `(x, y)` and the tile raster, all non-nullable (every emitted row
/// carries a concrete tile).
pub fn tile_explode_appended_fields() -> Result<Vec<Field>> {
    Ok(vec![
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Int32, false),
        RASTER.to_storage_field("tile", false)?,
    ])
}

/// Logical extension node used as a planning hook for `RS_TileExplode`.
///
/// Carries the single input plan, the tile-generator argument expressions (with
/// their [`TileArgLayout`]), and the precomputed output schema — the input's
/// fields followed by the appended `(x, y, tile)` columns. The tile-explode
/// extension planner recognizes it and produces a streaming `TileExplodeExec`.
#[derive(PartialEq, Eq, Hash)]
pub struct TileExplodePlanNode {
    pub input: LogicalPlan,
    pub args: Vec<Expr>,
    pub layout: TileArgLayout,
    pub schema: DFSchemaRef,
}

impl TileExplodePlanNode {
    /// Build a node from its input, the tile-generator argument expressions, and
    /// their layout, computing the output schema (input fields ++ `x`, `y`,
    /// `tile`).
    pub fn try_new(input: LogicalPlan, args: Vec<Expr>, layout: TileArgLayout) -> Result<Self> {
        let schema = tile_explode_output_schema(input.schema())?;
        Ok(Self {
            input,
            args,
            layout,
            schema,
        })
    }
}

/// The output schema of a tile-explode node: every input field (qualifiers
/// preserved) followed by the appended `(x, y, tile)` columns.
fn tile_explode_output_schema(input: &DFSchema) -> Result<DFSchemaRef> {
    let mut qualified_fields: Vec<_> = input
        .iter()
        .map(|(qualifier, field)| (qualifier.cloned(), Arc::clone(field)))
        .collect();
    for field in tile_explode_appended_fields()? {
        qualified_fields.push((None, Arc::new(field)));
    }
    let schema = DFSchema::new_with_metadata(qualified_fields, input.metadata().clone())?;
    Ok(Arc::new(schema))
}

// Manual implementation needed because of the `schema` field. Comparison
// excludes it, matching `SpatialJoinPlanNode`.
impl PartialOrd for TileExplodePlanNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        #[derive(PartialEq, PartialOrd)]
        struct Comparable<'a> {
            input: &'a LogicalPlan,
            args: &'a Vec<Expr>,
            layout: &'a TileArgLayout,
        }
        let comparable_self = Comparable {
            input: &self.input,
            args: &self.args,
            layout: &self.layout,
        };
        let comparable_other = Comparable {
            input: &other.input,
            args: &other.args,
            layout: &other.layout,
        };
        comparable_self
            .partial_cmp(&comparable_other)
            .filter(|cmp| *cmp != Ordering::Equal || self == other)
    }
}

impl fmt::Debug for TileExplodePlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        UserDefinedLogicalNodeCore::fmt_for_explain(self, f)
    }
}

impl UserDefinedLogicalNodeCore for TileExplodePlanNode {
    fn name(&self) -> &str {
        "TileExplode"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        self.args.clone()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let args: Vec<String> = self.args.iter().map(|e| e.to_string()).collect();
        write!(f, "TileExplode: args=[{}]", args.join(", "))
    }

    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        // Request every input column so projection pushdown does not prune the
        // sibling columns the node replicates per tile. See
        // `SpatialJoinPlanNode::necessary_children_exprs` and
        // https://github.com/apache/datafusion/pull/20393.
        let input_indices: Vec<usize> = (0..self.input.schema().fields().len()).collect();
        Some(vec![input_indices])
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if inputs.len() != 1 {
            return sedona_internal_err!("TileExplodePlanNode expects 1 input");
        }
        if exprs.len() != self.args.len() {
            return sedona_internal_err!(
                "TileExplodePlanNode expects {} exprs, got {}",
                self.args.len(),
                exprs.len()
            );
        }
        Ok(Self {
            input: inputs.swap_remove(0),
            args: exprs,
            layout: self.layout.clone(),
            schema: Arc::clone(&self.schema),
        })
    }
}
