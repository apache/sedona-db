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

//! Rewrite lazy tile list expansion into streaming execution.

use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::unnest::UnnestExec;
use sedona_expr::scalar_udf::SedonaScalarUDF;
use std::sync::Arc;

/// Replace the list-producing projection and its single unnest with a cursor.
/// Runs before RasterMaterializationRule, so the argument projection it creates
/// receives the usual byte-bounded input loading and source distribution.
#[derive(Debug)]
pub struct RasterTileRule {
    create_exec: fn(&ProjectionExec, &UnnestExec) -> Result<Arc<dyn ExecutionPlan>>,
}

impl RasterTileRule {
    /// Supply the raster implementation without coupling the planner to GDAL.
    pub fn new(
        create_exec: fn(&ProjectionExec, &UnnestExec) -> Result<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        Self { create_exec }
    }
}

impl PhysicalOptimizerRule for RasterTileRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(unnest) = node.downcast_ref::<UnnestExec>() else {
                return Ok(Transformed::no(node));
            };
            let [list] = unnest.list_column_indices() else {
                return Ok(Transformed::no(node));
            };
            if list.depth != 1 || !unnest.struct_column_indices().is_empty() {
                return Ok(Transformed::no(node));
            }
            // DataFusion may merge partitions between the projection and unnest.
            // Expand per partition instead, then merge the emitted tile batches.
            let merge = unnest.input().downcast_ref::<CoalescePartitionsExec>();
            if merge.is_some_and(|merge| merge.fetch().is_some()) {
                return Ok(Transformed::no(node));
            }
            let input = merge.map_or(unnest.input(), |merge| merge.input());
            let Some(projection) = input.downcast_ref::<ProjectionExec>() else {
                return Ok(Transformed::no(node));
            };
            let tile_index = list.index_in_input_schema;
            let Some(tile_expr) = projection.expr().get(tile_index) else {
                return Ok(Transformed::no(node));
            };
            let Some(function) = tile_expr.expr.downcast_ref::<ScalarFunctionExpr>() else {
                return Ok(Transformed::no(node));
            };
            // Do not rewrite arbitrary user functions sharing the SQL name.
            if function.fun().name() != "rs_lazytile"
                || function
                    .fun()
                    .inner()
                    .downcast_ref::<SedonaScalarUDF>()
                    .is_none()
            {
                return Ok(Transformed::no(node));
            }
            let mut result = (self.create_exec)(projection, unnest)?;
            if merge.is_some() {
                result = Arc::new(CoalescePartitionsExec::new(result));
            }
            Ok(Transformed::yes(result))
        })
        .data()
    }

    fn name(&self) -> &str {
        "sedona.stream_raster_tiles"
    }
    fn schema_check(&self) -> bool {
        true
    }
}
