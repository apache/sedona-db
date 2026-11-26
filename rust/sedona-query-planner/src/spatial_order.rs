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

//! Use the registered spatial ordering kernel for geometry and geography sort keys.

use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{DFSchema, Result};
use datafusion_expr::expr_rewriter::NamePreserver;
use datafusion_expr::{Expr, ExprSchemable, LogicalPlan, SortExpr};
use datafusion_optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use sedona_schema::datatypes::SedonaType;

#[derive(Default, Debug)]
pub struct SpatialOrderOptimizerRule;

impl OptimizerRule for SpatialOrderOptimizerRule {
    fn name(&self) -> &str {
        "sedona.spatial_order"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::Sort(mut sort) => {
                let changed = rewrite_sorts(&mut sort.expr, sort.input.schema(), config)?;
                Ok(Transformed::new_transformed(
                    LogicalPlan::Sort(sort),
                    changed,
                ))
            }
            LogicalPlan::Aggregate(_) | LogicalPlan::Window(_) => {
                let schema = plan.inputs()[0].schema().clone();
                let names = NamePreserver::new(&plan);
                plan.map_expressions(|expr| {
                    let name = names.save(&expr);
                    expr.transform_up(|mut expr| {
                        let changed = match &mut expr {
                            Expr::AggregateFunction(function) => {
                                rewrite_sorts(&mut function.params.order_by, &schema, config)?
                            }
                            Expr::WindowFunction(function) => {
                                rewrite_sorts(&mut function.params.order_by, &schema, config)?
                            }
                            _ => false,
                        };
                        Ok(Transformed::new_transformed(expr, changed))
                    })?
                    .map_data(|expr| Ok(name.restore(expr)))
                })
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

fn is_geometry(expr: &Expr, schema: &DFSchema) -> Result<bool> {
    // get_type() alone loses the extension metadata distinguishing WKB from binary.
    let (_, field) = expr.to_field(schema)?;
    Ok(matches!(
        SedonaType::from_storage_field(&field)?,
        SedonaType::Wkb(_, _) | SedonaType::WkbLarge(_, _) | SedonaType::WkbView(_, _)
    ))
}

fn rewrite_sorts(
    sorts: &mut [SortExpr],
    schema: &DFSchema,
    config: &dyn OptimizerConfig,
) -> Result<bool> {
    let mut changed = false;
    for sort in sorts {
        if !is_geometry(&sort.expr, schema)? {
            continue;
        }
        let Some(registry) = config.function_registry() else {
            continue;
        };
        let order = registry.udf("sd_order")?.call(vec![sort.expr.clone()]);
        // Without a spatial kernel (e.g. a build without S2), SD_Order is an
        // identity function. Keep storage ordering in that case. This also
        // prevents repeated wrapping, including after CSE hoists sort keys.
        if is_geometry(&order, schema)? {
            continue;
        }
        sort.expr = order;
        changed = true;
    }
    Ok(changed)
}
