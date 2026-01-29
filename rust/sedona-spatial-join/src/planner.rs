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

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use arrow_schema::Schema;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion_common::tree_node::Transformed;
use datafusion_common::{plan_err, DFSchema, DFSchemaRef, NullEquality, Result};
use datafusion_expr::logical_plan::{
    Extension, UserDefinedLogicalNode, UserDefinedLogicalNodeCore,
};
use datafusion_expr::{Expr, JoinConstraint, JoinType, LogicalPlan, Operator};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion_physical_plan::joins::NestedLoopJoinExec;

use crate::exec::SpatialJoinExec;
use crate::optimizer::{is_spatial_predicate_supported, transform_join_filter};
use sedona_common::option::SedonaOptions;

pub struct SedonaSpatialQueryPlanner;

impl fmt::Debug for SedonaSpatialQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SedonaSpatialQueryPlanner").finish()
    }
}

#[async_trait]
impl QueryPlanner for SedonaSpatialQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let physical_planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(SpatialJoinPlanner {})]);
        physical_planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

/// Register logical rewrite and a query planner that can produce [`SpatialJoinExec`]
/// directly from a rewritten extension node.
pub fn register_spatial_join_planner(builder: SessionStateBuilder) -> SessionStateBuilder {
    builder
        .with_optimizer_rule(Arc::new(SpatialJoinLogicalRewrite::default()))
        .with_query_planner(Arc::new(SedonaSpatialQueryPlanner))
}

#[derive(Default, Debug)]
struct SpatialJoinLogicalRewrite;

impl OptimizerRule for SpatialJoinLogicalRewrite {
    fn name(&self) -> &str {
        "spatial_join_logical_rewrite"
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
        let options = config.options();
        let Some(ext) = options.extensions.get::<SedonaOptions>() else {
            return Ok(Transformed::no(plan));
        };
        if !ext.spatial_join.enable {
            return Ok(Transformed::no(plan));
        }

        let LogicalPlan::Join(join) = &plan else {
            return Ok(Transformed::no(plan));
        };

        // v1: only rewrite joins that already have a spatial predicate in `filter`.
        let Some(filter) = join.filter.as_ref() else {
            return Ok(Transformed::no(plan));
        };

        let spatial_predicate_names = collect_spatial_predicate_names(filter);
        if spatial_predicate_names.is_empty() {
            return Ok(Transformed::no(plan));
        }

        // Join with with equi-join condition and spatial join condition. Only handle it
        // when the join condition contains ST_KNN. KNN join is not a regular join and
        // ST_KNN is also not a regular predicate. It must be handled by our spatial join exec.
        if !join.on.is_empty() && !spatial_predicate_names.contains("st_knn") {
            return Ok(Transformed::no(plan));
        }

        let schema = Arc::clone(&join.schema);
        let node = SpatialJoinPlanNode {
            left: join.left.as_ref().clone(),
            right: join.right.as_ref().clone(),
            join_type: join.join_type,
            filter: filter.clone(),
            schema,
            join_constraint: join.join_constraint,
            null_equality: join.null_equality,
        };

        Ok(Transformed::yes(LogicalPlan::Extension(Extension {
            node: Arc::new(node),
        })))
    }
}

/// Check if a given logical expression contains a spatial predicate component or not. We assume that the given
/// `expr` evaluates to a boolean value and originates from a filter logical node.
fn collect_spatial_predicate_names(expr: &Expr) -> HashSet<String> {
    fn collect(expr: &Expr, acc: &mut HashSet<String>) {
        match expr {
            Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
                left, right, op, ..
            }) => match op {
                Operator::And => {
                    collect(left, acc);
                    collect(right, acc);
                }
                Operator::Lt | Operator::LtEq => {
                    if is_distance_expr(left) {
                        acc.insert("st_dwithin".to_string());
                    }
                }
                Operator::Gt | Operator::GtEq => {
                    if is_distance_expr(right) {
                        acc.insert("st_dwithin".to_string());
                    }
                }
                _ => (),
            },
            Expr::ScalarFunction(datafusion_expr::expr::ScalarFunction { func, .. }) => {
                let func_name = func.name().to_lowercase();
                if matches!(
                    func_name.as_str(),
                    "st_intersects"
                        | "st_contains"
                        | "st_within"
                        | "st_covers"
                        | "st_covered_by"
                        | "st_coveredby"
                        | "st_touches"
                        | "st_crosses"
                        | "st_overlaps"
                        | "st_equals"
                        | "st_dwithin"
                        | "st_knn"
                ) {
                    acc.insert(func_name);
                }
            }
            _ => (),
        }
    }

    fn is_distance_expr(expr: &Expr) -> bool {
        let Expr::ScalarFunction(datafusion_expr::expr::ScalarFunction { func, .. }) = expr else {
            return false;
        };
        func.name().to_lowercase() == "st_distance"
    }

    let mut acc = HashSet::new();
    collect(expr, &mut acc);
    acc
}

#[derive(PartialEq, Eq, Hash)]
struct SpatialJoinPlanNode {
    left: LogicalPlan,
    right: LogicalPlan,
    join_type: JoinType,
    filter: Expr,
    schema: DFSchemaRef,
    join_constraint: JoinConstraint,
    null_equality: NullEquality,
}

// Manual implementation needed because of `schema` field. Comparison excludes this field.
// See https://github.com/apache/datafusion/blob/52.1.0/datafusion/expr/src/logical_plan/plan.rs#L3886
impl PartialOrd for SpatialJoinPlanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        #[derive(PartialEq, PartialOrd)]
        struct ComparableJoin<'a> {
            pub left: &'a LogicalPlan,
            pub right: &'a LogicalPlan,
            pub filter: &'a Expr,
            pub join_type: &'a JoinType,
            pub join_constraint: &'a JoinConstraint,
            pub null_equality: &'a NullEquality,
        }
        let comparable_self = ComparableJoin {
            left: &self.left,
            right: &self.right,
            filter: &self.filter,
            join_type: &self.join_type,
            join_constraint: &self.join_constraint,
            null_equality: &self.null_equality,
        };
        let comparable_other = ComparableJoin {
            left: &other.left,
            right: &other.right,
            filter: &other.filter,
            join_type: &other.join_type,
            join_constraint: &other.join_constraint,
            null_equality: &self.null_equality,
        };
        comparable_self
            .partial_cmp(&comparable_other)
            // TODO (https://github.com/apache/datafusion/issues/17477) avoid recomparing all fields
            .filter(|cmp| *cmp != Ordering::Equal || self == other)
    }
}

impl fmt::Debug for SpatialJoinPlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        UserDefinedLogicalNodeCore::fmt_for_explain(self, f)
    }
}

impl UserDefinedLogicalNodeCore for SpatialJoinPlanNode {
    fn name(&self) -> &str {
        "SpatialJoin"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.left, &self.right]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.filter.clone()]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SpatialJoin: join_type={:?}, filter={}",
            self.join_type, self.filter
        )
    }

    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if exprs.len() != 1 {
            return plan_err!("SpatialJoinPlanNode expects 1 expr");
        }
        if inputs.len() != 2 {
            return plan_err!("SpatialJoinPlanNode expects 2 inputs");
        }
        Ok(Self {
            left: inputs.swap_remove(0),
            right: inputs.swap_remove(0),
            join_type: self.join_type,
            filter: exprs.swap_remove(0),
            schema: Arc::clone(&self.schema),
            join_constraint: self.join_constraint,
            null_equality: self.null_equality,
        })
    }
}

struct SpatialJoinPlanner;

#[async_trait]
impl ExtensionPlanner for SpatialJoinPlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(spatial_node) = node.as_any().downcast_ref::<SpatialJoinPlanNode>() else {
            return Ok(None);
        };
        if logical_inputs.len() != 2 || physical_inputs.len() != 2 {
            return plan_err!("SpatialJoinPlanNode expects 2 inputs");
        }

        let ext = session_state
            .config_options()
            .extensions
            .get::<SedonaOptions>();

        let left = physical_inputs[0].clone();
        let right = physical_inputs[1].clone();

        let join_filter = build_join_filter(
            planner,
            &spatial_node.filter,
            spatial_node.left.schema(),
            spatial_node.right.schema(),
            &left.schema(),
            &right.schema(),
            session_state,
        )?;

        let join_filter = join_filter.map(|filter| {
            if filter.column_indices().len()
                == left.schema().fields().len() + right.schema().fields().len()
            {
                filter
            } else {
                JoinFilter::new(
                    filter.expression().clone(),
                    build_full_join_column_indices(&left.schema(), &right.schema()),
                    filter.schema().clone(),
                )
            }
        });

        let nlj = NestedLoopJoinExec::try_new(
            left,
            right,
            join_filter.clone(),
            &spatial_node.join_type,
            None,
        )?;

        // Try to emit SpatialJoinExec (optimized). If anything doesn't match, fall back
        // to DataFusion's NestedLoopJoinExec to preserve correctness.
        if let Some(ext) = ext {
            if ext.spatial_join.enable {
                let Some(join_filter) = nlj.filter() else {
                    return Ok(Some(Arc::new(nlj)));
                };
                if let Some((spatial_predicate, remainder)) = transform_join_filter(join_filter) {
                    // Match the optimizer conversion: drop CoalescePartitions on the build side
                    // so SpatialJoinExec can consume partitions directly.
                    let left = if let Some(coalesce_partitions) =
                        nlj.left().as_any().downcast_ref::<CoalescePartitionsExec>()
                    {
                        coalesce_partitions.input().clone()
                    } else {
                        nlj.left().clone()
                    };
                    let right = nlj.right().clone();

                    if is_spatial_predicate_supported(
                        &spatial_predicate,
                        &left.schema(),
                        &right.schema(),
                    )? {
                        let exec = SpatialJoinExec::try_new(
                            left,
                            right,
                            spatial_predicate,
                            remainder,
                            nlj.join_type(),
                            nlj.projection().cloned(),
                            &ext.spatial_join,
                        )?;
                        return Ok(Some(Arc::new(exec)));
                    }
                }
            }
        }

        Ok(Some(Arc::new(nlj)))
    }
}

fn build_join_filter(
    planner: &dyn PhysicalPlanner,
    expr: &Expr,
    left_df_schema: &DFSchemaRef,
    right_df_schema: &DFSchemaRef,
    left_schema: &Schema,
    right_schema: &Schema,
    session_state: &SessionState,
) -> Result<Option<JoinFilter>> {
    // Build JoinFilter the same way DataFusion's physical planner does:
    // - build an *intermediate* schema containing only referenced columns
    // - create the physical expression against that intermediate schema
    // - column_indices map intermediate columns back to left/right inputs

    let cols = expr.column_refs();
    let mut left_field_indices = cols
        .iter()
        .filter_map(|c| left_df_schema.index_of_column(c).ok())
        .collect::<Vec<_>>();
    left_field_indices.sort_unstable();
    let mut right_field_indices = cols
        .iter()
        .filter_map(|c| right_df_schema.index_of_column(c).ok())
        .collect::<Vec<_>>();
    right_field_indices.sort_unstable();

    let mut filter_df_fields =
        Vec::with_capacity(left_field_indices.len() + right_field_indices.len());
    let mut filter_fields =
        Vec::with_capacity(left_field_indices.len() + right_field_indices.len());

    for i in &left_field_indices {
        let (qual, field) = left_df_schema.qualified_field(*i);
        filter_df_fields.push((qual.cloned(), Arc::new(field.clone())));
        filter_fields.push(left_schema.field(*i).clone());
    }
    for i in &right_field_indices {
        let (qual, field) = right_df_schema.qualified_field(*i);
        filter_df_fields.push((qual.cloned(), Arc::new(field.clone())));
        filter_fields.push(right_schema.field(*i).clone());
    }

    let metadata: std::collections::HashMap<String, String> = left_df_schema
        .metadata()
        .clone()
        .into_iter()
        .chain(right_df_schema.metadata().clone())
        .collect();
    let filter_df_schema = DFSchema::new_with_metadata(filter_df_fields, metadata.clone())?;
    let filter_schema = Arc::new(Schema::new_with_metadata(filter_fields, metadata));

    let filter_expr = planner.create_physical_expr(expr, &filter_df_schema, session_state)?;
    let column_indices = JoinFilter::build_column_indices(left_field_indices, right_field_indices);

    Ok(Some(JoinFilter::new(
        filter_expr,
        column_indices,
        filter_schema,
    )))
}

fn build_full_join_column_indices(left_schema: &Schema, right_schema: &Schema) -> Vec<ColumnIndex> {
    let mut column_indices =
        Vec::with_capacity(left_schema.fields().len() + right_schema.fields().len());
    for (index, _field) in left_schema.fields().iter().enumerate() {
        column_indices.push(ColumnIndex {
            index,
            side: datafusion_common::JoinSide::Left,
        });
    }
    for (index, _field) in right_schema.fields().iter().enumerate() {
        column_indices.push(ColumnIndex {
            index,
            side: datafusion_common::JoinSide::Right,
        });
    }
    column_indices
}
