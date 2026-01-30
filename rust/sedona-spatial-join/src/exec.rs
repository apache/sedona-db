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
use std::{fmt::Formatter, sync::Arc};

use arrow_schema::SchemaRef;
use datafusion_common::{project_schema, JoinSide, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::JoinType;
use datafusion_physical_expr::equivalence::{join_equivalence_properties, ProjectionMapping};
use datafusion_physical_expr::projection::update_expr;
use datafusion_physical_plan::{
    common::can_project,
    joins::utils::{build_join_schema, check_join_is_valid, ColumnIndex, JoinFilter},
    joins::utils::{reorder_output_after_swap, swap_join_projection},
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
    projection::{
        join_allows_pushdown, join_table_borders, new_join_children, physical_to_column_exprs,
        try_embed_projection, update_join_filter, EmbeddedProjection, ProjectionExec,
        ProjectionExpr,
    },
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use parking_lot::Mutex;
use sedona_common::{sedona_internal_err, SpatialJoinOptions};

use crate::{
    prepare::{SpatialJoinComponents, SpatialJoinComponentsBuilder},
    spatial_predicate::{DistancePredicate, KNNPredicate, RelationPredicate, SpatialPredicate},
    stream::{SpatialJoinProbeMetrics, SpatialJoinStream},
    utils::{
        join_utils::{
            asymmetric_join_output_partitioning, boundedness_from_children,
            compute_join_emission_type,
        },
        once_fut::OnceAsync,
    },
};

/// Type alias for build and probe execution plans
type BuildProbePlans<'a> = (&'a Arc<dyn ExecutionPlan>, &'a Arc<dyn ExecutionPlan>);

fn update_spatial_predicate_for_child_projections(
    on: &SpatialPredicate,
    projected_left_exprs: &[ProjectionExpr],
    projected_right_exprs: &[ProjectionExpr],
) -> Result<Option<SpatialPredicate>> {
    let updated = match on {
        SpatialPredicate::Relation(pred) => {
            let Some(left) = update_expr(&pred.left, projected_left_exprs, false)? else {
                return Ok(None);
            };
            let Some(right) = update_expr(&pred.right, projected_right_exprs, false)? else {
                return Ok(None);
            };

            SpatialPredicate::Relation(RelationPredicate {
                left,
                right,
                relation_type: pred.relation_type,
            })
        }
        SpatialPredicate::Distance(pred) => {
            let Some(left) = update_expr(&pred.left, projected_left_exprs, false)? else {
                return Ok(None);
            };
            let Some(right) = update_expr(&pred.right, projected_right_exprs, false)? else {
                return Ok(None);
            };

            let distance = match pred.distance_side {
                JoinSide::Left => {
                    let Some(distance) = update_expr(&pred.distance, projected_left_exprs, false)?
                    else {
                        return Ok(None);
                    };
                    distance
                }
                JoinSide::Right => {
                    let Some(distance) = update_expr(&pred.distance, projected_right_exprs, false)?
                    else {
                        return Ok(None);
                    };
                    distance
                }
                JoinSide::None => Arc::clone(&pred.distance),
            };

            SpatialPredicate::Distance(DistancePredicate {
                left,
                right,
                distance,
                distance_side: pred.distance_side,
            })
        }
        SpatialPredicate::KNearestNeighbors(pred) => {
            let (probe_exprs, build_exprs) = match pred.probe_side {
                JoinSide::Left => (projected_left_exprs, projected_right_exprs),
                JoinSide::Right => (projected_right_exprs, projected_left_exprs),
                JoinSide::None => {
                    return sedona_internal_err!(
                        "KNN join requires explicit probe_side designation"
                    )
                }
            };

            let Some(left) = update_expr(&pred.left, probe_exprs, false)? else {
                return Ok(None);
            };
            let Some(right) = update_expr(&pred.right, build_exprs, false)? else {
                return Ok(None);
            };

            SpatialPredicate::KNearestNeighbors(KNNPredicate {
                left,
                right,
                k: pred.k,
                use_spheroid: pred.use_spheroid,
                probe_side: pred.probe_side,
            })
        }
    };

    Ok(Some(updated))
}

/// Determine the correct build/probe execution plan assignment for KNN joins.
///
/// For KNN joins, we need to determine which execution plan should be used as the build side
/// (indexed candidates) and which should be the probe side (queries that search the index).
///
/// The key insight is that the KNNPredicate expressions have already been correctly reprojected
/// by the optimizer, so we can use the join schema structure to determine the mapping:
/// - KNNPredicate.left should always be the probe side (queries)
/// - KNNPredicate.right should always be the build side (candidates)
///
/// We determine which execution plan corresponds to probe/build by analyzing the column indices
/// in the context of the overall join schema structure.
fn determine_knn_build_probe_plans<'a>(
    knn_pred: &KNNPredicate,
    left_plan: &'a Arc<dyn ExecutionPlan>,
    right_plan: &'a Arc<dyn ExecutionPlan>,
) -> Result<BuildProbePlans<'a>> {
    // Use the probe_side information from the optimizer to determine build/probe assignment
    match knn_pred.probe_side {
        JoinSide::Left => Ok((right_plan, left_plan)),
        JoinSide::Right => Ok((left_plan, right_plan)),
        JoinSide::None => sedona_internal_err!("KNN join requires explicit probe_side designation"),
    }
}

/// Physical execution plan for performing spatial joins between two tables. It uses a spatial
/// index to speed up the join operation.
///
/// ## Algorithm Overview
///
/// The spatial join execution follows a hash-join-like pattern:
/// 1. **Build Phase**: The left (smaller) table geometries are indexed using a spatial index
/// 2. **Probe Phase**: Each geometry from the right table is used to query the spatial index
/// 3. **Refinement**: Candidate pairs from the index are refined using exact spatial predicates
/// 4. **Output**: Matching pairs are combined according to the specified join type
#[derive(Debug)]
pub struct SpatialJoinExec {
    /// left (build) side which gets hashed
    pub left: Arc<dyn ExecutionPlan>,
    /// right (probe) side which are filtered by the hash table
    pub right: Arc<dyn ExecutionPlan>,
    /// Primary spatial join condition (the expression in the ON clause of the join)
    pub on: SpatialPredicate,
    /// Additional filters which are applied while finding matching rows. It could contain part of
    /// the ON clause, or expressions in the WHERE clause.
    pub filter: Option<JoinFilter>,
    /// How the join is performed (`OUTER`, `INNER`, etc)
    pub join_type: JoinType,
    /// The schema after join. Please be careful when using this schema,
    /// if there is a projection, the schema isn't the same as the output schema.
    join_schema: SchemaRef,
    /// Metrics for tracking execution statistics (public for wrapper implementations)
    pub metrics: ExecutionPlanMetricsSet,
    /// The projection indices of the columns in the output schema of join
    projection: Option<Vec<usize>>,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: PlanProperties,
    /// Once future for creating the partitioned index provider shared by all probe partitions.
    /// This future runs only once before probing starts, and can be disposed by the last finished
    /// stream so the provider does not outlive the execution plan unnecessarily.
    once_async_spatial_join_components: Arc<Mutex<Option<OnceAsync<SpatialJoinComponents>>>>,
    /// A random seed for making random procedures in spatial join deterministic
    seed: u64,
}

impl SpatialJoinExec {
    /// Try to create a new [`SpatialJoinExec`]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: SpatialPredicate,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
        options: &SpatialJoinOptions,
    ) -> Result<Self> {
        let seed = options
            .debug
            .random_seed
            .unwrap_or(fastrand::u64(0..0xFFFF));
        Self::try_new_internal(left, right, on, filter, join_type, projection, seed)
    }

    /// Create a new SpatialJoinExec with additional options
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_internal(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: SpatialPredicate,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
        seed: u64,
    ) -> Result<Self> {
        let left_schema = left.schema();
        let right_schema = right.schema();
        check_join_is_valid(&left_schema, &right_schema, &[])?;
        let (join_schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, join_type);
        let join_schema = Arc::new(join_schema);
        let cache = Self::compute_properties(
            &left,
            &right,
            &on,
            Arc::clone(&join_schema),
            *join_type,
            projection.as_ref(),
        )?;

        Ok(SpatialJoinExec {
            left,
            right,
            on,
            filter,
            join_type: *join_type,
            join_schema,
            column_indices,
            projection,
            metrics: Default::default(),
            cache,
            once_async_spatial_join_components: Arc::new(Mutex::new(None)),
            seed,
        })
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    /// Does this join has a projection on the joined columns
    pub fn contains_projection(&self) -> bool {
        self.projection.is_some()
    }

    /// Returns a new `ExecutionPlan` that runs NestedLoopsJoins with the left
    /// and right inputs swapped.
    ///
    /// # Notes:
    ///
    /// This function should be called BEFORE inserting any repartitioning
    /// operators on the join's children. Check [`super::HashJoinExec::swap_inputs`]
    /// for more details.
    pub fn swap_inputs(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let left_schema = self.left.schema();
        let right_schema = self.right.schema();

        let swapped_on = self.on.swap_for_swapped_children();

        let swapped_projection = swap_join_projection(
            left_schema.fields().len(),
            right_schema.fields().len(),
            self.projection.as_ref(),
            &self.join_type,
        );

        let swapped_join = SpatialJoinExec::try_new_internal(
            Arc::clone(&self.right),
            Arc::clone(&self.left),
            swapped_on,
            self.filter.as_ref().map(|f| f.swap()),
            &self.join_type.swap(),
            swapped_projection,
            self.seed,
        )?;

        let swapped_join: Arc<dyn ExecutionPlan> = Arc::new(swapped_join);

        match self.join_type {
            JoinType::LeftAnti
            | JoinType::LeftSemi
            | JoinType::RightAnti
            | JoinType::RightSemi
            | JoinType::LeftMark
            | JoinType::RightMark => Ok(swapped_join),
            _ if self.contains_projection() => Ok(swapped_join),
            _ => {
                reorder_output_after_swap(swapped_join, left_schema.as_ref(), right_schema.as_ref())
            }
        }
    }

    pub fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        can_project(&self.schema(), projection.as_ref())?;

        let projection = match (&self.projection, projection) {
            (Some(existing), Some(new)) => Some(new.into_iter().map(|i| existing[i]).collect()),
            (Some(_), None) => self.projection.clone(),
            (None, other) => other,
        };

        SpatialJoinExec::try_new_internal(
            Arc::clone(&self.left),
            Arc::clone(&self.right),
            self.on.clone(),
            self.filter.clone(),
            &self.join_type,
            projection,
            self.seed,
        )
    }

    /// This function creates the cache object that stores the plan properties such as schema,
    /// equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        on: &SpatialPredicate,
        schema: SchemaRef,
        join_type: JoinType,
        projection: Option<&Vec<usize>>,
    ) -> Result<PlanProperties> {
        let mut eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            &join_type,
            Arc::clone(&schema),
            &[false, false],
            None,
            // Pass extracted equality conditions to preserve equivalences
            &[],
        )?;

        let probe_side = if let SpatialPredicate::KNearestNeighbors(knn) = on {
            knn.probe_side
        } else {
            JoinSide::Right
        };
        let mut output_partitioning =
            asymmetric_join_output_partitioning(left, right, &join_type, probe_side)?;

        if let Some(projection) = projection {
            // construct a map from the input expressions to the output expression of the Projection
            let projection_mapping = ProjectionMapping::from_indices(projection, &schema)?;
            let out_schema = project_schema(&schema, Some(projection))?;
            output_partitioning = output_partitioning.project(&projection_mapping, &eq_properties);
            eq_properties = eq_properties.project(&projection_mapping, out_schema);
        }

        let emission_type = compute_join_emission_type(left, right, join_type, probe_side);

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            emission_type,
            boundedness_from_children([left, right]),
        ))
    }
}

impl DisplayAs for SpatialJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let display_on = format!(", on={}", self.on);
                let display_filter = self.filter.as_ref().map_or_else(
                    || "".to_string(),
                    |f| format!(", filter={}", f.expression()),
                );
                let display_projections = if self.contains_projection() {
                    format!(
                        ", projection=[{}]",
                        self.projection
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|index| format!(
                                "{}@{}",
                                self.join_schema.fields().get(*index).unwrap().name(),
                                index
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    "".to_string()
                };
                write!(
                    f,
                    "SpatialJoinExec: join_type={:?}{}{}{}",
                    self.join_type, display_on, display_filter, display_projections
                )
            }
            DisplayFormatType::TreeRender => {
                if *self.join_type() != JoinType::Inner {
                    writeln!(f, "join_type={:?}", self.join_type)
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl ExecutionPlan for SpatialJoinExec {
    fn name(&self) -> &str {
        "SpatialJoinExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false, false]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    /// Tries to push `projection` down through `SpatialJoinExec`. If possible, performs the
    /// pushdown and returns a new [`SpatialJoinExec`] as the top plan which has projections
    /// as its children. Otherwise, returns `None`.
    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if self.contains_projection() {
            return Ok(None);
        }

        // For join types where the output is not a concatenation of left + right columns
        // (e.g. semi/anti/mark), the generic join projection pushdown logic does not apply.
        // Fall back to embedding.
        match self.join_type {
            JoinType::LeftAnti
            | JoinType::LeftSemi
            | JoinType::RightAnti
            | JoinType::RightSemi
            | JoinType::LeftMark
            | JoinType::RightMark => return try_embed_projection(projection, self),
            _ => {}
        }

        // Try full pushdown first, and fall back to embedding.
        let Some(projection_as_columns) = physical_to_column_exprs(projection.expr()) else {
            return try_embed_projection(projection, self);
        };

        let (far_right_left_col_ind, far_left_right_col_ind) =
            join_table_borders(self.left.schema().fields().len(), &projection_as_columns);

        if !join_allows_pushdown(
            &projection_as_columns,
            &self.join_schema,
            far_right_left_col_ind,
            far_left_right_col_ind,
        ) {
            return try_embed_projection(projection, self);
        }

        let (projected_left_child, projected_right_child) = new_join_children(
            &projection_as_columns,
            far_right_left_col_ind,
            far_left_right_col_ind,
            &self.left,
            &self.right,
        )?;

        let new_filter = if let Some(filter) = self.filter.as_ref() {
            let left_cols = &projection_as_columns[0..=far_right_left_col_ind as usize];
            let right_cols = &projection_as_columns[far_left_right_col_ind as usize..];
            match update_join_filter(
                left_cols,
                right_cols,
                filter,
                self.left.schema().fields().len(),
            ) {
                Some(updated) => Some(updated),
                None => return try_embed_projection(projection, self),
            }
        } else {
            None
        };

        let projected_left_exprs = projected_left_child.expr();
        let projected_right_exprs = projected_right_child.expr();
        let Some(new_on) = update_spatial_predicate_for_child_projections(
            &self.on,
            projected_left_exprs,
            projected_right_exprs,
        )?
        else {
            return try_embed_projection(projection, self);
        };

        let new_exec = SpatialJoinExec::try_new_internal(
            Arc::new(projected_left_child),
            Arc::new(projected_right_child),
            new_on,
            new_filter,
            &self.join_type,
            None,
            self.seed,
        )?;
        Ok(Some(Arc::new(new_exec)))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let new_exec = SpatialJoinExec::try_new_internal(
            Arc::clone(&children[0]),
            Arc::clone(&children[1]),
            self.on.clone(),
            self.filter.clone(),
            &self.join_type,
            self.projection.clone(),
            self.seed,
        )?;
        Ok(Arc::new(new_exec))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let session_config = context.session_config();

        // Determine build/probe plans based on predicate type.
        // For KNN joins, the probe/build assignment is dynamic based on the KNN predicate's
        // probe_side. For regular spatial joins, left is always build and right is always probe.
        let (build_plan, probe_plan, probe_side) = match &self.on {
            SpatialPredicate::KNearestNeighbors(knn_pred) => {
                let (build_plan, probe_plan) = determine_knn_build_probe_plans(
                    knn_pred,
                    &self.left,
                    &self.right,
                    &self.join_schema,
                )?;
                (build_plan, probe_plan, knn_pred.probe_side)
            }
            _ => (&self.left, &self.right, JoinSide::Right),
        };

        // Determine which input index corresponds to the probe side for ordering checks
        let probe_input_index = if probe_side == JoinSide::Left { 0 } else { 1 };

        // A OnceFut for preparing the spatial join components once.
        let once_fut_spatial_join_components = {
            let mut once_async = self.once_async_spatial_join_components.lock();
            once_async
                .get_or_insert(OnceAsync::default())
                .try_once(|| {
                    let num_partitions = build_plan.output_partitioning().partition_count();
                    let mut build_streams = Vec::with_capacity(num_partitions);
                    for k in 0..num_partitions {
                        let stream = build_plan.execute(k, Arc::clone(&context))?;
                        build_streams.push(stream);
                    }

                    let probe_thread_count = probe_plan.output_partitioning().partition_count();
                    let spatial_join_components_builder = SpatialJoinComponentsBuilder::new(
                        Arc::clone(&context),
                        build_plan.schema(),
                        self.on.clone(),
                        self.join_type,
                        probe_thread_count,
                        self.metrics.clone(),
                        self.seed,
                    );
                    Ok(spatial_join_components_builder.build(build_streams))
                })?
        };

        let column_indices_after_projection = match &self.projection {
            Some(projection) => projection
                .iter()
                .map(|i| self.column_indices[*i].clone())
                .collect(),
            None => self.column_indices.clone(),
        };

        let probe_stream = probe_plan.execute(partition, Arc::clone(&context))?;

        let probe_side_ordered = self.maintains_input_order()[probe_input_index]
            && probe_plan.output_ordering().is_some();

        Ok(Box::pin(SpatialJoinStream::new(
            partition,
            self.schema(),
            &self.on,
            self.filter.clone(),
            self.join_type,
            probe_stream,
            column_indices_after_projection,
            probe_side_ordered,
            session_config,
            context.runtime_env(),
            &self.metrics,
            once_fut_spatial_join_components,
            Arc::clone(&self.once_async_spatial_join_components),
        )))
    }
}

impl EmbeddedProjection for SpatialJoinExec {
    fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        self.with_projection(projection)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        catalog::{MemTable, TableProvider},
        execution::SessionStateBuilder,
        prelude::{SessionConfig, SessionContext},
    };
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion_expr::ColumnarValue;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_plan::empty::EmptyExec;
    use datafusion_physical_plan::joins::NestedLoopJoinExec;
    use geo::{Distance, Euclidean};
    use geo_types::{Coord, Rect};
    use rstest::rstest;
    use sedona_geo::to_geo::GeoTypesExecutor;
    use sedona_geometry::types::GeometryTypeId;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY};
    use sedona_testing::datagen::RandomPartitionedDataBuilder;
    use tokio::sync::OnceCell;

    use sedona_common::{
        option::{add_sedona_option_extension, ExecutionMode, SpatialJoinOptions},
        SpatialLibrary,
    };

    use super::*;

    fn make_schema(fields: &[(&str, DataType)]) -> SchemaRef {
        Arc::new(Schema::new(
            fields
                .iter()
                .map(|(name, dt)| Field::new(*name, dt.clone(), true))
                .collect::<Vec<_>>(),
        ))
    }

    fn proj_expr(
        schema: &SchemaRef,
        index: usize,
    ) -> (Arc<dyn datafusion_physical_expr::PhysicalExpr>, String) {
        let name = schema.field(index).name().to_string();
        (Arc::new(Column::new(&name, index)), name)
    }

    #[test]
    fn test_try_swapping_with_projection_pushes_down_and_rewrites_relation_predicate() -> Result<()>
    {
        use crate::spatial_predicate::SpatialRelationType;

        // left: [l0, l1, l2], right: [r0, r1]
        let left_schema = make_schema(&[
            ("l0", DataType::Int32),
            ("l1", DataType::Int32),
            ("l2", DataType::Int32),
        ]);
        let right_schema = make_schema(&[("r0", DataType::Int32), ("r1", DataType::Int32)]);
        let left_len = left_schema.fields().len();

        let left: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&left_schema)));
        let right: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&right_schema)));

        // on: ST_Intersects(l2, r1) (types don't matter for rewrite-only test)
        let on = SpatialPredicate::Relation(RelationPredicate {
            left: Arc::new(Column::new("l2", 2)),
            right: Arc::new(Column::new("r1", 1)),
            relation_type: SpatialRelationType::Intersects,
        });

        let exec = Arc::new(SpatialJoinExec::try_new_internal(
            left,
            right,
            on,
            None,
            &JoinType::Inner,
            None,
            0,
        )?);

        // Project only columns used by the predicate: l2 then r1.
        let join_schema = exec.schema();
        let exprs = vec![
            proj_expr(&join_schema, 2),
            proj_expr(&join_schema, left_len + 1),
        ];
        let proj = ProjectionExec::try_new(exprs, Arc::clone(&exec) as Arc<dyn ExecutionPlan>)?;

        let Some(new_plan) = exec.try_swapping_with_projection(&proj)? else {
            return sedona_internal_err!("expected try_swapping_with_projection to succeed");
        };

        let new_exec = new_plan
            .as_any()
            .downcast_ref::<SpatialJoinExec>()
            .expect("expected SpatialJoinExec");

        // Projection is pushed down into children; join has no embedded projection.
        assert!(!new_exec.contains_projection());
        assert!(new_exec
            .children()
            .iter()
            .all(|c| c.as_any().downcast_ref::<ProjectionExec>().is_some()));

        // Predicate columns should be remapped to match the projected children (both become 0).
        let SpatialPredicate::Relation(new_on) = &new_exec.on else {
            return sedona_internal_err!("expected Relation predicate");
        };
        let new_left = new_on
            .left
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column expr");
        let new_right = new_on
            .right
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column expr");
        assert_eq!(new_left.index(), 0);
        assert_eq!(new_right.index(), 0);

        Ok(())
    }

    #[test]
    fn test_try_swapping_with_projection_pushes_down_and_rewrites_knn_predicate_by_probe_side(
    ) -> Result<()> {
        // left: [l0, lgeom], right: [r0, rgeom]
        let left_schema = make_schema(&[("l0", DataType::Int32), ("lgeom", DataType::Binary)]);
        let right_schema = make_schema(&[("r0", DataType::Int32), ("rgeom", DataType::Binary)]);
        let left_len = left_schema.fields().len();

        let left: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&left_schema)));
        let right: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&right_schema)));

        // KNN where queries are on the RIGHT plan (probe_side=Right): ST_KNN(rgeom, lgeom, ...)
        let on = SpatialPredicate::KNearestNeighbors(KNNPredicate {
            left: Arc::new(Column::new("rgeom", 1)),
            right: Arc::new(Column::new("lgeom", 1)),
            k: 3,
            use_spheroid: false,
            probe_side: JoinSide::Right,
        });

        let exec = Arc::new(SpatialJoinExec::try_new_internal(
            left,
            right,
            on,
            None,
            &JoinType::Inner,
            None,
            0,
        )?);

        // Project only geometry columns (left then right) so pushdown is allowed.
        let join_schema = exec.schema();
        let exprs = vec![
            proj_expr(&join_schema, 1),
            proj_expr(&join_schema, left_len + 1),
        ];
        let proj = ProjectionExec::try_new(exprs, Arc::clone(&exec) as Arc<dyn ExecutionPlan>)?;

        let Some(new_plan) = exec.try_swapping_with_projection(&proj)? else {
            return sedona_internal_err!("expected try_swapping_with_projection to succeed");
        };
        let new_exec = new_plan
            .as_any()
            .downcast_ref::<SpatialJoinExec>()
            .expect("expected SpatialJoinExec");

        let SpatialPredicate::KNearestNeighbors(new_on) = &new_exec.on else {
            return sedona_internal_err!("expected KNN predicate");
        };

        // Both sides should be remapped to 0 in their respective projected children.
        let new_probe = new_on
            .left
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column expr");
        let new_build = new_on
            .right
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column expr");
        assert_eq!(new_probe.index(), 0);
        assert_eq!(new_build.index(), 0);
        assert_eq!(new_on.probe_side, JoinSide::Right);

        Ok(())
    }

    #[test]
    fn test_swap_inputs_flips_knn_probe_side_without_swapping_exprs() -> Result<()> {
        let left_schema = make_schema(&[("l0", DataType::Int32), ("lgeom", DataType::Binary)]);
        let right_schema = make_schema(&[("r0", DataType::Int32), ("rgeom", DataType::Binary)]);

        let left: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&left_schema)));
        let right: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&right_schema)));

        let on = SpatialPredicate::KNearestNeighbors(KNNPredicate {
            left: Arc::new(Column::new("rgeom", 1)),
            right: Arc::new(Column::new("lgeom", 1)),
            k: 3,
            use_spheroid: false,
            probe_side: JoinSide::Right,
        });
        let exec =
            SpatialJoinExec::try_new_internal(left, right, on, None, &JoinType::Inner, None, 0)?;

        let swapped = exec.swap_inputs()?;
        let spatial_execs = collect_spatial_join_exec(&swapped)?;
        assert_eq!(spatial_execs.len(), 1);

        let swapped_exec = spatial_execs[0];
        let SpatialPredicate::KNearestNeighbors(knn) = &swapped_exec.on else {
            return sedona_internal_err!("expected KNN predicate");
        };

        // Children swapped, so probe_side flips.
        assert_eq!(knn.probe_side, JoinSide::Left);

        // Expressions are not swapped (remain pointing at original table schemas).
        let probe_expr = knn
            .left
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column expr");
        let build_expr = knn
            .right
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column expr");
        assert_eq!(probe_expr.name(), "rgeom");
        assert_eq!(probe_expr.index(), 1);
        assert_eq!(build_expr.name(), "lgeom");
        assert_eq!(build_expr.index(), 1);

        Ok(())
    }
}
