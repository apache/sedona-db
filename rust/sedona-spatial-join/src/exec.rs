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
use datafusion_physical_plan::{
    common::can_project,
    joins::utils::{build_join_schema, check_join_is_valid, ColumnIndex, JoinFilter},
    joins::utils::{reorder_output_after_swap, swap_join_projection},
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
    projection::{
        join_allows_pushdown, join_table_borders, new_join_children, physical_to_column_exprs,
        try_embed_projection, update_join_filter, EmbeddedProjection, ProjectionExec,
    },
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use parking_lot::Mutex;
use sedona_common::{sedona_internal_err, SpatialJoinOptions};

use crate::{
    prepare::{SpatialJoinComponents, SpatialJoinComponentsBuilder},
    spatial_predicate::{KNNPredicate, SpatialPredicate},
    stream::{SpatialJoinProbeMetrics, SpatialJoinStream},
    utils::{
        join_utils::{
            asymmetric_join_output_partitioning, boundedness_from_children,
            compute_join_emission_type,
        },
        once_fut::OnceAsync,
    },
    SedonaOptions,
};

/// Type alias for build and probe execution plans
type BuildProbePlans<'a> = (&'a Arc<dyn ExecutionPlan>, &'a Arc<dyn ExecutionPlan>);

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
        // check if the projection is valid
        can_project(&self.schema(), projection.as_ref())?;
        let projection = match projection {
            Some(projection) => match &self.projection {
                Some(p) => Some(projection.iter().map(|i| p[*i]).collect()),
                None => Some(projection),
            },
            None => None,
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
        // TODO: currently if there is projection in SpatialJoinExec, we can't push down projection to
        // left or right input. Maybe we can pushdown the mixed projection later.
        // This restriction is inherited from NestedLoopJoinExec and HashJoinExec in DataFusion.
        if self.contains_projection() {
            return Ok(None);
        }

        // TODO: mark joins make `new_join_children` below fail since the projection contains
        // a mark column, which is not a direct column from left or right child. `new_join_children`
        // is not designed to handle this case. We need to improve it later.
        match self.join_type {
            JoinType::LeftMark | JoinType::RightMark => {
                return try_embed_projection(projection, self)
            }
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
        let Some(new_on) = self
            .on
            .update_for_child_projections(projected_left_exprs, projected_right_exprs)?
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
        match &self.on {
            SpatialPredicate::KNearestNeighbors(_) => self.execute_knn(partition, context),
            _ => self.execute(partition, context),
        }
    }
}

impl EmbeddedProjection for SpatialJoinExec {
    fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        self.with_projection(projection)
    }
}

impl SpatialJoinExec {
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Regular spatial join logic - standard left=build, right=probe semantics
        let session_config = context.session_config();
        let target_output_batch_size = session_config.options().execution.batch_size;
        let sedona_options = session_config
            .options()
            .extensions
            .get::<SedonaOptions>()
            .cloned()
            .unwrap_or_default();

        // Regular join semantics: left is build, right is probe
        let (build_plan, probe_plan) = (&self.left, &self.right);

        // Build the spatial join components using shared OnceAsync
        let once_fut_spatial_join_components = {
            let mut once_async = self.once_async_spatial_join_components.lock();
            once_async
                .get_or_insert(OnceAsync::default())
                .try_once(|| {
                    let build_side = build_plan;

                    let num_partitions = build_side.output_partitioning().partition_count();
                    let mut build_streams = Vec::with_capacity(num_partitions);
                    for k in 0..num_partitions {
                        let stream = build_side.execute(k, Arc::clone(&context))?;
                        build_streams.push(stream);
                    }

                    let probe_thread_count = probe_plan.output_partitioning().partition_count();
                    let spatial_join_components_builder = SpatialJoinComponentsBuilder::new(
                        Arc::clone(&context),
                        build_side.schema(),
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

        let join_metrics = SpatialJoinProbeMetrics::new(partition, &self.metrics);
        let probe_stream = probe_plan.execute(partition, Arc::clone(&context))?;

        // For regular joins: probe is right side (index 1)
        let probe_side_ordered =
            self.maintains_input_order()[1] && self.right.output_ordering().is_some();

        Ok(Box::pin(SpatialJoinStream::new(
            partition,
            self.schema(),
            &self.on,
            self.filter.clone(),
            self.join_type,
            probe_stream,
            column_indices_after_projection,
            probe_side_ordered,
            join_metrics,
            sedona_options.spatial_join,
            target_output_batch_size,
            once_fut_spatial_join_components,
            Arc::clone(&self.once_async_spatial_join_components),
        )))
    }

    /// Execute KNN (K-Nearest Neighbors) spatial join with specialized logic for asymmetric KNN semantics
    fn execute_knn(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let session_config = context.session_config();
        let target_output_batch_size = session_config.options().execution.batch_size;
        let sedona_options = session_config
            .options()
            .extensions
            .get::<SedonaOptions>()
            .cloned()
            .unwrap_or_default();

        // Extract KNN predicate for type safety
        let knn_pred = match &self.on {
            SpatialPredicate::KNearestNeighbors(knn_pred) => knn_pred,
            _ => unreachable!("execute_knn called with non-KNN predicate"),
        };

        // Determine which execution plan should be build vs probe using join schema analysis
        let (build_plan, probe_plan) =
            determine_knn_build_probe_plans(knn_pred, &self.left, &self.right)?;

        // Determine if probe plan is the left execution plan (for column index swapping logic)
        let actual_probe_plan_is_left = std::ptr::eq(probe_plan.as_ref(), self.left.as_ref());

        // Build the spatial index
        let once_fut_spatial_join_components = {
            let mut once_async = self.once_async_spatial_join_components.lock();
            once_async
                .get_or_insert(OnceAsync::default())
                .try_once(|| {
                    let build_side = build_plan;

                    let num_partitions = build_side.output_partitioning().partition_count();
                    let mut build_streams = Vec::with_capacity(num_partitions);
                    for k in 0..num_partitions {
                        let stream = build_side.execute(k, Arc::clone(&context))?;
                        build_streams.push(stream);
                    }

                    let probe_thread_count = probe_plan.output_partitioning().partition_count();
                    let spatial_join_components_builder = SpatialJoinComponentsBuilder::new(
                        Arc::clone(&context),
                        build_side.schema(),
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

        let join_metrics = SpatialJoinProbeMetrics::new(partition, &self.metrics);
        let probe_stream = probe_plan.execute(partition, Arc::clone(&context))?;

        // Determine if probe side ordering is maintained for KNN
        let probe_side_ordered = if actual_probe_plan_is_left {
            // Actual probe is left plan
            self.maintains_input_order()[0] && self.left.output_ordering().is_some()
        } else {
            // Actual probe is right plan
            self.maintains_input_order()[1] && self.right.output_ordering().is_some()
        };

        Ok(Box::pin(SpatialJoinStream::new(
            partition,
            self.schema(),
            &self.on,
            self.filter.clone(),
            self.join_type,
            probe_stream,
            column_indices_after_projection,
            probe_side_ordered,
            join_metrics,
            sedona_options.spatial_join,
            target_output_batch_size,
            once_fut_spatial_join_components,
            Arc::clone(&self.once_async_spatial_join_components),
        )))
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

    type TestPartitions = (SchemaRef, Vec<Vec<RecordBatch>>);

    /// Creates standard test data with left (Polygon) and right (Point) partitions
    fn create_default_test_data() -> Result<(TestPartitions, TestPartitions)> {
        create_test_data_with_size_range((1.0, 10.0), WKB_GEOMETRY)
    }

    /// Creates test data with custom size range
    fn create_test_data_with_size_range(
        size_range: (f64, f64),
        sedona_type: SedonaType,
    ) -> Result<(TestPartitions, TestPartitions)> {
        let bounds = Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 100.0 });

        let left_data = RandomPartitionedDataBuilder::new()
            .seed(11584)
            .num_partitions(2)
            .batches_per_partition(2)
            .rows_per_batch(30)
            .geometry_type(GeometryTypeId::Polygon)
            .sedona_type(sedona_type.clone())
            .bounds(bounds)
            .size_range(size_range)
            .null_rate(0.1)
            .build()?;

        let right_data = RandomPartitionedDataBuilder::new()
            .seed(54843)
            .num_partitions(4)
            .batches_per_partition(4)
            .rows_per_batch(30)
            .geometry_type(GeometryTypeId::Point)
            .sedona_type(sedona_type)
            .bounds(bounds)
            .size_range(size_range)
            .null_rate(0.1)
            .build()?;

        Ok((left_data, right_data))
    }

    /// Creates test data with empty partitions inserted at beginning and end
    fn create_test_data_with_empty_partitions() -> Result<(TestPartitions, TestPartitions)> {
        let (mut left_data, mut right_data) = create_default_test_data()?;

        // Add empty partitions
        left_data.1.insert(0, vec![]);
        left_data.1.push(vec![]);
        right_data.1.insert(0, vec![]);
        right_data.1.push(vec![]);

        Ok((left_data, right_data))
    }

    /// Creates test data for KNN join (Point-Point)
    fn create_knn_test_data(
        size_range: (f64, f64),
        sedona_type: SedonaType,
    ) -> Result<(TestPartitions, TestPartitions)> {
        let bounds = Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 100.0 });

        let left_data = RandomPartitionedDataBuilder::new()
            .seed(1)
            .num_partitions(2)
            .batches_per_partition(2)
            .rows_per_batch(30)
            .geometry_type(GeometryTypeId::Point)
            .sedona_type(sedona_type.clone())
            .bounds(bounds)
            .size_range(size_range)
            .null_rate(0.1)
            .build()?;

        let right_data = RandomPartitionedDataBuilder::new()
            .seed(2)
            .num_partitions(4)
            .batches_per_partition(4)
            .rows_per_batch(30)
            .geometry_type(GeometryTypeId::Point)
            .sedona_type(sedona_type)
            .bounds(bounds)
            .size_range(size_range)
            .null_rate(0.1)
            .build()?;

        Ok((left_data, right_data))
    }

    fn setup_context(
        options: Option<SpatialJoinOptions>,
        batch_size: usize,
    ) -> Result<SessionContext> {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut session_config = SessionConfig::from_env()?
            .with_information_schema(true)
            .with_batch_size(batch_size);
        session_config = add_sedona_option_extension(session_config);
        let mut state_builder = SessionStateBuilder::new();
        if let Some(options) = options {
            // Logical rewrite (Filter(CrossJoin)->Join(filter)) + extension-based planning
            // (Join(filter)->SpatialJoinExec). Intentionally avoid physical plan rewrites.
            state_builder = crate::register_planner(state_builder);
            let opts = session_config
                .options_mut()
                .extensions
                .get_mut::<SedonaOptions>()
                .unwrap();
            opts.spatial_join = options;
        }
        let state = state_builder.with_config(session_config).build();
        let ctx = SessionContext::new_with_state(state);

        let mut function_set = sedona_functions::register::default_function_set();
        let scalar_kernels = sedona_geos::register::scalar_kernels();

        function_set.scalar_udfs().for_each(|udf| {
            ctx.register_udf(udf.clone().into());
        });

        for (name, kernel) in scalar_kernels.into_iter() {
            let udf = function_set.add_scalar_udf_impl(name, kernel)?;
            ctx.register_udf(udf.clone().into());
        }

        Ok(ctx)
    }

    #[tokio::test]
    async fn test_empty_data() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("dist", DataType::Float64, false),
            WKB_GEOMETRY.to_storage_field("geometry", true).unwrap(),
        ]));

        let test_data_vec = vec![vec![vec![]], vec![vec![], vec![]]];

        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareNone,
            ..Default::default()
        };
        let ctx = setup_context(Some(options.clone()), 10)?;
        for test_data in test_data_vec {
            let left_partitions = test_data.clone();
            let right_partitions = test_data;

            let mem_table_left: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                left_partitions.clone(),
            )?);
            let mem_table_right: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
                Arc::clone(&schema),
                right_partitions.clone(),
            )?);

            ctx.deregister_table("L")?;
            ctx.deregister_table("R")?;
            ctx.register_table("L", Arc::clone(&mem_table_left))?;
            ctx.register_table("R", Arc::clone(&mem_table_right))?;

            let sql = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id";
            let df = ctx.sql(sql).await?;
            let result_batches = df.collect().await?;
            for result_batch in result_batches {
                assert_eq!(result_batch.num_rows(), 0);
            }
        }

        Ok(())
    }

    // Shared test data and expected results - computed only once across all parameterized test cases
    // Using tokio::sync::OnceCell for async lazy initialization to avoid recomputing expensive
    // test data generation and nested loop join results for each test parameter combination
    static TEST_DATA: OnceCell<(TestPartitions, TestPartitions)> = OnceCell::const_new();
    static RANGE_JOIN_EXPECTED_RESULTS: OnceCell<Vec<RecordBatch>> = OnceCell::const_new();
    static DIST_JOIN_EXPECTED_RESULTS: OnceCell<Vec<RecordBatch>> = OnceCell::const_new();

    const RANGE_JOIN_SQL1: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id";
    const RANGE_JOIN_SQL2: &str =
        "SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY L.id, R.id";
    const RANGE_JOIN_SQLS: &[&str] = &[RANGE_JOIN_SQL1, RANGE_JOIN_SQL2];

    const DIST_JOIN_SQL1: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Distance(L.geometry, R.geometry) < 1.0 ORDER BY l_id, r_id";
    const DIST_JOIN_SQL2: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Distance(L.geometry, R.geometry) < L.dist / 10.0 ORDER BY l_id, r_id";
    const DIST_JOIN_SQL3: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Distance(L.geometry, R.geometry) < R.dist / 10.0 ORDER BY l_id, r_id";
    const DIST_JOIN_SQL4: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_DWithin(L.geometry, R.geometry, 1.0) ORDER BY l_id, r_id";
    const DIST_JOIN_SQLS: &[&str] = &[
        DIST_JOIN_SQL1,
        DIST_JOIN_SQL2,
        DIST_JOIN_SQL3,
        DIST_JOIN_SQL4,
    ];

    /// Get test data, computing it only once
    async fn get_default_test_data() -> &'static (TestPartitions, TestPartitions) {
        TEST_DATA
            .get_or_init(|| async {
                create_default_test_data().expect("Failed to create test data")
            })
            .await
    }

    /// Get expected results, computing them only once
    async fn get_expected_range_join_results() -> &'static Vec<RecordBatch> {
        get_or_init_expected_join_results(&RANGE_JOIN_EXPECTED_RESULTS, RANGE_JOIN_SQLS).await
    }

    async fn get_expected_distance_join_results() -> &'static Vec<RecordBatch> {
        get_or_init_expected_join_results(&DIST_JOIN_EXPECTED_RESULTS, DIST_JOIN_SQLS).await
    }

    async fn get_or_init_expected_join_results<'a>(
        lazy_init_results: &'a OnceCell<Vec<RecordBatch>>,
        sql_queries: &[&str],
    ) -> &'a Vec<RecordBatch> {
        lazy_init_results
            .get_or_init(|| async {
                let test_data = get_default_test_data().await;
                let ((left_schema, left_partitions), (right_schema, right_partitions)) = test_data;

                let batch_size = 10;

                // Run nested loop join to get expected results
                let mut expected_results = Vec::with_capacity(sql_queries.len());

                for (i, sql) in sql_queries.iter().enumerate() {
                    let result = run_spatial_join_query(
                        left_schema,
                        right_schema,
                        left_partitions.clone(),
                        right_partitions.clone(),
                        None,
                        batch_size,
                        sql,
                    )
                    .await
                    .unwrap_or_else(|_| panic!("Failed to generate expected result {}", i + 1));
                    expected_results.push(result);
                }

                expected_results
            })
            .await
    }

    #[rstest]
    #[tokio::test]
    async fn test_range_join_with_conf(
        #[values(10, 30, 1000)] max_batch_size: usize,
        #[values(
            ExecutionMode::PrepareNone,
            ExecutionMode::PrepareBuild,
            ExecutionMode::PrepareProbe,
            ExecutionMode::Speculative(20)
        )]
        execution_mode: ExecutionMode,
        #[values(SpatialLibrary::Geo, SpatialLibrary::Geos, SpatialLibrary::Tg)]
        spatial_library: SpatialLibrary,
    ) -> Result<()> {
        let test_data = get_default_test_data().await;
        let expected_results = get_expected_range_join_results().await;
        let ((left_schema, left_partitions), (right_schema, right_partitions)) = test_data;

        let options = SpatialJoinOptions {
            spatial_library,
            execution_mode,
            ..Default::default()
        };
        for (idx, sql) in RANGE_JOIN_SQLS.iter().enumerate() {
            let actual_result = run_spatial_join_query(
                left_schema,
                right_schema,
                left_partitions.clone(),
                right_partitions.clone(),
                Some(options.clone()),
                max_batch_size,
                sql,
            )
            .await?;
            assert_eq!(&actual_result, &expected_results[idx]);
        }

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_distance_join_with_conf(
        #[values(30, 1000)] max_batch_size: usize,
        #[values(SpatialLibrary::Geo, SpatialLibrary::Geos, SpatialLibrary::Tg)]
        spatial_library: SpatialLibrary,
    ) -> Result<()> {
        let test_data = get_default_test_data().await;
        let expected_results = get_expected_distance_join_results().await;
        let ((left_schema, left_partitions), (right_schema, right_partitions)) = test_data;

        let options = SpatialJoinOptions {
            spatial_library,
            ..Default::default()
        };
        for (idx, sql) in DIST_JOIN_SQLS.iter().enumerate() {
            let actual_result = run_spatial_join_query(
                left_schema,
                right_schema,
                left_partitions.clone(),
                right_partitions.clone(),
                Some(options.clone()),
                max_batch_size,
                sql,
            )
            .await?;
            assert_eq!(&actual_result, &expected_results[idx]);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_spatial_join_with_filter() -> Result<()> {
        let ((left_schema, left_partitions), (right_schema, right_partitions)) =
            create_test_data_with_size_range((0.1, 10.0), WKB_GEOMETRY)?;

        for max_batch_size in [10, 30, 100] {
            let options = SpatialJoinOptions {
                execution_mode: ExecutionMode::PrepareNone,
                ..Default::default()
            };
            test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
                "SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) AND L.dist < R.dist ORDER BY L.id, R.id").await?;
            test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
                "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) AND L.dist < R.dist ORDER BY l_id, r_id").await?;
            test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
                "SELECT L.id l_id, R.id r_id, L.dist l_dist, R.dist r_dist FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) AND L.dist < R.dist ORDER BY l_id, r_id").await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_range_join_with_empty_partitions() -> Result<()> {
        let ((left_schema, left_partitions), (right_schema, right_partitions)) =
            create_test_data_with_empty_partitions()?;

        for max_batch_size in [10, 30, 1000] {
            let options = SpatialJoinOptions {
                execution_mode: ExecutionMode::PrepareNone,
                ..Default::default()
            };
            test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
                "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id").await?;
            test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
                "SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY L.id, R.id").await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_inner_join() -> Result<()> {
        test_with_join_types(JoinType::Inner).await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_left_joins(
        #[values(JoinType::Left, JoinType::LeftSemi, JoinType::LeftAnti)] join_type: JoinType,
    ) -> Result<()> {
        test_with_join_types(join_type).await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_right_joins(
        #[values(JoinType::Right, JoinType::RightSemi, JoinType::RightAnti)] join_type: JoinType,
    ) -> Result<()> {
        test_with_join_types(join_type).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_full_outer_join() -> Result<()> {
        test_with_join_types(JoinType::Full).await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_mark_joins(
        #[values(JoinType::LeftMark, JoinType::RightMark)] join_type: JoinType,
    ) -> Result<()> {
        let options = SpatialJoinOptions::default();
        test_mark_join(join_type, options, 10).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_mark_join_via_correlated_exists_sql() -> Result<()> {
        let ((left_schema, left_partitions), (right_schema, right_partitions)) =
            create_test_data_with_size_range((0.1, 10.0), WKB_GEOMETRY)?;

        let mem_table_left: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
            left_schema.clone(),
            left_partitions.clone(),
        )?);
        let mem_table_right: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
            right_schema.clone(),
            right_partitions.clone(),
        )?);

        // DataFusion doesn't have explicit SQL syntax for MARK joins. Predicate subqueries embedded
        // in a more complex boolean expression (e.g. OR) are planned using a MARK join.
        //
        // Using EXISTS here (rather than IN) keeps the join filter as the pulled-up correlated
        // predicate (ST_Intersects), which is what SpatialJoinExec can optimize.
        let sql = "SELECT L.id FROM L WHERE L.id = 1 OR EXISTS (SELECT 1 FROM R WHERE ST_Intersects(L.geometry, R.geometry)) ORDER BY L.id";

        let batch_size = 10;
        let options = SpatialJoinOptions::default();

        // Optimized plan should include a SpatialJoinExec with Mark join type.
        let ctx = setup_context(Some(options), batch_size)?;
        ctx.register_table("L", Arc::clone(&mem_table_left))?;
        ctx.register_table("R", Arc::clone(&mem_table_right))?;
        let df = ctx.sql(sql).await?;
        let plan = df.clone().create_physical_plan().await?;
        let spatial_join_execs = collect_spatial_join_exec(&plan)?;
        assert!(
            spatial_join_execs
                .iter()
                .any(|exec| matches!(*exec.join_type(), JoinType::LeftMark | JoinType::RightMark)),
            "expected correlated IN-subquery to plan using a MARK join when optimized"
        );
        let actual_schema = df.schema().as_arrow().clone();
        let actual_batches = df.collect().await?;
        let actual_batch =
            arrow::compute::concat_batches(&Arc::new(actual_schema), &actual_batches)?;

        // Unoptimized plan should still contain a Mark join, but implemented as NestedLoopJoinExec.
        let ctx_no_opt = setup_context(None, batch_size)?;
        ctx_no_opt.register_table("L", mem_table_left)?;
        ctx_no_opt.register_table("R", mem_table_right)?;
        let df_no_opt = ctx_no_opt.sql(sql).await?;
        let plan_no_opt = df_no_opt.clone().create_physical_plan().await?;
        let nlj_execs = collect_nested_loop_join_exec(&plan_no_opt)?;
        assert!(
            nlj_execs
                .iter()
                .any(|exec| matches!(*exec.join_type(), JoinType::LeftMark | JoinType::RightMark)),
            "expected correlated IN-subquery to plan using a MARK join when not optimized"
        );
        let expected_schema = df_no_opt.schema().as_arrow().clone();
        let expected_batches = df_no_opt.collect().await?;
        let expected_batch =
            arrow::compute::concat_batches(&Arc::new(expected_schema), &expected_batches)?;

        assert!(expected_batch.num_rows() > 0);
        assert_eq!(expected_batch, actual_batch);

        Ok(())
    }

    #[tokio::test]
    async fn test_geography_join_is_not_optimized() -> Result<()> {
        let options = SpatialJoinOptions::default();
        let ctx = setup_context(Some(options), 10)?;

        // Prepare geography tables
        let ((left_schema, left_partitions), (right_schema, right_partitions)) =
            create_test_data_with_size_range((0.1, 10.0), WKB_GEOGRAPHY)?;
        let mem_table_left: Arc<dyn TableProvider> =
            Arc::new(MemTable::try_new(left_schema, left_partitions)?);
        let mem_table_right: Arc<dyn TableProvider> =
            Arc::new(MemTable::try_new(right_schema, right_partitions)?);
        ctx.register_table("L", mem_table_left)?;
        ctx.register_table("R", mem_table_right)?;

        // Execute geography join query
        let df = ctx
            .sql("SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry)")
            .await?;
        let plan = df.create_physical_plan().await?;

        // Verify that no SpatialJoinExec is present (geography join should not be optimized)
        let spatial_joins = collect_spatial_join_exec(&plan)?;
        assert!(
            spatial_joins.is_empty(),
            "Geography joins should not be optimized to SpatialJoinExec"
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_query_window_in_subquery() -> Result<()> {
        let ((left_schema, left_partitions), (right_schema, right_partitions)) =
            create_test_data_with_size_range((50.0, 60.0), WKB_GEOMETRY)?;
        let options = SpatialJoinOptions::default();
        test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, 10,
                "SELECT id FROM L WHERE ST_Intersects(L.geometry, (SELECT R.geometry FROM R WHERE R.id = 1))").await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_parallel_refinement_for_large_candidate_set() -> Result<()> {
        let ((left_schema, left_partitions), (right_schema, right_partitions)) =
            create_test_data_with_size_range((1.0, 50.0), WKB_GEOMETRY)?;

        for max_batch_size in [10, 30, 100] {
            let options = SpatialJoinOptions {
                execution_mode: ExecutionMode::PrepareNone,
                parallel_refinement_chunk_size: 10,
                ..Default::default()
            };
            test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
                "SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) AND L.dist < R.dist ORDER BY L.id, R.id").await?;
        }

        Ok(())
    }

    async fn test_with_join_types(join_type: JoinType) -> Result<RecordBatch> {
        let ((left_schema, left_partitions), (right_schema, right_partitions)) =
            create_test_data_with_empty_partitions()?;

        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareNone,
            ..Default::default()
        };
        let batch_size = 30;

        let inner_sql = "SELECT L.id l_id, R.id r_id FROM L INNER JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id";
        let sql = match join_type {
            JoinType::Inner => inner_sql,
            JoinType::Left => "SELECT L.id l_id, R.id r_id FROM L LEFT JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
            JoinType::Right => "SELECT L.id l_id, R.id r_id FROM L RIGHT JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
            JoinType::Full => "SELECT L.id l_id, R.id r_id FROM L FULL OUTER JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
            JoinType::LeftSemi => "SELECT L.id l_id FROM L LEFT SEMI JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id",
            JoinType::RightSemi => "SELECT R.id r_id FROM L RIGHT SEMI JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY r_id",
            JoinType::LeftAnti => "SELECT L.id l_id FROM L LEFT ANTI JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id",
            JoinType::RightAnti => "SELECT R.id r_id FROM L RIGHT ANTI JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY r_id",
            JoinType::LeftMark => {
                unreachable!("LeftMark is not directly supported in SQL, will be tested in other tests");
            }
            JoinType::RightMark => {
                unreachable!("RightMark is not directly supported in SQL, will be tested in other tests");
            }
        };

        let batches = test_spatial_join_query(
            &left_schema,
            &right_schema,
            left_partitions.clone(),
            right_partitions.clone(),
            &options,
            batch_size,
            sql,
        )
        .await?;

        if matches!(join_type, JoinType::Left | JoinType::Right | JoinType::Full) {
            // Make sure that we are effectively testing outer joins. If outer joins produces the same result as inner join,
            // it means that the test data is not suitable for testing outer joins.
            let inner_batches = run_spatial_join_query(
                &left_schema,
                &right_schema,
                left_partitions,
                right_partitions,
                Some(options),
                batch_size,
                inner_sql,
            )
            .await?;
            assert!(inner_batches.num_rows() < batches.num_rows());
        }

        Ok(batches)
    }

    async fn test_spatial_join_query(
        left_schema: &SchemaRef,
        right_schema: &SchemaRef,
        left_partitions: Vec<Vec<RecordBatch>>,
        right_partitions: Vec<Vec<RecordBatch>>,
        options: &SpatialJoinOptions,
        batch_size: usize,
        sql: &str,
    ) -> Result<RecordBatch> {
        // Run spatial join using SpatialJoinExec
        let actual = run_spatial_join_query(
            left_schema,
            right_schema,
            left_partitions.clone(),
            right_partitions.clone(),
            Some(options.clone()),
            batch_size,
            sql,
        )
        .await?;

        // Run spatial join using NestedLoopJoinExec
        let expected = run_spatial_join_query(
            left_schema,
            right_schema,
            left_partitions.clone(),
            right_partitions.clone(),
            None,
            batch_size,
            sql,
        )
        .await?;

        // Should produce the same result
        assert!(expected.num_rows() > 0);
        assert_eq!(expected, actual);

        Ok(actual)
    }

    async fn run_spatial_join_query(
        left_schema: &SchemaRef,
        right_schema: &SchemaRef,
        left_partitions: Vec<Vec<RecordBatch>>,
        right_partitions: Vec<Vec<RecordBatch>>,
        options: Option<SpatialJoinOptions>,
        batch_size: usize,
        sql: &str,
    ) -> Result<RecordBatch> {
        let mem_table_left: Arc<dyn TableProvider> =
            Arc::new(MemTable::try_new(left_schema.to_owned(), left_partitions)?);
        let mem_table_right: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
            right_schema.to_owned(),
            right_partitions,
        )?);

        let is_optimized_spatial_join = options.is_some();
        let ctx = setup_context(options, batch_size)?;
        ctx.register_table("L", Arc::clone(&mem_table_left))?;
        ctx.register_table("R", Arc::clone(&mem_table_right))?;
        let df = ctx.sql(sql).await?;
        let actual_schema = df.schema().as_arrow().clone();
        let plan = df.clone().create_physical_plan().await?;
        let spatial_join_execs = collect_spatial_join_exec(&plan)?;
        if is_optimized_spatial_join {
            assert_eq!(spatial_join_execs.len(), 1);
        } else {
            assert!(spatial_join_execs.is_empty());
        }
        let result_batches = df.collect().await?;
        let result_batch =
            arrow::compute::concat_batches(&Arc::new(actual_schema), &result_batches)?;
        Ok(result_batch)
    }

    fn collect_spatial_join_exec(plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<&SpatialJoinExec>> {
        let mut spatial_join_execs = Vec::new();
        plan.apply(|node| {
            if let Some(spatial_join_exec) = node.as_any().downcast_ref::<SpatialJoinExec>() {
                spatial_join_execs.push(spatial_join_exec);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(spatial_join_execs)
    }

    fn collect_nested_loop_join_exec(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Vec<&NestedLoopJoinExec>> {
        let mut execs = Vec::new();
        plan.apply(|node| {
            if let Some(exec) = node.as_any().downcast_ref::<NestedLoopJoinExec>() {
                execs.push(exec);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(execs)
    }

    async fn test_mark_join(
        join_type: JoinType,
        options: SpatialJoinOptions,
        batch_size: usize,
    ) -> Result<()> {
        let ((left_schema, left_partitions), (right_schema, right_partitions)) =
            create_test_data_with_size_range((0.1, 10.0), WKB_GEOMETRY)?;
        let mem_table_left: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
            left_schema.clone(),
            left_partitions.clone(),
        )?);
        let mem_table_right: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
            right_schema.clone(),
            right_partitions.clone(),
        )?);

        // We use a Left Join as a template to create the plan, then modify it to Mark Join
        let sql = "SELECT * FROM L LEFT JOIN R ON ST_Intersects(L.geometry, R.geometry)";

        // Create SpatialJoinExec plan
        let ctx = setup_context(Some(options.clone()), batch_size)?;
        ctx.register_table("L", mem_table_left.clone())?;
        ctx.register_table("R", mem_table_right.clone())?;
        let df = ctx.sql(sql).await?;
        let plan = df.create_physical_plan().await?;
        let spatial_join_execs = collect_spatial_join_exec(&plan)?;
        assert_eq!(spatial_join_execs.len(), 1);
        let original_exec = spatial_join_execs[0];
        let mark_exec = SpatialJoinExec::try_new(
            original_exec.left.clone(),
            original_exec.right.clone(),
            original_exec.on.clone(),
            original_exec.filter.clone(),
            &join_type,
            None,
            &options,
        )?;

        // Create NestedLoopJoinExec plan for comparison
        let ctx_no_opt = setup_context(None, batch_size)?;
        ctx_no_opt.register_table("L", mem_table_left)?;
        ctx_no_opt.register_table("R", mem_table_right)?;
        let df_no_opt = ctx_no_opt.sql(sql).await?;
        let plan_no_opt = df_no_opt.create_physical_plan().await?;
        let nlj_execs = collect_nested_loop_join_exec(&plan_no_opt)?;
        assert_eq!(nlj_execs.len(), 1);
        let original_nlj = nlj_execs[0];
        let mark_nlj = NestedLoopJoinExec::try_new(
            original_nlj.children()[0].clone(),
            original_nlj.children()[1].clone(),
            original_nlj.filter().cloned(),
            &join_type,
            None,
        )?;

        async fn run_and_sort(
            plan: Arc<dyn ExecutionPlan>,
            ctx: &SessionContext,
        ) -> Result<RecordBatch> {
            let results = datafusion_physical_plan::collect(plan, ctx.task_ctx()).await?;
            let batch = arrow::compute::concat_batches(&results[0].schema(), &results)?;
            let sort_col = batch.column(0);
            let indices = arrow::compute::sort_to_indices(sort_col, None, None)?;
            let sorted_batch = arrow::compute::take_record_batch(&batch, &indices)?;
            Ok(sorted_batch)
        }

        // Run both Mark Join plans and compare results
        let mark_batch = run_and_sort(Arc::new(mark_exec), &ctx).await?;
        let mark_nlj_batch = run_and_sort(Arc::new(mark_nlj), &ctx_no_opt).await?;
        assert_eq!(mark_batch, mark_nlj_batch);

        Ok(())
    }

    fn extract_geoms_and_ids(partitions: &[Vec<RecordBatch>]) -> Vec<(i32, geo::Geometry<f64>)> {
        let mut result = Vec::new();
        for partition in partitions {
            for batch in partition {
                let id_idx = batch.schema().index_of("id").expect("Id column not found");
                let ids = batch
                    .column(id_idx)
                    .as_any()
                    .downcast_ref::<arrow_array::Int32Array>()
                    .expect("Column 'id' should be Int32");

                let geom_idx = batch
                    .schema()
                    .index_of("geometry")
                    .expect("Geometry column not found");

                let geoms_col = batch.column(geom_idx);
                let geom_type = SedonaType::from_storage_field(batch.schema().field(geom_idx))
                    .expect("Failed to get SedonaType from geometry field");
                let arg_types = [geom_type];
                let arg_values = [ColumnarValue::Array(Arc::clone(geoms_col))];

                let executor = GeoTypesExecutor::new(&arg_types, &arg_values);
                let mut id_iter = ids.iter();
                executor
                    .execute_wkb_void(|maybe_geom| {
                        if let Some(id_opt) = id_iter.next() {
                            if let (Some(id), Some(geom)) = (id_opt, maybe_geom) {
                                result.push((id, geom))
                            }
                        }
                        Ok(())
                    })
                    .expect("Failed to extract geoms and ids from RecordBatch");
            }
        }
        result
    }

    fn compute_knn_ground_truth(
        left_partitions: &[Vec<RecordBatch>],
        right_partitions: &[Vec<RecordBatch>],
        k: usize,
    ) -> Vec<(i32, i32, f64)> {
        let left_data = extract_geoms_and_ids(left_partitions);
        let right_data = extract_geoms_and_ids(right_partitions);

        let mut results = Vec::new();

        for (l_id, l_geom) in left_data {
            let mut distances: Vec<(i32, f64)> = right_data
                .iter()
                .map(|(r_id, r_geom)| (*r_id, Euclidean.distance(&l_geom, r_geom)))
                .collect();

            // Sort by distance, then by ID for stability
            distances.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

            for (r_id, dist) in distances.iter().take(k.min(distances.len())) {
                results.push((l_id, *r_id, *dist));
            }
        }

        // Sort results by L.id, R.id
        results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        results
    }

    #[tokio::test]
    async fn test_knn_join_correctness() -> Result<()> {
        // Generate slightly larger data
        let ((left_schema, left_partitions), (right_schema, right_partitions)) =
            create_knn_test_data((0.1, 10.0), WKB_GEOMETRY)?;

        let options = SpatialJoinOptions::default();
        let k = 3;

        let sql1 = format!(
            "SELECT L.id, R.id, ST_Distance(L.geometry, R.geometry) FROM L JOIN R ON ST_KNN(L.geometry, R.geometry, {}, false) ORDER BY L.id, R.id",
            k
        );
        let expected1 = compute_knn_ground_truth(&left_partitions, &right_partitions, k)
            .into_iter()
            .map(|(l, r, _)| (l, r))
            .collect::<Vec<_>>();

        let sql2 = format!(
            "SELECT R.id, L.id, ST_Distance(L.geometry, R.geometry) FROM L JOIN R ON ST_KNN(R.geometry, L.geometry, {}, false) ORDER BY R.id, L.id",
            k
        );
        let expected2 = compute_knn_ground_truth(&right_partitions, &left_partitions, k)
            .into_iter()
            .map(|(l, r, _)| (l, r))
            .collect::<Vec<_>>();

        let sqls = [(&sql1, &expected1), (&sql2, &expected2)];

        for (sql, expected_results) in sqls {
            let batches = run_spatial_join_query(
                &left_schema,
                &right_schema,
                left_partitions.clone(),
                right_partitions.clone(),
                Some(options.clone()),
                10,
                sql,
            )
            .await?;

            // Collect actual results
            let mut actual_results = Vec::new();
            let combined_batch = arrow::compute::concat_batches(&batches.schema(), &[batches])?;
            let l_ids = combined_batch
                .column(0)
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .unwrap();
            let r_ids = combined_batch
                .column(1)
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .unwrap();

            for i in 0..combined_batch.num_rows() {
                actual_results.push((l_ids.value(i), r_ids.value(i)));
            }
            actual_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

            assert_eq!(actual_results, *expected_results);
        }

        Ok(())
    }

    #[test]
    fn test_try_swapping_with_projection_pushes_down_and_rewrites_relation_predicate() -> Result<()>
    {
        use crate::spatial_predicate::{RelationPredicate, SpatialRelationType};

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
