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
use datafusion_common::{project_schema, DataFusionError, JoinSide, Result};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::JoinType;
use datafusion_physical_expr::equivalence::{join_equivalence_properties, ProjectionMapping};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::{
    execution_plan::EmissionType,
    joins::utils::{build_join_schema, check_join_is_valid, ColumnIndex, JoinFilter},
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use parking_lot::Mutex;

use crate::{
    index::{build_index, SpatialIndex, SpatialJoinBuildMetrics},
    once_fut::OnceAsync,
    spatial_predicate::{KNNPredicate, SpatialPredicate},
    stream::{SpatialJoinProbeMetrics, SpatialJoinStream},
    utils::{asymmetric_join_output_partitioning, boundedness_from_children},
    // Re-export from sedona-common
    SedonaOptions,
};

/// Type alias for build and probe execution plans
type BuildProbePlans<'a> = (&'a Arc<dyn ExecutionPlan>, &'a Arc<dyn ExecutionPlan>);

/// Extract equality join conditions from a JoinFilter
/// Returns column pairs that represent equality conditions as PhysicalExprs
fn extract_equality_conditions(
    filter: &JoinFilter,
) -> Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> {
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, Column};

    let mut equalities = Vec::new();

    if let Some(binary_expr) = filter.expression().as_any().downcast_ref::<BinaryExpr>() {
        if binary_expr.op() == &Operator::Eq {
            // Check if both sides are column references
            if let (Some(_left_col), Some(_right_col)) = (
                binary_expr.left().as_any().downcast_ref::<Column>(),
                binary_expr.right().as_any().downcast_ref::<Column>(),
            ) {
                equalities.push((binary_expr.left().clone(), binary_expr.right().clone()));
            }
        }
    }

    equalities
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
    _join_schema: &SchemaRef,
) -> Result<BuildProbePlans<'a>> {
    // Use the probe_side information from the optimizer to determine build/probe assignment
    match knn_pred.probe_side {
        JoinSide::Left => Ok((right_plan, left_plan)),
        JoinSide::Right => Ok((left_plan, right_plan)),
        JoinSide::None => Err(DataFusionError::Internal(
            "KNN join requires explicit probe_side designation".to_string(),
        )),
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
    metrics: ExecutionPlanMetricsSet,
    /// The projection indices of the columns in the output schema of join
    projection: Option<Vec<usize>>,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: PlanProperties,
    /// Once future for building the spatial index.
    /// This futures run only once before the spatial index probing phase. It can also be disposed
    /// by the last finished stream so that the spatial index does not have to live as long as
    /// `SpatialJoinExec`.
    once_async_spatial_index: Arc<Mutex<Option<OnceAsync<SpatialIndex>>>>,
    /// Indicates if this SpatialJoin was converted from a HashJoin
    /// When true, we preserve HashJoin's equivalence properties and partitioning
    converted_from_hash_join: bool,
}

impl SpatialJoinExec {
    // Try to create a new [`SpatialJoinExec`]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: SpatialPredicate,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        Self::try_new_with_options(left, right, on, filter, join_type, projection, false)
    }

    /// Create a new SpatialJoinExec with additional options
    pub fn try_new_with_options(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: SpatialPredicate,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
        converted_from_hash_join: bool,
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
            Arc::clone(&join_schema),
            *join_type,
            projection.as_ref(),
            filter.as_ref(),
            converted_from_hash_join,
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
            once_async_spatial_index: Arc::new(Mutex::new(None)),
            converted_from_hash_join,
        })
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    /// Returns a vector indicating whether the left and right inputs maintain their order.
    /// The first element corresponds to the left input, and the second to the right.
    ///
    /// The left (build-side) input's order may change, but the right (probe-side) input's
    /// order is maintained for INNER, RIGHT, RIGHT ANTI, and RIGHT SEMI joins.
    ///
    /// Maintaining the right input's order helps optimize the nodes down the pipeline
    /// (See [`ExecutionPlan::maintains_input_order`]).
    ///
    /// This is a separate method because it is also called when computing properties, before
    /// a [`NestedLoopJoinExec`] is created. It also takes [`JoinType`] as an argument, as
    /// opposed to `Self`, for the same reason.
    fn maintains_input_order(join_type: JoinType) -> Vec<bool> {
        vec![
            false,
            matches!(
                join_type,
                JoinType::Inner | JoinType::Right | JoinType::RightAnti | JoinType::RightSemi
            ),
        ]
    }

    /// Does this join has a projection on the joined columns
    pub fn contains_projection(&self) -> bool {
        self.projection.is_some()
    }

    /// This function creates the cache object that stores the plan properties such as schema,
    /// equivalence properties, ordering, partitioning, etc.
    ///
    /// NOTICE: The implementation of this function should be identical to the one in
    /// [`datafusion_physical_plan::physical_plan::join::NestedLoopJoinExec::compute_properties`].
    /// This is because SpatialJoinExec is transformed from NestedLoopJoinExec in physical plan
    /// optimization phase. If the properties are not the same, the plan will be incorrect.
    ///
    /// When converted from HashJoin, we preserve HashJoin's equivalence properties by extracting
    /// equality conditions from the filter.
    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        join_type: JoinType,
        projection: Option<&Vec<usize>>,
        filter: Option<&JoinFilter>,
        converted_from_hash_join: bool,
    ) -> Result<PlanProperties> {
        // Extract equality conditions from filter if this was converted from HashJoin
        let on_columns = if converted_from_hash_join {
            filter.map_or(vec![], extract_equality_conditions)
        } else {
            vec![]
        };

        let mut eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            &join_type,
            Arc::clone(&schema),
            &[false, false],
            None,
            // Pass extracted equality conditions to preserve equivalences
            &on_columns,
        );

        // Use symmetric partitioning (like HashJoin) when converted from HashJoin
        // Otherwise use asymmetric partitioning (like NestedLoopJoin)
        let mut output_partitioning = if converted_from_hash_join {
            // Replicate HashJoin's symmetric partitioning logic
            // HashJoin preserves partitioning from both sides for inner joins
            // and from one side for outer joins
            use datafusion_physical_plan::Partitioning;
            match join_type {
                JoinType::Inner | JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti => {
                    left.output_partitioning().clone()
                }
                JoinType::Right | JoinType::RightSemi | JoinType::RightAnti => {
                    right.output_partitioning().clone()
                }
                JoinType::Full => {
                    // For full outer join, we can't preserve partitioning
                    Partitioning::UnknownPartitioning(left.output_partitioning().partition_count())
                }
                _ => asymmetric_join_output_partitioning(left, right, &join_type),
            }
        } else {
            asymmetric_join_output_partitioning(left, right, &join_type)
        };

        if let Some(projection) = projection {
            // construct a map from the input expressions to the output expression of the Projection
            let projection_mapping = ProjectionMapping::from_indices(projection, &schema)?;
            let out_schema = project_schema(&schema, Some(projection))?;
            let eq_props = eq_properties?;
            output_partitioning = output_partitioning.project(&projection_mapping, &eq_props);
            eq_properties = Ok(eq_props.project(&projection_mapping, out_schema));
        }

        let emission_type = if left.boundedness().is_unbounded() {
            EmissionType::Final
        } else if right.pipeline_behavior() == EmissionType::Incremental {
            match join_type {
                // If we only need to generate matched rows from the probe side,
                // we can emit rows incrementally.
                JoinType::Inner
                | JoinType::LeftSemi
                | JoinType::RightSemi
                | JoinType::Right
                | JoinType::RightAnti => EmissionType::Incremental,
                // If we need to generate unmatched rows from the *build side*,
                // we need to emit them at the end.
                JoinType::Left
                | JoinType::LeftAnti
                | JoinType::LeftMark
                | JoinType::RightMark
                | JoinType::Full => EmissionType::Both,
            }
        } else {
            right.pipeline_behavior()
        };

        Ok(PlanProperties::new(
            eq_properties?,
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
        Self::maintains_input_order(self.join_type)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(SpatialJoinExec {
            left: children[0].clone(),
            right: children[1].clone(),
            on: self.on.clone(),
            filter: self.filter.clone(),
            join_type: self.join_type,
            join_schema: self.join_schema.clone(),
            column_indices: self.column_indices.clone(),
            projection: self.projection.clone(),
            metrics: Default::default(),
            cache: self.cache.clone(),
            once_async_spatial_index: Arc::new(Mutex::new(None)),
            converted_from_hash_join: self.converted_from_hash_join,
        }))
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
            _ => {
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

                // Build the spatial index
                let once_fut_spatial_index = {
                    let mut once_async = self.once_async_spatial_index.lock();
                    once_async
                        .get_or_insert(OnceAsync::default())
                        .try_once(|| {
                            let build_side = build_plan;

                            let num_partitions = build_side.output_partitioning().partition_count();
                            let mut build_streams = Vec::with_capacity(num_partitions);
                            let mut build_metrics = Vec::with_capacity(num_partitions);
                            for k in 0..num_partitions {
                                let stream = build_side.execute(k, Arc::clone(&context))?;
                                build_streams.push(stream);
                                build_metrics.push(SpatialJoinBuildMetrics::new(k, &self.metrics));
                            }

                            let probe_thread_count =
                                self.right.output_partitioning().partition_count();

                            Ok(build_index(
                                build_side.schema(),
                                build_streams,
                                self.on.clone(),
                                sedona_options.spatial_join.clone(),
                                build_metrics,
                                Arc::clone(context.memory_pool()),
                                self.join_type,
                                probe_thread_count,
                            ))
                        })?
                };

                // Column indices for regular joins - no swapping needed
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
                    once_fut_spatial_index,
                    Arc::clone(&self.once_async_spatial_index),
                )))
            }
        }
    }
}

impl SpatialJoinExec {
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
            determine_knn_build_probe_plans(knn_pred, &self.left, &self.right, &self.join_schema)?;

        // Determine if probe plan is the left execution plan (for column index swapping logic)
        let actual_probe_plan_is_left = std::ptr::eq(probe_plan.as_ref(), self.left.as_ref());

        // Build the spatial index
        let once_fut_spatial_index = {
            let mut once_async = self.once_async_spatial_index.lock();
            once_async
                .get_or_insert(OnceAsync::default())
                .try_once(|| {
                    let build_side = build_plan;

                    let num_partitions = build_side.output_partitioning().partition_count();
                    let mut build_streams = Vec::with_capacity(num_partitions);
                    let mut build_metrics = Vec::with_capacity(num_partitions);
                    for k in 0..num_partitions {
                        let stream = build_side.execute(k, Arc::clone(&context))?;
                        build_streams.push(stream);
                        build_metrics.push(SpatialJoinBuildMetrics::new(k, &self.metrics));
                    }

                    let probe_thread_count = self.right.output_partitioning().partition_count();

                    Ok(build_index(
                        build_side.schema(),
                        build_streams,
                        self.on.clone(),
                        sedona_options.spatial_join.clone(),
                        build_metrics,
                        Arc::clone(context.memory_pool()),
                        self.join_type,
                        probe_thread_count,
                    ))
                })?
        };

        // Handle column indices for KNN - need to swap if we swapped execution plans
        let mut column_indices_after_projection = match &self.projection {
            Some(projection) => projection
                .iter()
                .map(|i| self.column_indices[*i].clone())
                .collect(),
            None => self.column_indices.clone(),
        };

        // If we swapped execution plans for KNN, we need to swap the column indices too
        if !actual_probe_plan_is_left {
            for col_idx in &mut column_indices_after_projection {
                match col_idx.side {
                    datafusion_common::JoinSide::Left => {
                        col_idx.side = datafusion_common::JoinSide::Right
                    }
                    datafusion_common::JoinSide::Right => {
                        col_idx.side = datafusion_common::JoinSide::Left
                    }
                    datafusion_common::JoinSide::None => {} // No change needed
                }
            }
        }

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
            once_fut_spatial_index,
            Arc::clone(&self.once_async_spatial_index),
        )))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        catalog::{MemTable, TableProvider},
        execution::SessionStateBuilder,
        prelude::{SessionConfig, SessionContext},
    };
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use geo_types::{Coord, Rect};
    use rstest::rstest;
    use sedona_geometry::types::GeometryTypeId;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY};
    use sedona_testing::datagen::RandomPartitionedDataBuilder;
    use tokio::sync::OnceCell;

    use crate::register_spatial_join_optimizer;
    use sedona_common::{
        option::{add_sedona_option_extension, ExecutionMode, SpatialJoinOptions},
        SpatialLibrary,
    };

    use super::*;

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
            .seed(1)
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

    fn setup_context(
        options: Option<SpatialJoinOptions>,
        batch_size: usize,
    ) -> Result<SessionContext> {
        let mut session_config = SessionConfig::from_env()?
            .with_information_schema(true)
            .with_batch_size(batch_size);
        session_config = add_sedona_option_extension(session_config);
        let mut state_builder = SessionStateBuilder::new();
        if let Some(options) = options {
            state_builder = register_spatial_join_optimizer(state_builder);
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
            let udf = function_set.add_scalar_udf_kernel(name, kernel)?;
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
        #[values(JoinType::Left, /* JoinType::LeftSemi, JoinType::LeftAnti */)] join_type: JoinType,
    ) -> Result<()> {
        test_with_join_types(join_type).await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_right_joins(
        #[values(JoinType::Right, /* JoinType::RightSemi, JoinType::RightAnti */)]
        join_type: JoinType,
    ) -> Result<()> {
        test_with_join_types(join_type).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_full_outer_join() -> Result<()> {
        test_with_join_types(JoinType::Full).await?;
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
            JoinType::LeftSemi => "SELECT L.id l_id FROM L WHERE EXISTS (SELECT 1 FROM R WHERE ST_Intersects(L.geometry, R.geometry)) ORDER BY l_id",
            JoinType::RightSemi => "SELECT R.id r_id FROM R WHERE EXISTS (SELECT 1 FROM L WHERE ST_Intersects(L.geometry, R.geometry)) ORDER BY r_id",
            JoinType::LeftAnti => "SELECT L.id l_id FROM L WHERE NOT EXISTS (SELECT 1 FROM R WHERE ST_Intersects(L.geometry, R.geometry)) ORDER BY l_id",
            JoinType::RightAnti => "SELECT R.id r_id FROM R WHERE NOT EXISTS (SELECT 1 FROM L WHERE ST_Intersects(L.geometry, R.geometry)) ORDER BY r_id",
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
}
