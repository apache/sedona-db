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
use datafusion_expr::{JoinType, Operator};
use datafusion_physical_expr::{
    equivalence::{join_equivalence_properties, ProjectionMapping},
    expressions::{BinaryExpr, Column},
    PhysicalExpr,
};
use datafusion_physical_plan::{
    execution_plan::EmissionType,
    joins::utils::{build_join_schema, check_join_is_valid, ColumnIndex, JoinFilter},
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties,
};
use parking_lot::Mutex;
use sedona_common::{sedona_internal_err, SpatialJoinOptions};

use crate::{
    prepare::{SpatialJoinComponents, SpatialJoinComponentsBuilder},
    spatial_predicate::{KNNPredicate, SpatialPredicate},
    stream::SpatialJoinStream,
    utils::{
        join_utils::{asymmetric_join_output_partitioning, boundedness_from_children},
        once_fut::OnceAsync,
    },
};

/// Type alias for build and probe execution plans
type BuildProbePlans<'a> = (&'a Arc<dyn ExecutionPlan>, &'a Arc<dyn ExecutionPlan>);

/// Extract equality join conditions from a JoinFilter
/// Returns column pairs that represent equality conditions as PhysicalExprs
fn extract_equality_conditions(
    filter: &JoinFilter,
) -> Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> {
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
    /// Indicates if this SpatialJoin was converted from a HashJoin
    /// When true, we preserve HashJoin's equivalence properties and partitioning
    converted_from_hash_join: bool,
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
        Self::try_new_with_options(
            left, right, on, filter, join_type, projection, options, false,
        )
    }

    /// Create a new SpatialJoinExec with additional options
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_with_options(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: SpatialPredicate,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
        options: &SpatialJoinOptions,
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
            &on,
            Arc::clone(&join_schema),
            *join_type,
            projection.as_ref(),
            filter.as_ref(),
            converted_from_hash_join,
        )?;
        let seed = options
            .debug
            .random_seed
            .unwrap_or(fastrand::u64(0..0xFFFF));

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
            converted_from_hash_join,
            seed,
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
    #[allow(clippy::too_many_arguments)]
    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        on: &SpatialPredicate,
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
        let mut output_partitioning = if let SpatialPredicate::KNearestNeighbors(knn) = on {
            match knn.probe_side {
                JoinSide::Left => left.output_partitioning().clone(),
                JoinSide::Right => right.output_partitioning().clone(),
                _ => asymmetric_join_output_partitioning(left, right, &join_type)?,
            }
        } else if converted_from_hash_join {
            // Replicate HashJoin's symmetric partitioning logic
            // HashJoin preserves partitioning from both sides for inner joins
            // and from one side for outer joins

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
                _ => asymmetric_join_output_partitioning(left, right, &join_type)?,
            }
        } else {
            asymmetric_join_output_partitioning(left, right, &join_type)?
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
            once_async_spatial_join_components: Arc::new(Mutex::new(None)),
            converted_from_hash_join: self.converted_from_hash_join,
            seed: self.seed,
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
