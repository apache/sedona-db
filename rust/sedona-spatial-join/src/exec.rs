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
    joins::utils::{build_join_schema, check_join_is_valid, ColumnIndex, JoinFilter},
    metrics::{ExecutionPlanMetricsSet, MetricsSet},
    projection::{EmbeddedProjection, ProjectionExec},
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use parking_lot::Mutex;
use sedona_common::{sedona_internal_err, SpatialJoinOptions};

use crate::{
    prepare::{SpatialJoinComponents, SpatialJoinComponentsBuilder},
    spatial_predicate::{KNNPredicate, SpatialPredicate},
    stream::SpatialJoinStream,
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
        // TODO: implement this
        todo!()
    }

    pub fn with_projection(&self, _projection: Option<Vec<usize>>) -> Result<Self> {
        // TODO: implement this
        todo!()
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
        _projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        // TODO: implement this
        Ok(None)
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
