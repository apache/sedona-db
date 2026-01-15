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

use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use crate::build_index::build_index;
use crate::config::GpuSpatialJoinConfig;
use crate::index::SpatialIndex;
use crate::spatial_predicate::SpatialPredicate;
use crate::stream::GpuSpatialJoinMetrics;
use crate::utils::join_utils::{asymmetric_join_output_partitioning, boundedness_from_children};
use crate::utils::once_fut::OnceAsync;
use crate::GpuSpatialJoinStream;
use arrow::datatypes::SchemaRef;
use datafusion::error::Result;
use datafusion::execution::context::TaskContext;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::{
    joins::utils::build_join_schema, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion_common::{project_schema, JoinType};
use datafusion_physical_expr::equivalence::{join_equivalence_properties, ProjectionMapping};
use datafusion_physical_expr::expressions::{BinaryExpr, Column};
use datafusion_physical_expr::Partitioning;
use datafusion_physical_plan::joins::utils::{check_join_is_valid, ColumnIndex, JoinFilter};
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::ExecutionPlanProperties;
use parking_lot::Mutex;
use sedona_common::SedonaOptions;

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
/// GPU-accelerated spatial join execution plan
///
/// This execution plan accepts two child inputs (e.g., ParquetExec) and performs:
/// 1. Reading data from child streams
/// 2. Data transfer to GPU memory
/// 3. GPU spatial join execution
/// 4. Result materialization
#[derive(Debug)]
pub struct GpuSpatialJoinExec {
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
    /// Spatial index built asynchronously on first execute() call and shared across all partitions.
    /// Uses OnceAsync for lazy initialization coordinated via async runtime.
    once_async_spatial_index: Arc<Mutex<Option<OnceAsync<Arc<SpatialIndex>>>>>,
    /// Indicates if this SpatialJoin was converted from a HashJoin
    /// When true, we preserve HashJoin's equivalence properties and partitioning
    converted_from_hash_join: bool,
    config: GpuSpatialJoinConfig,
}

impl GpuSpatialJoinExec {
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        on: SpatialPredicate,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
        config: GpuSpatialJoinConfig,
    ) -> Result<Self> {
        Self::try_new_with_options(
            left, right, on, filter, join_type, projection, false, config,
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
        converted_from_hash_join: bool,
        config: GpuSpatialJoinConfig,
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

        Ok(GpuSpatialJoinExec {
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
            config,
        })
    }
    fn maintains_input_order(join_type: JoinType) -> Vec<bool> {
        vec![
            false,
            matches!(
                join_type,
                JoinType::Inner | JoinType::Right | JoinType::RightAnti | JoinType::RightSemi
            ),
        ]
    }
    pub fn config(&self) -> &GpuSpatialJoinConfig {
        &self.config
    }

    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Does this join has a projection on the joined columns
    pub fn contains_projection(&self) -> bool {
        self.projection.is_some()
    }

    /// Get the projection indices
    pub fn projection(&self) -> Option<&Vec<usize>> {
        self.projection.as_ref()
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

impl DisplayAs for GpuSpatialJoinExec {
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
                    "GpuSpatialJoinExec: join_type={:?}{}{}{}",
                    self.join_type, display_on, display_filter, display_projections
                )
            }
            DisplayFormatType::TreeRender => {
                if self.join_type != JoinType::Inner {
                    writeln!(f, "join_type={:?}", self.join_type)
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl ExecutionPlan for GpuSpatialJoinExec {
    fn name(&self) -> &str {
        "GpuSpatialJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
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
        Ok(Arc::new(GpuSpatialJoinExec {
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
            config: Default::default(),
        }))
    }

    fn metrics(&self) -> Option<datafusion_physical_plan::metrics::MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        // Regular spatial join logic - standard left=build, right=probe semantics
        let session_config = context.session_config();
        let sedona_options = session_config
            .options()
            .extensions
            .get::<SedonaOptions>()
            .cloned()
            .unwrap_or_default();

        // Regular join semantics: left is build, right is probe
        let (build_plan, probe_plan) = (&self.left, &self.right);

        // Phase 1: Build Phase (runs once, shared across all output partitions)
        // Get or create the shared build data future
        let once_fut_spatial_index = {
            let mut once = self.once_async_spatial_index.lock();
            once.get_or_insert(OnceAsync::default()).try_once(|| {
                let build_side = build_plan;

                let num_partitions = build_side.output_partitioning().partition_count();
                let mut build_streams = Vec::with_capacity(num_partitions);
                for k in 0..num_partitions {
                    let stream = build_side.execute(k, Arc::clone(&context))?;
                    build_streams.push(stream);
                }
                let probe_thread_count = self.right.output_partitioning().partition_count();

                Ok(build_index(
                    Arc::clone(&context),
                    build_streams,
                    self.on.clone(),
                    self.join_type,
                    probe_thread_count,
                    self.metrics.clone(),
                    self.config.clone(),
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
        let join_metrics = GpuSpatialJoinMetrics::new(partition, &self.metrics);
        let probe_stream = probe_plan.execute(partition, Arc::clone(&context))?;

        // For regular joins: probe is right side (index 1)
        let probe_side_ordered =
            self.maintains_input_order()[1] && self.right.output_ordering().is_some();

        Ok(Box::pin(GpuSpatialJoinStream::new(
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
            once_fut_spatial_index,
            Arc::clone(&self.once_async_spatial_index),
        )))
    }
}
