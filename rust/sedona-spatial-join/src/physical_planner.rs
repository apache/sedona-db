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

use std::sync::Arc;

use arrow_schema::Schema;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::{JoinSide, Result};
use datafusion_physical_expr::PhysicalExpr;
use sedona_common::{sedona_internal_err, SpatialJoinOptions};
use sedona_query_planner::probe_shuffle_exec::ProbeShuffleExec;
use sedona_query_planner::spatial_join_physical_planner::{
    PlanSpatialJoinArgs, SpatialJoinPhysicalPlanner,
};
use sedona_query_planner::spatial_predicate::{DistancePredicate, KNNPredicate, RelationPredicate};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::exec::SpatialJoinExec;
use crate::spatial_predicate::SpatialPredicate;

/// [SpatialJoinFactory] implementation for the default spatial join
///
/// This struct is the entrypoint to ensuring the SedonaQueryPlanner is able
/// to instantiate the [ExecutionPlan] implemented in this crate.
#[derive(Debug)]
pub struct DefaultSpatialJoinPhysicalPlanner;

impl DefaultSpatialJoinPhysicalPlanner {
    /// Create a new default join factory
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DefaultSpatialJoinPhysicalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialJoinPhysicalPlanner for DefaultSpatialJoinPhysicalPlanner {
    fn plan_spatial_join(
        &self,
        args: &PlanSpatialJoinArgs<'_>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if !is_spatial_predicate_supported(
            args.spatial_predicate,
            &args.physical_left.schema(),
            &args.physical_right.schema(),
        )? {
            return Ok(None);
        }

        let should_swap = !matches!(
            args.spatial_predicate,
            SpatialPredicate::KNearestNeighbors(_)
        ) && args.join_type.supports_swap()
            && should_swap_join_order(
                args.join_options,
                args.physical_left.as_ref(),
                args.physical_right.as_ref(),
            )?;

        // Repartition the probe side when enabled. This breaks spatial locality in sorted/skewed
        // datasets, leading to more balanced workloads during out-of-core spatial join.
        // We determine which pre-swap input will be the probe AFTER any potential swap, and
        // repartition it here. swap_inputs() will then carry the RepartitionExec to the correct
        // child position.
        let (physical_left, physical_right) = if args.join_options.repartition_probe_side {
            repartition_probe_side(
                args.physical_left.clone(),
                args.physical_right.clone(),
                args.spatial_predicate,
                should_swap,
            )?
        } else {
            (args.physical_left.clone(), args.physical_right.clone())
        };

        let exec = SpatialJoinExec::try_new(
            physical_left,
            physical_right,
            args.spatial_predicate.clone(),
            args.remainder.cloned(),
            args.join_type,
            None,
            args.join_options,
        )?;

        if should_swap {
            exec.swap_inputs().map(Some)
        } else {
            Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
        }
    }
}

/// Spatial join reordering heuristic:
/// 1. Put the input with fewer rows on the build side, because fewer entries
///    produce a smaller and more efficient spatial index (R-tree).
/// 2. If row-count statistics are unavailable (for example, for CSV sources),
///    fall back to total input size as an estimate.
/// 3. Do not swap the join order if join reordering is disabled or no relevant
///    statistics are available.
pub fn should_swap_join_order(
    spatial_join_options: &SpatialJoinOptions,
    left: &dyn ExecutionPlan,
    right: &dyn ExecutionPlan,
) -> Result<bool> {
    if !spatial_join_options.spatial_join_reordering {
        log::info!(
            "spatial join swap heuristic disabled via sedona.spatial_join.spatial_join_reordering"
        );
        return Ok(false);
    }

    let left_stats = left.partition_statistics(None)?;
    let right_stats = right.partition_statistics(None)?;

    let left_num_rows = left_stats.num_rows;
    let right_num_rows = right_stats.num_rows;
    let left_total_byte_size = left_stats.total_byte_size;
    let right_total_byte_size = right_stats.total_byte_size;

    let should_swap = match (left_num_rows.get_value(), right_num_rows.get_value()) {
        (Some(l), Some(r)) => l > r,
        _ => match (
            left_total_byte_size.get_value(),
            right_total_byte_size.get_value(),
        ) {
            (Some(l), Some(r)) => l > r,
            _ => false,
        },
    };

    log::info!(
        "spatial join swap heuristic: left_num_rows={left_num_rows:?}, right_num_rows={right_num_rows:?}, left_total_byte_size={left_total_byte_size:?}, right_total_byte_size={right_total_byte_size:?}, should_swap={should_swap}"
    );

    Ok(should_swap)
}

/// Repartition the probe side of a spatial join using `RoundRobinBatch` partitioning.
///
/// The purpose is to break spatial locality in sorted or skewed datasets, which can cause
/// imbalanced partitions when running out-of-core spatial join. The number of partitions is
/// preserved; only the distribution of rows across partitions is shuffled.
///
/// The `should_swap` parameter indicates whether `swap_inputs()` will be called after
/// `SpatialJoinExec` is constructed. This affects which pre-swap input will become the
/// probe side:
/// - For non-KNN predicates: probe is always `Right` after any swap. If `should_swap` is true,
///   the current `left` will become `right` (probe) after swap, so we repartition `left`.
/// - For KNN predicates: `should_swap` is always false, and the probe side is determined by
///   `KNNPredicate::probe_side`.
pub fn repartition_probe_side(
    mut physical_left: Arc<dyn ExecutionPlan>,
    mut physical_right: Arc<dyn ExecutionPlan>,
    spatial_predicate: &SpatialPredicate,
    should_swap: bool,
) -> Result<(Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>)> {
    let probe_plan = match spatial_predicate {
        SpatialPredicate::KNearestNeighbors(knn) => match knn.probe_side {
            JoinSide::Left => &mut physical_left,
            JoinSide::Right => &mut physical_right,
            JoinSide::None => {
                // KNNPredicate::probe_side is asserted not to be None in its constructor;
                // treat this as a debug-only invariant violation and default to right.
                debug_assert!(false, "KNNPredicate::probe_side must not be JoinSide::None");
                &mut physical_right
            }
        },
        _ => {
            // For Relation/Distance predicates, probe is always Right after swap.
            // If should_swap, the current left will be moved to the right (probe) by swap_inputs().
            if should_swap {
                &mut physical_left
            } else {
                &mut physical_right
            }
        }
    };

    *probe_plan = Arc::new(ProbeShuffleExec::try_new(Arc::clone(probe_plan))?);

    Ok((physical_left, physical_right))
}

pub fn is_spatial_predicate_supported(
    spatial_predicate: &SpatialPredicate,
    left_schema: &Schema,
    right_schema: &Schema,
) -> Result<bool> {
    /// Only spatial predicates working with planar geometry are supported for optimization.
    /// Geography (spherical) types are explicitly excluded and will not trigger optimized spatial joins.
    fn is_geometry_type_supported(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Result<bool> {
        let left_return_field = expr.return_field(schema)?;
        let sedona_type = SedonaType::from_storage_field(&left_return_field)?;
        let matcher = ArgMatcher::is_geometry();
        Ok(matcher.match_type(&sedona_type))
    }

    match spatial_predicate {
        SpatialPredicate::Relation(RelationPredicate { left, right, .. })
        | SpatialPredicate::Distance(DistancePredicate { left, right, .. }) => {
            Ok(is_geometry_type_supported(left, left_schema)?
                && is_geometry_type_supported(right, right_schema)?)
        }
        SpatialPredicate::KNearestNeighbors(KNNPredicate {
            left,
            right,
            probe_side,
            ..
        }) => {
            let (left, right) = match probe_side {
                JoinSide::Left => (left, right),
                JoinSide::Right => (right, left),
                _ => {
                    return sedona_internal_err!(
                        "Invalid probe side in KNN predicate: {:?}",
                        probe_side
                    )
                }
            };
            Ok(is_geometry_type_supported(left, left_schema)?
                && is_geometry_type_supported(right, right_schema)?)
        }
    }
}

#[cfg(test)]
mod test {
    use arrow_schema::{DataType, Field};
    use datafusion_physical_expr::expressions::Column;
    use sedona_query_planner::spatial_predicate::SpatialRelationType;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_GEOMETRY};

    use super::*;

    #[test]
    fn test_is_spatial_predicate_supported() {
        // Planar geometry field
        let geom_field = WKB_GEOMETRY.to_storage_field("geom", false).unwrap();
        let schema = Arc::new(Schema::new(vec![geom_field.clone()]));
        let col_expr = Arc::new(Column::new("geom", 0)) as Arc<dyn PhysicalExpr>;
        let rel_pred = RelationPredicate::new(
            col_expr.clone(),
            col_expr.clone(),
            SpatialRelationType::Intersects,
        );
        let spatial_pred = SpatialPredicate::Relation(rel_pred);
        assert!(is_spatial_predicate_supported(&spatial_pred, &schema, &schema).unwrap());

        // Geography field (should NOT be supported)
        let geog_field = WKB_GEOGRAPHY.to_storage_field("geog", false).unwrap();
        let geog_schema = Arc::new(Schema::new(vec![geog_field.clone()]));
        let geog_col_expr = Arc::new(Column::new("geog", 0)) as Arc<dyn PhysicalExpr>;
        let rel_pred_geog = RelationPredicate::new(
            geog_col_expr.clone(),
            geog_col_expr.clone(),
            SpatialRelationType::Intersects,
        );
        let spatial_pred_geog = SpatialPredicate::Relation(rel_pred_geog);
        assert!(
            !is_spatial_predicate_supported(&spatial_pred_geog, &geog_schema, &geog_schema)
                .unwrap()
        );
    }

    #[test]
    fn test_is_knn_predicate_supported() {
        // ST_KNN(left, right)
        let left_schema = Arc::new(Schema::new(vec![WKB_GEOMETRY
            .to_storage_field("geom", false)
            .unwrap()]));
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            WKB_GEOMETRY.to_storage_field("geom", false).unwrap(),
        ]));
        let left_col_expr = Arc::new(Column::new("geom", 0)) as Arc<dyn PhysicalExpr>;
        let right_col_expr = Arc::new(Column::new("geom", 1)) as Arc<dyn PhysicalExpr>;
        let knn_pred = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            left_col_expr.clone(),
            right_col_expr.clone(),
            5,
            false,
            JoinSide::Left,
        ));
        assert!(is_spatial_predicate_supported(&knn_pred, &left_schema, &right_schema).unwrap());

        // ST_KNN(right, left)
        let knn_pred = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            right_col_expr.clone(),
            left_col_expr.clone(),
            5,
            false,
            JoinSide::Right,
        ));
        assert!(is_spatial_predicate_supported(&knn_pred, &left_schema, &right_schema).unwrap());

        // ST_KNN with geography (should NOT be supported)
        let left_geog_schema = Arc::new(Schema::new(vec![WKB_GEOGRAPHY
            .to_storage_field("geog", false)
            .unwrap()]));
        assert!(
            !is_spatial_predicate_supported(&knn_pred, &left_geog_schema, &right_schema).unwrap()
        );

        let right_geog_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            WKB_GEOGRAPHY.to_storage_field("geog", false).unwrap(),
        ]));
        assert!(
            !is_spatial_predicate_supported(&knn_pred, &left_schema, &right_geog_schema).unwrap()
        );
    }
}
