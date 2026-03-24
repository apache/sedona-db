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

//! DataFusion planner integration for Sedona spatial joins.
//!
//! This module wires Sedona's logical optimizer rules and physical planning extensions that
//! can produce `SpatialJoinExec`.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::{JoinSide, Result};
use sedona_spatial_join_common::extension_planner::{
    PlanSpatialJoinArgs, SedonaSpatialJoinFactory,
};
use sedona_spatial_join_common::probe_shuffle_exec::ProbeShuffleExec;
use sedona_spatial_join_common::spatial_expr_utils::is_spatial_predicate_supported;

use crate::exec::SpatialJoinExec;
use crate::spatial_predicate::SpatialPredicate;

#[derive(Debug)]
pub struct DefaultSpatialJoinFactory;

impl DefaultSpatialJoinFactory {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DefaultSpatialJoinFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl SedonaSpatialJoinFactory for DefaultSpatialJoinFactory {
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
            && should_swap_join_order(args.physical_left.as_ref(), args.physical_right.as_ref())?;

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
/// 3. Do not swap the join order if no relevant statistics are available.
fn should_swap_join_order(left: &dyn ExecutionPlan, right: &dyn ExecutionPlan) -> Result<bool> {
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
fn repartition_probe_side(
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
