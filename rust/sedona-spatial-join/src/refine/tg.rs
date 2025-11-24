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
use std::{
    marker::PhantomData,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

use datafusion_common::{DataFusionError, Result};
use sedona_common::{sedona_internal_err, ExecutionMode, SpatialJoinOptions, TgIndexType};
use sedona_expr::statistics::GeoStatistics;
use sedona_tg::tg::{self, BinaryPredicate};
use wkb::reader::Wkb;

use crate::{
    index::IndexQueryResult,
    refine::{
        exec_mode_selector::{get_or_update_execution_mode, ExecModeSelector, SelectOptimalMode},
        IndexQueryResultRefiner,
    },
    spatial_predicate::{RelationPredicate, SpatialPredicate, SpatialRelationType},
    utils::init_once_array::InitOnceArray,
};

/// TG-specific optimal mode selector that chooses the best execution mode
/// based on geometry complexity and TG library characteristics.
struct TgOptimalModeSelector {
    predicate: SpatialPredicate,
}

impl TgOptimalModeSelector {
    fn select_intersects(
        &self,
        build_stats: &GeoStatistics,
        probe_stats: &GeoStatistics,
    ) -> ExecutionMode {
        let build_mean_points_per_geometry = build_stats.mean_points_per_geometry().unwrap_or(0.0);
        let probe_mean_points_per_geometry = probe_stats.mean_points_per_geometry().unwrap_or(0.0);

        let max_mean_points_per_geometry =
            build_mean_points_per_geometry.max(probe_mean_points_per_geometry);
        if max_mean_points_per_geometry <= 32.0 {
            // If the mean points per geometry is less than 32, the geometries are not complex enough to
            // benefit from prepared geometries. TG itself will skip creating index for such geometries.
            // Please refer to `default_index_spread` in tg.c for more details.
            //
            // We select PrepareProbe here because we want TG to automatically figure out whether to create
            // index for each individual probe geometry. We don't use PrepareBuild because it will take
            // lots of memory storing not-prepared geometries, while it does not trade much for the
            // performance.
            ExecutionMode::PrepareProbe
        } else {
            // Choose a more complex side to prepare the geometries
            if build_mean_points_per_geometry > probe_mean_points_per_geometry {
                ExecutionMode::PrepareBuild
            } else {
                ExecutionMode::PrepareProbe
            }
        }
    }

    fn select_contains_covers(&self, build_stats: &GeoStatistics) -> ExecutionMode {
        let build_mean_points = build_stats.mean_points_per_geometry().unwrap_or(0.0);
        if build_mean_points >= 32.0 {
            ExecutionMode::PrepareBuild
        } else {
            ExecutionMode::PrepareNone
        }
    }
}

impl SelectOptimalMode for TgOptimalModeSelector {
    fn select(&self, build_stats: &GeoStatistics, probe_stats: &GeoStatistics) -> ExecutionMode {
        if matches!(
            &self.predicate,
            SpatialPredicate::Relation(RelationPredicate {
                relation_type: SpatialRelationType::Intersects,
                ..
            })
        ) {
            self.select_intersects(build_stats, probe_stats)
        } else {
            self.select_without_probe_stats(build_stats)
                .unwrap_or(ExecutionMode::PrepareNone)
        }
    }

    fn select_without_probe_stats(&self, build_stats: &GeoStatistics) -> Option<ExecutionMode> {
        match &self.predicate {
            SpatialPredicate::Distance(_) => Some(ExecutionMode::PrepareNone),
            SpatialPredicate::KNearestNeighbors(_) => Some(ExecutionMode::PrepareNone),
            SpatialPredicate::Relation(predicate) => {
                match predicate.relation_type {
                    SpatialRelationType::Intersects => {
                        // Both PrepareBuild and PrepareProbe can be used for intersects predicate.
                        // We need statistics from the probe side to select the optimal execution mode.
                        None
                    }
                    SpatialRelationType::Contains | SpatialRelationType::Covers => {
                        // PrepareBuild is the only execution mode that works for Contains and Covers.
                        // However, it needs additional memory so it is not always beneficial to
                        // use it.
                        Some(self.select_contains_covers(build_stats))
                    }
                    SpatialRelationType::Within | SpatialRelationType::CoveredBy => {
                        Some(ExecutionMode::PrepareProbe)
                    }
                    _ => {
                        // Other predicates cannot be accelerated by prepared geometries.
                        Some(ExecutionMode::PrepareNone)
                    }
                }
            }
        }
    }
}

/// A refiner that uses the tiny geometry library to evaluate spatial predicates.
pub(crate) struct TgRefiner {
    evaluator: Box<dyn TgPredicateEvaluator>,
    prepared_geoms: InitOnceArray<Option<tg::Geom>>,
    index_type: tg::IndexType,
    mem_usage: AtomicUsize,
    exec_mode: OnceLock<ExecutionMode>,
    exec_mode_selector: Option<ExecModeSelector>,
}

impl TgRefiner {
    pub fn try_new(
        predicate: &SpatialPredicate,
        options: SpatialJoinOptions,
        num_build_geoms: usize,
        build_stats: GeoStatistics,
    ) -> Result<Self> {
        let evaluator: Box<dyn TgPredicateEvaluator> = create_evaluator(predicate)?;
        let index_type = match options.tg.index_type {
            TgIndexType::Natural => tg::IndexType::Natural,
            TgIndexType::YStripes => tg::IndexType::YStripes,
        };

        let exec_mode = OnceLock::new();
        let exec_mode_selector = match options.execution_mode {
            ExecutionMode::Speculative(n) => {
                let selector = TgOptimalModeSelector {
                    predicate: predicate.clone(),
                };
                if let Some(mode) = selector.select_without_probe_stats(&build_stats) {
                    exec_mode.set(mode).unwrap();
                    None
                } else {
                    Some(ExecModeSelector::new(build_stats, n, Arc::new(selector)))
                }
            }
            _ => {
                exec_mode.set(options.execution_mode).unwrap();
                None
            }
        };

        let prepared_geom_array_size =
            if matches!(exec_mode.get(), Some(ExecutionMode::PrepareBuild) | None) {
                num_build_geoms
            } else {
                0
            };

        let prepared_geoms = InitOnceArray::new(prepared_geom_array_size);
        let mem_usage = prepared_geoms.allocated_size();
        Ok(Self {
            evaluator,
            prepared_geoms,
            index_type,
            mem_usage: AtomicUsize::new(mem_usage),
            exec_mode,
            exec_mode_selector,
        })
    }

    fn refine_prepare_build(
        &self,
        probe: &wkb::reader::Wkb<'_>,
        index_query_results: &[IndexQueryResult],
    ) -> Result<Vec<(i32, i32)>> {
        let mut build_batch_positions = Vec::with_capacity(index_query_results.len());
        let probe_geom = tg::Geom::parse_wkb(probe.buf(), self.index_type)?;
        for index_result in index_query_results {
            let (build_geom, is_newly_created) =
                self.prepared_geoms
                    .get_or_create(index_result.geom_idx, || {
                        let geom = tg::Geom::parse_wkb(index_result.wkb.buf(), self.index_type)?;
                        Ok(Some(geom))
                    })?;
            let Some(build_geom) = build_geom else {
                continue;
            };
            if is_newly_created {
                let prep_geom_size = build_geom.memsize();
                self.mem_usage.fetch_add(prep_geom_size, Ordering::Relaxed);
            }
            if self
                .evaluator
                .evaluate(build_geom, &probe_geom, index_result.distance)?
            {
                build_batch_positions.push(index_result.position);
            }
        }
        Ok(build_batch_positions)
    }

    fn refine_not_prepare_build(
        &self,
        probe: &wkb::reader::Wkb<'_>,
        index_query_results: &[IndexQueryResult],
        probe_index_type: tg::IndexType,
    ) -> Result<Vec<(i32, i32)>> {
        let mut build_batch_positions = Vec::with_capacity(index_query_results.len());
        let probe_geom = tg::Geom::parse_wkb(probe.buf(), probe_index_type)?;
        for index_result in index_query_results {
            let build_geom = tg::Geom::parse_wkb(index_result.wkb.buf(), tg::IndexType::Unindexed)?;
            if self
                .evaluator
                .evaluate(&build_geom, &probe_geom, index_result.distance)?
            {
                build_batch_positions.push(index_result.position);
            }
        }
        Ok(build_batch_positions)
    }
}

impl IndexQueryResultRefiner for TgRefiner {
    fn refine(
        &self,
        probe: &Wkb<'_>,
        index_query_results: &[IndexQueryResult],
    ) -> Result<Vec<(i32, i32)>> {
        let exec_mode = self.actual_execution_mode();
        match exec_mode {
            ExecutionMode::PrepareNone => {
                self.refine_not_prepare_build(probe, index_query_results, tg::IndexType::Unindexed)
            }
            ExecutionMode::PrepareBuild => self.refine_prepare_build(probe, index_query_results),
            ExecutionMode::PrepareProbe => {
                self.refine_not_prepare_build(probe, index_query_results, self.index_type)
            }
            ExecutionMode::Speculative(_) => {
                sedona_internal_err!(
                    "Speculative execution mode should be translated to other execution modes"
                )
            }
        }
    }

    fn mem_usage(&self) -> usize {
        self.mem_usage.load(Ordering::Relaxed)
    }

    fn actual_execution_mode(&self) -> ExecutionMode {
        get_or_update_execution_mode(
            &self.exec_mode,
            &self.exec_mode_selector,
            ExecutionMode::PrepareProbe,
        )
    }

    fn need_more_probe_stats(&self) -> bool {
        self.exec_mode.get().is_none()
    }

    fn merge_probe_stats(&self, stats: GeoStatistics) {
        if let Some(selector) = self.exec_mode_selector.as_ref() {
            selector.merge_probe_stats(stats);
        }
    }
}

trait TgPredicateEvaluator: Send + Sync {
    fn evaluate(&self, build: &tg::Geom, probe: &tg::Geom, distance: Option<f64>) -> Result<bool>;
}

struct TgPredicateEvaluatorImpl<Op: BinaryPredicate + Send + Sync> {
    _marker: PhantomData<Op>,
}

impl<Op: BinaryPredicate + Send + Sync> TgPredicateEvaluatorImpl<Op> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<Op: BinaryPredicate + Send + Sync> TgPredicateEvaluator for TgPredicateEvaluatorImpl<Op> {
    fn evaluate(&self, build: &tg::Geom, probe: &tg::Geom, _distance: Option<f64>) -> Result<bool> {
        Ok(Op::evaluate(build, probe))
    }
}

fn create_evaluator(predicate: &SpatialPredicate) -> Result<Box<dyn TgPredicateEvaluator>> {
    let evaluator: Box<dyn TgPredicateEvaluator> = match predicate {
        SpatialPredicate::Distance(_) => {
            return Err(DataFusionError::Internal(
                "Distance predicate is not supported for TG".to_string(),
            ))
        }
        SpatialPredicate::KNearestNeighbors(_) => {
            return Err(DataFusionError::Internal(
                "KNN predicate is not supported for TG".to_string(),
            ))
        }
        SpatialPredicate::Relation(predicate) => match predicate.relation_type {
            SpatialRelationType::Intersects => {
                Box::new(TgPredicateEvaluatorImpl::<tg::Intersects>::new())
            }
            SpatialRelationType::Contains => {
                Box::new(TgPredicateEvaluatorImpl::<tg::Contains>::new())
            }
            SpatialRelationType::Within => Box::new(TgPredicateEvaluatorImpl::<tg::Within>::new()),
            SpatialRelationType::Covers => Box::new(TgPredicateEvaluatorImpl::<tg::Covers>::new()),
            SpatialRelationType::CoveredBy => {
                Box::new(TgPredicateEvaluatorImpl::<tg::CoveredBy>::new())
            }
            SpatialRelationType::Touches => {
                Box::new(TgPredicateEvaluatorImpl::<tg::Touches>::new())
            }
            SpatialRelationType::Equals => Box::new(TgPredicateEvaluatorImpl::<tg::Equals>::new()),
            _ => {
                return Err(DataFusionError::Internal(
                    "Unsupported spatial relation type for TG".to_string(),
                ))
            }
        },
    };
    Ok(evaluator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial_predicate::{DistancePredicate, RelationPredicate, SpatialRelationType};
    use datafusion_common::JoinSide;
    use datafusion_common::ScalarValue;
    use datafusion_physical_expr::expressions::{Column, Literal};
    use datafusion_physical_expr::PhysicalExpr;
    use sedona_common::DEFAULT_SPECULATIVE_THRESHOLD;
    use std::sync::Arc;

    /// Helper function to create a dummy PhysicalExpr for testing
    fn create_dummy_column(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new(name, index))
    }

    /// Helper function to create GeoStatistics with specific mean points per geometry
    fn create_geo_stats(total_geometries: i64, total_points: i64) -> GeoStatistics {
        GeoStatistics::empty()
            .with_total_geometries(total_geometries)
            .with_total_points(total_points)
    }

    /// Helper function to create a relation predicate
    fn create_relation_predicate(relation_type: SpatialRelationType) -> SpatialPredicate {
        SpatialPredicate::Relation(RelationPredicate::new(
            create_dummy_column("left_geom", 0),
            create_dummy_column("right_geom", 1),
            relation_type,
        ))
    }

    /// Helper function to create a distance predicate
    fn create_distance_predicate() -> SpatialPredicate {
        SpatialPredicate::Distance(DistancePredicate::new(
            create_dummy_column("left_geom", 0),
            create_dummy_column("right_geom", 1),
            Arc::new(Literal::new(ScalarValue::Float64(Some(100.0)))),
            JoinSide::None,
        ))
    }

    #[test]
    fn test_tg_refiner_distance_predicate_unsupported() {
        let predicate = create_distance_predicate();
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        // Distance predicate should fail for TG
        let result = TgRefiner::try_new(&predicate, options, 100, build_stats);
        assert!(result.is_err());
    }

    #[test]
    fn test_tg_refiner_intersects_predicate_needs_stats() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = TgRefiner::try_new(&predicate, options, 100, build_stats).unwrap();

        // Intersects predicate should need probe stats to determine execution mode
        assert!(refiner.need_more_probe_stats());

        // Before probe stats, should return default mode
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareProbe);
    }

    #[test]
    fn test_tg_refiner_contains_predicate_immediate_selection() {
        let predicate = create_relation_predicate(SpatialRelationType::Contains);
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = TgRefiner::try_new(&predicate, options.clone(), 100, build_stats).unwrap();

        // Contains predicate should immediately select PrepareNone, since build side is not complex enough
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareNone);
        assert!(!refiner.need_more_probe_stats());

        let build_stats = create_geo_stats(100, 8000);
        let refiner = TgRefiner::try_new(&predicate, options, 100, build_stats).unwrap();

        // Contains predicate should immediately select PrepareBuild, since build side is complex enough
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareBuild);
        assert!(!refiner.need_more_probe_stats());
    }

    #[test]
    fn test_tg_refiner_covers_predicate_immediate_selection() {
        let predicate = create_relation_predicate(SpatialRelationType::Covers);
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = TgRefiner::try_new(&predicate, options.clone(), 100, build_stats).unwrap();

        // Covers predicate should immediately select PrepareNone, since build side is not complex enough
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareNone);
        assert!(!refiner.need_more_probe_stats());

        let build_stats = create_geo_stats(100, 8000);
        let refiner = TgRefiner::try_new(&predicate, options, 100, build_stats).unwrap();

        // Covers predicate should immediately select PrepareBuild, since build side is complex enough
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareBuild);
        assert!(!refiner.need_more_probe_stats());
    }

    #[test]
    fn test_tg_refiner_within_predicate_immediate_selection() {
        let predicate = create_relation_predicate(SpatialRelationType::Within);
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = TgRefiner::try_new(&predicate, options, 100, build_stats).unwrap();

        // Within predicate should immediately select PrepareProbe
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareProbe);
        assert!(!refiner.need_more_probe_stats());
    }

    #[test]
    fn test_tg_refiner_covered_by_predicate_immediate_selection() {
        let predicate = create_relation_predicate(SpatialRelationType::CoveredBy);
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = TgRefiner::try_new(&predicate, options, 100, build_stats).unwrap();

        // CoveredBy predicate should immediately select PrepareProbe
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareProbe);
        assert!(!refiner.need_more_probe_stats());
    }

    #[test]
    fn test_tg_refiner_intersects_with_simple_geometries() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 1000); // 10 points per geometry (simple)
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = TgRefiner::try_new(&predicate, options, 100, build_stats).unwrap();

        // Merge probe stats with simple geometries (20 points per geometry)
        let probe_stats = create_geo_stats(50, 1000); // 20 points per geometry
        refiner.merge_probe_stats(probe_stats);

        // With simple geometries (max 20 points per geometry < 32 threshold), should select PrepareProbe
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareProbe);
    }

    #[test]
    fn test_tg_refiner_intersects_with_complex_geometries_prefer_build() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 8000); // 80 points per geometry (complex)
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(50), // Use smaller threshold for testing
            ..Default::default()
        };

        let refiner = TgRefiner::try_new(&predicate, options, 100, build_stats).unwrap();

        // Merge probe stats with simpler geometries (30 points per geometry)
        let probe_stats = create_geo_stats(50, 1500); // 30 points per geometry
        refiner.merge_probe_stats(probe_stats);

        // Build side has more complex geometries (80 > 30), should select PrepareBuild
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareBuild);
    }

    #[test]
    fn test_tg_refiner_intersects_with_complex_geometries_prefer_probe() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 3000); // 30 points per geometry
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = TgRefiner::try_new(&predicate, options, 100, build_stats).unwrap();

        // Merge probe stats with more complex geometries (80 points per geometry)
        let probe_stats = create_geo_stats(50, 4000); // 80 points per geometry
        refiner.merge_probe_stats(probe_stats);

        // Probe side has more complex geometries (80 > 30), should select PrepareProbe
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareProbe);
    }

    #[test]
    fn test_tg_refiner_non_speculative_mode() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };

        let refiner = TgRefiner::try_new(&predicate, options, 100, build_stats).unwrap();

        // Non-speculative mode should immediately select the specified mode
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareBuild);
        assert!(!refiner.need_more_probe_stats());
    }

    #[test]
    fn test_tg_refiner_select_optimal_mode_function() {
        // Test the select_optimal_mode function directly
        let build_stats = create_geo_stats(100, 1000); // 10 points per geometry
        let probe_stats = create_geo_stats(50, 1000); // 20 points per geometry
        let selector = TgOptimalModeSelector {
            predicate: create_relation_predicate(SpatialRelationType::Intersects),
        };

        // Simple geometries (max 20 < 32 threshold) should select PrepareProbe
        let result = selector.select(&build_stats, &probe_stats);
        assert_eq!(result, ExecutionMode::PrepareProbe);

        // Complex geometries should select based on which side is more complex
        let complex_build_stats = create_geo_stats(100, 8000); // 80 points per geometry
        let simple_probe_stats = create_geo_stats(50, 1500); // 30 points per geometry

        let result = selector.select(&complex_build_stats, &simple_probe_stats);
        assert_eq!(result, ExecutionMode::PrepareBuild);

        let simple_build_stats = create_geo_stats(100, 3000); // 30 points per geometry
        let complex_probe_stats = create_geo_stats(50, 4000); // 80 points per geometry

        let result = selector.select(&simple_build_stats, &complex_probe_stats);
        assert_eq!(result, ExecutionMode::PrepareProbe);
    }

    #[test]
    fn test_tg_refiner_prepared_geom_array_sizing() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 1000);
        let num_build_geoms = 500;

        // Test PrepareBuild mode - should allocate array for all build geometries
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let refiner =
            TgRefiner::try_new(&predicate, options, num_build_geoms, build_stats.clone()).unwrap();
        assert!(refiner.prepared_geoms.len() == num_build_geoms);

        // Test PrepareProbe mode - should not allocate array for build geometries
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareProbe,
            ..Default::default()
        };
        let refiner =
            TgRefiner::try_new(&predicate, options, num_build_geoms, build_stats.clone()).unwrap();
        assert!(refiner.prepared_geoms.is_empty());

        // Test PrepareNone mode - should not allocate array for build geometries
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareNone,
            ..Default::default()
        };
        let refiner =
            TgRefiner::try_new(&predicate, options, num_build_geoms, build_stats).unwrap();
        assert!(refiner.prepared_geoms.is_empty());
    }
}
