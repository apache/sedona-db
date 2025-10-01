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
use std::sync::{Arc, OnceLock};

use datafusion_common::Result;
use geo::{Contains, Relate, Within};
use sedona_common::{sedona_internal_err, ExecutionMode, SpatialJoinOptions};
use sedona_expr::statistics::GeoStatistics;
use sedona_geo::to_geo::item_to_geometry;
use sedona_geo_generic_alg::{line_measures::DistanceExt, Intersects};
use wkb::reader::Wkb;

use crate::{
    index::IndexQueryResult,
    refine::{
        exec_mode_selector::{get_or_update_execution_mode, ExecModeSelector, SelectOptimalMode},
        IndexQueryResultRefiner,
    },
    spatial_predicate::{SpatialPredicate, SpatialRelationType},
};

/// Geo-specific optimal mode selector that chooses the best execution mode
/// based on probe-side geometry complexity.
struct GeoOptimalModeSelector {
    predicate: SpatialPredicate,
}

impl SelectOptimalMode for GeoOptimalModeSelector {
    fn select(&self, _build_stats: &GeoStatistics, probe_stats: &GeoStatistics) -> ExecutionMode {
        match self.predicate {
            SpatialPredicate::Distance(_) => ExecutionMode::PrepareNone,
            SpatialPredicate::KNearestNeighbors(_) => ExecutionMode::PrepareNone,
            SpatialPredicate::Relation(_) => {
                // We only support PrepareProbe and PrepareNone. Only the stats from the probe side is used to
                // select the execution mode.
                let probe_mean_points_per_geometry =
                    probe_stats.mean_points_per_geometry().unwrap_or(0.0);
                if probe_mean_points_per_geometry <= 50.0 {
                    ExecutionMode::PrepareNone
                } else {
                    ExecutionMode::PrepareProbe
                }
            }
        }
    }

    fn select_without_probe_stats(&self, _build_stats: &GeoStatistics) -> Option<ExecutionMode> {
        match self.predicate {
            SpatialPredicate::Distance(_) => Some(ExecutionMode::PrepareNone),
            SpatialPredicate::KNearestNeighbors(_) => Some(ExecutionMode::PrepareNone),
            _ => None,
        }
    }
}

/// A refiner that uses the geo library to evaluate spatial predicates.
pub(crate) struct GeoRefiner {
    evaluator: Box<dyn GeoPredicateEvaluator>,
    exec_mode: OnceLock<ExecutionMode>,
    exec_mode_selector: Option<ExecModeSelector>,
}

impl GeoRefiner {
    pub fn new(
        predicate: &SpatialPredicate,
        options: SpatialJoinOptions,
        build_stats: GeoStatistics,
    ) -> Self {
        let evaluator = create_evaluator(predicate);

        let exec_mode = OnceLock::new();
        let exec_mode_selector = match options.execution_mode {
            ExecutionMode::Speculative(n) => {
                let selector = GeoOptimalModeSelector {
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

        Self {
            evaluator,
            exec_mode,
            exec_mode_selector,
        }
    }

    fn refine_prepare_none(
        &self,
        probe: &Wkb<'_>,
        index_query_results: &[IndexQueryResult],
    ) -> Result<Vec<(i32, i32)>> {
        let mut build_batch_positions = Vec::with_capacity(index_query_results.len());
        for index_result in index_query_results {
            if self
                .evaluator
                .evaluate(index_result.wkb, probe, index_result.distance)?
            {
                build_batch_positions.push(index_result.position);
            }
        }
        Ok(build_batch_positions)
    }

    fn refine_prepare_probe(
        &self,
        probe: &Wkb<'_>,
        index_query_results: &[IndexQueryResult],
    ) -> Result<Vec<(i32, i32)>> {
        let mut build_batch_positions = Vec::with_capacity(index_query_results.len());
        let probe_geom = match item_to_geometry(probe) {
            Ok(geom) => geom,
            Err(_) => return Ok(Vec::new()),
        };
        let probe_geom = geo::PreparedGeometry::from(probe_geom);

        for index_result in index_query_results {
            if self.evaluator.evaluate_prepare_probe(
                index_result.wkb,
                &probe_geom,
                index_result.distance,
            )? {
                build_batch_positions.push(index_result.position);
            }
        }
        Ok(build_batch_positions)
    }
}

impl IndexQueryResultRefiner for GeoRefiner {
    fn refine(
        &self,
        probe: &Wkb<'_>,
        index_query_results: &[IndexQueryResult],
    ) -> Result<Vec<(i32, i32)>> {
        let exec_mode = self.actual_execution_mode();
        match exec_mode {
            ExecutionMode::PrepareNone => self.refine_prepare_none(probe, index_query_results),
            ExecutionMode::PrepareBuild => {
                // Prepare build mode is not implemented for geo, because geo's prepared geometry
                // is not thread-safe and runs slowly, we suggest using other libraries that have
                // faster prepared geometry, such as tg or GEOS.
                self.refine_prepare_none(probe, index_query_results)
            }
            ExecutionMode::PrepareProbe => self.refine_prepare_probe(probe, index_query_results),
            ExecutionMode::Speculative(_) => {
                sedona_internal_err!(
                    "Speculative execution mode should be translated to other execution modes"
                )
            }
        }
    }

    fn mem_usage(&self) -> usize {
        0
    }

    fn actual_execution_mode(&self) -> ExecutionMode {
        get_or_update_execution_mode(
            &self.exec_mode,
            &self.exec_mode_selector,
            ExecutionMode::PrepareNone,
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

trait GeoPredicateEvaluator: Send + Sync {
    fn evaluate(&self, build: &Wkb, probe: &Wkb, distance: Option<f64>) -> Result<bool>;

    fn evaluate_prepare_probe(
        &self,
        build: &Wkb,
        probe: &geo::PreparedGeometry<'static, geo_types::Geometry>,
        distance: Option<f64>,
    ) -> Result<bool>;
}

fn create_evaluator(predicate: &SpatialPredicate) -> Box<dyn GeoPredicateEvaluator> {
    match predicate {
        SpatialPredicate::Distance(_) => Box::new(GeoDistance),
        SpatialPredicate::Relation(predicate) => match predicate.relation_type {
            SpatialRelationType::Intersects => Box::new(GeoIntersects),
            SpatialRelationType::Contains => Box::new(GeoContains),
            SpatialRelationType::Within => Box::new(GeoWithin),
            SpatialRelationType::Covers => Box::new(GeoCovers),
            SpatialRelationType::CoveredBy => Box::new(GeoCoveredBy),
            SpatialRelationType::Touches => Box::new(GeoTouches),
            SpatialRelationType::Crosses => Box::new(GeoCrosses),
            SpatialRelationType::Overlaps => Box::new(GeoOverlaps),
            SpatialRelationType::Equals => Box::new(GeoEquals),
        },
        SpatialPredicate::KNearestNeighbors(_) => Box::new(GeoDistance), // Use distance for KNN
    }
}

struct GeoIntersects;

impl GeoPredicateEvaluator for GeoIntersects {
    fn evaluate(&self, build: &Wkb, probe: &Wkb, _distance: Option<f64>) -> Result<bool> {
        Ok(build.intersects(probe))
    }

    fn evaluate_prepare_probe(
        &self,
        build: &Wkb,
        probe: &geo::PreparedGeometry<'static, geo_types::Geometry>,
        _distance: Option<f64>,
    ) -> Result<bool> {
        let build_geom = match item_to_geometry(build) {
            Ok(geom) => geom,
            Err(_) => return Ok(false),
        };
        Ok(probe.relate(&build_geom).is_intersects())
    }
}

struct GeoContains;

impl GeoPredicateEvaluator for GeoContains {
    fn evaluate(&self, build: &Wkb, probe: &Wkb, _distance: Option<f64>) -> Result<bool> {
        let build_geom = match item_to_geometry(build) {
            Ok(geom) => geom,
            Err(_) => return Ok(false),
        };
        let probe_geom = match item_to_geometry(probe) {
            Ok(geom) => geom,
            Err(_) => return Ok(false),
        };
        Ok(build_geom.contains(&probe_geom))
    }

    fn evaluate_prepare_probe(
        &self,
        build: &Wkb,
        probe: &geo::PreparedGeometry<'static, geo_types::Geometry>,
        _distance: Option<f64>,
    ) -> Result<bool> {
        let build_geom = match item_to_geometry(build) {
            Ok(geom) => geom,
            Err(_) => return Ok(false),
        };
        Ok(probe.relate(&build_geom).is_within())
    }
}

struct GeoWithin;

impl GeoPredicateEvaluator for GeoWithin {
    fn evaluate(&self, build: &Wkb, probe: &Wkb, _distance: Option<f64>) -> Result<bool> {
        let build_geom = match item_to_geometry(build) {
            Ok(geom) => geom,
            Err(_) => return Ok(false),
        };
        let probe_geom = match item_to_geometry(probe) {
            Ok(geom) => geom,
            Err(_) => return Ok(false),
        };
        Ok(build_geom.is_within(&probe_geom))
    }

    fn evaluate_prepare_probe(
        &self,
        build: &Wkb,
        probe: &geo::PreparedGeometry<'static, geo_types::Geometry>,
        _distance: Option<f64>,
    ) -> Result<bool> {
        let build_geom = match item_to_geometry(build) {
            Ok(geom) => geom,
            Err(_) => return Ok(false),
        };
        Ok(probe.relate(&build_geom).is_contains())
    }
}

struct GeoDistance;

impl GeoPredicateEvaluator for GeoDistance {
    fn evaluate(&self, build: &Wkb, probe: &Wkb, distance: Option<f64>) -> Result<bool> {
        let Some(distance) = distance else {
            return Ok(false);
        };
        let dist = build.distance_ext(probe);
        Ok(dist <= distance)
    }

    fn evaluate_prepare_probe(
        &self,
        build: &Wkb,
        probe: &geo::PreparedGeometry<'static, geo_types::Geometry>,
        distance: Option<f64>,
    ) -> Result<bool> {
        let Some(distance) = distance else {
            return Ok(false);
        };
        let dist = build.distance_ext(probe.geometry());
        Ok(dist <= distance)
    }
}

/// Macro to generate relation evaluators that use the relate() method
macro_rules! impl_relate_evaluator {
    ($struct_name:ident, $geo_method:ident $(,)?) => {
        #[derive(Debug)]
        struct $struct_name;

        impl GeoPredicateEvaluator for $struct_name {
            fn evaluate(&self, build: &Wkb, probe: &Wkb, _distance: Option<f64>) -> Result<bool> {
                let build_geom = match item_to_geometry(build) {
                    Ok(geom) => geom,
                    Err(_) => return Ok(false),
                };
                let probe_geom = match item_to_geometry(probe) {
                    Ok(geom) => geom,
                    Err(_) => return Ok(false),
                };
                Ok(build_geom.relate(&probe_geom).$geo_method())
            }

            fn evaluate_prepare_probe(
                &self,
                build: &Wkb,
                probe: &geo::PreparedGeometry<'static, geo_types::Geometry>,
                _distance: Option<f64>,
            ) -> Result<bool> {
                let build_geom = match item_to_geometry(build) {
                    Ok(geom) => geom,
                    Err(_) => return Ok(false),
                };
                Ok(probe.relate(&build_geom).$geo_method())
            }
        }
    };
}

// Generate relate-based evaluators using the macro
impl_relate_evaluator!(GeoTouches, is_touches);
impl_relate_evaluator!(GeoCrosses, is_crosses);
impl_relate_evaluator!(GeoOverlaps, is_overlaps);
impl_relate_evaluator!(GeoCovers, is_covers);
impl_relate_evaluator!(GeoCoveredBy, is_coveredby);
impl_relate_evaluator!(GeoEquals, is_equal_topo);

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
    fn test_geo_refiner_distance_predicate_immediate_selection() {
        let predicate = create_distance_predicate();
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = GeoRefiner::new(&predicate, options, build_stats);

        // Distance predicate should immediately select PrepareNone
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareNone);
        assert!(!refiner.need_more_probe_stats());
    }

    #[test]
    fn test_geo_refiner_intersects_predicate_needs_stats() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = GeoRefiner::new(&predicate, options, build_stats);

        // Intersects predicate should need probe stats to determine execution mode
        assert!(refiner.need_more_probe_stats());

        // Before probe stats, should return default mode
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareNone);
    }

    #[test]
    fn test_geo_refiner_intersects_with_simple_geometries() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 1000); // 10 points per geometry (simple)
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
            ..Default::default()
        };

        let refiner = GeoRefiner::new(&predicate, options, build_stats);

        // Merge probe stats with simple geometries (20 points per geometry)
        let probe_stats = create_geo_stats(50, 1000); // 20 points per geometry
        refiner.merge_probe_stats(probe_stats);

        // With simple geometries (max 20 points per geometry < 50 threshold), should select PrepareNone
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareNone);
    }

    #[test]
    fn test_geo_refiner_intersects_ignore_build_complexity() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 8000); // 80 points per geometry (complex)
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(50), // Use smaller threshold for testing
            ..Default::default()
        };

        let refiner = GeoRefiner::new(&predicate, options, build_stats);

        // Merge probe stats with simpler geometries (30 points per geometry)
        let probe_stats = create_geo_stats(50, 1500); // 30 points per geometry
        refiner.merge_probe_stats(probe_stats);

        // Build side has more complex geometries (80 > 30), but we still select PrepareNone
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareNone);
    }

    #[test]
    fn test_geo_refiner_intersects_with_complex_geometries_prefer_probe() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 3000); // 30 points per geometry
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::Speculative(50), // Use smaller threshold for testing
            ..Default::default()
        };

        let refiner = GeoRefiner::new(&predicate, options, build_stats);

        // Merge probe stats with more complex geometries (80 points per geometry)
        let probe_stats = create_geo_stats(50, 4000); // 80 points per geometry
        refiner.merge_probe_stats(probe_stats);

        // Probe side has more complex geometries (80 > 30), should select PrepareProbe
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareProbe);
    }

    #[test]
    fn test_geo_refiner_non_speculative_mode() {
        let predicate = create_relation_predicate(SpatialRelationType::Intersects);
        let build_stats = create_geo_stats(100, 1000);
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };

        let refiner = GeoRefiner::new(&predicate, options, build_stats);

        // Non-speculative mode should immediately select the specified mode
        assert_eq!(refiner.actual_execution_mode(), ExecutionMode::PrepareBuild);
        assert!(!refiner.need_more_probe_stats());
    }

    #[test]
    fn test_geo_refiner_all_relation_types() {
        let relation_types = vec![
            SpatialRelationType::Intersects,
            SpatialRelationType::Contains,
            SpatialRelationType::Within,
            SpatialRelationType::Covers,
            SpatialRelationType::CoveredBy,
            SpatialRelationType::Touches,
            SpatialRelationType::Crosses,
            SpatialRelationType::Overlaps,
            SpatialRelationType::Equals,
        ];

        for relation_type in relation_types {
            let predicate = create_relation_predicate(relation_type);
            let build_stats = create_geo_stats(100, 1000);
            let options = SpatialJoinOptions {
                execution_mode: ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD),
                ..Default::default()
            };

            let refiner = GeoRefiner::new(&predicate, options, build_stats);

            // All relation types should need probe stats for GeoRefiner
            // Because GeoRefiner doesn't have predicate-specific optimizations
            assert!(
                refiner.need_more_probe_stats(),
                "Failed for relation type: {relation_type:?}"
            );
        }
    }

    #[test]
    fn test_geo_refiner_select_optimal_mode_function() {
        // Test the select_optimal_mode function directly
        let build_stats = create_geo_stats(100, 1000); // 10 points per geometry
        let probe_stats = create_geo_stats(50, 1000); // 20 points per geometry
        let selector = GeoOptimalModeSelector {
            predicate: create_relation_predicate(SpatialRelationType::Intersects),
        };

        // Simple geometries (max 20 < 50 threshold) should select PrepareNone
        let result = selector.select(&build_stats, &probe_stats);
        assert_eq!(result, ExecutionMode::PrepareNone);

        // Complex geometries should select based on which side is more complex
        let complex_build_stats = create_geo_stats(100, 8000); // 80 points per geometry
        let simple_probe_stats = create_geo_stats(50, 1500); // 30 points per geometry

        let result = selector.select(&complex_build_stats, &simple_probe_stats);
        assert_eq!(result, ExecutionMode::PrepareNone);

        let simple_build_stats = create_geo_stats(100, 3000); // 30 points per geometry
        let complex_probe_stats = create_geo_stats(50, 4000); // 80 points per geometry

        let result = selector.select(&simple_build_stats, &complex_probe_stats);
        assert_eq!(result, ExecutionMode::PrepareProbe);
    }
}
