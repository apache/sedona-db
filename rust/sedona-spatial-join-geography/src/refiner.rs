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

//! Geography-specific refiner for spatial join predicate evaluation using s2geography.
//!
//! This module provides a refiner that evaluates spatial predicates on the sphere
//! using s2geography, rather than the default Cartesian predicates.

use std::sync::Arc;

use datafusion_common::{exec_datafusion_err, Result};
use sedona_common::{sedona_internal_err, ExecutionMode, SpatialJoinOptions};
use sedona_expr::statistics::GeoStatistics;
use sedona_s2geography::{
    geography::{Geography, GeographyFactory},
    operator::{Op, OpType},
};
use sedona_spatial_join::{
    refine::IndexQueryResultRefinerFactory,
    spatial_predicate::{RelationPredicate, SpatialRelationType},
    IndexQueryResult, IndexQueryResultRefiner, SpatialPredicate,
};
use wkb::reader::Wkb;

#[derive(Debug)]
pub struct GeographyRefiner {
    op_type: OpType,
}

impl GeographyRefiner {
    pub fn new(predicate: SpatialPredicate) -> Result<Self> {
        let op_type = match predicate {
            SpatialPredicate::Relation(RelationPredicate {
                left: _,
                right: _,
                relation_type,
            }) => match relation_type {
                SpatialRelationType::Intersects => OpType::Intersects,
                SpatialRelationType::Contains => OpType::Contains,
                SpatialRelationType::Within => OpType::Within,
                SpatialRelationType::Equals => OpType::Equals,
                _ => {
                    return sedona_internal_err!(
                        "GeographyRefiner crated with unsupported relation type {relation_type}"
                    )
                }
            },
            SpatialPredicate::Distance(_) => OpType::DWithin,
            _ => {
                return sedona_internal_err!(
                    "GeographyRefiner crated with unsupported predicate {predicate}"
                )
            }
        };

        Ok(Self { op_type })
    }
}

impl IndexQueryResultRefiner for GeographyRefiner {
    fn refine(
        &self,
        probe: &Wkb<'_>,
        index_query_results: &[IndexQueryResult],
    ) -> Result<Vec<(i32, i32)>> {
        let mut results = Vec::with_capacity(index_query_results.len());
        let mut op = Op::new(self.op_type);

        // TODO: thread local factories?
        let mut factory = GeographyFactory::new();
        let probe_geog = factory
            .from_wkb(probe.buf())
            .map_err(|e| exec_datafusion_err!("{e}"))?;

        // TODO: exploit preparedness in the same way as the tg geometries
        let mut build_geog = Geography::new();

        for result in index_query_results {
            factory
                .init_from_wkb(result.wkb.buf(), &mut build_geog)
                .map_err(|e| exec_datafusion_err!("{e}"))?;

            // TODO: evaluation order left vs right?
            let eval = if matches!(self.op_type, OpType::DWithin) {
                op.eval_binary_distance_predicate(
                    &build_geog,
                    &probe_geog,
                    result.distance.unwrap_or(f64::INFINITY),
                )
            } else {
                op.eval_binary_predicate(&build_geog, &probe_geog)
            };

            if eval.map_err(|e| exec_datafusion_err!("{e}"))? {
                results.push(result.position);
            }
        }

        Ok(results)
    }

    fn estimate_max_memory_usage(&self, _build_stats: &GeoStatistics) -> usize {
        // TODO: Calculate based on whether we cache prepared geometries
        0
    }

    fn mem_usage(&self) -> usize {
        // TODO: Track actual memory usage
        0
    }

    fn actual_execution_mode(&self) -> ExecutionMode {
        // TODO: preparedness
        ExecutionMode::PrepareNone
    }

    fn need_more_probe_stats(&self) -> bool {
        false
    }

    fn merge_probe_stats(&self, _stats: GeoStatistics) {
        // TODO: Use stats for adaptive execution mode selection
    }
}

#[derive(Debug)]
pub struct GeographyRefinerFactory;

impl IndexQueryResultRefinerFactory for GeographyRefinerFactory {
    fn create_refiner(
        &self,
        predicate: &SpatialPredicate,
        _options: SpatialJoinOptions,
        _num_build_geoms: usize,
        _build_stats: GeoStatistics,
    ) -> Result<Arc<dyn IndexQueryResultRefiner>> {
        Ok(Arc::new(GeographyRefiner::new(predicate.clone())?))
    }
}
