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
    utils::init_once_array::InitOnceArray,
    IndexQueryResult, IndexQueryResultRefiner, SpatialPredicate,
};
use wkb::reader::Wkb;

#[derive(Debug)]
pub struct GeographyRefiner {
    op_type: OpType,
    prepared_geoms: InitOnceArray<Option<Geography<'static>>>,
}

impl GeographyRefiner {
    pub fn new(predicate: SpatialPredicate, num_build_geoms: usize) -> Result<Self> {
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
                        "GeographyRefiner created with unsupported relation type {relation_type}"
                    )
                }
            },
            SpatialPredicate::Distance(_) => OpType::DWithin,
            _ => {
                return sedona_internal_err!(
                    "GeographyRefiner created with unsupported predicate {predicate}"
                )
            }
        };

        let prepared_geoms = InitOnceArray::new(num_build_geoms);
        Ok(Self {
            op_type,
            prepared_geoms,
        })
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
        let mut probe_geog = factory
            .from_wkb(probe.buf())
            .map_err(|e| exec_datafusion_err!("{e}"))?;

        let mut build_geog = Geography::new();

        // Crude heuristic used by the S2Loop (build an index after 20 unindexed
        // contains queries even for small looops).
        if probe.buf().len() > (32 * 2 * size_of::<f64>()) || index_query_results.len() > 20 {
            probe_geog
                .prepare()
                .map_err(|e| exec_datafusion_err!("{e}"))?;
        }

        // We're in prepared build mode
        if !self.prepared_geoms.is_empty() {
            for result in index_query_results {
                let (prepared_build_geom, _) =
                    self.prepared_geoms.get_or_create(result.geom_idx, || {
                        // Basically, prepare anything except points on the build side
                        if result.wkb.buf().len() > 32 {
                            let mut geog = factory
                                .from_wkb(result.wkb.buf())
                                .map_err(|e| exec_datafusion_err!("{e}"))?;
                            geog.prepare().map_err(|e| exec_datafusion_err!("{e}"))?;
                            Ok(Some(unsafe {
                                // Safety: the evaluated batches keep the required WKB alive
                                std::mem::transmute::<Geography<'_>, Geography<'static>>(geog)
                            }))
                        } else {
                            Ok(None)
                        }
                    })?;

                let build_geog_ref = if let Some(prepared_geog) = prepared_build_geom {
                    prepared_geog
                } else {
                    factory
                        .init_from_wkb(result.wkb.buf(), &mut build_geog)
                        .map_err(|e| exec_datafusion_err!("{e}"))?;
                    &build_geog
                };

                let eval = if matches!(self.op_type, OpType::DWithin) {
                    op.eval_binary_distance_predicate(
                        build_geog_ref,
                        &probe_geog,
                        result.distance.unwrap_or(f64::INFINITY),
                    )
                } else {
                    op.eval_binary_predicate(build_geog_ref, &probe_geog)
                };

                if eval.map_err(|e| exec_datafusion_err!("{e}"))? {
                    results.push(result.position);
                }
            }
        } else {
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
        num_build_geoms: usize,
        _build_stats: GeoStatistics,
    ) -> Result<Arc<dyn IndexQueryResultRefiner>> {
        Ok(Arc::new(GeographyRefiner::new(
            predicate.clone(),
            num_build_geoms,
        )?))
    }
}
