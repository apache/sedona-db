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

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

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
pub(crate) struct GeographyRefiner {
    op_type: OpType,
    prepared_geoms: InitOnceArray<Option<Geography<'static>>>,
    mem_used: AtomicUsize,
}

impl GeographyRefiner {
    pub fn new(
        predicate: SpatialPredicate,
        options: &SpatialJoinOptions,
        num_build_geoms: usize,
    ) -> Result<Self> {
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

        // Allow join options to turn off preparedness
        let prepared_geoms = if !matches!(
            options.execution_mode,
            ExecutionMode::PrepareNone | ExecutionMode::PrepareProbe
        ) {
            InitOnceArray::new(num_build_geoms)
        } else {
            InitOnceArray::new(0)
        };

        let mem_used = AtomicUsize::new(prepared_geoms.allocated_size());

        Ok(Self {
            op_type,
            prepared_geoms,
            mem_used,
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

        // We may want thread local factories/build_geog/probe_geog here
        let mut factory = GeographyFactory::new();
        let mut probe_geog = factory
            .from_wkb(probe.buf())
            .map_err(|e| exec_datafusion_err!("{e}"))?;

        let mut build_geog = Geography::new();

        // Crude heuristic used by the S2Loop (build an index after 20 unindexed
        // contains queries even for small loops).
        if probe.buf().len() > (32 * 2 * size_of::<f64>()) || index_query_results.len() > 20 {
            probe_geog
                .prepare()
                .map_err(|e| exec_datafusion_err!("{e}"))?;
        }

        if !self.prepared_geoms.is_empty() {
            // We're in prepared build mode
            for result in index_query_results {
                let (prepared_build_geom, is_newly_inserted) =
                    self.prepared_geoms.get_or_create(result.geom_idx, || {
                        // Basically, prepare anything except points on the build side
                        if result.wkb.buf().len() > 32 {
                            let mut geog = factory
                                .from_wkb(result.wkb.buf())
                                .map_err(|e| exec_datafusion_err!("{e}"))?;
                            geog.prepare().map_err(|e| exec_datafusion_err!("{e}"))?;
                            Ok(Some(unsafe {
                                // Safety: the evaluated batches keep the required WKB alive. The
                                // refiner always outlives the evaluated batches.
                                std::mem::transmute::<Geography<'_>, Geography<'static>>(geog)
                            }))
                        } else {
                            Ok(None)
                        }
                    })?;

                let build_geog_ref = if let Some(prepared_geog) = prepared_build_geom {
                    if is_newly_inserted {
                        self.mem_used
                            .fetch_add(prepared_geog.mem_used(), Ordering::Relaxed);
                    }

                    prepared_geog
                } else {
                    factory
                        .init_from_wkb(result.wkb.buf(), &mut build_geog)
                        .map_err(|e| exec_datafusion_err!("{e}"))?;
                    &build_geog
                };

                let eval = if matches!(self.op_type, OpType::DWithin) {
                    if let Some(distance) = result.distance {
                        op.eval_binary_distance_predicate(build_geog_ref, &probe_geog, distance)
                    } else {
                        Ok(false)
                    }
                } else {
                    op.eval_binary_predicate(build_geog_ref, &probe_geog)
                };

                if eval.map_err(|e| exec_datafusion_err!("{e}"))? {
                    results.push(result.position);
                }
            }
        } else {
            // We are not in prepared build mode
            for result in index_query_results {
                factory
                    .init_from_wkb(result.wkb.buf(), &mut build_geog)
                    .map_err(|e| exec_datafusion_err!("{e}"))?;

                let eval = if matches!(self.op_type, OpType::DWithin) {
                    if let Some(distance) = result.distance {
                        op.eval_binary_distance_predicate(&build_geog, &probe_geog, distance)
                    } else {
                        Ok(false)
                    }
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

    fn estimate_max_memory_usage(&self, build_stats: &GeoStatistics) -> usize {
        // This estimate could be improved but is a crude heuristic based on:
        //
        // - In prepared build mode, we allocate the prepared geometry array
        // - Points are never prepared and don't occupy any extra memory
        // - The size of a prepared geometry once indexed is usually >500 bytes
        //   but is usually larger based on the number of coordinates. This could
        //   use a better estimate based on the mean byte size.
        //
        // Note this is called after the object has already been created but before
        // the index has ever been probed. It should be an error to have build_stats
        // where some components are None.

        // If we never prepare anything, our memory usage is negligible
        if self.prepared_geoms.is_empty() {
            return 0;
        }

        let num_polygons = build_stats.polygonal_count().unwrap_or(0);
        let num_lines = build_stats.lineal_count().unwrap_or(0);
        (num_polygons + num_lines).try_into().unwrap_or(usize::MAX) * 500
            + self.prepared_geoms.allocated_size()
    }

    fn mem_usage(&self) -> usize {
        self.mem_used.load(Ordering::Relaxed)
    }

    fn actual_execution_mode(&self) -> ExecutionMode {
        if self.prepared_geoms.is_empty() {
            ExecutionMode::PrepareNone
        } else {
            ExecutionMode::PrepareBuild
        }
    }

    fn need_more_probe_stats(&self) -> bool {
        false
    }

    fn merge_probe_stats(&self, _stats: GeoStatistics) {
        // Statistics are not currently used to optimize preparedness
    }
}

#[derive(Debug)]
pub struct GeographyRefinerFactory;

impl IndexQueryResultRefinerFactory for GeographyRefinerFactory {
    fn create_refiner(
        &self,
        predicate: &SpatialPredicate,
        options: SpatialJoinOptions,
        num_build_geoms: usize,
        _build_stats: GeoStatistics,
    ) -> Result<Arc<dyn IndexQueryResultRefiner>> {
        Ok(Arc::new(GeographyRefiner::new(
            predicate.clone(),
            &options,
            num_build_geoms,
        )?))
    }
}
