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

use arrow_array::ArrayRef;
use datafusion_common::{exec_datafusion_err, not_impl_err, JoinType, Result};
use datafusion_physical_plan::ColumnarValue;
use geo_index::rtree::util::f64_box_to_f32;
use geo_types::{coord, Rect};
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::executor::IterGeo;
use sedona_s2geography::{
    geography::{Geography, GeographyFactory},
    rect_bounder::RectBounder,
};
use sedona_schema::datatypes::SedonaType;
use sedona_spatial_join::{
    index::{spatial_index_builder::SpatialJoinBuildMetrics, SpatialIndexBuilder},
    join_provider::SpatialJoinProvider,
    operand_evaluator::{EvaluatedGeometryArray, EvaluatedGeometryArrayFactory},
    SpatialJoinOptions, SpatialPredicate,
};

use crate::spatial_index_builder::GeographySpatialIndexBuilder;

#[derive(Debug)]
pub struct GeographyJoinProvider;

impl SpatialJoinProvider for GeographyJoinProvider {
    fn try_new_spatial_index_builder(
        &self,
        schema: arrow_schema::SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Box<dyn SpatialIndexBuilder>> {
        // Create the inner (default) builder
        let builder = GeographySpatialIndexBuilder::new(
            schema,
            spatial_predicate.clone(),
            options,
            join_type,
            probe_threads_count,
            metrics,
        )?;

        Ok(Box::new(builder))
    }

    fn estimate_extra_memory_usage(
        &self,
        _geo_stats: &GeoStatistics,
        _spatial_predicate: &SpatialPredicate,
        _options: &SpatialJoinOptions,
    ) -> usize {
        // TODO: calculate
        0
    }

    fn evaluated_array_factory(&self) -> Arc<dyn EvaluatedGeometryArrayFactory> {
        Arc::new(GeographyEvaluatedArrayFactory)
    }
}

#[derive(Debug)]
struct GeographyEvaluatedArrayFactory;

impl EvaluatedGeometryArrayFactory for GeographyEvaluatedArrayFactory {
    fn try_new_evaluated_array(
        &self,
        geometry_array: ArrayRef,
        sedona_type: &SedonaType,
        distance_columnar_value: Option<&ColumnarValue>,
    ) -> Result<EvaluatedGeometryArray> {
        // We don't support expansion yet
        if distance_columnar_value.is_some() {
            return not_impl_err!(
                "rectangle expansion by distance is not yet supported for geography joins"
            );
        }

        // compute rectangles from wkb using the RectBounder from s2geography
        let num_rows = geometry_array.len();
        let mut bounder = RectBounder::new();
        let mut factory = GeographyFactory::new();
        let mut geog = Geography::new();
        let mut rect_vec = Vec::with_capacity(num_rows);
        geometry_array.iter_as_wkb_bytes(sedona_type, num_rows, |wkb_opt| {
            if let Some(wkb) = wkb_opt {
                bounder.clear();
                factory.init_from_wkb(wkb, &mut geog).map_err(|e| {
                    exec_datafusion_err!("Failed to read WKB in evaluated array factory: {e}")
                })?;
                bounder.bound(&geog).map_err(|e| {
                    exec_datafusion_err!(
                        "Failed to bound geography in evaluated array factory: {e}"
                    )
                })?;
                let maybe_bounds = bounder.finish().map_err(|e| {
                    exec_datafusion_err!(
                        "Failed to finish bounding in evaluated array factory: {e}"
                    )
                })?;
                if let Some((mut min_x, min_y, mut max_x, max_y)) = maybe_bounds {
                    // The evaluated geometry array currently needs Cartesian rectangles; however
                    // we can still recalculate these when we ingest into the index. In the
                    // partitioned join we may want to ensure we can express bounds in a way that
                    // the partitioner understands (if it doesn't already) to do a better job
                    // partitioning wraparounds.
                    if min_x > max_x {
                        min_x = -180.0;
                        max_x = 180.0;
                    }

                    let (min_x, min_y, max_x, max_y) = f64_box_to_f32(min_x, min_y, max_x, max_y);
                    let rect = Rect::new(coord!(x: min_x, y: min_y), coord!(x: max_x, y: max_y));
                    rect_vec.push(Some(rect));
                } else {
                    rect_vec.push(None);
                }
            } else {
                rect_vec.push(None);
            }

            Ok(())
        })?;

        EvaluatedGeometryArray::try_new_with_rects(geometry_array, rect_vec, sedona_type)
    }
}
