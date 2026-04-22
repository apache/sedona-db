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

use crate::index::GpuSpatialIndexBuilder;
use crate::options::GpuOptions;
use arrow_array::ArrayRef;
use arrow_schema::SchemaRef;
use datafusion::logical_expr::ColumnarValue;
use datafusion_common::not_impl_err;
use datafusion_common::JoinType;
use datafusion_common::Result;
use geo_index::rtree::util::f64_box_to_f32;
use geo_types::{coord, Rect};
use sedona_common::SpatialJoinOptions;
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::executor::IterGeo;
use sedona_geo_generic_alg::BoundingRect;
use sedona_schema::datatypes::SedonaType;
use sedona_spatial_join::index::spatial_index_builder::{
    SpatialIndexBuilder, SpatialJoinBuildMetrics,
};
use sedona_spatial_join::join_provider::SpatialJoinProvider;
use sedona_spatial_join::operand_evaluator::{
    EvaluatedGeometryArray, EvaluatedGeometryArrayFactory,
};
use sedona_spatial_join::SpatialPredicate;
use std::sync::Arc;
use wkb::reader::GeometryType;

#[derive(Debug)]
pub(crate) struct GpuSpatialJoinProvider {
    gpu_options: GpuOptions,
}

impl GpuSpatialJoinProvider {
    pub(crate) fn new(gpu_options: GpuOptions) -> Self {
        Self { gpu_options }
    }
}

impl SpatialJoinProvider for GpuSpatialJoinProvider {
    fn try_new_spatial_index_builder(
        &self,
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        _options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Box<dyn SpatialIndexBuilder>> {
        let builder = GpuSpatialIndexBuilder::new(
            schema,
            spatial_predicate,
            self.gpu_options.clone(),
            join_type,
            probe_threads_count,
            metrics,
        );
        Ok(Box::new(builder))
    }

    fn estimate_extra_memory_usage(
        &self,
        geo_stats: &GeoStatistics,
        spatial_predicate: &SpatialPredicate,
        options: &SpatialJoinOptions,
    ) -> usize {
        GpuSpatialIndexBuilder::estimate_extra_memory_usage(geo_stats, spatial_predicate, options)
    }

    fn evaluated_array_factory(&self) -> Arc<dyn EvaluatedGeometryArrayFactory> {
        Arc::new(DefaultGeometryArrayFactory)
    }
}

#[derive(Debug)]
pub(crate) struct DefaultGeometryArrayFactory;

impl EvaluatedGeometryArrayFactory for DefaultGeometryArrayFactory {
    fn try_new_evaluated_array(
        &self,
        geometry_array: ArrayRef,
        sedona_type: &SedonaType,
        distance_columnar_value: Option<&ColumnarValue>,
    ) -> Result<EvaluatedGeometryArray> {
        if distance_columnar_value.is_some() {
            return not_impl_err!(
                "rectangle expansion by distance is not yet supported for GPU joins"
            );
        }

        let num_rows = geometry_array.len();
        let mut rect_vec = Vec::with_capacity(num_rows);
        geometry_array.iter_as_wkb(sedona_type, num_rows, |wkb_opt| {
            let rect_opt = if let Some(wkb) = &wkb_opt {
                // This piece of code checks whether the underlying geometry is a point
                // By representing the point with an MBR with the same min corner and max corner,
                // libgpuspatial treats the MBR as a point, which triggers an optimized point query
                // instead of using rect-rect query for high performance
                // Ref: https://github.com/apache/sedona-db/blob/9187f8b8c4ca52b64837fab5fddd377703f7331b/c/sedona-libgpuspatial/libgpuspatial/src/rt_spatial_index.cu#L374
                if let Some(rect) = wkb.bounding_rect() {
                    let min = rect.min();
                    let max = rect.max();
                    // Why conservative bounding boxes prevent false negatives:
                    // 1. P32 = round_nearest(P64), so P32 is the closest possible float to P64.
                    // 2. Min32 = round_down(Min64), guaranteeing Min32 <= Min64.
                    // 3. Max32 = round_up(Max64), guaranteeing Max32 >= Max64.
                    // If P64 is inside Box64 (Min64 <= P64 <= Max64), P32 cannot fall outside Box32.
                    // If P32 < Min32, it would mean Min32 is closer to P64 than P32 is, which
                    // contradicts P32 being the nearest float. Therefore, false negatives are impossible.
                    if wkb.geometry_type() == GeometryType::Point {
                        Some(Rect::new(
                            coord!(x: min.x as f32, y: min.y as f32),
                            coord!(x: max.x as f32, y: max.y as f32),
                        ))
                    } else {
                        let (min_x, min_y, max_x, max_y) =
                            f64_box_to_f32(min.x, min.y, max.x, max.y);
                        Some(Rect::new(
                            coord!(x: min_x, y: min_y),
                            coord!(x: max_x, y: max_y),
                        ))
                    }
                } else {
                    None
                }
            } else {
                None
            };

            rect_vec.push(rect_opt);
            Ok(())
        })?;

        EvaluatedGeometryArray::try_new_with_rects(geometry_array, rect_vec, sedona_type)
    }
}
