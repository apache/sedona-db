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

use arrow_array::{ArrayRef, Float64Array};
use arrow_schema::DataType;
use datafusion_common::{exec_datafusion_err, JoinType, Result, ScalarValue};
use datafusion_physical_plan::ColumnarValue;
use sedona_common::sedona_internal_err;
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::executor::IterGeo;
use sedona_s2geography::{
    geography::{Geography, GeographyFactory},
    rect_bounder::RectBounder,
};
use sedona_schema::datatypes::SedonaType;
use sedona_spatial_join::{
    index::{
        default_spatial_index_builder::DefaultSpatialIndexBuilder,
        spatial_index_builder::SpatialJoinBuildMetrics, SpatialIndexBuilder,
    },
    join_provider::SpatialJoinProvider,
    operand_evaluator::{EvaluatedGeometryArray, EvaluatedGeometryArrayFactory},
    utils::bounds::Bounds2D,
    SpatialJoinOptions, SpatialPredicate,
};

use crate::{
    refiner::GeographyRefinerFactory, spatial_index_builder::GeographySpatialIndexBuilder,
};

#[derive(Debug)]
pub(crate) struct GeographyJoinProvider;

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
        geo_stats: &GeoStatistics,
        spatial_predicate: &SpatialPredicate,
        options: &SpatialJoinOptions,
    ) -> usize {
        DefaultSpatialIndexBuilder::estimate_extra_memory_usage(
            geo_stats,
            spatial_predicate,
            options,
            Arc::new(GeographyRefinerFactory),
        )
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
        // Without distance expansion use the impl without a bounder modifier
        let Some(distance_columnar_value) = distance_columnar_value else {
            return try_new_evaluated_array_impl(geometry_array, sedona_type, |_bounder| {});
        };

        // Cast to float if the input was some other numeric
        let distance_columnar_value = distance_columnar_value.cast_to(&DataType::Float64, None)?;

        let result = match &distance_columnar_value {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(distance))) => {
                try_new_evaluated_array_impl(geometry_array, sedona_type, |bounder| {
                    bounder.expand_by_distance(*distance);
                })
            }
            ColumnarValue::Scalar(ScalarValue::Float64(None)) => {
                try_new_evaluated_array_impl(geometry_array, sedona_type, |bounder| {
                    bounder.clear();
                })
            }
            ColumnarValue::Array(array) => {
                if let Some(array) = array.as_any().downcast_ref::<Float64Array>() {
                    let mut distance_iter = array.iter();
                    try_new_evaluated_array_impl(geometry_array, sedona_type, |bounder| {
                        if let Some(next_distance) = distance_iter
                            .next()
                            .expect("distance array has correct length")
                        {
                            // Non null distance
                            bounder.expand_by_distance(next_distance);
                        } else {
                            // Null distance
                            bounder.clear();
                        }
                    })
                } else {
                    return sedona_internal_err!("Distance columnar value is not a Float64Array");
                }
            }
            _ => {
                return sedona_internal_err!("Distance columnar value is not a Float64");
            }
        }?;

        // Store the distance value so it can be used during refinement
        Ok(result.with_distance(Some(distance_columnar_value.clone())))
    }
}

fn try_new_evaluated_array_impl(
    geometry_array: ArrayRef,
    sedona_type: &SedonaType,
    mut modify_bounder: impl FnMut(&mut RectBounder),
) -> Result<EvaluatedGeometryArray> {
    // Compute rectangles from wkb using the RectBounder from s2geography
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
                exec_datafusion_err!("Failed to bound geography in evaluated array factory: {e}")
            })?;

            // Account for distance (either by expanding the bounds or by emptying them for a null
            // distance).
            modify_bounder(&mut bounder);

            let maybe_bounds = bounder.finish().map_err(|e| {
                exec_datafusion_err!("Failed to finish bounding in evaluated array factory: {e}")
            })?;
            if let Some((min_x, min_y, max_x, max_y)) = maybe_bounds {
                rect_vec.push(Bounds2D::new((min_x, max_x), (min_y, max_y)));
            } else {
                rect_vec.push(Bounds2D::empty());
            }
        } else {
            // Also call the bounder modifier here to ensure it's called exactly once for every
            // element
            modify_bounder(&mut bounder);
            rect_vec.push(Bounds2D::empty());
        }

        Ok(())
    })?;

    EvaluatedGeometryArray::try_new_with_rects(geometry_array, rect_vec, sedona_type)
}
