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
use sedona_expr::aggregate_udf::SedonaAccumulatorRef;
use sedona_expr::scalar_udf::ScalarKernelRef;

use crate::st_intersection_agg::st_intersection_agg_impl;
use crate::st_line_interpolate_point::st_line_interpolate_point_impl;
use crate::st_union_agg::st_union_agg_impl;
use crate::{
    st_area::st_area_impl, st_centroid::st_centroid_impl, st_distance::st_distance_impl,
    st_dwithin::st_dwithin_impl, st_intersects::st_intersects_impl, st_length::st_length_impl,
    st_perimeter::st_perimeter_impl,
};

pub fn scalar_kernels() -> Vec<(&'static str, ScalarKernelRef)> {
    vec![
        ("st_intersects", st_intersects_impl()),
        ("st_area", st_area_impl()),
        ("st_centroid", st_centroid_impl()),
        ("st_distance", st_distance_impl()),
        ("st_dwithin", st_dwithin_impl()),
        ("st_length", st_length_impl()),
        ("st_perimeter", st_perimeter_impl()),
        ("st_lineinterpolatepoint", st_line_interpolate_point_impl()),
    ]
}

pub fn aggregate_kernels() -> Vec<(&'static str, SedonaAccumulatorRef)> {
    vec![
        ("st_intersection_agg", st_intersection_agg_impl()),
        ("st_union_agg", st_union_agg_impl()),
    ]
}
