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

use crate::{
    distance::st_distance_impl,
    st_area::st_area_impl,
    st_boundary::st_boundary_impl,
    st_buffer::{st_buffer_impl, st_buffer_style_impl},
    st_centroid::st_centroid_impl,
    st_convexhull::st_convex_hull_impl,
    st_dwithin::st_dwithin_impl,
    st_isring::st_is_ring_impl,
    st_issimple::st_is_simple_impl,
    st_isvalid::st_is_valid_impl,
    st_isvalidreason::st_is_valid_reason_impl,
    st_length::st_length_impl,
    st_makevalid::st_make_valid_impl,
    st_perimeter::st_perimeter_impl,
    st_polygonize_agg::st_polygonize_agg_impl,
    st_reverse::st_reverse_impl,
    st_simplify::st_simplify_impl,
    st_simplifypreservetopology::st_simplify_preserve_topology_impl,
    st_snap::st_snap_impl,
    st_unaryunion::st_unary_union_impl,
};

use crate::binary_predicates::{
    st_contains_impl, st_covered_by_impl, st_covers_impl, st_crosses_impl, st_disjoint_impl,
    st_equals_impl, st_intersects_impl, st_overlaps_impl, st_touches_impl, st_within_impl,
};

use crate::overlay::{
    st_difference_impl, st_intersection_impl, st_sym_difference_impl, st_union_impl,
};

pub fn scalar_kernels() -> Vec<(&'static str, ScalarKernelRef)> {
    vec![
        ("st_area", st_area_impl()),
        ("st_boundary", st_boundary_impl()),
        ("st_buffer", st_buffer_impl()),
        ("st_buffer", st_buffer_style_impl()),
        ("st_centroid", st_centroid_impl()),
        ("st_contains", st_contains_impl()),
        ("st_convexhull", st_convex_hull_impl()),
        ("st_coveredby", st_covered_by_impl()),
        ("st_covers", st_covers_impl()),
        ("st_crosses", st_crosses_impl()),
        ("st_difference", st_difference_impl()),
        ("st_disjoint", st_disjoint_impl()),
        ("st_distance", st_distance_impl()),
        ("st_dwithin", st_dwithin_impl()),
        ("st_equals", st_equals_impl()),
        ("st_intersection", st_intersection_impl()),
        ("st_intersects", st_intersects_impl()),
        ("st_isring", st_is_ring_impl()),
        ("st_issimple", st_is_simple_impl()),
        ("st_isvalid", st_is_valid_impl()),
        ("st_isvalidreason", st_is_valid_reason_impl()),
        ("st_length", st_length_impl()),
        ("st_makevalid", st_make_valid_impl()),
        ("st_overlaps", st_overlaps_impl()),
        ("st_perimeter", st_perimeter_impl()),
        ("st_reverse", st_reverse_impl()),
        ("st_simplify", st_simplify_impl()),
        (
            "st_simplifypreservetopology",
            st_simplify_preserve_topology_impl(),
        ),
        ("st_snap", st_snap_impl()),
        ("st_symdifference", st_sym_difference_impl()),
        ("st_touches", st_touches_impl()),
        ("st_unaryunion", st_unary_union_impl()),
        ("st_union", st_union_impl()),
        ("st_within", st_within_impl()),
    ]
}

pub fn aggregate_kernels() -> Vec<(&'static str, SedonaAccumulatorRef)> {
    vec![("st_polygonize_agg", st_polygonize_agg_impl())]
}
