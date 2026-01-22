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
use sedona_expr::scalar_udf::{IntoScalarKernelRefs, ScalarKernelRef};

macro_rules! define_scalar_kernels {
    ($($name:expr => $impl:expr),* $(,)?) => {
        vec![
            $(
                ($name, $impl().into_scalar_kernel_refs()),
            )*
        ]
    };
}

macro_rules! define_aggregate_kernels {
    ($($name:expr => $impl:expr),* $(,)?) => {
        vec![
            $(
                ($name, $impl()),
            )*
        ]
    };
}

pub fn scalar_kernels() -> Vec<(&'static str, Vec<ScalarKernelRef>)> {
    define_scalar_kernels!(
        "st_area" => crate::st_area::st_area_impl,
        "st_boundary" => crate::st_boundary::st_boundary_impl,
        "st_buffer" => crate::st_buffer::st_buffer_impl,
        "st_buffer" => crate::st_buffer::st_buffer_style_impl,
        "st_centroid" => crate::st_centroid::st_centroid_impl,
        "st_concavehull" => crate::st_concavehull::st_concave_hull_allow_holes_impl,
        "st_concavehull" => crate::st_concavehull::st_concave_hull_impl,
        "st_contains" => crate::binary_predicates::st_contains_impl,
        "st_convexhull" => crate::st_convexhull::st_convex_hull_impl,
        "st_coveredby" => crate::binary_predicates::st_covered_by_impl,
        "st_covers" => crate::binary_predicates::st_covers_impl,
        "st_crosses" => crate::binary_predicates::st_crosses_impl,
        "st_difference" => crate::overlay::st_difference_impl,
        "st_disjoint" => crate::binary_predicates::st_disjoint_impl,
        "st_distance" => crate::distance::st_distance_impl,
        "st_dwithin" => crate::st_dwithin::st_dwithin_impl,
        "st_equals" => crate::binary_predicates::st_equals_impl,
        "st_intersection" => crate::overlay::st_intersection_impl,
        "st_intersects" => crate::binary_predicates::st_intersects_impl,
        "st_isring" => crate::st_isring::st_is_ring_impl,
        "st_issimple" => crate::st_issimple::st_is_simple_impl,
        "st_isvalid" => crate::st_isvalid::st_is_valid_impl,
        "st_isvalidreason" => crate::st_isvalidreason::st_is_valid_reason_impl,
        "st_length" => crate::st_length::st_length_impl,
        "st_linemerge" => crate::st_line_merge::st_line_merge_impl,
        "st_makevalid" => crate::st_makevalid::st_make_valid_impl,
        "st_minimumclearance" => crate::st_minimumclearance::st_minimum_clearance_impl,
        "st_minimumclearanceline" => crate::st_minimumclearance_line::st_minimum_clearance_line_impl,
        "st_nrings" => crate::st_nrings::st_nrings_impl,
        "st_numinteriorrings" => crate::st_numinteriorrings::st_num_interior_rings_impl,
        "st_numpoints" => crate::st_numpoints::st_num_points_impl,
        "st_overlaps" => crate::binary_predicates::st_overlaps_impl,
        "st_perimeter" => crate::st_perimeter::st_perimeter_impl,
        "st_polygonize" => crate::st_polygonize::st_polygonize_impl,
        "st_simplify" => crate::st_simplify::st_simplify_impl,
        "st_simplifypreservetopology" => crate::st_simplifypreservetopology::st_simplify_preserve_topology_impl,
        "st_snap" => crate::st_snap::st_snap_impl,
        "st_symdifference" => crate::overlay::st_sym_difference_impl,
        "st_touches" => crate::binary_predicates::st_touches_impl,
        "st_unaryunion" => crate::st_unaryunion::st_unary_union_impl,
        "st_union" => crate::overlay::st_union_impl,
        "st_within" => crate::binary_predicates::st_within_impl,
    )
}

pub fn aggregate_kernels() -> Vec<(&'static str, Vec<SedonaAccumulatorRef>)> {
    define_aggregate_kernels!(
        "st_polygonize_agg" => crate::st_polygonize_agg::st_polygonize_agg_impl,
    )
}
