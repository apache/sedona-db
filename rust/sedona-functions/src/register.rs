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
use sedona_expr::function_set::FunctionSet;

/// Export the set of functions defined in this crate
pub fn default_function_set() -> FunctionSet {
    let mut function_set = FunctionSet::new();

    macro_rules! register_scalar_udfs {
        ($function_set:expr, $($udf:expr),* $(,)?) => {
            $(
                $function_set.insert_scalar_udf($udf());
            )*
        };
    }

    macro_rules! register_aggregate_udfs {
        ($function_set:expr, $($udf:expr),* $(,)?) => {
            $(
                $function_set.insert_aggregate_udf($udf());
            )*
        };
    }

    register_scalar_udfs!(
        function_set,
        crate::barrier::barrier_udf,
        crate::distance::st_distance_sphere_udf,
        crate::distance::st_distance_spheroid_udf,
        crate::distance::st_distance_udf,
        crate::distance::st_frechet_distance_udf,
        crate::distance::st_hausdorff_distance_udf,
        crate::distance::st_max_distance_udf,
        crate::overlay::st_difference_udf,
        crate::overlay::st_intersection_udf,
        crate::overlay::st_sym_difference_udf,
        crate::overlay::st_union_udf,
        crate::predicates::st_contains_udf,
        crate::predicates::st_covered_by_udf,
        crate::predicates::st_covers_udf,
        crate::predicates::st_disjoint_udf,
        crate::predicates::st_equals_udf,
        crate::predicates::st_intersects_udf,
        crate::predicates::st_knn_udf,
        crate::predicates::st_touches_udf,
        crate::predicates::st_within_udf,
        crate::referencing::st_line_interpolate_point_udf,
        crate::referencing::st_line_locate_point_udf,
        crate::sd_format::sd_format_udf,
        crate::st_area::st_area_udf,
        crate::st_asbinary::st_asbinary_udf,
        crate::st_astext::st_astext_udf,
        crate::st_buffer::st_buffer_udf,
        crate::st_centroid::st_centroid_udf,
        crate::st_dimension::st_dimension_udf,
        crate::st_dwithin::st_dwithin_udf,
        crate::st_envelope::st_envelope_udf,
        crate::st_flipcoordinates::st_flipcoordinates_udf,
        crate::st_geometrytype::st_geometry_type_udf,
        crate::st_geomfromwkb::st_geogfromwkb_udf,
        crate::st_geomfromwkb::st_geomfromwkb_udf,
        crate::st_geomfromwkt::st_geogfromwkt_udf,
        crate::st_geomfromwkt::st_geomfromwkt_udf,
        crate::st_haszm::st_hasm_udf,
        crate::st_haszm::st_hasz_udf,
        crate::st_isempty::st_isempty_udf,
        crate::st_length::st_length_udf,
        crate::st_makeline::st_makeline_udf,
        crate::st_perimeter::st_perimeter_udf,
        crate::st_point::st_geogpoint_udf,
        crate::st_point::st_point_udf,
        crate::st_pointzm::st_pointz_udf,
        crate::st_pointzm::st_pointm_udf,
        crate::st_pointzm::st_pointzm_udf,
        crate::st_transform::st_transform_udf,
        crate::st_setsrid::st_set_crs_udf,
        crate::st_setsrid::st_set_srid_udf,
        crate::st_srid::st_crs_udf,
        crate::st_srid::st_srid_udf,
        crate::st_xyzm::st_m_udf,
        crate::st_xyzm::st_x_udf,
        crate::st_xyzm::st_y_udf,
        crate::st_xyzm::st_z_udf,
        crate::st_xyzm_minmax::st_xmin_udf,
        crate::st_xyzm_minmax::st_ymin_udf,
        crate::st_xyzm_minmax::st_xmax_udf,
        crate::st_xyzm_minmax::st_ymax_udf,
        crate::st_xyzm_minmax::st_zmin_udf,
        crate::st_xyzm_minmax::st_zmax_udf,
        crate::st_xyzm_minmax::st_mmin_udf,
        crate::st_xyzm_minmax::st_mmax_udf,
    );

    register_aggregate_udfs!(
        function_set,
        crate::st_analyze_aggr::st_analyze_aggr_udf,
        crate::st_collect::st_collect_udf,
        crate::st_envelope_aggr::st_envelope_aggr_udf,
        crate::st_intersection_aggr::st_intersection_aggr_udf,
        crate::st_union_aggr::st_union_aggr_udf,
    );

    function_set
}

/// Functions whose implementations are registered independently
///
/// These functions are included in the default function set; however,
/// it is useful to expose them individually for testing in crates that
/// implement them.
pub mod stubs {
    pub use crate::overlay::*;
    pub use crate::predicates::*;
    pub use crate::referencing::*;
    pub use crate::st_area::st_area_udf;
    pub use crate::st_centroid::st_centroid_udf;
    pub use crate::st_length::st_length_udf;
    pub use crate::st_perimeter::st_perimeter_udf;
    pub use crate::st_setsrid::st_set_crs_with_engine_udf;
    pub use crate::st_setsrid::st_set_srid_with_engine_udf;
    pub use crate::st_transform::st_transform_udf;
}
