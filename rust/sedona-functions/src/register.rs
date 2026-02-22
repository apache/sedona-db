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
        crate::sd_format::sd_format_udf,
        crate::sd_order::sd_order_udf,
        crate::st_affine::st_affine_udf,
        crate::st_asbinary::st_asbinary_udf,
        crate::st_asewkb::st_asewkb_udf,
        crate::st_astext::st_astext_udf,
        crate::st_azimuth::st_azimuth_udf,
        crate::st_dimension::st_dimension_udf,
        crate::st_dump::st_dump_udf,
        crate::st_envelope::st_envelope_udf,
        crate::st_flipcoordinates::st_flipcoordinates_udf,
        crate::st_geometryn::st_geometryn_udf,
        crate::st_geometrytype::st_geometry_type_udf,
        crate::st_geomfromewkb::st_geomfromewkb_udf,
        crate::st_geomfromwkb::st_geogfromwkb_udf,
        crate::st_geomfromwkb::st_geomfromwkb_udf,
        crate::st_geomfromwkb::st_geomfromwkbunchecked_udf,
        crate::st_geomfromwkt::st_geogfromwkt_udf,
        crate::st_geomfromwkt::st_geomfromewkt_udf,
        crate::st_geomfromwkt::st_geomfromwkt_udf,
        crate::st_haszm::st_hasm_udf,
        crate::st_haszm::st_hasz_udf,
        crate::st_interiorringn::st_interiorringn_udf,
        crate::st_isclosed::st_isclosed_udf,
        crate::st_iscollection::st_iscollection_udf,
        crate::st_isempty::st_isempty_udf,
        crate::st_knn::st_knn_udf,
        crate::st_makeline::st_makeline_udf,
        crate::st_numgeometries::st_numgeometries_udf,
        crate::st_point::st_geogpoint_udf,
        crate::st_point::st_point_udf,
        crate::st_pointn::st_pointn_udf,
        crate::st_points::st_npoints_udf,
        crate::st_points::st_points_udf,
        crate::st_pointzm::st_pointm_udf,
        crate::st_pointzm::st_pointz_udf,
        crate::st_pointzm::st_pointzm_udf,
        crate::st_reverse::st_reverse_udf,
        crate::st_rotate::st_rotate_udf,
        crate::st_rotate::st_rotate_x_udf,
        crate::st_rotate::st_rotate_y_udf,
        crate::st_scale::st_scale_udf,
        crate::st_setsrid::st_set_crs_udf,
        crate::st_setsrid::st_set_srid_udf,
        crate::st_srid::st_crs_udf,
        crate::st_srid::st_srid_udf,
        crate::st_start_point::st_end_point_udf,
        crate::st_start_point::st_start_point_udf,
        crate::st_translate::st_translate_udf,
        crate::st_force_dim::st_force2d_udf,
        crate::st_force_dim::st_force3d_udf,
        crate::st_force_dim::st_force3dm_udf,
        crate::st_force_dim::st_force4d_udf,
        crate::st_xyzm_minmax::st_mmax_udf,
        crate::st_xyzm_minmax::st_mmin_udf,
        crate::st_xyzm_minmax::st_xmax_udf,
        crate::st_xyzm_minmax::st_xmin_udf,
        crate::st_xyzm_minmax::st_ymax_udf,
        crate::st_xyzm_minmax::st_ymin_udf,
        crate::st_xyzm_minmax::st_zmax_udf,
        crate::st_xyzm_minmax::st_zmin_udf,
        crate::st_xyzm::st_m_udf,
        crate::st_xyzm::st_x_udf,
        crate::st_xyzm::st_y_udf,
        crate::st_xyzm::st_z_udf,
        crate::st_zmflag::st_zmflag_udf,
    );

    register_aggregate_udfs!(
        function_set,
        crate::st_analyze_agg::st_analyze_agg_udf,
        crate::st_collect_agg::st_collect_agg_udf,
        crate::st_envelope_agg::st_envelope_agg_udf,
    );

    function_set
}
