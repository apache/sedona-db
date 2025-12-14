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
        crate::rs_envelope::rs_envelope_udf,
        crate::rs_example::rs_example_udf,
        crate::rs_geotransform::rs_rotation_udf,
        crate::rs_geotransform::rs_scalex_udf,
        crate::rs_geotransform::rs_scaley_udf,
        crate::rs_geotransform::rs_skewx_udf,
        crate::rs_geotransform::rs_skewy_udf,
        crate::rs_geotransform::rs_upperleftx_udf,
        crate::rs_geotransform::rs_upperlefty_udf,
        crate::rs_rastercoordinate::rs_worldtorastercoord_udf,
        crate::rs_rastercoordinate::rs_worldtorastercoordx_udf,
        crate::rs_rastercoordinate::rs_worldtorastercoordy_udf,
        crate::rs_size::rs_height_udf,
        crate::rs_size::rs_width_udf,
        crate::rs_worldcoordinate::rs_rastertoworldcoord_udf,
        crate::rs_worldcoordinate::rs_rastertoworldcoordx_udf,
        crate::rs_worldcoordinate::rs_rastertoworldcoordy_udf,
    );

    register_aggregate_udfs!(function_set,);

    function_set
}
