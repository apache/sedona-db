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
use sedona_expr::scalar_udf::ScalarKernelRef;

use crate::scalar_kernel;

pub fn scalar_kernels() -> Vec<(&'static str, ScalarKernelRef)> {
    vec![
        ("st_area", scalar_kernel::st_area_impl()),
        ("st_centroid", scalar_kernel::st_centroid_impl()),
        ("st_closestpoint", scalar_kernel::st_closest_point_impl()),
        ("st_contains", scalar_kernel::st_contains_impl()),
        ("st_convexhull", scalar_kernel::st_convex_hull_impl()),
        ("st_difference", scalar_kernel::st_difference_impl()),
        ("st_distance", scalar_kernel::st_distance_impl()),
        ("st_equals", scalar_kernel::st_equals_impl()),
        ("st_intersection", scalar_kernel::st_intersection_impl()),
        ("st_intersects", scalar_kernel::st_intersects_impl()),
        (
            "st_lineinterpolatepoint",
            scalar_kernel::st_line_interpolate_point_impl(),
        ),
        (
            "st_linelocatepoint",
            scalar_kernel::st_line_locate_point_impl(),
        ),
        ("st_length", scalar_kernel::st_length_impl()),
        ("st_symdifference", scalar_kernel::st_sym_difference_impl()),
        ("st_maxdistance", scalar_kernel::st_max_distance_impl()),
        ("st_perimeter", scalar_kernel::st_perimeter_impl()),
        ("st_shortestline", scalar_kernel::st_shortest_line_impl()),
        ("st_union", scalar_kernel::st_union_impl()),
    ]
}
