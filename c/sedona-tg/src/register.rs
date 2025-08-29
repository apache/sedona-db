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

use crate::binary_predicate;

pub fn scalar_kernels() -> Vec<(&'static str, ScalarKernelRef)> {
    vec![
        ("st_equals", binary_predicate::st_equals_impl()),
        ("st_intersects", binary_predicate::st_intersects_impl()),
        ("st_disjoint", binary_predicate::st_disjoint_impl()),
        ("st_contains", binary_predicate::st_contains_impl()),
        ("st_within", binary_predicate::st_within_impl()),
        ("st_covers", binary_predicate::st_covers_impl()),
        ("st_coveredby", binary_predicate::st_covered_by_impl()),
        ("st_touches", binary_predicate::st_touches_impl()),
    ]
}
