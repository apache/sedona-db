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
use sedona_expr::{item_crs::ItemCrsKernel, scalar_udf::ScalarKernelRef};

use crate::kernels::{
    st_astext_impl, st_astext_precision_impl, st_geogfromwkb_impl, st_geogfromwkt_impl,
    st_geomfromwkb_impl, st_geomfromwkt_impl,
};

fn st_astext_kernels() -> Vec<ScalarKernelRef> {
    let mut kernels = ItemCrsKernel::wrap_impl(vec![st_astext_impl()]);
    kernels.extend(st_astext_precision_impl());
    kernels
}

pub fn scalar_kernels() -> Vec<(&'static str, Vec<ScalarKernelRef>)> {
    vec![
        ("st_astext", st_astext_kernels()),
        ("st_geogfromwkb", vec![st_geogfromwkb_impl()]),
        ("st_geogfromwkt", vec![st_geogfromwkt_impl()]),
        ("st_geomfromwkb", vec![st_geomfromwkb_impl()]),
        ("st_geomfromwkt", vec![st_geomfromwkt_impl()]),
    ]
}
