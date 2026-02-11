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

use criterion::{criterion_group, criterion_main, Criterion};
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*};

fn criterion_benchmark(c: &mut Criterion) {
    let f = sedona_raster_functions::register::default_function_set();

    // RS_Rotation(raster)
    benchmark::scalar(c, &f, "rs_geotransform", "rs_rotation", Raster(64, 64));

    // RS_ScaleX(raster)
    benchmark::scalar(c, &f, "rs_geotransform", "rs_scalex", Raster(64, 64));

    // RS_ScaleY(raster)
    benchmark::scalar(c, &f, "rs_geotransform", "rs_scaley", Raster(64, 64));

    // RS_SkewX(raster)
    benchmark::scalar(c, &f, "rs_geotransform", "rs_skewx", Raster(64, 64));

    // RS_SkewY(raster)
    benchmark::scalar(c, &f, "rs_geotransform", "rs_skewy", Raster(64, 64));

    // RS_UpperLeftX(raster)
    benchmark::scalar(c, &f, "rs_geotransform", "rs_upperleftx", Raster(64, 64));

    // RS_UpperLeftY(raster)
    benchmark::scalar(c, &f, "rs_geotransform", "rs_upperlefty", Raster(64, 64));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
