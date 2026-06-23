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

//! Benchmarks for RS_Clip UDF
//!
//! RS_Clip clips rasters to a geometry boundary.
//! Signature: RS_Clip(raster, geometry, [nodata], [crop])
//!
//! NOTE: This benchmark requires geometry creation which needs additional setup.
//! The underlying GDAL segfault issue has been fixed.
//! TODO: Implement full benchmark once geometry utilities are available in bench context.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_rs_clip_placeholder(c: &mut Criterion) {
    let mut group = c.benchmark_group("rs_clip");

    // Placeholder benchmark - needs geometry creation utilities
    // The underlying GDAL memory dataset issue has been fixed
    group.bench_function("placeholder", |b| b.iter(|| std::hint::black_box(42)));

    group.finish();
}

criterion_group!(benches, bench_rs_clip_placeholder);
criterion_main!(benches);
