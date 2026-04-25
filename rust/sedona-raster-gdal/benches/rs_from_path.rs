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

//! Benchmarks for RS_FromPath UDF
//!
//! RS_FromPath creates out-db rasters from file paths.
//!
//! NOTE: This benchmark is currently disabled because RS_FromPath has a known issue
//! with RasterBuilder not correctly handling null data for out-db rasters.
//! The out-db path support is still evolving; this file currently contains a placeholder benchmark.
//!
//! Once the out-db raster support is fixed, this benchmark should cover:
//! - Loading rasters with and without extent calculation
//! - Different raster files
//! - Batch processing

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_rs_from_path_placeholder(c: &mut Criterion) {
    let mut group = c.benchmark_group("rs_from_path");

    // Placeholder benchmark - actual benchmarks disabled due to known issue
    // with RasterBuilder not handling null data for out-db rasters
    group.bench_function("placeholder", |b| {
        b.iter(|| {
            // No-op: RS_FromPath benchmarks disabled until out-db raster issue is resolved
            std::hint::black_box(42)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_rs_from_path_placeholder);
criterion_main!(benches);
