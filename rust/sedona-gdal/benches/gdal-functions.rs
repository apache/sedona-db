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
use sedona_expr::function_set::FunctionSet;
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*, BenchmarkArgs};

fn criterion_benchmark(c: &mut Criterion) {
    let mut f = FunctionSet::new();
    for (name, kernel) in sedona_gdal::register::scalar_kernels() {
        f.add_scalar_udf_kernel(name, kernel).unwrap();
    }

    let args = BenchmarkArgs::ArrayScalarScalarScalar(
        Raster(128, 128, 1),
        Int32(0, 127),
        Int32(0, 127),
        Int32(1, 2),
    );

    benchmark::scalar(c, &f, "sedona-gdal", "rs_value", args);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
