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
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*, BenchmarkArgs::*};

fn criterion_benchmark(c: &mut Criterion) {
    let mut f = FunctionSet::new();
    for (name, kernel) in sedona_geo::register::scalar_kernels() {
        f.add_scalar_udf_kernel(name, kernel).unwrap();
    }

    benchmark::scalar(c, &f, "geo", "st_area", Polygon(10));
    benchmark::scalar(c, &f, "geo", "st_area", Polygon(500));

    benchmark::scalar(
        c,
        &f,
        "geo",
        "st_buffer",
        ArrayScalar(Polygon(10), Float64(1.0, 10.0)),
    );
    benchmark::scalar(
        c,
        &f,
        "geo",
        "st_buffer",
        ArrayScalar(Polygon(50), Float64(1.0, 10.0)), // Reduced from 500 so that it can finish
    );

    benchmark::scalar(c, &f, "geo", "st_perimeter", Polygon(10));
    benchmark::scalar(c, &f, "geo", "st_perimeter", Polygon(500));

    benchmark::scalar(
        c,
        &f,
        "geo",
        "st_intersects",
        ArrayScalar(Point, Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geo",
        "st_intersects",
        ArrayScalar(Point, Polygon(500)),
    );

    benchmark::scalar(c, &f, "geo", "st_distance", ArrayScalar(Point, Polygon(10)));
    benchmark::scalar(
        c,
        &f,
        "geo",
        "st_distance",
        ArrayScalar(Point, Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geo",
        "st_dwithin",
        ArrayArrayScalar(Polygon(10), Polygon(10), Float64(1.0, 2.0)),
    );
    benchmark::scalar(
        c,
        &f,
        "geo",
        "st_dwithin",
        ArrayArrayScalar(Polygon(10), Polygon(500), Float64(1.0, 2.0)),
    );
}

fn criterion_benchmark_aggr(c: &mut Criterion) {
    let mut f = sedona_functions::register::default_function_set();
    for (name, kernel) in sedona_geo::register::aggregate_kernels() {
        f.add_aggregate_udf_kernel(name, kernel).unwrap();
    }

    // st_intersection_agg would need its own configuration because most of the generated
    // polygons would not intersect and result in empty output almost immediately
    benchmark::aggregate(c, &f, "geo", "st_union_agg", Polygon(10));
}

criterion_group!(benches, criterion_benchmark);
criterion_group! {
    name = benches_aggr;
    // These are currently very slow, so only run 10 samples
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark_aggr
}

criterion_main!(benches, benches_aggr);
