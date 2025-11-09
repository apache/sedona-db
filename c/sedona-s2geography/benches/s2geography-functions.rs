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
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_expr::ScalarUDF;
use sedona_expr::{
    function_set::FunctionSet,
    scalar_udf::{SedonaScalarUDF, SimpleSedonaScalarKernel},
};
use sedona_schema::{datatypes::WKB_GEOGRAPHY, matchers::ArgMatcher};
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*, BenchmarkArgs};

fn criterion_benchmark(c: &mut Criterion) {
    let mut f = FunctionSet::new();
    for (name, kernel) in sedona_s2geography::register::scalar_kernels() {
        f.add_scalar_udf_kernel(name, kernel).unwrap();
    }

    // Single geometry functions
    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_area",
        Transformed(Polygon(10).into(), to_geography()),
    );

    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_length",
        Transformed(LineString(10).into(), to_geography()),
    );

    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_centroid",
        Transformed(Polygon(10).into(), to_geography()),
    );

    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_convexhull",
        Transformed(Polygon(10).into(), to_geography()),
    );

    // Binary geometry functions
    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_distance",
        BenchmarkArgs::ArrayScalar(
            Transformed(Point.into(), to_geography()),
            Transformed(Point.into(), to_geography()),
        ),
    );

    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_closestpoint",
        BenchmarkArgs::ArrayScalar(
            Transformed(LineString(10).into(), to_geography()),
            Transformed(LineString(10).into(), to_geography()),
        ),
    );
    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_intersection",
        BenchmarkArgs::ArrayScalar(
            Transformed(Polygon(10).into(), to_geography()),
            Transformed(Polygon(10).into(), to_geography()),
        ),
    );

    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_union",
        BenchmarkArgs::ArrayScalar(
            Transformed(Polygon(10).into(), to_geography()),
            Transformed(Polygon(10).into(), to_geography()),
        ),
    );

    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_difference",
        BenchmarkArgs::ArrayScalar(
            Transformed(Polygon(10).into(), to_geography()),
            Transformed(Polygon(10).into(), to_geography()),
        ),
    );

    // Predicate functions
    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_contains",
        BenchmarkArgs::ScalarArray(
            Transformed(Polygon(10).into(), to_geography()),
            Transformed(Point.into(), to_geography()),
        ),
    );

    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_intersects",
        BenchmarkArgs::ArrayScalar(
            Transformed(Point.into(), to_geography()),
            Transformed(Polygon(10).into(), to_geography()),
        ),
    );

    benchmark::scalar(
        c,
        &f,
        "s2geography",
        "st_equals",
        BenchmarkArgs::ArrayScalar(
            Transformed(Polygon(10).into(), to_geography()),
            Transformed(Polygon(10).into(), to_geography()),
        ),
    );
}

fn to_geography() -> ScalarUDF {
    let kernel = SimpleSedonaScalarKernel::new_ref(
        ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOGRAPHY),
        Arc::new(move |_, args| args[0].cast_to(WKB_GEOGRAPHY.storage_type(), None)),
    );
    SedonaScalarUDF::from_kernel("geog", kernel).into()
}

criterion_group! {
    name = benches;
    // These are currently very slow, so only run 10 samples
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
}
criterion_main!(benches);
