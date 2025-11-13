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
use sedona_testing::benchmark_util::BenchmarkArgs::{ArrayArrayScalar, ArrayScalar};
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*};

fn criterion_benchmark(c: &mut Criterion) {
    let mut f = FunctionSet::new();
    for (name, kernel) in sedona_geos::register::scalar_kernels() {
        f.add_scalar_udf_kernel(name, kernel).unwrap();
    }

    benchmark::scalar(c, &f, "geos", "st_area", Polygon(10));
    benchmark::scalar(c, &f, "geos", "st_area", Polygon(500));

    benchmark::scalar(c, &f, "geos", "st_boundary", Polygon(10));
    benchmark::scalar(c, &f, "geos", "st_boundary", Polygon(500));

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_buffer",
        ArrayScalar(Polygon(10), Float64(1.0, 10.0)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_buffer",
        ArrayScalar(Polygon(500), Float64(1.0, 10.0)),
    );

    benchmark::scalar(c, &f, "geos", "st_centroid", Polygon(10));
    benchmark::scalar(c, &f, "geos", "st_centroid", Polygon(500));

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_contains",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_contains",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(c, &f, "geos", "st_convexhull", MultiPoint(10));

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_coveredby",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_coveredby",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_covers",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_covers",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_crosses",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_crosses",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_difference",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_difference",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_disjoint",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_disjoint",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_distance",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_distance",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_dwithin",
        ArrayArrayScalar(Polygon(10), Polygon(10), Float64(1.0, 2.0)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_dwithin",
        ArrayArrayScalar(Polygon(10), Polygon(500), Float64(1.0, 2.0)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_equals",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_equals",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_intersection",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_intersection",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_intersects",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_intersects",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(c, &f, "geos", "st_issimple", Polygon(10));
    benchmark::scalar(c, &f, "geos", "st_issimple", Polygon(500));

    benchmark::scalar(c, &f, "geos", "st_isvalid", Polygon(10));
    benchmark::scalar(c, &f, "geos", "st_isvalid", Polygon(500));

    benchmark::scalar(c, &f, "geos", "st_isvalidreason", Polygon(10));
    benchmark::scalar(c, &f, "geos", "st_isvalidreason", Polygon(500));

    benchmark::scalar(c, &f, "geos", "st_length", LineString(10));
    benchmark::scalar(c, &f, "geos", "st_length", LineString(500));

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_overlaps",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_overlaps",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(c, &f, "geos", "st_perimeter", Polygon(10));
    benchmark::scalar(c, &f, "geos", "st_perimeter", Polygon(500));

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_symdifference",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_symdifference",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_touches",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_touches",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(c, &f, "geos", "st_unaryunion", Polygon(10));
    benchmark::scalar(c, &f, "geos", "st_unaryunion", Polygon(500));

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_union",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_union",
        ArrayScalar(Polygon(10), Polygon(500)),
    );

    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_within",
        ArrayScalar(Polygon(10), Polygon(10)),
    );
    benchmark::scalar(
        c,
        &f,
        "geos",
        "st_within",
        ArrayScalar(Polygon(10), Polygon(500)),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
