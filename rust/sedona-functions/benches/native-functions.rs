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
use datafusion_expr::ScalarUDF;
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*, BenchmarkArgs};

fn criterion_benchmark(c: &mut Criterion) {
    let f = sedona_functions::register::default_function_set();

    let st_asbinary: ScalarUDF = f.scalar_udf("st_asbinary").unwrap().clone().into();
    let st_astext: ScalarUDF = f.scalar_udf("st_astext").unwrap().clone().into();

    benchmark::scalar(c, &f, "native", "st_astext", Point);
    benchmark::scalar(c, &f, "native", "st_astext", LineString(10));

    benchmark::scalar(c, &f, "native", "st_dimension", Point);
    benchmark::scalar(c, &f, "native", "st_dimension", LineString(10));

    benchmark::scalar(c, &f, "native", "st_envelope", Point);
    benchmark::scalar(c, &f, "native", "st_envelope", LineString(10));

    benchmark::scalar(c, &f, "native", "st_flipcoordinates", Point);
    benchmark::scalar(c, &f, "native", "st_flipcoordinates", LineString(10));

    benchmark::scalar(c, &f, "native", "st_geometrytype", Point);
    benchmark::scalar(c, &f, "native", "st_geometrytype", LineString(10));

    benchmark::scalar(
        c,
        &f,
        "native",
        "st_geomfromwkb",
        Transformed(Point.into(), st_asbinary.clone()),
    );
    benchmark::scalar(
        c,
        &f,
        "native",
        "st_geomfromwkt",
        Transformed(Point.into(), st_astext.clone()),
    );
    benchmark::scalar(
        c,
        &f,
        "native",
        "st_geomfromwkt",
        Transformed(LineString(10).into(), st_astext.clone()),
    );

    benchmark::scalar(c, &f, "native", "st_hasz", Point);
    benchmark::scalar(c, &f, "native", "st_hasz", LineString(10));

    benchmark::scalar(c, &f, "native", "st_hasm", Point);
    benchmark::scalar(c, &f, "native", "st_hasm", LineString(10));

    benchmark::scalar(c, &f, "native", "st_isempty", Point);
    benchmark::scalar(c, &f, "native", "st_isempty", LineString(10));

    benchmark::scalar(
        c,
        &f,
        "native",
        "st_makeline",
        BenchmarkArgs::ArrayArray(Point, Point),
    );

    benchmark::scalar(
        c,
        &f,
        "native",
        "st_point",
        BenchmarkArgs::ArrayArray(Float64(0.0, 100.0), Float64(0.0, 100.0)),
    );

    benchmark::scalar(
        c,
        &f,
        "native",
        "st_pointz",
        BenchmarkArgs::ArrayArrayArray(
            Float64(0.0, 100.0),
            Float64(0.0, 100.0),
            Float64(0.0, 100.0),
        ),
    );

    benchmark::scalar(
        c,
        &f,
        "native",
        "st_pointm",
        BenchmarkArgs::ArrayArrayArray(
            Float64(0.0, 100.0),
            Float64(0.0, 100.0),
            Float64(0.0, 100.0),
        ),
    );

    benchmark::scalar(
        c,
        &f,
        "native",
        "st_pointzm",
        BenchmarkArgs::ArrayArrayArrayArray(
            Float64(0.0, 100.0),
            Float64(0.0, 100.0),
            Float64(0.0, 100.0),
            Float64(0.0, 100.0),
        ),
    );

    benchmark::scalar(c, &f, "native", "st_x", Point);
    benchmark::scalar(c, &f, "native", "st_y", Point);
    benchmark::scalar(c, &f, "native", "st_z", Point);
    benchmark::scalar(c, &f, "native", "st_m", Point);

    benchmark::scalar(c, &f, "native", "st_xmin", LineString(10));
    benchmark::scalar(c, &f, "native", "st_xmax", LineString(10));
    benchmark::scalar(c, &f, "native", "st_ymin", LineString(10));
    benchmark::scalar(c, &f, "native", "st_ymax", LineString(10));

    benchmark::scalar(c, &f, "native", "st_zmin", LineString(10));
    benchmark::scalar(c, &f, "native", "st_zmax", LineString(10));
    benchmark::scalar(c, &f, "native", "st_mmin", LineString(10));
    benchmark::scalar(c, &f, "native", "st_mmax", LineString(10));

    benchmark::aggregate(c, &f, "native", "st_envelope_aggr", Point);
    benchmark::aggregate(c, &f, "native", "st_envelope_aggr", LineString(10));

    benchmark::aggregate(c, &f, "native", "st_analyze_aggr", Point);
    benchmark::aggregate(c, &f, "native", "st_analyze_aggr", LineString(10));

    benchmark::aggregate(c, &f, "native", "st_collect", Point);
    benchmark::aggregate(c, &f, "native", "st_collect", LineString(10));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
