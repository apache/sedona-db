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
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*};

fn criterion_benchmark(c: &mut Criterion) {
    let mut f = sedona_functions::register::default_function_set();
    for (name, kernel) in sedona_geoarrow_c::register::scalar_kernels() {
        f.add_scalar_udf_impl(name, kernel).unwrap();
    }

    let st_asbinary: ScalarUDF = f.scalar_udf("st_asbinary").unwrap().clone().into();
    let st_astext: ScalarUDF = f.scalar_udf("st_astext").unwrap().clone().into();

    benchmark::scalar(c, &f, "geoarrow_c", "st_astext", Point);
    benchmark::scalar(c, &f, "geoarrow_c", "st_astext", LineString(10));
    benchmark::scalar(
        c,
        &f,
        "geoarrow_c",
        "st_geomfromwkb",
        Transformed(Point.into(), st_asbinary.clone()),
    );
    benchmark::scalar(
        c,
        &f,
        "geoarrow_c",
        "st_geomfromwkt",
        Transformed(Point.into(), st_astext.clone()),
    );
    benchmark::scalar(
        c,
        &f,
        "geoarrow_c",
        "st_geomfromwkt",
        Transformed(LineString(10).into(), st_astext.clone()),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
