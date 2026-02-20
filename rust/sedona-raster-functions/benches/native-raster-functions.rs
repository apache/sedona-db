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
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*, BenchmarkArgs};

fn criterion_benchmark(c: &mut Criterion) {
    let f = sedona_raster_functions::register::default_function_set();

    // RS_BandNoDataValue
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_bandnodatavalue",
        BenchmarkArgs::Array(Raster(64, 64)),
    );
    // RS_BandPath
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_bandpath",
        BenchmarkArgs::Array(Raster(64, 64)),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_bandpath",
        BenchmarkArgs::ArrayScalar(Raster(64, 64), Int32(1, 2)),
    );
    // RS_BandPixelType
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_bandpixeltype",
        BenchmarkArgs::Array(Raster(64, 64)),
    );

    benchmark::scalar(c, &f, "native-raster", "rs_convexhull", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_crs", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_envelope", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_georeference", Raster(64, 64));
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_georeference",
        BenchmarkArgs::ArrayScalar(Raster(64, 64), String("ESRI".to_string())),
    );
    benchmark::scalar(c, &f, "native-raster", "rs_height", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_numbands", Raster(64, 64));
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_rastertoworldcoord",
        BenchmarkArgs::ArrayScalarScalar(Raster(64, 64), Int32(0, 63), Int32(0, 63)),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_rastertoworldcoordx",
        BenchmarkArgs::ArrayScalarScalar(Raster(64, 64), Int32(0, 63), Int32(0, 63)),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_rastertoworldcoordy",
        BenchmarkArgs::ArrayScalarScalar(Raster(64, 64), Int32(0, 63), Int32(0, 63)),
    );
    benchmark::scalar(c, &f, "native-raster", "rs_rotation", Raster(64, 64));
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_setcrs",
        BenchmarkArgs::ArrayScalar(Raster(64, 64), String("EPSG:3857".to_string())),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_setsrid",
        BenchmarkArgs::ArrayScalar(Raster(64, 64), Int32(3857, 3858)),
    );
    benchmark::scalar(c, &f, "native-raster", "rs_scalex", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_scaley", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_skewx", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_skewy", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_srid", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_upperleftx", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_upperlefty", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_width", Raster(64, 64));
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_worldtorastercoord",
        BenchmarkArgs::ArrayScalarScalar(
            Raster(64, 64),
            Float64(-45.0, 45.0),
            Float64(-45.0, 45.0),
        ),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_worldtorastercoordx",
        BenchmarkArgs::ArrayScalarScalar(
            Raster(64, 64),
            Float64(-45.0, 45.0),
            Float64(-45.0, 45.0),
        ),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_worldtorastercoordy",
        BenchmarkArgs::ArrayScalarScalar(
            Raster(64, 64),
            Float64(-45.0, 45.0),
            Float64(-45.0, 45.0),
        ),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
