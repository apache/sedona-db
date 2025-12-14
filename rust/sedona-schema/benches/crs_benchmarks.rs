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
use sedona_schema::crs::deserialize_crs;
use std::hint::black_box;

fn bench_crs_equality(c: &mut Criterion) {
    c.bench_function("equality_lnglat_crs", |b| {
        b.iter(|| {
            for _ in 0..100000 {
                let crs1 = deserialize_crs(black_box("EPSG:4326")).unwrap().unwrap();
                let crs3 = deserialize_crs(black_box("OGC:CRS84")).unwrap().unwrap();
                let eq = crs1.crs_equals(crs3.as_ref());
                black_box(eq);
            }
        })
    });

    c.bench_function("equality_different_crs", |b| {
        b.iter(|| {
            for _ in 0..100000 {
                let crs1 = deserialize_crs(black_box("EPSG:3827")).unwrap().unwrap();
                let crs2 = deserialize_crs(black_box("EPSG:3857")).unwrap().unwrap();
                let eq = crs1.crs_equals(crs2.as_ref());
                black_box(eq);
            }
        })
    });
}

criterion_group!(benches, bench_crs_equality);
criterion_main!(benches);
