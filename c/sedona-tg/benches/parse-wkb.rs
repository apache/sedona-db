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
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use geo::{LineString, Point};
use sedona_tg::tg::Geom;

fn criterion_benchmark(c: &mut Criterion) {
    let points = (0..10000)
        .map(|i| Point::new(i as f64, (i + 1) as f64))
        .collect::<Vec<_>>();
    let large_geom = LineString::from(points);
    let mut large_geom_wkb_big_endian = Vec::new();
    wkb::writer::write_geometry(
        &mut large_geom_wkb_big_endian,
        &large_geom,
        &wkb::writer::WriteOptions {
            endianness: wkb::Endianness::BigEndian,
        },
    )
    .unwrap();
    let mut large_geom_wkb_little_endian = Vec::new();
    wkb::writer::write_geometry(
        &mut large_geom_wkb_little_endian,
        &large_geom,
        &wkb::writer::WriteOptions {
            endianness: wkb::Endianness::LittleEndian,
        },
    )
    .unwrap();

    c.bench_function("parse_wkb_big_endian", |b| {
        b.iter(|| {
            let result = Geom::parse_wkb(
                &large_geom_wkb_big_endian,
                sedona_tg::tg::IndexType::Unindexed,
            );
            black_box(result)
        })
    });

    c.bench_function("parse_wkb_little_endian", |b| {
        b.iter(|| {
            let result = Geom::parse_wkb(
                &large_geom_wkb_little_endian,
                sedona_tg::tg::IndexType::Unindexed,
            );
            black_box(result)
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
