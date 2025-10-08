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
use geo_traits::to_geo::ToGeoGeometry;
use sedona_geo_generic_alg::algorithm::line_measures::{Euclidean, LengthMeasurableExt};
use sedona_geo_generic_alg::Polygon;

#[path = "utils/wkb_util.rs"]
mod wkb_util;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("perimeter_f32", |bencher| {
        let norway = sedona_testing::fixtures::norway_main::<f32>();
        let polygon = Polygon::new(norway, vec![]);

        bencher.iter(|| {
            criterion::black_box(criterion::black_box(&polygon).perimeter_ext(&Euclidean));
        });
    });

    c.bench_function("perimeter", |bencher| {
        let norway = sedona_testing::fixtures::norway_main::<f64>();
        let polygon = Polygon::new(norway, vec![]);

        bencher.iter(|| {
            criterion::black_box(criterion::black_box(&polygon).perimeter_ext(&Euclidean));
        });
    });

    c.bench_function("perimeter_wkb", |bencher| {
        let norway = sedona_testing::fixtures::norway_main::<f64>();
        let polygon = Polygon::new(norway, vec![]);
        let wkb_bytes = wkb_util::geo_to_wkb(polygon);

        bencher.iter(|| {
            let wkb_geom = wkb::reader::read_wkb(&wkb_bytes).unwrap();
            criterion::black_box(wkb_geom.perimeter_ext(&Euclidean));
        });
    });

    c.bench_function("perimeter_wkb_convert", |bencher| {
        let norway = sedona_testing::fixtures::norway_main::<f64>();
        let polygon = Polygon::new(norway, vec![]);
        let wkb_bytes = wkb_util::geo_to_wkb(polygon);

        bencher.iter(|| {
            let wkb_geom = wkb::reader::read_wkb(&wkb_bytes).unwrap();
            let geom = wkb_geom.to_geometry();
            criterion::black_box(geom.perimeter_ext(&Euclidean));
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
