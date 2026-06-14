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

//! Benchmarks for rs_geotiff_tiles.

use std::hint::black_box;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion::catalog::TableProvider;
use sedona_gdal::driver::DriverManager;
use sedona_gdal::global::with_global_gdal_api;
use sedona_gdal::raster::types::Buffer;
use tempfile::TempDir;

fn write_test_geotiff(dir: &TempDir, name: &str) -> String {
    let path = dir.path().join(name);
    let path_str = path.to_string_lossy().to_string();
    with_global_gdal_api(|api| {
        let driver = DriverManager::get_driver_by_name(api, "GTiff").unwrap();
        let dataset = driver
            .create_with_band_type::<u8>(&path_str, 10, 10, 1)
            .unwrap();
        dataset
            .set_geo_transform(&[0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .unwrap();
        let band = dataset.rasterband(1).unwrap();
        let mut buffer = Buffer::new((10, 10), (0..100u8).collect::<Vec<_>>());
        band.write((0, 0), (10, 10), &mut buffer).unwrap();
    })
    .unwrap();
    path_str
}

fn provider_schema() -> SchemaRef {
    sedona_raster_gdal::rs_geotiff_tiles::GeoTiffTilesProvider::try_new("/tmp".to_string(), false)
        .unwrap()
        .schema()
}

fn bench_rs_geotiff_tiles(c: &mut Criterion) {
    let tmp = TempDir::new().unwrap();
    let path = write_test_geotiff(&tmp, "bench.tiff");
    let schema = Arc::new(provider_schema());
    let mut group = c.benchmark_group("rs_geotiff_tiles");
    group.throughput(Throughput::Elements(1));
    group.bench_with_input(
        BenchmarkId::new("fixture", "test4.tiff"),
        &path,
        |b, input| {
            b.iter(|| {
                black_box(
                    sedona_raster_gdal::rs_geotiff_tiles::build_batch_for_file(
                        input,
                        (*schema).clone(),
                    )
                    .unwrap(),
                )
            })
        },
    );
    group.finish();
}

criterion_group!(benches, bench_rs_geotiff_tiles);
criterion_main!(benches);
