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

//! Benchmarks for the RS_FromGDALRaster UDF (GeoTIFF bytes → in-db raster).

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion_expr::ScalarUDF;
use sedona_schema::datatypes::SedonaType;
use sedona_testing::{data::test_raster, testers::ScalarUdfTester};

/// Read a fixture GeoTIFF's bytes and replicate them across `rows` (mirrors the
/// fixture-driven inputs used by the rs_frompath / rs_metadata benchmarks).
fn geotiff_bytes_array(name: &str, rows: usize) -> ArrayRef {
    assert!(rows > 0, "benchmark rows must be positive");
    let path = test_raster(name).unwrap();
    let bytes = std::fs::read(path).unwrap();
    Arc::new(BinaryArray::from(vec![bytes.as_slice(); rows]))
}

fn bench_rs_from_gdal_raster(c: &mut Criterion) {
    let udf: ScalarUDF = sedona_raster_gdal::rs_from_gdal_raster_udf().into();
    let tester = ScalarUdfTester::new(udf, vec![SedonaType::Arrow(DataType::Binary)]);

    let mut group = c.benchmark_group("rs_from_gdal_raster");
    for rows in [1usize, 32] {
        let input = geotiff_bytes_array("test4.tiff", rows);
        group.throughput(Throughput::Elements(rows as u64));
        group.bench_with_input(BenchmarkId::new("decode", rows), &input, |b, input| {
            b.iter(|| black_box(tester.invoke_arrays(vec![input.clone()]).unwrap()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rs_from_gdal_raster);
criterion_main!(benches);
