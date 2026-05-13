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

//! Benchmarks for RS_FromPath UDF.
//!
//! RS_FromPath creates out-db rasters from file paths, so these benchmarks use
//! raster fixtures from the `sedona-testing` test module rather than synthetic input.

use std::{hint::black_box, sync::Arc};

use arrow_array::{ArrayRef, StringArray};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion_expr::ScalarUDF;
use sedona_schema::datatypes::SedonaType;
use sedona_testing::{data::test_raster, testers::ScalarUdfTester};

const SMALL_RASTER_FIXTURES: &[&str] = &[
    "test1.tiff",
    "test2.tif",
    "test3.tif",
    "test4.tiff",
    "test5.tiff",
];

fn raster_path_array(names: &[&str], rows: usize) -> ArrayRef {
    assert!(
        !names.is_empty(),
        "benchmark fixture list must not be empty"
    );

    let paths = names
        .iter()
        .map(|name| test_raster(name).unwrap())
        .collect::<Vec<_>>();

    let values = (0..rows)
        .map(|index| paths[index % paths.len()].as_str())
        .collect::<Vec<_>>();

    Arc::new(StringArray::from(values))
}

fn bench_rs_frompath(c: &mut Criterion) {
    let udf: ScalarUDF = sedona_raster_gdal::rs_frompath_udf().into();
    let tester = ScalarUdfTester::new(udf, vec![SedonaType::Arrow(DataType::Utf8)]);

    let single_small = raster_path_array(&["test4.tiff"], 1);
    let mixed_small = raster_path_array(SMALL_RASTER_FIXTURES, SMALL_RASTER_FIXTURES.len());
    let batched_small = raster_path_array(SMALL_RASTER_FIXTURES, 256);

    let mut group = c.benchmark_group("rs_frompath");

    group.throughput(Throughput::Elements(single_small.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("fixtures", "single_small"),
        &single_small,
        |b, input| b.iter(|| black_box(tester.invoke_array(input.clone()).unwrap())),
    );

    group.throughput(Throughput::Elements(mixed_small.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("fixtures", "mixed_small"),
        &mixed_small,
        |b, input| b.iter(|| black_box(tester.invoke_array(input.clone()).unwrap())),
    );

    group.throughput(Throughput::Elements(batched_small.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("fixtures", "batched_small"),
        &batched_small,
        |b, input| b.iter(|| black_box(tester.invoke_array(input.clone()).unwrap())),
    );

    group.finish();
}

criterion_group!(benches, bench_rs_frompath);
criterion_main!(benches);
