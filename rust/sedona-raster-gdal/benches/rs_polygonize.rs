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

//! Benchmarks for RS_Polygonize UDF.

use std::{hint::black_box, sync::Arc};

use arrow_array::{ArrayRef, Int32Array};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion_expr::ColumnarValue;
use datafusion_expr::ScalarUDF;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_testing::testers::ScalarUdfTester;

fn raster_array(rows: usize) -> ArrayRef {
    assert!(rows > 0, "benchmark rows must be positive");

    let example: ScalarUDF = sedona_raster_functions::rs_example::rs_example_udf().into();
    let tester = ScalarUdfTester::new(example, vec![]);
    let scalar = tester.invoke(vec![]).unwrap();
    match scalar {
        ColumnarValue::Scalar(value) => value.to_array_of_size(rows).unwrap(),
        ColumnarValue::Array(array) => array,
    }
}

fn band_array(rows: usize) -> ArrayRef {
    Arc::new(Int32Array::from(vec![1; rows]))
}

fn bench_rs_polygonize(c: &mut Criterion) {
    let udf: ScalarUDF = sedona_raster_gdal::rs_polygonize_udf().into();
    let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

    let single = raster_array(1);
    let batched = raster_array(32);
    let single_band = band_array(single.len());
    let batched_band = band_array(batched.len());

    let mut group = c.benchmark_group("rs_polygonize");

    group.throughput(Throughput::Elements(single.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("fixtures", "single_example"),
        &single,
        |b, input| {
            b.iter(|| {
                black_box(
                    tester
                        .invoke_arrays(vec![input.clone(), single_band.clone()])
                        .unwrap(),
                )
            })
        },
    );

    group.throughput(Throughput::Elements(batched.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("fixtures", "batched_example"),
        &batched,
        |b, input| {
            b.iter(|| {
                black_box(
                    tester
                        .invoke_arrays(vec![input.clone(), batched_band.clone()])
                        .unwrap(),
                )
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_rs_polygonize);
criterion_main!(benches);
