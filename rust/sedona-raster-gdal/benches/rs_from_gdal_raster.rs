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

//! Benchmarks for RS_FromGDALRaster UDF
//!
//! RS_FromGDALRaster parses binary raster data (GeoTIFF, etc.) into in-db rasters.
//! This benchmark covers:
//! - Different raster sizes
//! - Single vs batch parsing
//! - Parsing efficiency

mod bench_common;

use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion_expr::{ColumnarValue, ScalarUDF};
use sedona_raster_gdal::rs_from_gdal_raster_udf;
use sedona_schema::datatypes::SedonaType;

use bench_common::*;

fn load_geotiff_bytes(name: &str) -> Vec<u8> {
    let test_file = sedona_testing::data::test_raster(name).expect("Failed to find test raster");
    std::fs::read(&test_file).expect("Failed to read test raster file")
}

fn bench_rs_from_gdal_raster_single(c: &mut Criterion) {
    let udf: ScalarUDF = rs_from_gdal_raster_udf().into();

    // Arg types: (binary)
    let arg_types = vec![SedonaType::Arrow(DataType::Binary)];

    let mut group = c.benchmark_group("rs_from_gdal_raster_single");

    for raster_name in &["test1.tiff", "test4.tiff"] {
        let bytes = load_geotiff_bytes(raster_name);

        group.throughput(Throughput::Bytes(bytes.len() as u64));

        let binary_scalar =
            ColumnarValue::Scalar(datafusion_common::ScalarValue::Binary(Some(bytes.clone())));

        group.bench_with_input(
            BenchmarkId::new("parse", raster_name),
            &(&binary_scalar, &arg_types),
            |b, (binary, arg_types)| {
                b.iter(|| {
                    let args = vec![(*binary).clone()];
                    invoke_udf(&udf, &args, arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_rs_from_gdal_raster_batch(c: &mut Criterion) {
    let udf: ScalarUDF = rs_from_gdal_raster_udf().into();

    // Arg types: (binary)
    let arg_types = vec![SedonaType::Arrow(DataType::Binary)];

    let mut group = c.benchmark_group("rs_from_gdal_raster_batch");

    for batch_size in &[10, 50, 100] {
        let bytes = load_geotiff_bytes("test4.tiff");
        let total_bytes = bytes.len() * batch_size;

        group.throughput(Throughput::Bytes(total_bytes as u64));

        // Create batch of binary data
        let binary_data: Vec<Option<&[u8]>> =
            (0..*batch_size).map(|_| Some(bytes.as_slice())).collect();
        let binary_array = BinaryArray::from(binary_data);
        let binary_columnar = ColumnarValue::Array(Arc::new(binary_array) as ArrayRef);

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &(&binary_columnar, &arg_types),
            |b, (binary, arg_types)| {
                b.iter(|| {
                    let args = vec![(*binary).clone()];
                    invoke_udf(&udf, &args, arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rs_from_gdal_raster_single,
    bench_rs_from_gdal_raster_batch
);
criterion_main!(benches);
