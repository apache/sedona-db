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

//! Benchmarks for RS_Value UDF
//!
//! RS_Value extracts pixel values from raster at specified coordinates.
//! This benchmark covers:
//! - Grid coordinate (col/row) based lookup with different positions
//! - Different raster sizes
//! - Batch processing

mod bench_common;

use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion_expr::ScalarUDF;
use sedona_raster_gdal::rs_value_udf;
use sedona_schema::datatypes::{SedonaType, RASTER};

use bench_common::*;

fn bench_rs_value_grid(c: &mut Criterion) {
    let udf: ScalarUDF = rs_value_udf().into();

    // Arg types for grid-based RS_Value: (raster, col_x, row_y, band)
    let arg_types = vec![
        RASTER,
        SedonaType::Arrow(DataType::Int32),
        SedonaType::Arrow(DataType::Int32),
        SedonaType::Arrow(DataType::Int32),
    ];

    let mut group = c.benchmark_group("rs_value_grid");

    for raster_name in &["test1.tiff", "test4.tiff"] {
        let raster = load_raster_as_scalar(raster_name);

        // Test at pixel (0, 0)
        let col_x = int32_scalar(0);
        let row_y = int32_scalar(0);
        let band = int32_scalar(1);

        group.bench_with_input(
            BenchmarkId::new("pos_0_0", raster_name),
            &(&raster, &col_x, &row_y, &band, &arg_types),
            |b, (raster, col_x, row_y, band, arg_types)| {
                b.iter(|| {
                    let args = vec![
                        (*raster).clone(),
                        (*col_x).clone(),
                        (*row_y).clone(),
                        (*band).clone(),
                    ];
                    invoke_udf(&udf, &args, arg_types).unwrap()
                })
            },
        );

        // Test at a different position
        let col_x = int32_scalar(5);
        let row_y = int32_scalar(5);
        group.bench_with_input(
            BenchmarkId::new("pos_5_5", raster_name),
            &(&raster, &col_x, &row_y, &band, &arg_types),
            |b, (raster, col_x, row_y, band, arg_types)| {
                b.iter(|| {
                    let args = vec![
                        (*raster).clone(),
                        (*col_x).clone(),
                        (*row_y).clone(),
                        (*band).clone(),
                    ];
                    invoke_udf(&udf, &args, arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_rs_value_bands(c: &mut Criterion) {
    let udf: ScalarUDF = rs_value_udf().into();

    // Arg types for grid-based RS_Value: (raster, col_x, row_y, band)
    let arg_types = vec![
        RASTER,
        SedonaType::Arrow(DataType::Int32),
        SedonaType::Arrow(DataType::Int32),
        SedonaType::Arrow(DataType::Int32),
    ];

    let mut group = c.benchmark_group("rs_value_bands");

    let raster = load_raster_as_scalar("test4.tiff");
    let col_x = int32_scalar(0);
    let row_y = int32_scalar(0);

    // Different band numbers
    for band_num in &[1, 2, 3] {
        let band = int32_scalar(*band_num);

        group.bench_with_input(
            BenchmarkId::new("band", band_num),
            &(&raster, &col_x, &row_y, &band, &arg_types),
            |b, (raster, col_x, row_y, band, arg_types)| {
                b.iter(|| {
                    let args = vec![
                        (*raster).clone(),
                        (*col_x).clone(),
                        (*row_y).clone(),
                        (*band).clone(),
                    ];
                    invoke_udf(&udf, &args, arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_rs_value_batch(c: &mut Criterion) {
    let udf: ScalarUDF = rs_value_udf().into();

    // Arg types for grid-based RS_Value: (raster, col_x, row_y, band)
    let arg_types = vec![
        RASTER,
        SedonaType::Arrow(DataType::Int32),
        SedonaType::Arrow(DataType::Int32),
        SedonaType::Arrow(DataType::Int32),
    ];

    let mut group = c.benchmark_group("rs_value_batch");

    // Batch processing: multiple rasters with same query position
    for batch_size in &[10, 50, 100] {
        let rasters = load_rasters_as_array("test4.tiff", *batch_size);
        let col_x = int32_scalar(0);
        let row_y = int32_scalar(0);
        let band = int32_scalar(1);

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &(&rasters, &arg_types),
            |b, (rasters, arg_types)| {
                b.iter(|| {
                    let args = vec![
                        (*rasters).clone(),
                        col_x.clone(),
                        row_y.clone(),
                        band.clone(),
                    ];
                    invoke_udf(&udf, &args, arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rs_value_grid,
    bench_rs_value_bands,
    bench_rs_value_batch
);
criterion_main!(benches);
