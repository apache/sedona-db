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

//! Benchmarks for RS_MapAlgebra UDF
//!
//! RS_MapAlgebra applies mathematical expressions to raster pixels.
//! This benchmark covers:
//! - Different expression complexities
//! - Single vs multiple output bands
//! - Different raster sizes
//! - Different output pixel types

mod bench_common;

use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use datafusion_expr::ScalarUDF;
use sedona_raster_gdal::rs_map_algebra_udf;
use sedona_schema::datatypes::{SedonaType, RASTER};

use bench_common::*;

fn bench_rs_map_algebra_expressions(c: &mut Criterion) {
    let udf: ScalarUDF = rs_map_algebra_udf().into();

    // Arg types: (raster, pixel_type, expr)
    let arg_types = vec![
        RASTER,
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Utf8),
    ];

    let mut group = c.benchmark_group("rs_map_algebra_expressions");
    group.sample_size(20);

    let raster = load_raster_as_scalar("test4.tiff");
    let pixel_type = string_scalar("D"); // Float64 output

    // Simple expression: identity
    let expr_identity = string_scalar("rast0");
    group.bench_with_input(
        BenchmarkId::new("expr", "identity"),
        &(&expr_identity, &arg_types),
        |b, (expr, arg_types)| {
            b.iter(|| {
                let args = vec![raster.clone(), pixel_type.clone(), (*expr).clone()];
                invoke_udf(&udf, &args, arg_types).unwrap()
            })
        },
    );

    // Simple arithmetic
    let expr_arithmetic = string_scalar("rast0 * 2 + 1");
    group.bench_with_input(
        BenchmarkId::new("expr", "arithmetic"),
        &(&expr_arithmetic, &arg_types),
        |b, (expr, arg_types)| {
            b.iter(|| {
                let args = vec![raster.clone(), pixel_type.clone(), (*expr).clone()];
                invoke_udf(&udf, &args, arg_types).unwrap()
            })
        },
    );

    // More complex arithmetic (single band, but exercises the expression parser)
    let expr_complex_arith = string_scalar("(rast0 * 0.5 + 128) / 2.0 - rast0 * 0.1");
    group.bench_with_input(
        BenchmarkId::new("expr", "complex_arithmetic"),
        &(&expr_complex_arith, &arg_types),
        |b, (expr, arg_types)| {
            b.iter(|| {
                let args = vec![raster.clone(), pixel_type.clone(), (*expr).clone()];
                invoke_udf(&udf, &args, arg_types).unwrap()
            })
        },
    );

    // Using position variables (x, y) in expressions
    let expr_position = string_scalar("rast0 + x + y");
    group.bench_with_input(
        BenchmarkId::new("expr", "with_position"),
        &(&expr_position, &arg_types),
        |b, (expr, arg_types)| {
            b.iter(|| {
                let args = vec![raster.clone(), pixel_type.clone(), (*expr).clone()];
                invoke_udf(&udf, &args, arg_types).unwrap()
            })
        },
    );

    // Using width/height constants in expressions
    let expr_normalized = string_scalar("rast0 / 255.0 * (width + height)");
    group.bench_with_input(
        BenchmarkId::new("expr", "with_dimensions"),
        &(&expr_normalized, &arg_types),
        |b, (expr, arg_types)| {
            b.iter(|| {
                let args = vec![raster.clone(), pixel_type.clone(), (*expr).clone()];
                invoke_udf(&udf, &args, arg_types).unwrap()
            })
        },
    );

    group.finish();
}

fn bench_rs_map_algebra_pixel_types(c: &mut Criterion) {
    let udf: ScalarUDF = rs_map_algebra_udf().into();

    // Arg types: (raster, pixel_type, expr)
    let arg_types = vec![
        RASTER,
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Utf8),
    ];

    let mut group = c.benchmark_group("rs_map_algebra_pixel_types");
    group.sample_size(20);

    let raster = load_raster_as_scalar("test4.tiff");
    let expr = string_scalar("rast0 * 2");

    // Different output pixel types
    for (pixel_type, type_name) in &[
        ("B", "UInt8"),
        ("S", "Int16"),
        ("I", "Int32"),
        ("F", "Float32"),
        ("D", "Float64"),
    ] {
        let pixel_type_scalar = string_scalar(pixel_type);

        group.bench_with_input(
            BenchmarkId::new("output_type", type_name),
            &(&pixel_type_scalar, &arg_types),
            |b, (pt, arg_types)| {
                b.iter(|| {
                    let args = vec![raster.clone(), (*pt).clone(), expr.clone()];
                    invoke_udf(&udf, &args, arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_rs_map_algebra_with_nodata(c: &mut Criterion) {
    let udf: ScalarUDF = rs_map_algebra_udf().into();

    // Different arg types for each signature
    let arg_types_basic = vec![
        RASTER,
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Utf8),
    ];
    let arg_types_nodata = vec![
        RASTER,
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Float64),
    ];
    let arg_types_bands = vec![
        RASTER,
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Float64),
        SedonaType::Arrow(DataType::Int32),
    ];

    let mut group = c.benchmark_group("rs_map_algebra_with_nodata");
    group.sample_size(20);

    let raster = load_raster_as_scalar("test4.tiff");
    let pixel_type = string_scalar("D");
    let expr = string_scalar("rast0 * 2 + 1");

    // Without nodata
    group.bench_function("without_nodata", |b| {
        b.iter(|| {
            let args = vec![raster.clone(), pixel_type.clone(), expr.clone()];
            invoke_udf(&udf, &args, &arg_types_basic).unwrap()
        })
    });

    // With nodata
    let nodata = float64_scalar(-9999.0);
    group.bench_function("with_nodata", |b| {
        b.iter(|| {
            let args = vec![
                raster.clone(),
                pixel_type.clone(),
                expr.clone(),
                nodata.clone(),
            ];
            invoke_udf(&udf, &args, &arg_types_nodata).unwrap()
        })
    });

    // With nodata and multiple output bands
    let num_bands = int32_scalar(3);
    group.bench_function("multi_band_output", |b| {
        b.iter(|| {
            let args = vec![
                raster.clone(),
                pixel_type.clone(),
                expr.clone(),
                nodata.clone(),
                num_bands.clone(),
            ];
            invoke_udf(&udf, &args, &arg_types_bands).unwrap()
        })
    });

    group.finish();
}

fn bench_rs_map_algebra_raster_sizes(c: &mut Criterion) {
    let udf: ScalarUDF = rs_map_algebra_udf().into();

    // Arg types: (raster, pixel_type, expr)
    let arg_types = vec![
        RASTER,
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Utf8),
    ];

    let mut group = c.benchmark_group("rs_map_algebra_raster_sizes");
    group.sample_size(20);

    let pixel_type = string_scalar("D");
    let expr = string_scalar("rast0 * 2 + 1");

    for raster_name in &["test1.tiff", "test4.tiff"] {
        let raster = load_raster_as_scalar(raster_name);

        group.bench_with_input(
            BenchmarkId::new("raster", raster_name),
            &(&raster, &arg_types),
            |b, (raster, arg_types)| {
                b.iter(|| {
                    let args = vec![(*raster).clone(), pixel_type.clone(), expr.clone()];
                    invoke_udf(&udf, &args, arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_rs_map_algebra_batch(c: &mut Criterion) {
    let udf: ScalarUDF = rs_map_algebra_udf().into();

    // Arg types: (raster, pixel_type, expr)
    let arg_types = vec![
        RASTER,
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Utf8),
    ];

    let mut group = c.benchmark_group("rs_map_algebra_batch");
    group.sample_size(10);

    let pixel_type = string_scalar("D");
    let expr = string_scalar("rast0 * 2");

    for batch_size in &[5, 10, 20] {
        let rasters = load_rasters_as_array("test4.tiff", *batch_size);

        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &(&rasters, &arg_types),
            |b, (rasters, arg_types)| {
                b.iter(|| {
                    let args = vec![(*rasters).clone(), pixel_type.clone(), expr.clone()];
                    invoke_udf(&udf, &args, arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rs_map_algebra_expressions,
    bench_rs_map_algebra_pixel_types,
    bench_rs_map_algebra_with_nodata,
    bench_rs_map_algebra_raster_sizes,
    bench_rs_map_algebra_batch
);
criterion_main!(benches);
