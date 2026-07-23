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

//! Benchmarks for the RS_Resample UDF.
//!
//! RS_Resample reads the full source raster through GDAL's RasterIO resampling
//! into a new grid. The cost is dominated by the resampling kernel over the
//! larger of the source/output pixel counts, so the axes are:
//!
//! - **Resolution sweep** downsampling 2:1 with nearest neighbour, over
//!   `256²`, `512²`, `1024²` sources — the kernel scans O(source pixels).
//! - **Upsample 1:2** with nearest neighbour, where the output pixel count
//!   dominates.
//! - **Bilinear vs nearest** at a fixed resolution, isolating the kernel cost.
//!
//! The ground-truth comparison for correctness lives in the Python parity test
//! (`test_rs_resample.py`), which cross-checks these same operations against
//! rasterio's `Dataset.read(out_shape=..., resampling=...)` — the identical
//! GDAL RasterIO path — so nearest is bit-exact and bilinear matches within a
//! float tolerance.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Float64Array, StringArray};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_expr::ScalarUDF;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_testing::{raster_spec::RasterSpec, testers::ScalarUdfTester};

fn criterion_benchmark(c: &mut Criterion) {
    let f = sedona_raster_gdal::register::default_function_set();
    let udf: ScalarUDF = f
        .scalar_udf("rs_resample")
        .expect("rs_resample is registered")
        .clone()
        .into();

    // The 5-argument dimension overload:
    // RS_Resample(raster, widthOrScale, heightOrScale, useScale, algorithm).
    let tester = ScalarUdfTester::new(
        udf,
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Float64),
            SedonaType::Arrow(DataType::Float64),
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Utf8),
        ],
    );

    // A north-up single-band raster covering [0, size]² at unit pixels.
    let build_raster = |size: i64| -> ArrayRef {
        let values = vec![1u8; (size * size) as usize];
        Arc::new(
            RasterSpec::d2(size, size)
                .band_values(&values)
                .crs(None)
                .bbox(0.0, 0.0, size as f64, size as f64)
                .build(),
        )
    };

    // Dimension mode (useScale=false), so widthOrScale/heightOrScale are the
    // output pixel dimensions.
    let run =
        |c: &mut Criterion, label: &str, raster: ArrayRef, width: f64, height: f64, alg: &str| {
            let width: ArrayRef = Arc::new(Float64Array::from(vec![width]));
            let height: ArrayRef = Arc::new(Float64Array::from(vec![height]));
            let use_scale: ArrayRef = Arc::new(BooleanArray::from(vec![false]));
            let algorithm: ArrayRef = Arc::new(StringArray::from(vec![alg.to_string()]));
            c.bench_function(label, |b| {
                b.iter(|| {
                    tester
                        .invoke_arrays(vec![
                            raster.clone(),
                            width.clone(),
                            height.clone(),
                            use_scale.clone(),
                            algorithm.clone(),
                        ])
                        .unwrap()
                })
            });
        };

    // Resolution sweep: 2:1 nearest downsample.
    for size in [256i64, 512, 1024] {
        let half = (size / 2) as f64;
        let label = format!("raster-gdal rs_resample Downsample2x(Nearest, Raster({size}x{size}))");
        run(c, &label, build_raster(size), half, half, "NearestNeighbor");
    }

    // 1:2 nearest upsample: output pixel count dominates.
    run(
        c,
        "raster-gdal rs_resample Upsample2x(Nearest, Raster(512x512))",
        build_raster(512),
        1024.0,
        1024.0,
        "NearestNeighbor",
    );

    // Bilinear vs nearest at a fixed resolution isolates the kernel cost.
    run(
        c,
        "raster-gdal rs_resample Downsample2x(Bilinear, Raster(1024x1024))",
        build_raster(1024),
        512.0,
        512.0,
        "Bilinear",
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
