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

//! Benchmarks for the RS_MapAlgebra UDF.
//!
//! RS_MapAlgebra evaluates a per-pixel expression over a raster's bands. Each
//! case runs the expression `rast0 * 2 + 1` over a Float64 raster at a few
//! resolutions.
//!
//! Alongside the UDF, a `native-baseline` case computes the identical result
//! with a plain Rust loop over the band's `f64` values — the ground-truth lower
//! bound a native per-pixel implementation would approach. The gap between the
//! two is the cost of the general-purpose expression engine (compiling the
//! expression per row and re-binding string-keyed variables per pixel), which
//! is what a native implementation would remove.

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, StringArray};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_expr::ScalarUDF;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_testing::{raster_spec::RasterSpec, testers::ScalarUdfTester};

const EXPR: &str = "rast0 * 2 + 1";

fn criterion_benchmark(c: &mut Criterion) {
    let f = sedona_raster_gdal::register::default_function_set();
    let udf: ScalarUDF = f
        .scalar_udf("rs_mapalgebra")
        .expect("rs_mapalgebra is registered")
        .clone()
        .into();

    // RS_MapAlgebra(raster, expr, options).
    let tester = ScalarUdfTester::new(
        udf,
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Utf8),
        ],
    );

    let expr: ArrayRef = Arc::new(StringArray::from(vec![EXPR]));
    let options: ArrayRef = Arc::new(StringArray::from(vec![Some(r#"{"pixel_type": "D"}"#)]));

    // A north-up Float64 raster with a simple ramp of pixel values.
    let build = |w: i64, h: i64| -> (ArrayRef, Vec<f64>) {
        let values: Vec<f64> = (0..(w * h)).map(|i| i as f64).collect();
        let raster: ArrayRef =
            Arc::new(RasterSpec::d2(w, h).crs(None).band_values(&values).build());
        (raster, values)
    };

    for (w, h) in [(64i64, 64i64), (256, 256), (512, 512)] {
        let (raster, values) = build(w, h);

        c.bench_function(
            &format!("raster-gdal rs_map_algebra MapAlgebra(Raster({w}x{h}), '{EXPR}')"),
            |b| {
                b.iter(|| {
                    tester
                        .invoke_arrays(vec![raster.clone(), expr.clone(), options.clone()])
                        .unwrap()
                })
            },
        );

        // Ground-truth baseline: the same arithmetic with a plain Rust loop,
        // producing the Float64 output bytes directly.
        c.bench_function(
            &format!("raster-gdal rs_map_algebra native-baseline({w}x{h})"),
            |b| {
                b.iter(|| {
                    let mut out = Vec::with_capacity(values.len() * 8);
                    for &v in black_box(&values) {
                        out.extend_from_slice(&(v * 2.0 + 1.0).to_le_bytes());
                    }
                    black_box(out)
                })
            },
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
