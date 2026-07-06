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

//! Benchmarks for the RS_Clip UDF.
//!
//! RS_Clip rasterizes the clip geometry into a mask, sets pixels outside it to
//! nodata, and (by default) crops to the geometry's bounding box.
//!
//! Each case builds a raster whose world extent is exactly the clip-polygon
//! generator's `[-10, 10]²` bounds at the requested resolution, so every
//! generated polygon lands on the raster and the full mask/crop path runs.
//! `all_touched = true` guarantees an overlapping polygon burns at least one
//! pixel even when it is smaller than a cell (otherwise a sub-pixel polygon
//! would produce an empty mask and hit the no-intersection early return — which
//! is what an earlier version of this benchmark accidentally measured, because
//! it placed the rasters far outside the polygons entirely).
//!
//! Two axes are swept:
//! - **Raster resolution** (`64²`, `256²`, `1024²`): per-clip cost is dominated
//!   by O(width × height) mask handling (rasterize + mask scan + crop window).
//! - **Clip polygon complexity** (vertex count) at a fixed resolution, which
//!   drives the GDAL rasterization cost.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Int32Array};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_expr::ScalarUDF;
use sedona_schema::datatypes::{SedonaType, RASTER, WKB_GEOMETRY};
use sedona_testing::{
    benchmark_util::BenchmarkArgSpec, raster_spec::RasterSpec, testers::ScalarUdfTester,
};

fn criterion_benchmark(c: &mut Criterion) {
    let f = sedona_raster_gdal::register::default_function_set();
    let udf: ScalarUDF = f
        .scalar_udf("rs_clip")
        .expect("rs_clip is registered")
        .clone()
        .into();

    // RS_Clip(raster, band, geom, all_touched).
    let tester = ScalarUdfTester::new(
        udf,
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            WKB_GEOMETRY,
            SedonaType::Arrow(DataType::Boolean),
        ],
    );

    // A north-up raster covering exactly the polygon generator's [-10, 10]²
    // bounds at the requested resolution, so every generated polygon overlaps.
    let build_raster = |w: i64, h: i64| -> ArrayRef {
        let transform = [-10.0, 20.0 / w as f64, 0.0, 10.0, 0.0, -20.0 / h as f64];
        let values = vec![1u8; (w * h) as usize];
        Arc::new(
            RasterSpec::d2(w, h)
                .band_values(&values)
                .crs(None)
                .transform(transform)
                .build(),
        )
    };

    let band: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    let all_touched: ArrayRef = Arc::new(BooleanArray::from(vec![true]));

    let mut bench_clip = |w: i64, h: i64, vertices: usize| {
        let raster = build_raster(w, h);
        let geom = BenchmarkArgSpec::Polygon(vertices)
            .build_arrays(0, 1, 1)
            .expect("build clip polygon")
            .remove(0);
        let label = format!("raster-gdal rs_clip Clip(Raster({w}x{h}), Polygon({vertices}))");
        c.bench_function(&label, |b| {
            b.iter(|| {
                tester
                    .invoke_arrays(vec![
                        raster.clone(),
                        band.clone(),
                        geom.clone(),
                        all_touched.clone(),
                    ])
                    .unwrap()
            })
        });
    };

    // Resolution sweep (simple 8-vertex polygon).
    bench_clip(64, 64, 8);
    bench_clip(256, 256, 8);
    bench_clip(1024, 1024, 8);

    // Polygon-complexity axis at a fixed 64×64 resolution.
    bench_clip(64, 64, 50);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
