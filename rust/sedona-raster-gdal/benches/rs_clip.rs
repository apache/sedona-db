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
//! Axes:
//! - **Raster resolution** (`64²`, `256²`, `1024²`) with a small polygon: cost
//!   is dominated by O(width × height) mask handling (rasterize + mask scan).
//! - **Clip polygon complexity** (vertex count) at a fixed resolution, driving
//!   the GDAL rasterization cost.
//! - **Large clip**: a polygon covering most of the raster, so the mask/crop
//!   copy (`apply_mask_and_crop`) dominates rather than mask rasterization.

use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, BooleanArray, Int32Array};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_expr::ScalarUDF;
use sedona_schema::datatypes::{SedonaType, RASTER, WKB_GEOMETRY};
use sedona_testing::{
    benchmark_util::BenchmarkArgSpec, create::make_wkb, raster_spec::RasterSpec,
    testers::ScalarUdfTester,
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

    let band: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    let all_touched: ArrayRef = Arc::new(BooleanArray::from(vec![true]));

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

    // `c` is passed in rather than captured, so several cases can share it.
    let run = |c: &mut Criterion, label: &str, raster: ArrayRef, geom: ArrayRef| {
        c.bench_function(label, |b| {
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

    let gen_polygon = |vertices: usize| -> ArrayRef {
        BenchmarkArgSpec::Polygon(vertices)
            .build_arrays(0, 1, 1)
            .expect("build clip polygon")
            .remove(0)
    };

    // Resolution sweep (simple 8-vertex polygon).
    for (w, h) in [(64i64, 64i64), (256, 256), (1024, 1024)] {
        let label = format!("raster-gdal rs_clip Clip(Raster({w}x{h}), Polygon(8))");
        run(c, &label, build_raster(w, h), gen_polygon(8));
    }

    // Polygon-complexity axis at a fixed 64×64 resolution.
    run(
        c,
        "raster-gdal rs_clip Clip(Raster(64x64), Polygon(50))",
        build_raster(64, 64),
        gen_polygon(50),
    );

    // Large clip: the polygon covers nearly the whole raster, so the crop copy
    // (apply_mask_and_crop) dominates — this is where its row memcpy matters.
    let big_geom: ArrayRef = Arc::new(BinaryArray::from_iter_values([make_wkb(
        "POLYGON ((-9.5 -9.5, 9.5 -9.5, 9.5 9.5, -9.5 9.5, -9.5 -9.5))",
    )
    .as_slice()]));
    run(
        c,
        "raster-gdal rs_clip Clip(Raster(1024x1024), Polygon(large))",
        build_raster(1024, 1024),
        big_geom,
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
