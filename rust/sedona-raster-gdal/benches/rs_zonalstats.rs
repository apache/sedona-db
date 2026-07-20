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

//! Benchmarks for the RS_ZonalStats / RS_ZonalStatsAll UDFs.
//!
//! Both functions rasterize the zone geometry into a mask, walk the masked
//! window collecting the selected pixel values, and reduce them to statistics.
//!
//! Each case builds a raster whose world extent is exactly the zone-polygon
//! generator's `[-10, 10]²` bounds at the requested resolution, so every
//! generated polygon lands on the raster and the full mask/collect/reduce path
//! runs. `all_touched = true` (passed via the JSON options) guarantees a
//! polygon smaller than a cell still burns at least one pixel rather than
//! hitting the empty-zone early return.
//!
//! Axes:
//! - **Raster resolution** (`64²`, `256²`, `1024²`) with a small polygon:
//!   rasterization + window scan dominate.
//! - **Zone polygon complexity** (vertex count) at a fixed resolution, driving
//!   the GDAL rasterization cost.
//! - **Large zone**: a polygon covering most of the raster, so collecting the
//!   masked values and the reduction (sort for median, frequency map for mode)
//!   dominate — the `RS_ZonalStatsAll` case is the heaviest since it computes
//!   every statistic.
//!
//! Numerical correctness against a reference (rasterio / numpy) is pinned by
//! the Python parity tests, not here; this bench only measures throughput.

use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, StringArray};
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_expr::ScalarUDF;
use sedona_schema::datatypes::{SedonaType, RASTER, WKB_GEOMETRY};
use sedona_testing::{
    benchmark_util::BenchmarkArgSpec, create::make_wkb, raster_spec::RasterSpec,
    testers::ScalarUdfTester,
};

fn criterion_benchmark(c: &mut Criterion) {
    let f = sedona_raster_gdal::register::default_function_set();
    let stats_udf: ScalarUDF = f
        .scalar_udf("rs_zonalstats")
        .expect("rs_zonalstats is registered")
        .clone()
        .into();
    let stats_all_udf: ScalarUDF = f
        .scalar_udf("rs_zonalstatsall")
        .expect("rs_zonalstatsall is registered")
        .clone()
        .into();

    // RS_ZonalStats(raster, zone, stat, options) and
    // RS_ZonalStatsAll(raster, zone, options).
    let stats_tester = ScalarUdfTester::new(
        stats_udf,
        vec![
            RASTER,
            WKB_GEOMETRY,
            SedonaType::Arrow(arrow_schema::DataType::Utf8),
            SedonaType::Arrow(arrow_schema::DataType::Utf8),
        ],
    );
    let stats_all_tester = ScalarUdfTester::new(
        stats_all_udf,
        vec![
            RASTER,
            WKB_GEOMETRY,
            SedonaType::Arrow(arrow_schema::DataType::Utf8),
        ],
    );

    let mean_stat: ArrayRef = Arc::new(StringArray::from(vec!["mean"]));
    let all_touched_opts: ArrayRef = Arc::new(StringArray::from(vec![
        r#"{"band": 1, "all_touched": true}"#,
    ]));

    // A north-up raster covering exactly the polygon generator's [-10, 10]²
    // bounds at the requested resolution, so every generated polygon overlaps.
    let build_raster = |w: i64, h: i64| -> ArrayRef {
        let transform = [-10.0, 20.0 / w as f64, 0.0, 10.0, 0.0, -20.0 / h as f64];
        let values: Vec<f64> = (0..(w * h)).map(|v| v as f64).collect();
        Arc::new(
            RasterSpec::d2(w, h)
                .band_values(&values)
                .crs(None)
                .transform(transform)
                .build(),
        )
    };

    let gen_polygon = |vertices: usize| -> ArrayRef {
        BenchmarkArgSpec::Polygon(vertices)
            .build_arrays(0, 1, 1)
            .expect("build zone polygon")
            .remove(0)
    };

    let run_single = |c: &mut Criterion, label: &str, raster: ArrayRef, geom: ArrayRef| {
        c.bench_function(label, |b| {
            b.iter(|| {
                stats_tester
                    .invoke_arrays(vec![
                        raster.clone(),
                        geom.clone(),
                        mean_stat.clone(),
                        all_touched_opts.clone(),
                    ])
                    .unwrap()
            })
        });
    };

    let run_all = |c: &mut Criterion, label: &str, raster: ArrayRef, geom: ArrayRef| {
        c.bench_function(label, |b| {
            b.iter(|| {
                stats_all_tester
                    .invoke_arrays(vec![raster.clone(), geom.clone(), all_touched_opts.clone()])
                    .unwrap()
            })
        });
    };

    // Resolution sweep (simple 8-vertex polygon), single-stat mean.
    for (w, h) in [(64i64, 64i64), (256, 256), (1024, 1024)] {
        let label =
            format!("raster-gdal rs_zonalstats ZonalStats(Raster({w}x{h}), Polygon(8), mean)");
        run_single(c, &label, build_raster(w, h), gen_polygon(8));
    }

    // Zone-complexity axis at a fixed 64×64 resolution.
    run_single(
        c,
        "raster-gdal rs_zonalstats ZonalStats(Raster(64x64), Polygon(50), mean)",
        build_raster(64, 64),
        gen_polygon(50),
    );

    // Large zone: the polygon covers nearly the whole raster, so collecting the
    // masked values and the reduction dominate. RS_ZonalStatsAll is the heaviest
    // (median sort + mode frequency map over every selected pixel).
    let big_geom = || -> ArrayRef {
        Arc::new(BinaryArray::from_iter_values([make_wkb(
            "POLYGON ((-9.5 -9.5, 9.5 -9.5, 9.5 9.5, -9.5 9.5, -9.5 -9.5))",
        )
        .as_slice()]))
    };
    run_single(
        c,
        "raster-gdal rs_zonalstats ZonalStats(Raster(1024x1024), Polygon(large), mean)",
        build_raster(1024, 1024),
        big_geom(),
    );
    run_all(
        c,
        "raster-gdal rs_zonalstats ZonalStatsAll(Raster(1024x1024), Polygon(large))",
        build_raster(1024, 1024),
        big_geom(),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
