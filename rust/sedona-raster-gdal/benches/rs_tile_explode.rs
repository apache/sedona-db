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

//! Benchmarks for the RS_TileExplode UDF.
//!
//! RS_TileExplode cuts a raster into a grid of `tile_width` x `tile_height`
//! tiles, copying each tile's pixel window into its own buffer. The hot path is
//! the per-tile window copy (`copy_tile_window`), which memcpys each source row
//! segment into the tile; total work is O(width x height) regardless of tile
//! size, so the cost is dominated by that copy plus per-tile bookkeeping.
//!
//! Axes:
//! - **Raster resolution** (`256²`, `1024²`) at a fixed `256²` tile: how the
//!   copy scales with pixel count (1 tile vs 16 tiles).
//! - **Tile size** at a fixed `1024²` raster (`1024²` = one whole-raster tile,
//!   `256²`, `64²`): smaller tiles mean more per-tile setup for the same total
//!   copy.
//! - **Padding** on the `1024²`/`100²`-tile case, where every edge tile does
//!   extra nodata-fill work.

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int32Array};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_expr::ScalarUDF;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_testing::{raster_spec::RasterSpec, testers::ScalarUdfTester};

fn udf() -> ScalarUDF {
    sedona_raster_gdal::register::default_function_set()
        .scalar_udf("rs_tileexplode")
        .expect("rs_tileexplode is registered")
        .clone()
        .into()
}

/// A north-up single-band UInt8 raster of `w` x `h` pixels.
fn build_raster(w: i64, h: i64) -> ArrayRef {
    let values = vec![1u8; (w * h) as usize];
    Arc::new(
        RasterSpec::d2(w, h)
            .band_values(&values)
            .crs(None)
            .transform([0.0, 1.0, 0.0, h as f64, 0.0, -1.0])
            .build(),
    )
}

fn criterion_benchmark(c: &mut Criterion) {
    let tester3 = ScalarUdfTester::new(
        udf(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ],
    );

    let run = |c: &mut Criterion, label: &str, raster: ArrayRef, tile: i32| {
        let tw: ArrayRef = Arc::new(Int32Array::from(vec![tile]));
        let th: ArrayRef = Arc::new(Int32Array::from(vec![tile]));
        c.bench_function(label, |b| {
            b.iter(|| {
                tester3
                    .invoke_arrays(vec![raster.clone(), tw.clone(), th.clone()])
                    .unwrap()
            })
        });
    };

    // Resolution sweep at a fixed 256x256 tile.
    for (w, h) in [(256i64, 256i64), (1024, 1024)] {
        let label = format!("raster-gdal rs_tileexplode TileExplode(Raster({w}x{h}), 256, 256)");
        run(c, &label, build_raster(w, h), 256);
    }

    // Tile-size sweep at a fixed 1024x1024 raster.
    for tile in [1024i32, 256, 64] {
        let label =
            format!("raster-gdal rs_tileexplode TileExplode(Raster(1024x1024), {tile}, {tile})");
        run(c, &label, build_raster(1024, 1024), tile);
    }

    // Padding: 100x100 tiles over 1024x1024 leaves partial edge tiles that each
    // do the extra nodata-fill work. Uses the (raster, width, height,
    // padWithNoData, noDataVal) overload.
    let tester5 = ScalarUdfTester::new(
        udf(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Float64),
        ],
    );
    let raster = build_raster(1024, 1024);
    let tw: ArrayRef = Arc::new(Int32Array::from(vec![100]));
    let th: ArrayRef = Arc::new(Int32Array::from(vec![100]));
    let pad: ArrayRef = Arc::new(BooleanArray::from(vec![true]));
    let nodata: ArrayRef = Arc::new(Float64Array::from(vec![0.0]));
    c.bench_function(
        "raster-gdal rs_tileexplode TileExplode(Raster(1024x1024), 100, 100, pad)",
        |b| {
            b.iter(|| {
                tester5
                    .invoke_arrays(vec![
                        raster.clone(),
                        tw.clone(),
                        th.clone(),
                        pad.clone(),
                        nodata.clone(),
                    ])
                    .unwrap()
            })
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
