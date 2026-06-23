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
//! nodata, and (by default) crops to the geometry's bounding box. The geometry
//! is reprojected into the raster CRS first.
//!
//! Two axes are swept:
//! - **Raster resolution** (`64²`, `256²`, `1024²`): the per-row cost is
//!   dominated by O(width × height) mask handling, so the batch row count is
//!   scaled inversely with the pixel count to keep wall-clock comparable.
//! - **Clip polygon complexity** (vertex count) at a fixed resolution, which
//!   drives the GDAL rasterization cost.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, ScalarUDF, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{
    crs::{lnglat, Crs},
    datatypes::SedonaType,
};
use sedona_testing::benchmark_util::{BenchmarkArgSpec::*, BenchmarkArgs};

/// Tags CRS-less geometries with lng/lat so they match the CRS of the rasters
/// produced by the benchmark generator (avoiding an RS_Clip CRS mismatch).
fn sd_apply_default_crs_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "sd_applydefaultcrs",
        vec![Arc::new(SDApplyDefaultCRS { crs: lnglat() })],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct SDApplyDefaultCRS {
    crs: Crs,
}

impl SedonaScalarKernel for SDApplyDefaultCRS {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        if args.len() != 1 {
            return Ok(None);
        }

        match &args[0] {
            SedonaType::Wkb(edges, crs) if crs.is_none() => {
                Ok(Some(SedonaType::Wkb(*edges, self.crs.clone())))
            }
            SedonaType::WkbView(edges, crs) if crs.is_none() => {
                Ok(Some(SedonaType::WkbView(*edges, self.crs.clone())))
            }
            SedonaType::Wkb(..) | SedonaType::WkbView(..) => Ok(Some(args[0].clone())),
            _ => Ok(None),
        }
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        Ok(args[0].clone())
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let f = sedona_raster_gdal::register::default_function_set();
    let udf: ScalarUDF = f
        .scalar_udf("rs_clip")
        .expect("rs_clip is registered")
        .clone()
        .into();

    // Helper: build a fixed-size batch for one (raster, band, polygon) config
    // and register it as a criterion benchmark. `rows` is the batch size, set
    // per case so total work stays comparable. crop defaults to true.
    let mut bench_clip = |w, h, vertices, rows| {
        let data = BenchmarkArgs::ArrayArrayArray(
            Raster(w, h),
            Int32(1, 2),
            Transformed(
                Box::new(Polygon(vertices)),
                sd_apply_default_crs_udf().into(),
            ),
        )
        .build_data(1, rows)
        .expect("build benchmark data");
        c.bench_function(&data.make_label("raster-gdal", "rs_clip"), |b| {
            b.iter(|| data.invoke_scalar(&udf).unwrap())
        });
    };

    // Resolution sweep (simple polygon); row count scales inversely with the
    // pixel count so each case does comparable total work.
    bench_clip(64, 64, 8, 2048);
    bench_clip(256, 256, 8, 128);
    bench_clip(1024, 1024, 8, 8);

    // Polygon-complexity axis at a fixed 64×64 resolution (the simple-polygon
    // case at this size is already covered by the sweep above).
    bench_clip(64, 64, 50, 2048);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
