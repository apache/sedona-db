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
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{
    crs::{lnglat, Crs},
    datatypes::SedonaType,
};
use sedona_testing::benchmark_util::{benchmark, BenchmarkArgSpec::*, BenchmarkArgs};

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
    let f = sedona_raster_functions::register::default_function_set();

    // RS_BandNoDataValue
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_bandnodatavalue",
        BenchmarkArgs::Array(Raster(64, 64)),
    );
    // RS_BandPath
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_bandpath",
        BenchmarkArgs::Array(Raster(64, 64)),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_bandpath",
        BenchmarkArgs::ArrayScalar(Raster(64, 64), Int32(1, 2)),
    );
    // RS_BandPixelType
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_bandpixeltype",
        BenchmarkArgs::Array(Raster(64, 64)),
    );

    benchmark::scalar(c, &f, "native-raster", "rs_convexhull", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_crs", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_envelope", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_georeference", Raster(64, 64));
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_georeference",
        BenchmarkArgs::ArrayScalar(Raster(64, 64), String("ESRI".to_string())),
    );
    benchmark::scalar(c, &f, "native-raster", "rs_height", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_numbands", Raster(64, 64));
    // RS_PixelAsPoint, RS_PixelAsCentroid, RS_PixelAsPolygon
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_pixelaspoint",
        BenchmarkArgs::ArrayScalarScalar(Raster(64, 64), Int32(1, 64), Int32(1, 64)),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_pixelascentroid",
        BenchmarkArgs::ArrayScalarScalar(Raster(64, 64), Int32(1, 64), Int32(1, 64)),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_pixelaspolygon",
        BenchmarkArgs::ArrayScalarScalar(Raster(64, 64), Int32(1, 64), Int32(1, 64)),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_rastertoworldcoord",
        BenchmarkArgs::ArrayScalarScalar(Raster(64, 64), Int32(0, 63), Int32(0, 63)),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_rastertoworldcoordx",
        BenchmarkArgs::ArrayScalarScalar(Raster(64, 64), Int32(0, 63), Int32(0, 63)),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_rastertoworldcoordy",
        BenchmarkArgs::ArrayScalarScalar(Raster(64, 64), Int32(0, 63), Int32(0, 63)),
    );
    benchmark::scalar(c, &f, "native-raster", "rs_rotation", Raster(64, 64));
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_setcrs",
        BenchmarkArgs::ArrayScalar(Raster(64, 64), String("EPSG:3857".to_string())),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_setsrid",
        BenchmarkArgs::ArrayScalar(Raster(64, 64), Int32(3857, 3858)),
    );
    benchmark::scalar(c, &f, "native-raster", "rs_scalex", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_scaley", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_skewx", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_skewy", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_srid", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_upperleftx", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_upperlefty", Raster(64, 64));
    benchmark::scalar(c, &f, "native-raster", "rs_width", Raster(64, 64));
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_worldtorastercoord",
        BenchmarkArgs::ArrayScalarScalar(
            Raster(64, 64),
            Float64(-45.0, 45.0),
            Float64(-45.0, 45.0),
        ),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_worldtorastercoordx",
        BenchmarkArgs::ArrayScalarScalar(
            Raster(64, 64),
            Float64(-45.0, 45.0),
            Float64(-45.0, 45.0),
        ),
    );
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_worldtorastercoordy",
        BenchmarkArgs::ArrayScalarScalar(
            Raster(64, 64),
            Float64(-45.0, 45.0),
            Float64(-45.0, 45.0),
        ),
    );

    // RS_Intersects(raster, geometry) - point
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_intersects",
        BenchmarkArgs::ArrayArray(
            Raster(64, 64),
            Transformed(Box::new(Point), sd_apply_default_crs_udf().into()),
        ),
    );
    // RS_Intersects(raster, geometry) - polygon
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_intersects",
        BenchmarkArgs::ArrayArray(
            Raster(64, 64),
            Transformed(Box::new(Polygon(4)), sd_apply_default_crs_udf().into()),
        ),
    );
    // RS_Intersects(raster, raster)
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_intersects",
        BenchmarkArgs::ArrayArray(Raster(64, 64), Raster(64, 64)),
    );
    // RS_Contains(raster, geometry) - point
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_contains",
        BenchmarkArgs::ArrayArray(
            Raster(64, 64),
            Transformed(Box::new(Point), sd_apply_default_crs_udf().into()),
        ),
    );
    // RS_Contains(raster, geometry) - polygon
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_contains",
        BenchmarkArgs::ArrayArray(
            Raster(64, 64),
            Transformed(Box::new(Polygon(4)), sd_apply_default_crs_udf().into()),
        ),
    );
    // RS_Within(raster, geometry) - polygon
    benchmark::scalar(
        c,
        &f,
        "native-raster",
        "rs_within",
        BenchmarkArgs::ArrayArray(
            Raster(64, 64),
            Transformed(Box::new(Polygon(4)), sd_apply_default_crs_udf().into()),
        ),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
