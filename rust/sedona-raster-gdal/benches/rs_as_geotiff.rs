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

//! Benchmarks for RS_AsGeoTiff UDF
//!
//! RS_AsGeoTiff converts rasters to GeoTIFF binary format.
//! Supported variants:
//! - RS_AsGeoTiff(raster)
//! - RS_AsGeoTiff(raster, compression)
//! - RS_AsGeoTiff(raster, compression, quality)
//! - RS_AsGeoTiff(raster, compression, quality, tileWidth, tileHeight)

mod bench_common;

use arrow_array::Array;
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, ScalarUDF};
use sedona_raster_gdal::rs_as_geotiff_udf;
use sedona_schema::datatypes::{SedonaType, RASTER};

use bench_common::*;

fn bench_rs_as_geotiff_basic(c: &mut Criterion) {
    let udf: ScalarUDF = rs_as_geotiff_udf().into();

    // Arg types: (raster)
    let arg_types = vec![RASTER];

    let mut group = c.benchmark_group("rs_as_geotiff");

    for raster_name in &["test1.tiff", "test4.tiff"] {
        let raster_scalar = load_raster_as_scalar(raster_name);

        // Estimate throughput based on source raster size
        let raster_arr = load_rasters_from_geotiff(raster_name, 1);
        let estimated_size = raster_arr.get_array_memory_size();
        group.throughput(Throughput::Bytes(estimated_size as u64));

        group.bench_with_input(
            BenchmarkId::new("basic", raster_name),
            &raster_scalar,
            |b, raster| {
                b.iter(|| {
                    let args = vec![raster.clone()];
                    invoke_udf(&udf, &args, &arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_rs_as_geotiff_with_compression(c: &mut Criterion) {
    let udf: ScalarUDF = rs_as_geotiff_udf().into();

    // Arg types: (raster, compression, quality)
    let arg_types = vec![
        RASTER,
        SedonaType::Arrow(DataType::Utf8),
        SedonaType::Arrow(DataType::Float64),
    ];

    let mut group = c.benchmark_group("rs_as_geotiff_compression");

    let raster_name = "test4.tiff";
    let raster_scalar = load_raster_as_scalar(raster_name);
    let quality_scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(75.0)));

    for compression in &["none", "lzw", "deflate"] {
        let compression_scalar =
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(compression.to_string())));

        group.bench_with_input(
            BenchmarkId::new("compression", compression),
            &(&raster_scalar, &compression_scalar, &quality_scalar),
            |b, (raster, comp, qual)| {
                b.iter(|| {
                    let args = vec![(*raster).clone(), (*comp).clone(), (*qual).clone()];
                    invoke_udf(&udf, &args, &arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rs_as_geotiff_basic,
    bench_rs_as_geotiff_with_compression
);
criterion_main!(benches);
