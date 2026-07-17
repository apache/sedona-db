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

//! Benchmarks for the RS_AsGeoTiff UDF.
//!
//! Exercises the raster → GeoTIFF export path (`RS_Example` rasters, no external
//! fixtures): the uncompressed default and the compression axis (none/lzw/deflate).

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, ScalarUDF};
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::raster::BandDataType;
use sedona_testing::raster_spec::RasterSpec;
use sedona_testing::testers::ScalarUdfTester;

fn raster_array(rows: usize) -> ArrayRef {
    assert!(rows > 0, "benchmark rows must be positive");

    let example: ScalarUDF = sedona_raster_functions::rs_example::rs_example_udf().into();
    let tester = ScalarUdfTester::new(example, vec![]);
    match tester.invoke(vec![]).unwrap() {
        ColumnarValue::Scalar(value) => value.to_array_of_size(rows).unwrap(),
        ColumnarValue::Array(array) => array,
    }
}

fn bench_rs_as_geotiff_basic(c: &mut Criterion) {
    let udf: ScalarUDF = sedona_raster_gdal::rs_as_geotiff_udf().into();
    let tester = ScalarUdfTester::new(udf, vec![RASTER]);

    let mut group = c.benchmark_group("rs_as_geotiff");
    for rows in [1usize, 32] {
        let rasters = raster_array(rows);
        group.throughput(Throughput::Elements(rows as u64));
        group.bench_with_input(BenchmarkId::new("basic", rows), &rasters, |b, input| {
            b.iter(|| black_box(tester.invoke_arrays(vec![input.clone()]).unwrap()))
        });
    }

    // A single ~16 MB raster: large enough that per-row byte handling
    // (vsimem buffer -> output array) registers against the GDAL encode.
    const LARGE: i64 = 4096;
    let large: ArrayRef = Arc::new(
        RasterSpec::d2(LARGE, LARGE)
            .band(BandDataType::UInt8)
            .build(),
    );
    group.throughput(Throughput::Bytes((LARGE * LARGE) as u64));
    group.bench_with_input(BenchmarkId::new("large", "16MiB"), &large, |b, input| {
        b.iter(|| black_box(tester.invoke_arrays(vec![input.clone()]).unwrap()))
    });
    group.finish();
}

fn bench_rs_as_geotiff_compression(c: &mut Criterion) {
    let udf: ScalarUDF = sedona_raster_gdal::rs_as_geotiff_udf().into();
    let tester = ScalarUdfTester::new(
        udf,
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Float64),
        ],
    );

    let rasters = raster_array(32);
    // Quality is a 0.0-1.0 fraction (ignored by these codecs, but must be valid).
    let quality = ColumnarValue::Scalar(ScalarValue::Float64(Some(0.75)));

    let mut group = c.benchmark_group("rs_as_geotiff_compression");
    group.throughput(Throughput::Elements(rasters.len() as u64));
    for compression in ["none", "lzw", "deflate"] {
        let comp = ColumnarValue::Scalar(ScalarValue::Utf8(Some(compression.to_string())));
        group.bench_with_input(
            BenchmarkId::new("compression", compression),
            &(&rasters, &comp, &quality),
            |b, (rasters, comp, quality)| {
                b.iter(|| {
                    black_box(
                        tester
                            .invoke(vec![
                                ColumnarValue::Array((*rasters).clone()),
                                (*comp).clone(),
                                (*quality).clone(),
                            ])
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rs_as_geotiff_basic,
    bench_rs_as_geotiff_compression
);
criterion_main!(benches);
