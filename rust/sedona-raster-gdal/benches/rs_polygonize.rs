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

//! Benchmarks for RS_Polygonize UDF
//!
//! RS_Polygonize converts raster pixels to polygon geometries.
//! Signature: RS_Polygonize(raster, band)
//!
//! This benchmark covers:
//! - Different raster sizes
//! - Different band numbers

mod bench_common;

use arrow_array::Array;
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion_common::ScalarValue;
use datafusion_expr::{ColumnarValue, ScalarUDF};
use sedona_raster_gdal::rs_polygonize_udf;
use sedona_schema::datatypes::{SedonaType, RASTER};

use bench_common::*;

fn bench_rs_polygonize_single(c: &mut Criterion) {
    let udf: ScalarUDF = rs_polygonize_udf().into();

    // Arg types: (raster, band_index)
    let arg_types = vec![RASTER, SedonaType::Arrow(DataType::Int32)];

    let mut group = c.benchmark_group("rs_polygonize");

    for raster_name in &["test1.tiff", "test4.tiff"] {
        let raster_scalar = load_raster_as_scalar(raster_name);
        let band_scalar = ColumnarValue::Scalar(ScalarValue::Int32(Some(1)));

        // Estimate throughput based on source raster size
        let raster_arr = load_rasters_from_geotiff(raster_name, 1);
        let estimated_size = raster_arr.get_array_memory_size();
        group.throughput(Throughput::Bytes(estimated_size as u64));

        group.bench_with_input(
            BenchmarkId::new("single", raster_name),
            &(&raster_scalar, &band_scalar),
            |b, (raster, band)| {
                b.iter(|| {
                    let args = vec![(*raster).clone(), (*band).clone()];
                    invoke_udf(&udf, &args, &arg_types).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_rs_polygonize_single);
criterion_main!(benches);
