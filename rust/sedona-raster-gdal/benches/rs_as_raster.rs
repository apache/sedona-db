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

//! Benchmarks for RS_AsRaster UDF.

use std::{hint::black_box, sync::Arc};

use arrow_array::{ArrayRef, BooleanArray, Float64Array, StringArray};
use arrow_schema::DataType;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion_expr::ScalarUDF;
use sedona_schema::crs::deserialize_crs;
use sedona_schema::datatypes::{Edges, SedonaType, RASTER};
use sedona_testing::{
    create::create_array,
    raster_spec::{raster_array as build_raster_array, RasterSpec},
    testers::ScalarUdfTester,
};

fn reference_raster_spec() -> RasterSpec {
    RasterSpec::d2(4, 3)
        .transform([10.0, 2.0, 0.0, 20.0, 0.0, -2.0])
        .crs(Some("EPSG:4326"))
        .band_values(&[0u8; 12])
        .nodata(0u8)
}

fn raster_array(rows: usize) -> ArrayRef {
    Arc::new(build_raster_array(vec![
        Some(reference_raster_spec());
        rows
    ]))
}

fn geometry_array(rows: usize, geom_type: &SedonaType) -> ArrayRef {
    let polygon = "POLYGON((10 18, 10 20, 12 20, 12 18, 10 18))";

    create_array(&vec![Some(polygon); rows], geom_type)
}

fn bench_rs_as_raster(c: &mut Criterion) {
    let udf: ScalarUDF = sedona_raster_gdal::rs_as_raster_udf().into();
    let geom_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:4326").unwrap());
    let tester = ScalarUdfTester::new(
        udf,
        vec![
            geom_type.clone(),
            RASTER,
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Float64),
            SedonaType::Arrow(DataType::Float64),
            SedonaType::Arrow(DataType::Boolean),
        ],
    );

    let single_geom = geometry_array(1, &geom_type);
    let single_raster = raster_array(1);
    let batch_geom = geometry_array(128, &geom_type);
    let batch_raster = raster_array(128);
    let pixel_type_single = Arc::new(StringArray::from(vec!["D"]));
    let pixel_type_batch = Arc::new(StringArray::from(vec!["D"; 128]));
    let all_touched_single = Arc::new(BooleanArray::from(vec![false]));
    let all_touched_batch = Arc::new(BooleanArray::from(vec![false; 128]));
    let burn_single = Arc::new(Float64Array::from(vec![1.0]));
    let burn_batch = Arc::new(Float64Array::from(vec![1.0; 128]));
    let nodata_single = Arc::new(Float64Array::from(vec![0.0]));
    let nodata_batch = Arc::new(Float64Array::from(vec![0.0; 128]));
    let extent_single = Arc::new(BooleanArray::from(vec![true]));
    let extent_batch = Arc::new(BooleanArray::from(vec![true; 128]));

    let mut group = c.benchmark_group("rs_as_raster");

    group.throughput(Throughput::Elements(1));
    group.bench_with_input(BenchmarkId::new("fixtures", "single"), &(), |b, _| {
        b.iter(|| {
            black_box(
                tester
                    .invoke_arrays(vec![
                        single_geom.clone(),
                        single_raster.clone(),
                        pixel_type_single.clone(),
                        all_touched_single.clone(),
                        burn_single.clone(),
                        nodata_single.clone(),
                        extent_single.clone(),
                    ])
                    .unwrap(),
            )
        })
    });

    group.throughput(Throughput::Elements(128));
    group.bench_with_input(BenchmarkId::new("fixtures", "batch_128"), &(), |b, _| {
        b.iter(|| {
            black_box(
                tester
                    .invoke_arrays(vec![
                        batch_geom.clone(),
                        batch_raster.clone(),
                        pixel_type_batch.clone(),
                        all_touched_batch.clone(),
                        burn_batch.clone(),
                        nodata_batch.clone(),
                        extent_batch.clone(),
                    ])
                    .unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_rs_as_raster);
criterion_main!(benches);
