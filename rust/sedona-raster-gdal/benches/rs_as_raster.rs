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
use sedona_gdal::global::with_global_gdal;
use sedona_gdal::raster::types::Buffer;
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_schema::datatypes::{SedonaType, RASTER, WKB_GEOMETRY};
use sedona_testing::{create::create_array, testers::ScalarUdfTester};

fn base_raster() -> arrow_array::StructArray {
    with_global_gdal(|gdal| {
        let driver = gdal.get_driver_by_name("MEM").unwrap();
        let dataset = driver.create_with_band_type::<u8>("", 4, 3, 1).unwrap();
        dataset
            .set_geo_transform(&[10.0, 2.0, 0.0, 20.0, 0.0, -2.0])
            .unwrap();
        dataset.set_projection("EPSG:4326").unwrap();
        let band = dataset.rasterband(1).unwrap();
        band.set_no_data_value(Some(0.0)).unwrap();
        let mut buffer = Buffer::new((4, 3), vec![0u8; 12]);
        band.write((0, 0), (4, 3), &mut buffer).unwrap();
        sedona_raster_gdal::dataset_to_indb_raster(&dataset).unwrap()
    })
    .unwrap()
}

fn raster_array(rows: usize) -> ArrayRef {
    let raster = base_raster();

    if rows == 1 {
        Arc::new(raster)
    } else {
        let raster_struct = RasterStructArray::new(&raster);
        let raster_ref = raster_struct.get(0).unwrap();
        let mut builder = RasterBuilder::new(rows);
        for _ in 0..rows {
            builder
                .start_raster(&raster_ref.metadata(), raster_ref.crs())
                .unwrap();
            let band = raster_ref.bands().band(1).unwrap();
            builder.start_band(band.metadata()).unwrap();
            builder
                .band_data_writer()
                .append_value(band.nd_buffer().unwrap().as_contiguous().unwrap());
            builder.finish_band().unwrap();
            builder.finish_raster().unwrap();
        }
        Arc::new(builder.finish().unwrap())
    }
}

fn geometry_array(rows: usize) -> ArrayRef {
    let polygon = with_global_gdal(|_gdal| {
        let raster = base_raster();
        let raster_struct = RasterStructArray::new(&raster);
        let raster = raster_struct.get(0).unwrap();
        let md = raster.metadata();
        format!(
            "POLYGON(({x0} {y1}, {x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}))",
            x0 = md.upper_left_x(),
            x1 = md.upper_left_x() + md.scale_x(),
            y0 = md.upper_left_y(),
            y1 = md.upper_left_y() + md.scale_y(),
        )
    })
    .unwrap();

    create_array(&vec![Some(polygon.as_str()); rows], &WKB_GEOMETRY)
}

fn bench_rs_as_raster(c: &mut Criterion) {
    let udf: ScalarUDF = sedona_raster_gdal::rs_as_raster_udf().into();
    let tester = ScalarUdfTester::new(
        udf,
        vec![
            WKB_GEOMETRY,
            RASTER,
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Float64),
            SedonaType::Arrow(DataType::Float64),
            SedonaType::Arrow(DataType::Boolean),
        ],
    );

    let single_geom = geometry_array(1);
    let single_raster = raster_array(1);
    let batch_geom = geometry_array(128);
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
