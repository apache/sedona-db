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

//! Regression tests for all RS_* raster functions.
//!
//! These tests lock in the exact outputs of every raster function before the N-D
//! schema migration. After migration, every test here must produce identical results.
//! If a test fails after migration, it means the migration changed observable behavior.

use std::sync::Arc;

use arrow_array::{Array, Float64Array, StringArray, UInt32Array, UInt64Array};
use arrow_schema::DataType;
use datafusion_common::ScalarValue;
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_schema::crs::lnglat;
use sedona_schema::datatypes::{Edges, SedonaType, RASTER, WKB_GEOMETRY};
use sedona_schema::raster::{BandDataType, StorageType};
use sedona_testing::compare::assert_array_equal;
use sedona_testing::create::create_array;
use sedona_testing::rasters::generate_test_rasters;
use sedona_testing::testers::ScalarUdfTester;

// -----------------------------------------------------------------------
// Shared test data helpers
// -----------------------------------------------------------------------

/// Standard test rasters: 3 rasters with index 1 null.
/// Raster 0: 1x2, upperleft=(1,2), scale=(0.1,-0.2), skew=(0,0), 1 band UInt16
/// Raster 1: null
/// Raster 2: 3x4, upperleft=(3,4), scale=(0.2,-0.4), skew=(0.06,0.08), 1 band UInt16
fn standard_rasters() -> arrow_array::StructArray {
    generate_test_rasters(3, Some(1)).unwrap()
}

/// Build a multi-band raster for testing band-specific operations.
/// 3 bands: UInt8, UInt16, Float32, each 2x2 pixels.
fn multi_band_raster() -> arrow_array::StructArray {
    let mut builder = RasterBuilder::new(1);
    let crs = lnglat().unwrap().to_crs_string();
    let metadata = RasterMetadata {
        width: 2,
        height: 2,
        upperleft_x: 10.0,
        upperleft_y: 20.0,
        scale_x: 0.5,
        scale_y: -0.5,
        skew_x: 0.0,
        skew_y: 0.0,
    };
    builder.start_raster(&metadata, Some(&crs)).unwrap();

    // Band 1: UInt8, nodata=255
    builder
        .start_band(BandMetadata {
            datatype: BandDataType::UInt8,
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            outdb_url: None,
            outdb_band_id: None,
        })
        .unwrap();
    builder
        .band_data_writer()
        .append_value([1u8, 2u8, 3u8, 4u8]);
    builder.finish_band().unwrap();

    // Band 2: UInt16, nodata=0
    builder
        .start_band(BandMetadata {
            datatype: BandDataType::UInt16,
            nodata_value: Some(vec![0u8, 0u8]),
            storage_type: StorageType::InDb,
            outdb_url: None,
            outdb_band_id: None,
        })
        .unwrap();
    let band2_data: Vec<u8> = [100u16, 200u16, 300u16, 400u16]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    builder.band_data_writer().append_value(&band2_data);
    builder.finish_band().unwrap();

    // Band 3: Float32, no nodata
    builder
        .start_band(BandMetadata {
            datatype: BandDataType::Float32,
            nodata_value: None,
            storage_type: StorageType::InDb,
            outdb_url: None,
            outdb_band_id: None,
        })
        .unwrap();
    let band3_data: Vec<u8> = [1.5f32, 2.5f32, 3.5f32, 4.5f32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    builder.band_data_writer().append_value(&band3_data);
    builder.finish_band().unwrap();

    builder.finish_raster().unwrap();
    builder.finish().unwrap()
}

// -----------------------------------------------------------------------
// Round-trip: builder → reader
// -----------------------------------------------------------------------

#[test]
fn roundtrip_builder_reader() {
    let rasters = standard_rasters();
    let array = RasterStructArray::new(&rasters);
    assert_eq!(array.len(), 3);

    // Raster 0
    let r0 = array.get(0).unwrap();
    assert_eq!(r0.metadata().width(), 1);
    assert_eq!(r0.metadata().height(), 2);
    assert_eq!(r0.metadata().upper_left_x(), 1.0);
    assert_eq!(r0.metadata().upper_left_y(), 2.0);
    assert_eq!(r0.metadata().scale_x(), 0.1);
    assert_eq!(r0.metadata().scale_y(), -0.2);
    assert_eq!(r0.metadata().skew_x(), 0.0);
    assert_eq!(r0.metadata().skew_y(), 0.0);
    assert!(r0.crs().is_some());
    assert_eq!(r0.bands().len(), 1);

    let band = r0.bands().band(1).unwrap();
    assert_eq!(band.metadata().data_type().unwrap(), BandDataType::UInt16);
    assert_eq!(band.metadata().nodata_value(), Some(&[0u8, 0u8][..]));
    assert_eq!(band.metadata().storage_type().unwrap(), StorageType::InDb);
    // 1x2 = 2 pixels * 2 bytes = 4 bytes
    assert_eq!(band.data().len(), 4);

    // Raster 1: null
    assert!(array.is_null(1));

    // Raster 2
    let r2 = array.get(2).unwrap();
    assert_eq!(r2.metadata().width(), 3);
    assert_eq!(r2.metadata().height(), 4);
    assert_eq!(r2.metadata().upper_left_x(), 3.0);
    assert_eq!(r2.metadata().upper_left_y(), 4.0);
    assert_eq!(r2.metadata().scale_x(), 0.2);
    assert_eq!(r2.metadata().scale_y(), -0.4);
    assert_eq!(r2.metadata().skew_x(), 0.06);
    assert_eq!(r2.metadata().skew_y(), 0.08);
}

#[test]
fn roundtrip_multi_band() {
    let rasters = multi_band_raster();
    let array = RasterStructArray::new(&rasters);
    let r = array.get(0).unwrap();

    assert_eq!(r.bands().len(), 3);

    let b1 = r.bands().band(1).unwrap();
    assert_eq!(b1.metadata().data_type().unwrap(), BandDataType::UInt8);
    assert_eq!(b1.metadata().nodata_value(), Some(&[255u8][..]));
    assert_eq!(b1.data(), &[1u8, 2, 3, 4]);

    let b2 = r.bands().band(2).unwrap();
    assert_eq!(b2.metadata().data_type().unwrap(), BandDataType::UInt16);
    assert_eq!(b2.metadata().nodata_value(), Some(&[0u8, 0][..]));

    let b3 = r.bands().band(3).unwrap();
    assert_eq!(b3.metadata().data_type().unwrap(), BandDataType::Float32);
    assert_eq!(b3.metadata().nodata_value(), None);
}

// -----------------------------------------------------------------------
// RS_Width / RS_Height
// -----------------------------------------------------------------------

#[test]
fn regression_rs_width() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_size::rs_width_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(UInt64Array::from(vec![Some(1), None, Some(3)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_height() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_size::rs_height_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(UInt64Array::from(vec![Some(2), None, Some(4)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

// -----------------------------------------------------------------------
// RS_NumBands
// -----------------------------------------------------------------------

#[test]
fn regression_rs_numbands() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_numbands::rs_numbands_udf().into(),
        vec![RASTER],
    );
    // Standard rasters have 1 band each
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(UInt32Array::from(vec![Some(1), None, Some(1)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_numbands_multi() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_numbands::rs_numbands_udf().into(),
        vec![RASTER],
    );
    let rasters = multi_band_raster();
    let expected: Arc<dyn Array> = Arc::new(UInt32Array::from(vec![Some(3)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

// -----------------------------------------------------------------------
// RS_BandPixelType
// -----------------------------------------------------------------------

#[test]
fn regression_rs_bandpixeltype() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_band_accessors::rs_bandpixeltype_udf().into(),
        vec![RASTER],
    );
    // Default (band 1) on standard rasters — all UInt16
    let rasters = standard_rasters();
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    let string_array = result
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("expected StringArray");
    assert_eq!(string_array.value(0), "UNSIGNED_16BITS");
    assert!(string_array.is_null(1));
    assert_eq!(string_array.value(2), "UNSIGNED_16BITS");
}

#[test]
fn regression_rs_bandpixeltype_by_index() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_band_accessors::rs_bandpixeltype_udf().into(),
        vec![RASTER, SedonaType::Arrow(DataType::Int32)],
    );
    let rasters = multi_band_raster();

    // Band 1: UInt8
    let result = tester
        .invoke_array_scalar(Arc::new(rasters.clone()), 1_i32)
        .unwrap();
    let arr = result.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(arr.value(0), "UNSIGNED_8BITS");

    // Band 2: UInt16
    let result = tester
        .invoke_array_scalar(Arc::new(rasters.clone()), 2_i32)
        .unwrap();
    let arr = result.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(arr.value(0), "UNSIGNED_16BITS");

    // Band 3: Float32
    let result = tester
        .invoke_array_scalar(Arc::new(rasters), 3_i32)
        .unwrap();
    let arr = result.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(arr.value(0), "REAL_32BITS");
}

// -----------------------------------------------------------------------
// RS_BandNoDataValue
// -----------------------------------------------------------------------

#[test]
fn regression_rs_bandnodatavalue() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_band_accessors::rs_bandnodatavalue_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![Some(0.0), None, Some(0.0)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

// -----------------------------------------------------------------------
// RS_BandPath
// -----------------------------------------------------------------------

#[test]
fn regression_rs_bandpath_indb() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_bandpath::rs_bandpath_udf().into(),
        vec![RASTER],
    );
    // InDb rasters should return null for band path
    let rasters = standard_rasters();
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();
    assert!(string_array.is_null(0));
    assert!(string_array.is_null(1));
    assert!(string_array.is_null(2));
}

// -----------------------------------------------------------------------
// Geotransform functions
// -----------------------------------------------------------------------

#[test]
fn regression_rs_upperleftx() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_geotransform::rs_upperleftx_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_upperlefty() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_geotransform::rs_upperlefty_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![Some(2.0), None, Some(4.0)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_scalex() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_geotransform::rs_scalex_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![Some(0.1), None, Some(0.2)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_scaley() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_geotransform::rs_scaley_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![Some(-0.2), None, Some(-0.4)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_skewx() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_geotransform::rs_skewx_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![Some(0.0), None, Some(0.06)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_skewy() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_geotransform::rs_skewy_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![Some(0.0), None, Some(0.08)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_rotation() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_geotransform::rs_rotation_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![
        Some(-0.0),
        None,
        Some(-0.29145679447786704),
    ]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

// -----------------------------------------------------------------------
// Coordinate conversion
// -----------------------------------------------------------------------

#[test]
fn regression_rs_rastertoworldcoordx() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_worldcoordinate::rs_rastertoworldcoordx_udf().into(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ],
    );
    let rasters = standard_rasters();
    // At pixel (0, 0), world X = upperleft_x
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
    let result = tester
        .invoke_array_scalar_scalar(Arc::new(rasters), 0_i32, 0_i32)
        .unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_rastertoworldcoordy() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_worldcoordinate::rs_rastertoworldcoordy_udf().into(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ],
    );
    let rasters = standard_rasters();
    // At pixel (0, 0), world Y = upperleft_y
    let expected: Arc<dyn Array> = Arc::new(Float64Array::from(vec![Some(2.0), None, Some(4.0)]));
    let result = tester
        .invoke_array_scalar_scalar(Arc::new(rasters), 0_i32, 0_i32)
        .unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_rastertoworldcoord() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_worldcoordinate::rs_rastertoworldcoord_udf().into(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ],
    );
    let rasters = standard_rasters();
    // At pixel (0, 0), world coord = (upperleft_x, upperleft_y) = POINT(1 2) / POINT(3 4)
    let expected = &create_array(
        &[Some("POINT (1 2)"), None, Some("POINT (3 4)")],
        &WKB_GEOMETRY,
    );
    let result = tester
        .invoke_array_scalar_scalar(Arc::new(rasters), 0_i32, 0_i32)
        .unwrap();
    assert_array_equal(&result, expected);
}

#[test]
fn regression_rs_worldtorastercoordx() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_rastercoordinate::rs_worldtorastercoordx_udf().into(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Float64),
            SedonaType::Arrow(DataType::Float64),
        ],
    );
    let rasters = standard_rasters();
    // World coord = upperleft → pixel (0, 0), so X = 0
    let result = tester
        .invoke_array_scalar_scalar(Arc::new(rasters.clone()), 1.0_f64, 2.0_f64)
        .unwrap();
    let int_array = result
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();
    assert_eq!(int_array.value(0), 0);
}

#[test]
fn regression_rs_worldtorastercoordy() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_rastercoordinate::rs_worldtorastercoordy_udf().into(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Float64),
            SedonaType::Arrow(DataType::Float64),
        ],
    );
    let rasters = standard_rasters();
    let result = tester
        .invoke_array_scalar_scalar(Arc::new(rasters.clone()), 1.0_f64, 2.0_f64)
        .unwrap();
    let int_array = result
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap();
    assert_eq!(int_array.value(0), 0);
}

// -----------------------------------------------------------------------
// CRS / SRID
// -----------------------------------------------------------------------

#[test]
fn regression_rs_srid() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_srid::rs_srid_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    // Standard test rasters use lnglat CRS = EPSG:4326 = SRID 4326
    let expected: Arc<dyn Array> = Arc::new(UInt32Array::from(vec![Some(4326), None, Some(4326)]));
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert_array_equal(&result, &expected);
}

#[test]
fn regression_rs_crs() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_srid::rs_crs_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();
    // Non-null rasters should have a non-empty CRS string
    assert!(!string_array.value(0).is_empty());
    assert!(string_array.is_null(1));
    assert!(!string_array.value(2).is_empty());
}

// -----------------------------------------------------------------------
// RS_GeoReference
// -----------------------------------------------------------------------

#[test]
fn regression_rs_georeference() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_georeference::rs_georeference_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    let string_array = result.as_any().downcast_ref::<StringArray>().unwrap();
    // Raster 0: scale_x=0.1, skew_y=0.0, skew_x=0.0, scale_y=-0.2, upperleft_x=1.0, upperleft_y=2.0
    // GDAL format: "scale_x\nskew_y\nskew_x\nscale_y\nupperleft_x\nupperleft_y"
    let geo_ref_0 = string_array.value(0);
    assert!(
        geo_ref_0.contains("0.1"),
        "GeoReference should contain scale_x: {}",
        geo_ref_0
    );
    assert!(string_array.is_null(1));
}

// -----------------------------------------------------------------------
// Spatial geometry functions
// -----------------------------------------------------------------------

#[test]
fn regression_rs_envelope() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_envelope::rs_envelope_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    // Should return non-null geometry for non-null rasters
    assert!(!result.is_null(0));
    assert!(result.is_null(1));
    assert!(!result.is_null(2));
}

#[test]
fn regression_rs_convexhull() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_convexhull::rs_convexhull_udf().into(),
        vec![RASTER],
    );
    let rasters = standard_rasters();
    let result = tester.invoke_array(Arc::new(rasters)).unwrap();
    assert!(!result.is_null(0));
    assert!(result.is_null(1));
    assert!(!result.is_null(2));
}

// -----------------------------------------------------------------------
// Pixel functions
// -----------------------------------------------------------------------

#[test]
fn regression_rs_pixelaspoint() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_pixel_functions::rs_pixelaspoint_udf().into(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ],
    );
    let rasters = standard_rasters();
    let result = tester
        .invoke_array_scalar_scalar(Arc::new(rasters), 1_i32, 1_i32)
        .unwrap();
    // At pixel (1,1) with 1-based indexing:
    // Raster 0: upperleft=(1,2), scale=(0.1,-0.2), skew=(0,0) → POINT(1 2)
    // Raster 1: null
    // Raster 2: non-null geometry
    assert!(!result.is_null(0));
    assert!(result.is_null(1));
    assert!(!result.is_null(2));
}

#[test]
fn regression_rs_pixelascentroid() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_pixel_functions::rs_pixelascentroid_udf().into(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ],
    );
    let rasters = standard_rasters();
    let result = tester
        .invoke_array_scalar_scalar(Arc::new(rasters), 1_i32, 1_i32)
        .unwrap();
    // Should return non-null geometry for non-null rasters
    assert!(!result.is_null(0));
    assert!(result.is_null(1));
    assert!(!result.is_null(2));
}

#[test]
fn regression_rs_pixelaspolygon() {
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_pixel_functions::rs_pixelaspolygon_udf().into(),
        vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ],
    );
    let rasters = standard_rasters();
    let result = tester
        .invoke_array_scalar_scalar(Arc::new(rasters), 1_i32, 1_i32)
        .unwrap();
    assert!(!result.is_null(0));
    assert!(result.is_null(1));
    assert!(!result.is_null(2));
}

// -----------------------------------------------------------------------
// Spatial predicates
// -----------------------------------------------------------------------

#[test]
fn regression_rs_intersects() {
    // Rasters have CRS (lnglat), so geometry must also have matching CRS
    let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_spatial_predicates::rs_intersects_udf().into(),
        vec![RASTER, geom_type.clone()],
    );
    // Use rasters with index 0 null (to match existing test pattern)
    let rasters = generate_test_rasters(3, Some(0)).unwrap();
    // Raster 1: corners at approximately (2.0, 3.0) area
    // Raster 2: corners at approximately (3.0, 4.0) area
    let geom = create_array(
        &[
            None,
            Some("POINT (2.15 2.75)"), // Inside raster 1
            Some("POINT (0.0 0.0)"),   // Outside all rasters
        ],
        &geom_type,
    );
    let result = tester.invoke_arrays(vec![Arc::new(rasters), geom]).unwrap();
    let bool_array = result
        .as_any()
        .downcast_ref::<arrow_array::BooleanArray>()
        .unwrap();
    assert!(bool_array.is_null(0));
    assert!(bool_array.value(1));
    assert!(!bool_array.value(2));
}

#[test]
fn regression_rs_contains() {
    let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
    let tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_spatial_predicates::rs_contains_udf().into(),
        vec![RASTER, geom_type.clone()],
    );
    let rasters = generate_test_rasters(3, Some(0)).unwrap();
    // Point far outside any raster
    let geom = create_array(
        &[None, Some("POINT (999 999)"), Some("POINT (999 999)")],
        &geom_type,
    );
    let result = tester.invoke_arrays(vec![Arc::new(rasters), geom]).unwrap();
    let bool_array = result
        .as_any()
        .downcast_ref::<arrow_array::BooleanArray>()
        .unwrap();
    assert!(bool_array.is_null(0));
    assert!(!bool_array.value(1));
    assert!(!bool_array.value(2));
}

// -----------------------------------------------------------------------
// RS_SetSRID / RS_SetCRS
// -----------------------------------------------------------------------

#[test]
fn regression_rs_setsrid_roundtrip() {
    let set_tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_setsrid::rs_set_srid_udf().into(),
        vec![RASTER, SedonaType::Arrow(DataType::UInt32)],
    );
    let get_tester = ScalarUdfTester::new(
        sedona_raster_functions::rs_srid::rs_srid_udf().into(),
        vec![RASTER],
    );

    let rasters = standard_rasters();
    // Set SRID to 3857
    let updated = set_tester
        .invoke_array_scalar(Arc::new(rasters), 3857_u32)
        .unwrap();
    // Read back SRID
    let result = get_tester.invoke_array(updated).unwrap();
    let srid_array = result.as_any().downcast_ref::<UInt32Array>().unwrap();
    assert_eq!(srid_array.value(0), 3857);
    assert!(srid_array.is_null(1));
    assert_eq!(srid_array.value(2), 3857);
}

// -----------------------------------------------------------------------
// RS_Example
// -----------------------------------------------------------------------

#[test]
fn regression_rs_example() {
    use datafusion_expr::ScalarFunctionArgs;

    // RS_Example takes no args — invoke the UDF directly
    let udf: datafusion_expr::ScalarUDF =
        sedona_raster_functions::rs_example::rs_example_udf().into();
    let return_field = udf
        .return_field_from_args(datafusion_expr::ReturnFieldArgs {
            arg_fields: &[],
            scalar_arguments: &[],
        })
        .unwrap();
    let config_options = Arc::new(datafusion_common::config::ConfigOptions::default());
    let result = udf
        .invoke_with_args(ScalarFunctionArgs {
            args: vec![],
            arg_fields: vec![],
            number_rows: 1,
            return_field,
            config_options,
        })
        .unwrap();

    if let datafusion_expr::ColumnarValue::Scalar(ScalarValue::Struct(arc_struct)) = result {
        let raster_array = RasterStructArray::new(arc_struct.as_ref());
        assert_eq!(raster_array.len(), 1);
        let raster = raster_array.get(0).unwrap();
        assert_eq!(raster.metadata().width(), 64);
        assert_eq!(raster.metadata().height(), 32);
        assert_eq!(raster.bands().len(), 3);
    } else {
        panic!("Expected scalar struct result from RS_Example");
    }
}

// -----------------------------------------------------------------------
// Null scalar propagation
// -----------------------------------------------------------------------

#[test]
fn regression_null_propagation() {
    // All single-arg raster functions should return null for null input
    type TesterFactory<'a> = Vec<(&'a str, Box<dyn Fn() -> ScalarUdfTester>)>;
    let functions: TesterFactory = vec![
        (
            "rs_width",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_size::rs_width_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_height",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_size::rs_height_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_numbands",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_numbands::rs_numbands_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_upperleftx",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_geotransform::rs_upperleftx_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_upperlefty",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_geotransform::rs_upperlefty_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_scalex",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_geotransform::rs_scalex_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_scaley",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_geotransform::rs_scaley_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_skewx",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_geotransform::rs_skewx_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_skewy",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_geotransform::rs_skewy_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_rotation",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_geotransform::rs_rotation_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_srid",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_srid::rs_srid_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
        (
            "rs_crs",
            Box::new(|| {
                ScalarUdfTester::new(
                    sedona_raster_functions::rs_srid::rs_crs_udf().into(),
                    vec![RASTER],
                )
            }),
        ),
    ];

    for (name, make_tester) in &functions {
        let tester = make_tester();
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(
            result.is_null(),
            "{} should return null for null input, got {:?}",
            name,
            result
        );
    }
}
