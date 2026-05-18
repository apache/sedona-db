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

//! Process-isolated assertion for the "no OutDb loader registered" error
//! message. Lives in a separate integration binary so the global
//! `OUTDB_BAND_LOADER` `OnceLock` stays unset — the in-crate unit tests
//! all install a mock and would mask this path.

use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_schema::raster::BandDataType;

fn build_outdb_band() -> arrow_array::StructArray {
    let mut b = RasterBuilder::new(1);
    b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 3], None)
        .unwrap();
    b.start_band_nd(
        None,
        &["y", "x"],
        &[2, 3],
        BandDataType::UInt8,
        None,
        Some("file:///nonexistent.tif"),
        Some("geotiff"),
    )
    .unwrap();
    b.band_data_writer().append_value([0u8; 0]); // empty → schema-OutDb
    b.finish_band().unwrap();
    b.finish_raster().unwrap();
    b.finish().unwrap()
}

#[test]
fn nd_buffer_errors_with_clear_message_when_no_loader_registered() {
    let arr = build_outdb_band();
    let rasters = RasterStructArray::new(&arr);
    let raster = rasters.get(0).unwrap();
    let band = raster.band(0).unwrap();
    let err = band.nd_buffer().unwrap_err().to_string();
    assert!(
        err.contains("no OutDb loader registered"),
        "expected the no-loader diagnostic, got: {err}"
    );
}
