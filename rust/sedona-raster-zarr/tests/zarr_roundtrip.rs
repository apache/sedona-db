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

//! End-to-end fixture test: build a small Zarr group on disk with the
//! `zarrs` crate, then read it back through `group_to_rasters` and
//! verify the resulting raster `StructArray`. The loader always emits
//! OutDb-style rows (empty `data`, populated `outdb_uri`), so these
//! tests assert on metadata and chunk-anchor URIs rather than pixel
//! bytes. Pixel-byte coverage lives in the loader's unit tests against
//! `retrieve_chunk_bytes`.

use std::sync::Arc;

use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::RasterRef;
use sedona_raster_zarr::group_to_rasters;
use sedona_schema::raster::BandDataType;
use tempfile::TempDir;
use zarrs::array::data_type;
use zarrs::array::ArrayBuilder;
use zarrs::group::GroupBuilder;
use zarrs_filesystem::FilesystemStore;

/// Build a 2-band group on disk:
///   - dims:  [t, y, x]
///   - shape: [2, 4, 4]
///   - chunks: [1, 2, 2]    → chunk grid [2, 2, 2] = 8 chunk positions
///   - arrays: "temperature" (UInt8) and "pressure" (UInt8)
///
/// Returns the temp dir (kept alive by the caller so files persist).
fn build_fixture() -> TempDir {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());

    // Group with a known affine transform so we can verify per-chunk
    // transforms below.
    let mut group_attrs = serde_json::Map::new();
    group_attrs.insert(
        "spatial:transform".into(),
        serde_json::json!([100.0, 1.0, 0.0, 200.0, 0.0, -1.0]),
    );
    group_attrs.insert("proj:epsg".into(), serde_json::json!(4326));
    GroupBuilder::new()
        .attributes(group_attrs)
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    for name in ["temperature", "pressure"] {
        let array = ArrayBuilder::new(
            vec![2u64, 4u64, 4u64],
            vec![1u64, 2u64, 2u64],
            data_type::uint8(),
            0u8,
        )
        .dimension_names(Some(["t", "y", "x"]))
        .build(store.clone(), &format!("/{name}"))
        .unwrap();
        array.store_metadata().unwrap();

        // Bytes don't matter for these tests — the loader doesn't read
        // them. Write zeros so the chunk files exist for any future
        // resolver fixture.
        for t in 0..2u64 {
            for yc in 0..2u64 {
                for xc in 0..2u64 {
                    array.store_chunk(&[t, yc, xc], vec![0u8; 4]).unwrap();
                }
            }
        }
    }

    tmp
}

#[test]
fn round_trip_emits_one_row_per_chunk_position_with_outdb_anchors() {
    let tmp = build_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let arr = group_to_rasters(&uri, None).unwrap();

    let rasters = RasterStructArray::new(&arr);
    assert_eq!(rasters.len(), 8, "expected 8 chunk rows (2*2*2)");

    // First row corresponds to chunk (t=0, y=0, x=0). With group transform
    // [100, 1, 0, 200, 0, -1] and chunk shape [1, 2, 2], chunk (0,0,0) has
    // origin (100, 200) and spatial_shape [2, 2].
    let r0 = rasters.get(0).unwrap();
    let r0_transform: Vec<f64> = r0.transform().to_vec();
    assert_eq!(r0_transform, vec![100.0, 1.0, 0.0, 200.0, 0.0, -1.0]);
    assert_eq!(r0.spatial_shape(), &[2, 2]);
    assert_eq!(r0.num_bands(), 2);
    assert_eq!(r0.crs(), Some("EPSG:4326"));

    // Bands are sorted by array path for determinism — `pressure` sorts
    // before `temperature` lexicographically, so band 0 is pressure and
    // band 1 is temperature.
    let pressure = r0.band(0).unwrap();
    assert_eq!(pressure.raw_source_shape(), &[1, 2, 2]);
    assert_eq!(pressure.data_type(), BandDataType::UInt8);
    assert!(
        !pressure.is_indb(),
        "loader emits OutDb rows — is_indb() must be false"
    );
    assert_eq!(pressure.outdb_format(), Some("zarr"));
    let anchor = pressure.outdb_uri().expect("outdb_uri set");
    // Anchor is the group URI verbatim plus a fragment carrying array
    // path + chunk indices. No `zarr://` prefix.
    assert!(anchor.starts_with("file://"), "got: {anchor}");
    assert!(!anchor.starts_with("zarr://"), "got: {anchor}");
    assert!(anchor.contains("#array=pressure"), "got: {anchor}");
    assert!(anchor.contains("&chunk=0,0,0"), "got: {anchor}");

    // Last row corresponds to chunk (t=1, y=1, x=1). Anchor + transform
    // both reflect the translation.
    let last = rasters.get(7).unwrap();
    let last_transform: Vec<f64> = last.transform().to_vec();
    assert_eq!(last_transform[0], 100.0 + 2.0); // x_off = 2
    assert_eq!(last_transform[3], 200.0 - 2.0); // y_off = 2, sy = -1
    let temp = last.band(1).unwrap();
    let anchor = temp.outdb_uri().expect("outdb_uri set");
    assert!(anchor.contains("#array=temperature"), "got: {anchor}");
    assert!(anchor.contains("&chunk=1,1,1"), "got: {anchor}");
}

#[test]
fn errors_on_empty_group() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();
    let uri = format!("file://{}", tmp.path().display());
    let err = group_to_rasters(&uri, None).unwrap_err().to_string();
    assert!(err.contains("no child arrays"), "got: {err}");
}

/// Build a group with two 3-D data arrays and 1-D `t`/`y`/`x` coord
/// variables alongside them — the xarray-on-Zarr pattern. The loader's
/// default behaviour must drop the coord variables and read only the
/// data arrays.
fn build_xarray_style_fixture() -> TempDir {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    // Two 3-D data arrays sharing the same chunk grid.
    for name in ["temperature", "pressure"] {
        let arr = ArrayBuilder::new(
            vec![2u64, 4u64, 4u64],
            vec![1u64, 2u64, 2u64],
            data_type::uint8(),
            0u8,
        )
        .dimension_names(Some(["t", "y", "x"]))
        .build(store.clone(), &format!("/{name}"))
        .unwrap();
        arr.store_metadata().unwrap();
    }

    // 1-D coord variables. Different chunk grid than the data arrays —
    // would trip validate_group_constraints if not auto-skipped.
    for (name, len, dim) in [("t", 2u64, "t"), ("y", 4u64, "y"), ("x", 4u64, "x")] {
        let arr = ArrayBuilder::new(vec![len], vec![len], data_type::uint8(), 0u8)
            .dimension_names(Some([dim]))
            .build(store.clone(), &format!("/{name}"))
            .unwrap();
        arr.store_metadata().unwrap();
    }

    tmp
}

#[test]
fn auto_skips_1d_coord_variables() {
    let tmp = build_xarray_style_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let arr = group_to_rasters(&uri, None).unwrap();
    let rasters = RasterStructArray::new(&arr);
    // 2*2*2 = 8 chunk positions, with 2 bands per row (pressure, temperature).
    assert_eq!(rasters.len(), 8);
    let r0 = rasters.get(0).unwrap();
    assert_eq!(r0.num_bands(), 2);
}

#[test]
fn explicit_arrays_filter_selects_subset() {
    let tmp = build_xarray_style_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let filter = vec!["temperature".to_string()];
    let arr = group_to_rasters(&uri, Some(&filter)).unwrap();
    let rasters = RasterStructArray::new(&arr);
    assert_eq!(rasters.len(), 8);
    let r0 = rasters.get(0).unwrap();
    assert_eq!(r0.num_bands(), 1, "only temperature should be read");
}

#[test]
fn explicit_arrays_filter_rejects_unknown_name() {
    let tmp = build_xarray_style_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let filter = vec!["humidity".to_string()];
    let err = group_to_rasters(&uri, Some(&filter))
        .unwrap_err()
        .to_string();
    assert!(err.contains("humidity"), "got: {err}");
    assert!(err.contains("no array named"), "got: {err}");
}

#[test]
fn errors_when_crs_declared_without_transform() {
    // CRS-without-transform is almost certainly malformed metadata —
    // the user thinks they have full georef but downstream spatial
    // joins would silently use the identity pixel transform. The
    // loader refuses rather than producing wrong results.
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
    let mut group_attrs = serde_json::Map::new();
    group_attrs.insert("proj:epsg".into(), serde_json::json!(4326));
    GroupBuilder::new()
        .attributes(group_attrs)
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();
    ArrayBuilder::new(vec![2u64, 2], vec![1u64, 2], data_type::uint8(), 0u8)
        .dimension_names(Some(["y", "x"]))
        .build(store.clone(), "/temperature")
        .unwrap()
        .store_metadata()
        .unwrap();

    let uri = format!("file://{}", tmp.path().display());
    let err = group_to_rasters(&uri, None).unwrap_err().to_string();
    assert!(err.contains("CRS"), "got: {err}");
    assert!(err.contains("spatial:transform"), "got: {err}");
}

#[test]
fn explicit_arrays_filter_rejects_1d_arrays() {
    // A user explicitly naming a 1-D array gets a clear "needs 2 dims"
    // error at parse time, not a confusing downstream spatial-dim
    // resolution failure.
    let tmp = build_xarray_style_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let filter = vec!["t".to_string()];
    let err = group_to_rasters(&uri, Some(&filter))
        .unwrap_err()
        .to_string();
    assert!(err.contains("\"t\""), "got: {err}");
    assert!(err.contains("rank 1"), "got: {err}");
    assert!(err.contains("at least 2 dimensions"), "got: {err}");
}

#[test]
fn errors_when_group_has_only_1d_arrays() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();
    ArrayBuilder::new(vec![4u64], vec![2u64], data_type::uint8(), 0u8)
        .dimension_names(Some(["x"]))
        .build(store.clone(), "/x")
        .unwrap()
        .store_metadata()
        .unwrap();

    let uri = format!("file://{}", tmp.path().display());
    let err = group_to_rasters(&uri, None).unwrap_err().to_string();
    assert!(err.contains("only 1-D arrays"), "got: {err}");
}

#[test]
fn falls_back_to_array_dimensions_attribute() {
    // Simulates a Zarr v2 array (or any v3 array that lacks a first-class
    // `dimension_names` field) by leaving `.dimension_names(None)` and
    // setting xarray's `_ARRAY_DIMENSIONS` attribute instead. The loader
    // must accept it and treat the attribute as authoritative.
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());

    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    let mut attrs = serde_json::Map::new();
    attrs.insert("_ARRAY_DIMENSIONS".into(), serde_json::json!(["y", "x"]));
    let array = ArrayBuilder::new(vec![2u64, 2], vec![1u64, 2], data_type::uint8(), 0u8)
        .attributes(attrs)
        .build(store.clone(), "/temperature")
        .unwrap();
    array.store_metadata().unwrap();

    let uri = format!("file://{}", tmp.path().display());
    let arr = group_to_rasters(&uri, None).unwrap();
    let rasters = RasterStructArray::new(&arr);
    assert_eq!(rasters.len(), 2);
    let r0 = rasters.get(0).unwrap();
    let band = r0.band(0).unwrap();
    // Anchor URI populated even though the array used the v2 dim-name
    // fallback — proves the fallback feeds the rest of the pipeline.
    assert_eq!(band.outdb_format(), Some("zarr"));
    assert!(band.outdb_uri().unwrap().contains("#array=temperature"));
}

#[test]
fn errors_on_mismatched_chunk_grids() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();
    ArrayBuilder::new(vec![4u64, 4], vec![2u64, 2], data_type::uint8(), 0u8)
        .dimension_names(Some(["y", "x"]))
        .build(store.clone(), "/array_a")
        .unwrap()
        .store_metadata()
        .unwrap();
    ArrayBuilder::new(vec![4u64, 4], vec![4u64, 4], data_type::uint8(), 0u8)
        .dimension_names(Some(["y", "x"]))
        .build(store.clone(), "/array_b")
        .unwrap()
        .store_metadata()
        .unwrap();

    let uri = format!("file://{}", tmp.path().display());
    let err = group_to_rasters(&uri, None).unwrap_err().to_string();
    assert!(
        err.contains("chunk") && err.contains("array_a") && err.contains("array_b"),
        "got: {err}"
    );
}
