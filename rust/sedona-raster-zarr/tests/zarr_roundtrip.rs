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
//! `zarrs` crate, then read it back through `group_to_*_rasters` and
//! verify the resulting raster `StructArray`.

use std::sync::Arc;

use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::RasterRef;
use sedona_raster_zarr::{group_to_indb_rasters, group_to_outdb_rasters};
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
///
/// `store_chunk_elements` is deprecated in zarrs 0.23 in favour of
/// `store_chunk` (which takes raw bytes); the typed convenience wrapper
/// is still the cleanest path for fixture code so we suppress the
/// warning here.
#[allow(deprecated)]
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

    for (name, base) in [("temperature", 0u8), ("pressure", 100u8)] {
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

        // Fill each chunk with a deterministic pattern so we can verify
        // the right chunk lands in the right row:
        //   pixel(t, y, x) = base + (t*16 + y*4 + x)
        // Each chunk is 1×2×2 = 4 pixels. Chunk (t_idx, y_idx, x_idx)
        // covers (t_idx, [2*y_idx..2*y_idx+2], [2*x_idx..2*x_idx+2]).
        for t in 0..2u64 {
            for yc in 0..2u64 {
                for xc in 0..2u64 {
                    let mut chunk = Vec::with_capacity(4);
                    for dy in 0..2u64 {
                        for dx in 0..2u64 {
                            let y = yc * 2 + dy;
                            let x = xc * 2 + dx;
                            chunk.push(base.wrapping_add((t * 16 + y * 4 + x) as u8));
                        }
                    }
                    array
                        .store_chunk_elements::<u8>(&[t, yc, xc], &chunk)
                        .unwrap();
                }
            }
        }
    }

    tmp
}

#[test]
fn indb_round_trip_emits_one_row_per_chunk_position() {
    let tmp = build_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let arr = group_to_indb_rasters(&uri, None).unwrap();

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
    //
    // Pressure has base=100; chunk (t=0, y=0, x=0) covers y∈{0,1}, x∈{0,1}
    // → pixel offsets {0, 1, 4, 5} → values {100, 101, 104, 105}.
    let pressure = r0.band(0).unwrap();
    assert_eq!(pressure.raw_source_shape(), &[1, 2, 2]);
    assert_eq!(pressure.data_type(), BandDataType::UInt8);
    assert!(pressure.is_indb());
    assert_eq!(
        &*pressure.contiguous_data().unwrap(),
        &[100u8, 101, 104, 105]
    );
    // Chunk anchor is populated on InDb rows too as provenance — the
    // `data` column carries the bytes, and `outdb_uri` records where
    // they came from.
    assert_eq!(pressure.outdb_format(), Some("zarr"));
    let anchor = pressure.outdb_uri().expect("outdb_uri set on InDb band");
    assert!(anchor.contains("#array=pressure"), "got: {anchor}");
    assert!(anchor.contains("&chunk=0,0,0"), "got: {anchor}");

    // Temperature has base=0 → same chunk holds {0, 1, 4, 5}.
    let temperature = r0.band(1).unwrap();
    assert_eq!(&*temperature.contiguous_data().unwrap(), &[0u8, 1, 4, 5]);

    // Last row corresponds to chunk (t=1, y=1, x=1). Temperature pixels:
    //   t=1, y∈{2,3}, x∈{2,3} → 1*16 + y*4 + x → 26, 27, 30, 31.
    let last = rasters.get(7).unwrap();
    let last_transform: Vec<f64> = last.transform().to_vec();
    assert_eq!(last_transform[0], 100.0 + 2.0); // x_off = 2
    assert_eq!(last_transform[3], 200.0 - 2.0); // y_off = 2, sy = -1
                                                // band 1 is temperature (per the sort-by-path order above).
    let last_temp = last.band(1).unwrap();
    assert_eq!(&*last_temp.contiguous_data().unwrap(), &[26u8, 27, 30, 31]);
}

#[test]
fn outdb_emits_chunk_anchors() {
    let tmp = build_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let arr = group_to_outdb_rasters(&uri, None).unwrap();

    let rasters = RasterStructArray::new(&arr);
    assert_eq!(rasters.len(), 8);

    // OutDb rows have empty data column and chunk anchor URIs.
    // Bands sort alphabetically by array path: pressure (band 0), then
    // temperature (band 1).
    let r0 = rasters.get(0).unwrap();
    let pressure = r0.band(0).unwrap();
    assert!(
        !pressure.is_indb(),
        "OutDb band must report is_indb() = false"
    );
    // "This is zarr" lives in outdb_format, not a URI scheme prefix.
    assert_eq!(pressure.outdb_format(), Some("zarr"));
    let anchor = pressure.outdb_uri().expect("outdb_uri set");
    // Anchor is the group URI verbatim plus a fragment carrying array
    // path + chunk indices. No `zarr://` prefix.
    assert!(anchor.starts_with("file://"), "got: {anchor}");
    assert!(!anchor.starts_with("zarr://"), "got: {anchor}");
    assert!(anchor.contains("#array=pressure"), "got: {anchor}");
    assert!(anchor.contains("&chunk=0,0,0"), "got: {anchor}");

    // Last chunk position's temperature band points at chunk (1,1,1).
    let last = rasters.get(7).unwrap();
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
    let err = group_to_indb_rasters(&uri, None).unwrap_err().to_string();
    assert!(err.contains("no child arrays"), "got: {err}");
}

/// Build a group with two 3-D data arrays and 1-D `t`/`y`/`x` coord
/// variables alongside them — the xarray-on-Zarr pattern. The loader's
/// default behaviour must drop the coord variables and read only the
/// data arrays.
#[allow(deprecated)]
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
        for t in 0..2u64 {
            for yc in 0..2u64 {
                for xc in 0..2u64 {
                    arr.store_chunk_elements::<u8>(&[t, yc, xc], &[0u8; 4])
                        .unwrap();
                }
            }
        }
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
    let arr = group_to_indb_rasters(&uri, None).unwrap();
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
    let arr = group_to_indb_rasters(&uri, Some(&filter)).unwrap();
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
    let err = group_to_indb_rasters(&uri, Some(&filter))
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
    let err = group_to_indb_rasters(&uri, None).unwrap_err().to_string();
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
    let err = group_to_indb_rasters(&uri, Some(&filter))
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
    let err = group_to_indb_rasters(&uri, None).unwrap_err().to_string();
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
    #[allow(deprecated)]
    {
        let array = ArrayBuilder::new(vec![2u64, 2], vec![1u64, 2], data_type::uint8(), 0u8)
            .attributes(attrs)
            .build(store.clone(), "/temperature")
            .unwrap();
        array.store_metadata().unwrap();
        array
            .store_chunk_elements::<u8>(&[0, 0], &[10u8, 11])
            .unwrap();
        array
            .store_chunk_elements::<u8>(&[1, 0], &[20u8, 21])
            .unwrap();
    }

    let uri = format!("file://{}", tmp.path().display());
    let arr = group_to_indb_rasters(&uri, None).unwrap();
    let rasters = RasterStructArray::new(&arr);
    assert_eq!(rasters.len(), 2);
    let r0 = rasters.get(0).unwrap();
    let band = r0.band(0).unwrap();
    assert_eq!(&*band.contiguous_data().unwrap(), &[10u8, 11]);
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
    let err = group_to_indb_rasters(&uri, None).unwrap_err().to_string();
    assert!(
        err.contains("chunk") && err.contains("array_a") && err.contains("array_b"),
        "got: {err}"
    );
}
