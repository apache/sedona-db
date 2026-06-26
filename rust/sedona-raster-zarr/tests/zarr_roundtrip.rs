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
//! `zarrs` crate, then read it back through `read_all` and
//! verify the resulting raster `StructArray`. The loader always emits
//! OutDb-style rows (empty `data`, populated `outdb_uri`), so these
//! tests assert on metadata and chunk-anchor URIs rather than pixel
//! bytes. Pixel-byte coverage lives in the loader's unit tests against
//! `retrieve_chunk_bytes`.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::StructArray;
use arrow_schema::ArrowError;
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::RasterRef;
use sedona_raster_zarr::{open_storage_from_uri, ZarrChunkReader};

/// Drain a `ZarrChunkReader` into a single `StructArray`. Fixtures in
/// this file are small (≤8 chunk rows) so they fit in one batch with a
/// generous batch_size. Anything bigger would need `arrow::compute::concat`.
async fn read_all(uri: &str, arrays: Option<&[String]>) -> Result<StructArray, ArrowError> {
    // Fixtures are file:// URIs rooted at an absolute temp path. Pass a
    // LocalFileSystem rooted at `/` — the same shape the host's
    // ObjectStoreRegistry yields for file:// — and let
    // open_storage_from_uri prefix it at the group's path.
    let store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let storage = open_storage_from_uri(uri, store)?;
    let reader = ZarrChunkReader::try_new(storage, uri, arrays, 1024).await?;
    let batches: Vec<_> = reader.collect::<Result<Vec<_>, _>>()?;
    assert!(
        batches.len() <= 1,
        "test helper assumes ≤1 batch; got {} — bump batch_size or add concat",
        batches.len()
    );
    let batch = batches.into_iter().next().ok_or_else(|| {
        ArrowError::InvalidArgumentError("ZarrChunkReader produced zero batches".into())
    })?;
    Ok(batch.column(0).as_struct().clone())
}
use sedona_schema::raster::BandDataType;
use tempfile::TempDir;
use zarrs::array::data_type;
use zarrs::array::ArrayBuilder;
use zarrs::group::{Group, GroupBuilder};
use zarrs::metadata_ext::group::consolidated_metadata::{
    ConsolidatedMetadata, ConsolidatedMetadataKind,
};
use zarrs::node::Node;
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
    // transforms below. `spatial:transform` is in affine order
    // [a, b, c, d, e, f]; the affine [1, 0, 100, 0, -1, 200] is north-up
    // with origin (100, 200), which the reader stores internally in GDAL
    // order as [100, 1, 0, 200, 0, -1].
    let mut group_attrs = serde_json::Map::new();
    group_attrs.insert(
        "spatial:transform".into(),
        serde_json::json!([1.0, 0.0, 100.0, 0.0, -1.0, 200.0]),
    );
    group_attrs.insert("proj:code".into(), serde_json::json!("EPSG:4326"));
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

/// Like [`build_fixture`], but folds a `consolidated_metadata` block into
/// the group's `zarr.json` and then *removes* each child array's
/// `zarr.json`. Discovery can then only succeed by reading the
/// consolidated block — a list-then-open-each path would re-read the
/// now-deleted per-array metadata and find nothing. This is the
/// regression fixture for stores that can't list (e.g. plain HTTPS),
/// where consolidated metadata is the only viable discovery mechanism.
fn build_consolidated_fixture() -> TempDir {
    let tmp = build_fixture();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());

    // Build the consolidated map from the on-disk hierarchy (this lists,
    // which is fine — the fixture is local) and fold it into the group.
    let metadata = Node::open(&store, "/")
        .unwrap()
        .consolidate_metadata()
        .expect("root group yields consolidated metadata");
    let consolidated = ConsolidatedMetadata {
        metadata,
        kind: ConsolidatedMetadataKind::Inline,
    };
    Group::open(store.clone(), "/")
        .unwrap()
        .set_consolidated_metadata(Some(consolidated))
        .store_metadata()
        .unwrap();

    // Remove the per-array metadata so a list-then-open read would find
    // no openable arrays; only the consolidated block remains.
    for name in ["temperature", "pressure"] {
        std::fs::remove_file(tmp.path().join(name).join("zarr.json")).unwrap();
    }

    tmp
}

#[tokio::test]
async fn discovers_child_arrays_from_consolidated_metadata() {
    let tmp = build_consolidated_fixture();
    let uri = format!("file://{}", tmp.path().display());

    // Per-array zarr.json are gone, so this only succeeds via the group's
    // consolidated_metadata block: the pre-fix list-then-open path would
    // discover zero arrays and error.
    let arr = read_all(&uri, None).await.unwrap();

    let rasters = RasterStructArray::try_new(&arr).unwrap();
    assert_eq!(
        rasters.len(),
        8,
        "8 chunk rows, discovered from consolidated metadata without per-array reads"
    );
    assert_eq!(rasters.get(0).unwrap().num_bands(), 2);
}

#[tokio::test]
async fn round_trip_emits_one_row_per_chunk_position_with_outdb_anchors() {
    let tmp = build_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let arr = read_all(&uri, None).await.unwrap();

    let rasters = RasterStructArray::try_new(&arr).unwrap();
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

#[tokio::test]
async fn errors_on_empty_group() {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();
    let uri = format!("file://{}", tmp.path().display());
    let err = read_all(&uri, None).await.unwrap_err().to_string();
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

#[tokio::test]
async fn auto_skips_1d_coord_variables() {
    let tmp = build_xarray_style_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let arr = read_all(&uri, None).await.unwrap();
    let rasters = RasterStructArray::try_new(&arr).unwrap();
    // 2*2*2 = 8 chunk positions, with 2 bands per row (pressure, temperature).
    assert_eq!(rasters.len(), 8);
    let r0 = rasters.get(0).unwrap();
    assert_eq!(r0.num_bands(), 2);
}

#[tokio::test]
async fn explicit_arrays_filter_selects_subset() {
    let tmp = build_xarray_style_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let filter = vec!["temperature".to_string()];
    let arr = read_all(&uri, Some(&filter)).await.unwrap();
    let rasters = RasterStructArray::try_new(&arr).unwrap();
    assert_eq!(rasters.len(), 8);
    let r0 = rasters.get(0).unwrap();
    assert_eq!(r0.num_bands(), 1, "only temperature should be read");
}

#[tokio::test]
async fn explicit_arrays_filter_rejects_unknown_name() {
    let tmp = build_xarray_style_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let filter = vec!["humidity".to_string()];
    let err = read_all(&uri, Some(&filter)).await.unwrap_err().to_string();
    assert!(err.contains("humidity"), "got: {err}");
    assert!(err.contains("no array named"), "got: {err}");
}

#[tokio::test]
async fn errors_when_crs_declared_without_transform() {
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
    let err = read_all(&uri, None).await.unwrap_err().to_string();
    assert!(err.contains("CRS"), "got: {err}");
    assert!(err.contains("spatial:transform"), "got: {err}");
}

#[tokio::test]
async fn explicit_arrays_filter_rejects_1d_arrays() {
    // A user explicitly naming a 1-D array gets a clear "needs 2 dims"
    // error at parse time, not a confusing downstream spatial-dim
    // resolution failure.
    let tmp = build_xarray_style_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let filter = vec!["t".to_string()];
    let err = read_all(&uri, Some(&filter)).await.unwrap_err().to_string();
    assert!(err.contains("\"t\""), "got: {err}");
    assert!(err.contains("rank 1"), "got: {err}");
    assert!(err.contains("at least 2 dimensions"), "got: {err}");
}

#[tokio::test]
async fn errors_when_group_has_only_1d_arrays() {
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
    let err = read_all(&uri, None).await.unwrap_err().to_string();
    assert!(err.contains("only 1-D arrays"), "got: {err}");
}

#[tokio::test]
async fn falls_back_to_array_dimensions_attribute() {
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
    let arr = read_all(&uri, None).await.unwrap();
    let rasters = RasterStructArray::try_new(&arr).unwrap();
    assert_eq!(rasters.len(), 2);
    let r0 = rasters.get(0).unwrap();
    let band = r0.band(0).unwrap();
    // Anchor URI populated even though the array used the v2 dim-name
    // fallback — proves the fallback feeds the rest of the pipeline.
    assert_eq!(band.outdb_format(), Some("zarr"));
    assert!(band.outdb_uri().unwrap().contains("#array=temperature"));
}

#[tokio::test]
async fn errors_on_mismatched_chunk_grids() {
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
    let err = read_all(&uri, None).await.unwrap_err().to_string();
    assert!(
        err.contains("chunk") && err.contains("array_a") && err.contains("array_b"),
        "got: {err}"
    );
}

/// A CF-style group with no `spatial:transform`: a single-chunk 2-D data array
/// `temperature` with dims `[y_name, x_name]` plus matching 1-D coordinate
/// arrays. Georeferencing must be derived from the (regularly spaced)
/// coordinate values. `&[f64]` implements `IntoArrayBytes`, so the element
/// slice goes straight to `store_chunk` (typed `store_chunk_elements` is
/// deprecated).
fn build_coord_fixture(y_name: &str, y_vals: &[f64], x_name: &str, x_vals: &[f64]) -> TempDir {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());

    // No spatial:transform, no proj:* — only coordinate arrays.
    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    // 1-D coordinate arrays, single chunk each.
    for (name, vals) in [(y_name, y_vals), (x_name, x_vals)] {
        let n = vals.len() as u64;
        let arr = ArrayBuilder::new(vec![n], vec![n], data_type::float64(), 0.0f64)
            .dimension_names(Some([name]))
            .build(store.clone(), &format!("/{name}"))
            .unwrap();
        arr.store_metadata().unwrap();
        arr.store_chunk(&[0], vals).unwrap();
    }

    // 2-D data array, dims [y_name, x_name], single chunk.
    let (ny, nx) = (y_vals.len() as u64, x_vals.len() as u64);
    let data = ArrayBuilder::new(vec![ny, nx], vec![ny, nx], data_type::uint8(), 0u8)
        .dimension_names(Some([y_name, x_name]))
        .build(store.clone(), "/temperature")
        .unwrap();
    data.store_metadata().unwrap();
    data.store_chunk(&[0, 0], vec![0u8; (ny * nx) as usize])
        .unwrap();

    tmp
}

#[tokio::test]
async fn derives_geotransform_and_crs_from_coordinate_arrays() {
    let tmp = build_coord_fixture("latitude", &[20.0, 19.0], "longitude", &[10.0, 11.0, 12.0]);
    let uri = format!("file://{}", tmp.path().display());
    let arr = read_all(&uri, None).await.unwrap();

    let rasters = RasterStructArray::try_new(&arr).unwrap();
    assert_eq!(
        rasters.len(),
        1,
        "single-chunk data array -> one raster row"
    );
    let raster = rasters.get(0).unwrap();

    // lon [10,11,12] step +1 -> scale_x 1, origin_x 10 - 0.5 = 9.5
    // lat [20,19]    step -1 -> scale_y -1, origin_y 20 - (-0.5) = 20.5
    assert_eq!(
        raster.transform().to_vec(),
        vec![9.5, 1.0, 0.0, 20.5, 0.0, -1.0]
    );
    // latitude/longitude dim names imply geographic EPSG:4326.
    assert_eq!(raster.crs(), Some("EPSG:4326"));
}

#[tokio::test]
async fn derives_transform_without_crs_for_generic_xy_dims() {
    // `y`/`x` (not lat/lon): transform is still derived from the coordinates,
    // but no CRS is inferred — the user attaches one with RS_SetCRS.
    let tmp = build_coord_fixture("y", &[20.0, 19.0], "x", &[10.0, 11.0, 12.0]);
    let uri = format!("file://{}", tmp.path().display());
    let arr = read_all(&uri, None).await.unwrap();

    let raster = RasterStructArray::try_new(&arr).unwrap().get(0).unwrap();
    assert_eq!(
        raster.transform().to_vec(),
        vec![9.5, 1.0, 0.0, 20.5, 0.0, -1.0]
    );
    assert_eq!(raster.crs(), None);
}

#[tokio::test]
async fn irregular_coordinate_arrays_fall_back_to_identity() {
    // x is irregularly spaced (0, 1, 3), so no affine can represent it; with no
    // declared CRS the reader falls back to the identity pixel transform.
    let tmp = build_coord_fixture("y", &[0.0, 1.0], "x", &[0.0, 1.0, 3.0]);
    let uri = format!("file://{}", tmp.path().display());
    let arr = read_all(&uri, None).await.unwrap();

    let raster = RasterStructArray::try_new(&arr).unwrap().get(0).unwrap();
    assert_eq!(
        raster.transform().to_vec(),
        vec![0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    );
    assert_eq!(raster.crs(), None);
}

/// A CF / rioxarray group whose CRS lives on a separate scalar grid-mapping
/// variable rather than in the group attributes. `temperature` has generic
/// `y`/`x` dims (so no CRS is inferred from the names) and links to the
/// `spatial_ref` variable via `link_attr` (`"coordinates"` as rioxarray
/// writes it, or the CF `"grid_mapping"` attribute). The scalar `spatial_ref`
/// variable carries the WKT in `crs_wkt` when `crs_wkt` is `Some` — `None`
/// builds the variable with no CRS attribute, exercising the
/// stays-CRS-less path. Regularly spaced coordinate arrays supply the
/// transform either way.
fn build_grid_mapping_fixture(crs_wkt: Option<&str>, link_attr: &str) -> TempDir {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());

    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    // 1-D coordinate arrays (regular spacing -> derivable transform).
    for (name, vals) in [("y", &[20.0, 19.0][..]), ("x", &[10.0, 11.0, 12.0][..])] {
        let n = vals.len() as u64;
        let arr = ArrayBuilder::new(vec![n], vec![n], data_type::float64(), 0.0f64)
            .dimension_names(Some([name]))
            .build(store.clone(), &format!("/{name}"))
            .unwrap();
        arr.store_metadata().unwrap();
        arr.store_chunk(&[0], vals).unwrap();
    }

    // Scalar grid-mapping variable, with the CRS attribute only when asked.
    let mut crs_attrs = serde_json::Map::new();
    if let Some(wkt) = crs_wkt {
        crs_attrs.insert("crs_wkt".into(), serde_json::json!(wkt));
    }
    let crs_var = ArrayBuilder::new(
        Vec::<u64>::new(),
        Vec::<u64>::new(),
        data_type::int64(),
        0i64,
    )
    .attributes(crs_attrs)
    .build(store.clone(), "/spatial_ref")
    .unwrap();
    crs_var.store_metadata().unwrap();

    // 2-D data array linking to the CRS variable via `link_attr`.
    let mut data_attrs = serde_json::Map::new();
    data_attrs.insert(link_attr.into(), serde_json::json!("spatial_ref"));
    let data = ArrayBuilder::new(vec![2u64, 3], vec![2u64, 3], data_type::uint8(), 0u8)
        .dimension_names(Some(["y", "x"]))
        .attributes(data_attrs)
        .build(store.clone(), "/temperature")
        .unwrap();
    data.store_metadata().unwrap();
    data.store_chunk(&[0, 0], vec![0u8; 6]).unwrap();

    tmp
}

const POLAR_STEREO_WKT: &str = "PROJCS[\"unknown\",GEOGCS[\"unknown\",DATUM[\"WGS_1984\",\
    SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],\
    UNIT[\"degree\",0.0174532925199433]],PROJECTION[\"Polar_Stereographic\"],\
    PARAMETER[\"latitude_of_origin\",-71],UNIT[\"metre\",1]]";

#[tokio::test]
async fn resolves_crs_from_cf_grid_mapping_variable() {
    // CRS is on the `spatial_ref` variable, not the group; the loader must
    // open it and lift the WKT even though `y`/`x` imply no CRS themselves.
    // rioxarray links it via the `coordinates` attribute.
    let tmp = build_grid_mapping_fixture(Some(POLAR_STEREO_WKT), "coordinates");
    let uri = format!("file://{}", tmp.path().display());
    let arr = read_all(&uri, None).await.unwrap();

    let raster = RasterStructArray::try_new(&arr).unwrap().get(0).unwrap();
    // Transform still derived from the regular coordinate arrays.
    assert_eq!(
        raster.transform().to_vec(),
        vec![9.5, 1.0, 0.0, 20.5, 0.0, -1.0]
    );
    assert_eq!(raster.crs(), Some(POLAR_STEREO_WKT));
}

#[tokio::test]
async fn resolves_crs_via_cf_grid_mapping_attribute() {
    // The CF-standard linkage: the data variable's `grid_mapping` attribute
    // names the CRS variable directly (vs. rioxarray's `coordinates` list).
    let tmp = build_grid_mapping_fixture(Some(POLAR_STEREO_WKT), "grid_mapping");
    let uri = format!("file://{}", tmp.path().display());
    let arr = read_all(&uri, None).await.unwrap();

    let raster = RasterStructArray::try_new(&arr).unwrap().get(0).unwrap();
    assert_eq!(raster.crs(), Some(POLAR_STEREO_WKT));
}

#[tokio::test]
async fn grid_mapping_variable_without_crs_attribute_stays_crs_less() {
    // The `spatial_ref` variable exists and opens, but carries no CRS
    // attribute — the group must read back CRS-less rather than erroring.
    let tmp = build_grid_mapping_fixture(None, "coordinates");
    let uri = format!("file://{}", tmp.path().display());
    let arr = read_all(&uri, None).await.unwrap();

    let raster = RasterStructArray::try_new(&arr).unwrap().get(0).unwrap();
    // Transform still derived from the coordinate arrays; just no CRS.
    assert_eq!(
        raster.transform().to_vec(),
        vec![9.5, 1.0, 0.0, 20.5, 0.0, -1.0]
    );
    assert_eq!(raster.crs(), None);
}

#[tokio::test]
async fn resolves_crs_from_grid_mapping_variable_with_arrays_filter() {
    // The grid-mapping variable isn't in the filter, but the loader opens it
    // by name — mirroring how coordinate arrays are resolved.
    let tmp = build_grid_mapping_fixture(Some(POLAR_STEREO_WKT), "coordinates");
    let uri = format!("file://{}", tmp.path().display());
    let arrays = ["temperature".to_string()];
    let arr = read_all(&uri, Some(&arrays)).await.unwrap();

    let raster = RasterStructArray::try_new(&arr).unwrap().get(0).unwrap();
    assert_eq!(raster.crs(), Some(POLAR_STEREO_WKT));
}

#[tokio::test]
async fn derives_transform_with_explicit_arrays_filter() {
    // The coordinate arrays are not in the `arrays` filter, but the reader
    // opens them by name, so derivation still works.
    let tmp = build_coord_fixture("latitude", &[20.0, 19.0], "longitude", &[10.0, 11.0, 12.0]);
    let uri = format!("file://{}", tmp.path().display());
    let arrays = ["temperature".to_string()];
    let arr = read_all(&uri, Some(&arrays)).await.unwrap();

    let raster = RasterStructArray::try_new(&arr).unwrap().get(0).unwrap();
    assert_eq!(
        raster.transform().to_vec(),
        vec![9.5, 1.0, 0.0, 20.5, 0.0, -1.0]
    );
    assert_eq!(raster.crs(), Some("EPSG:4326"));
}

// A real EPSG:3857 WKT (carries the embedded authority), as rioxarray writes to
// the `spatial_ref` variable's `crs_wkt` attribute.
const WKT_3857: &str = "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\",\
DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],\
PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]],\
PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],\
PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],\
PARAMETER[\"false_northing\",0],UNIT[\"metre\",1],AUTHORITY[\"EPSG\",\"3857\"]]";

/// CF / rioxarray group: no `proj:`/`spatial:` group attributes; georeferencing
/// lives on a scalar `spatial_ref` variable (linked via `grid_mapping`) carrying
/// `crs_wkt` + a GDAL-order `GeoTransform` string.
fn build_cf_spatial_ref_fixture() -> TempDir {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());

    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    // Data variable, linked to the grid-mapping variable by name.
    let mut data_attrs = serde_json::Map::new();
    data_attrs.insert("grid_mapping".into(), serde_json::json!("spatial_ref"));
    ArrayBuilder::new(vec![4u64, 4u64], vec![2u64, 2u64], data_type::uint8(), 0u8)
        .dimension_names(Some(["y", "x"]))
        .attributes(data_attrs)
        .build(store.clone(), "/variables")
        .unwrap()
        .store_metadata()
        .unwrap();

    // Scalar grid-mapping variable: crs_wkt + GeoTransform (already GDAL order).
    let mut sr_attrs = serde_json::Map::new();
    sr_attrs.insert("crs_wkt".into(), serde_json::json!(WKT_3857));
    sr_attrs.insert(
        "GeoTransform".into(),
        serde_json::json!("-8630308.0188 1.2874707 0 4772553.2794 0 -1.2874707"),
    );
    ArrayBuilder::new(vec![1u64], vec![1u64], data_type::uint8(), 0u8)
        .attributes(sr_attrs)
        .build(store.clone(), "/spatial_ref")
        .unwrap()
        .store_metadata()
        .unwrap();

    tmp
}

#[tokio::test]
async fn cf_spatial_ref_georeferences_from_crs_wkt_and_geotransform() {
    // The rioxarray/CF convention: CRS + affine on a `spatial_ref` variable, not
    // in group attributes. The reader resolves the CRS from `crs_wkt` and uses
    // the `GeoTransform` verbatim (GDAL order, no reorder). First chunk row sits
    // at the grid origin, so its transform equals the GeoTransform.
    let tmp = build_cf_spatial_ref_fixture();
    let uri = format!("file://{}", tmp.path().display());
    let arr = read_all(&uri, None).await.unwrap();

    let raster = RasterStructArray::try_new(&arr).unwrap().get(0).unwrap();
    assert_eq!(
        raster.transform().to_vec(),
        vec![-8630308.0188, 1.2874707, 0.0, 4772553.2794, 0.0, -1.2874707]
    );
    let crs = raster.crs().expect("CRS resolved from spatial_ref crs_wkt");
    assert!(crs.contains("3857"), "expected EPSG:3857 WKT, got: {crs}");
}
