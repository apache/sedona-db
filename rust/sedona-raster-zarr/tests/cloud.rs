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

//! Transport and interoperability tests using the fixtures in
//! `apache/sedona-testing`. HTTP reads use a small in-process Rust server,
//! keeping the tests deterministic while exercising `object_store::http`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::StructArray;
use arrow_array::cast::AsArray;
use axum::Router;
use object_store::ObjectStore;
use sedona_raster::array::RasterStructArray;
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest};
use sedona_raster::traits::RasterRef;
use sedona_raster_zarr::{
    ZarrChunkReader, ZarrLoader, object_store_for_uri, open_storage_from_uri,
};
use tokio::task::JoinHandle;
use tower_http::services::ServeDir;

#[derive(Clone, Copy, Debug)]
struct FixtureExpectation {
    name: &'static str,
    crs_authority: &'static str,
    geotransform: [f64; 6],
    raster_arrays: &'static [&'static str],
}

fn fixture_root() -> PathBuf {
    PathBuf::from(sedona_testing::data::test_zarr("v2-cf-grid-mapping.zarr").unwrap())
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap()
}

async fn read_all(uri: &str, store: Arc<dyn ObjectStore>, arrays: Option<&[&str]>) -> StructArray {
    let arrays = arrays.map(|arrays| {
        arrays
            .iter()
            .map(|array| (*array).to_string())
            .collect::<Vec<_>>()
    });
    let storage = open_storage_from_uri(uri, store).expect("open fixture storage");
    let reader = ZarrChunkReader::try_new(storage, uri, arrays.as_deref(), 1024)
        .await
        .expect("create fixture reader");
    let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(batches.len(), 1, "fixture rows should fit in one batch");
    batches[0].column(0).as_struct().clone()
}

fn assert_fixture(array: &StructArray, expected: &FixtureExpectation) {
    let rasters = RasterStructArray::try_new(array).unwrap();
    assert_eq!(
        rasters.len(),
        8,
        "fixture arrays share one 2 x 2 x 2 chunk grid"
    );

    let authority_code = expected.crs_authority.split(':').next_back().unwrap();
    let mut found_grid_origin = false;
    for i in 0..rasters.len() {
        let raster = rasters.get(i).unwrap();
        assert_eq!(raster.num_bands(), expected.raster_arrays.len());
        for (band, expected_name) in expected.raster_arrays.iter().enumerate() {
            assert_eq!(
                raster
                    .band_name(band)
                    .map(|name| name.trim_start_matches('/')),
                Some(*expected_name)
            );
        }
        let transform = raster.transform().to_vec();
        assert_eq!(
            [transform[1], transform[2], transform[4], transform[5]],
            [
                expected.geotransform[1],
                expected.geotransform[2],
                expected.geotransform[4],
                expected.geotransform[5],
            ]
        );
        found_grid_origin |= transform == expected.geotransform;
        let crs = raster.crs().expect("fixture CRS");
        assert!(
            crs.contains(authority_code),
            "expected {} in fixture CRS, got {crs}",
            expected.crs_authority
        );
    }
    assert!(
        found_grid_origin,
        "expected a chunk at the fixture's grid origin"
    );
}

async fn load_first_band(array: &StructArray) -> Vec<u8> {
    let rasters = RasterStructArray::try_new(array).unwrap();
    let raster = rasters.get(0).unwrap();
    let band = raster.band(0).unwrap();
    let dim_names = band.dim_names();
    let request = RasterLoadRequest {
        uri: band.outdb_uri().unwrap(),
        dim_names: &dim_names,
        source_shape: band.raw_source_shape(),
        view: band.view(),
        data_type: band.data_type(),
    };
    let result = ZarrLoader::new().load(&[&request]).await.unwrap();
    result[0].bytes.to_vec()
}

async fn assert_local_fixture(expected: &FixtureExpectation, arrays: Option<&[&str]>) {
    let store: Arc<dyn ObjectStore> = Arc::new(object_store::local::LocalFileSystem::new());
    let path = PathBuf::from(sedona_testing::data::test_zarr(expected.name).unwrap())
        .canonicalize()
        .unwrap();
    let uri = format!("file://{}", path.display());
    let array = read_all(&uri, store, arrays).await;
    assert_fixture(&array, expected);
}

#[tokio::test]
async fn reads_v2_cf_grid_mapping_from_local_filesystem() {
    let expected = FixtureExpectation {
        name: "v2-cf-grid-mapping.zarr",
        crs_authority: "EPSG:3857",
        geotransform: [100.0, 1.0, 0.0, 200.0, 0.0, -1.0],
        raster_arrays: &["temperature"],
    };
    assert_local_fixture(&expected, Some(expected.raster_arrays)).await;
}

#[tokio::test]
async fn reads_v3_cf_grid_mapping_from_local_filesystem() {
    let expected = FixtureExpectation {
        name: "v3-cf-grid-mapping.zarr",
        crs_authority: "EPSG:3857",
        geotransform: [100.0, 1.0, 0.0, 200.0, 0.0, -1.0],
        raster_arrays: &["precipitation"],
    };
    assert_local_fixture(&expected, Some(expected.raster_arrays)).await;
}

#[tokio::test]
async fn reads_v3_geozarr_node_bbox_from_local_filesystem() {
    let expected = FixtureExpectation {
        name: "v3-geozarr-bbox-node.zarr",
        crs_authority: "EPSG:32610",
        geotransform: [500000.0, 30.0, 0.0, 5200000.0, 0.0, -30.0],
        raster_arrays: &["elevation"],
    };
    assert_local_fixture(&expected, Some(expected.raster_arrays)).await;
}

#[tokio::test]
async fn reads_v3_geozarr_pixel_bbox_from_local_filesystem() {
    let expected = FixtureExpectation {
        name: "v3-geozarr-bbox-pixel.zarr",
        crs_authority: "EPSG:4326",
        geotransform: [-124.0, 0.5, 0.0, 50.0, 0.0, -0.5],
        raster_arrays: &["air_temperature"],
    };
    assert_local_fixture(&expected, Some(expected.raster_arrays)).await;
}

#[tokio::test]
async fn discovers_v3_geozarr_consolidated_arrays_from_local_filesystem() {
    let expected = FixtureExpectation {
        name: "v3-geozarr-consolidated.zarr",
        crs_authority: "EPSG:32610",
        geotransform: [500000.0, 30.0, 0.0, 5200000.0, 0.0, -30.0],
        raster_arrays: &["quality", "reflectance"],
    };
    assert_local_fixture(&expected, None).await;
}

#[tokio::test]
async fn reads_v3_lon_lat_coordinates_from_local_filesystem() {
    let expected = FixtureExpectation {
        name: "v3-lon-lat-coordinates.zarr",
        crs_authority: "EPSG:4326",
        geotransform: [-124.0, 0.5, 0.0, 50.0, 0.0, -0.5],
        raster_arrays: &["air_temperature"],
    };
    assert_local_fixture(&expected, Some(expected.raster_arrays)).await;
}

/// Static HTTP server for fixture tests. [`ServeDir`] implements GET, HEAD,
/// byte ranges, and safe path handling for `object_store::http`.
///
/// The local filesystem tests cover Zarr parsing, while these HTTP tests also
/// verify metadata discovery through the remote object-store path. The v2
/// test additionally dereferences one emitted chunk anchor, covering payload
/// and byte-range reads.
struct StaticHttpServer {
    address: SocketAddr,
    task: JoinHandle<()>,
}

impl StaticHttpServer {
    async fn start(root: PathBuf) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new().fallback_service(ServeDir::new(root));
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self { address, task }
    }

    fn authority(&self) -> String {
        format!("http://{}", self.address)
    }

    fn uri(&self, fixture: &str) -> String {
        format!("{}/{fixture}", self.authority())
    }
}

impl Drop for StaticHttpServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tokio::test]
async fn reads_v2_fixture_over_http() {
    let expected = FixtureExpectation {
        name: "v2-cf-grid-mapping.zarr",
        crs_authority: "EPSG:3857",
        geotransform: [100.0, 1.0, 0.0, 200.0, 0.0, -1.0],
        raster_arrays: &["temperature"],
    };
    let server = StaticHttpServer::start(fixture_root()).await;
    let uri = server.uri(expected.name);
    let store = object_store_for_uri(&uri).unwrap();
    let array = read_all(&uri, store, Some(expected.raster_arrays)).await;
    assert_fixture(&array, &expected);
    let bytes = load_first_band(&array).await;
    let pixels = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| i16::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(pixels, [0, 1, 2, 6, 7, 8]);
}

#[tokio::test]
async fn discovers_inline_consolidated_v3_fixture_over_http() {
    let expected = FixtureExpectation {
        name: "v3-geozarr-consolidated.zarr",
        crs_authority: "EPSG:32610",
        geotransform: [500000.0, 30.0, 0.0, 5200000.0, 0.0, -30.0],
        raster_arrays: &["quality", "reflectance"],
    };
    let server = StaticHttpServer::start(fixture_root()).await;
    let uri = server.uri(expected.name);
    let store = object_store_for_uri(&uri).unwrap();
    let array = read_all(&uri, store, None).await;
    assert_fixture(&array, &expected);
}
