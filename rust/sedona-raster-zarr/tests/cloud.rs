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

use std::collections::BTreeMap;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::StructArray;
use arrow_array::cast::AsArray;
use axum::Router;
use object_store::ObjectStore;
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::RasterRef;
use sedona_raster_zarr::{ZarrChunkReader, object_store_for_uri, open_storage_from_uri};
use serde::Deserialize;
use tokio::task::JoinHandle;
use tower_http::services::ServeDir;

#[derive(Debug, Deserialize)]
struct FixtureExpectation {
    crs_authority: String,
    geotransform: [f64; 6],
    raster_arrays: Vec<String>,
}

fn fixture_manifest() -> BTreeMap<String, FixtureExpectation> {
    let path = sedona_testing::data::test_zarr("manifest.json").unwrap();
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn fixture_root() -> PathBuf {
    PathBuf::from(sedona_testing::data::test_zarr("manifest.json").unwrap())
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap()
}

async fn read_all(
    uri: &str,
    store: Arc<dyn ObjectStore>,
    arrays: Option<&[String]>,
) -> StructArray {
    let storage = open_storage_from_uri(uri, store).expect("open fixture storage");
    let reader = ZarrChunkReader::try_new(storage, uri, arrays, 1024)
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
                Some(expected_name.as_str())
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

#[tokio::test]
async fn all_fixtures_match_manifest_on_local_filesystem() {
    let store: Arc<dyn ObjectStore> = Arc::new(object_store::local::LocalFileSystem::new());
    for (name, expected) in fixture_manifest() {
        let path = PathBuf::from(sedona_testing::data::test_zarr(&name).unwrap())
            .canonicalize()
            .unwrap();
        let uri = format!("file://{}", path.display());
        let arrays =
            (name != "v3-geozarr-consolidated.zarr").then_some(expected.raster_arrays.as_slice());
        let array = read_all(&uri, Arc::clone(&store), arrays).await;
        assert_fixture(&array, &expected);
    }
}

/// Static HTTP server for fixture tests. [`ServeDir`] implements GET, HEAD,
/// byte ranges, and safe path handling for `object_store::http`.
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
    let name = "v2-cf-grid-mapping.zarr";
    let expected = fixture_manifest().remove(name).unwrap();
    let server = StaticHttpServer::start(fixture_root()).await;
    let uri = server.uri(name);
    let store = object_store_for_uri(&uri).unwrap();
    let array = read_all(&uri, store, Some(&expected.raster_arrays)).await;
    assert_fixture(&array, &expected);
}

#[tokio::test]
async fn discovers_inline_consolidated_v3_fixture_over_http() {
    let name = "v3-geozarr-consolidated.zarr";
    let expected = fixture_manifest().remove(name).unwrap();
    let server = StaticHttpServer::start(fixture_root()).await;
    let uri = server.uri(name);
    let store = object_store_for_uri(&uri).unwrap();
    let array = read_all(&uri, store, None).await;
    assert_fixture(&array, &expected);
}
