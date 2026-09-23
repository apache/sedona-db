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

//! Cross-call cost of `ZarrLoader::load` over a local store: the default
//! handle cache against re-opening every array on every call
//! (`with_array_handle_ttl(Duration::ZERO)`). Each call loads 64 chunks
//! spread over 4 arrays of one group, the shape a batch of scan rows has.
//! Both variants keep the store client, so the difference isolates the
//! array opens. On a page-cached local store those cost microseconds; on
//! object storage each is a network round trip, which this number does
//! not show.
//!
//! Run with `cargo bench -p sedona-raster-zarr --bench zarr_loader`.

use std::sync::Arc;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest};
use sedona_raster::view_entries::ViewEntries;
use sedona_raster_zarr::ZarrLoader;
use sedona_schema::raster::BandDataType;
use tempfile::TempDir;
use zarrs::array::{ArrayBuilder, FillValue, data_type as zarr_dtype};
use zarrs::group::GroupBuilder;
use zarrs_filesystem::FilesystemStore;

const ARRAYS: usize = 4;
const CHUNKS_PER_ARRAY: u64 = 16;
const CHUNK_SIDE: u64 = 64;

/// A group of `ARRAYS` uint8 arrays, each one row of `CHUNKS_PER_ARRAY`
/// square chunks, written with `zarrs_filesystem` and read back through
/// `object_store` like production does.
fn build_store(dir: &TempDir) -> String {
    let store_path = dir.path().join("bench.zarr");
    let store = Arc::new(FilesystemStore::new(&store_path).unwrap());
    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();
    for a in 0..ARRAYS {
        let array = ArrayBuilder::new(
            vec![CHUNK_SIDE, CHUNK_SIDE * CHUNKS_PER_ARRAY],
            vec![CHUNK_SIDE, CHUNK_SIDE],
            zarr_dtype::uint8(),
            FillValue::from(0u8),
        )
        .build(store.clone(), &format!("/band{a}"))
        .unwrap();
        array.store_metadata().unwrap();
        let chunk = vec![a as u8; (CHUNK_SIDE * CHUNK_SIDE) as usize];
        for j in 0..CHUNKS_PER_ARRAY {
            array.store_chunk(&[0, j], chunk.clone()).unwrap();
        }
    }
    format!("file://{}", store_path.display())
}

fn bench_load(c: &mut Criterion) {
    let tmp = TempDir::new().unwrap();
    let store_uri = build_store(&tmp);
    // Same anchor format the Zarr reader emits: `{store}#array=..&chunk=..`.
    let uris: Vec<String> = (0..ARRAYS)
        .flat_map(|a| (0..CHUNKS_PER_ARRAY).map(move |j| (a, j)))
        .map(|(a, j)| format!("{store_uri}#array=band{a}&chunk=0,{j}"))
        .collect();
    let shape = [CHUNK_SIDE as i64, CHUNK_SIDE as i64];
    let view = ViewEntries::identity_for_shape(&shape);
    let reqs: Vec<RasterLoadRequest> = uris
        .iter()
        .map(|uri| RasterLoadRequest {
            uri,
            dim_names: &["y", "x"],
            source_shape: &shape,
            view: &view,
            data_type: BandDataType::UInt8,
        })
        .collect();
    let refs: Vec<&RasterLoadRequest> = reqs.iter().collect();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("zarr_loader_load_64_chunks_4_arrays");
    let loaders = [
        ("handle_cache", ZarrLoader::new()),
        (
            "reopen_every_call",
            ZarrLoader::new().with_array_handle_ttl(Duration::ZERO),
        ),
    ];
    for (name, loader) in loaders {
        rt.block_on(loader.load(&refs)).unwrap();
        group.bench_function(name, |b| {
            b.iter(|| rt.block_on(loader.load(&refs)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_load);
criterion_main!(benches);
