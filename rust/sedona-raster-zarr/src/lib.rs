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

//! Zarr-backed N-D raster loader for SedonaDB.
//!
//! [`ZarrChunkReader`] is a `RecordBatchReader` that can represent a Zarr
//! group as one full raster row or walk its chunk grid lazily, emitting one
//! row per inner chunk. Each band carries either a whole-array anchor or a
//! chunk anchor in `outdb_uri`; the `data` column stays empty. The async
//! [`ZarrLoader`] resolves both forms and pushes ordinary rectangular views
//! into Zarr subset reads, assembling all intersecting chunks when needed.
//!
//! Supported stores are local files, S3, and HTTP(S).

mod coords;
mod dtype;
mod geozarr;
mod loader;
mod raster_loader;
mod source_uri;

pub use loader::ZarrChunkReader;
pub use raster_loader::{
    DEFAULT_ARRAY_HANDLE_CAPACITY, DEFAULT_ARRAY_HANDLE_TTL, DEFAULT_LOAD_CONCURRENCY, HandleStats,
    ZARR_FORMAT, ZarrLoader,
};
pub use source_uri::{object_store_for_uri, open_storage_from_uri};
