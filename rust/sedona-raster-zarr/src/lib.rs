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
//! [`ZarrChunkReader`] is a `RecordBatchReader` that walks a Zarr
//! group's chunk grid lazily, emitting one raster row per chunk
//! position with one band per array. Each row carries a chunk-anchor
//! URI in `outdb_uri`; the `data` column stays empty. Metadata-only
//! operations (`count(*)`, `RS_Envelope`, `RS_Width`, …) work directly
//! against these rows; byte-consuming kernels need an async resolver
//! that's not part of this crate.
//!
//! Local filesystem stores only — `file://` URIs or bare paths.

mod dtype;
mod geozarr;
mod loader;
mod outdb_loader;
mod source_uri;

pub use loader::ZarrChunkReader;
pub use outdb_loader::{ZarrLoader, ZARR_FORMAT};
