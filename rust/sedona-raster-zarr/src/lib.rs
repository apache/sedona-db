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
//! Opens a Zarr group via the `zarrs` crate and emits one raster row per
//! chunk position. Each row's bands are the corresponding chunks of each
//! array in the group, mapped onto SedonaDB's canonical N-D raster Arrow
//! schema.
//!
//! Single entry point: [`ZarrChunkReader`] is a `RecordBatchReader`
//! that walks the group's chunk grid lazily, emitting one batch per
//! `next()` call. Each row carries chunk-anchor URIs in `outdb_uri`;
//! `data` is empty until the async OutDb resolver (registered
//! separately, lands in a follow-up) materialises the bytes.
//! Metadata-only operations (`count(*)`, `RS_Envelope`, `RS_Width`, …)
//! work today; byte-consuming kernels require the resolver to be
//! registered.
//!
//! Local filesystem stores only — `file://` URIs or bare paths.

pub mod dtype;
pub mod format_spec;
pub mod geozarr;
pub mod loader;
pub mod source_uri;
pub mod udtf;

pub use format_spec::ZarrFormatSpec;
pub use loader::ZarrChunkReader;
pub use udtf::ZarrReadFunction;
