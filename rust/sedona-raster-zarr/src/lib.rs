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
//! Two entry points:
//!
//! - [`group_to_indb_rasters`] — eagerly fetches every chunk's bytes into
//!   the Arrow `data` column. Suitable for snapshots / small groups.
//! - [`group_to_outdb_rasters`] — emits chunk-anchor URIs only; bytes
//!   fetch on demand through the process-wide OutDb loader (registered
//!   separately via `sedona-raster`'s loader hook).
//!
//! Phase 1 supports local filesystem stores only; cloud backends arrive
//! with the resolver work.

pub mod dtype;
pub mod geozarr;
pub mod loader;
pub mod source_uri;

pub use loader::{group_to_indb_rasters, group_to_outdb_rasters};
