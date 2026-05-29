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

//! Cloud-backed smoke tests for `ZarrChunkReader`.
//!
//! These tests reach the network and are `#[ignore]` by default. Both
//! target the public, anonymous ITS_LIVE v2 ice-velocity datacubes
//! (NASA MEaSUREs), hosted at `s3://its-live-data/` in `us-west-2`. The
//! dataset is a Zarr v2 group whose data arrays declare
//! `dimension_names = ['mid_date', 'y', 'x']` — passing the loader's
//! default spatial-dim policy — and whose coordinate variables include
//! a few short fixed-length Unicode (`<U…`) arrays with null
//! fill_values that current zarrs (0.23) can't open. The loader's
//! list-then-open-each path tolerates those per-array failures, so a
//! valid raster row still streams out.
//!
//! Run with:
//!
//! ```bash
//! AWS_SKIP_SIGNATURE=true AWS_REGION=us-west-2 \
//!     cargo test -p sedona-raster-zarr --test cloud -- --ignored --nocapture
//! ```

use std::sync::Arc;

use object_store::aws::AmazonS3Builder;
use object_store::http::HttpBuilder;
use object_store::ObjectStore;
use sedona_raster::array::RasterStructArray;
use sedona_raster_zarr::{open_storage_from_uri, ZarrChunkReader};

/// ITS_LIVE bucket layout: `s3://<BUCKET>/<KEY>` ↔
/// `https://<BUCKET>.s3.us-west-2.amazonaws.com/<KEY>`.
const ITS_LIVE_BUCKET: &str = "its-live-data";
const ITS_LIVE_KEY: &str =
    "datacubes/v2/N40W120/ITS_LIVE_vel_EPSG32610_G0120_X250000_Y5450000.zarr";

/// Arrays known to share the canonical (mid_date, y, x) layout and
/// chunk grid in the ITS_LIVE v2 datacubes. The same group also holds
/// arrays with incompatible chunk grids (e.g. `floatingice`) and
/// U-typed coord variables zarrs can't currently open, so both smokes
/// pin the read to this whitelist instead of relying on discovery.
const ITS_LIVE_ARRAYS: &[&str] = &["M11", "M12"];

fn count_rows(reader: ZarrChunkReader) -> usize {
    let mut rows = 0;
    for batch in reader {
        let batch = batch.expect("batch read ok");
        let s = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .expect("raster column is a StructArray");
        rows += RasterStructArray::new(s).len();
    }
    rows
}

fn its_live_arrays() -> Vec<String> {
    ITS_LIVE_ARRAYS.iter().map(|s| (*s).into()).collect()
}

/// `s3://its-live-data/...` via [`object_store::aws::AmazonS3Builder`].
///
/// Requires AWS credentials in env (`AWS_REGION`, `AWS_ACCESS_KEY_ID`,
/// `AWS_SECRET_ACCESS_KEY`) — or for anonymous public reads,
/// `AWS_SKIP_SIGNATURE=true` plus `AWS_REGION=us-west-2`.
#[tokio::test]
#[ignore]
async fn s3_zarr_smoke() {
    let uri = format!("s3://{ITS_LIVE_BUCKET}/{ITS_LIVE_KEY}");
    let arrays = its_live_arrays();
    let store: Arc<dyn ObjectStore> = Arc::new(
        AmazonS3Builder::from_env()
            .with_url(&uri)
            .build()
            .expect("build AmazonS3 store from env"),
    );
    let storage = open_storage_from_uri(&uri, store).expect("open_storage_from_uri");
    let reader = ZarrChunkReader::try_new(storage, &uri, Some(&arrays), 1024)
        .await
        .expect("ZarrChunkReader::try_new against ITS_LIVE on s3://");
    let rows = count_rows(reader);
    assert!(rows > 0, "expected at least one chunk row from {uri}");
}

/// `https://...s3.us-west-2.amazonaws.com/...` via
/// [`object_store::http::HttpStore`]. Same bucket, different URI scheme
/// and different storage backend — exercises the HTTPS code path
/// without depending on `PROPFIND`-style listing, which AWS S3 doesn't
/// support.
#[tokio::test]
#[ignore]
async fn https_zarr_smoke() {
    let uri = format!("https://{ITS_LIVE_BUCKET}.s3.us-west-2.amazonaws.com/{ITS_LIVE_KEY}");
    let arrays = its_live_arrays();
    // open_storage_from_uri expects a store rooted at scheme+authority
    // and applies the path as a PrefixStore itself, so build the
    // HttpStore against the bucket host only — not the full key.
    let authority = format!("https://{ITS_LIVE_BUCKET}.s3.us-west-2.amazonaws.com");
    let store: Arc<dyn ObjectStore> = Arc::new(
        HttpBuilder::new()
            .with_url(authority)
            .build()
            .expect("build HttpStore"),
    );
    let storage = open_storage_from_uri(&uri, store).expect("open_storage_from_uri");
    let reader = ZarrChunkReader::try_new(storage, &uri, Some(&arrays), 1024)
        .await
        .expect("ZarrChunkReader::try_new against ITS_LIVE on https://");
    let rows = count_rows(reader);
    assert!(rows > 0, "expected at least one chunk row from {uri}");
}
