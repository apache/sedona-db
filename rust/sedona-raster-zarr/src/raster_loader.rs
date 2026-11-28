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

//! Zarr backend implementing [`sedona_raster::raster_loader::AsyncRasterLoader`].
//!
//! Resolves a band's OutDb URI back into a Zarr array or chunk read. The URI
//! is an anchor of the form
//! `<store_uri>#array=<array_path>[&chunk=<i0>,<i1>,...]`. Whole-array
//! anchors may materialize several chunks, and ordinary unit-stride views are
//! pushed into Zarr subset reads. For an array stored with
//! the `sharding_indexed` codec the indices address an inner chunk, and
//! the read fetches just that chunk's byte range out of its shard (plus
//! the shard's index, once per shard and opened array) rather than
//! decoding the whole shard.
//!
//! `load` treats its request slice as a batch (see the trait docs): the
//! anchors are first reduced to the distinct stores, arrays and read regions
//! they touch, then each store is built once, each array is
//! opened once (one metadata round trip per distinct array rather than
//! per request), and each distinct chunk is fetched once. Reads in flight
//! are bounded by the loader's I/O budget ([`ZarrLoader::concurrency`]),
//! which every concurrent `load` call on the loader and its clones shares:
//! DataFusion runs one call per partition at once, and the budget is what
//! the store sees in total, not per partition. Requests that name the
//! same region share the loaded bytes. Across calls the loader keeps the
//! store clients and opened arrays (see [`ZarrLoader`]), so a later batch
//! over the same group pays neither again. All I/O goes through `zarrs`'s
//! async API over `object_store`, so nothing blocks the caller's runtime.
//!
//! Registered against the per-session
//! [`RasterLoaderRegistry`](sedona_raster::raster_loader::RasterLoaderRegistry);
//! the loader claims the [`ZARR_FORMAT`] `outdb_format` via
//! `supports_format`. As an out-of-tree plugin, `sedona-raster-zarr` does
//! not depend on `sedona` — callers wire the registration themselves from
//! their `SedonaContext` setup:
//!
//! ```ignore
//! ctx.register_raster_loader(std::sync::Arc::new(sedona_raster_zarr::ZarrLoader::new()));
//! ```

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use lru::LruCache;
use object_store::ObjectStore;
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest, RasterLoadResult};
use sedona_raster::view_entries::ViewEntries;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use zarrs::array::{
    Array, ArrayBytes, ArrayShardedExt, ArraySubset, AsyncArrayShardedReadableExt,
    AsyncArrayShardedReadableExtCache, ChunkGrid, CodecOptions,
};
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{
    AsyncMaybeBytesIterator, AsyncReadableStorage, AsyncReadableStorageTraits, MaybeBytes,
    StorageError, StoreKey,
};

use crate::dtype::zarr_to_band_data_type;
use crate::source_uri::{
    object_store_for_uri, open_storage_from_uri, parse_zarr_anchor, store_client_key,
};

/// Format key the loader registers under. Keep in sync with
/// `outdb_format` values emitted by the Zarr reader's band builder
/// (see `crate::loader`).
pub const ZARR_FORMAT: &str = "zarr";

/// Default I/O budget: chunk reads in flight at once across every
/// concurrent `load` call on a loader (all partitions together), not per
/// call.
///
/// Chosen from `benches/zarr_io_concurrency.rs` (256 chunks of 16 KiB,
/// one caller and eight callers sharing the budget, Apple M-series, 12
/// cores). Against an HTTP store adding 10 ms to every request, time halves
/// with every doubling of the budget up to 64 (3.2 s at 1, 415 ms at 8,
/// 114 ms at 32, 64 ms at 64, with one or eight callers alike) and is worse
/// again at 128 (89-104 ms). Against the page-cached local filesystem the
/// best budget is 8 to 16 (3.2 ms with one caller, 4.1 ms with eight) and
/// 64 costs 21% more with one caller (3.9 ms) and 5% more with eight
/// (4.3 ms). 64 also equals the fan-out eight partitions had before the
/// budget was shared (8 per call). Memory in flight is the budget times
/// the chunk size, so a larger default was not taken.
pub const DEFAULT_LOAD_CONCURRENCY: usize = 64;

/// Default number of opened arrays a loader keeps across `load` calls.
pub const DEFAULT_ARRAY_HANDLE_CAPACITY: usize = 256;

/// Default time an opened array stays usable. A `load` that needs the
/// array after this re-reads its metadata, which is the only guard
/// against an array rewritten or removed under a running session: with
/// a live handle, a chunk whose key has been deleted comes back as the
/// array's fill value rather than as an open error, and a rewritten shard
/// is read through its cached index, until the TTL lapses.
pub const DEFAULT_ARRAY_HANDLE_TTL: Duration = Duration::from_secs(300);

/// Async raster byte loader for Zarr-backed bands.
///
/// Within one `load` call the batch is reduced to its distinct stores,
/// arrays and chunks (`LoadPlan`). Across calls the loader keeps two
/// kinds of handle, never chunk bytes: the `ObjectStore` client per
/// scheme and authority (a connection pool; see `store_client_key`) and
/// the opened [`Array`] per `(store URI, array path)`, whose open is a
/// metadata round trip. Clones share the same handles.
///
/// Building the store client here is the in-process bridge until a
/// credentialed store can be passed in from DataFusion's
/// `ObjectStoreRegistry`; see the loader-FFI follow-up. Because the client
/// is kept, the environment (credentials, endpoint, region) is read once
/// per scheme and authority for the life of the loader, not on every call.
#[derive(Debug, Clone)]
pub struct ZarrLoader {
    concurrency: usize,
    /// Shared by the storage backing every opened array, including cached
    /// shard decoders, so codec fan-out cannot bypass the budget.
    budget: Arc<IoBudget>,
    handles: Arc<HandleCache>,
}

/// In-flight storage reads, for tests and benchmarks.
#[derive(Debug, Default)]
struct IoCounters {
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
}

impl IoCounters {
    fn enter(&self, permits: usize) {
        let now = self.in_flight.fetch_add(permits, Ordering::SeqCst) + permits;
        self.peak_in_flight.fetch_max(now, Ordering::SeqCst);
    }

    fn exit(&self, permits: usize) {
        self.in_flight.fetch_sub(permits, Ordering::SeqCst);
    }
}

/// Bound actual storage reads: zarrs' codec concurrency is only a target,
/// and its nested chunk/shard scheduling can exceed that target.
#[derive(Debug)]
struct IoBudget {
    permits: Arc<Semaphore>,
    io: Arc<IoCounters>,
}

impl IoBudget {
    fn new(concurrency: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(concurrency)),
            io: Arc::new(IoCounters::default()),
        }
    }

    async fn acquire(&self) -> Result<IoPermit, StorageError> {
        let permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| StorageError::Other("Zarr loader: I/O budget closed".into()))?;
        self.io.enter(1);
        Ok(IoPermit {
            _permit: permit,
            io: self.io.clone(),
        })
    }
}

struct IoPermit {
    _permit: OwnedSemaphorePermit,
    io: Arc<IoCounters>,
}

impl Drop for IoPermit {
    fn drop(&mut self) {
        self.io.exit(1);
    }
}

struct BudgetedStorage {
    inner: AsyncReadableStorage,
    budget: Arc<IoBudget>,
}

#[async_trait]
impl AsyncReadableStorageTraits for BudgetedStorage {
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let _permit = self.budget.acquire().await?;
        self.inner.get(key).await
    }

    async fn get_partial(
        &self,
        key: &StoreKey,
        range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        let _permit = self.budget.acquire().await?;
        self.inner.get_partial(key, range).await
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        // Do not hand a batch of ranges to the underlying store under one
        // permit: it may fetch those ranges concurrently. Each range goes
        // through the same budget as single-range and whole-chunk reads.
        let mut bytes = Vec::new();
        for range in ranges {
            let Some(value) = self.get_partial(key, range).await? else {
                return Ok(None);
            };
            bytes.push(Ok(value));
        }
        Ok(Some(stream::iter(bytes).boxed()))
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let _permit = self.budget.acquire().await?;
        self.inner.size_key(key).await
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

impl Default for ZarrLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ZarrLoader {
    pub fn new() -> Self {
        let budget = Arc::new(IoBudget::new(DEFAULT_LOAD_CONCURRENCY));
        Self {
            concurrency: DEFAULT_LOAD_CONCURRENCY,
            budget: budget.clone(),
            handles: Arc::new(HandleCache::new(
                DEFAULT_ARRAY_HANDLE_CAPACITY,
                DEFAULT_ARRAY_HANDLE_TTL,
                budget,
            )),
        }
    }

    /// The I/O budget: chunk reads in flight at once, counted across every
    /// concurrent `load` call on this loader and its clones. Clamped to at
    /// least 1. Starts a fresh budget and handle cache, so configure before
    /// the loader is shared.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        debug_assert!(
            Arc::strong_count(&self.handles) == 1,
            "configure the I/O budget before sharing the loader"
        );
        self.concurrency = concurrency.max(1);
        self.budget = Arc::new(IoBudget::new(self.concurrency));
        // Cached decoders retain their storage and budget. Reopen them when
        // configuring a new budget, rather than leaving reads on the old one.
        self.handles = Arc::new(HandleCache::new(
            self.handles.capacity,
            self.handles.ttl,
            self.budget.clone(),
        ));
        self
    }

    /// The configured I/O budget: the maximum number of chunk reads in
    /// flight at once across all concurrent calls.
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// The most storage reads this loader has had in flight at once under
    /// its current budget. Includes shard-index reads. For tests and benchmarks.
    pub fn peak_in_flight(&self) -> usize {
        self.budget.io.peak_in_flight.load(Ordering::SeqCst)
    }

    /// Number of opened arrays kept across calls. Clamped to at least 1.
    /// Starts a fresh handle cache, so configure before the loader is
    /// shared.
    pub fn with_array_handle_capacity(mut self, capacity: usize) -> Self {
        debug_assert!(
            Arc::strong_count(&self.handles) == 1,
            "configure the handle cache before sharing the loader"
        );
        self.handles = Arc::new(HandleCache::new(
            capacity,
            self.handles.ttl,
            self.budget.clone(),
        ));
        self
    }

    /// How long an opened array stays usable before it is re-opened.
    /// `Duration::ZERO` re-opens on every call. Starts a fresh handle
    /// cache, so configure before the loader is shared.
    pub fn with_array_handle_ttl(mut self, ttl: Duration) -> Self {
        debug_assert!(
            Arc::strong_count(&self.handles) == 1,
            "configure the handle cache before sharing the loader"
        );
        self.handles = Arc::new(HandleCache::new(
            self.handles.capacity,
            ttl,
            self.budget.clone(),
        ));
        self
    }

    /// Counters for the cross-call handle reuse, for tests and diagnostics.
    pub fn handle_stats(&self) -> HandleStats {
        self.handles.stats()
    }
}

/// Counters exposed by [`ZarrLoader::handle_stats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct HandleStats {
    /// `ObjectStore` clients reused for a scheme and authority seen before.
    pub store_hits: u64,
    /// `ObjectStore` clients built: one per distinct scheme and authority.
    pub store_misses: u64,
    /// Array opens avoided by an unexpired handle from an earlier call.
    pub array_hits: u64,
    /// Arrays opened, each a metadata round trip: first use, or the handle
    /// had expired or been evicted.
    pub array_misses: u64,
}

/// An opened array together with the shard-index caches its chunk reads
/// go through. The caches are what make a sharded array's inner-chunk
/// reads cheap: a shard's index is fetched and decoded on the first read
/// into that shard and reused by every later read into it, by this call
/// and by later calls while the handle lives. Each holds the shard's
/// decoded index (16 bytes per inner chunk) and nothing of the pixel data;
/// they are dropped with the handle. For an unsharded array the one cache
/// stays empty and reads go straight to the chunk.
///
/// There is one cache per shard, not one per array, because zarrs' cache
/// holds its lock while it fetches an index. Shared by every shard, it
/// would make the first reads of different shards wait for each other's
/// fetches; per shard, only reads of the same shard wait, and they would
/// have waited for that index anyway.
///
/// This per-shard map is a workaround for zarrs 0.23. Unreleased zarrs
/// removes `ArrayShardedReadableExt` in favour of `ArrayCached` and async
/// chunk caches that load each key once without a cache-wide lock. Remove
/// the map when we move to the zarrs release that has them.
struct OpenedArray {
    array: Array<dyn AsyncReadableStorageTraits>,
    /// The inner-chunk grid, for finding the shard an inner chunk is in;
    /// `None` for an unsharded array.
    subchunk_grid: Option<ChunkGrid>,
    shards: Mutex<HashMap<Vec<u64>, Arc<AsyncArrayShardedReadableExtCache>>>,
}

impl OpenedArray {
    fn new(array: Array<dyn AsyncReadableStorageTraits>) -> Self {
        let subchunk_grid = array.is_sharded().then(|| array.subchunk_grid());
        Self {
            array,
            subchunk_grid,
            shards: Mutex::new(HashMap::new()),
        }
    }

    /// The shard-index cache for the shard holding inner chunk
    /// `chunk_indices`, or the array's one cache when it is unsharded.
    fn shard_cache(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<AsyncArrayShardedReadableExtCache>, String> {
        let shard = match &self.subchunk_grid {
            Some(grid) => {
                let subset = grid
                    .subset(chunk_indices)
                    .map_err(|e| e.to_string())?
                    .ok_or("inner chunk is outside the array")?;
                let shards = self
                    .array
                    .chunks_in_array_subset(&subset)
                    .map_err(|e| e.to_string())?
                    .ok_or("inner chunk is outside the shard grid")?;
                shards.start().to_vec()
            }
            None => Vec::new(),
        };
        Ok(lock(&self.shards)
            .entry(shard)
            .or_insert_with(|| Arc::new(AsyncArrayShardedReadableExtCache::new(&self.array)))
            .clone())
    }

    /// Effective inner-chunk shape in array axis order.
    fn chunk_shape(&self) -> Result<Vec<u64>, ArrowError> {
        if let Some(shape) = self.array.effective_subchunk_shape() {
            return Ok(shape.iter().map(|n| n.get()).collect());
        }
        self.array
            .chunk_shape(&vec![0; self.array.dimensionality()])
            .map(|shape| shape.iter().map(|n| n.get()).collect())
            .map_err(|e| {
                ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                    "failed to query Zarr chunk shape for {}: {e}",
                    self.array.path()
                )))
            })
    }

    /// Cache used for an array subset. For sharded arrays this is keyed by
    /// the first shard the subset touches, preserving reuse for repeated
    /// views while avoiding a single cache-wide lock across unrelated reads.
    fn subset_cache(
        &self,
        subset: &ArraySubset,
    ) -> Result<Arc<AsyncArrayShardedReadableExtCache>, ArrowError> {
        let shard = if self.subchunk_grid.is_some() {
            self.array
                .chunks_in_array_subset(subset)
                .map_err(|e| {
                    ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                        "failed to find shards for subset {subset} of {}: {e}",
                        self.array.path()
                    )))
                })?
                .filter(|shards| shards.num_elements() > 0)
                .map(|shards| shards.start().to_vec())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(lock(&self.shards)
            .entry(shard)
            .or_insert_with(|| Arc::new(AsyncArrayShardedReadableExtCache::new(&self.array)))
            .clone())
    }
}

type SharedArray = Arc<OpenedArray>;

struct ArrayHandle {
    array: SharedArray,
    opened_at: Instant,
}

/// `(store URI, array path)`: what identifies an opened array.
type ArrayKey = (String, String);

/// Taken by the calls racing to open one array, so that one of them does.
type OpenGate = Arc<tokio::sync::Mutex<()>>;

/// Removes an array's gate from `opening` when the call holding it finishes
/// or is dropped. Only that gate: a later call may have put in a new one.
struct GateRelease<'a> {
    opening: &'a Mutex<HashMap<ArrayKey, OpenGate>>,
    key: &'a ArrayKey,
    gate: &'a OpenGate,
}

impl Drop for GateRelease<'_> {
    fn drop(&mut self) {
        let mut opening = lock(self.opening);
        if opening
            .get(self.key)
            .is_some_and(|gate| Arc::ptr_eq(gate, self.gate))
        {
            opening.remove(self.key);
        }
    }
}

/// The cross-call state of a [`ZarrLoader`]. The map locks are held only
/// around map access, never across an await. Calls racing to open the same
/// array (partitions loading chunks of one array at once) take that array's
/// gate in `opening` in turn: the first opens it and the rest find the
/// handle in the cache, so an array is opened once however its rows are
/// partitioned.
struct HandleCache {
    budget: Arc<IoBudget>,
    capacity: usize,
    ttl: Duration,
    stores: Mutex<HashMap<String, Arc<dyn ObjectStore>>>,
    arrays: Mutex<LruCache<ArrayKey, ArrayHandle>>,
    /// One gate per array being opened right now.
    opening: Mutex<HashMap<ArrayKey, OpenGate>>,
    store_hits: AtomicU64,
    store_misses: AtomicU64,
    array_misses: AtomicU64,
    array_hits: AtomicU64,
}

impl HandleCache {
    fn new(capacity: usize, ttl: Duration, budget: Arc<IoBudget>) -> Self {
        let capacity = capacity.max(1);
        Self {
            budget,
            capacity,
            ttl,
            stores: Mutex::new(HashMap::new()),
            arrays: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity).expect("capacity clamped to at least 1"),
            )),
            opening: Mutex::new(HashMap::new()),
            store_hits: AtomicU64::new(0),
            store_misses: AtomicU64::new(0),
            array_misses: AtomicU64::new(0),
            array_hits: AtomicU64::new(0),
        }
    }

    fn stats(&self) -> HandleStats {
        HandleStats {
            store_hits: self.store_hits.load(Ordering::Relaxed),
            store_misses: self.store_misses.load(Ordering::Relaxed),
            array_misses: self.array_misses.load(Ordering::Relaxed),
            array_hits: self.array_hits.load(Ordering::Relaxed),
        }
    }

    /// The `ObjectStore` client for `store_uri`, built on first use.
    fn store(&self, store_uri: &str) -> Result<Arc<dyn ObjectStore>, ArrowError> {
        let key = store_client_key(store_uri)?;
        if let Some(store) = lock(&self.stores).get(&key) {
            self.store_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(store.clone());
        }
        let store = object_store_for_uri(store_uri)?;
        self.store_misses.fetch_add(1, Ordering::Relaxed);
        Ok(lock(&self.stores).entry(key).or_insert(store).clone())
    }

    /// The opened array at `array_path` in `store_uri`, reusing an
    /// unexpired handle from an earlier call or from a concurrent one.
    async fn array(&self, store_uri: &str, array_path: &str) -> Result<SharedArray, ArrowError> {
        let key: ArrayKey = (store_uri.to_string(), array_path.to_string());
        if let Some(array) = self.cached(&key) {
            return Ok(array);
        }
        let gate = lock(&self.opening)
            .entry(key.clone())
            .or_insert_with(|| OpenGate::new(tokio::sync::Mutex::new(())))
            .clone();
        let _opening = gate.lock().await;
        // Later callers start from the cache; the ones still queued on this
        // gate hold their own reference to it and re-check the cache too.
        // Removed on drop so a call cancelled during the open leaves no entry.
        let _release = GateRelease {
            opening: &self.opening,
            key: &key,
            gate: &gate,
        };
        // Another call may have opened it while this one waited for the gate.
        if let Some(array) = self.cached(&key) {
            return Ok(array);
        }
        self.open(store_uri, array_path).await
    }

    /// The unexpired handle for `key`, counted as a hit.
    fn cached(&self, key: &ArrayKey) -> Option<SharedArray> {
        let mut arrays = lock(&self.arrays);
        if let Some(handle) = arrays.get(key) {
            if handle.opened_at.elapsed() < self.ttl {
                self.array_hits.fetch_add(1, Ordering::Relaxed);
                return Some(handle.array.clone());
            }
            arrays.pop(key);
        }
        None
    }

    /// Open the array and cache its handle, counted as a miss.
    async fn open(&self, store_uri: &str, array_path: &str) -> Result<SharedArray, ArrowError> {
        let storage = open_storage_from_uri(store_uri, self.store(store_uri)?)?;
        let array = Array::async_open(storage.clone(), array_path)
            .await
            .map_err(|e| {
                ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                    "failed to open Zarr array {array_path} in {store_uri}: {e}"
                )))
            })?;
        self.array_misses.fetch_add(1, Ordering::Relaxed);
        let storage: AsyncReadableStorage = Arc::new(BudgetedStorage {
            inner: storage,
            budget: self.budget.clone(),
        });
        let array: SharedArray = Arc::new(OpenedArray::new(array.with_storage(storage)));
        lock(&self.arrays).put(
            (store_uri.to_string(), array_path.to_string()),
            ArrayHandle {
                array: array.clone(),
                opened_at: Instant::now(),
            },
        );
        Ok(array)
    }
}

impl fmt::Debug for HandleCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HandleCache")
            .field("capacity", &self.capacity)
            .field("ttl", &self.ttl)
            .field("stats", &self.stats())
            .finish()
    }
}

/// A poisoned lock only means another `load` panicked mid-update; the
/// maps stay usable, so recover rather than propagate.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A batch of array/chunk anchors reduced to its distinct stores and arrays.
#[derive(Debug, PartialEq, Eq)]
struct AnchorPlan {
    /// Distinct store URIs.
    stores: Vec<String>,
    /// Distinct `(store index, absolute array path)` pairs.
    arrays: Vec<(usize, String)>,
    /// Parsed anchor for each request, in request order.
    requests: Vec<PlannedAnchor>,
}

#[derive(Debug, PartialEq, Eq)]
struct PlannedAnchor {
    array: usize,
    chunk_indices: Option<Vec<u64>>,
}

impl AnchorPlan {
    fn build<'a>(uris: impl IntoIterator<Item = &'a str>) -> Result<Self, ArrowError> {
        let mut plan = Self {
            stores: Vec::new(),
            arrays: Vec::new(),
            requests: Vec::new(),
        };
        let mut store_ids = HashMap::new();
        let mut array_ids = HashMap::new();

        for uri in uris {
            let anchor = parse_zarr_anchor(uri)?;
            let array_path = if anchor.array_path.starts_with('/') {
                anchor.array_path
            } else {
                format!("/{}", anchor.array_path)
            };
            let store = intern(&mut store_ids, &mut plan.stores, anchor.store_uri);
            let array = intern(&mut array_ids, &mut plan.arrays, (store, array_path));
            plan.requests.push(PlannedAnchor {
                array,
                chunk_indices: anchor.chunk_indices,
            });
        }
        Ok(plan)
    }
}

/// One physical read. Array subsets may cross any number of chunks; chunk
/// reads preserve the fill-padded edge-chunk behaviour of tiled scans.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ReadRegion {
    Chunk(Vec<u64>),
    ArraySubset { start: Vec<u64>, shape: Vec<u64> },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReadKey {
    array: usize,
    region: ReadRegion,
}

struct PlannedOutput {
    read: usize,
    source_shape: Vec<i64>,
    view: ViewEntries,
}

/// Index of `key` in `items`, appending it (and recording it in `ids`) on
/// first sight.
fn intern<K: Clone + Eq + Hash>(ids: &mut HashMap<K, usize>, items: &mut Vec<K>, key: K) -> usize {
    if let Some(&id) = ids.get(&key) {
        return id;
    }
    let id = items.len();
    items.push(key.clone());
    ids.insert(key, id);
    id
}

#[async_trait]
impl AsyncRasterLoader for ZarrLoader {
    fn name(&self) -> &str {
        ZARR_FORMAT
    }

    /// Zarr-specific: claims only bands whose `outdb_format` is
    /// [`ZARR_FORMAT`]. Everything else (including the unset `None` that
    /// `RS_FromPath` emits) falls through to the catch-all GDAL loader.
    fn supports_format(&self, format: Option<&str>) -> bool {
        format == Some(ZARR_FORMAT)
    }

    async fn load(&self, reqs: &[&RasterLoadRequest]) -> Result<Vec<RasterLoadResult>, ArrowError> {
        let plan = AnchorPlan::build(reqs.iter().map(|req| req.uri))?;

        // One opened array per distinct array in the batch. The open is a
        // metadata round trip, so it is paid once per batch here and, via
        // the handle cache, usually not even that across batches.
        let mut arrays: Vec<SharedArray> = Vec::with_capacity(plan.arrays.len());
        for (store, array_path) in &plan.arrays {
            arrays.push(self.handles.array(&plan.stores[*store], array_path).await?);
        }

        // Validate and reduce requests to distinct physical reads. The read
        // key includes its subset, so two views of one array only share bytes
        // when they resolve to the same region.
        let mut reads = Vec::new();
        let mut read_ids = HashMap::new();
        let mut outputs = Vec::with_capacity(reqs.len());
        for (req, anchor) in reqs.iter().zip(&plan.requests) {
            let array = &arrays[anchor.array];
            let array_path = &plan.arrays[anchor.array].1;
            let file_dtype = zarr_to_band_data_type(array.array.data_type())?;
            if file_dtype != req.data_type {
                return Err(ArrowError::ExternalError(Box::new(
                    sedona_internal_datafusion_err!(
                        "Zarr OutDb band metadata claims {:?} but array {} is {:?} ({})",
                        req.data_type,
                        array_path,
                        file_dtype,
                        req.uri
                    ),
                )));
            }

            req.view.validate(req.source_shape).map_err(|e| {
                ArrowError::InvalidArgumentError(format!(
                    "Zarr OutDb request has an invalid view for source_shape {:?}: {e}",
                    req.source_shape
                ))
            })?;

            let array_shape = array.array.shape();
            let chunk_shape = array.chunk_shape()?;
            let expected_shape = match &anchor.chunk_indices {
                Some(indices) => {
                    if indices.len() != array_shape.len() {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "Zarr chunk anchor has {} indices but array {array_path} has {} dimensions ({})",
                            indices.len(),
                            array_shape.len(),
                            req.uri
                        )));
                    }
                    &chunk_shape
                }
                None => array_shape,
            };
            let expected_shape_i64 = shape_i64(expected_shape, req.uri)?;
            if req.source_shape != expected_shape_i64 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Zarr OutDb source_shape {:?} does not match {} shape {:?} ({})",
                    req.source_shape,
                    if anchor.chunk_indices.is_some() {
                        "chunk"
                    } else {
                        "array"
                    },
                    expected_shape_i64,
                    req.uri
                )));
            }

            let native = native_subset(req.view);
            let (region, source_shape, view) = match (&anchor.chunk_indices, native) {
                // Preserve whole edge chunks, including fill padding past the
                // array shape, exactly as tiled scans have always done.
                (Some(indices), _) if req.view.is_identity(req.source_shape) => (
                    ReadRegion::Chunk(indices.clone()),
                    req.source_shape.to_vec(),
                    req.view.clone(),
                ),
                (Some(indices), Some((local_start, shape))) => {
                    let start = offset_chunk_subset(indices, &chunk_shape, &local_start)?;
                    if subset_in_bounds(&start, &shape, array_shape) {
                        let visible_shape = shape_i64(&shape, req.uri)?;
                        let identity = ViewEntries::identity_for_shape(&visible_shape);
                        (
                            ReadRegion::ArraySubset { start, shape },
                            visible_shape,
                            identity,
                        )
                    } else {
                        (
                            ReadRegion::Chunk(indices.clone()),
                            req.source_shape.to_vec(),
                            req.view.clone(),
                        )
                    }
                }
                (Some(indices), None) => (
                    ReadRegion::Chunk(indices.clone()),
                    req.source_shape.to_vec(),
                    req.view.clone(),
                ),
                (None, Some((start, shape))) => {
                    let visible_shape = shape_i64(&shape, req.uri)?;
                    let identity = ViewEntries::identity_for_shape(&visible_shape);
                    (
                        ReadRegion::ArraySubset { start, shape },
                        visible_shape,
                        identity,
                    )
                }
                (None, None) => (
                    ReadRegion::ArraySubset {
                        start: vec![0; array_shape.len()],
                        shape: array_shape.to_vec(),
                    },
                    req.source_shape.to_vec(),
                    req.view.clone(),
                ),
            };

            let read = intern(
                &mut read_ids,
                &mut reads,
                ReadKey {
                    array: anchor.array,
                    region,
                },
            );
            outputs.push(PlannedOutput {
                read,
                source_shape,
                view,
            });
        }

        // Fetch distinct regions concurrently. The storage adapter enforces
        // the shared I/O budget even when zarrs fans a region out over chunks.
        // Results complete out of order, so retain the index for scattering.
        let mut fetches = Vec::with_capacity(reads.len());
        for (read_idx, read) in reads.iter().enumerate() {
            let array: &OpenedArray = &arrays[read.array];
            let (store, array_path) = &plan.arrays[read.array];
            let store_uri = plan.stores[*store].as_str();
            fetches.push(async move {
                retrieve_region(array, store_uri, array_path, &read.region, self.concurrency)
                    .await
                    .map(|buffer| (read_idx, buffer))
            });
        }
        let fetched: Vec<(usize, Buffer)> = stream::iter(fetches)
            .buffer_unordered(self.concurrency)
            .try_collect()
            .await?;
        let mut buffers: Vec<Option<Buffer>> = vec![None; reads.len()];
        for (read_idx, buffer) in fetched {
            buffers[read_idx] = Some(buffer);
        }

        // Scatter back in request order. Requests naming the same region
        // share one loaded `Buffer` (refcounted clone, no copy).
        outputs
            .into_iter()
            .map(|output| {
                let buffer = buffers[output.read].clone().ok_or_else(|| {
                    ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                        "Zarr loader: read {} of {} was never fetched",
                        output.read,
                        reads.len()
                    )))
                })?;
                Ok(RasterLoadResult {
                    bytes: buffer,
                    source_shape: output.source_shape,
                    view: output.view,
                })
            })
            .collect()
    }
}

/// Return a rectangular, unit-stride subset in source-axis order. Views with
/// permutation, stride, reversal, or broadcast retain the full-source fallback.
fn native_subset(view: &ViewEntries) -> Option<(Vec<u64>, Vec<u64>)> {
    let mut start = Vec::with_capacity(view.len());
    let mut shape = Vec::with_capacity(view.len());
    for (axis, entry) in view.iter().enumerate() {
        if entry.source_axis != axis as i64 || entry.step != 1 || entry.start < 0 || entry.steps < 0
        {
            return None;
        }
        start.push(u64::try_from(entry.start).ok()?);
        shape.push(u64::try_from(entry.steps).ok()?);
    }
    Some((start, shape))
}

fn shape_i64(shape: &[u64], uri: &str) -> Result<Vec<i64>, ArrowError> {
    shape
        .iter()
        .map(|&n| {
            i64::try_from(n).map_err(|_| {
                ArrowError::InvalidArgumentError(format!(
                    "Zarr dimension {n} exceeds i64::MAX ({uri})"
                ))
            })
        })
        .collect()
}

fn offset_chunk_subset(
    chunk_indices: &[u64],
    chunk_shape: &[u64],
    local_start: &[u64],
) -> Result<Vec<u64>, ArrowError> {
    chunk_indices
        .iter()
        .zip(chunk_shape)
        .zip(local_start)
        .map(|((&index, &size), &local)| {
            index
                .checked_mul(size)
                .and_then(|base| base.checked_add(local))
                .ok_or_else(|| {
                    ArrowError::InvalidArgumentError(format!(
                        "Zarr chunk subset offset overflows u64: {index} * {size} + {local}"
                    ))
                })
        })
        .collect()
}

fn subset_in_bounds(start: &[u64], shape: &[u64], array_shape: &[u64]) -> bool {
    start
        .iter()
        .zip(shape)
        .zip(array_shape)
        .all(|((&start, &shape), &extent)| {
            start.checked_add(shape).is_some_and(|end| end <= extent)
        })
}

async fn retrieve_region(
    opened: &OpenedArray,
    store_uri: &str,
    array_path: &str,
    region: &ReadRegion,
    concurrent_target: usize,
) -> Result<Buffer, ArrowError> {
    match region {
        ReadRegion::Chunk(indices) => retrieve_chunk(opened, store_uri, array_path, indices).await,
        ReadRegion::ArraySubset { start, shape } => {
            retrieve_array_subset(
                opened,
                store_uri,
                array_path,
                start,
                shape,
                concurrent_target,
            )
            .await
        }
    }
}

/// Read a rectangular array subset, assembling all intersecting chunks in
/// C-order. The codec concurrency target encourages parallel reads; the
/// storage adapter enforces the loader-wide I/O limit.
async fn retrieve_array_subset(
    opened: &OpenedArray,
    store_uri: &str,
    array_path: &str,
    start: &[u64],
    shape: &[u64],
    concurrent_target: usize,
) -> Result<Buffer, ArrowError> {
    let subset = ArraySubset::new_with_start_shape(start.to_vec(), shape.to_vec()).map_err(|e| {
        ArrowError::InvalidArgumentError(format!(
            "invalid Zarr subset start={start:?} shape={shape:?} for {array_path} in {store_uri}: {e}"
        ))
    })?;
    let cache = opened.subset_cache(&subset)?;
    let options = CodecOptions::default()
        .with_chunk_concurrent_minimum(1)
        .with_concurrent_target(concurrent_target);
    let bytes = opened
        .array
        .async_retrieve_array_subset_sharded_opt::<ArrayBytes<'static>>(
            &cache, &subset, &options,
        )
        .await
        .map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                "failed to retrieve subset start={start:?} shape={shape:?} from {array_path} in {store_uri}: {e}"
            )))
        })?;
    let raw = bytes.into_fixed().map_err(|_| {
        ArrowError::InvalidArgumentError(format!(
            "array {array_path} in {store_uri}: variable-length subset bytes are not supported"
        ))
    })?;
    Ok(Buffer::from_vec(raw.into_owned()))
}

/// Read one chunk's raw bytes out of an already-open array.
///
/// `chunk_indices` are on the grid the reader walks: the inner-chunk grid
/// of a sharded array, the chunk grid of any other. zarrs' subchunk read
/// falls through to a plain chunk read for an unsharded array, so there
/// is one path for both. Within a shard, reads of distinct inner chunks
/// proceed concurrently once the shard's index is cached; the first read
/// into a shard fetches the index while holding that shard's cache lock,
/// so the other reads of that shard wait for that one small ranged
/// request. Reads of other shards do not.
async fn retrieve_chunk(
    opened: &OpenedArray,
    store_uri: &str,
    array_path: &str,
    chunk_indices: &[u64],
) -> Result<Buffer, ArrowError> {
    let shards = opened.shard_cache(chunk_indices).map_err(|e| {
        ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
            "failed to find the shard of chunk {chunk_indices:?} in {array_path} in {store_uri}: {e}"
        )))
    })?;
    let bytes = opened
        .array
        .async_retrieve_subchunk_opt::<ArrayBytes<'static>>(
            &shards,
            chunk_indices,
            &CodecOptions::default(),
        )
        .await
        .map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                "failed to retrieve chunk {chunk_indices:?} from {array_path} in {store_uri}: {e}"
            )))
        })?;
    let raw = bytes.into_fixed().map_err(|_| {
        ArrowError::InvalidArgumentError(format!(
            "array {array_path} in {store_uri}: variable-length chunk bytes are not supported"
        ))
    })?;
    Ok(Buffer::from_vec(raw.into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::view_entries::{ViewEntries, ViewEntry};
    use std::sync::Arc;

    use sedona_schema::raster::BandDataType;
    use tempfile::TempDir;
    use zarrs::array::ArrayBuilder;
    use zarrs::array::{FillValue, data_type as zarr_dtype};
    use zarrs::group::GroupBuilder;
    use zarrs_filesystem::FilesystemStore;

    use crate::source_uri::{build_array_anchor, build_chunk_anchor};

    /// Build a Zarr group at `<tempdir>/store.zarr` containing one array
    /// `temperature` of UInt8 with shape [2, 3] and chunk shape [2, 3]
    /// (one chunk). Returns the store URI and array path.
    fn build_uint8_zarr(dir: &TempDir) -> (String, &'static str, Vec<u8>) {
        let store_path = dir.path().join("store.zarr");
        let store = Arc::new(FilesystemStore::new(&store_path).unwrap());

        // Root group metadata — Zarr v3 stores need this for
        // `Group::open(store, "/")` to succeed.
        GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();

        let array = ArrayBuilder::new(
            vec![2, 3],
            vec![2, 3],
            zarr_dtype::uint8(),
            FillValue::from(0u8),
        )
        .build(store.clone(), "/temperature")
        .unwrap();
        array.store_metadata().unwrap();

        let pixels: Vec<u8> = vec![10, 11, 12, 13, 14, 15];
        array.store_chunk(&[0, 0], pixels.clone()).unwrap();

        let store_uri = format!("file://{}", store_path.display());
        (store_uri, "temperature", pixels)
    }

    #[tokio::test]
    async fn zarr_loader_reads_uint8_chunk() {
        let tmp = TempDir::new().unwrap();
        let (store_uri, array_path, expected_pixels) = build_uint8_zarr(&tmp);
        let uri = build_chunk_anchor(&store_uri, array_path, &[0, 0]);

        let loader = ZarrLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result[0].bytes.as_slice(), expected_pixels.as_slice());
    }

    #[tokio::test]
    async fn zarr_loader_materializes_a_full_array_across_chunks() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let uri = build_array_anchor(&store_uri, "temperature");
        let shape = [2, 6];
        let view = ViewEntries::identity_for_shape(&shape);
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &shape,
            view: &view,
            data_type: BandDataType::UInt8,
        };

        let result = ZarrLoader::new().load(&[&req]).await.unwrap();
        assert_eq!(result[0].source_shape, shape);
        assert!(result[0].view.is_identity(&shape));
        assert_eq!(
            result[0].bytes.as_slice(),
            &[10, 11, 12, 20, 21, 22, 13, 14, 15, 23, 24, 25]
        );
    }

    #[tokio::test]
    async fn zarr_loader_pushes_a_cross_chunk_view_into_the_array_read() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let uri = build_array_anchor(&store_uri, "temperature");
        let view = ViewEntries::new(vec![
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 2,
            },
            ViewEntry {
                source_axis: 1,
                start: 2,
                step: 1,
                steps: 3,
            },
        ]);
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 6],
            view: &view,
            data_type: BandDataType::UInt8,
        };

        let result = ZarrLoader::new().load(&[&req]).await.unwrap();
        assert_eq!(result[0].bytes.as_slice(), &[12, 20, 21, 15, 23, 24]);
        assert_eq!(result[0].source_shape, [2, 3]);
        assert!(result[0].view.is_identity(&[2, 3]));
    }

    #[tokio::test]
    async fn zarr_loader_pushes_a_view_into_one_chunk() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let uri = build_chunk_anchor(&store_uri, "temperature", &[0, 0]);
        let view = ViewEntries::new(vec![
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 2,
            },
            ViewEntry {
                source_axis: 1,
                start: 1,
                step: 1,
                steps: 2,
            },
        ]);
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &view,
            data_type: BandDataType::UInt8,
        };

        let result = ZarrLoader::new().load(&[&req]).await.unwrap();
        assert_eq!(result[0].bytes.as_slice(), &[11, 12, 14, 15]);
        assert_eq!(result[0].source_shape, [2, 2]);
        assert!(result[0].view.is_identity(&[2, 2]));
    }

    /// A sharded array's anchor indexes an inner chunk: the bytes come
    /// back for that 2×2 block alone, not the 4×4 shard it sits in. The
    /// array has one shard; the second load reuses its index.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn zarr_loader_opens_an_array_once_under_concurrent_loads() {
        // Partitions loading chunks of one array at the same time (the
        // planner spreads a raster's rows over partitions) must not each
        // open it: the first call opens, the rest reuse the handle.
        let tmp = TempDir::new().unwrap();
        let (store_uri, array_path, pixels) = build_uint8_zarr(&tmp);
        let uri = build_chunk_anchor(&store_uri, array_path, &[0, 0]);
        let loader = ZarrLoader::new();
        let loads = (0..8).map(|_| {
            let loader = loader.clone();
            let uri = uri.clone();
            tokio::spawn(async move {
                let req = RasterLoadRequest {
                    uri: &uri,
                    dim_names: &["y", "x"],
                    source_shape: &[2, 3],
                    view: &ViewEntries::identity_for_shape(&[2, 3]),
                    data_type: BandDataType::UInt8,
                };
                loader.load(&[&req]).await.unwrap()[0].bytes.clone()
            })
        });
        for loaded in futures::future::join_all(loads).await {
            assert_eq!(loaded.unwrap().as_slice(), pixels.as_slice());
        }
        let stats = loader.handle_stats();
        assert_eq!(stats.array_misses, 1, "{stats:?}");
        assert_eq!(stats.array_hits, 7, "{stats:?}");
        assert_eq!(stats.store_misses, 1, "{stats:?}");
    }

    #[tokio::test]
    async fn canceled_storage_reads_release_io_budget() {
        use object_store::memory::InMemory;
        use object_store::throttle::{ThrottleConfig, ThrottledStore};
        use zarrs_object_store::AsyncObjectStore;

        let budget = Arc::new(IoBudget::new(1));
        let storage = BudgetedStorage {
            inner: Arc::new(AsyncObjectStore::new(ThrottledStore::new(
                InMemory::new(),
                ThrottleConfig {
                    wait_get_per_call: Duration::from_secs(60),
                    ..Default::default()
                },
            ))),
            budget: budget.clone(),
        };
        let key = StoreKey::new("pixels/c/0/0").unwrap();
        let mut read = Box::pin(storage.get(&key));
        assert!(futures::poll!(&mut read).is_pending());
        assert_eq!(budget.permits.available_permits(), 0);
        assert_eq!(budget.io.in_flight.load(Ordering::SeqCst), 1);
        drop(read);
        assert_eq!(budget.permits.available_permits(), 1);
        assert_eq!(budget.io.in_flight.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn canceled_array_opens_release_gate_entries() {
        use object_store::memory::InMemory;
        use object_store::throttle::{ThrottleConfig, ThrottledStore};

        let cache = HandleCache::new(1, Duration::from_secs(300), Arc::new(IoBudget::new(1)));
        let store = Arc::new(ThrottledStore::new(
            InMemory::new(),
            ThrottleConfig {
                wait_get_per_call: Duration::from_secs(60),
                ..Default::default()
            },
        ));
        lock(&cache.stores).insert("file://".into(), store);

        for i in 0..16 {
            let path = format!("/array{i}");
            let mut opening = Box::pin(cache.array("file:///review.zarr", &path));
            assert!(futures::poll!(&mut opening).is_pending());
            drop(opening);
        }

        assert_eq!(lock(&cache.arrays).len(), 0);
        assert_eq!(lock(&cache.opening).len(), 0);
    }

    /// Concurrent first reads of different shards share the opened array
    /// but fetch their shard indexes at the same time. Every store call
    /// waits 100 ms of paused (virtual) time, so a batch that fetched the
    /// 32 indexes one after another would take over 3.2 s.
    #[tokio::test(start_paused = true)]
    async fn cold_reads_of_different_shards_fetch_their_indexes_concurrently() {
        use object_store::ObjectStoreExt;
        use object_store::memory::InMemory;
        use object_store::path::Path;
        use object_store::throttle::{ThrottleConfig, ThrottledStore};

        // 32 shards of shape [4, 4] side by side, each four 2×2 inner chunks.
        let tmp = TempDir::new().unwrap();
        let filesystem = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
        let array = ArrayBuilder::new(
            vec![4, 128],
            vec![4, 4],
            zarr_dtype::uint8(),
            FillValue::from(0u8),
        )
        .dimension_names(Some(["y", "x"]))
        .subchunk_shape(vec![2, 2])
        .build(filesystem, "/sharded")
        .unwrap();
        array.store_metadata().unwrap();
        let memory = InMemory::new();
        let metadata = std::fs::read(tmp.path().join("sharded/zarr.json")).unwrap();
        memory
            .put(
                &Path::from("review.zarr/sharded/zarr.json"),
                metadata.into(),
            )
            .await
            .unwrap();
        for shard in 0..32u8 {
            let pixels: Vec<u8> = (0..16u8).map(|v| v + shard).collect();
            array.store_chunk(&[0, shard as u64], pixels).unwrap();
            let key = format!("sharded/c/0/{shard}");
            let bytes = std::fs::read(tmp.path().join(&key)).unwrap();
            memory
                .put(&Path::from(format!("review.zarr/{key}")), bytes.into())
                .await
                .unwrap();
        }
        let store = Arc::new(ThrottledStore::new(
            memory,
            ThrottleConfig {
                wait_get_per_call: Duration::from_millis(100),
                ..Default::default()
            },
        ));

        let loader = ZarrLoader::new();
        lock(&loader.handles.stores).insert("file://".into(), store);
        let start = tokio::time::Instant::now();
        let loads = (0..32u8).map(|shard| {
            let loader = loader.clone();
            tokio::spawn(async move {
                // Inner chunk (0, 2 * shard) is the first one of that shard.
                let uri = format!("file:///review.zarr#array=sharded&chunk=0,{}", 2 * shard);
                let req = RasterLoadRequest {
                    uri: &uri,
                    dim_names: &["y", "x"],
                    source_shape: &[2, 2],
                    view: &ViewEntries::identity_for_shape(&[2, 2]),
                    data_type: BandDataType::UInt8,
                };
                let loaded = loader.load(&[&req]).await.unwrap()[0].bytes.clone();
                assert_eq!(loaded.as_slice(), [shard, shard + 1, shard + 4, shard + 5]);
            })
        });
        for load in futures::future::join_all(loads).await {
            load.unwrap();
        }
        let elapsed = start.elapsed();

        assert!(elapsed < Duration::from_secs(1), "{elapsed:?}");
        let stats = loader.handle_stats();
        assert_eq!(stats.array_misses, 1, "{stats:?}");
        assert_eq!(stats.array_hits, 31, "{stats:?}");
    }

    #[tokio::test]
    async fn zarr_loader_reads_one_inner_chunk_of_a_sharded_array() {
        let tmp = TempDir::new().unwrap();
        let store_path = tmp.path().join("store.zarr");
        let store = Arc::new(FilesystemStore::new(&store_path).unwrap());
        GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();
        let array = ArrayBuilder::new(
            vec![4, 4],
            vec![4, 4],
            zarr_dtype::uint8(),
            FillValue::from(0u8),
        )
        .dimension_names(Some(["y", "x"]))
        .subchunk_shape(vec![2, 2])
        .build(store.clone(), "/temperature")
        .unwrap();
        array.store_metadata().unwrap();
        // Pixel (y, x) holds 10*y + x.
        let pixels: Vec<u8> = (0..4u8)
            .flat_map(|y| (0..4u8).map(move |x| 10 * y + x))
            .collect();
        array.store_chunk(&[0, 0], pixels).unwrap();

        let store_uri = format!("file://{}", store_path.display());
        let uri = build_chunk_anchor(&store_uri, "temperature", &[1, 0]);
        let loader = ZarrLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 2],
            view: &ViewEntries::identity_for_shape(&[2, 2]),
            data_type: BandDataType::UInt8,
        };
        for _ in 0..2 {
            let result = loader.load(&[&req]).await.unwrap();
            assert_eq!(result[0].bytes.as_slice(), &[20, 21, 30, 31]);
        }
        assert_eq!(loader.handle_stats().array_misses, 1);
    }

    #[tokio::test]
    async fn zarr_loader_errors_when_dtype_disagrees_with_array() {
        let tmp = TempDir::new().unwrap();
        let (store_uri, array_path, _) = build_uint8_zarr(&tmp);
        let uri = build_chunk_anchor(&store_uri, array_path, &[0, 0]);

        let loader = ZarrLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]), // Array is UInt8 but the band claims Int16.
            data_type: BandDataType::Int16,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("metadata claims") && (msg.contains("UInt8") || msg.contains("Int16")),
            "expected dtype-mismatch diagnostic, got: {msg}"
        );
    }

    #[tokio::test]
    async fn zarr_loader_errors_on_malformed_chunk_anchor_uri() {
        let loader = ZarrLoader::new();
        let req = RasterLoadRequest {
            uri: "file:///tmp/foo.zarr", // missing fragment
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("missing"),
            "expected missing-fragment diagnostic, got: {err}"
        );
    }

    #[tokio::test]
    async fn zarr_loader_errors_on_missing_array_path() {
        let tmp = TempDir::new().unwrap();
        let (store_uri, _, _) = build_uint8_zarr(&tmp);
        // Anchor a chunk against a non-existent array.
        let uri = build_chunk_anchor(&store_uri, "nonexistent", &[0, 0]);

        let loader = ZarrLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("/nonexistent") && err.to_string().contains(&store_uri),
            "expected diagnostic to name the missing array path and its store, got: {err}"
        );
    }

    #[tokio::test]
    async fn zarr_loader_errors_on_unsupported_scheme() {
        // `s3`/`http` are supported now; an unsupported scheme (e.g. `gs://`)
        // is rejected at store construction, without touching the network.
        let loader = ZarrLoader::new();
        let uri = build_chunk_anchor("gs://bucket/foo.zarr", "temperature", &[0, 0]);
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("unsupported Zarr URI scheme"),
            "expected unsupported-scheme rejection, got: {err}"
        );
    }

    /// Two arrays, each shape `[2, 6]` chunked `[2, 3]` (two chunks apiece),
    /// every chunk holding distinct pixels: temperature chunks are 10..16
    /// and 20..26, pressure chunks are 30..36 and 40..46.
    fn build_two_array_zarr(dir: &TempDir) -> String {
        let store_path = dir.path().join("store.zarr");
        let store = Arc::new(FilesystemStore::new(&store_path).unwrap());
        GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();
        for (name, base) in [("temperature", 10u8), ("pressure", 30u8)] {
            let array = ArrayBuilder::new(
                vec![2, 6],
                vec![2, 3],
                zarr_dtype::uint8(),
                FillValue::from(0u8),
            )
            .build(store.clone(), &format!("/{name}"))
            .unwrap();
            array.store_metadata().unwrap();
            array
                .store_chunk(&[0, 0], (base..base + 6).collect::<Vec<u8>>())
                .unwrap();
            array
                .store_chunk(&[0, 1], (base + 10..base + 16).collect::<Vec<u8>>())
                .unwrap();
        }
        format!("file://{}", store_path.display())
    }

    #[test]
    fn anchor_plan_dedupes_stores_and_arrays() {
        let uris = [
            build_chunk_anchor("file:///a.zarr", "t", &[0, 0]),
            build_chunk_anchor("file:///a.zarr", "t", &[0, 1]),
            build_chunk_anchor("file:///a.zarr", "/p", &[0, 0]),
            build_chunk_anchor("file:///b.zarr", "t", &[0, 0]),
            build_array_anchor("file:///a.zarr", "t"),
        ];
        let plan = AnchorPlan::build(uris.iter().map(String::as_str)).unwrap();
        assert_eq!(plan.stores, vec!["file:///a.zarr", "file:///b.zarr"]);
        assert_eq!(
            plan.arrays,
            vec![
                (0, "/t".to_string()),
                (0, "/p".to_string()),
                (1, "/t".to_string())
            ]
        );
        assert_eq!(
            plan.requests,
            vec![
                PlannedAnchor {
                    array: 0,
                    chunk_indices: Some(vec![0, 0])
                },
                PlannedAnchor {
                    array: 0,
                    chunk_indices: Some(vec![0, 1])
                },
                PlannedAnchor {
                    array: 1,
                    chunk_indices: Some(vec![0, 0])
                },
                PlannedAnchor {
                    array: 2,
                    chunk_indices: Some(vec![0, 0])
                },
                PlannedAnchor {
                    array: 0,
                    chunk_indices: None
                },
            ]
        );
    }

    #[test]
    fn load_plan_rejects_malformed_anchor() {
        let err = AnchorPlan::build(["file:///a.zarr"]).unwrap_err();
        assert!(err.to_string().contains("missing"), "{err}");
    }

    #[test]
    fn with_concurrency_clamps_to_at_least_one() {
        assert_eq!(ZarrLoader::new().concurrency(), DEFAULT_LOAD_CONCURRENCY);
        assert_eq!(ZarrLoader::new().with_concurrency(0).concurrency(), 1);
        assert_eq!(ZarrLoader::new().with_concurrency(3).concurrency(), 3);
        assert_eq!(
            ZarrLoader::new()
                .with_concurrency(3)
                .budget
                .permits
                .available_permits(),
            3
        );
    }

    /// Eight concurrent calls on clones of one loader with a budget of two
    /// never have more than two chunk reads in flight between them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_loads_share_one_io_budget() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let uris = two_array_uris(&store_uri);
        let loader = ZarrLoader::new().with_concurrency(2);

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let loader = loader.clone();
            let uris = uris.clone();
            tasks.push(tokio::spawn(async move {
                load_two_arrays(&loader, &uris).await
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }

        let peak = loader.peak_in_flight();
        assert!((1..=2).contains(&peak), "peak in flight {peak}");
    }

    #[tokio::test]
    async fn zarr_loader_batches_requests_across_arrays_and_chunks() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let view = ViewEntries::identity_for_shape(&[2, 3]);
        let uris = [
            build_chunk_anchor(&store_uri, "temperature", &[0, 0]),
            build_chunk_anchor(&store_uri, "pressure", &[0, 1]),
            build_chunk_anchor(&store_uri, "temperature", &[0, 1]),
            build_chunk_anchor(&store_uri, "pressure", &[0, 0]),
            // Duplicate anchor: must come back with the same bytes, shared.
            build_chunk_anchor(&store_uri, "temperature", &[0, 0]),
        ];
        let reqs: Vec<RasterLoadRequest> = uris
            .iter()
            .map(|uri| RasterLoadRequest {
                uri,
                dim_names: &["y", "x"],
                source_shape: &[2, 3],
                view: &view,
                data_type: BandDataType::UInt8,
            })
            .collect();
        let req_refs: Vec<&RasterLoadRequest> = reqs.iter().collect();
        let expected: [Vec<u8>; 5] = [
            (10..16).collect(),
            (40..46).collect(),
            (20..26).collect(),
            (30..36).collect(),
            (10..16).collect(),
        ];

        // Serial and fanned-out must agree, and both must preserve request order.
        for concurrency in [1, DEFAULT_LOAD_CONCURRENCY] {
            let results = ZarrLoader::new()
                .with_concurrency(concurrency)
                .load(&req_refs)
                .await
                .unwrap();
            assert_eq!(results.len(), 5);
            for (i, (result, want)) in results.iter().zip(&expected).enumerate() {
                assert_eq!(
                    result.bytes.as_slice(),
                    want.as_slice(),
                    "request {i} at concurrency {concurrency}"
                );
            }
            assert_eq!(
                results[0].bytes.as_ptr(),
                results[4].bytes.as_ptr(),
                "duplicate anchors must share one loaded buffer"
            );
        }
    }

    #[tokio::test]
    async fn zarr_loader_batch_fails_when_any_request_dtype_disagrees() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let view = ViewEntries::identity_for_shape(&[2, 3]);
        let good = build_chunk_anchor(&store_uri, "temperature", &[0, 0]);
        let bad = build_chunk_anchor(&store_uri, "pressure", &[0, 0]);
        let reqs = [
            RasterLoadRequest {
                uri: &good,
                dim_names: &["y", "x"],
                source_shape: &[2, 3],
                view: &view,
                data_type: BandDataType::UInt8,
            },
            RasterLoadRequest {
                uri: &bad,
                dim_names: &["y", "x"],
                source_shape: &[2, 3],
                view: &view,
                // Array is UInt8 but this band claims Int16.
                data_type: BandDataType::Int16,
            },
        ];
        let err = ZarrLoader::new()
            .load(&[&reqs[0], &reqs[1]])
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("metadata claims") && err.contains("/pressure"),
            "expected the dtype-mismatch diagnostic naming the array, got: {err}"
        );
    }

    fn two_array_uris(store_uri: &str) -> Vec<String> {
        vec![
            build_chunk_anchor(store_uri, "temperature", &[0, 0]),
            build_chunk_anchor(store_uri, "pressure", &[0, 1]),
        ]
    }

    async fn load_two_arrays(loader: &ZarrLoader, uris: &[String]) -> Vec<RasterLoadResult> {
        let view = ViewEntries::identity_for_shape(&[2, 3]);
        let reqs: Vec<RasterLoadRequest> = uris
            .iter()
            .map(|uri| RasterLoadRequest {
                uri,
                dim_names: &["y", "x"],
                source_shape: &[2, 3],
                view: &view,
                data_type: BandDataType::UInt8,
            })
            .collect();
        let refs: Vec<&RasterLoadRequest> = reqs.iter().collect();
        loader.load(&refs).await.unwrap()
    }

    #[tokio::test]
    async fn zarr_loader_reuses_array_handles_across_calls() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let uris = two_array_uris(&store_uri);
        let loader = ZarrLoader::new();

        let first = load_two_arrays(&loader, &uris).await;
        assert_eq!(
            loader.handle_stats(),
            HandleStats {
                store_hits: 1,
                store_misses: 1,
                array_hits: 0,
                array_misses: 2,
            }
        );

        let second = load_two_arrays(&loader, &uris).await;
        assert_eq!(
            loader.handle_stats(),
            HandleStats {
                store_hits: 1,
                store_misses: 1,
                array_hits: 2,
                array_misses: 2,
            }
        );
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.bytes.as_slice(), b.bytes.as_slice());
        }

        // A clone shares the handles rather than starting its own.
        let clone = loader.clone();
        load_two_arrays(&clone, &uris).await;
        assert_eq!(loader.handle_stats().array_hits, 4);
        assert_eq!(loader.handle_stats().array_misses, 2);

        // A second group under the same scheme and authority (`file://`)
        // shares the store client: its arrays are new opens, the client
        // is not rebuilt.
        let other = TempDir::new().unwrap();
        let other_uris = two_array_uris(&build_two_array_zarr(&other));
        load_two_arrays(&loader, &other_uris).await;
        assert_eq!(loader.handle_stats().store_misses, 1);
        assert_eq!(loader.handle_stats().store_hits, 3);
        assert_eq!(loader.handle_stats().array_misses, 4);
    }

    #[tokio::test]
    async fn zarr_loader_reopens_arrays_once_the_handle_ttl_expires() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let uris = two_array_uris(&store_uri);
        let loader = ZarrLoader::new().with_array_handle_ttl(Duration::ZERO);

        load_two_arrays(&loader, &uris).await;
        load_two_arrays(&loader, &uris).await;
        // Arrays are re-opened; the store client has no TTL and is kept.
        assert_eq!(
            loader.handle_stats(),
            HandleStats {
                store_hits: 3,
                store_misses: 1,
                array_hits: 0,
                array_misses: 4,
            }
        );
    }

    #[tokio::test]
    async fn zarr_loader_bounds_the_number_of_kept_array_handles() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let uris = two_array_uris(&store_uri);
        // Room for one array while every call touches two: each open evicts
        // the other, so nothing is ever reused. Zero clamps to one.
        let loader = ZarrLoader::new().with_array_handle_capacity(0);

        load_two_arrays(&loader, &uris).await;
        load_two_arrays(&loader, &uris).await;
        assert_eq!(
            loader.handle_stats(),
            HandleStats {
                store_hits: 3,
                store_misses: 1,
                array_hits: 0,
                array_misses: 4,
            }
        );
    }
}
