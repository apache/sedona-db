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

//! Session-scoped cache of loaded OutDb band bytes.
//!
//! [`RasterChunkCache`] is the interface `RS_EnsureLoaded` caches through.
//! Before dispatching to a loader it looks each OutDb band up by
//! [`ChunkKey`] — the band's `outdb_uri` plus its data type — and
//! afterwards it stores what the loader returned, as the bytes and the
//! source shape they are laid out in. A Zarr anchor URI names one chunk
//! and a GDAL URI names one band, so the unit cached is the unit loaded.
//! The requesting band's own view is re-applied by the caller on a hit,
//! which is why only whole-source ("unresolved") results belong here: the
//! caller checks that before inserting.
//!
//! [`InMemoryChunkCache`] is what a `SedonaContext` builds, and the rest
//! of this page describes it. [`NoChunkCache`] keeps nothing, for a
//! session built without a cache. A tier that keeps evicted entries on
//! local disk (DataFusion's `DiskManager` would be the natural home) or in
//! an object store would implement the same trait: a hit may hand back
//! bytes read from anywhere, and such a tier evicts by plain byte LRU,
//! since nothing outside memory is refcounted. It only pays off where the
//! tier is faster than the store the loader reads from, so it would be
//! opt-in.
//!
//! # Zero-copy hits
//!
//! A hit hands back a refcounted clone of the cached [`Buffer`]. The raster
//! builder attaches it as a shared data block of the output `BinaryView`
//! column, so a hit costs no pixel copy and every downstream operator that
//! keeps view blocks shared keeps sharing it.
//!
//! # Idle versus shared entries
//!
//! Because hits share the allocation, evicting an entry only frees memory
//! when the cache is its last holder. [`Buffer::strong_count`] tells the
//! two apart: an entry with count one is *idle* (only the cache holds it)
//! and is what the budget counts and what eviction removes; an entry with
//! a higher count is *shared* with some in-flight batch, costs the cache
//! nothing extra, is by definition hot, and is left alone until its last
//! holder drops. There is no callback on that drop, so idle bytes are
//! recomputed by scanning on insert and on request. The count is a
//! snapshot: evicting an entry someone just cloned frees nothing and
//! skipping one whose batch just dropped costs one more pass, neither of
//! which is unsafe.
//!
//! This is also why the structure is a plain map and LRU list rather than
//! an off-the-shelf cache: a weigher fixed at insert time cannot see
//! refcounts, so evicting a shared entry would look like progress and
//! free nothing.
//!
//! # Identity and staleness
//!
//! The key is the URI and the data type; the entry also records the source
//! shape and source-order dimension names the bytes were laid out over,
//! and a lookup only counts as a hit when those match the requesting band
//! (a stale entry is a miss, and an insert with a different layout replaces
//! it). The `outdb_format` is not part of the key: two loaders registered
//! for the same URI string would alias, which no in-tree loader does.
//! Bytes behind a URI are assumed not to change for the life of the
//! session. A store rewritten under a running session serves the old bytes
//! until the entry is evicted; `SET sedona.raster.cache_max_bytes = 0`
//! (then back) or [`RasterChunkCache::clear`] drops everything.
//!
//! # Accounting
//!
//! With a [`MemoryPool`], each cached allocation is claimed via
//! [`Buffer::claim`] so the pool charges it until the allocation is
//! actually freed — when the last holder drops, not when the cache
//! evicts. Arrow's pool interface accounts but cannot refuse, so the
//! cache checks [`MemoryPool::available`] before inserting and skips the
//! insert when the pool is full; a load never fails because of the cache.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use arrow_buffer::{Buffer, MemoryPool};
use lru::LruCache;
use sedona_schema::raster::BandDataType;

/// What `RS_EnsureLoaded` caches loaded OutDb bytes through; see the
/// [module docs](self) for what is stored and why.
///
/// One instance serves every partition and call of a session, so each
/// method takes `&self` and must tolerate concurrent callers. Nothing here
/// can fail: a load never fails because of its cache.
pub trait RasterChunkCache: Send + Sync + fmt::Debug {
    /// The entry for `key`, if one is held. The bytes may share the cached
    /// allocation, and the lookup may promote the entry.
    fn get(&self, key: &ChunkKey) -> Option<CachedChunk>;

    /// Store each chunk under its key and return, in order, the buffer the
    /// caller should use from now on: the cached one where the entry was
    /// stored, so that the caller's output and the cache share a single
    /// allocation, and the caller's own bytes unchanged where it was not.
    /// All of a load's results arrive together so an implementation can
    /// account for them once.
    fn insert_batch(&self, entries: Vec<(ChunkKey, CachedChunk)>) -> Vec<Buffer>;

    /// [`Self::insert_batch`] for one entry.
    fn insert(&self, key: ChunkKey, chunk: CachedChunk) -> Buffer {
        self.insert_batch(vec![(key, chunk)])
            .pop()
            .expect("one buffer per entry")
    }

    /// The budget, in bytes, for entries only the cache holds. Zero means
    /// the cache is disabled.
    fn max_bytes(&self) -> usize;

    /// Change the budget. `sedona.raster.cache_max_bytes` reaches the cache
    /// through here before every load, so a `SET` applies live: lowering
    /// the budget evicts down to the new value, and zero clears the cache
    /// and disables it until raised again.
    fn set_max_bytes(&self, max_bytes: usize);

    /// Evict down to the budget. Callers run this before every load, since
    /// an entry only stops being referenced when an earlier batch drops,
    /// which a cache cannot observe; this is where a session that only
    /// hits gives memory back.
    fn trim(&self);

    /// Drop every entry. Bytes shared with in-flight batches live on until
    /// those batches drop.
    fn clear(&self);

    /// Counters and a snapshot of what is held.
    fn stats(&self) -> ChunkCacheStats;
}

/// Identity of a cached load: the band's `outdb_uri` and data type. The
/// data type is defensive — a URI identifies its bytes, but a mismatched
/// dtype request must never alias an entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub uri: String,
    pub data_type: BandDataType,
}

/// A cached whole-source load: the bytes, the shape they are laid out in,
/// and the source-order dimension names that shape is over. The view is
/// deliberately absent; the requesting band's own view is re-applied on a
/// hit.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedChunk {
    pub bytes: Buffer,
    pub source_shape: Vec<i64>,
    pub dim_names: Vec<String>,
}

impl CachedChunk {
    /// Whether `other` describes the same layout: shape, dimension order
    /// and byte count. Bytes are not compared.
    pub fn same_layout(&self, other: &CachedChunk) -> bool {
        self.source_shape == other.source_shape
            && self.dim_names == other.dim_names
            && self.bytes.len() == other.bytes.len()
    }
}

/// Counters and a snapshot of the cache's contents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChunkCacheStats {
    /// Lookups that found an entry.
    pub hits: u64,
    /// Lookups that found nothing.
    pub misses: u64,
    /// Entries stored.
    pub inserts: u64,
    /// Inserts skipped: cache disabled, entry over budget, or pool full.
    pub skipped: u64,
    /// Entries removed: idle ones to make room, plus everything dropped
    /// by [`RasterChunkCache::clear`] or a budget of zero.
    pub evictions: u64,
    /// Entries currently held.
    pub entries: usize,
    /// Bytes of entries only the cache holds — what the budget counts.
    pub idle_bytes: usize,
    /// Bytes of entries some batch still references — the query's cost,
    /// not the cache's.
    pub shared_bytes: usize,
}

struct Inner {
    /// Most-recently used first; eviction walks from the other end.
    entries: LruCache<ChunkKey, CachedChunk>,
    max_bytes: usize,
    evictions: u64,
}

impl Inner {
    fn weight(chunk: &CachedChunk) -> usize {
        // Capacity, not length: that is what the allocation occupies and
        // what `Buffer::claim` charges.
        chunk.bytes.capacity()
    }

    fn is_idle(chunk: &CachedChunk) -> bool {
        chunk.bytes.strong_count() == 1
    }

    fn idle_and_shared_bytes(&self) -> (usize, usize) {
        self.entries
            .iter()
            .fold((0, 0), |(idle, shared), (_, chunk)| {
                if Self::is_idle(chunk) {
                    (idle + Self::weight(chunk), shared)
                } else {
                    (idle, shared + Self::weight(chunk))
                }
            })
    }

    /// Evict idle entries, least recently used first, until at least
    /// `needed` bytes have been freed or no idle entry is left, and return
    /// the bytes freed. Shared entries are skipped: dropping the cache's
    /// reference to one frees nothing.
    fn evict_idle(&mut self, needed: usize) -> usize {
        let mut freed = 0usize;
        let victims: Vec<ChunkKey> = self
            .entries
            .iter()
            .rev()
            .filter(|(_, chunk)| Self::is_idle(chunk))
            .take_while(|(_, chunk)| {
                let take = freed < needed;
                if take {
                    freed += Self::weight(chunk);
                }
                take
            })
            .map(|(key, _)| key.clone())
            .collect();
        for key in victims {
            self.entries.pop(&key);
            self.evictions += 1;
        }
        freed
    }

    /// Evict idle entries down to the budget.
    fn trim(&mut self) {
        let (idle, _) = self.idle_and_shared_bytes();
        if idle > self.max_bytes {
            self.evict_idle(idle - self.max_bytes);
        }
    }
}

/// The in-memory [`RasterChunkCache`]: an LRU map of whole-source entries
/// whose budget counts only the entries that no batch still references,
/// each allocation charged to an optional [`MemoryPool`]. See the
/// [module docs](self).
pub struct InMemoryChunkCache {
    inner: Mutex<Inner>,
    pool: Option<Arc<dyn MemoryPool>>,
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    skipped: AtomicU64,
}

impl InMemoryChunkCache {
    /// A cache holding up to `max_bytes` of idle entries. Zero disables
    /// it: every lookup misses and nothing is stored.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: LruCache::unbounded(),
                max_bytes,
                evictions: 0,
            }),
            pool: None,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        }
    }

    /// Charge cached allocations to `pool` for as long as they live, and
    /// skip inserts the pool has no room for.
    pub fn with_memory_pool(mut self, pool: Arc<dyn MemoryPool>) -> Self {
        self.pool = Some(pool);
        self
    }
}

impl RasterChunkCache for InMemoryChunkCache {
    /// Promotes the entry to most recently used.
    fn get(&self, key: &ChunkKey) -> Option<CachedChunk> {
        let mut inner = lock(&self.inner);
        if inner.max_bytes == 0 {
            return None;
        }
        match inner.entries.get(key) {
            Some(chunk) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(chunk.clone())
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// A buffer is shrunk to its length first when nothing else holds it,
    /// since a decode may leave spare capacity and both the budget and the
    /// pool charge capacity.
    ///
    /// An insert is skipped, and its bytes handed back unchanged, when the
    /// cache is disabled, when the entry alone exceeds the budget, or when
    /// the memory pool has no room for it even after evicting idle entries.
    /// An existing entry for the key with the same layout wins over the new
    /// bytes; one with a different layout is stale and is replaced.
    ///
    /// One lock and one idle scan serve the whole batch.
    fn insert_batch(&self, entries: Vec<(ChunkKey, CachedChunk)>) -> Vec<Buffer> {
        let mut inner = lock(&self.inner);
        if inner.max_bytes == 0 {
            self.skipped
                .fetch_add(entries.len() as u64, Ordering::Relaxed);
            return entries.into_iter().map(|(_, chunk)| chunk.bytes).collect();
        }

        // Entries inserted below are shared (the caller holds the returned
        // clone), so idle bytes only move when eviction frees some.
        let (mut idle, _) = inner.idle_and_shared_bytes();
        let budget = inner.max_bytes;
        let mut out = Vec::with_capacity(entries.len());
        for (key, mut chunk) in entries {
            if let Some(existing) = inner.entries.get(&key) {
                if existing.same_layout(&chunk) {
                    out.push(existing.bytes.clone());
                    continue;
                }
                if let Some(stale) = inner.entries.pop(&key) {
                    if Inner::is_idle(&stale) {
                        idle -= Inner::weight(&stale);
                    }
                    inner.evictions += 1;
                }
            }

            chunk.bytes.shrink_to_fit();
            let weight = Inner::weight(&chunk);
            if weight > budget {
                self.skipped.fetch_add(1, Ordering::Relaxed);
                out.push(chunk.bytes);
                continue;
            }
            if idle + weight > budget {
                idle -= inner.evict_idle(idle + weight - budget);
            }
            if let Some(pool) = &self.pool {
                // Idle entries are the cache's to give back: evicting them
                // releases their reservations, so make room in the pool the
                // same way as in the budget before giving up.
                let room = pool_room(pool.as_ref());
                if room < weight {
                    idle -= inner.evict_idle(weight - room);
                }
                if pool_room(pool.as_ref()) < weight {
                    self.skipped.fetch_add(1, Ordering::Relaxed);
                    out.push(chunk.bytes);
                    continue;
                }
                chunk.bytes.claim(pool.as_ref());
            }

            out.push(chunk.bytes.clone());
            inner.entries.put(key, chunk);
            self.inserts.fetch_add(1, Ordering::Relaxed);
        }
        out
    }

    fn max_bytes(&self) -> usize {
        lock(&self.inner).max_bytes
    }

    fn set_max_bytes(&self, max_bytes: usize) {
        let mut inner = lock(&self.inner);
        inner.max_bytes = max_bytes;
        if max_bytes == 0 {
            inner.evictions += inner.entries.len() as u64;
            inner.entries.clear();
            return;
        }
        inner.trim();
    }

    /// Idle entries over the budget go, least recently used first.
    fn trim(&self) {
        let mut inner = lock(&self.inner);
        if inner.max_bytes > 0 {
            inner.trim();
        }
    }

    fn clear(&self) {
        let mut inner = lock(&self.inner);
        inner.evictions += inner.entries.len() as u64;
        inner.entries.clear();
    }

    /// The idle and shared byte counts come from a scan under the lock.
    fn stats(&self) -> ChunkCacheStats {
        let inner = lock(&self.inner);
        let (idle_bytes, shared_bytes) = inner.idle_and_shared_bytes();
        ChunkCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            skipped: self.skipped.load(Ordering::Relaxed),
            evictions: inner.evictions,
            entries: inner.entries.len(),
            idle_bytes,
            shared_bytes,
        }
    }
}

impl fmt::Debug for InMemoryChunkCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InMemoryChunkCache")
            .field("max_bytes", &self.max_bytes())
            .field("pool", &self.pool.is_some())
            .field("stats", &self.stats())
            .finish()
    }
}

/// A [`RasterChunkCache`] that keeps nothing: every lookup misses, every
/// insert hands the caller's bytes straight back, the budget reads as zero
/// and setting it changes nothing, and the counters stay at zero. A loader
/// config built without a cache carries one, so `RS_EnsureLoaded` never
/// branches on whether a session has a cache.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoChunkCache;

impl RasterChunkCache for NoChunkCache {
    fn get(&self, _key: &ChunkKey) -> Option<CachedChunk> {
        None
    }

    fn insert_batch(&self, entries: Vec<(ChunkKey, CachedChunk)>) -> Vec<Buffer> {
        entries.into_iter().map(|(_, chunk)| chunk.bytes).collect()
    }

    fn max_bytes(&self) -> usize {
        0
    }

    fn set_max_bytes(&self, _max_bytes: usize) {}

    fn trim(&self) {}

    fn clear(&self) {}

    fn stats(&self) -> ChunkCacheStats {
        ChunkCacheStats::default()
    }
}

/// Bytes the pool can still take. Computed from `capacity` and `used`
/// rather than `MemoryPool::available`, which some adapters (DataFusion's)
/// report as `isize::MIN` when an unbounded capacity does not fit an
/// `isize`.
fn pool_room(pool: &dyn MemoryPool) -> usize {
    pool.capacity().saturating_sub(pool.used())
}

/// A poisoned lock means another insert panicked mid-update; the map is
/// still consistent (every mutation is a single `put`/`pop`), so recover
/// rather than propagate.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn key(name: &str) -> ChunkKey {
        ChunkKey {
            uri: name.to_string(),
            data_type: BandDataType::UInt8,
        }
    }

    fn chunk(len: usize) -> CachedChunk {
        CachedChunk {
            bytes: Buffer::from_vec(vec![7u8; len]),
            source_shape: vec![len as i64],
            dim_names: vec!["x".to_string()],
        }
    }

    #[test]
    fn hit_shares_the_cached_allocation() {
        let cache = InMemoryChunkCache::new(1024);
        let stored = cache.insert(key("a"), chunk(100));
        let hit = cache.get(&key("a")).expect("hit");
        assert_eq!(hit.bytes.as_ptr(), stored.as_ptr());
        assert_eq!(hit.source_shape, vec![100]);
        assert_eq!(hit.dim_names, vec!["x".to_string()]);
        assert!(cache.get(&key("b")).is_none());
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.inserts), (1, 1, 1));
        // `stored` and `hit` are still alive, so the entry is shared.
        assert_eq!(stats.shared_bytes, 100);
        assert_eq!(stats.idle_bytes, 0);
        drop(stored);
        drop(hit);
        assert_eq!(cache.stats().idle_bytes, 100);
    }

    #[test]
    fn a_disabled_cache_stores_nothing_and_never_hits() {
        let cache = InMemoryChunkCache::new(0);
        let c = chunk(10);
        let ptr = c.bytes.as_ptr();
        let returned = cache.insert(key("a"), c);
        assert_eq!(returned.as_ptr(), ptr);
        assert!(cache.get(&key("a")).is_none());
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.skipped, stats.misses), (0, 1, 0));
    }

    #[test]
    fn an_entry_over_the_whole_budget_is_skipped() {
        let cache = InMemoryChunkCache::new(50);
        cache.insert(key("big"), chunk(100));
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().skipped, 1);
    }

    #[test]
    fn eviction_takes_idle_entries_lru_first_and_leaves_shared_ones() {
        let cache = InMemoryChunkCache::new(300);
        for name in ["a", "b", "c"] {
            cache.insert(key(name), chunk(100));
        }
        // Hold `a` as a batch would. It is the least recently used entry
        // but shared, so it neither counts nor gets evicted; `d` fits
        // beside the 200 idle bytes of `b` and `c` without evicting.
        let held = cache.get(&key("a")).unwrap();
        cache.insert(key("d"), chunk(100));
        assert_eq!(cache.stats().evictions, 0);
        // `e` pushes idle bytes over budget: the idle LRU victim is `b`.
        cache.insert(key("e"), chunk(100));
        assert_eq!(cache.stats().evictions, 1);
        assert!(cache.get(&key("b")).is_none());
        for name in ["a", "c", "d", "e"] {
            assert!(cache.get(&key(name)).is_some(), "{name}");
        }

        // Once the holder drops, `a` is idle. The lookups above promoted
        // it, so touch the others to make it least recently used again.
        drop(held);
        for name in ["c", "d", "e"] {
            cache.get(&key(name));
        }
        // 400 idle bytes plus `f` need 200 freed: `a` then `c`.
        cache.insert(key("f"), chunk(100));
        assert_eq!(cache.stats().evictions, 3);
        assert!(cache.get(&key("a")).is_none());
        assert!(cache.get(&key("c")).is_none());
        assert_eq!(cache.stats().idle_bytes, 300);
    }

    #[test]
    fn shared_entries_do_not_count_against_the_budget() {
        let cache = InMemoryChunkCache::new(200);
        let held_a = cache.insert(key("a"), chunk(100));
        let held_b = cache.insert(key("b"), chunk(100));
        // Both are shared, so a third fits without evicting either.
        cache.insert(key("c"), chunk(100));
        let stats = cache.stats();
        assert_eq!(stats.entries, 3);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.shared_bytes, 200);
        assert_eq!(stats.idle_bytes, 100);
        drop(held_a);
        drop(held_b);
        // Now 300 idle bytes sit over a 200 budget; the next insert trims.
        cache.insert(key("d"), chunk(100));
        assert_eq!(cache.stats().idle_bytes, 200);
    }

    #[test]
    fn trim_evicts_idle_bytes_over_budget_without_an_insert() {
        let cache = InMemoryChunkCache::new(200);
        let held: Vec<Buffer> = ["a", "b", "c"]
            .into_iter()
            .map(|name| cache.insert(key(name), chunk(100)))
            .collect();
        // Nothing to trim while a batch holds everything.
        cache.trim();
        assert_eq!(cache.stats().entries, 3);
        drop(held);
        cache.trim();
        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.idle_bytes, 200);
        assert!(cache.get(&key("a")).is_none(), "LRU idle entry went first");
    }

    #[test]
    fn lowering_the_budget_evicts_and_zero_clears() {
        let cache = InMemoryChunkCache::new(300);
        for name in ["a", "b", "c"] {
            cache.insert(key(name), chunk(100));
        }
        cache.set_max_bytes(150);
        assert_eq!(cache.stats().entries, 1);
        assert!(cache.get(&key("c")).is_some());
        cache.set_max_bytes(0);
        assert_eq!(cache.stats().entries, 0);
        assert!(cache.get(&key("c")).is_none());
        cache.set_max_bytes(300);
        cache.insert(key("z"), chunk(10));
        assert!(cache.get(&key("z")).is_some());
    }

    #[test]
    fn an_existing_entry_with_the_same_layout_wins_over_a_duplicate_insert() {
        let cache = InMemoryChunkCache::new(1024);
        let stored = cache.insert(key("a"), chunk(10));
        let returned = cache.insert(key("a"), chunk(10));
        assert_eq!(returned.as_ptr(), stored.as_ptr());
        assert_eq!(cache.stats().inserts, 1);
    }

    #[test]
    fn an_existing_entry_with_a_different_layout_is_replaced() {
        let cache = InMemoryChunkCache::new(1024);
        let old = cache.insert(key("a"), chunk(12));
        // Same URI and dtype, same byte count, transposed layout: the new
        // bytes must win and the stale entry must go, even while held.
        let transposed = CachedChunk {
            bytes: Buffer::from_vec(vec![1u8; 12]),
            source_shape: vec![3, 4],
            dim_names: vec!["y".to_string(), "x".to_string()],
        };
        let new_ptr = transposed.bytes.as_ptr();
        let returned = cache.insert(key("a"), transposed);
        assert_eq!(returned.as_ptr(), new_ptr);
        assert_ne!(returned.as_ptr(), old.as_ptr());
        let hit = cache.get(&key("a")).unwrap();
        assert_eq!(hit.source_shape, vec![3, 4]);
        let stats = cache.stats();
        assert_eq!((stats.entries, stats.inserts, stats.evictions), (1, 2, 1));
    }

    #[test]
    fn insert_shrinks_spare_capacity_when_it_owns_the_allocation() {
        let cache = InMemoryChunkCache::new(1024);
        let mut vec = Vec::with_capacity(512);
        vec.extend_from_slice(&[1u8; 64]);
        let bytes = Buffer::from_vec(vec);
        assert!(bytes.capacity() >= 512);
        let stored = cache.insert(
            key("a"),
            CachedChunk {
                bytes,
                source_shape: vec![64],
                dim_names: vec!["x".to_string()],
            },
        );
        assert_eq!(stored.len(), 64);
        assert_eq!(stored.capacity(), 64);
        assert_eq!(cache.stats().shared_bytes, 64);
    }

    #[test]
    fn a_batch_is_inserted_under_one_idle_scan() {
        let cache = InMemoryChunkCache::new(150);
        cache.insert(key("old"), chunk(100));
        let entries: Vec<(ChunkKey, CachedChunk)> = ["a", "b", "c"]
            .into_iter()
            .map(|name| (key(name), chunk(100)))
            .collect();
        let held = cache.insert_batch(entries);
        assert_eq!(held.len(), 3);
        // The batch's own entries are shared and do not count, so only
        // `old` was idle: it goes for the first of them to fit, and the
        // rest fit beside 0 idle bytes.
        let stats = cache.stats();
        assert_eq!(stats.entries, 3);
        assert_eq!(stats.evictions, 1);
        assert!(cache.get(&key("old")).is_none());
    }

    #[test]
    fn no_chunk_cache_keeps_nothing_and_ignores_its_budget() {
        let cache: &dyn RasterChunkCache = &NoChunkCache;
        let c = chunk(10);
        let ptr = c.bytes.as_ptr();
        let returned = cache.insert(key("a"), c);
        assert_eq!(returned.as_ptr(), ptr);
        assert!(cache.get(&key("a")).is_none());
        cache.set_max_bytes(1 << 20);
        assert_eq!(cache.max_bytes(), 0);
        cache.trim();
        cache.clear();
        assert_eq!(cache.stats(), ChunkCacheStats::default());
    }

    /// A pool with a hard capacity, since arrow's tracking pool is
    /// unbounded.
    #[derive(Debug)]
    struct CappedPool {
        capacity: usize,
        used: Arc<AtomicUsize>,
    }

    #[derive(Debug)]
    struct CappedReservation {
        size: usize,
        used: Arc<AtomicUsize>,
    }

    impl arrow_buffer::MemoryReservation for CappedReservation {
        fn size(&self) -> usize {
            self.size
        }
        fn resize(&mut self, new_size: usize) {
            if new_size >= self.size {
                self.used.fetch_add(new_size - self.size, Ordering::Relaxed);
            } else {
                self.used.fetch_sub(self.size - new_size, Ordering::Relaxed);
            }
            self.size = new_size;
        }
    }

    impl Drop for CappedReservation {
        fn drop(&mut self) {
            self.used.fetch_sub(self.size, Ordering::Relaxed);
        }
    }

    impl MemoryPool for CappedPool {
        fn reserve(&self, size: usize) -> Box<dyn arrow_buffer::MemoryReservation> {
            self.used.fetch_add(size, Ordering::Relaxed);
            Box::new(CappedReservation {
                size,
                used: self.used.clone(),
            })
        }
        fn available(&self) -> isize {
            self.capacity as isize - self.used.load(Ordering::Relaxed) as isize
        }
        fn used(&self) -> usize {
            self.used.load(Ordering::Relaxed)
        }
        fn capacity(&self) -> usize {
            self.capacity
        }
    }

    #[test]
    fn the_pool_is_charged_until_the_last_holder_drops() {
        let used = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(CappedPool {
            capacity: 1024,
            used: used.clone(),
        });
        let cache = InMemoryChunkCache::new(1024).with_memory_pool(pool);
        let held = cache.insert(key("a"), chunk(100));
        assert_eq!(used.load(Ordering::Relaxed), 100);
        // Evicting while a batch still holds the block frees nothing.
        cache.clear();
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(used.load(Ordering::Relaxed), 100);
        drop(held);
        assert_eq!(used.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_full_pool_skips_the_insert_after_evicting_idle_entries() {
        let used = Arc::new(AtomicUsize::new(0));
        let pool = Arc::new(CappedPool {
            capacity: 150,
            used: used.clone(),
        });
        let cache = InMemoryChunkCache::new(1024).with_memory_pool(pool);
        let held = cache.insert(key("a"), chunk(100));
        // `a` is shared and fills most of the pool: `b` cannot be charged.
        let b = chunk(100);
        let b_ptr = b.bytes.as_ptr();
        let returned = cache.insert(key("b"), b);
        assert_eq!(returned.as_ptr(), b_ptr);
        assert_eq!(cache.stats().skipped, 1);
        assert_eq!(used.load(Ordering::Relaxed), 100);
        // Once `a` is idle it can be evicted to make room for `b`.
        drop(held);
        cache.insert(key("b"), chunk(100));
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.evictions, 1);
        assert_eq!(used.load(Ordering::Relaxed), 100);
    }
}
