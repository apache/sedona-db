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

//! A session-wide budget for blocking raster I/O.
//!
//! [`RasterIoBudget`] bounds how many blocking file operations (an open, a
//! header read, a pixel read) are in flight at once across everything that
//! shares it. A session holds one and hands it to every user: the GDAL pixel
//! loader takes a permit per file it reads, and `RS_FromPath` takes one per
//! file it opens. Both kinds of permit come from one
//! `tokio::sync::Semaphore`, which works with or without a runtime, so the
//! cap is on their total.
//!
//! The budget's size is `sedona.raster.io_concurrency`; it can change while
//! permits are out (see [`RasterIoBudget::set_limit`]).
//!
//! ## Every waiter is a thread
//!
//! A permit is waited for only by [`RasterIoBudget::acquire_blocking`], on
//! the thread that will do the I/O, and held only by that thread. There is
//! deliberately no async `acquire`. Tokio's semaphore hands a released
//! permit to the first queued waiter at release time, whether or not that
//! waiter's future is ever polled again. A future queued as a waiter inside
//! a task that then blocks its thread (for example a task that polls a
//! pending load and then evaluates `RS_FromPath` on the same thread) would
//! receive the permit and never run to use or return it, and every later
//! waiter would queue behind it forever. A waiting thread makes progress on
//! its own, and a holder needs nothing but its own thread to finish, so a
//! permit released anywhere always reaches a thread that will use it and
//! give it back. That is what makes `acquire_blocking` safe to call from any
//! thread.
//!
//! A user must also never wait for a permit while it holds another: two
//! such users could each hold part of the budget and wait forever for the
//! rest. [`RasterIoBudget::acquire_blocking`] checks this in debug builds.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

pub use sedona_common::option::DEFAULT_RASTER_IO_CONCURRENCY;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// A counting budget of blocking raster I/O operations, shared by clones.
///
/// Permits are owned values that give their slot back when dropped,
/// including during unwinding.
#[derive(Debug, Clone)]
pub struct RasterIoBudget {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    semaphore: Arc<Semaphore>,
    size: Mutex<Size>,
    in_use: AtomicUsize,
    peak_in_use: AtomicUsize,
}

#[derive(Debug)]
struct Size {
    limit: usize,
    /// Permits still to be retired after a shrink: when the budget shrinks
    /// below the permits out, the excess is retired as those permits drop.
    debt: usize,
}

impl Default for RasterIoBudget {
    fn default() -> Self {
        Self::new(DEFAULT_RASTER_IO_CONCURRENCY)
    }
}

impl RasterIoBudget {
    /// A budget of `limit` operations at once, clamped to at least 1.
    pub fn new(limit: usize) -> Self {
        let limit = limit.max(1);
        Self {
            inner: Arc::new(Inner {
                semaphore: Arc::new(Semaphore::new(limit)),
                size: Mutex::new(Size { limit, debt: 0 }),
                in_use: AtomicUsize::new(0),
                peak_in_use: AtomicUsize::new(0),
            }),
        }
    }

    /// The most operations allowed in flight at once.
    pub fn limit(&self) -> usize {
        self.lock_size().limit
    }

    /// Resize the budget, clamped to at least 1. Growing frees the new
    /// permits at once. Shrinking takes free permits out of circulation, and
    /// retires the rest as permits now in use are dropped, so the operations
    /// in flight never exceed the new limit once those have finished.
    pub fn set_limit(&self, limit: usize) {
        let limit = limit.max(1);
        let mut size = self.lock_size();
        if limit > size.limit {
            let grow = limit - size.limit;
            let cancelled = grow.min(size.debt);
            size.debt -= cancelled;
            self.inner.semaphore.add_permits(grow - cancelled);
        } else if limit < size.limit {
            let shrink = size.limit - limit;
            let forgotten = self.inner.semaphore.forget_permits(shrink);
            size.debt += shrink - forgotten;
        }
        size.limit = limit;
    }

    /// Permits free right now.
    pub fn available(&self) -> usize {
        self.inner.semaphore.available_permits()
    }

    /// Permits out right now.
    pub fn in_use(&self) -> usize {
        self.inner.in_use.load(Ordering::SeqCst)
    }

    /// The most permits ever out at once. For tests and benchmarks.
    pub fn peak_in_use(&self) -> usize {
        self.inner.peak_in_use.load(Ordering::SeqCst)
    }

    /// Whether `self` and `other` are the same budget.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Take a permit if one is free, without waiting. The permit can move to
    /// the thread that does the I/O; like any permit, it must not be held
    /// while waiting for another.
    pub fn try_acquire(&self) -> Option<IoPermit> {
        Arc::clone(&self.inner.semaphore)
            .try_acquire_owned()
            .ok()
            .map(|permit| self.issue(permit))
    }

    /// Block the calling thread until a permit is free.
    ///
    /// Callable from any thread: with no runtime, from a `spawn_blocking`
    /// thread, and from an async worker of either runtime flavor, since every
    /// waiter on the budget is a thread (see the module docs). On an async
    /// worker it stalls that worker's other tasks while it waits, so on a
    /// multi-threaded runtime wrap the wait (and the I/O) in
    /// `tokio::task::block_in_place`.
    ///
    /// The permit stays on the thread that took it. In debug builds this
    /// panics if the thread already holds a permit from this method, since
    /// waiting while holding one can deadlock (see the module docs).
    pub fn acquire_blocking(&self) -> BlockingIoPermit {
        debug_assert_eq!(
            BLOCKING_PERMITS_HELD.with(Cell::get),
            0,
            "waiting for a raster I/O permit while this thread holds another"
        );
        let permit = futures::executor::block_on(Arc::clone(&self.inner.semaphore).acquire_owned())
            .expect("the I/O budget's semaphore is never closed");
        BLOCKING_PERMITS_HELD.with(|held| held.set(held.get() + 1));
        BlockingIoPermit {
            _permit: self.issue(permit),
            _not_send: PhantomData,
        }
    }

    fn issue(&self, permit: OwnedSemaphorePermit) -> IoPermit {
        let now = self.inner.in_use.fetch_add(1, Ordering::SeqCst) + 1;
        self.inner.peak_in_use.fetch_max(now, Ordering::SeqCst);
        IoPermit {
            permit: Some(permit),
            inner: Arc::clone(&self.inner),
        }
    }

    fn lock_size(&self) -> std::sync::MutexGuard<'_, Size> {
        self.inner
            .size
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

thread_local! {
    /// Permits from [`RasterIoBudget::acquire_blocking`] held by this thread.
    static BLOCKING_PERMITS_HELD: Cell<usize> = const { Cell::new(0) };
}

/// One slot of a [`RasterIoBudget`], given back when dropped.
#[derive(Debug)]
pub struct IoPermit {
    permit: Option<OwnedSemaphorePermit>,
    inner: Arc<Inner>,
}

impl Drop for IoPermit {
    fn drop(&mut self) {
        self.inner.in_use.fetch_sub(1, Ordering::SeqCst);
        let mut size = self
            .inner
            .size
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        // Dropping the permit returns it to the semaphore, unless the budget
        // shrank while it was out: then it is retired instead.
        if let Some(permit) = self.permit.take()
            && size.debt > 0
        {
            size.debt -= 1;
            permit.forget();
        }
    }
}

/// A permit from [`RasterIoBudget::acquire_blocking`]. It cannot leave the
/// thread that took it, which lets debug builds catch a thread waiting for a
/// second permit while it holds one.
#[derive(Debug)]
pub struct BlockingIoPermit {
    _permit: IoPermit,
    _not_send: PhantomData<*const ()>,
}

impl Drop for BlockingIoPermit {
    fn drop(&mut self) {
        BLOCKING_PERMITS_HELD.with(|held| held.set(held.get() - 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    #[test]
    fn limit_is_clamped_to_one() {
        assert_eq!(RasterIoBudget::new(0).limit(), 1);
        assert_eq!(RasterIoBudget::new(0).available(), 1);
        let budget = RasterIoBudget::new(4);
        budget.set_limit(0);
        assert_eq!(budget.limit(), 1);
        assert_eq!(budget.available(), 1);
        assert_eq!(
            RasterIoBudget::default().limit(),
            DEFAULT_RASTER_IO_CONCURRENCY
        );
    }

    #[test]
    fn clones_share_one_budget() {
        let budget = RasterIoBudget::new(2);
        let clone = budget.clone();
        assert!(budget.ptr_eq(&clone));
        assert!(!budget.ptr_eq(&RasterIoBudget::new(2)));
        let _a = clone.try_acquire().unwrap();
        let _b = budget.try_acquire().unwrap();
        assert!(budget.try_acquire().is_none());
        assert_eq!(clone.in_use(), 2);
    }

    #[test]
    fn acquire_blocking_without_a_runtime() {
        let budget = RasterIoBudget::new(1);
        let held = budget.try_acquire().unwrap();
        let acquired = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _permit = budget.acquire_blocking();
                acquired.store(true, Ordering::SeqCst);
            });
            std::thread::sleep(Duration::from_millis(20));
            assert!(!acquired.load(Ordering::SeqCst), "the budget was full");
            drop(held);
        });
        assert!(acquired.load(Ordering::SeqCst));
        assert_eq!(budget.available(), 1);
        assert_eq!(budget.in_use(), 0);
    }

    #[test]
    fn acquire_blocking_on_a_current_thread_runtime() {
        // The permit is released by a `spawn_blocking` thread, which keeps
        // running while the runtime's only worker is blocked.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let budget = RasterIoBudget::new(1);
        runtime.block_on(async {
            let held = budget.try_acquire().unwrap();
            let release = tokio::task::spawn_blocking(move || {
                std::thread::sleep(Duration::from_millis(20));
                drop(held);
            });
            let permit = budget.acquire_blocking();
            assert_eq!(budget.in_use(), 1);
            drop(permit);
            release.await.unwrap();
        });
        assert_eq!(budget.available(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acquire_blocking_in_block_in_place() {
        let budget = RasterIoBudget::new(1);
        let held = budget.try_acquire().unwrap();
        let releaser = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            drop(held);
        });
        tokio::task::block_in_place(|| {
            let _permit = budget.acquire_blocking();
            assert_eq!(budget.in_use(), 1);
        });
        releaser.await.unwrap();
        assert_eq!(budget.available(), 1);
        assert_eq!(budget.in_use(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn blocking_pool_and_plain_thread_users_share_the_cap() {
        let budget = RasterIoBudget::new(3);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        // The first two operations to start wait for each other, so two are
        // certainly in flight at once however the threads are scheduled.
        let arrivals = Arc::new(AtomicUsize::new(0));
        let first_two = Arc::new(std::sync::Barrier::new(2));
        let work = {
            let in_flight = Arc::clone(&in_flight);
            let peak = Arc::clone(&peak);
            move || {
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                if arrivals.fetch_add(1, Ordering::SeqCst) < 2 {
                    first_two.wait();
                }
                std::thread::sleep(Duration::from_millis(2));
                in_flight.fetch_sub(1, Ordering::SeqCst);
            }
        };

        // Users on the runtime's blocking pool, as the GDAL loader's reads.
        let mut tasks = Vec::new();
        for _ in 0..24 {
            let budget = budget.clone();
            let work = work.clone();
            tasks.push(tokio::task::spawn_blocking(move || {
                let _permit = budget.acquire_blocking();
                work();
            }));
        }
        // Users on plain threads, as `RS_FromPath`'s, at the same time.
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let budget = budget.clone();
                let work = work.clone();
                std::thread::spawn(move || {
                    for _ in 0..6 {
                        let _permit = budget.acquire_blocking();
                        work();
                    }
                })
            })
            .collect();
        for task in tasks {
            task.await.unwrap();
        }
        tokio::task::block_in_place(|| {
            for thread in threads {
                thread.join().unwrap();
            }
        });

        let peak = peak.load(Ordering::SeqCst);
        assert!((2..=3).contains(&peak), "peak in flight {peak}");
        assert!(budget.peak_in_use() <= 3);
        assert_eq!(budget.available(), 3);
        assert_eq!(budget.in_use(), 0);
    }

    #[test]
    fn permits_return_when_their_holder_panics() {
        let budget = RasterIoBudget::new(2);
        let unwound = std::thread::spawn({
            let budget = budget.clone();
            move || {
                let _permit = budget.acquire_blocking();
                panic!("holder panicked");
            }
        })
        .join();
        assert!(unwound.is_err());
        let moved = budget.try_acquire().unwrap();
        let unwound = std::thread::spawn(move || {
            let _permit = moved;
            panic!("holder panicked");
        })
        .join();
        assert!(unwound.is_err());
        assert_eq!(budget.available(), 2);
        assert_eq!(budget.in_use(), 0);
    }

    #[test]
    fn growing_frees_permits_at_once() {
        let budget = RasterIoBudget::new(1);
        let _a = budget.try_acquire().unwrap();
        assert!(budget.try_acquire().is_none());
        budget.set_limit(3);
        assert_eq!(budget.limit(), 3);
        let _b = budget.try_acquire().unwrap();
        let _c = budget.try_acquire().unwrap();
        assert!(budget.try_acquire().is_none());
    }

    #[test]
    fn shrinking_retires_permits_as_they_return() {
        let budget = RasterIoBudget::new(4);
        let a = budget.try_acquire().unwrap();
        let b = budget.try_acquire().unwrap();
        let c = budget.try_acquire().unwrap();
        // One permit is free; shrinking to 1 forgets it and owes two more.
        budget.set_limit(1);
        assert_eq!(budget.available(), 0);
        drop(a);
        drop(b);
        assert_eq!(budget.available(), 0, "both returns went to the debt");
        drop(c);
        assert_eq!(budget.available(), 1);

        // Growing again cancels outstanding debt before adding permits.
        budget.set_limit(2);
        let a = budget.try_acquire().unwrap();
        let b = budget.try_acquire().unwrap();
        budget.set_limit(1);
        budget.set_limit(2);
        assert_eq!(budget.available(), 0);
        drop(a);
        drop(b);
        assert_eq!(budget.available(), 2);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn nested_blocking_acquire_is_caught_in_debug_builds() {
        // Waiting for a second permit while holding one can deadlock two
        // such waiters against each other; the debug assertion documents
        // that no user does it.
        let budget = RasterIoBudget::new(4);
        let nested = std::thread::spawn(move || {
            let _outer = budget.acquire_blocking();
            let _inner = budget.acquire_blocking();
        })
        .join();
        assert!(nested.is_err());
    }
}
