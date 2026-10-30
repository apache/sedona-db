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
//! file it opens, so the cap is on their total.
//!
//! The budget's size is `sedona.raster.io_concurrency`; it can change while
//! permits are out (see [`RasterIoBudget::set_limit`]).
//!
//! ## Every waiter is a thread
//!
//! The budget is a counter behind a mutex and a condition variable, with no
//! runtime underneath. A permit is waited for only by
//! [`RasterIoBudget::acquire_blocking`], on the thread that will do the I/O,
//! and held only by that thread. There is deliberately no async `acquire`: a
//! future parked on the budget inside a task that then blocks its thread
//! (one that polls a pending load and then evaluates `RS_FromPath` on the
//! same thread, as `SortPreservingMergeExec` or `EnsureLoadedExec` can)
//! would be in line for a slot it could never take, and every waiter behind
//! it would starve. A waiting thread makes progress on its own, and a holder
//! needs nothing but its own thread to finish, so a slot freed anywhere
//! always reaches a thread that will use it and give it back. That is what
//! makes `acquire_blocking` safe to call from any thread: with no runtime,
//! from a blocking-pool thread, on an async worker of either runtime
//! flavor, and inside another executor's `block_on`.
//!
//! Waiters are served in arrival order, and a slot that a queued waiter is
//! due is not given to a newcomer, so no user starves behind the churn of
//! the others.
//!
//! A user must also never wait for a permit while it holds another: two
//! such users could each hold part of the budget and wait forever for the
//! rest. [`RasterIoBudget::acquire_blocking`] checks this in debug builds.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

pub use sedona_common::option::DEFAULT_RASTER_IO_CONCURRENCY;

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
    state: Mutex<State>,
    /// Signalled whenever a slot may have become free for the waiter next in
    /// line: a permit returned, the limit changed, or the waiter ahead of it
    /// took its slot.
    changed: Condvar,
}

#[derive(Debug)]
struct State {
    limit: usize,
    in_use: usize,
    peak_in_use: usize,
    /// The ticket the next waiter takes.
    next_ticket: u64,
    /// The ticket next in line for a slot. `next_ticket - now_serving`
    /// waiters are queued.
    now_serving: u64,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn has_waiters(state: &State) -> bool {
        state.next_ticket != state.now_serving
    }

    /// Count a permit as out.
    fn issue(self: &Arc<Self>, state: &mut State) -> IoPermit {
        state.in_use += 1;
        state.peak_in_use = state.peak_in_use.max(state.in_use);
        IoPermit {
            inner: Arc::clone(self),
        }
    }
}

impl Default for RasterIoBudget {
    fn default() -> Self {
        Self::new(DEFAULT_RASTER_IO_CONCURRENCY)
    }
}

impl RasterIoBudget {
    /// A budget of `limit` operations at once, clamped to at least 1.
    pub fn new(limit: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    limit: limit.max(1),
                    in_use: 0,
                    peak_in_use: 0,
                    next_ticket: 0,
                    now_serving: 0,
                }),
                changed: Condvar::new(),
            }),
        }
    }

    /// The most operations allowed in flight at once.
    pub fn limit(&self) -> usize {
        self.inner.lock().limit
    }

    /// Resize the budget, clamped to at least 1. Growing admits waiters at
    /// once. Shrinking admits no one until the operations in flight have
    /// dropped below the new limit, so it holds once those have finished.
    pub fn set_limit(&self, limit: usize) {
        self.inner.lock().limit = limit.max(1);
        self.inner.changed.notify_all();
    }

    /// Permits free right now for a newcomer: none while waiters are queued,
    /// since the slots that free up are theirs.
    pub fn available(&self) -> usize {
        let state = self.inner.lock();
        if Inner::has_waiters(&state) {
            0
        } else {
            state.limit.saturating_sub(state.in_use)
        }
    }

    /// Permits out right now.
    pub fn in_use(&self) -> usize {
        self.inner.lock().in_use
    }

    /// The most permits ever out at once. For tests and benchmarks.
    pub fn peak_in_use(&self) -> usize {
        self.inner.lock().peak_in_use
    }

    /// Threads waiting for a permit right now.
    pub fn waiting(&self) -> usize {
        let state = self.inner.lock();
        usize::try_from(state.next_ticket - state.now_serving).unwrap_or(usize::MAX)
    }

    /// Whether `self` and `other` are the same budget.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Take a permit if one is free, without waiting; never ahead of a queued
    /// waiter. The permit can move to the thread that does the I/O; like any
    /// permit, it must not be held while waiting for another.
    pub fn try_acquire(&self) -> Option<IoPermit> {
        let mut state = self.inner.lock();
        if Inner::has_waiters(&state) || state.in_use >= state.limit {
            return None;
        }
        Some(self.inner.issue(&mut state))
    }

    /// Block the calling thread until a permit is free.
    ///
    /// Callable from any thread: with no runtime, from a `spawn_blocking`
    /// thread, from an async worker of either runtime flavor, and inside
    /// another executor's `block_on`, since every waiter on the budget is a
    /// thread (see the module docs). On an async worker it stalls that
    /// worker's other tasks while it waits, so on a multi-threaded runtime
    /// wrap the wait (and the I/O) in `tokio::task::block_in_place`.
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
        let mut state = self.inner.lock();
        let ticket = state.next_ticket;
        state.next_ticket += 1;
        while state.now_serving != ticket || state.in_use >= state.limit {
            state = self
                .inner
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.now_serving += 1;
        let permit = self.inner.issue(&mut state);
        drop(state);
        // The waiter now at the head of the line may be due a slot as well.
        self.inner.changed.notify_all();
        BLOCKING_PERMITS_HELD.with(|held| held.set(held.get() + 1));
        BlockingIoPermit {
            _permit: permit,
            _not_send: PhantomData,
        }
    }
}

thread_local! {
    /// Permits from [`RasterIoBudget::acquire_blocking`] held by this thread.
    static BLOCKING_PERMITS_HELD: Cell<usize> = const { Cell::new(0) };
}

/// One slot of a [`RasterIoBudget`], given back when dropped.
#[derive(Debug)]
pub struct IoPermit {
    inner: Arc<Inner>,
}

impl Drop for IoPermit {
    fn drop(&mut self) {
        self.inner.lock().in_use -= 1;
        self.inner.changed.notify_all();
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Spin until `budget` has exactly `n` threads queued.
    fn queued(budget: &RasterIoBudget, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while budget.waiting() != n {
            assert!(
                Instant::now() < deadline,
                "never saw {n} waiter(s) queued; {} are",
                budget.waiting()
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

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
    fn any_limit_is_accepted() {
        // `SET sedona.raster.io_concurrency = 18446744073709551615` parses.
        let budget = RasterIoBudget::new(usize::MAX);
        assert_eq!(budget.limit(), usize::MAX);
        let _permit = budget.try_acquire().unwrap();
        let budget = RasterIoBudget::new(4);
        budget.set_limit(usize::MAX);
        assert_eq!(budget.limit(), usize::MAX);
        assert_eq!(budget.available(), usize::MAX);
        let _permit = budget.acquire_blocking();
        assert_eq!(budget.available(), usize::MAX - 1);
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
            queued(&budget, 1);
            std::thread::sleep(Duration::from_millis(20));
            assert!(!acquired.load(Ordering::SeqCst), "the budget was full");
            drop(held);
        });
        assert!(acquired.load(Ordering::SeqCst));
        assert_eq!(budget.available(), 1);
        assert_eq!(budget.in_use(), 0);
        assert_eq!(budget.waiting(), 0);
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

    #[test]
    fn acquire_blocking_inside_another_executor() {
        // A host that drives its futures with `futures::executor::block_on`
        // evaluates `RS_FromPath` inside that executor; the wait must not
        // need an executor of its own.
        let budget = RasterIoBudget::new(1);
        let held = budget.try_acquire().unwrap();
        let release = std::thread::spawn({
            let budget = budget.clone();
            move || {
                queued(&budget, 1);
                drop(held);
            }
        });
        futures::executor::block_on(async {
            let _permit = budget.acquire_blocking();
            assert_eq!(budget.in_use(), 1);
        });
        release.join().unwrap();
        assert_eq!(budget.available(), 1);
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
        assert_eq!(budget.waiting(), 0);
    }

    #[test]
    fn waiters_are_served_in_arrival_order() {
        let budget = RasterIoBudget::new(1);
        let held = budget.try_acquire().unwrap();
        let order = Mutex::new(Vec::new());
        // The first waiter keeps its permit until the test has looked at the
        // budget, so the second is still in line when the assertions run.
        let (release_first, first_released) = std::sync::mpsc::channel::<()>();
        let (first_acquired, acquired_first) = std::sync::mpsc::channel::<()>();
        std::thread::scope(|scope| {
            let (budget, order) = (&budget, &order);
            scope.spawn(move || {
                let _permit = budget.acquire_blocking();
                order.lock().unwrap().push("first");
                first_acquired.send(()).unwrap();
                first_released.recv().unwrap();
            });
            queued(budget, 1);
            scope.spawn(|| {
                let _permit = budget.acquire_blocking();
                order.lock().unwrap().push("second");
            });
            queued(budget, 2);
            // The freed slot goes to the first waiter; the second stays in
            // line, and a newcomer gets nothing.
            drop(held);
            // Leaving the queue happens before acquire_blocking() returns.
            // Wait for the recorded acquisition before inspecting the order.
            acquired_first
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            assert_eq!(budget.waiting(), 1);
            assert_eq!(*order.lock().unwrap(), ["first"]);
            assert_eq!(budget.in_use(), 1);
            assert!(budget.try_acquire().is_none());
            assert_eq!(budget.available(), 0);
            release_first.send(()).unwrap();
        });
        assert_eq!(*order.lock().unwrap(), ["first", "second"]);
        assert_eq!(budget.waiting(), 0);
        assert!(budget.try_acquire().is_some());
    }

    #[test]
    fn growing_admits_a_waiter() {
        let budget = RasterIoBudget::new(1);
        let _held = budget.try_acquire().unwrap();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let _permit = budget.acquire_blocking();
            });
            queued(&budget, 1);
            budget.set_limit(2);
        });
        assert_eq!(budget.in_use(), 1);
        assert_eq!(budget.available(), 1);
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
    fn shrinking_holds_once_permits_return() {
        let budget = RasterIoBudget::new(4);
        let a = budget.try_acquire().unwrap();
        let b = budget.try_acquire().unwrap();
        let c = budget.try_acquire().unwrap();
        // One permit is free; shrinking to 1 takes it away and admits no one
        // until the three out have come back below the limit.
        budget.set_limit(1);
        assert_eq!(budget.available(), 0);
        assert!(budget.try_acquire().is_none());
        drop(a);
        drop(b);
        assert_eq!(budget.available(), 0, "still at the limit");
        drop(c);
        assert_eq!(budget.available(), 1);

        // Growing again before they return admits at once.
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
