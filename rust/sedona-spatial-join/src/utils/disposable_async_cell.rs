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

use std::fmt;

use parking_lot::Mutex;
use tokio::sync::Notify;

/// Error returned when writing to a [`DisposableAsyncCell`] fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CellSetError {
    /// The cell has already been disposed, so new values are rejected.
    Disposed,

    /// The cell already has a value.
    AlreadySet,
}

/// An asynchronous cell that can be set at most once before either being
/// disposed or read by any number of waiters.
///
/// This is used as a lightweight one-shot coordination primitive in the spatial
/// join implementation. For example, `PartitionedIndexProvider` keeps one
/// `DisposableAsyncCell` per regular partition to publish either a successfully
/// built `SpatialIndex` (or the build error) exactly once. Concurrent
/// `SpatialJoinStream`s racing to probe the same partition can then await the
/// same shared result instead of building duplicate indexes.
///
/// When an index is no longer needed (e.g. the last stream finishes a
/// partition), the cell can be disposed to free resources.
///
/// Awaiters calling [`DisposableAsyncCell::get`] will park until a value is set
/// or the cell is disposed. Once disposed, `get` returns `None` and `set`
/// returns [`CellSetError::Disposed`].
pub(crate) struct DisposableAsyncCell<T> {
    state: Mutex<CellState<T>>,
    notify: Notify,
}

impl<T> fmt::Debug for DisposableAsyncCell<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisposableAsyncCell")
    }
}

impl<T> Default for DisposableAsyncCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DisposableAsyncCell<T> {
    /// Creates a new empty cell with no stored value.
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(CellState::Empty),
            notify: Notify::new(),
        }
    }

    /// Marks the cell as disposed and wakes every waiter.
    pub(crate) fn dispose(&self) {
        {
            let mut state = self.state.lock();
            *state = CellState::Disposed;
        }
        self.notify.notify_waiters();
    }

    /// Check whether the cell has a value or not.
    pub(crate) fn is_set(&self) -> bool {
        let state = self.state.lock();
        matches!(*state, CellState::Value(_))
    }

    /// Check whether the cell is empty (not set or disposed)
    pub(crate) fn is_empty(&self) -> bool {
        let state = self.state.lock();
        matches!(*state, CellState::Empty)
    }
}

impl<T: Clone> DisposableAsyncCell<T> {
    /// Waits until a value is set or the cell is disposed.
    /// Returns `None` if the cell is disposed without a value.
    pub(crate) async fn get(&self) -> Option<T> {
        loop {
            let notified = self.notify.notified();
            {
                let state = self.state.lock();
                match &*state {
                    CellState::Value(val) => return Some(val.clone()),
                    CellState::Disposed => return None,
                    CellState::Empty => {}
                }
            }
            notified.await;
        }
    }

    /// Stores the provided value if the cell is still empty.
    /// Fails if a value already exists or the cell has been disposed.
    pub(crate) fn set(&self, value: T) -> std::result::Result<(), CellSetError> {
        {
            let mut state = self.state.lock();
            match &mut *state {
                CellState::Empty => *state = CellState::Value(value),
                CellState::Disposed => return Err(CellSetError::Disposed),
                CellState::Value(_) => return Err(CellSetError::AlreadySet),
            }
        }

        self.notify.notify_waiters();
        Ok(())
    }
}

enum CellState<T> {
    Empty,
    Value(T),
    Disposed,
}

#[cfg(test)]
mod tests {
    use super::{CellSetError, DisposableAsyncCell};
    use std::sync::Arc;
    use tokio::task;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn get_returns_value_once_set() {
        let cell = DisposableAsyncCell::new();
        cell.set(42).expect("set succeeds");
        assert_eq!(Some(42), cell.get().await);
    }

    #[tokio::test]
    async fn multiple_waiters_receive_same_value() {
        let cell = Arc::new(DisposableAsyncCell::new());
        let cloned = Arc::clone(&cell);
        let waiter_one = task::spawn(async move { cloned.get().await });
        let cloned = Arc::clone(&cell);
        let waiter_two = task::spawn(async move { cloned.get().await });

        cell.set(String::from("value")).expect("set succeeds");
        assert_eq!(Some("value".to_string()), waiter_one.await.unwrap());
        assert_eq!(Some("value".to_string()), waiter_two.await.unwrap());
    }

    #[tokio::test]
    async fn dispose_unblocks_waiters() {
        let cell = Arc::new(DisposableAsyncCell::<i32>::new());
        let waiter = tokio::spawn({
            let cloned = Arc::clone(&cell);
            async move { cloned.get().await }
        });

        cell.dispose();
        assert_eq!(None, waiter.await.unwrap());
    }

    #[tokio::test]
    async fn set_after_dispose_fails() {
        let cell = DisposableAsyncCell::new();
        cell.dispose();
        assert_eq!(Err(CellSetError::Disposed), cell.set(5));
    }

    #[tokio::test]
    async fn set_twice_rejects_second_value() {
        let cell = DisposableAsyncCell::new();
        cell.set("first").expect("initial set succeeds");
        assert_eq!(Err(CellSetError::AlreadySet), cell.set("second"));
        assert_eq!(Some("first"), cell.get().await);
    }

    #[tokio::test]
    async fn get_waits_until_value_is_set() {
        let cell = Arc::new(DisposableAsyncCell::new());
        let cloned = Arc::clone(&cell);
        let waiter = tokio::spawn(async move { cloned.get().await });

        sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished());

        cell.set(99).expect("set succeeds");
        assert_eq!(Some(99), waiter.await.unwrap());
    }
}
