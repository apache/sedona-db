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

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::executor::{ThreadPool, ThreadPoolBuilder};
use futures::task::AtomicWaker;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Consumer-side limits. Each read/list gets its own budget and cancellation state.
#[derive(Clone, Debug)]
pub struct ObjectStoreOptions {
    pub call_timeout: Duration,
    pub idle_timeout: Duration,
    pub operation_timeout: Duration,
}
impl Default for ObjectStoreOptions {
    fn default() -> Self {
        Self {
            call_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(30),
            operation_timeout: Duration::from_secs(300),
        }
    }
}
impl ObjectStoreOptions {
    pub(crate) fn validate(&self) -> object_store::Result<()> {
        for timeout in [self.call_timeout, self.idle_timeout, self.operation_timeout] {
            if timeout.is_zero() || Instant::now().checked_add(timeout).is_none() {
                return Err(failure(
                    io::ErrorKind::InvalidInput,
                    "invalid ObjectStore timeout",
                ));
            }
        }
        Ok(())
    }
}

/// Fixed foreign polling threads and a bounded number of live stores/operations.
///
/// Pending streams yield these threads normally. A callback that blocks forever
/// consumes a thread and its admission slot permanently. Saturation fails fast;
/// it never creates replacement threads or an unbounded queue. Share an executor
/// across imports (the default is shared within this linked copy of the crate) to bound aggregate resources.
pub struct ObjectStoreExecutor {
    pool: ThreadPool,
    slots: Arc<Semaphore>,
}
impl ObjectStoreExecutor {
    pub fn new(threads: usize, capacity: usize) -> io::Result<Arc<Self>> {
        if threads == 0 || !(2..=Semaphore::MAX_PERMITS).contains(&capacity) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "need threads > 0 and capacity >= 2",
            ));
        }
        timer_runtime()?;
        Ok(Arc::new(Self {
            pool: ThreadPoolBuilder::new()
                .pool_size(threads)
                .name_prefix("sedona-ffi-store-")
                .create()?,
            slots: Arc::new(Semaphore::new(capacity)),
        }))
    }

    pub fn shared() -> io::Result<Arc<Self>> {
        static SHARED: OnceLock<Arc<ObjectStoreExecutor>> = OnceLock::new();
        // Initialization is rare; a losing initializer drops its unused pool.
        if let Some(pool) = SHARED.get() {
            return Ok(pool.clone());
        }
        let pool = Self::new(4, 256)?;
        Ok(SHARED.get_or_init(|| pool).clone())
    }

    pub(crate) fn admit(&self) -> object_store::Result<OwnedSemaphorePermit> {
        self.slots.clone().try_acquire_owned().map_err(|_| {
            failure(
                io::ErrorKind::WouldBlock,
                "FFI ObjectStore executor is at capacity",
            )
        })
    }

    pub(crate) fn spawn(&self, future: impl Future<Output = ()> + Send + 'static) {
        self.pool.spawn_ok(future);
    }
}

pub(crate) fn failure(kind: io::ErrorKind, message: impl Into<String>) -> object_store::Error {
    object_store::Error::Generic {
        store: "SedonaCObjectStore",
        source: Box::new(io::Error::new(kind, message.into())),
    }
}
pub(crate) fn invalid(message: impl Into<String>) -> object_store::Error {
    failure(io::ErrorKind::InvalidData, message)
}
pub(crate) fn unsupported(message: impl Into<String>) -> object_store::Error {
    object_store::Error::NotSupported {
        source: message.into().into(),
    }
}

/// Timer driver is deliberately independent of the foreign polling threads and
/// of the caller's executor. Futures can be used from futures::executor::block_on.
fn timer_runtime() -> io::Result<&'static tokio::runtime::Runtime> {
    static TIMERS: OnceLock<io::Result<tokio::runtime::Runtime>> = OnceLock::new();
    TIMERS
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_name("sedona-ffi-deadlines")
                .enable_time()
                .build()
        })
        .as_ref()
        .map_err(|error| io::Error::new(error.kind(), error.to_string()))
}
pub(crate) fn timer(deadline: Instant) -> Pin<Box<tokio::time::Sleep>> {
    // Executor construction has already checked timer initialization.
    let runtime = timer_runtime().expect("timer initialized by ObjectStoreExecutor");
    let _entered = runtime.enter();
    Box::pin(tokio::time::sleep_until(deadline.into()))
}

#[derive(Default)]
pub(crate) struct Cancellation {
    cancelled: AtomicBool,
    waker: AtomicWaker,
}
impl Cancellation {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.waker.wake();
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
    pub async fn cancelled(&self) {
        futures::future::poll_fn(|cx| {
            self.waker.register(cx.waker());
            if self.is_cancelled() {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        })
        .await
    }
}

/// Only local state is touched on the query thread, even when a foreign cancel or
/// destructor never returns. The worker owns all raw FFI values until cleanup.
pub(crate) struct CancelOnDrop(pub Arc<Cancellation>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub(crate) async fn wait<T>(
    receiver: impl Future<Output = T>,
    deadline: Instant,
) -> object_store::Result<T> {
    let timeout = timer(deadline);
    futures::pin_mut!(receiver);
    match futures::future::select(timeout, receiver).await {
        futures::future::Either::Left(_) => Err(failure(
            io::ErrorKind::TimedOut,
            "FFI ObjectStore call timed out",
        )),
        futures::future::Either::Right((result, _)) => Ok(result),
    }
}

pub(crate) struct ResponseStream<T> {
    pub receiver: tokio::sync::mpsc::Receiver<object_store::Result<T>>,
    pub cancel: CancelOnDrop,
    pub deadline: Pin<Box<tokio::time::Sleep>>,
    pub idle_timeout: Duration,
    pub idle: Option<Pin<Box<tokio::time::Sleep>>>,
    pub finished: bool,
}
impl<T> futures::Stream for ResponseStream<T> {
    type Item = object_store::Result<T>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        if self.idle.is_none() {
            self.idle = Some(timer(Instant::now() + self.idle_timeout));
        }
        if self.deadline.as_mut().poll(cx).is_ready()
            || self.idle.as_mut().unwrap().as_mut().poll(cx).is_ready()
        {
            self.finished = true;
            self.cancel.0.cancel();
            self.receiver.close();
            return Poll::Ready(Some(Err(failure(
                io::ErrorKind::TimedOut,
                "FFI ObjectStore stream timed out",
            ))));
        }
        match self.receiver.poll_recv(cx) {
            Poll::Ready(result) => {
                self.idle = None;
                if result.is_none() || matches!(result, Some(Err(_))) {
                    self.finished = true;
                    self.cancel.0.cancel();
                }
                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
