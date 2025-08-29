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
/// This module contains the OnceAsync and OnceFut types, which are used to
/// run an async closure once. The source code was copied from DataFusion
/// https://github.com/apache/datafusion/blob/48.0.0/datafusion/physical-plan/src/joins/utils.rs
use std::task::{Context, Poll};
use std::{
    fmt::{self, Debug},
    future::Future,
    sync::Arc,
};

use datafusion_common::{DataFusionError, Result, SharedResult};
use futures::{
    future::{BoxFuture, Shared},
    ready, FutureExt,
};
use parking_lot::Mutex;

/// A [`OnceAsync`] runs an `async` closure once, where multiple calls to
/// [`OnceAsync::try_once`] return a [`OnceFut`] that resolves to the result of the
/// same computation.
///
/// This is useful for joins where the results of one child are needed to proceed
/// with multiple output stream
///
///
/// For example, in a hash join, one input is buffered and shared across
/// potentially multiple output partitions. Each output partition must wait for
/// the hash table to be built before proceeding.
///
/// Each output partition waits on the same `OnceAsync` before proceeding.
pub(crate) struct OnceAsync<T> {
    fut: Mutex<Option<SharedResult<OnceFut<T>>>>,
}

impl<T> Default for OnceAsync<T> {
    fn default() -> Self {
        Self {
            fut: Mutex::new(None),
        }
    }
}

impl<T> Debug for OnceAsync<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OnceAsync")
    }
}

impl<T: 'static> OnceAsync<T> {
    /// If this is the first call to this function on this object, will invoke
    /// `f` to obtain a future and return a [`OnceFut`] referring to this. `f`
    /// may fail, in which case its error is returned.
    ///
    /// If this is not the first call, will return a [`OnceFut`] referring
    /// to the same future as was returned by the first call - or the same
    /// error if the initial call to `f` failed.
    pub(crate) fn try_once<F, Fut>(&self, f: F) -> Result<OnceFut<T>>
    where
        F: FnOnce() -> Result<Fut>,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        self.fut
            .lock()
            .get_or_insert_with(|| f().map(OnceFut::new).map_err(Arc::new))
            .clone()
            .map_err(DataFusionError::Shared)
    }
}

/// The shared future type used internally within [`OnceAsync`]
type OnceFutPending<T> = Shared<BoxFuture<'static, SharedResult<Arc<T>>>>;

/// A [`OnceFut`] represents a shared asynchronous computation, that will be evaluated
/// once for all [`Clone`]'s, with [`OnceFut::get`] providing a non-consuming interface
/// to drive the underlying [`Future`] to completion
pub(crate) struct OnceFut<T> {
    state: OnceFutState<T>,
}

impl<T> Clone for OnceFut<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

enum OnceFutState<T> {
    Pending(OnceFutPending<T>),
    Ready(SharedResult<Arc<T>>),
}

impl<T> Clone for OnceFutState<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Pending(p) => Self::Pending(p.clone()),
            Self::Ready(r) => Self::Ready(r.clone()),
        }
    }
}

impl<T: 'static> OnceFut<T> {
    /// Create a new [`OnceFut`] from a [`Future`]
    pub(crate) fn new<Fut>(fut: Fut) -> Self
    where
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        Self {
            state: OnceFutState::Pending(
                fut.map(|res| res.map(Arc::new).map_err(Arc::new))
                    .boxed()
                    .shared(),
            ),
        }
    }

    /// Get the result of the computation if it is ready, without consuming it
    #[allow(unused)]
    pub(crate) fn get(&mut self, cx: &mut Context<'_>) -> Poll<Result<&T>> {
        if let OnceFutState::Pending(fut) = &mut self.state {
            let r = ready!(fut.poll_unpin(cx));
            self.state = OnceFutState::Ready(r);
        }

        // Cannot use loop as this would trip up the borrow checker
        match &self.state {
            OnceFutState::Pending(_) => unreachable!(),
            OnceFutState::Ready(r) => Poll::Ready(
                r.as_ref()
                    .map(|r| r.as_ref())
                    .map_err(DataFusionError::from),
            ),
        }
    }

    /// Get shared reference to the result of the computation if it is ready, without consuming it
    pub(crate) fn get_shared(&mut self, cx: &mut Context<'_>) -> Poll<Result<Arc<T>>> {
        if let OnceFutState::Pending(fut) = &mut self.state {
            let r = ready!(fut.poll_unpin(cx));
            self.state = OnceFutState::Ready(r);
        }

        match &self.state {
            OnceFutState::Pending(_) => unreachable!(),
            OnceFutState::Ready(r) => Poll::Ready(r.clone().map_err(DataFusionError::Shared)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;

    use super::*;

    #[tokio::test]
    async fn check_error_nesting() {
        let once_fut =
            OnceFut::<()>::new(async { Err(DataFusionError::Internal("some error".to_string())) });

        struct TestFut(OnceFut<()>);
        impl Future for TestFut {
            type Output = Result<()>;

            fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
                match ready!(self.0.get(cx)) {
                    Ok(()) => Poll::Ready(Ok(())),
                    Err(e) => Poll::Ready(Err(e)),
                }
            }
        }

        let res = TestFut(once_fut).await;
        let arrow_err_from_fut = res.expect_err("once_fut always return error");

        let wrapped_err = arrow_err_from_fut;
        let root_err = wrapped_err.find_root();

        let _expected = DataFusionError::Internal("some error".to_string());

        assert!(matches!(root_err, _expected))
    }
}
