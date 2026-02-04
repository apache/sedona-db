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

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use datafusion_common::Result;
use futures::{Stream, StreamExt};

use crate::evaluated_batch::{
    evaluated_batch_stream::{EvaluatedBatchStream, SendableEvaluatedBatchStream},
    EvaluatedBatch,
};
use crate::probe::ProbeStreamMetrics;

/// A non-partitioned evaluated batch stream that simply forwards batches from an inner stream,
/// while updating probe stream metrics. This is for running non-partitioned fully in-memory
/// spatial joins.
pub(crate) struct NonPartitionedStream {
    inner: SendableEvaluatedBatchStream,
    metrics: ProbeStreamMetrics,
}

impl NonPartitionedStream {
    pub fn new(inner: SendableEvaluatedBatchStream, metrics: ProbeStreamMetrics) -> Self {
        Self { inner, metrics }
    }
}

impl Stream for NonPartitionedStream {
    type Item = Result<EvaluatedBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                self.metrics.probe_input_batches.add(1);
                self.metrics.probe_input_rows.add(batch.num_rows());
                Poll::Ready(Some(Ok(batch)))
            }
            other => other,
        }
    }
}

impl EvaluatedBatchStream for NonPartitionedStream {
    fn is_external(&self) -> bool {
        self.inner.as_ref().get_ref().is_external()
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.inner.schema()
    }
}
