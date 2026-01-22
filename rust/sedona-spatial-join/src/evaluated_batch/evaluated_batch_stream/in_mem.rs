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
    sync::Arc,
    task::{Context, Poll},
    vec::IntoIter,
};

use arrow_schema::SchemaRef;
use datafusion_common::Result;

use crate::evaluated_batch::{evaluated_batch_stream::EvaluatedBatchStream, EvaluatedBatch};

pub struct InMemoryEvaluatedBatchStream {
    schema: SchemaRef,
    iter: IntoIter<EvaluatedBatch>,
}

impl InMemoryEvaluatedBatchStream {
    pub fn new(schema: SchemaRef, batches: Vec<EvaluatedBatch>) -> Self {
        InMemoryEvaluatedBatchStream {
            schema,
            iter: batches.into_iter(),
        }
    }
}

impl EvaluatedBatchStream for InMemoryEvaluatedBatchStream {
    fn is_external(&self) -> bool {
        false
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl futures::Stream for InMemoryEvaluatedBatchStream {
    type Item = Result<EvaluatedBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut()
            .iter
            .next()
            .map(|batch| Poll::Ready(Some(Ok(batch))))
            .unwrap_or(Poll::Ready(None))
    }
}
