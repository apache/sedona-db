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

use std::pin::Pin;

use futures::Stream;

use crate::evaluated_batch::EvaluatedBatch;
use datafusion_common::Result;

/// A stream that produces [`EvaluatedBatch`] items. This stream may have purely in-memory or
/// out-of-core implementations. The type of the stream could be queried calling `is_external()`.
pub(crate) trait EvaluatedBatchStream: Stream<Item = Result<EvaluatedBatch>> {
    /// Returns true if this stream is an external stream, where batch data were spilled to disk.
    fn is_external(&self) -> bool;
}

pub(crate) type SendableEvaluatedBatchStream = Pin<Box<dyn EvaluatedBatchStream + Send>>;

pub(crate) mod in_mem;
