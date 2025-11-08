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

use arrow_array::RecordBatch;
use datafusion_expr::ColumnarValue;
use geo::Rect;
use wkb::reader::Wkb;

use crate::operand_evaluator::EvaluatedGeometryArray;

/// EvaluatedBatch contains the original record batch from the input stream and the evaluated
/// geometry array.
pub(crate) struct EvaluatedBatch {
    /// Original record batch polled from the stream
    pub batch: RecordBatch,
    /// Evaluated geometry array, containing the geometry array containing geometries to be joined,
    /// rects of joined geometries, evaluated distance columnar values if we are running a distance
    /// join, etc.
    pub geom_array: EvaluatedGeometryArray,
}

impl EvaluatedBatch {
    pub fn in_mem_size(&self) -> usize {
        // NOTE: sometimes `geom_array` will reuse the memory of `batch`, especially when
        // the expression for evaluating the geometry is a simple column reference. In this case,
        // the in_mem_size will be overestimated. It is a conservative estimation so there's no risk
        // of running out of memory because of underestimation.
        self.batch.get_array_memory_size() + self.geom_array.in_mem_size()
    }

    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn wkb(&self, idx: usize) -> Option<&Wkb<'_>> {
        let wkbs = self.geom_array.wkbs();
        wkbs[idx].as_ref()
    }

    pub fn rects(&self) -> &Vec<Option<Rect<f32>>> {
        &self.geom_array.rects
    }

    pub fn distance(&self) -> &Option<ColumnarValue> {
        &self.geom_array.distance
    }
}

pub(crate) mod evaluated_batch_stream;
