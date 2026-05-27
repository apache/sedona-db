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
use arrow_schema::SchemaRef;
use datafusion_common::Result;

use crate::{
    operand_evaluator::EvaluatedGeometryArray, utils::arrow_utils::get_record_batch_memory_size,
};

/// EvaluatedBatch contains the original record batch from the input stream and the evaluated
/// geometry array.
pub struct EvaluatedBatch {
    /// Original record batch polled from the stream
    pub batch: RecordBatch,
    /// Evaluated geometry array, containing the geometry array containing geometries to be joined,
    /// rects of joined geometries, evaluated distance columnar values if we are running a distance
    /// join, etc.
    pub geom_array: EvaluatedGeometryArray,
}

impl EvaluatedBatch {
    pub fn in_mem_size(&self) -> Result<usize> {
        let record_batch_size = get_record_batch_memory_size(&self.batch)?;

        // If the geometry expression was a simple input column reference, DataFusion returns the
        // same Arrow array that is already owned by `batch`. Count that buffer once and only add the
        // evaluated metadata (bounds, distance, WKB handles) here.
        if self.geometry_array_is_batch_column() {
            Ok(record_batch_size + self.geom_array.evaluation_overhead_in_mem_size()?)
        } else {
            Ok(record_batch_size + self.geom_array.in_mem_size()?)
        }
    }

    fn geometry_array_is_batch_column(&self) -> bool {
        self.batch
            .columns()
            .iter()
            .any(|array| std::sync::Arc::ptr_eq(array, self.geom_array.geometry_array()))
    }

    pub fn schema(&self) -> SchemaRef {
        self.batch.schema()
    }

    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }
}

pub mod evaluated_batch_stream;
pub mod spill;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::arrow_utils::get_array_memory_size;
    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::{DataType, Field, Schema};
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_array;
    use std::sync::Arc;

    fn geometry_array() -> ArrayRef {
        create_array(&[Some("POINT (1 2)"), Some("POINT (3 4)")], &WKB_GEOMETRY)
    }

    fn batch_with_geometry(geom: ArrayRef) -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            WKB_GEOMETRY.to_storage_field("geom", true)?,
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef, geom],
        )
        .map_err(Into::into)
    }

    #[test]
    fn in_mem_size_does_not_double_count_reused_geometry_column() -> Result<()> {
        let geom = geometry_array();
        let batch = batch_with_geometry(Arc::clone(&geom))?;
        let geom_array = EvaluatedGeometryArray::try_new(geom, &WKB_GEOMETRY)?;
        let evaluated_batch = EvaluatedBatch { batch, geom_array };

        let expected = get_record_batch_memory_size(&evaluated_batch.batch)?
            + evaluated_batch
                .geom_array
                .evaluation_overhead_in_mem_size()?;

        assert_eq!(evaluated_batch.in_mem_size()?, expected);
        Ok(())
    }

    #[test]
    fn in_mem_size_counts_computed_geometry_array() -> Result<()> {
        let input_geom = geometry_array();
        let evaluated_geom = geometry_array();
        let batch = batch_with_geometry(input_geom)?;
        let geom_array = EvaluatedGeometryArray::try_new(evaluated_geom, &WKB_GEOMETRY)?;
        let evaluated_batch = EvaluatedBatch { batch, geom_array };

        let expected = get_record_batch_memory_size(&evaluated_batch.batch)?
            + get_array_memory_size(evaluated_batch.geom_array.geometry_array())?
            + evaluated_batch
                .geom_array
                .evaluation_overhead_in_mem_size()?;

        assert_eq!(evaluated_batch.in_mem_size()?, expected);
        Ok(())
    }
}
