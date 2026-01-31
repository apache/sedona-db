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

use std::sync::Arc;

use arrow::array::Float64Array;
use arrow_array::{Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion::config::SpillCompression;
use datafusion_common::{Result, ScalarValue};
use datafusion_execution::{disk_manager::RefCountedTempFile, runtime_env::RuntimeEnv};
use datafusion_expr::ColumnarValue;
use datafusion_physical_plan::metrics::SpillMetrics;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use sedona_schema::datatypes::SedonaType;

use crate::{
    evaluated_batch::EvaluatedBatch,
    operand_evaluator::EvaluatedGeometryArray,
    utils::spill::{RecordBatchSpillReader, RecordBatchSpillWriter},
};

/// Writer for spilling evaluated batches to disk
pub struct EvaluatedBatchSpillWriter {
    /// The temporary spill file being written to
    inner: RecordBatchSpillWriter,

    /// Schema of the spilled record batches. It is augmented from the schema of original record batches
    /// The spill_schema has 4 fields:
    /// * `data`: StructArray containing the original record batch columns
    /// * `geom`: geometry array in storage format
    /// * `dist`: distance field
    spill_schema: Schema,
    /// Inner fields of the "data" StructArray in the spilled record batches
    data_inner_fields: Fields,
}

const SPILL_FIELD_DATA_INDEX: usize = 0;
const SPILL_FIELD_GEOM_INDEX: usize = 1;
const SPILL_FIELD_DIST_INDEX: usize = 2;

impl EvaluatedBatchSpillWriter {
    /// Create a new SpillWriter
    pub fn try_new(
        env: Arc<RuntimeEnv>,
        schema: SchemaRef,
        sedona_type: &SedonaType,
        request_description: &str,
        compression: SpillCompression,
        metrics: SpillMetrics,
        batch_size_threshold: Option<usize>,
    ) -> Result<Self> {
        // Construct schema of record batches to be written. The written batches are augmented from the original record batches.
        let data_inner_fields = schema.fields().clone();
        let data_struct_field =
            Field::new("data", DataType::Struct(data_inner_fields.clone()), false);
        let geom_field = sedona_type.to_storage_field("geom", true)?;
        let dist_field = Field::new("dist", DataType::Float64, true);
        let spill_schema = Schema::new(vec![data_struct_field, geom_field, dist_field]);

        // Create spill file
        let inner = RecordBatchSpillWriter::try_new(
            env,
            Arc::new(spill_schema.clone()),
            request_description,
            compression,
            metrics,
            batch_size_threshold,
        )?;

        Ok(Self {
            inner,
            spill_schema,
            data_inner_fields,
        })
    }

    /// Append an EvaluatedBatch to the spill file
    pub fn append(&mut self, evaluated_batch: &EvaluatedBatch) -> Result<()> {
        let record_batch = self.spilled_record_batch(evaluated_batch)?;

        // Splitting/compaction and spill bytes/rows metrics are handled by `RecordBatchSpillWriter`.
        self.inner.write_batch(record_batch)?;
        Ok(())
    }

    /// Finish writing and return the temporary file
    pub fn finish(self) -> Result<RefCountedTempFile> {
        self.inner.finish()
    }

    fn spilled_record_batch(&self, evaluated_batch: &EvaluatedBatch) -> Result<RecordBatch> {
        let num_rows = evaluated_batch.num_rows();

        // Store the original data batch into a StructArray
        let data_batch = &evaluated_batch.batch;
        let data_arrays = data_batch.columns().to_vec();
        let data_struct_array =
            StructArray::try_new(self.data_inner_fields.clone(), data_arrays, None)?;

        // Store dist into a Float64Array
        let mut dist_builder = arrow::array::Float64Builder::with_capacity(num_rows);
        let geom_array = &evaluated_batch.geom_array;
        match &geom_array.distance {
            Some(ColumnarValue::Scalar(scalar)) => match scalar {
                ScalarValue::Float64(dist_value) => {
                    for _ in 0..num_rows {
                        dist_builder.append_option(*dist_value);
                    }
                }
                _ => {
                    return sedona_internal_err!("Distance columnar value is not a Float64Array");
                }
            },
            Some(ColumnarValue::Array(array)) => {
                let float_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                dist_builder.append_array(float_array);
            }
            None => {
                for _ in 0..num_rows {
                    dist_builder.append_null();
                }
            }
        }
        let dist_array = dist_builder.finish();

        // Assemble the final spilled RecordBatch
        let columns = vec![
            Arc::new(data_struct_array) as Arc<dyn arrow::array::Array>,
            Arc::clone(&geom_array.geometry_array),
            Arc::new(dist_array) as Arc<dyn arrow::array::Array>,
        ];
        let spilled_record_batch =
            RecordBatch::try_new(Arc::new(self.spill_schema.clone()), columns)?;
        Ok(spilled_record_batch)
    }
}
/// Reader for reading spilled evaluated batches from disk
pub struct EvaluatedBatchSpillReader {
    inner: RecordBatchSpillReader,
}
impl EvaluatedBatchSpillReader {
    /// Create a new SpillReader
    pub fn try_new(temp_file: &RefCountedTempFile) -> Result<Self> {
        Ok(Self {
            inner: RecordBatchSpillReader::try_new(temp_file)?,
        })
    }

    /// Get the schema of the spilled data
    pub fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    /// Read the next EvaluatedBatch from the spill file
    #[allow(unused)]
    pub fn next_batch(&mut self) -> Option<Result<EvaluatedBatch>> {
        self.next_raw_batch()
            .map(|record_batch| record_batch.and_then(spilled_batch_to_evaluated_batch))
    }

    /// Read the next raw RecordBatch from the spill file
    pub fn next_raw_batch(&mut self) -> Option<Result<RecordBatch>> {
        self.inner.next_batch()
    }
}

pub(crate) fn spilled_batch_to_evaluated_batch(
    record_batch: RecordBatch,
) -> Result<EvaluatedBatch> {
    // Extract the data struct array (column 0) and convert back to the original RecordBatch
    let data_array = record_batch
        .column(SPILL_FIELD_DATA_INDEX)
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            sedona_internal_datafusion_err!("Expected data column to be a StructArray")
        })?;

    let data_schema = Arc::new(Schema::new(match data_array.data_type() {
        DataType::Struct(fields) => fields.clone(),
        _ => return sedona_internal_err!("Expected data column to have Struct data type"),
    }));

    let data_columns = (0..data_array.num_columns())
        .map(|i| Arc::clone(data_array.column(i)))
        .collect::<Vec<_>>();

    let batch = RecordBatch::try_new(data_schema, data_columns)?;

    // Extract the geometry array (column 1)
    let geom_array = Arc::clone(record_batch.column(SPILL_FIELD_GEOM_INDEX));

    // Determine the SedonaType from the geometry field in the record batch schema
    let schema = record_batch.schema();
    let geom_field = schema.field(SPILL_FIELD_GEOM_INDEX);
    let sedona_type = SedonaType::from_storage_field(geom_field)?;

    // Extract the distance array (column 3) and convert back to ColumnarValue
    let dist_array = record_batch
        .column(SPILL_FIELD_DIST_INDEX)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            sedona_internal_datafusion_err!("Expected dist column to be Float64Array")
        })?;

    let distance = if !dist_array.is_empty() {
        // Check if all values are the same (scalar case)
        let first_value = if dist_array.is_null(0) {
            None
        } else {
            Some(dist_array.value(0))
        };

        let all_same = (1..dist_array.len()).all(|i| {
            let current_value = if dist_array.is_null(i) {
                None
            } else {
                Some(dist_array.value(i))
            };
            current_value == first_value
        });

        if all_same {
            Some(ColumnarValue::Scalar(ScalarValue::Float64(first_value)))
        } else {
            Some(ColumnarValue::Array(Arc::clone(
                record_batch.column(SPILL_FIELD_DIST_INDEX),
            )))
        }
    } else {
        None
    };

    // Create EvaluatedGeometryArray
    let mut geom_array = EvaluatedGeometryArray::try_new(geom_array, &sedona_type)?;
    geom_array.distance = distance;

    Ok(EvaluatedBatch { batch, geom_array })
}

pub(crate) fn spilled_schema_to_evaluated_schema(spilled_schema: &SchemaRef) -> Result<SchemaRef> {
    if spilled_schema.fields().is_empty() {
        return Ok(SchemaRef::new(Schema::empty()));
    }

    let data_field = spilled_schema.field(SPILL_FIELD_DATA_INDEX);
    let inner_fields = match data_field.data_type() {
        DataType::Struct(fields) => fields.clone(),
        _ => {
            return sedona_internal_err!("Invalid schema of spilled file: {:?}", spilled_schema);
        }
    };
    Ok(SchemaRef::new(Schema::new(inner_fields)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::arrow_utils::get_record_batch_memory_size;
    use arrow_array::{ArrayRef, BinaryArray, Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::{DataFusionError, Result};
    use datafusion_execution::runtime_env::RuntimeEnv;
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use std::sync::Arc;

    fn create_test_runtime_env() -> Result<Arc<RuntimeEnv>> {
        Ok(Arc::new(RuntimeEnv::default()))
    }

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_record_batch() -> Result<RecordBatch> {
        let schema = create_test_schema();
        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let name_array = Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob"), None]));
        RecordBatch::try_new(schema, vec![id_array, name_array]).map_err(|e| e.into())
    }

    fn create_test_geometry_array() -> Result<(ArrayRef, SedonaType)> {
        // Create WKB encoded points (simple binary data for testing)
        // WKB for POINT (1 2): 01 01000000 0000000000000000F03F 0000000000000040
        let point1_wkb: Vec<u8> = vec![
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64,
        ];
        let point2_wkb: Vec<u8> = vec![
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 0, 16, 64,
        ];
        let point3_wkb: Vec<u8> = vec![
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 64, 0, 0, 0, 0, 0, 0, 24, 64,
        ];

        let sedona_type = WKB_GEOMETRY;
        let geom_array: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(point1_wkb.as_slice()),
            Some(point2_wkb.as_slice()),
            Some(point3_wkb.as_slice()),
        ]));

        Ok((geom_array, sedona_type))
    }

    fn create_test_evaluated_batch() -> Result<EvaluatedBatch> {
        let batch = create_test_record_batch()?;
        let (geom_array, sedona_type) = create_test_geometry_array()?;
        let mut geom_array = EvaluatedGeometryArray::try_new(geom_array, &sedona_type)?;

        // Add distance as a scalar value
        geom_array.distance = Some(ColumnarValue::Scalar(ScalarValue::Float64(Some(10.0))));

        Ok(EvaluatedBatch { batch, geom_array })
    }

    fn create_test_evaluated_batch_with_array_distance() -> Result<EvaluatedBatch> {
        let batch = create_test_record_batch()?;
        let (geom_array, sedona_type) = create_test_geometry_array()?;
        let mut geom_array = EvaluatedGeometryArray::try_new(geom_array, &sedona_type)?;

        // Add distance as an array value
        let dist_array = Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0)]));
        geom_array.distance = Some(ColumnarValue::Array(dist_array));

        Ok(EvaluatedBatch { batch, geom_array })
    }

    fn create_test_evaluated_batch_with_nulls() -> Result<EvaluatedBatch> {
        let schema = create_test_schema();
        let id_array = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let name_array = Arc::new(StringArray::from(vec![
            Some("Alice"),
            None,
            Some("Charlie"),
        ]));
        let batch = RecordBatch::try_new(schema, vec![id_array, name_array])
            .map_err(DataFusionError::from)?;

        // Create geometry array with null in the middle
        let point1_wkb: Vec<u8> = vec![
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64,
        ];
        let point3_wkb: Vec<u8> = vec![
            1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 64, 0, 0, 0, 0, 0, 0, 24, 64,
        ];

        let sedona_type = WKB_GEOMETRY;
        let geom_array: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(point1_wkb.as_slice()),
            None,
            Some(point3_wkb.as_slice()),
        ]));

        let mut geom_array = EvaluatedGeometryArray::try_new(geom_array, &sedona_type)?;

        // Add distance with nulls
        let dist_array = Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]));
        geom_array.distance = Some(ColumnarValue::Array(dist_array));

        Ok(EvaluatedBatch { batch, geom_array })
    }

    #[test]
    fn test_spill_writer_creation() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;

        // Verify the spill schema has the expected structure
        assert_eq!(writer.spill_schema.fields().len(), 3);
        assert_eq!(
            writer.spill_schema.field(SPILL_FIELD_DATA_INDEX).name(),
            "data"
        );
        assert_eq!(
            writer.spill_schema.field(SPILL_FIELD_GEOM_INDEX).name(),
            "geom"
        );
        assert_eq!(
            writer.spill_schema.field(SPILL_FIELD_DIST_INDEX).name(),
            "dist"
        );

        Ok(())
    }

    #[test]
    fn test_spill_write_and_read_basic() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let mut writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;

        let evaluated_batch = create_test_evaluated_batch()?;
        let original_num_rows = evaluated_batch.num_rows();

        writer.append(&evaluated_batch)?;
        let temp_file = writer.finish()?;

        // Read back the spilled data
        let mut reader = EvaluatedBatchSpillReader::try_new(&temp_file)?;
        let read_batch_result = reader.next_batch();

        assert!(read_batch_result.is_some());
        let read_batch = read_batch_result.unwrap()?;

        // Verify the data
        assert_eq!(read_batch.num_rows(), original_num_rows);
        assert_eq!(read_batch.batch.num_columns(), 2); // id and name columns

        // Verify that there are no more batches
        assert!(reader.next_batch().is_none());

        Ok(())
    }

    #[test]
    fn test_spill_write_and_read_with_array_distance() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let mut writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;

        let evaluated_batch = create_test_evaluated_batch_with_array_distance()?;
        writer.append(&evaluated_batch)?;
        let temp_file = writer.finish()?;

        // Read back the spilled data
        let mut reader = EvaluatedBatchSpillReader::try_new(&temp_file)?;
        let read_batch = reader.next_batch().unwrap()?;

        // Verify distance is read back as array
        match &read_batch.geom_array.distance {
            Some(ColumnarValue::Array(array)) => {
                let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                assert_eq!(float_array.len(), 3);
                assert_eq!(float_array.value(0), 1.0);
                assert_eq!(float_array.value(1), 2.0);
                assert_eq!(float_array.value(2), 3.0);
            }
            _ => panic!("Expected distance to be an array"),
        }

        Ok(())
    }

    #[test]
    fn test_spill_write_and_read_with_nulls() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let mut writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;

        let evaluated_batch = create_test_evaluated_batch_with_nulls()?;
        writer.append(&evaluated_batch)?;
        let temp_file = writer.finish()?;

        // Read back the spilled data
        let mut reader = EvaluatedBatchSpillReader::try_new(&temp_file)?;
        let read_batch = reader.next_batch().unwrap()?;

        // Verify nulls are preserved
        assert_eq!(read_batch.num_rows(), 3);
        assert!(read_batch.geom_array.rects[1].is_none()); // Null geometry

        // Verify distance nulls
        match &read_batch.geom_array.distance {
            Some(ColumnarValue::Array(array)) => {
                let float_array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                assert!(float_array.is_valid(0));
                assert!(float_array.is_null(1));
                assert!(float_array.is_valid(2));
            }
            _ => panic!("Expected distance to be an array"),
        }

        Ok(())
    }

    #[test]
    fn test_spill_multiple_batches() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let mut writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;

        // Write multiple batches
        let batch1 = create_test_evaluated_batch()?;
        let batch2 = create_test_evaluated_batch_with_array_distance()?;
        let batch3 = create_test_evaluated_batch_with_nulls()?;

        writer.append(&batch1)?;
        writer.append(&batch2)?;
        writer.append(&batch3)?;
        let temp_file = writer.finish()?;

        // Read back all batches
        let mut reader = EvaluatedBatchSpillReader::try_new(&temp_file)?;

        let read_batch1 = reader.next_batch().unwrap()?;
        assert_eq!(read_batch1.num_rows(), 3);

        let read_batch2 = reader.next_batch().unwrap()?;
        assert_eq!(read_batch2.num_rows(), 3);

        let read_batch3 = reader.next_batch().unwrap()?;
        assert_eq!(read_batch3.num_rows(), 3);

        // Verify no more batches
        assert!(reader.next_batch().is_none());

        Ok(())
    }

    #[test]
    fn test_spill_metrics_updated() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let mut writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics.clone(),
            None,
        )?;

        let evaluated_batch = create_test_evaluated_batch()?;
        writer.append(&evaluated_batch)?;
        writer.finish()?;

        // Verify spill metrics were updated
        assert!(metrics.spilled_rows.value() > 0);
        assert!(metrics.spilled_bytes.value() > 0);

        // Verify spill file count was updated
        assert_eq!(metrics.spill_file_count.value(), 1);

        Ok(())
    }

    #[test]
    fn test_spill_rect_preservation() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let mut writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;

        let evaluated_batch = create_test_evaluated_batch()?;
        let original_rects = evaluated_batch.rects().clone();

        writer.append(&evaluated_batch)?;
        let temp_file = writer.finish()?;

        // Read back and verify rects
        let mut reader = EvaluatedBatchSpillReader::try_new(&temp_file)?;
        let read_batch = reader.next_batch().unwrap()?;

        assert_eq!(read_batch.rects().len(), original_rects.len());
        for (original, read) in original_rects.iter().zip(read_batch.rects().iter()) {
            match (original, read) {
                (Some(orig_rect), Some(read_rect)) => {
                    assert_eq!(orig_rect.min().x, read_rect.min().x);
                    assert_eq!(orig_rect.min().y, read_rect.min().y);
                    assert_eq!(orig_rect.max().x, read_rect.max().x);
                    assert_eq!(orig_rect.max().y, read_rect.max().y);
                }
                (None, None) => {}
                _ => panic!("Rect mismatch between original and read"),
            }
        }

        Ok(())
    }

    #[test]
    fn test_spill_scalar_distance_preserved() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let mut writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;

        let evaluated_batch = create_test_evaluated_batch()?;
        writer.append(&evaluated_batch)?;
        let temp_file = writer.finish()?;

        // Read back and verify scalar distance is preserved
        let mut reader = EvaluatedBatchSpillReader::try_new(&temp_file)?;
        let read_batch = reader.next_batch().unwrap()?;

        match &read_batch.geom_array.distance {
            Some(ColumnarValue::Scalar(ScalarValue::Float64(Some(val)))) => {
                assert_eq!(*val, 10.0);
            }
            _ => panic!("Expected scalar distance value"),
        }

        Ok(())
    }

    #[test]
    fn test_spill_empty_batch() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let mut writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema.clone(),
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;

        // Create an empty batch
        let id_array = Arc::new(Int32Array::from(Vec::<i32>::new()));
        let name_array = Arc::new(StringArray::from(Vec::<Option<&str>>::new()));
        let empty_batch = RecordBatch::try_new(schema, vec![id_array, name_array])
            .map_err(DataFusionError::from)?;

        let geom_array: ArrayRef = Arc::new(BinaryArray::from(Vec::<Option<&[u8]>>::new()));
        let geom_array = EvaluatedGeometryArray::try_new(geom_array, &sedona_type)?;

        let evaluated_batch = EvaluatedBatch {
            batch: empty_batch,
            geom_array,
        };

        writer.append(&evaluated_batch)?;
        let temp_file = writer.finish()?;

        // Read back and verify
        let mut reader = EvaluatedBatchSpillReader::try_new(&temp_file)?;
        let read_batch = reader.next_batch().unwrap()?;
        assert_eq!(read_batch.num_rows(), 0);

        Ok(())
    }

    #[test]
    fn test_spill_batch_splitting() -> Result<()> {
        let env = create_test_runtime_env()?;
        let schema = create_test_schema();
        let sedona_type = WKB_GEOMETRY;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        // Create a batch
        let evaluated_batch = create_test_evaluated_batch()?;
        let batch_size = get_record_batch_memory_size(&evaluated_batch.batch)?;

        // Set threshold to be smaller than batch size, so it splits into at least 2 parts
        let threshold = batch_size / 2;

        let mut writer = EvaluatedBatchSpillWriter::try_new(
            env,
            schema,
            &sedona_type,
            "test_spill",
            SpillCompression::Uncompressed,
            metrics,
            Some(threshold),
        )?;

        writer.append(&evaluated_batch)?;
        let temp_file = writer.finish()?;

        // Read back the spilled data
        let mut reader = EvaluatedBatchSpillReader::try_new(&temp_file)?;

        // We expect multiple batches
        let mut num_batches = 0;
        let mut total_rows = 0;
        while let Some(batch_result) = reader.next_batch() {
            let batch = batch_result?;
            num_batches += 1;
            total_rows += batch.num_rows();
        }

        assert!(
            num_batches > 1,
            "Batch should have been split into multiple batches"
        );
        assert_eq!(
            total_rows,
            evaluated_batch.num_rows(),
            "Total rows should match"
        );

        Ok(())
    }
}
