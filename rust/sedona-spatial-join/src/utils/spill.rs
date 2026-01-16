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

use std::{fs::File, io::BufReader, sync::Arc};

use arrow::ipc::{
    reader::StreamReader,
    writer::{IpcWriteOptions, StreamWriter},
};
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::config::SpillCompression;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{disk_manager::RefCountedTempFile, runtime_env::RuntimeEnv};
use datafusion_physical_plan::metrics::SpillMetrics;

use crate::utils::arrow_utils::{compact_batch, get_record_batch_memory_size};

/// Generic Arrow IPC stream spill writer for [`RecordBatch`].
///
/// Shared between multiple components so spill metrics are updated consistently.
pub(crate) struct RecordBatchSpillWriter {
    in_progress_file: RefCountedTempFile,
    writer: StreamWriter<File>,
    metrics: SpillMetrics,
    batch_size_threshold: Option<usize>,
}

impl RecordBatchSpillWriter {
    pub fn try_new(
        env: Arc<RuntimeEnv>,
        schema: SchemaRef,
        request_description: &str,
        compression: SpillCompression,
        metrics: SpillMetrics,
        batch_size_threshold: Option<usize>,
    ) -> Result<Self> {
        let in_progress_file = env.disk_manager.create_tmp_file(request_description)?;
        let file = File::create(in_progress_file.path())?;

        let mut write_options = IpcWriteOptions::default();
        write_options = write_options.try_with_compression(compression.into())?;

        let writer = StreamWriter::try_new_with_options(file, schema.as_ref(), write_options)?;
        metrics.spill_file_count.add(1);

        Ok(Self {
            in_progress_file,
            writer,
            metrics,
            batch_size_threshold,
        })
    }

    /// Write a record batch to the spill file.
    ///
    /// If `batch_size_threshold` is configured and the in-memory size of the batch exceeds the
    /// threshold, this will automatically split the batch into smaller slices and (optionally)
    /// compact each slice before writing.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            // Preserve "empty batch" semantics: callers may rely on spilling and reading back a
            // zero-row batch (e.g. as a sentinel for an empty stream).
            return self.write_one_batch(batch);
        }

        let rows_per_split = self.calculate_rows_per_split(batch, num_rows)?;
        if rows_per_split < num_rows {
            let mut offset = 0;
            while offset < num_rows {
                let length = std::cmp::min(rows_per_split, num_rows - offset);
                let slice = batch.slice(offset, length);
                let compacted = compact_batch(slice)?;
                self.write_one_batch(&compacted)?;
                offset += length;
            }
        } else {
            self.write_one_batch(batch)?;
        }
        Ok(())
    }

    fn calculate_rows_per_split(&self, batch: &RecordBatch, num_rows: usize) -> Result<usize> {
        let Some(threshold) = self.batch_size_threshold else {
            return Ok(num_rows);
        };
        if threshold == 0 {
            return Ok(num_rows);
        }

        let batch_size = get_record_batch_memory_size(batch)?;
        if batch_size <= threshold {
            return Ok(num_rows);
        }

        let num_splits = batch_size.div_ceil(threshold);
        let rows = num_rows.div_ceil(num_splits);
        Ok(std::cmp::max(1, rows))
    }

    fn write_one_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer.write(batch).map_err(|e| {
            DataFusionError::Execution(format!(
                "Failed to write RecordBatch to spill file {:?}: {}",
                self.in_progress_file.path(),
                e
            ))
        })?;

        self.metrics.spilled_rows.add(batch.num_rows());
        self.metrics
            .spilled_bytes
            .add(get_record_batch_memory_size(batch)?);

        Ok(())
    }

    pub fn finish(mut self) -> Result<RefCountedTempFile> {
        self.writer.finish()?;

        let mut in_progress_file = self.in_progress_file;
        in_progress_file.update_disk_usage()?;
        let size = in_progress_file.current_disk_usage();
        self.metrics.spilled_bytes.add(size as usize);
        Ok(in_progress_file)
    }
}

/// Generic Arrow IPC stream spill reader for [`RecordBatch`].
pub(crate) struct RecordBatchSpillReader {
    stream_reader: StreamReader<BufReader<File>>,
}

impl RecordBatchSpillReader {
    pub fn try_new(temp_file: &RefCountedTempFile) -> Result<Self> {
        let file = File::open(temp_file.path())?;
        let mut stream_reader = StreamReader::try_new_buffered(file, None)?;

        // SAFETY: spill writers in this crate strictly follow Arrow IPC specifications.
        // Skip redundant validation during read to speed up.
        unsafe {
            stream_reader = stream_reader.with_skip_validation(true);
        }

        Ok(Self { stream_reader })
    }

    pub fn schema(&self) -> SchemaRef {
        self.stream_reader.schema()
    }

    pub fn next_batch(&mut self) -> Option<Result<RecordBatch>> {
        self.stream_reader
            .next()
            .map(|result| result.map_err(|e| e.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;

    fn create_test_runtime_env() -> Result<Arc<RuntimeEnv>> {
        Ok(Arc::new(RuntimeEnv::default()))
    }

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_record_batch(num_rows: usize) -> RecordBatch {
        let ids: Int32Array = (0..num_rows as i32).collect();

        let names: StringArray = (0..num_rows)
            .map(|i| {
                if i % 3 == 0 {
                    None
                } else {
                    Some(format!("name_{i}"))
                }
            })
            .collect();

        RecordBatch::try_new(create_test_schema(), vec![Arc::new(ids), Arc::new(names)]).unwrap()
    }

    #[test]
    fn test_record_batch_spill_empty_batch_round_trip() -> Result<()> {
        let env = create_test_runtime_env()?;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let schema = create_test_schema();
        let mut writer = RecordBatchSpillWriter::try_new(
            env,
            schema.clone(),
            "test_record_batch_spill_empty",
            SpillCompression::Uncompressed,
            metrics.clone(),
            None,
        )?;

        let empty = create_test_record_batch(0);
        writer.write_batch(&empty)?;
        let file = writer.finish()?;

        assert_eq!(metrics.spill_file_count.value(), 1);
        assert_eq!(metrics.spilled_rows.value(), 0);

        let mut reader = RecordBatchSpillReader::try_new(&file)?;
        let read = reader.next_batch().unwrap()?;
        assert_eq!(read.num_rows(), 0);
        assert_eq!(read.schema(), schema);
        assert!(reader.next_batch().is_none());

        Ok(())
    }

    #[test]
    fn test_record_batch_spill_round_trip() -> Result<()> {
        let env = create_test_runtime_env()?;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let schema = create_test_schema();
        let mut writer = RecordBatchSpillWriter::try_new(
            env,
            schema.clone(),
            "test_record_batch_spill",
            SpillCompression::Uncompressed,
            metrics.clone(),
            None,
        )?;

        let batch1 = create_test_record_batch(5);
        let batch2 = create_test_record_batch(3);
        writer.write_batch(&batch1)?;
        writer.write_batch(&batch2)?;

        let file = writer.finish()?;

        assert_eq!(metrics.spill_file_count.value(), 1);
        assert_eq!(metrics.spilled_rows.value(), 8);
        assert!(metrics.spilled_bytes.value() > 0);

        let mut reader = RecordBatchSpillReader::try_new(&file)?;
        assert_eq!(reader.schema(), schema);

        let read1 = reader.next_batch().unwrap()?;
        assert_eq!(read1.num_rows(), 5);
        let read2 = reader.next_batch().unwrap()?;
        assert_eq!(read2.num_rows(), 3);
        assert!(reader.next_batch().is_none());

        Ok(())
    }

    #[test]
    fn test_record_batch_spill_auto_splitting() -> Result<()> {
        let env = create_test_runtime_env()?;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        let schema = create_test_schema();
        // Force splitting by setting a tiny threshold.
        let mut writer = RecordBatchSpillWriter::try_new(
            env,
            schema.clone(),
            "test_record_batch_spill_split",
            SpillCompression::Uncompressed,
            metrics.clone(),
            Some(1),
        )?;

        let batch = create_test_record_batch(10);
        writer.write_batch(&batch)?;
        let file = writer.finish()?;

        // Rows should reflect the logical input rows, even if internally split.
        assert_eq!(metrics.spilled_rows.value(), 10);
        assert!(metrics.spilled_bytes.value() > 0);

        // Reader should be able to read all rows back across multiple batches.
        let mut reader = RecordBatchSpillReader::try_new(&file)?;
        let mut total_rows = 0;
        while let Some(batch) = reader.next_batch() {
            total_rows += batch?.num_rows();
        }
        assert_eq!(total_rows, 10);

        Ok(())
    }
}
