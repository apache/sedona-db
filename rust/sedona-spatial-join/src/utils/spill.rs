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
    fmt,
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::config::SpillCompression;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::{
    SendableRecordBatchStream, SpillFile, SpillWriter, runtime_env::RuntimeEnv,
};
use datafusion_physical_plan::{SpillManager, metrics::SpillMetrics};
use futures::StreamExt;

use crate::utils::arrow_utils::{
    compact_batch, get_record_batch_memory_size, schema_contains_view_types,
};

/// Generic Arrow IPC stream spill writer for [`RecordBatch`].
///
/// Shared between multiple components so spill metrics are updated consistently.
pub(crate) struct RecordBatchSpillWriter {
    spill_file: Arc<dyn SpillFile>,
    spill_manager: SpillManager,
    writer: StreamWriter<CountingSpillWriter>,
    bytes_written: Arc<AtomicUsize>,
    metrics: SpillMetrics,
    batch_in_memory_size_threshold: Option<usize>,
    gc_view_arrays: bool,
}

impl RecordBatchSpillWriter {
    pub fn try_new(
        env: Arc<RuntimeEnv>,
        schema: SchemaRef,
        request_description: &str,
        compression: SpillCompression,
        metrics: SpillMetrics,
        batch_in_memory_size_threshold: Option<usize>,
    ) -> Result<Self> {
        let spill_file = env.disk_manager.create_tmp_file(request_description)?;
        let bytes_written = Arc::new(AtomicUsize::new(0));
        let spill_writer = CountingSpillWriter {
            inner: spill_file.open_writer()?,
            bytes_written: Arc::clone(&bytes_written),
        };

        let mut write_options = IpcWriteOptions::default();
        write_options = write_options.try_with_compression(compression.into())?;

        let writer =
            StreamWriter::try_new_with_options(spill_writer, schema.as_ref(), write_options)?;
        let spill_manager = SpillManager::new(env, metrics.clone(), Arc::clone(&schema))
            .with_compression_type(compression);
        metrics.spill_file_count.add(1);

        let gc_view_arrays = schema_contains_view_types(&schema);

        Ok(Self {
            spill_file,
            spill_manager,
            writer,
            bytes_written,
            metrics,
            batch_in_memory_size_threshold,
            gc_view_arrays,
        })
    }

    /// Write a record batch to the spill file.
    ///
    /// If `batch_size_threshold` is configured and the in-memory size of the batch exceeds the
    /// threshold, this will automatically split the batch into smaller slices and (optionally)
    /// compact each slice before writing.
    pub fn write_batch(&mut self, batch: RecordBatch) -> Result<()> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            // Preserve "empty batch" semantics: callers may rely on spilling and reading back a
            // zero-row batch (e.g. as a sentinel for an empty stream).
            return self.write_one_batch(batch);
        }

        let rows_per_split = self.calculate_rows_per_split(&batch, num_rows)?;
        if rows_per_split < num_rows {
            let mut offset = 0;
            while offset < num_rows {
                let length = std::cmp::min(rows_per_split, num_rows - offset);
                let slice = batch.slice(offset, length);
                self.write_one_batch(slice)?;
                offset += length;
            }
        } else {
            self.write_one_batch(batch)?;
        }
        Ok(())
    }

    fn calculate_rows_per_split(&self, batch: &RecordBatch, num_rows: usize) -> Result<usize> {
        let Some(threshold) = self.batch_in_memory_size_threshold else {
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

    fn write_one_batch(&mut self, batch: RecordBatch) -> Result<()> {
        // Writing record batches containing sparse binary view arrays may lead to excessive
        // disk usage and slow read performance later. Compact such batches before writing.
        let batch = if self.gc_view_arrays {
            compact_batch(batch)?
        } else {
            batch
        };
        self.writer.write(&batch).map_err(|e| {
            DataFusionError::Execution(format!(
                "Failed to write RecordBatch to spill file {:?}: {}",
                self.spill_file.path(),
                e
            ))
        })?;

        self.metrics.spilled_rows.add(batch.num_rows());
        Ok(())
    }

    pub fn finish(mut self) -> Result<SpillArtifact> {
        self.writer.finish()?;
        self.writer.get_mut().finish()?;
        self.metrics
            .spilled_bytes
            .add(self.bytes_written.load(Ordering::Relaxed));
        Ok(SpillArtifact {
            file: self.spill_file,
            manager: self.spill_manager,
        })
    }
}

struct CountingSpillWriter {
    inner: Box<dyn SpillWriter>,
    bytes_written: Arc<AtomicUsize>,
}

impl Write for CountingSpillWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.bytes_written.fetch_add(written, Ordering::Relaxed);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl SpillWriter for CountingSpillWriter {
    fn finish(&mut self) -> Result<()> {
        self.inner.finish()
    }
}

/// A finalized spill file together with the schema-aware manager needed to read it.
#[derive(Clone)]
pub struct SpillArtifact {
    file: Arc<dyn SpillFile>,
    manager: SpillManager,
}

impl SpillArtifact {
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(self.manager.schema())
    }

    pub fn size(&self) -> Option<u64> {
        self.file.size()
    }

    pub fn read_stream(&self) -> Result<SendableRecordBatchStream> {
        self.manager
            .read_spill_as_stream_unbuffered(Arc::clone(&self.file), None)
    }
}

impl fmt::Debug for SpillArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpillArtifact")
            .field("path", &self.file.path())
            .field("size", &self.file.size())
            .finish()
    }
}

/// Generic Arrow IPC stream spill reader for [`RecordBatch`].
pub(crate) struct RecordBatchSpillReader {
    stream_reader: SendableRecordBatchStream,
}

impl RecordBatchSpillReader {
    pub fn try_new(spill: &SpillArtifact) -> Result<Self> {
        Ok(Self {
            stream_reader: spill.read_stream()?,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.stream_reader.schema()
    }

    pub async fn next_batch(&mut self) -> Option<Result<RecordBatch>> {
        self.stream_reader.next().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::BinaryViewBuilder;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use datafusion_execution::{
        TempFileFactory, disk_manager::DiskManagerBuilder, runtime_env::RuntimeEnvBuilder,
    };
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
    use futures::stream;
    use std::{pin::Pin, sync::Mutex};

    #[derive(Default)]
    struct MemorySpillFile {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl SpillFile for MemorySpillFile {
        fn size(&self) -> Option<u64> {
            Some(self.bytes.lock().unwrap().len() as u64)
        }

        fn read_stream(
            &self,
        ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<Bytes>> + Send>>> {
            let bytes = Bytes::from(self.bytes.lock().unwrap().clone());
            Ok(Box::pin(stream::once(async move { Ok(bytes) })))
        }

        fn open_writer(&self) -> Result<Box<dyn SpillWriter>> {
            self.bytes.lock().unwrap().clear();
            Ok(Box::new(MemorySpillWriter {
                bytes: Arc::clone(&self.bytes),
            }))
        }
    }

    struct MemorySpillWriter {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for MemorySpillWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.bytes.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl SpillWriter for MemorySpillWriter {
        fn finish(&mut self) -> Result<()> {
            Ok(())
        }
    }

    struct MemoryTempFileFactory;

    impl TempFileFactory for MemoryTempFileFactory {
        fn create_temp_file(&self, _description: &str) -> Result<Arc<dyn SpillFile>> {
            Ok(Arc::new(MemorySpillFile::default()))
        }
    }

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

    fn create_test_binary_view_batch(num_rows: usize, value_len: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "payload",
            DataType::BinaryView,
            false,
        )]));

        let mut builder = BinaryViewBuilder::new();
        for i in 0..num_rows {
            let byte = b'a' + (i % 26) as u8;
            let bytes = vec![byte; value_len];
            builder.append_value(bytes.as_slice());
        }

        let array = Arc::new(builder.finish());
        RecordBatch::try_new(schema, vec![array]).unwrap()
    }

    #[tokio::test]
    async fn test_record_batch_spill_empty_batch_round_trip() -> Result<()> {
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
        writer.write_batch(empty)?;
        let file = writer.finish()?;

        assert_eq!(metrics.spill_file_count.value(), 1);
        assert_eq!(metrics.spilled_rows.value(), 0);

        let mut reader = RecordBatchSpillReader::try_new(&file)?;
        let read = reader.next_batch().await.unwrap()?;
        assert_eq!(read.num_rows(), 0);
        assert_eq!(read.schema(), schema);
        assert!(reader.next_batch().await.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_record_batch_spill_round_trip() -> Result<()> {
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
        writer.write_batch(batch1)?;
        writer.write_batch(batch2)?;

        let file = writer.finish()?;

        assert_eq!(metrics.spill_file_count.value(), 1);
        assert_eq!(metrics.spilled_rows.value(), 8);
        assert!(metrics.spilled_bytes.value() > 0);

        let mut reader = RecordBatchSpillReader::try_new(&file)?;
        assert_eq!(reader.schema(), schema);

        let read1 = reader.next_batch().await.unwrap()?;
        assert_eq!(read1.num_rows(), 5);
        let read2 = reader.next_batch().await.unwrap()?;
        assert_eq!(read2.num_rows(), 3);
        assert!(reader.next_batch().await.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_record_batch_spill_round_trip_without_local_path() -> Result<()> {
        let disk_manager =
            DiskManagerBuilder::default().with_temp_file_factory(Arc::new(MemoryTempFileFactory));
        let env = Arc::new(
            RuntimeEnvBuilder::new()
                .with_disk_manager_builder(disk_manager)
                .build()?,
        );
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);
        let schema = create_test_schema();
        let mut writer = RecordBatchSpillWriter::try_new(
            env,
            Arc::clone(&schema),
            "test_non_local_spill",
            SpillCompression::Uncompressed,
            metrics,
            None,
        )?;
        writer.write_batch(create_test_record_batch(4))?;
        let spill = writer.finish()?;

        assert!(spill.file.path().is_none());
        let mut reader = RecordBatchSpillReader::try_new(&spill)?;
        let batch = reader.next_batch().await.unwrap()?;
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.schema(), schema);
        assert!(reader.next_batch().await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_record_batch_spill_auto_splitting() -> Result<()> {
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
        writer.write_batch(batch)?;
        let file = writer.finish()?;

        // Rows should reflect the logical input rows, even if internally split.
        assert_eq!(metrics.spilled_rows.value(), 10);
        assert!(metrics.spilled_bytes.value() > 0);

        // Reader should be able to read all rows back across multiple batches.
        let mut reader = RecordBatchSpillReader::try_new(&file)?;
        let mut total_rows = 0;
        while let Some(batch) = reader.next_batch().await {
            total_rows += batch?.num_rows();
        }
        assert_eq!(total_rows, 10);

        Ok(())
    }

    #[tokio::test]
    async fn test_record_batch_spill_sliced_binary_view_not_excessive() -> Result<()> {
        let env = create_test_runtime_env()?;
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = SpillMetrics::new(&metrics_set, 0);

        // Use a long payload so the view-value buffers dominate overhead, making the
        // size comparison stable across platforms.
        const NUM_ROWS: usize = 100;
        const NUM_SLICES: usize = 10;
        const VALUE_LEN: usize = 8 * 1024;

        let batch = create_test_binary_view_batch(NUM_ROWS, VALUE_LEN);
        let batch_size = get_record_batch_memory_size(&batch)?;

        let mut writer = RecordBatchSpillWriter::try_new(
            env,
            batch.schema(),
            "test_record_batch_spill_sliced_binary_view",
            SpillCompression::Uncompressed,
            metrics.clone(),
            None,
        )?;

        let rows_per_slice = NUM_ROWS / NUM_SLICES;
        assert_eq!(rows_per_slice * NUM_SLICES, NUM_ROWS);
        for i in 0..NUM_SLICES {
            let slice = batch.slice(i * rows_per_slice, rows_per_slice);
            writer.write_batch(slice)?;
        }

        let file = writer.finish()?;
        let spill_size = file.size().unwrap_or_default() as usize;
        assert!(
            spill_size <= (batch_size as f64 * 1.2) as usize,
            "spill file unexpectedly large for sliced BinaryView batch: spill_size={spill_size}, batch_size={batch_size}"
        );

        Ok(())
    }
}
