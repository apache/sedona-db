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

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{BinaryArray, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion::config::SpillCompression;
use datafusion_common::ScalarValue;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_expr::ColumnarValue;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
use sedona_schema::datatypes::WKB_GEOMETRY;
use sedona_spatial_join::evaluated_batch::spill::{
    EvaluatedBatchSpillReader, EvaluatedBatchSpillWriter,
};
use sedona_spatial_join::evaluated_batch::EvaluatedBatch;
use sedona_spatial_join::operand_evaluator::EvaluatedGeometryArray;

const ROWS: usize = 1024;
const BATCHES_PER_FILE: usize = 64;

fn make_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn make_evaluated_batch(num_rows: usize) -> EvaluatedBatch {
    let schema = make_schema();
    let ids: Vec<i32> = (0..num_rows).map(|v| v as i32).collect();
    let id_array = Arc::new(Int32Array::from(ids));
    let name_array = Arc::new(StringArray::from(vec![Some("Alice"); num_rows]));
    let batch = RecordBatch::try_new(schema, vec![id_array, name_array])
        .expect("failed to build record batch for benchmark");

    let point_wkb: &[u8] = &[
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64,
    ];
    let geom_array = Arc::new(BinaryArray::from_iter_values(std::iter::repeat_n(
        point_wkb, num_rows,
    )));

    let mut geom_array = EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY)
        .expect("failed to build geometry array for benchmark");

    // Use a scalar distance so the spilled dist column is constant.
    geom_array.distance = Some(ColumnarValue::Scalar(ScalarValue::Float64(Some(10.0))));

    EvaluatedBatch { batch, geom_array }
}

fn write_spill_file(
    env: Arc<RuntimeEnv>,
    schema: SchemaRef,
    metrics_set: &ExecutionPlanMetricsSet,
    evaluated_batch: &EvaluatedBatch,
) -> Arc<datafusion_execution::disk_manager::RefCountedTempFile> {
    let metrics = SpillMetrics::new(metrics_set, 0);
    let mut writer = EvaluatedBatchSpillWriter::try_new(
        env,
        schema,
        &WKB_GEOMETRY,
        "bench_spill",
        SpillCompression::Uncompressed,
        metrics,
        None,
    )
    .expect("failed to create spill writer for benchmark");

    for _ in 0..BATCHES_PER_FILE {
        writer
            .append(evaluated_batch)
            .expect("failed to append batch in benchmark");
    }

    Arc::new(writer.finish().expect("failed to finish spill writer"))
}

fn bench_spill_writer_and_reader(c: &mut Criterion) {
    let env = Arc::new(RuntimeEnv::default());
    let schema = make_schema();
    let evaluated_batch = make_evaluated_batch(ROWS);
    let metrics_set = ExecutionPlanMetricsSet::new();

    // Prepare a stable spill file for read benchmarks.
    let spill_file = write_spill_file(
        Arc::clone(&env),
        Arc::clone(&schema),
        &metrics_set,
        &evaluated_batch,
    );

    let mut group = c.benchmark_group("evaluated_batch_spill");
    group.throughput(Throughput::Elements((ROWS * BATCHES_PER_FILE) as u64));

    group.bench_with_input(
        BenchmarkId::new(
            "spill_writer",
            format!("rows_{ROWS}_batches_{BATCHES_PER_FILE}"),
        ),
        &evaluated_batch,
        |b, batch| {
            b.iter(|| {
                let metrics = SpillMetrics::new(&metrics_set, 0);
                let mut writer = EvaluatedBatchSpillWriter::try_new(
                    Arc::clone(&env),
                    Arc::clone(&schema),
                    &WKB_GEOMETRY,
                    "bench_spill",
                    SpillCompression::Uncompressed,
                    metrics,
                    None,
                )
                .expect("failed to create spill writer");

                for _ in 0..BATCHES_PER_FILE {
                    writer.append(black_box(batch)).unwrap();
                }

                let file = writer.finish().unwrap();
                black_box(file);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "spill_reader",
            format!("rows_{ROWS}_batches_{BATCHES_PER_FILE}"),
        ),
        &spill_file,
        |b, file| {
            b.iter(|| {
                let mut reader = EvaluatedBatchSpillReader::try_new(black_box(file.as_ref()))
                    .expect("failed to create spill reader");
                let mut rows = 0usize;

                while let Some(batch) = reader.next_batch() {
                    let batch = batch.expect("failed to read evaluated batch");
                    rows += batch.num_rows();
                    black_box(batch);
                }

                black_box(rows);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new(
            "spill_reader_raw",
            format!("rows_{ROWS}_batches_{BATCHES_PER_FILE}"),
        ),
        &spill_file,
        |b, file| {
            b.iter(|| {
                let mut reader = EvaluatedBatchSpillReader::try_new(black_box(file.as_ref()))
                    .expect("failed to create spill reader");
                let mut rows = 0usize;

                while let Some(batch) = reader.next_raw_batch() {
                    let batch = batch.expect("failed to read record batch");
                    rows += batch.num_rows();
                    black_box(batch);
                }

                black_box(rows);
            })
        },
    );

    // Measure the per-call overhead of `next_batch()` (including decoding) for a single batch.
    group.throughput(Throughput::Elements(ROWS as u64));
    group.bench_with_input(
        BenchmarkId::new("spill_reader_next_batch_single", format!("rows_{ROWS}")),
        &spill_file,
        |b, file| {
            b.iter(|| {
                let mut reader = EvaluatedBatchSpillReader::try_new(black_box(file.as_ref()))
                    .expect("failed to create spill reader");

                let batch = reader
                    .next_batch()
                    .expect("expected at least one batch")
                    .expect("failed to read evaluated batch");
                black_box(batch);
            })
        },
    );

    group.finish();
}

criterion_group!(evaluated_batch_spill, bench_spill_writer_and_reader);
criterion_main!(evaluated_batch_spill);
