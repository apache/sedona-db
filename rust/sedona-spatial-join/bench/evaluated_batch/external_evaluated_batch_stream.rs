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

use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use datafusion::config::SpillCompression;
use datafusion_common::ScalarValue;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_expr::ColumnarValue;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, SpillMetrics};
use futures::StreamExt;
use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
use sedona_spatial_join::evaluated_batch::evaluated_batch_stream::external::ExternalEvaluatedBatchStream;
use sedona_spatial_join::evaluated_batch::spill::EvaluatedBatchSpillWriter;
use sedona_spatial_join::evaluated_batch::EvaluatedBatch;
use sedona_spatial_join::operand_evaluator::EvaluatedGeometryArray;
use sedona_testing::create::create_array_storage;

const ROWS: usize = 8192;
const BATCHES_PER_FILE: usize = 64;

fn make_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn make_evaluated_batch(num_rows: usize, sedona_type: &SedonaType) -> EvaluatedBatch {
    let schema = make_schema();
    let ids: Vec<i32> = (0..num_rows).map(|v| v as i32).collect();
    let id_array = Arc::new(Int32Array::from(ids));
    let name_array = Arc::new(StringArray::from(vec![Some("Alice"); num_rows]));
    let batch = RecordBatch::try_new(schema, vec![id_array, name_array])
        .expect("failed to build record batch for benchmark");

    // Use sedona-testing helpers so this benchmark stays focused on spill + stream I/O.
    // This builds either a Binary (WKB_GEOMETRY) or BinaryView (WKB_VIEW_GEOMETRY) array.
    let wkt_values = vec![Some("POINT (0 0)"); num_rows];
    let geom_array = create_array_storage(&wkt_values, sedona_type);

    let mut geom_array = EvaluatedGeometryArray::try_new(geom_array, sedona_type)
        .expect("failed to build geometry array for benchmark");

    geom_array.distance = Some(ColumnarValue::Scalar(ScalarValue::Float64(Some(10.0))));

    EvaluatedBatch { batch, geom_array }
}

fn write_spill_file(
    env: Arc<RuntimeEnv>,
    schema: SchemaRef,
    metrics_set: &ExecutionPlanMetricsSet,
    sedona_type: &SedonaType,
    compression: SpillCompression,
    evaluated_batch: &EvaluatedBatch,
) -> Arc<datafusion_execution::disk_manager::RefCountedTempFile> {
    let metrics = SpillMetrics::new(metrics_set, 0);
    let mut writer = EvaluatedBatchSpillWriter::try_new(
        env,
        schema,
        sedona_type,
        "bench_external_stream",
        compression,
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

fn bench_external_evaluated_batch_stream(c: &mut Criterion) {
    let env = Arc::new(RuntimeEnv::default());
    let schema = make_schema();
    let metrics_set = ExecutionPlanMetricsSet::new();

    let compressions = [
        ("uncompressed", SpillCompression::Uncompressed),
        ("lz4", SpillCompression::Lz4Frame),
    ];

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("failed to create tokio runtime");

    for (label, sedona_type) in [("wkb", WKB_GEOMETRY), ("wkb_view", WKB_VIEW_GEOMETRY)] {
        let evaluated_batch = make_evaluated_batch(ROWS, &sedona_type);

        for (compression_label, compression) in compressions {
            let spill_file = write_spill_file(
                Arc::clone(&env),
                Arc::clone(&schema),
                &metrics_set,
                &sedona_type,
                compression,
                &evaluated_batch,
            );

            let mut group = c.benchmark_group(format!(
                "external_evaluated_batch_stream/{label}/{compression_label}"
            ));
            group.throughput(Throughput::Elements((ROWS * BATCHES_PER_FILE) as u64));

            group.bench_with_input(
                BenchmarkId::new(
                    "external_stream",
                    format!("rows_{ROWS}_batches_{BATCHES_PER_FILE}"),
                ),
                &spill_file,
                |b, file| {
                    b.iter(|| {
                        runtime.block_on(async {
                            let stream =
                                ExternalEvaluatedBatchStream::try_from_spill_file(Arc::clone(file))
                                    .expect("failed to create external evaluated batch stream");
                            futures::pin_mut!(stream);

                            let mut rows = 0usize;
                            while let Some(batch) = stream.next().await {
                                let batch =
                                    batch.expect("failed to read evaluated batch from stream");
                                rows += batch.num_rows();
                                black_box(batch);
                            }
                            black_box(rows);
                        })
                    })
                },
            );

            group.finish();
        }
    }
}

criterion_group!(
    external_evaluated_batch_stream,
    bench_external_evaluated_batch_stream
);
criterion_main!(external_evaluated_batch_stream);
