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

//! Overhead of `EnsureLoadedExec` in the non-materialized case.
//!
//! The async UDF is an identity that stands in for `rs_ensureloaded` over
//! rasters that are already in memory, so the only work measured is the
//! operator's own: the per-row byte estimate, the slicing, and one async
//! invocation plus one output batch per slice. DataFusion's stock
//! `AsyncFuncExec` over the same plan is the baseline, and the budget is
//! swept so the 8192-row input batch is emitted as 1, 8, 64, 512 and 8192
//! slices.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::prelude::SessionConfig;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_execution::TaskContext;
use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::async_scalar_function::AsyncFuncExpr;
use datafusion_physical_expr::expressions::col;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::async_func::AsyncFuncExec;
use datafusion_physical_plan::common::collect;
use sedona_common::option::{RasterOptions, SedonaOptions};
use sedona_query_planner::raster_batch_budget::RasterBatchBudgetRule;
use sedona_raster::builder::{RasterBuilder, StartBandArgs};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::raster::BandDataType;

const ROWS: usize = 8192;
/// 16 × 16 UInt8: 256 bytes per raster, so a budget of `256 × rows_per_slice`
/// bytes yields slices of exactly that many rows.
const SIDE: i64 = 16;
const BYTES_PER_ROW: usize = (SIDE * SIDE) as usize;

/// Identity stand-in for `rs_ensureloaded` (same name, so the rule picks it
/// up), costing nothing beyond the argument array clone.
#[derive(Debug)]
struct IdentityEnsureLoaded {
    signature: Signature,
}

impl PartialEq for IdentityEnsureLoaded {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for IdentityEnsureLoaded {}
impl Hash for IdentityEnsureLoaded {
    fn hash<H: Hasher>(&self, state: &mut H) {
        "rs_ensureloaded".hash(state);
    }
}

impl ScalarUDFImpl for IdentityEnsureLoaded {
    fn name(&self) -> &str {
        "rs_ensureloaded"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[0].clone())
    }
    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        unreachable!("async only")
    }
}

#[async_trait]
impl AsyncScalarUDFImpl for IdentityEnsureLoaded {
    async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(ColumnarValue::Array(
            args.args[0].to_array(args.number_rows)?,
        ))
    }
}

/// One batch of `ROWS` in-db 16 × 16 UInt8 rasters.
fn raster_batch() -> (SchemaRef, RecordBatch) {
    let pixels = vec![7u8; BYTES_PER_ROW];
    let mut b = RasterBuilder::new(ROWS);
    for _ in 0..ROWS {
        b.start_raster_nd(
            &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            &["y", "x"],
            &[SIDE, SIDE],
            None,
        )
        .unwrap();
        b.start_band(StartBandArgs::new(
            &["y", "x"],
            &[SIDE, SIDE],
            BandDataType::UInt8,
        ))
        .unwrap();
        b.band_data_writer().append_value(&pixels);
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
    }
    let rasters: ArrayRef = Arc::new(b.finish().unwrap());
    let raster_field = SedonaType::Raster.to_storage_field("rast", true).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rast", rasters.data_type().clone(), true)
            .with_metadata(raster_field.metadata().clone()),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![rasters]).unwrap();
    (schema, batch)
}

/// `AsyncFuncExec(rs_ensureloaded(rast))` over the batch.
fn async_func_plan(schema: &SchemaRef, batch: &RecordBatch) -> Arc<dyn ExecutionPlan> {
    let input =
        MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(schema), None).unwrap();
    let udf = Arc::new(
        AsyncScalarUDF::new(Arc::new(IdentityEnsureLoaded {
            signature: Signature::any(1, Volatility::Stable),
        }))
        .into_scalar_udf(),
    );
    let func = Arc::new(
        ScalarFunctionExpr::try_new(
            udf,
            vec![col("rast", schema).unwrap()],
            schema,
            Arc::new(ConfigOptions::default()),
        )
        .unwrap(),
    );
    let expr = Arc::new(AsyncFuncExpr::try_new("__async_fn_0", func, schema).unwrap());
    Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap())
}

fn task_context(max_batch_bytes: usize) -> Arc<TaskContext> {
    let config = SessionConfig::new().with_option_extension(SedonaOptions {
        raster: RasterOptions { max_batch_bytes },
        ..Default::default()
    });
    SessionStateBuilder::new()
        .with_config(config)
        .build()
        .task_ctx()
}

fn bench(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let (schema, batch) = raster_batch();
    let stock = async_func_plan(&schema, &batch);
    let budgeted = RasterBatchBudgetRule
        .optimize(async_func_plan(&schema, &batch), &ConfigOptions::default())
        .unwrap();
    let ctx = task_context(0);

    let mut group = c.benchmark_group("ensure_loaded_exec_overhead");
    group.throughput(Throughput::Elements(ROWS as u64));

    group.bench_function("AsyncFuncExec (baseline)", |b| {
        b.iter(|| {
            let stream = stock.execute(0, Arc::clone(&ctx)).unwrap();
            rt.block_on(collect(stream)).unwrap()
        })
    });

    for rows_per_slice in [ROWS, 1024, 128, 16, 1] {
        let ctx = task_context(BYTES_PER_ROW * rows_per_slice);
        let slices = ROWS / rows_per_slice;
        group.bench_with_input(
            BenchmarkId::new("EnsureLoadedExec/slices", slices),
            &slices,
            |b, _| {
                b.iter(|| {
                    let stream = budgeted.execute(0, Arc::clone(&ctx)).unwrap();
                    rt.block_on(collect(stream)).unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
