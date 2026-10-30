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

//! Where rasters are loaded: the input of a loading node's
//! [`RasterBatchSplitExec`](crate::raster_materialization::RasterBatchSplitExec)
//! is hash-partitioned by raster source.
//!
//! The planner usually feeds that node through a round-robin repartition,
//! which deals whole batches, and a raster catalog is often tiny (one
//! Parquet row group, a couple of batches), so all the loading lands on a
//! couple of partitions while the rest idle. Hashing is per row, so even a
//! two-batch catalog spreads over every partition, and the rows of one
//! raster share a partition, so the chunk cache loads each raster once
//! instead of once per partition that happens to hold some of its rows.
//! [`raster_source_spread`] decides when the spread is safe and what it
//! replaces, [`RasterSourceKeyExpr`] is the key, and the
//! [`RasterMaterializationRule`](crate::raster_materialization::RasterMaterializationRule)
//! builds the repartition between the split and the input.

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use arrow_array::{Array, RecordBatch, StructArray, UInt64Array};
use arrow_schema::{DataType, Schema};
use datafusion_common::{Result, internal_err};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties, Partitioning};
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::outdb_source;

/// The spread of a loading node's `input` by raster source: the plan to
/// hash (`input` itself, or what the planner's own repartitioning below the
/// node reads, see [`spread_source`]) and the number of partitions, or
/// `None` when the spread is skipped.
///
/// Why hash rather than the round-robin the planner already put below the
/// node: round-robin deals whole batches, and a raster catalog is often one
/// or two batches, so only one or two partitions ever load anything. A hash
/// is computed per row, so it spreads any number of batches over every
/// partition, and it keeps all rows of one raster in one partition. That
/// locality is what keeps a load per raster: the chunk cache then sees each
/// raster's rows in order on one stream instead of several partitions
/// missing on the same raster at once.
///
/// The spread is skipped when repartitioning could only cost, or could
/// break a requirement of a parent. The rule runs after
/// `EnforceDistribution`, `EnforceSorting` and `SanityCheckPlan`, so
/// whatever it changes is not checked again:
///
/// - a single target partition: nothing to spread over;
/// - an input with an ordering: a parent may rely on it, and the node
///   otherwise maintains it;
/// - an input already hash partitioned (for example the probe side of a
///   partitioned join): a parent may require exactly that distribution, so
///   it keeps the spread it already has.
///
/// The node can rule it out too (see `split_under_loading_node` in
/// `raster_materialization`): a
/// `FilterExec` with a fetch, and an aggregate that is not a `Partial` one.
///
/// The planner's own repartitioning is replaced rather than stacked on: a
/// round-robin (`RepartitionExec(RoundRobinBatch)` without
/// `preserve_order`) or a merge (`CoalescePartitionsExec` without a fetch)
/// directly below the node is dropped and its input hashed instead, so the
/// rows cross one repartition, not two.
///
/// A single input partition is spread too, then the node is merged back
/// with a `CoalescePartitionsExec` so its output is still one partition.
/// The planner leaves an input that small when its statistics say it fits
/// in one batch (a catalog of a few thousand rasters does) or when a parent
/// requires a single partition (a `CollectLeft` build side, a global
/// limit); either way the loads are the expensive part and run in parallel,
/// and whatever is above still sees one partition.
///
/// The spread has `max(target_partitions, input partitions)` partitions, so
/// an input wider than the target (a `UnionExec`, say) keeps its width.
///
/// The repartition decouples the producer below from the loads: a fast
/// producer (a spatial join) fills the repartition's channels while the
/// loads drain them slowly, so rows queue there, still as metadata-only
/// references. Under a memory limit the repartition spills them; without one
/// they cost resident memory (about 2 GB on SpatialBench Raster Q5 at SF1).
/// Nor does the spread look at a `LIMIT` above, so a limited query may pull
/// more input than it needs.
///
/// The rule does not try to detect an input that is already well spread:
/// the plan cannot tell how many batches a partition will deliver, and the
/// cost of the repartition is one hash of rows that are still metadata-only
/// OutDb references (about 0.9 s for 6M joined rows, a few milliseconds for
/// a ten-thousand-raster catalog), while the locality it adds is worth
/// having either way.
pub(crate) fn raster_source_spread(
    input: &Arc<dyn ExecutionPlan>,
    target_partitions: usize,
) -> Option<(Arc<dyn ExecutionPlan>, usize)> {
    if target_partitions <= 1
        || input.output_ordering().is_some()
        || matches!(input.output_partitioning(), Partitioning::Hash(_, _))
    {
        return None;
    }
    let source = spread_source(input);
    let partitions = target_partitions
        .max(input.output_partitioning().partition_count())
        .max(source.output_partitioning().partition_count());
    Some((source, partitions))
}

/// The plan to hash for [`raster_source_spread`]: `input` itself, or the
/// input of a round-robin or merge directly below the loading node. Hashing
/// redistributes every row anyway, so a round-robin under it is a wasted
/// hop, and a merge under it is undone by the hash (the merge back to one
/// partition happens above the node instead).
fn spread_source(input: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if let Some(repartition) = input.downcast_ref::<RepartitionExec>()
        && matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_))
        && !repartition.preserve_order()
    {
        return Arc::clone(repartition.input());
    }
    if let Some(merge) = input.downcast_ref::<CoalescePartitionsExec>()
        && merge.fetch().is_none()
    {
        return Arc::clone(merge.input());
    }
    Arc::clone(input)
}

/// Per-row partitioning key: which file a raster reads its pixels from.
///
/// Evaluates its child (a raster) and returns, per row, a `UInt64` hash of
/// the `outdb_uri` of the first band that still has to be loaded (empty
/// `data` over a non-empty visible region), reduced to its source with
/// [`outdb_source`] (a trailing `#band=N` stripped), so every band and every
/// row reading the same file shares a key and therefore a partition. Rows
/// with nothing to load (InDb rasters, including loaded ones that keep their
/// `outdb_uri` as provenance; null rasters; rasters without bands) have no
/// file to keep together; they get values from a running counter instead,
/// so they spread like a round-robin rather than all hashing to one
/// partition.
///
/// That counter makes the expression non-deterministic: the same row can
/// get a different key on another evaluation. It therefore reports itself
/// volatile, and a
/// [`RasterBatchSplitExec`](crate::raster_materialization::RasterBatchSplitExec)
/// above the repartition advertises `UnknownPartitioning` rather than this
/// hash (see its `try_new`). Hashing fallback rows
/// deterministically was the alternative, but there is nothing cheap to
/// hash: an in-db raster's identity is its pixels, and a position in the
/// batch is no more reproducible than the counter. Equality and `Hash`
/// ignore the counter, which only matters for comparing plans.
#[derive(Debug)]
pub struct RasterSourceKeyExpr {
    arg: Arc<dyn PhysicalExpr>,
    fallback: Arc<AtomicU64>,
}

impl RasterSourceKeyExpr {
    pub fn new(arg: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            arg,
            fallback: Arc::new(AtomicU64::new(0)),
        }
    }

    fn keys(&self, rasters: &RasterStructArray) -> UInt64Array {
        let mut out = Vec::with_capacity(rasters.len());
        let mut without_source = 0u64;
        // Raster rows arrive in runs (a join emits a raster's matches
        // together), so reuse the previous hash when the source repeats.
        // The cache must stay keyed on the *stripped* source: then the
        // bands of one file (`a.tif#band=1`, `a.tif#band=2`) share a run
        // as well as a key.
        let mut last: Option<(&str, u64)> = None;
        for i in 0..rasters.len() {
            let key = match rasters.first_outdb_band_uri(i).map(outdb_source) {
                Some(source) => match last {
                    Some((prev, hash)) if prev == source => Some(hash),
                    _ => {
                        let mut hasher = DefaultHasher::new();
                        source.hash(&mut hasher);
                        let hash = hasher.finish();
                        last = Some((source, hash));
                        Some(hash)
                    }
                },
                None => {
                    without_source += 1;
                    None
                }
            };
            out.push(key);
        }
        // One atomic add per batch for all rows without a source.
        let mut next = self
            .fallback
            .fetch_add(without_source, AtomicOrdering::Relaxed);
        out.into_iter()
            .map(|key| {
                key.unwrap_or_else(|| {
                    let key = next;
                    next = next.wrapping_add(1);
                    key
                })
            })
            .collect()
    }
}

impl fmt::Display for RasterSourceKeyExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "raster_source({})", self.arg)
    }
}

impl PartialEq for RasterSourceKeyExpr {
    fn eq(&self, other: &Self) -> bool {
        self.arg.eq(&other.arg)
    }
}

impl Eq for RasterSourceKeyExpr {}

impl Hash for RasterSourceKeyExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.arg.hash(state);
    }
}

impl PhysicalExpr for RasterSourceKeyExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::UInt64)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let array = self.arg.evaluate(batch)?.into_array(batch.num_rows())?;
        let rasters = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                sedona_internal_datafusion_err!(
                    "raster_source: expected a Raster (Struct) argument, got {:?}",
                    array.data_type()
                )
            })?;
        let rasters = RasterStructArray::try_new(rasters)?;
        Ok(ColumnarValue::Array(Arc::new(self.keys(&rasters))))
    }

    /// Fallback keys come from a counter; see the type docs.
    fn is_volatile_node(&self) -> bool {
        true
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.arg]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if children.len() != 1 {
            return internal_err!("raster_source expects 1 child, got {}", children.len());
        }
        Ok(Arc::new(Self {
            arg: children.remove(0),
            fallback: Arc::clone(&self.fallback),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "raster_source(")?;
        self.arg.fmt_sql(f)?;
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{ArrayRef, Int32Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::functions_aggregate::count::count_udaf;
    use datafusion::physical_optimizer::PhysicalOptimizerRule;
    use datafusion_common::config::ConfigOptions;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::{col, is_not_null, lit};
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
    use datafusion_physical_plan::common::collect;
    use datafusion_physical_plan::filter::{FilterExec, FilterExecBuilder};
    use datafusion_physical_plan::projection::{ProjectionExec, ProjectionExpr};
    use datafusion_physical_plan::sorts::sort::SortExec;
    use datafusion_physical_plan::union::UnionExec;
    use sedona_common::option::DEFAULT_RASTER_MAX_BATCH_BYTES;
    use sedona_raster::builder::{RasterBuilder, StartBandArgs};
    use sedona_schema::datatypes::SedonaType;
    use sedona_schema::raster::BandDataType;

    use crate::raster_materialization::tests::{
        counted_raster_call, ensure_loaded_call, task_context,
    };
    use crate::raster_materialization::{
        PASSTHROUGH_BATCH_SIZE, RasterBatchSplitExec, RasterMaterializationRule,
    };

    /// `id` (the row number) and `rast`: one OutDb `[2, 2]` UInt8 band per
    /// row, reading `uris[id]`.
    fn outdb_raster_batch(uris: &[&str]) -> (SchemaRef, RecordBatch) {
        let mut b = RasterBuilder::new(uris.len());
        for uri in uris {
            b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 2], None)
                .unwrap();
            b.start_band(StartBandArgs {
                outdb_uri: Some(uri),
                outdb_format: Some("mock"),
                ..StartBandArgs::new(&["y", "x"], &[2, 2], BandDataType::UInt8)
            })
            .unwrap();
            b.band_data_writer().append_value([0u8; 0]);
            b.finish_band().unwrap();
            b.finish_raster().unwrap();
        }
        let rasters: ArrayRef = Arc::new(b.finish().unwrap());
        let raster_field = SedonaType::Raster.to_storage_field("rast", true).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("rast", rasters.data_type().clone(), true)
                .with_metadata(raster_field.metadata().clone()),
        ]));
        let ids = Arc::new(Int32Array::from_iter_values(0..uris.len() as i32));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, rasters]).unwrap();
        (schema, batch)
    }

    /// `ProjectionExec(id, rast, call AS loaded)` over `input`: the shape the
    /// planner gives `SELECT id, rast, RS_EnsureLoaded(..) AS loaded`.
    fn loading_projection(
        call: Arc<dyn PhysicalExpr>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = input.schema();
        Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(col("id", &schema).unwrap(), "id"),
                    ProjectionExpr::new(col("rast", &schema).unwrap(), "rast"),
                    ProjectionExpr::new(call, "loaded"),
                ],
                input,
            )
            .unwrap(),
        )
    }

    #[test]
    fn raster_source_key_groups_rows_by_file_and_spreads_rows_without_one() {
        // Rows 0-3 and 6-9 are OutDb, rows 4-5 InDb (row 5 loaded from
        // a.tif, which it keeps as provenance); then a null raster and a
        // raster without bands.
        let mut b = RasterBuilder::new(12);
        for (uri, loaded) in [
            (Some("s3://b/a.tif#band=1"), false),
            (Some("s3://b/a.tif#band=2"), false),
            (Some("s3://b/b.tif"), false),
            (Some("s3://b/a.tif"), false),
            (None, true),
            (Some("s3://b/a.tif"), true),
            (Some("s3://b/x.zarr#var=a"), false),
            (Some("s3://b/x.zarr#var=b"), false),
            (Some("s3://b/p.tif#anchor#band=1"), false),
            (Some("s3://b/p.tif#anchor"), false),
        ] {
            b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 2], None)
                .unwrap();
            b.start_band(StartBandArgs {
                outdb_uri: uri,
                outdb_format: uri.map(|_| "mock"),
                ..StartBandArgs::new(&["y", "x"], &[2, 2], BandDataType::UInt8)
            })
            .unwrap();
            if loaded {
                b.band_data_writer().append_value([0u8; 4]);
            } else {
                b.band_data_writer().append_value([0u8; 0]);
            }
            b.finish_band().unwrap();
            b.finish_raster().unwrap();
        }
        b.append_null().unwrap();
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 2], None)
            .unwrap();
        b.finish_raster().unwrap();
        let rasters: ArrayRef = Arc::new(b.finish().unwrap());
        let schema = Schema::new(vec![Field::new("rast", rasters.data_type().clone(), true)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![rasters]).unwrap();
        let key = RasterSourceKeyExpr::new(col("rast", &schema).unwrap());
        assert!(key.is_volatile_node());
        let keys = key.evaluate(&batch).unwrap().into_array(12).unwrap();
        let keys = keys
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();

        // Every band and every row of one file share a key; other files differ.
        assert_eq!(keys[0], keys[1]);
        assert_eq!(keys[0], keys[3]);
        assert_ne!(keys[0], keys[2]);
        // Only a trailing `#band=N` is stripped: other fragments name
        // different sources, and an earlier `#anchor` stays.
        assert_ne!(keys[6], keys[7]);
        assert_eq!(keys[8], keys[9]);
        // Rows with nothing to load draw consecutive counter values, one
        // batch at a time, even when they carry a URI.
        let mut fallback = vec![keys[4], keys[5], keys[10], keys[11]];
        fallback.sort();
        assert_eq!(fallback, (fallback[0]..fallback[0] + 4).collect::<Vec<_>>());
        // A sliced batch reads its own rows, and an empty one yields no keys.
        let sliced = key
            .evaluate(&batch.slice(2, 2))
            .unwrap()
            .into_array(2)
            .unwrap();
        assert_eq!(
            sliced
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .to_vec(),
            keys[2..4].to_vec()
        );
        let empty = key
            .evaluate(&batch.slice(0, 0))
            .unwrap()
            .into_array(0)
            .unwrap();
        assert_eq!(empty.len(), 0);
    }

    #[tokio::test]
    async fn loading_node_input_is_hash_partitioned_by_raster_source() {
        // Sixteen rasters, four rows each, interleaved over two input
        // partitions of one batch each: a whole-batch round-robin could only
        // ever keep two partitions busy.
        let uris: Vec<String> = (0..64)
            .map(|i| format!("s3://b/r{}.tif#band={}", i % 16, i / 16 + 1))
            .collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, first) = outdb_raster_batch(&uris[..32]);
        let (_, second) = outdb_raster_batch(&uris[32..]);
        let input = MemorySourceConfig::try_new_exec(
            &[vec![first], vec![second]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let projection = loading_projection(
            ensure_loaded_call(&calls, col("rast", &schema).unwrap(), &schema),
            input,
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Projection ← Split ← hash RepartitionExec ← scan
        assert_eq!(plan.schema(), expected_schema);
        let split = plan
            .downcast_ref::<ProjectionExec>()
            .expect("projection kept")
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the projection");
        assert_eq!(split.sized_columns(), &[1]);
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
        let Partitioning::Hash(keys, 4) = repartition.partitioning() else {
            panic!("expected a 4-way hash, got {}", repartition.partitioning());
        };
        assert_eq!(keys[0].to_string(), "raster_source(rast@1)");
        assert!(
            repartition
                .input()
                .downcast_ref::<DataSourceExec>()
                .is_some()
        );
        // The counter-drawn keys of rows without a file are not reproducible,
        // so the hash is not advertised to parents.
        assert!(matches!(
            split.properties().output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        assert!(matches!(
            plan.output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        assert!(plan.output_ordering().is_none());

        let key = RasterSourceKeyExpr::new(col("rast", &schema).unwrap());
        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let mut partition_of = HashMap::new();
        let mut rows = 0;
        let mut busy = 0;
        for partition in 0..4 {
            let batches = collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap();
            let mut here = 0;
            for batch in &batches {
                assert_eq!(batch.schema(), expected_schema);
                let keys = key
                    .evaluate(batch)
                    .unwrap()
                    .into_array(batch.num_rows())
                    .unwrap();
                for &k in keys
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .values()
                {
                    // Every row of a raster, from either input partition,
                    // lands in the same output partition.
                    assert_eq!(*partition_of.entry(k).or_insert(partition), partition);
                    here += 1;
                }
            }
            rows += here;
            busy += usize::from(here > 0);
        }
        assert_eq!(rows, 64);
        assert_eq!(partition_of.len(), 16);
        assert!(busy > 2, "only {busy} of 4 partitions received rows");
        assert_eq!(calls.lock().unwrap().iter().sum::<usize>(), 64);
    }

    #[test]
    fn raster_source_spread_is_skipped_where_it_could_break_a_parent() {
        let (schema, batch) = outdb_raster_batch(&["s3://b/a.tif", "s3://b/b.tif"]);
        let two_partitions = MemorySourceConfig::try_new_exec(
            &[vec![batch.clone()], vec![batch]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let by_id =
            LexOrdering::new([PhysicalSortExpr::new_default(col("id", &schema).unwrap())]).unwrap();
        let cases: Vec<(&str, Arc<dyn ExecutionPlan>, usize)> = vec![
            ("one target partition", two_partitions.clone(), 1),
            (
                "ordered input",
                Arc::new(
                    SortExec::new(by_id, two_partitions.clone()).with_preserve_partitioning(true),
                ),
                4,
            ),
            (
                "hash-partitioned input",
                Arc::new(
                    RepartitionExec::try_new(
                        two_partitions.clone(),
                        Partitioning::Hash(vec![col("id", &schema).unwrap()], 4),
                    )
                    .unwrap(),
                ),
                4,
            ),
        ];
        for (name, input, target_partitions) in cases {
            let projection = loading_projection(
                ensure_loaded_call(
                    &Arc::new(Mutex::new(Vec::new())),
                    col("rast", &schema).unwrap(),
                    &schema,
                ),
                Arc::clone(&input),
            );
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = target_partitions;
            let plan = RasterMaterializationRule
                .optimize(projection, &config)
                .unwrap();
            // The split still goes in, directly over the input.
            let split = plan
                .downcast_ref::<ProjectionExec>()
                .unwrap_or_else(|| panic!("{name}: got {}", plan.name()))
                .input()
                .downcast_ref::<RasterBatchSplitExec>()
                .unwrap_or_else(|| panic!("{name}: no split"));
            assert!(Arc::ptr_eq(split.input(), &input), "{name}");
        }
    }

    #[tokio::test]
    async fn single_input_partition_is_spread_and_merged_back() {
        // The planner leaves a small input (or one under a single-partition
        // requirement) on one partition; the loads still spread, and the
        // output is one partition again.
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, batch) = outdb_raster_batch(&uris);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let projection = loading_projection(
            ensure_loaded_call(&calls, col("rast", &schema).unwrap(), &schema),
            input,
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Coalesce ← Projection ← Split ← hash RepartitionExec ← scan
        assert_eq!(plan.schema(), expected_schema);
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        let merge = plan.downcast_ref::<CoalescePartitionsExec>().unwrap();
        let split = merge
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap()
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .unwrap();
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
        assert!(matches!(
            repartition.partitioning(),
            Partitioning::Hash(_, 4)
        ));

        let batches = collect(
            plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 16);
        // One invocation per busy partition, not one for the whole input.
        assert!(calls.lock().unwrap().len() > 1);
    }

    #[tokio::test]
    async fn expression_raster_argument_is_evaluated_once_below_the_repartition() {
        // Hashing `RS_FromPath(path)`-style arguments directly would evaluate
        // them twice (key, then load). The argument is hoisted into a column
        // under the repartition instead, so it is evaluated once per input
        // batch and both the hash and the sizing read the column.
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, first) = outdb_raster_batch(&uris[..8]);
        let (_, second) = outdb_raster_batch(&uris[8..]);
        let input = MemorySourceConfig::try_new_exec(
            &[vec![first], vec![second]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let projection = loading_projection(
            ensure_loaded_call(
                &calls,
                counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                &schema,
            ),
            input,
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Projection ← Split ← hash RepartitionExec ← Projection (evaluate
        // the argument) ← scan
        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        assert_eq!(
            top.expr()[2].expr.to_string(),
            "rs_ensureloaded(__ensure_loaded_arg_0@2)"
        );
        let split = top.input().downcast_ref::<RasterBatchSplitExec>().unwrap();
        assert_eq!(split.sized_columns(), &[2]);
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
        let Partitioning::Hash(keys, 4) = repartition.partitioning() else {
            panic!("expected a 4-way hash, got {}", repartition.partitioning());
        };
        assert_eq!(
            keys[0].to_string(),
            "raster_source(__ensure_loaded_arg_0@2)"
        );
        let below = repartition
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap();
        assert_eq!(
            below
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec!["id", "rast", "__ensure_loaded_arg_0"]
        );
        assert!(below.input().downcast_ref::<DataSourceExec>().is_some());
        assert!(matches!(
            plan.output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        // The raster columns keep their extension metadata through both
        // projections.
        let raster_metadata = SedonaType::Raster
            .to_storage_field("rast", true)
            .unwrap()
            .metadata()
            .clone();
        assert!(!raster_metadata.is_empty());
        assert_eq!(below.schema().field(1).metadata(), &raster_metadata);
        assert_eq!(below.schema().field(2).metadata(), &raster_metadata);
        assert_eq!(plan.schema().field(1).metadata(), &raster_metadata);
        assert_eq!(plan.schema().field(2).metadata(), &raster_metadata);

        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let mut rows = 0;
        for partition in 0..4 {
            for batch in collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap()
            {
                assert_eq!(batch.schema(), expected_schema);
                // Identity mocks: the loaded column is the input raster.
                assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
                rows += batch.num_rows();
            }
        }
        assert_eq!(rows, 16);
        assert_eq!(calls.lock().unwrap().iter().sum::<usize>(), 16);
        // Once per input batch, below the repartition; never again above it.
        assert_eq!(evaluations.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn expression_argument_over_a_single_input_partition_is_spread_and_merged_back() {
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, batch) = outdb_raster_batch(&uris);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let projection = loading_projection(
            ensure_loaded_call(
                &calls,
                counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                &schema,
            ),
            input,
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Coalesce ← Projection ← Split ← hash RepartitionExec ← Projection
        // (evaluate the argument) ← scan
        assert_eq!(plan.schema(), expected_schema);
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        let merge = plan.downcast_ref::<CoalescePartitionsExec>().unwrap();
        let split = merge
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap()
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .unwrap();
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
        assert!(matches!(
            repartition.partitioning(),
            Partitioning::Hash(_, 4)
        ));
        let below = repartition
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap();
        assert!(below.input().downcast_ref::<DataSourceExec>().is_some());

        let batches = collect(
            plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 16);
        for batch in &batches {
            assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
        }
        assert!(calls.lock().unwrap().len() > 1);
        // Evaluated once, on the single input partition, before the spread.
        assert_eq!(evaluations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn spread_keys_on_the_first_call_when_it_is_an_expression_and_the_second_a_column() {
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, first) = outdb_raster_batch(&uris[..8]);
        let (_, second) = outdb_raster_batch(&uris[8..]);
        let input = MemorySourceConfig::try_new_exec(
            &[vec![first], vec![second]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(col("id", &schema).unwrap(), "id"),
                    ProjectionExpr::new(col("rast", &schema).unwrap(), "rast"),
                    ProjectionExpr::new(
                        ensure_loaded_call(
                            &calls,
                            counted_raster_call(
                                &evaluations,
                                col("rast", &schema).unwrap(),
                                &schema,
                            ),
                            &schema,
                        ),
                        "a",
                    ),
                    ProjectionExpr::new(
                        ensure_loaded_call(&calls, col("rast", &schema).unwrap(), &schema),
                        "b",
                    ),
                ],
                input,
            )
            .unwrap(),
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        let calls_shown: Vec<String> = top.expr()[2..].iter().map(|p| p.expr.to_string()).collect();
        assert_eq!(
            calls_shown,
            vec![
                "rs_ensureloaded(__ensure_loaded_arg_0@2)",
                "rs_ensureloaded(rast@1)"
            ]
        );
        let split = top.input().downcast_ref::<RasterBatchSplitExec>().unwrap();
        assert_eq!(split.sized_columns(), &[1, 2]);
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
        let Partitioning::Hash(keys, 4) = repartition.partitioning() else {
            panic!("expected a 4-way hash, got {}", repartition.partitioning());
        };
        assert_eq!(
            keys[0].to_string(),
            "raster_source(__ensure_loaded_arg_0@2)"
        );

        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let mut rows = 0;
        for partition in 0..4 {
            for batch in collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap()
            {
                assert_eq!(batch.schema(), expected_schema);
                assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
                assert_eq!(batch.column(1).as_ref(), batch.column(3).as_ref());
                rows += batch.num_rows();
            }
        }
        assert_eq!(rows, 16);
        assert_eq!(calls.lock().unwrap().iter().sum::<usize>(), 32);
        assert_eq!(evaluations.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn parents_above_the_spread_are_rebuilt_and_filters_still_pass_through() {
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, first) = outdb_raster_batch(&uris[..8]);
        let (_, second) = outdb_raster_batch(&uris[8..]);
        let input = MemorySourceConfig::try_new_exec(
            &[vec![first], vec![second]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let loading = loading_projection(
            ensure_loaded_call(
                &Arc::new(Mutex::new(Vec::new())),
                counted_raster_call(
                    &Arc::new(AtomicUsize::new(0)),
                    col("rast", &schema).unwrap(),
                    &schema,
                ),
                &schema,
            ),
            input,
        );
        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(lit(true), loading).unwrap());
        let filter_schema = filter.schema();
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(col("id", &filter_schema).unwrap(), "id"),
                    ProjectionExpr::new(col("loaded", &filter_schema).unwrap(), "loaded"),
                ],
                filter,
            )
            .unwrap(),
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Projection ← Filter ← Projection (load) ← Split ← hash.
        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        let filter = top.input().downcast_ref::<FilterExec>().unwrap();
        assert_eq!(filter.batch_size(), PASSTHROUGH_BATCH_SIZE);
        let split = filter
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap()
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .unwrap();
        assert!(split.input().downcast_ref::<RepartitionExec>().is_some());
        assert_eq!(plan.output_partitioning().partition_count(), 4);

        // Four-byte rasters under a four-byte budget: one row per slice, and
        // the filter forwards every slice instead of merging them.
        let ctx = task_context(4);
        let mut sizes = Vec::new();
        for partition in 0..4 {
            let batches = collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap();
            sizes.extend(batches.iter().map(RecordBatch::num_rows));
        }
        assert_eq!(sizes, vec![1; 16]);
    }

    #[test]
    fn planner_repartitioning_below_the_node_is_replaced_not_stacked() {
        let (schema, batch) = outdb_raster_batch(&["s3://b/a.tif", "s3://b/b.tif"]);
        let one_partition: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                .unwrap();
        let two_partitions: Arc<dyn ExecutionPlan> = MemorySourceConfig::try_new_exec(
            &[vec![batch.clone()], vec![batch]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let round_robin: Arc<dyn ExecutionPlan> = Arc::new(
            RepartitionExec::try_new(Arc::clone(&one_partition), Partitioning::RoundRobinBatch(4))
                .unwrap(),
        );
        let merge: Arc<dyn ExecutionPlan> =
            Arc::new(CoalescePartitionsExec::new(Arc::clone(&two_partitions)));
        let limited_merge: Arc<dyn ExecutionPlan> =
            Arc::new(CoalescePartitionsExec::new(Arc::clone(&two_partitions)).with_fetch(Some(1)));
        let wide_union: Arc<dyn ExecutionPlan> = UnionExec::try_new(vec![
            Arc::clone(&two_partitions),
            Arc::clone(&two_partitions),
            Arc::clone(&two_partitions),
        ])
        .unwrap();
        // (case, input, plan hashed below the split, partitions, merged back)
        let cases = vec![
            ("round-robin", round_robin, one_partition, 4, false),
            ("merge", merge, Arc::clone(&two_partitions), 4, true),
            // A merge with a fetch limits rows; it stays.
            (
                "merge with fetch",
                Arc::clone(&limited_merge),
                limited_merge,
                4,
                true,
            ),
            // Six input partitions over a target of four keep all six.
            ("wide union", Arc::clone(&wide_union), wide_union, 6, false),
        ];
        for (name, input, hashed, partitions, merged) in cases {
            let projection = loading_projection(
                ensure_loaded_call(
                    &Arc::new(Mutex::new(Vec::new())),
                    col("rast", &schema).unwrap(),
                    &schema,
                ),
                input,
            );
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = 4;
            let plan = RasterMaterializationRule
                .optimize(projection, &config)
                .unwrap();

            let loading = if merged {
                assert_eq!(plan.output_partitioning().partition_count(), 1, "{name}");
                plan.downcast_ref::<CoalescePartitionsExec>()
                    .unwrap_or_else(|| panic!("{name}: got {}", plan.name()))
                    .input()
            } else {
                assert_eq!(
                    plan.output_partitioning().partition_count(),
                    partitions,
                    "{name}"
                );
                &plan
            };
            let split = loading
                .downcast_ref::<ProjectionExec>()
                .unwrap_or_else(|| panic!("{name}: got {}", loading.name()))
                .input()
                .downcast_ref::<RasterBatchSplitExec>()
                .unwrap_or_else(|| panic!("{name}: no split"));
            let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
            assert!(
                matches!(repartition.partitioning(), Partitioning::Hash(_, n) if *n == partitions),
                "{name}: {}",
                repartition.partitioning()
            );
            assert!(Arc::ptr_eq(repartition.input(), &hashed), "{name}");
        }
    }

    #[tokio::test]
    async fn a_user_column_named_like_the_hoisted_argument_is_not_read_as_it() {
        // The hoisted argument is appended at position 2 as
        // `__ensure_loaded_arg_0`; the user's Int32 column of that name sits
        // at position 0. Columns are bound by position, so the loader and the
        // hash still read the raster.
        let uris: Vec<String> = (0..8).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (outdb_schema, outdb) = outdb_raster_batch(&uris);
        let schema = Arc::new(Schema::new(vec![
            Field::new("__ensure_loaded_arg_0", DataType::Int32, false),
            outdb_schema.field(1).clone(),
        ]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), outdb.columns().to_vec()).unwrap();
        for target_partitions in [1, 4] {
            let input =
                MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                    .unwrap();
            let projection: Arc<dyn ExecutionPlan> = Arc::new(
                ProjectionExec::try_new(
                    vec![
                        ProjectionExpr::new(col("__ensure_loaded_arg_0", &schema).unwrap(), "n"),
                        ProjectionExpr::new(col("rast", &schema).unwrap(), "rast"),
                        ProjectionExpr::new(
                            ensure_loaded_call(
                                &Arc::new(Mutex::new(Vec::new())),
                                counted_raster_call(
                                    &Arc::new(AtomicUsize::new(0)),
                                    col("rast", &schema).unwrap(),
                                    &schema,
                                ),
                                &schema,
                            ),
                            "loaded",
                        ),
                    ],
                    input,
                )
                .unwrap(),
            );
            let expected_schema = projection.schema();
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = target_partitions;
            let plan = RasterMaterializationRule
                .optimize(projection, &config)
                .unwrap();
            assert_eq!(plan.schema(), expected_schema);
            let batches = collect(
                plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                    .unwrap(),
            )
            .await
            .unwrap();
            assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 8);
            for batch in &batches {
                assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
            }
        }
    }

    #[tokio::test]
    async fn only_partial_aggregates_and_unlimited_filters_are_spread() {
        // Over a single input partition: a `Partial` stage is spread and
        // merged back, while a `Single` stage must stay on one partition
        // (its result is the one aggregate) and a filter with a fetch is the
        // query's limit. Both still get the split.
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, batch) = outdb_raster_batch(&uris);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;

        for (mode, spread) in [
            (AggregateMode::Partial, true),
            (AggregateMode::Single, false),
        ] {
            let input =
                MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                    .unwrap();
            let count = AggregateExprBuilder::new(
                count_udaf(),
                vec![ensure_loaded_call(
                    &calls,
                    col("rast", &schema).unwrap(),
                    &schema,
                )],
            )
            .schema(Arc::clone(&schema))
            .alias("c")
            .build()
            .unwrap();
            let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
                AggregateExec::try_new(
                    mode,
                    PhysicalGroupBy::new_single(vec![]),
                    vec![Arc::new(count)],
                    vec![None],
                    input,
                    Arc::clone(&schema),
                )
                .unwrap(),
            );
            let expected_schema = aggregate.schema();
            let plan = RasterMaterializationRule
                .optimize(aggregate, &config)
                .unwrap();
            assert_eq!(plan.schema(), expected_schema, "{mode:?}");
            assert_eq!(plan.output_partitioning().partition_count(), 1, "{mode:?}");
            let aggregate = if spread {
                plan.downcast_ref::<CoalescePartitionsExec>()
                    .unwrap_or_else(|| panic!("{mode:?}: got {}", plan.name()))
                    .input()
            } else {
                &plan
            }
            .downcast_ref::<AggregateExec>()
            .unwrap_or_else(|| panic!("{mode:?}: no aggregate"));
            let split = aggregate
                .input()
                .downcast_ref::<RasterBatchSplitExec>()
                .unwrap_or_else(|| panic!("{mode:?}: no split"));
            assert_eq!(
                split.input().downcast_ref::<RepartitionExec>().is_some(),
                spread,
                "{mode:?}"
            );

            let batches = collect(
                plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                    .unwrap(),
            )
            .await
            .unwrap();
            let counts: Vec<i64> = batches
                .iter()
                .flat_map(|b| {
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect();
            // A partial stage emits one state per busy partition, a single
            // stage the one result.
            assert_eq!(counts.iter().sum::<i64>(), 16, "{mode:?}");
            assert_eq!(counts.len() > 1, spread, "{mode:?}");
        }

        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                .unwrap();
        let filter: Arc<dyn ExecutionPlan> = Arc::new(
            FilterExecBuilder::new(
                is_not_null(ensure_loaded_call(
                    &calls,
                    col("rast", &schema).unwrap(),
                    &schema,
                ))
                .unwrap(),
                input,
            )
            .with_fetch(Some(3))
            .build()
            .unwrap(),
        );
        let plan = RasterMaterializationRule.optimize(filter, &config).unwrap();
        let filter = plan.downcast_ref::<FilterExec>().expect("filter kept");
        assert_eq!(filter.fetch(), Some(3));
        let split = filter
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the filter");
        assert!(split.input().downcast_ref::<DataSourceExec>().is_some());
        let batches = collect(
            plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    }
}
