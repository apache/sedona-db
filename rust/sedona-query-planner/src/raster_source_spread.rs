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
//! is spread over partitions by raster source, by a
//! [`RasterSourceSpreadExec`].
//!
//! The planner usually feeds that node through a round-robin repartition,
//! which deals whole batches, and a raster catalog is often tiny (one
//! Parquet row group, a couple of batches), so all the loading lands on a
//! couple of partitions while the rest idle. Hashing is per row, so even a
//! two-batch catalog spreads over every partition, and the rows of one
//! raster share a partition, so the chunk cache loads each raster once
//! instead of once per partition that happens to hold some of its rows.
//! [`raster_source_spread`] decides when the spread is safe and what it
//! replaces, [`raster_source_keys`] is the key, and the
//! [`RasterMaterializationRule`](crate::raster_materialization::RasterMaterializationRule)
//! builds the spread between the split and the input.
//!
//! The spread is a Sedona-owned operator rather than DataFusion's
//! `RepartitionExec`, because the repartition gathers the rows it sends to
//! each output partition back up to `datafusion.execution.batch_size`
//! through a shared coalescer, and only accounts for memory after that. The
//! coalescer copies the `data` (BinaryView) bytes of rasters that are
//! already in memory (a 1/N hash take always leaves its buffers sparse), and
//! it buffers up to a full batch of them per partition before
//! `sedona.raster.max_batch_bytes` (enforced by the split above) or the
//! memory pool sees a byte. [`RasterSourceSpreadExec`] never merges: it
//! forwards a batch that has nothing to load whole, sends each hash piece of
//! the rest as is, and bounds and accounts what it queues per output.

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use arrow_array::{Array, RecordBatch, StructArray, UInt32Array};
use datafusion_common::arrow::compute::take_record_batch;
use datafusion_common::{DataFusionError, Result, internal_err};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::execution_plan::{CardinalityEffect, EvaluationType, SchedulingType};
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties,
};
use futures::{FutureExt, StreamExt, stream};
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::outdb_source;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};

use crate::raster_materialization::{budget_bytes, is_raster_field};

/// The spread of a loading node's `input` by raster source: the plan to
/// spread (`input` itself, or what the planner's own repartitioning below
/// the node reads, see [`spread_source`]) and the number of partitions, or
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
/// directly below the node is dropped and its input spread instead, so the
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
/// The spread decouples the producer below from the loads: a fast producer
/// (a spatial join) fills the spread's queues while the loads drain them
/// slowly, so rows queue there, still as metadata-only references, up to
/// the bound described on [`RasterSourceSpreadExec`]. Nor does the spread
/// look at a `LIMIT` above, so a limited query may pull more input than it
/// needs.
///
/// The rule does not try to detect an input that is already well spread:
/// the plan cannot tell how many batches a partition will deliver, and the
/// cost of the spread is one hash of rows that are still metadata-only
/// OutDb references, while the locality it adds is worth having either way.
/// Batches with nothing to load pass through whole.
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

/// The plan to spread for [`raster_source_spread`]: `input` itself, or the
/// input of a round-robin or merge directly below the loading node. The
/// spread redistributes every row anyway, so a round-robin under it is a
/// wasted hop, and a merge under it is undone by the spread (the merge back
/// to one partition happens above the node instead).
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

/// Per-row spread key: which file a raster still has to load its pixels
/// from.
///
/// For a row with a band to load (by the data/shape test of
/// [`RasterStructArray::first_outdb_band_uri`], not by whether it names a
/// file), a hash of that first band's `outdb_uri` reduced to its source with
/// [`outdb_source`] (a trailing `#band=N` stripped), so every band and every
/// row reading the same file shares a key and therefore a partition. Rows
/// with nothing to load (in-db rasters, rasters already loaded, which keep
/// their `outdb_uri` as provenance, null rasters, rasters without bands)
/// and rows to load without a URI get `None`; the spread deals them
/// round-robin, since there is no file to keep together.
pub fn raster_source_keys(rasters: &RasterStructArray) -> Vec<Option<u64>> {
    // Raster rows arrive in runs (a join emits a raster's matches
    // together), so reuse the previous hash when the source repeats. The
    // cache must stay keyed on the *stripped* source: then the bands of one
    // file (`a.tif#band=1`, `a.tif#band=2`) share a run as well as a key.
    let mut last: Option<(&str, u64)> = None;
    (0..rasters.len())
        .map(|idx| {
            let source = outdb_source(rasters.first_outdb_band_uri(idx)?);
            if let Some((prev, hash)) = last
                && prev == source
            {
                return Some(hash);
            }
            let mut hasher = DefaultHasher::new();
            source.hash(&mut hasher);
            let hash = hasher.finish();
            last = Some((source, hash));
            Some(hash)
        })
        .collect()
}

/// Spreads the rows of its input partitions over `partitions` output
/// partitions by raster source ([`raster_source_keys`] of the raster column
/// `key_column`), for the
/// [`RasterBatchSplitExec`](crate::raster_materialization::RasterBatchSplitExec)
/// and loading node above. See the module docs for why it is not a
/// `RepartitionExec`.
///
/// One task per input partition pulls its batches and, per batch:
///
/// - forwards it whole, to the next output in a round-robin, when no row
///   has a band to load, so rasters already in memory are never taken
///   apart, copied or merged;
/// - otherwise sends the rows of each source to output `key % partitions`
///   and deals rows without a source round-robin, one `take` per output
///   that receives rows (none when one output receives the whole batch).
///   A `take` copies the BinaryView views, not the bytes they point to.
///
/// Pieces are queued as they are, never merged. Each output's queue holds
/// at most `sedona.raster.max_batch_bytes` (by `get_array_memory_size`,
/// which counts a buffer a piece shares with its siblings in full, so the
/// bound is conservative), and always admits one piece when it is empty, so
/// a piece larger than the budget still goes through, alone. Queued pieces
/// are accounted to a per-output `MemoryReservation`. When the pool refuses
/// a piece, the sender waits for that output's queue to empty and then
/// queues it anyway: the input is not pulled further meanwhile, and the
/// query goes on one piece per output at a time instead of failing. Nothing
/// is spilled: the queues are bounded, and the rows usually queued here are
/// metadata-only OutDb references.
///
/// A full queue stalls the input task that feeds it until its output takes
/// a piece. Only outputs that have been polled are waited for (an output
/// not polled yet queues without a bound), so outputs drained one after
/// the other do not deadlock; an output that was polled once and then left
/// unread while another is read to its end would. Rows of one input partition keep
/// their relative order within an output partition.
///
/// The output partitioning is `UnknownPartitioning`: rows without a source
/// are dealt by position, so the spread is a load-balancing device, not a
/// distribution any other plan could reproduce and match.
#[derive(Debug)]
pub struct RasterSourceSpreadExec {
    input: Arc<dyn ExecutionPlan>,
    /// Input raster column whose sources key the spread.
    key_column: usize,
    cache: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    /// The output queues, created with the input tasks on the first
    /// `execute` and handed out one per output partition.
    outputs: Mutex<Option<Vec<Option<OutputQueue>>>>,
}

impl RasterSourceSpreadExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        key_column: usize,
        partitions: usize,
    ) -> Result<Self> {
        if partitions == 0 {
            return internal_err!("RasterSourceSpreadExec requires at least one partition");
        }
        match input.schema().fields().get(key_column) {
            Some(field) if is_raster_field(field) => {}
            _ => {
                return internal_err!(
                    "RasterSourceSpreadExec: key column {key_column} is not a raster column"
                );
            }
        }
        let mut eq_properties = input.equivalence_properties().clone();
        eq_properties.clear_orderings();
        if input.output_partitioning().partition_count() > 1 {
            eq_properties.clear_per_partition_constants();
        }
        let cache = PlanProperties::new(
            eq_properties,
            Partitioning::UnknownPartitioning(partitions),
            input.pipeline_behavior(),
            input.boundedness(),
        )
        .with_scheduling_type(SchedulingType::Cooperative)
        .with_evaluation_type(EvaluationType::Eager);
        Ok(Self {
            input,
            key_column,
            cache: Arc::new(cache),
            metrics: ExecutionPlanMetricsSet::new(),
            outputs: Mutex::new(None),
        })
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub fn key_column(&self) -> usize {
        self.key_column
    }

    pub fn partitions(&self) -> usize {
        self.cache.output_partitioning().partition_count()
    }

    /// Execute every input partition, each in a task that spreads it over
    /// the output queues this returns.
    fn start(&self, context: &Arc<TaskContext>) -> Result<Vec<Option<OutputQueue>>> {
        let partitions = self.partitions();
        // Permits are bytes. `0` (no budget) makes `budget_bytes` unbounded;
        // the semaphore takes at most `u32::MAX` per acquire.
        let capacity = budget_bytes(context).min(u64::from(u32::MAX)) as u32;
        let mut senders = Vec::with_capacity(partitions);
        let mut queues = Vec::with_capacity(partitions);
        for partition in 0..partitions {
            let (tx, rx) = mpsc::unbounded_channel();
            let space = Arc::new(Semaphore::new(capacity as usize));
            let polled = Arc::new(AtomicBool::new(false));
            let reservation = Arc::new(
                MemoryConsumer::new(format!("RasterSourceSpreadExec[{partition}]"))
                    .register(context.memory_pool()),
            );
            senders.push(OutputSender {
                tx,
                space: Arc::clone(&space),
                polled: Arc::clone(&polled),
                capacity,
                reservation: Arc::clone(&reservation),
            });
            queues.push((
                rx,
                space,
                polled,
                reservation,
                BaselineMetrics::new(&self.metrics, partition),
            ));
        }

        let input_partitions = self.input.output_partitioning().partition_count();
        let mut tasks = Vec::with_capacity(input_partitions);
        for input_partition in 0..input_partitions {
            let input = self.input.execute(input_partition, Arc::clone(context))?;
            let spread = spread_input(input, senders.clone(), self.key_column, input_partition);
            let senders = senders.clone();
            // One task per input, aborted directly when the outputs go. An
            // error or a panic reaches every output instead of ending it as
            // if the input were done.
            tasks.push(SpawnedTask::spawn(async move {
                let error = match AssertUnwindSafe(spread).catch_unwind().await {
                    Ok(Ok(())) => return,
                    Ok(Err(e)) => e,
                    Err(panic) => {
                        let message = panic
                            .downcast_ref::<&str>()
                            .map(|m| m.to_string())
                            .or_else(|| panic.downcast_ref::<String>().cloned())
                            .unwrap_or_default();
                        sedona_internal_datafusion_err!(
                            "RasterSourceSpreadExec: input task panicked: {message}"
                        )
                    }
                };
                let error = Arc::new(error);
                for sender in senders {
                    sender
                        .tx
                        .send(Err(DataFusionError::Shared(Arc::clone(&error))))
                        .ok();
                }
            }));
        }
        // Every output holds the tasks: they are aborted once all outputs
        // are dropped.
        let tasks = Arc::new(tasks);
        Ok(queues
            .into_iter()
            .map(|(rx, space, polled, reservation, baseline)| {
                Some(OutputQueue {
                    rx,
                    space,
                    polled,
                    reservation,
                    baseline,
                    _tasks: Arc::clone(&tasks),
                })
            })
            .collect())
    }
}

/// A piece of an input batch queued for an output, with the queue space
/// (none before the output is first polled) and the reserved bytes it holds
/// until the output takes it.
struct Piece {
    batch: RecordBatch,
    bytes: usize,
    _space: Option<OwnedSemaphorePermit>,
}

/// The input tasks' end of one output's queue.
#[derive(Clone)]
struct OutputSender {
    tx: mpsc::UnboundedSender<Result<Piece>>,
    /// Free queue space in bytes; closed when the output is dropped.
    space: Arc<Semaphore>,
    /// Whether the output has been polled yet; see [`OutputSender::send`].
    polled: Arc<AtomicBool>,
    capacity: u32,
    reservation: Arc<MemoryReservation>,
}

impl OutputSender {
    /// Queue `batch`, waiting for space (see [`RasterSourceSpreadExec`]).
    /// `false` when the output has been dropped.
    ///
    /// An output nobody has polled yet is not waited for: a consumer may
    /// drain the outputs one after the other (each to its end before the
    /// next is executed), and waiting for a later one would stall the one
    /// being read. Its pieces queue without a bound and are reserved even
    /// past the pool's limit, as they would be in a `RepartitionExec` that
    /// cannot spill; a parent that polls every output at once (a
    /// `CoalescePartitionsExec`) bounds every queue from its first poll.
    async fn send(&self, batch: RecordBatch) -> bool {
        let bytes = batch.get_array_memory_size();
        if !self.polled.load(AtomicOrdering::Acquire) {
            self.reservation.grow(bytes);
            return self
                .tx
                .send(Ok(Piece {
                    batch,
                    bytes,
                    _space: None,
                }))
                .is_ok();
        }
        let wanted = bytes.min(self.capacity as usize) as u32;
        let Ok(mut space) = Arc::clone(&self.space).acquire_many_owned(wanted).await else {
            return false;
        };
        if self.reservation.try_grow(bytes).is_err() {
            // Over the memory limit: wait for this queue to empty, then
            // queue the piece anyway. Release the partial space first, or two
            // senders each holding part of it would wait on each other.
            drop(space);
            space = match Arc::clone(&self.space)
                .acquire_many_owned(self.capacity)
                .await
            {
                Ok(space) => space,
                Err(_) => return false,
            };
            self.reservation.grow(bytes);
        }
        self.tx
            .send(Ok(Piece {
                batch,
                bytes,
                _space: Some(space),
            }))
            .is_ok()
    }
}

/// An output partition's end of its queue.
#[derive(Debug)]
struct OutputQueue {
    rx: mpsc::UnboundedReceiver<Result<Piece>>,
    space: Arc<Semaphore>,
    polled: Arc<AtomicBool>,
    reservation: Arc<MemoryReservation>,
    baseline: BaselineMetrics,
    _tasks: Arc<Vec<SpawnedTask<()>>>,
}

impl fmt::Debug for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Piece({} rows, {} bytes)",
            self.batch.num_rows(),
            self.bytes
        )
    }
}

impl OutputQueue {
    async fn next(&mut self) -> Option<Result<RecordBatch>> {
        self.polled.store(true, AtomicOrdering::Release);
        let piece = match self.rx.recv().await? {
            Ok(piece) => piece,
            Err(e) => return Some(Err(e)),
        };
        // The piece leaves the queue: its space goes back with the permit.
        self.reservation.shrink(piece.bytes);
        self.baseline.record_output(piece.batch.num_rows());
        Some(Ok(piece.batch))
    }
}

impl Drop for OutputQueue {
    fn drop(&mut self) {
        // Wake senders waiting for space in a queue nobody reads any more.
        self.space.close();
    }
}

/// Spread the batches of one input partition over `outputs` until the input
/// ends or every output is dropped. `first` staggers where each input's
/// round-robin starts.
async fn spread_input(
    mut input: SendableRecordBatchStream,
    outputs: Vec<OutputSender>,
    key_column: usize,
    first: usize,
) -> Result<()> {
    let partitions = outputs.len();
    let mut outputs: Vec<Option<OutputSender>> = outputs.into_iter().map(Some).collect();
    let mut open = partitions;
    let mut next = first;
    let mut until_yield = partitions;
    while open > 0 {
        let Some(batch) = input.next().await.transpose()? else {
            break;
        };
        if batch.num_rows() == 0 {
            continue;
        }
        for (partition, piece) in split_by_source(batch, key_column, partitions, &mut next)? {
            if let Some(output) = &outputs[partition]
                && !output.send(piece).await
            {
                // Dropped early (a limit above, say); stop feeding it.
                outputs[partition] = None;
                open -= 1;
            }
        }
        // Sending rarely waits, so yield now and then lest an input that is
        // always ready starve other tasks (as `RepartitionExec` does).
        if until_yield == 0 {
            tokio::task::yield_now().await;
            until_yield = partitions;
        } else {
            until_yield -= 1;
        }
    }
    Ok(())
}

/// The pieces of `batch` and the output each goes to (see
/// [`RasterSourceSpreadExec`]). `next` is the round-robin position for
/// batches and rows without a source.
fn split_by_source(
    batch: RecordBatch,
    key_column: usize,
    partitions: usize,
    next: &mut usize,
) -> Result<Vec<(usize, RecordBatch)>> {
    let column = batch.column(key_column);
    let rasters = column
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            sedona_internal_datafusion_err!(
                "RasterSourceSpreadExec: expected a Raster (Struct) column at {key_column}, got {:?}",
                column.data_type()
            )
        })?;
    let rasters = RasterStructArray::try_new(rasters)?;
    let mut round_robin = || {
        let partition = *next % partitions;
        *next = next.wrapping_add(1);
        partition
    };
    if !rasters.any_raster_needs_load() {
        return Ok(vec![(round_robin(), batch)]);
    }
    let mut rows: Vec<Vec<u32>> = vec![Vec::new(); partitions];
    for (row, key) in raster_source_keys(&rasters).into_iter().enumerate() {
        let partition = match key {
            Some(key) => (key % partitions as u64) as usize,
            None => round_robin(),
        };
        rows[partition].push(row as u32);
    }
    if let Some(partition) = rows.iter().position(|rows| rows.len() == batch.num_rows()) {
        return Ok(vec![(partition, batch)]);
    }
    rows.into_iter()
        .enumerate()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(partition, rows)| {
            Ok((
                partition,
                take_record_batch(&batch, &UInt32Array::from(rows))?,
            ))
        })
        .collect()
}

impl DisplayAs for RasterSourceSpreadExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let key = format!(
            "raster_source({}@{})",
            self.input.schema().field(self.key_column).name(),
            self.key_column
        );
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "RasterSourceSpreadExec: key={key}, partitions={}, input_partitions={}",
                self.partitions(),
                self.input.output_partitioning().partition_count()
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "key={key}")?;
                writeln!(f, "partitions={}", self.partitions())
            }
        }
    }
}

impl ExecutionPlan for RasterSourceSpreadExec {
    fn name(&self) -> &str {
        "RasterSourceSpreadExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!(
                "RasterSourceSpreadExec expects exactly 1 child, got {}",
                children.len()
            );
        }
        Ok(Arc::new(Self::try_new(
            children.remove(0),
            self.key_column,
            self.partitions(),
        )?))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let queue = {
            let mut outputs = self
                .outputs
                .lock()
                .map_err(|_| sedona_internal_datafusion_err!("RasterSourceSpreadExec: poisoned"))?;
            let outputs = match &mut *outputs {
                Some(outputs) => outputs,
                None => outputs.insert(self.start(&context)?),
            };
            outputs
                .get_mut(partition)
                .and_then(Option::take)
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!(
                        "RasterSourceSpreadExec: partition {partition} executed twice or out of range"
                    )
                })?
        };
        let output = stream::unfold(queue, |mut queue| async move {
            let item = queue.next().await?;
            Some((item, queue))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            output,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::util::pretty::pretty_format_batches;
    use datafusion::datasource::MemTable;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::ScalarUDF;
    use datafusion::physical_optimizer::sanity_checker::SanityCheckPlan;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_expr::Operator;
    use datafusion_physical_plan::displayable;
    use datafusion_physical_plan::limit::GlobalLimitExec;
    use sedona_common::option::{RasterOptions, SedonaOptions};

    use arrow_array::{ArrayRef, Int32Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::functions_aggregate::count::count_udaf;
    use datafusion::physical_optimizer::PhysicalOptimizerRule;
    use datafusion_common::config::ConfigOptions;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::{binary, col, is_not_null, lit};
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
        MockEnsureLoaded, counted_raster_call, ensure_loaded_call, task_context,
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
    fn raster_source_keys_group_rows_to_load_by_file() {
        // Rows 0-3 and 6-9 are OutDb, rows 4-5 InDb, row 10 loaded (its
        // bytes are here, and it still names the file it came from); then a
        // null raster and a raster without bands.
        let mut b = RasterBuilder::new(13);
        for (uri, bytes) in [
            (Some("s3://b/a.tif#band=1"), 0),
            (Some("s3://b/a.tif#band=2"), 0),
            (Some("s3://b/b.tif"), 0),
            (Some("s3://b/a.tif"), 0),
            (None, 4),
            (None, 4),
            (Some("s3://b/x.zarr#var=a"), 0),
            (Some("s3://b/x.zarr#var=b"), 0),
            (Some("s3://b/p.tif#anchor#band=1"), 0),
            (Some("s3://b/p.tif#anchor"), 0),
            (Some("s3://b/a.tif"), 4),
        ] {
            b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 2], None)
                .unwrap();
            b.start_band(StartBandArgs {
                outdb_uri: uri,
                outdb_format: uri.map(|_| "mock"),
                ..StartBandArgs::new(&["y", "x"], &[2, 2], BandDataType::UInt8)
            })
            .unwrap();
            b.band_data_writer().append_value(vec![0u8; bytes]);
            b.finish_band().unwrap();
            b.finish_raster().unwrap();
        }
        b.append_null().unwrap();
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 2], None)
            .unwrap();
        b.finish_raster().unwrap();
        let rasters = b.finish().unwrap();
        let keys = raster_source_keys(&RasterStructArray::try_new(&rasters).unwrap());
        assert_eq!(keys.len(), 13);

        // Every band and every row of one file share a key; other files differ.
        assert!(keys[0].is_some());
        assert_eq!(keys[0], keys[1]);
        assert_eq!(keys[0], keys[3]);
        assert_ne!(keys[0], keys[2]);
        // Only a trailing `#band=N` is stripped: other fragments name
        // different sources, and an earlier `#anchor` stays.
        assert_ne!(keys[6], keys[7]);
        assert_eq!(keys[8], keys[9]);
        // Rows with nothing to load have no key, whatever file they name.
        for idx in [4, 5, 10, 11, 12] {
            assert_eq!(keys[idx], None, "row {idx}");
        }
        // A sliced array reads its own rows, and an empty one yields no keys.
        let sliced = rasters.slice(2, 2);
        assert_eq!(
            raster_source_keys(&RasterStructArray::try_new(&sliced).unwrap()),
            keys[2..4].to_vec()
        );
        let empty = rasters.slice(0, 0);
        assert!(raster_source_keys(&RasterStructArray::try_new(&empty).unwrap()).is_empty());
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

        // Projection ← Split ← RasterSourceSpreadExec ← scan
        assert_eq!(plan.schema(), expected_schema);
        let split = plan
            .downcast_ref::<ProjectionExec>()
            .expect("projection kept")
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the projection");
        assert_eq!(split.sized_columns(), &[1]);
        let spread = split
            .input()
            .downcast_ref::<RasterSourceSpreadExec>()
            .unwrap();
        assert_eq!(spread.partitions(), 4);
        assert_eq!(spread.key_column(), 1);
        assert!(spread.input().downcast_ref::<DataSourceExec>().is_some());
        // Rows without a file are dealt by position, so no distribution is
        // advertised to parents.
        assert!(matches!(
            split.properties().output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        assert!(matches!(
            plan.output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        assert!(plan.output_ordering().is_none());

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
                let rasters = batch.column(1).as_any().downcast_ref::<StructArray>();
                let keys =
                    raster_source_keys(&RasterStructArray::try_new(rasters.unwrap()).unwrap());
                for k in keys.into_iter().map(Option::unwrap) {
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

        // Coalesce ← Projection ← Split ← RasterSourceSpreadExec ← scan
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
        let spread = split
            .input()
            .downcast_ref::<RasterSourceSpreadExec>()
            .unwrap();
        assert_eq!(spread.partitions(), 4);

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
    async fn expression_raster_argument_is_evaluated_once_below_the_spread() {
        // Hashing `RS_FromPath(path)`-style arguments directly would evaluate
        // them twice (key, then load). The argument is hoisted into a column
        // under the spread instead, so it is evaluated once per input
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

        // Projection ← Split ← RasterSourceSpreadExec ← Projection (evaluate
        // the argument) ← scan
        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        assert_eq!(
            top.expr()[2].expr.to_string(),
            "rs_ensureloaded(__ensure_loaded_arg_0@2)"
        );
        let split = top.input().downcast_ref::<RasterBatchSplitExec>().unwrap();
        assert_eq!(split.sized_columns(), &[2]);
        let spread = split
            .input()
            .downcast_ref::<RasterSourceSpreadExec>()
            .unwrap();
        assert_eq!(spread.partitions(), 4);
        assert_eq!(spread.key_column(), 2);
        let below = spread.input().downcast_ref::<ProjectionExec>().unwrap();
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
        // Once per input batch, below the spread; never again above it.
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

        // Coalesce ← Projection ← Split ← RasterSourceSpreadExec ← Projection
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
        let spread = split
            .input()
            .downcast_ref::<RasterSourceSpreadExec>()
            .unwrap();
        assert_eq!(spread.partitions(), 4);
        let below = spread.input().downcast_ref::<ProjectionExec>().unwrap();
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
        let spread = split
            .input()
            .downcast_ref::<RasterSourceSpreadExec>()
            .unwrap();
        assert_eq!(spread.partitions(), 4);
        assert_eq!(spread.key_column(), 2);

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
        assert!(
            split
                .input()
                .downcast_ref::<RasterSourceSpreadExec>()
                .is_some()
        );
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
            let spread = split
                .input()
                .downcast_ref::<RasterSourceSpreadExec>()
                .unwrap();
            assert_eq!(spread.partitions(), partitions, "{name}");
            assert!(Arc::ptr_eq(spread.input(), &hashed), "{name}");
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
                split
                    .input()
                    .downcast_ref::<RasterSourceSpreadExec>()
                    .is_some(),
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

    #[tokio::test]
    async fn batches_with_nothing_to_load_pass_through_whole_without_a_copy() {
        // Three batches of loaded 32 × 32 rasters that still name the file
        // they came from: nothing to load, so each batch goes whole to the
        // next output, and its pixels are the input's buffers.
        let mut batches = Vec::new();
        let mut schema = None;
        for batch_idx in 0..3 {
            let mut b = RasterBuilder::new(8);
            for i in 0..8 {
                b.start_raster_nd(
                    &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                    &["y", "x"],
                    &[32, 32],
                    None,
                )
                .unwrap();
                let uri = format!("s3://b/r{i}.tif");
                b.start_band(StartBandArgs {
                    outdb_uri: Some(&uri),
                    outdb_format: Some("mock"),
                    ..StartBandArgs::new(&["y", "x"], &[32, 32], BandDataType::UInt8)
                })
                .unwrap();
                b.band_data_writer().append_value(vec![i as u8; 1024]);
                b.finish_band().unwrap();
                b.finish_raster().unwrap();
            }
            let rasters: ArrayRef = Arc::new(b.finish().unwrap());
            let raster_field = SedonaType::Raster.to_storage_field("rast", true).unwrap();
            let batch_schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("rast", rasters.data_type().clone(), true)
                    .with_metadata(raster_field.metadata().clone()),
            ]));
            let ids = Arc::new(Int32Array::from_iter_values(
                batch_idx * 8..batch_idx * 8 + 8,
            ));
            batches
                .push(RecordBatch::try_new(Arc::clone(&batch_schema), vec![ids, rasters]).unwrap());
            schema = Some(batch_schema);
        }
        let schema = schema.unwrap();
        let input_buffers: Vec<*const u8> = batches
            .iter()
            .flat_map(|batch| {
                let rasters = batch.column(1).as_any().downcast_ref::<StructArray>();
                RasterStructArray::try_new(rasters.unwrap())
                    .unwrap()
                    .band_data_array()
                    .data_buffers()
                    .iter()
                    .map(|buffer| buffer.as_ptr())
                    .collect::<Vec<_>>()
            })
            .collect();
        let input =
            MemorySourceConfig::try_new_exec(&[batches], Arc::clone(&schema), None).unwrap();
        let spread = RasterSourceSpreadExec::try_new(input, 1, 4).unwrap();
        assert!(matches!(
            spread.properties().output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));

        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let outputs = futures::future::try_join_all(
            (0..4).map(|partition| collect(spread.execute(partition, Arc::clone(&ctx)).unwrap())),
        )
        .await
        .unwrap();
        // One whole batch each to outputs 0, 1 and 2, in round-robin.
        let rows: Vec<Vec<usize>> = outputs
            .iter()
            .map(|batches| batches.iter().map(RecordBatch::num_rows).collect())
            .collect();
        assert_eq!(rows, vec![vec![8], vec![8], vec![8], vec![]]);
        for batch in outputs.iter().flatten() {
            let rasters = batch.column(1).as_any().downcast_ref::<StructArray>();
            let rasters = RasterStructArray::try_new(rasters.unwrap()).unwrap();
            for buffer in rasters.band_data_array().data_buffers() {
                assert!(input_buffers.contains(&buffer.as_ptr()));
            }
        }
    }

    #[tokio::test]
    async fn mixed_batches_keep_sources_together_and_share_loaded_pixels() {
        // Rows 0-7 are OutDb over four files, rows 8-15 loaded (with the
        // URI they came from): the OutDb rows go by file, the loaded ones
        // round-robin, and every piece is a `take` over the same buffers.
        let mut b = RasterBuilder::new(16);
        for i in 0..16 {
            b.start_raster_nd(
                &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                &["y", "x"],
                &[32, 32],
                None,
            )
            .unwrap();
            let uri = format!("s3://b/r{}.tif#band={}", i % 4, i / 4 % 2 + 1);
            b.start_band(StartBandArgs {
                outdb_uri: Some(&uri),
                outdb_format: Some("mock"),
                ..StartBandArgs::new(&["y", "x"], &[32, 32], BandDataType::UInt8)
            })
            .unwrap();
            b.band_data_writer()
                .append_value(vec![i as u8; if i < 8 { 0 } else { 1024 }]);
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
        let ids = Arc::new(Int32Array::from_iter_values(0..16));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, rasters]).unwrap();
        let input_buffers: Vec<*const u8> = {
            let rasters = batch.column(1).as_any().downcast_ref::<StructArray>();
            RasterStructArray::try_new(rasters.unwrap())
                .unwrap()
                .band_data_array()
                .data_buffers()
                .iter()
                .map(|buffer| buffer.as_ptr())
                .collect()
        };
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let spread = RasterSourceSpreadExec::try_new(input, 1, 4).unwrap();

        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let outputs = futures::future::try_join_all(
            (0..4).map(|partition| collect(spread.execute(partition, Arc::clone(&ctx)).unwrap())),
        )
        .await
        .unwrap();
        let mut partition_of = HashMap::new();
        let mut ids = Vec::new();
        let mut loaded_per_partition = vec![0; 4];
        for (partition, batches) in outputs.iter().enumerate() {
            for batch in batches {
                let rasters = batch.column(1).as_any().downcast_ref::<StructArray>();
                let rasters = RasterStructArray::try_new(rasters.unwrap()).unwrap();
                for buffer in rasters.band_data_array().data_buffers() {
                    assert!(input_buffers.contains(&buffer.as_ptr()));
                }
                let row_ids = batch.column(0).as_any().downcast_ref::<Int32Array>();
                for (id, key) in row_ids
                    .unwrap()
                    .values()
                    .iter()
                    .zip(raster_source_keys(&rasters))
                {
                    ids.push(*id);
                    match key {
                        Some(key) => {
                            assert!(*id < 8);
                            assert_eq!(*partition_of.entry(key).or_insert(partition), partition);
                        }
                        None => {
                            assert!(*id >= 8);
                            loaded_per_partition[partition] += 1;
                        }
                    }
                }
            }
        }
        ids.sort();
        assert_eq!(ids, (0..16).collect::<Vec<_>>());
        assert_eq!(partition_of.len(), 4);
        assert_eq!(loaded_per_partition, vec![2; 4]);
    }

    #[tokio::test]
    async fn queued_bytes_are_bounded_and_dropped_outputs_stop_the_input() {
        // 64 one-row batches of a loaded 1 KiB raster under a 1 KiB budget.
        // `RepartitionExec` gathered all 64 before emitting one; here each
        // output queues one piece, and the input waits.
        let mut b = RasterBuilder::new(64);
        for i in 0..64 {
            b.start_raster_nd(
                &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                &["y", "x"],
                &[32, 32],
                None,
            )
            .unwrap();
            b.start_band(StartBandArgs::new(
                &["y", "x"],
                &[32, 32],
                BandDataType::UInt8,
            ))
            .unwrap();
            b.band_data_writer().append_value(vec![i as u8; 1024]);
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
        let ids = Arc::new(Int32Array::from_iter_values(0..64));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, rasters]).unwrap();
        let one_row_batches: Vec<RecordBatch> = (0..64).map(|i| batch.slice(i, 1)).collect();
        let input = MemorySourceConfig::try_new_exec(&[one_row_batches], Arc::clone(&schema), None)
            .unwrap();
        // Count the batches pulled from the input as they are evaluated.
        let pulled = Arc::new(AtomicUsize::new(0));
        let input: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(col("id", &schema).unwrap(), "id"),
                    ProjectionExpr::new(
                        counted_raster_call(&pulled, col("rast", &schema).unwrap(), &schema),
                        "rast",
                    ),
                ],
                input,
            )
            .unwrap(),
        );
        let spread: Arc<dyn ExecutionPlan> =
            Arc::new(RasterSourceSpreadExec::try_new(input, 1, 4).unwrap());
        let ctx = task_context(1024);

        let mut streams: Vec<_> = (0..4)
            .map(|partition| spread.execute(partition, Arc::clone(&ctx)).unwrap())
            .collect();
        // Every output is polled once (as a `CoalescePartitionsExec` would),
        // then only output 0 is read.
        for stream in &mut streams[1..] {
            assert!(stream.next().now_or_never().is_none());
        }
        let first = streams[0].next().await.unwrap().unwrap();
        assert_eq!(first.num_rows(), 1);
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        // Four queued (one per output), one waiting for output 0's space,
        // and the one output 0 took.
        let before_drop = pulled.load(Ordering::SeqCst);
        assert!(before_drop <= 8, "pulled {before_drop} of 64 batches");

        // Dropping every output stops the input task, also while it waits
        // for space.
        drop(streams);
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert_eq!(pulled.load(Ordering::SeqCst), before_drop);

        // Drained concurrently, every piece arrives as it was cut.
        let spread = Arc::clone(&spread).reset_state().unwrap();
        let outputs = futures::future::try_join_all(
            (0..4).map(|partition| collect(spread.execute(partition, Arc::clone(&ctx)).unwrap())),
        )
        .await
        .unwrap();
        let rows: Vec<usize> = outputs
            .iter()
            .flatten()
            .map(RecordBatch::num_rows)
            .collect();
        assert_eq!(rows, vec![1; 64]);
    }

    #[tokio::test]
    async fn a_memory_limit_slows_the_spread_without_failing_it() {
        // A 4 KiB pool and no batch budget: once the pool refuses a 1 KiB
        // piece, the input waits for that output's queue to empty and goes
        // on, one piece at a time, instead of erroring.
        let mut b = RasterBuilder::new(64);
        for i in 0..64 {
            b.start_raster_nd(
                &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                &["y", "x"],
                &[32, 32],
                None,
            )
            .unwrap();
            b.start_band(StartBandArgs::new(
                &["y", "x"],
                &[32, 32],
                BandDataType::UInt8,
            ))
            .unwrap();
            b.band_data_writer().append_value(vec![i as u8; 1024]);
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
        let ids = Arc::new(Int32Array::from_iter_values(0..64));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, rasters]).unwrap();
        let one_row_batches: Vec<RecordBatch> = (0..64).map(|i| batch.slice(i, 1)).collect();
        let input = MemorySourceConfig::try_new_exec(&[one_row_batches], Arc::clone(&schema), None)
            .unwrap();
        let spread = RasterSourceSpreadExec::try_new(input, 1, 4).unwrap();
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4096));
        let config = SessionConfig::new().with_option_extension(SedonaOptions {
            raster: RasterOptions {
                max_batch_bytes: 0,
                ..Default::default()
            },
            ..Default::default()
        });
        let ctx = SessionStateBuilder::new()
            .with_config(config)
            .with_runtime_env(
                RuntimeEnvBuilder::new()
                    .with_memory_pool(Arc::clone(&pool))
                    .build_arc()
                    .unwrap(),
            )
            .build()
            .task_ctx();

        let outputs = futures::future::try_join_all(
            (0..4).map(|partition| collect(spread.execute(partition, Arc::clone(&ctx)).unwrap())),
        )
        .await
        .unwrap();
        assert_eq!(outputs.iter().flatten().count(), 64);
        drop(outputs);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn empty_batches_are_skipped_and_input_errors_reach_every_output() {
        let (schema, batch) = outdb_raster_batch(&["s3://b/a.tif", "s3://b/b.tif"]);
        let input = MemorySourceConfig::try_new_exec(
            &[
                vec![batch.slice(0, 0), batch.clone()],
                vec![batch.slice(0, 0)],
            ],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let spread = RasterSourceSpreadExec::try_new(input, 1, 4).unwrap();
        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let outputs = futures::future::try_join_all(
            (0..4).map(|partition| collect(spread.execute(partition, Arc::clone(&ctx)).unwrap())),
        )
        .await
        .unwrap();
        let batches: Vec<&RecordBatch> = outputs.iter().flatten().collect();
        assert!(batches.iter().all(|batch| batch.num_rows() > 0));
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 2);

        // `id / 0` fails on the second input partition only.
        let input = MemorySourceConfig::try_new_exec(
            &[vec![batch.clone()], vec![batch]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let failing: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(
                        binary(
                            col("id", &schema).unwrap(),
                            Operator::Divide,
                            lit(0i32),
                            &schema,
                        )
                        .unwrap(),
                        "id",
                    ),
                    ProjectionExpr::new(col("rast", &schema).unwrap(), "rast"),
                ],
                input,
            )
            .unwrap(),
        );
        let spread = RasterSourceSpreadExec::try_new(failing, 1, 4).unwrap();
        let results = futures::future::join_all(
            (0..4).map(|partition| collect(spread.execute(partition, Arc::clone(&ctx)).unwrap())),
        )
        .await;
        for result in results {
            let error = result.expect_err("every output sees the error").to_string();
            assert!(error.contains("Divide by zero"), "{error}");
        }
    }

    #[tokio::test]
    async fn a_single_partition_consumer_above_a_spread_still_gets_one_partition() {
        // `GlobalLimitExec` and a non-preserving `SortExec` require a single
        // input partition. Over one input partition the spread is merged back
        // below them; over two, the planner's merge is kept above the node.
        // `SanityCheckPlan` runs right after the rule, without re-running
        // distribution or sorting enforcement that could repair the plan.
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{}.tif", i % 4)).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, batch) = outdb_raster_batch(&uris);
        let one_partition =
            MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                .unwrap();
        let two_partitions = MemorySourceConfig::try_new_exec(
            &[vec![batch.slice(0, 8)], vec![batch.slice(8, 8)]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let loading = |input: Arc<dyn ExecutionPlan>| {
            loading_projection(
                ensure_loaded_call(
                    &Arc::new(Mutex::new(Vec::new())),
                    col("rast", &schema).unwrap(),
                    &schema,
                ),
                input,
            )
        };
        let by_id =
            LexOrdering::new([PhysicalSortExpr::new_default(col("id", &schema).unwrap())]).unwrap();
        let cases: Vec<(&str, Arc<dyn ExecutionPlan>, bool)> = vec![
            (
                "limit with offset over one partition",
                Arc::new(GlobalLimitExec::new(
                    loading(one_partition.clone()),
                    2,
                    Some(5),
                )),
                false,
            ),
            (
                "limit with offset over a merge of two partitions",
                Arc::new(GlobalLimitExec::new(
                    Arc::new(CoalescePartitionsExec::new(loading(two_partitions.clone()))),
                    2,
                    Some(5),
                )),
                false,
            ),
            (
                "ordered limit with offset over one partition",
                Arc::new(GlobalLimitExec::new(
                    Arc::new(SortExec::new(by_id.clone(), loading(one_partition))),
                    2,
                    Some(5),
                )),
                true,
            ),
            (
                "ordered limit with offset over a merge of two partitions",
                Arc::new(GlobalLimitExec::new(
                    Arc::new(SortExec::new(
                        by_id,
                        Arc::new(CoalescePartitionsExec::new(loading(two_partitions))),
                    )),
                    2,
                    Some(5),
                )),
                true,
            ),
        ];
        for (name, plan, ordered) in cases {
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = 4;
            let rewritten = RasterMaterializationRule
                .optimize(Arc::clone(&plan), &config)
                .unwrap();
            let rewritten = SanityCheckPlan::new()
                .optimize(rewritten, &config)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let shown = displayable(rewritten.as_ref()).indent(true).to_string();
            assert!(shown.contains("raster_source(rast@1)"), "{name}: {shown}");
            assert_eq!(
                rewritten.output_partitioning().partition_count(),
                1,
                "{name}"
            );

            let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
            let ids = |batches: Vec<RecordBatch>| -> Vec<i32> {
                batches
                    .iter()
                    .flat_map(|b| {
                        b.column(0)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .unwrap()
                            .values()
                            .to_vec()
                    })
                    .collect()
            };
            let got = ids(collect(rewritten.execute(0, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap());
            let before = ids(collect(plan.execute(0, ctx).unwrap()).await.unwrap());
            assert_eq!(got.len(), 5, "{name}");
            assert_eq!(before.len(), 5, "{name}");
            if ordered {
                assert_eq!(got, (2..7).collect::<Vec<_>>(), "{name}");
                assert_eq!(before, got, "{name}");
            }
        }
    }

    #[tokio::test]
    async fn sql_plans_keep_parent_requirements_and_results() {
        // Queries in the shapes of DataFusion's own distribution, sorting,
        // limit and sanity-checker tests, planned by a session that runs the
        // rule after DataFusion's physical optimizer and `SanityCheckPlan`
        // after the rule, as `SedonaContext` does. A rewrite that broke a
        // parent's ordering or distribution requirement fails planning.
        // Results are compared with the same session without the rule.
        let mut b = RasterBuilder::new(32);
        for i in 0..32 {
            b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 2], None)
                .unwrap();
            let uri = format!("s3://b/r{}.tif", i % 8);
            b.start_band(StartBandArgs {
                outdb_uri: Some(&uri),
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
            Field::new("grp", DataType::Int32, false),
            Field::new("rast", rasters.data_type().clone(), true)
                .with_metadata(raster_field.metadata().clone()),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(0..32)),
                Arc::new(Int32Array::from_iter_values((0..32).map(|i| i / 8))),
                rasters,
            ],
        )
        .unwrap();
        let halves = vec![vec![batch.slice(0, 16)], vec![batch.slice(16, 16)]];

        let mut contexts = Vec::new();
        for with_rule in [true, false] {
            // Partitioned joins even for these small inputs, and
            // order-preserving repartitions over the sorted table.
            let config = SessionConfig::new()
                .with_target_partitions(4)
                .set_usize(
                    "datafusion.optimizer.hash_join_single_partition_threshold",
                    0,
                )
                .set_usize(
                    "datafusion.optimizer.hash_join_single_partition_threshold_rows",
                    0,
                )
                .set_bool("datafusion.optimizer.prefer_existing_sort", true);
            let mut state = SessionStateBuilder::new()
                .with_config(config)
                .with_default_features();
            if with_rule {
                state = state
                    .with_physical_optimizer_rule(Arc::new(RasterMaterializationRule))
                    .with_physical_optimizer_rule(Arc::new(SanityCheckPlan::new()));
            }
            let ctx = SessionContext::new_with_state(state.build());
            ctx.register_udf(ScalarUDF::new_from_impl(MockEnsureLoaded::new(Arc::new(
                Mutex::new(Vec::new()),
            ))));
            ctx.register_table(
                "t",
                Arc::new(MemTable::try_new(Arc::clone(&schema), halves.clone()).unwrap()),
            )
            .unwrap();
            ctx.register_table(
                "t1",
                Arc::new(
                    MemTable::try_new(Arc::clone(&schema), vec![vec![batch.clone()]]).unwrap(),
                ),
            )
            .unwrap();
            ctx.register_table(
                "ts",
                Arc::new(
                    MemTable::try_new(Arc::clone(&schema), halves.clone())
                        .unwrap()
                        .with_sort_order(vec![vec![
                            datafusion::prelude::col("grp").sort(true, false),
                            datafusion::prelude::col("id").sort(true, false),
                        ]]),
                ),
            )
            .unwrap();
            contexts.push(ctx);
        }

        // (name, query, whether the rule spreads somewhere in it, whether the
        // rows are fully determined; a LIMIT without ORDER BY fixes only the
        // count).
        let cases = [
            (
                "ordered limit with offset over a spread",
                "SELECT id, rs_ensureloaded(rast) AS l FROM t ORDER BY id LIMIT 5 OFFSET 2",
                true,
                true,
            ),
            (
                "hash join on a key aliased by the loading projection",
                "SELECT a.k, b.grp, a.l FROM \
                 (SELECT t.id AS k, rs_ensureloaded(t.rast) AS l FROM t JOIN t1 ON t.id = t1.id) a \
                 JOIN t b ON a.k = b.id",
                false,
                true,
            ),
            (
                "window over sorted input",
                "SELECT id, row_number() OVER (PARTITION BY grp ORDER BY id) AS rn, l \
                 FROM (SELECT id, grp, rs_ensureloaded(rast) AS l FROM ts)",
                false,
                true,
            ),
            (
                "window over a loading filter on sorted input",
                "SELECT id, count(*) OVER (ORDER BY grp, id) AS running FROM ts \
                 WHERE rs_ensureloaded(rast) IS NOT NULL",
                false,
                true,
            ),
            (
                "aggregate over input sorted on the group key",
                "SELECT grp, count(rs_ensureloaded(rast)) FROM ts GROUP BY grp",
                false,
                true,
            ),
            (
                "aggregate over input sorted on part of the group key",
                "SELECT grp, id % 3 AS m, count(rs_ensureloaded(rast)) FROM ts GROUP BY grp, id % 3",
                false,
                true,
            ),
            (
                "limit pushed into a loading filter and its merge",
                "SELECT id FROM t WHERE rs_ensureloaded(rast) IS NOT NULL LIMIT 3",
                false,
                false,
            ),
            (
                "hash-partitioned union branches under an interleave",
                "SELECT k, sum(c) FROM ( \
                 SELECT id AS k, count(rs_ensureloaded(rast)) AS c FROM t GROUP BY id \
                 UNION ALL SELECT id, count(rs_ensureloaded(rast)) FROM t1 GROUP BY id) \
                 GROUP BY k",
                true,
                true,
            ),
        ];
        for (name, sql, spreads, exact) in cases {
            let mut results = Vec::new();
            for ctx in &contexts {
                let df = ctx.sql(sql).await.unwrap();
                let plan = df
                    .clone()
                    .create_physical_plan()
                    .await
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                results.push((
                    displayable(plan.as_ref()).indent(true).to_string(),
                    df.collect().await.unwrap(),
                ));
            }
            let (shown, got) = &results[0];
            let (_, expected) = &results[1];
            assert!(shown.contains("RasterBatchSplitExec"), "{name}: {shown}");
            assert_eq!(shown.contains("raster_source("), spreads, "{name}: {shown}");
            let rows =
                |batches: &[RecordBatch]| batches.iter().map(|b| b.num_rows()).sum::<usize>();
            assert_eq!(rows(got), rows(expected), "{name}");
            if exact {
                let sorted_lines = |batches: &[RecordBatch]| {
                    let mut lines: Vec<String> = pretty_format_batches(batches)
                        .unwrap()
                        .to_string()
                        .lines()
                        .map(str::to_string)
                        .collect();
                    lines.sort();
                    lines
                };
                assert_eq!(sorted_lines(got), sorted_lines(expected), "{name}");
            }
        }
    }
}
