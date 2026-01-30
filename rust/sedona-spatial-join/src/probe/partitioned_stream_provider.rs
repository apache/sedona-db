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

use std::ops::DerefMut;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::config::SpillCompression;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::runtime_env::RuntimeEnv;
use parking_lot::Mutex;

use crate::probe::first_pass_stream::FirstPassStream;
use crate::probe::non_partitioned_stream::NonPartitionedStream;
use crate::probe::ProbeStreamMetrics;
use crate::{
    evaluated_batch::evaluated_batch_stream::{
        external::ExternalEvaluatedBatchStream, SendableEvaluatedBatchStream,
    },
    partitioning::{
        stream_repartitioner::{SpilledPartitions, StreamRepartitioner},
        PartitionedSide, SpatialPartition, SpatialPartitioner,
    },
};

#[derive(Clone)]
pub(crate) struct ProbeStreamOptions {
    pub partitioner: Option<Arc<dyn SpatialPartitioner>>,
    pub target_batch_rows: usize,
    pub spill_compression: SpillCompression,
    pub buffer_bytes_threshold: usize,
    pub spilled_batch_in_memory_size_threshold: Option<usize>,
}

pub(crate) struct PartitionedProbeStreamProvider {
    state: Arc<Mutex<ProbeStreamState>>,
    runtime_env: Arc<RuntimeEnv>,
    options: ProbeStreamOptions,
    schema: SchemaRef,
    metrics: ProbeStreamMetrics,
}

enum ProbeStreamState {
    Pending {
        source: SendableEvaluatedBatchStream,
    },
    FirstPass,
    SubsequentPass {
        manifest: ProbePartitionManifest,
    },
    NonPartitionedConsumed,
    Failed(Arc<DataFusionError>),
}

impl PartitionedProbeStreamProvider {
    pub fn new(
        runtime_env: Arc<RuntimeEnv>,
        options: ProbeStreamOptions,
        source: SendableEvaluatedBatchStream,
        metrics: ProbeStreamMetrics,
    ) -> Self {
        let schema = source.schema();
        Self {
            state: Arc::new(Mutex::new(ProbeStreamState::Pending { source })),
            runtime_env,
            options,
            schema,
            metrics,
        }
    }

    pub fn stream_for(&self, partition: SpatialPartition) -> Result<SendableEvaluatedBatchStream> {
        match partition {
            SpatialPartition::None => Err(DataFusionError::Execution(
                "SpatialPartition::None should be handled via outer join logic".into(),
            )),
            SpatialPartition::Regular(0) => self.first_pass_stream(),
            SpatialPartition::Regular(_) | SpatialPartition::Multi => {
                if self.options.partitioner.is_none() {
                    Err(DataFusionError::Execution(
                        "Non-partitioned probe stream only supports Regular(0)".into(),
                    ))
                } else {
                    self.subsequent_pass_stream(partition)
                }
            }
        }
    }

    fn first_pass_stream(&self) -> Result<SendableEvaluatedBatchStream> {
        if self.options.partitioner.is_none() {
            return self.non_partitioned_first_pass_stream();
        }

        let schema = Arc::clone(&self.schema);
        let mut state_guard = self.state.lock();
        match std::mem::replace(&mut *state_guard, ProbeStreamState::FirstPass) {
            ProbeStreamState::Pending { source } => {
                let partitioner = Arc::clone(
                    self.options
                        .partitioner
                        .as_ref()
                        .expect("Partitioned first pass requires a partitioner"),
                );
                let repartitioner = StreamRepartitioner::builder(
                    Arc::clone(&self.runtime_env),
                    Arc::clone(&partitioner),
                    PartitionedSide::ProbeSide,
                    self.metrics.spill_metrics.clone(),
                )
                .spill_compression(self.options.spill_compression)
                .buffer_bytes_threshold(self.options.buffer_bytes_threshold)
                .target_batch_size(self.options.target_batch_rows)
                .spilled_batch_in_memory_size_threshold(
                    self.options.spilled_batch_in_memory_size_threshold,
                )
                .build();

                let state = Arc::clone(&self.state);
                let callback = move |res: Result<SpilledPartitions>| {
                    let mut guard = state.lock();
                    *guard = match res {
                        Ok(mut spills) => {
                            let mut s = String::new();
                            if spills.debug_print(&mut s).is_ok() {
                                log::debug!("Probe side spilled partitions:\n{}", s);
                            }

                            // Sanity check: Regular(0) and None should be empty
                            let mut check_empty = |partition: SpatialPartition| -> Result<()> {
                                let spilled = spills.take_spilled_partition(partition)?;
                                if !spilled.into_spill_files().is_empty() {
                                    return Err(DataFusionError::Execution(format!(
                                        "{:?} partition should not have spilled data",
                                        partition
                                    )));
                                }
                                Ok(())
                            };

                            match check_empty(SpatialPartition::Regular(0))
                                .and_then(|_| check_empty(SpatialPartition::None))
                            {
                                Ok(_) => ProbeStreamState::SubsequentPass {
                                    manifest: ProbePartitionManifest::new(schema, spills),
                                },
                                Err(err) => ProbeStreamState::Failed(Arc::new(err)),
                            }
                        }
                        Err(err) => {
                            let err_arc = Arc::new(err);
                            ProbeStreamState::Failed(Arc::clone(&err_arc))
                        }
                    };
                    Ok(())
                };

                let first_pass = FirstPassStream::new(
                    source,
                    repartitioner,
                    partitioner,
                    self.metrics.clone(),
                    callback,
                );
                Ok(Box::pin(first_pass))
            }
            ProbeStreamState::FirstPass => Err(DataFusionError::Execution(
                "First pass already running for partitioned probe stream".into(),
            )),
            ProbeStreamState::SubsequentPass { .. } => Err(DataFusionError::Execution(
                "First pass already completed".into(),
            )),
            ProbeStreamState::NonPartitionedConsumed => Err(DataFusionError::Execution(
                "Non-partitioned probe stream already consumed".into(),
            )),
            ProbeStreamState::Failed(err) => Err(DataFusionError::Shared(err)),
        }
    }

    fn non_partitioned_first_pass_stream(&self) -> Result<SendableEvaluatedBatchStream> {
        let mut state_guard = self.state.lock();
        match std::mem::replace(&mut *state_guard, ProbeStreamState::NonPartitionedConsumed) {
            ProbeStreamState::Pending { source } => Ok(Box::pin(NonPartitionedStream::new(
                source,
                self.metrics.clone(),
            ))),
            ProbeStreamState::NonPartitionedConsumed => Err(DataFusionError::Execution(
                "Non-partitioned probe stream already consumed".into(),
            )),
            ProbeStreamState::Failed(err) => Err(DataFusionError::Shared(err)),
            _ => Err(DataFusionError::Execution(
                "Non-partitioned probe stream is not available".into(),
            )),
        }
    }

    fn subsequent_pass_stream(
        &self,
        partition: SpatialPartition,
    ) -> Result<SendableEvaluatedBatchStream> {
        if self.options.partitioner.is_none() {
            return Err(DataFusionError::Execution(
                "Non-partitioned probe stream cannot serve additional partitions".into(),
            ));
        }
        let mut locked = self.state.lock();
        let manifest = match locked.deref_mut() {
            ProbeStreamState::SubsequentPass { manifest } => manifest,
            ProbeStreamState::Failed(err) => return Err(DataFusionError::Shared(Arc::clone(err))),
            _ => {
                return Err(DataFusionError::Execution(
                    "Partitioned probe stream warm-up not finished".into(),
                ))
            }
        };

        {
            // let mut manifest = manifest.lock();
            manifest.stream_for(partition)
        }
    }

    pub fn get_partition_row_count(&self, partition: SpatialPartition) -> Result<usize> {
        let mut locked = self.state.lock();
        let manifest = match locked.deref_mut() {
            ProbeStreamState::SubsequentPass { manifest } => manifest,
            ProbeStreamState::Failed(err) => return Err(DataFusionError::Shared(Arc::clone(err))),
            _ => {
                return Err(DataFusionError::Execution(
                    "Partitioned probe stream warm-up not finished".into(),
                ))
            }
        };
        manifest.get_partition_row_count(partition)
    }
}

pub struct ProbePartitionManifest {
    schema: SchemaRef,
    slots: SpilledPartitions,
}

impl ProbePartitionManifest {
    fn new(schema: SchemaRef, spills: SpilledPartitions) -> Self {
        Self {
            schema,
            slots: spills,
        }
    }

    fn get_partition_row_count(&self, partition: SpatialPartition) -> Result<usize> {
        let spilled = self.slots.get_spilled_partition(partition)?;
        Ok(spilled.num_rows())
    }

    fn stream_for(&mut self, partition: SpatialPartition) -> Result<SendableEvaluatedBatchStream> {
        match partition {
            SpatialPartition::Regular(0) => Err(DataFusionError::Execution(
                "Partition 0 is only available during the first pass".into(),
            )),
            SpatialPartition::None => Err(DataFusionError::Execution(
                "SpatialPartition::None should not request a probe stream".into(),
            )),
            SpatialPartition::Regular(_) => {
                let spilled = self.slots.take_spilled_partition(partition)?;
                Ok(Box::pin(
                    ExternalEvaluatedBatchStream::try_from_spill_files(
                        Arc::clone(&self.schema),
                        spilled.into_spill_files(),
                    )?,
                ))
            }
            SpatialPartition::Multi => {
                let spilled = self.slots.get_spilled_partition(partition)?;
                Ok(Box::pin(
                    ExternalEvaluatedBatchStream::try_from_spill_files(
                        Arc::clone(&self.schema),
                        spilled.into_spill_files(),
                    )?,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluated_batch::EvaluatedBatch;
    use crate::operand_evaluator::EvaluatedGeometryArray;
    use crate::partitioning::flat::FlatPartitioner;
    use crate::{
        evaluated_batch::evaluated_batch_stream::in_mem::InMemoryEvaluatedBatchStream,
        probe::ProbeStreamMetrics,
    };
    use arrow_array::{ArrayRef, BinaryArray, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::config::SpillCompression;
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
    use futures::TryStreamExt;
    use sedona_geometry::bounding_box::BoundingBox;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use std::sync::Arc;

    fn sample_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
    }

    fn sample_batch(ids: &[i32], wkbs: Vec<Option<Vec<u8>>>) -> Result<EvaluatedBatch> {
        assert_eq!(ids.len(), wkbs.len());
        let id_array = Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef;
        let batch = RecordBatch::try_new(sample_schema(), vec![id_array])?;
        let geom_values: Vec<Option<&[u8]>> = wkbs
            .iter()
            .map(|maybe_wkb| maybe_wkb.as_ref().map(|buf| buf.as_slice()))
            .collect();
        let geom_array: ArrayRef = Arc::new(BinaryArray::from(geom_values));
        let geom_array = EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY)?;
        Ok(EvaluatedBatch { batch, geom_array })
    }

    fn point_wkb(x: f64, y: f64) -> Vec<u8> {
        let mut buf = vec![1u8, 1, 0, 0, 0];
        buf.extend_from_slice(&x.to_le_bytes());
        buf.extend_from_slice(&y.to_le_bytes());
        buf
    }

    fn rect_wkb(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1 + 4 + 4 + 4 + 5 * 16);
        buf.push(1u8);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&5u32.to_le_bytes());
        let coords = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ];
        for (x, y) in coords {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
        }
        buf
    }

    fn ids_from_batches(batches: &[EvaluatedBatch]) -> Vec<Vec<i32>> {
        batches
            .iter()
            .map(|batch| {
                let array = batch
                    .batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                (0..array.len())
                    .map(|idx| array.value(idx))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn create_probe_stream(
        batches: Vec<EvaluatedBatch>,
        partitioner: Option<Arc<dyn SpatialPartitioner>>,
    ) -> PartitionedProbeStreamProvider {
        let runtime_env = Arc::new(RuntimeEnv::default());
        assert!(!batches.is_empty(), "test batches should not be empty");
        let schema = batches[0].schema();
        let stream: SendableEvaluatedBatchStream =
            Box::pin(InMemoryEvaluatedBatchStream::new(schema, batches));
        PartitionedProbeStreamProvider::new(
            runtime_env,
            ProbeStreamOptions {
                partitioner,
                target_batch_rows: 1024,
                spill_compression: SpillCompression::Uncompressed,
                buffer_bytes_threshold: 0,
                spilled_batch_in_memory_size_threshold: None,
            },
            stream,
            ProbeStreamMetrics::new(0, &ExecutionPlanMetricsSet::new()),
        )
    }

    fn sample_partitioner() -> Result<Arc<dyn SpatialPartitioner>> {
        let partitions = vec![
            BoundingBox::xy((0.0, 50.0), (0.0, 50.0)),
            BoundingBox::xy((50.0, 100.0), (0.0, 50.0)),
        ];
        Ok(Arc::new(FlatPartitioner::try_new(partitions)?))
    }

    #[tokio::test]
    async fn first_pass_emits_partition_zero_and_spills_rest() -> Result<()> {
        let partitioner = sample_partitioner()?;
        let batch = sample_batch(
            &[0, 1, 2, 3],
            vec![
                Some(point_wkb(10.0, 10.0)),
                Some(point_wkb(60.0, 10.0)),
                Some(point_wkb(70.0, 20.0)),
                Some(point_wkb(200.0, 200.0)),
            ],
        )?;
        let probe_stream = create_probe_stream(vec![batch], Some(Arc::clone(&partitioner)));

        let first_pass = probe_stream.stream_for(SpatialPartition::Regular(0))?;
        let batches = first_pass.try_collect::<Vec<_>>().await?;
        assert_eq!(ids_from_batches(&batches), vec![vec![0, 3]]);

        let regular_one = probe_stream.stream_for(SpatialPartition::Regular(1))?;
        let replay_batches = regular_one.try_collect::<Vec<_>>().await?;
        assert_eq!(ids_from_batches(&replay_batches), vec![vec![1, 2]]);

        assert!(probe_stream
            .stream_for(SpatialPartition::Regular(1))
            .is_err());
        assert!(probe_stream
            .stream_for(SpatialPartition::Regular(0))
            .is_err());
        Ok(())
    }

    #[tokio::test]
    async fn requesting_regular_partition_before_first_pass_fails() -> Result<()> {
        let partitioner = sample_partitioner()?;
        let batch = sample_batch(&[0], vec![Some(point_wkb(60.0, 10.0))])?;
        let probe_stream = create_probe_stream(vec![batch], Some(Arc::clone(&partitioner)));
        assert!(probe_stream
            .stream_for(SpatialPartition::Regular(1))
            .is_err());

        // After running first pass, subsequent requests should succeed once.
        let first_pass = probe_stream.stream_for(SpatialPartition::Regular(0))?;
        first_pass.try_collect::<Vec<_>>().await?;
        let replay = probe_stream.stream_for(SpatialPartition::Regular(1))?;
        let batches = replay.try_collect::<Vec<_>>().await?;
        assert_eq!(ids_from_batches(&batches), vec![vec![0]]);
        Ok(())
    }

    #[tokio::test]
    async fn multi_partition_stream_consumed_multiple_times() -> Result<()> {
        let partitioner = sample_partitioner()?;
        let batch = sample_batch(&[0], vec![Some(rect_wkb(25.0, 0.0, 75.0, 20.0))])?;
        let probe_stream = create_probe_stream(vec![batch], Some(partitioner));

        let first_pass = probe_stream.stream_for(SpatialPartition::Regular(0))?;
        first_pass.try_collect::<Vec<_>>().await?;

        let multi_stream = probe_stream.stream_for(SpatialPartition::Multi)?;
        let multi_batches = multi_stream.try_collect::<Vec<_>>().await?;
        assert_eq!(ids_from_batches(&multi_batches), vec![vec![0]]);

        let multi_stream_2 = probe_stream.stream_for(SpatialPartition::Multi)?;
        let multi_batches_2 = multi_stream_2.try_collect::<Vec<_>>().await?;
        assert_eq!(ids_from_batches(&multi_batches_2), vec![vec![0]]);
        Ok(())
    }

    #[tokio::test]
    async fn non_partitioned_stream_exposes_only_partition_zero() -> Result<()> {
        let batch = sample_batch(
            &[0, 1],
            vec![Some(point_wkb(10.0, 10.0)), Some(point_wkb(12.0, 12.0))],
        )?;
        let probe_stream = create_probe_stream(vec![batch], None);

        let first_pass = probe_stream.stream_for(SpatialPartition::Regular(0))?;
        let batches = first_pass.try_collect::<Vec<_>>().await?;
        assert_eq!(ids_from_batches(&batches), vec![vec![0, 1]]);

        assert!(probe_stream
            .stream_for(SpatialPartition::Regular(0))
            .is_err());
        assert!(probe_stream
            .stream_for(SpatialPartition::Regular(1))
            .is_err());
        assert!(probe_stream.stream_for(SpatialPartition::Multi).is_err());
        Ok(())
    }
}
