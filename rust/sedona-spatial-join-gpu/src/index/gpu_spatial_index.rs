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

use crate::options::GpuOptions;
use arrow::array::BooleanBufferBuilder;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::{DataFusionError, Result};
use parking_lot::Mutex;
use sedona_common::ExecutionMode;
use sedona_expr::statistics::GeoStatistics;
use sedona_libgpuspatial::{
    GpuSpatialIndex, GpuSpatialOptions, GpuSpatialRefiner, GpuSpatialRelationPredicate,
};
use sedona_spatial_join::evaluated_batch::EvaluatedBatch;
use sedona_spatial_join::index::spatial_index::SpatialIndex;
use sedona_spatial_join::index::QueryResultMetrics;
use sedona_spatial_join::spatial_predicate::SpatialRelationType;
use sedona_spatial_join::SpatialPredicate;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wkb::reader::Wkb;

pub struct GPUSpatialIndex {
    pub(crate) schema: SchemaRef,
    /// GPU spatial index for performing GPU-accelerated filtering
    pub(crate) index: Arc<GpuSpatialIndex>,
    /// GPU spatial refiner for performing GPU-accelerated refinement
    pub(crate) refiner: Arc<GpuSpatialRefiner>,
    pub(crate) spatial_predicate: SpatialPredicate,
    /// Indexed batches containing evaluated geometry arrays. It contains the original record
    /// batches and geometry arrays obtained by evaluating the geometry expression on the build side.
    pub(crate) indexed_batches: Vec<EvaluatedBatch>,
    /// An array for translating data index to geometry batch index and row index
    pub(crate) data_id_to_batch_pos: Vec<(i32, i32)>,
    /// Shared bitmap builders for visited left indices, one per batch
    pub(crate) visited_build_side: Option<Mutex<Vec<BooleanBufferBuilder>>>,
    /// Counter of running probe-threads, potentially able to update `bitmap`.
    /// Each time a probe thread finished probing the index, it will decrement the counter.
    /// The last finished probe thread will produce the extra output batches for unmatched
    /// build side when running left-outer joins. See also [`report_probe_completed`].
    pub(crate) probe_threads_counter: AtomicUsize,
}
impl GPUSpatialIndex {
    pub fn empty(
        spatial_predicate: SpatialPredicate,
        schema: SchemaRef,
        gpu_options: GpuOptions,
        visited_build_side: Option<Mutex<Vec<BooleanBufferBuilder>>>,
        probe_threads_counter: AtomicUsize,
    ) -> Result<Self> {
        let gpu_libspatial_options = GpuSpatialOptions {
            cuda_use_memory_pool: gpu_options.use_memory_pool,
            cuda_memory_pool_init_percent: gpu_options.memory_pool_init_percentage as i32,
            concurrency: 1,
            device_id: gpu_options.device_id as i32,
            compress_bvh: gpu_options.compress_bvh,
            pipeline_batches: gpu_options.pipeline_batches as u32,
        };

        Ok(Self {
            schema,
            spatial_predicate,
            index: Arc::new(
                GpuSpatialIndex::try_new(&gpu_libspatial_options)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?,
            ),
            refiner: Arc::new(
                GpuSpatialRefiner::try_new(&gpu_libspatial_options)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?,
            ),
            indexed_batches: vec![],
            data_id_to_batch_pos: vec![],
            visited_build_side,
            probe_threads_counter,
        })
    }

    fn refine(
        &self,
        probe_geoms: &ArrayRef,
        predicate: &SpatialPredicate,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<()> {
        match predicate {
            SpatialPredicate::Relation(rel_p) => {
                self.refiner
                    .refine(
                        probe_geoms,
                        Self::convert_relation_type(&rel_p.relation_type)?,
                        build_indices,
                        probe_indices,
                    )
                    .map_err(|e| {
                        DataFusionError::Execution(format!(
                            "GPU spatial refinement failed: {:?}",
                            e
                        ))
                    })?;
                Ok(())
            }
            _ => Err(DataFusionError::NotImplemented(
                "Only Relation predicate is supported for GPU spatial query".to_string(),
            )),
        }
    }
    // Translate Sedona SpatialRelationType to GpuSpatialRelationPredicate
    fn convert_relation_type(t: &SpatialRelationType) -> Result<GpuSpatialRelationPredicate> {
        match t {
            SpatialRelationType::Equals => Ok(GpuSpatialRelationPredicate::Equals),
            SpatialRelationType::Touches => Ok(GpuSpatialRelationPredicate::Touches),
            SpatialRelationType::Contains => Ok(GpuSpatialRelationPredicate::Contains),
            SpatialRelationType::Covers => Ok(GpuSpatialRelationPredicate::Covers),
            SpatialRelationType::Intersects => Ok(GpuSpatialRelationPredicate::Intersects),
            SpatialRelationType::Within => Ok(GpuSpatialRelationPredicate::Within),
            SpatialRelationType::CoveredBy => Ok(GpuSpatialRelationPredicate::CoveredBy),
            _ => {
                // This should not happen as we check for supported predicates earlier
                Err(DataFusionError::Execution(format!(
                    "Unsupported spatial relation type for GPU: {:?}",
                    t
                )))
            }
        }
    }
}

#[async_trait]
impl SpatialIndex for GPUSpatialIndex {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn num_indexed_batches(&self) -> usize {
        self.indexed_batches.len()
    }
    fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch {
        &self.indexed_batches[batch_idx].batch
    }

    /// This method implements [`SpatialIndex::query_batch`] with GPU-accelerated spatial filtering
    /// and refinement. It takes a batch of probe geometries and a range of row indices to process,
    /// performs a spatial query on the GPU to find candidate matches,
    /// refines the candidates using the GPU refiner,
    /// and returns the matching build batch positions and probe indices.
    async fn query_batch(
        &self,
        evaluated_batch: &Arc<EvaluatedBatch>,
        range: Range<usize>,
        _max_result_size: usize,
        build_batch_positions: &mut Vec<(i32, i32)>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<(QueryResultMetrics, usize)> {
        if range.is_empty() {
            return Ok((
                QueryResultMetrics {
                    count: 0,
                    candidate_count: 0,
                },
                range.start,
            ));
        }
        let index = &self.index.as_ref();

        let empty_rect = [f32::NAN, f32::NAN, f32::NAN, f32::NAN];
        let rects: Vec<_> = range
            .clone()
            .map(|row_idx| {
                let bounds = &evaluated_batch.geom_array.rects()[row_idx];
                if bounds.is_empty() {
                    empty_rect
                } else {
                    let (x, y) = bounds.clone().into_inner();
                    [x.0, y.0, x.1, y.1]
                }
            })
            .collect();

        let (mut gpu_build_indices, mut gpu_probe_indices) =
            index.probe(rects.as_ref()).map_err(|e| {
                DataFusionError::Execution(format!("GPU spatial query failed: {:?}", e))
            })?;

        assert_eq!(gpu_build_indices.len(), gpu_probe_indices.len());

        let candidate_count = gpu_build_indices.len();

        self.refine(
            evaluated_batch.geom_array.geometry_array(),
            &self.spatial_predicate,
            &mut gpu_build_indices,
            &mut gpu_probe_indices,
        )?;

        assert_eq!(gpu_build_indices.len(), gpu_probe_indices.len());

        let total_count = gpu_build_indices.len();

        for (build_idx, probe_idx) in gpu_build_indices.iter().zip(gpu_probe_indices.iter()) {
            let data_id = *build_idx as usize;
            let (batch_idx, row_idx) = self.data_id_to_batch_pos[data_id];
            build_batch_positions.push((batch_idx, row_idx));
            probe_indices.push(range.start as u32 + probe_idx);
        }
        Ok((
            QueryResultMetrics {
                count: total_count,
                candidate_count,
            },
            range.end,
        ))
    }
    fn need_more_probe_stats(&self) -> bool {
        false
    }

    fn merge_probe_stats(&self, _stats: GeoStatistics) {}

    fn visited_build_side(&self) -> Option<&Mutex<Vec<BooleanBufferBuilder>>> {
        self.visited_build_side.as_ref()
    }

    fn report_probe_completed(&self) -> bool {
        self.probe_threads_counter.fetch_sub(1, Ordering::Relaxed) == 1
    }

    fn get_refiner_mem_usage(&self) -> usize {
        0
    }

    fn get_actual_execution_mode(&self) -> ExecutionMode {
        ExecutionMode::PrepareBuild // GPU-based spatial index is always on PrepareBuild mode
    }

    fn query_knn(
        &self,
        _probe_wkb: &Wkb,
        _k: u32,
        _use_spheroid: bool,
        _include_tie_breakers: bool,
        _build_batch_positions: &mut Vec<(i32, i32)>,
        _distances: Option<&mut Vec<f64>>,
    ) -> Result<QueryResultMetrics> {
        Err(DataFusionError::NotImplemented(
            "KNN query is not implemented for GPU spatial index".to_string(),
        ))
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use crate::index::gpu_spatial_index_builder::GPUSpatialIndexBuilder;
    use crate::options::GpuOptions;
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, SchemaRef};
    use datafusion_common::JoinType;
    use datafusion_physical_expr::expressions::Column;
    use futures::Stream;
    use sedona_expr::statistics::GeoStatistics;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_spatial_join::evaluated_batch::evaluated_batch_stream::{
        EvaluatedBatchStream, SendableEvaluatedBatchStream,
    };
    use sedona_spatial_join::evaluated_batch::EvaluatedBatch;
    use sedona_spatial_join::index::spatial_index::SpatialIndexRef;
    use sedona_spatial_join::index::spatial_index_builder::{
        SpatialIndexBuilder, SpatialJoinBuildMetrics,
    };
    use sedona_spatial_join::operand_evaluator::EvaluatedGeometryArray;
    use sedona_spatial_join::spatial_predicate::{RelationPredicate, SpatialRelationType};
    use sedona_spatial_join::SpatialPredicate;
    use sedona_testing::create::create_array;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::task::{Context, Poll};
    use std::vec::IntoIter;

    pub struct SingleBatchStream {
        // We use an Option so we can `take()` it on the first poll,
        // leaving `None` for subsequent polls to signal the end of the stream.
        batch: Option<EvaluatedBatch>,
        schema: SchemaRef,
    }

    impl SingleBatchStream {
        pub fn new(batch: EvaluatedBatch, schema: SchemaRef) -> Self {
            Self {
                batch: Some(batch),
                schema,
            }
        }
    }

    impl Stream for SingleBatchStream {
        type Item = datafusion_common::Result<EvaluatedBatch>; // Or Result<EvaluatedBatch, DataFusionError>

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            // `take()` removes the value from the Option, leaving `None` in its place.
            // If there is a batch, it maps it to `Some(Ok(batch))`.
            // If it's already empty, it returns `None`.
            Poll::Ready(self.batch.take().map(Ok))
        }
    }

    impl EvaluatedBatchStream for SingleBatchStream {
        fn is_external(&self) -> bool {
            false
        }

        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }
    }

    async fn build_index(
        builder: GPUSpatialIndexBuilder,
        indexed_batch: EvaluatedBatch,
        schema: SchemaRef,
    ) -> SpatialIndexRef {
        let single_batch_stream = SingleBatchStream::new(indexed_batch, schema);
        let sendable_stream: SendableEvaluatedBatchStream = Box::pin(single_batch_stream);
        let stats = GeoStatistics::empty();
        let mut builder = Box::new(builder);
        builder.add_stream(sendable_stream, stats).await.unwrap();
        builder.finish().unwrap()
    }

    // 1. Create a new stream struct for multiple batches
    pub struct VecBatchStream {
        // IntoIter allows us to cleanly pop items off the vector one by one
        batches: IntoIter<EvaluatedBatch>,
        schema: SchemaRef,
    }

    impl VecBatchStream {
        pub fn new(batches: Vec<EvaluatedBatch>, schema: SchemaRef) -> Self {
            Self {
                batches: batches.into_iter(),
                schema,
            }
        }
    }

    impl Stream for VecBatchStream {
        type Item = datafusion_common::Result<EvaluatedBatch>;

        fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            // `next()` on IntoIter returns Option<EvaluatedBatch>
            // We map it to Option<Result<EvaluatedBatch>> to match the stream's Item type
            Poll::Ready(self.batches.next().map(Ok))
        }
    }

    impl EvaluatedBatchStream for VecBatchStream {
        fn is_external(&self) -> bool {
            false
        }

        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }
    }

    // 2. Write the new build_index function that accepts the Vec
    async fn build_index_from_vec(
        mut builder: Box<GPUSpatialIndexBuilder>,
        indexed_batches: Vec<EvaluatedBatch>,
        schema: SchemaRef,
    ) -> SpatialIndexRef {
        let vec_batch_stream = VecBatchStream::new(indexed_batches, schema);
        let sendable_stream: SendableEvaluatedBatchStream = Box::pin(vec_batch_stream);

        let stats = GeoStatistics::empty();

        // Add the stream of multiple batches to the builder
        builder.add_stream(sendable_stream, stats).await.unwrap();
        builder.finish().unwrap()
    }
    #[test]
    fn test_spatial_index_builder_empty() {
        let options = GpuOptions {
            enable: true,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();
        let schema = Arc::new(arrow_schema::Schema::empty());
        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = Box::new(GPUSpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            metrics,
        ));

        // Test finishing with empty data
        let index = builder.finish().unwrap();
        assert_eq!(index.schema(), schema);
        assert_eq!(index.num_indexed_batches(), 0);
    }

    #[tokio::test]
    async fn test_spatial_index_builder_add_batch() {
        let options = GpuOptions {
            enable: true,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        // Create a simple test geometry batch
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let builder = GPUSpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            metrics,
        );

        let geom_batch = create_array(
            &[
                Some("POINT (0.25 0.25)"),
                Some("POINT (10 10)"),
                None,
                Some("POINT (0.25 0.25)"),
            ],
            &WKB_GEOMETRY,
        );
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(geom_batch.clone())]).unwrap();
        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        let index = build_index(builder, indexed_batch, schema.clone()).await;

        assert_eq!(index.schema(), schema);
        assert_eq!(index.num_indexed_batches(), 1);
    }

    #[tokio::test]
    async fn test_spatial_index_builder_add_multiple_batches() {
        let gpu_options = GpuOptions {
            enable: true,
            concat_build: false,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        // Create the schema
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        // Initialize the builder
        let builder = GPUSpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            gpu_options,
            JoinType::Inner,
            4,
            metrics,
        );

        // --- Create Batch 1 ---
        let geom_batch1 = create_array(
            &[
                Some("POINT (0.25 0.25)"),
                Some("POINT (10 10)"),
                None,
                Some("POINT (0.25 0.25)"),
            ],
            &WKB_GEOMETRY,
        );
        let batch1 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(geom_batch1.clone())]).unwrap();
        let evaluated_batch1 = EvaluatedBatch {
            batch: batch1,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch1, &WKB_GEOMETRY).unwrap(),
        };

        // --- Create Batch 2 ---
        let geom_batch2 = create_array(
            &[
                Some("POINT (1 1)"),
                Some("POINT (5 5)"),
                Some("POINT (20 20)"),
            ],
            &WKB_GEOMETRY,
        );
        let batch2 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(geom_batch2.clone())]).unwrap();
        let evaluated_batch2 = EvaluatedBatch {
            batch: batch2,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch2, &WKB_GEOMETRY).unwrap(),
        };

        // --- Build the Index ---
        // Combine them into a Vec and use the multi-batch builder function
        let indexed_batches = vec![evaluated_batch1, evaluated_batch2];

        // Note: This relies on the `build_index_from_vec` function we created earlier
        let index = build_index_from_vec(Box::new(builder), indexed_batches, schema.clone()).await;

        // --- Assertions ---
        assert_eq!(index.schema(), schema);

        // Assert that exactly 2 batches were indexed
        assert_eq!(index.num_indexed_batches(), 2);
    }

    async fn setup_index_for_batch_test(
        build_geoms: &[Option<&str>],
        options: GpuOptions,
    ) -> SpatialIndexRef {
        let metrics = SpatialJoinBuildMetrics::default();
        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("left", 0)),
            Arc::new(Column::new("right", 0)),
            SpatialRelationType::Intersects,
        ));
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let builder = GPUSpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            1,
            metrics,
        );

        let geom_array = create_array(build_geoms, &WKB_GEOMETRY);
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![Field::new(
                "geom",
                DataType::Binary,
                true,
            )])),
            vec![Arc::new(geom_array.clone())],
        )
        .unwrap();
        let evaluated_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY).unwrap(),
        };
        build_index(builder, evaluated_batch, schema).await
    }

    fn create_probe_batch(probe_geoms: &[Option<&str>]) -> Arc<EvaluatedBatch> {
        let geom_array = create_array(probe_geoms, &WKB_GEOMETRY);
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![Field::new(
                "geom",
                DataType::Binary,
                true,
            )])),
            vec![Arc::new(geom_array.clone())],
        )
        .unwrap();
        Arc::new(EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY).unwrap(),
        })
    }
    #[tokio::test]
    async fn test_query_batch_empty_results() {
        let build_geoms = &[Some("POINT (0 0)"), Some("POINT (1 1)")];
        let options = GpuOptions {
            enable: true,
            ..Default::default()
        };
        let index = setup_index_for_batch_test(build_geoms, options).await;

        // Probe with geometries that don't intersect
        let probe_geoms = &[Some("POINT (10 10)"), Some("POINT (20 20)")];
        let probe_batch = create_probe_batch(probe_geoms);

        let mut build_batch_positions = Vec::new();
        let mut probe_indices = Vec::new();
        let (metrics, next_idx) = index
            .query_batch(
                &probe_batch,
                0..2,
                usize::MAX,
                &mut build_batch_positions,
                &mut probe_indices,
            )
            .await
            .unwrap();

        assert_eq!(metrics.count, 0);
        assert_eq!(build_batch_positions.len(), 0);
        assert_eq!(probe_indices.len(), 0);
        assert_eq!(next_idx, 2);
    }

    #[tokio::test]
    async fn test_query_batch_non_empty_results_multiple_build_batches() {
        let gpu_options = GpuOptions {
            enable: true,
            concat_build: false,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let builder = GPUSpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            gpu_options,
            JoinType::Inner,
            4,
            metrics,
        );

        // --- Build Side: Multiple Batches of Polygons ---
        // Batch 0
        let build_geom_batch0 = create_array(
            &[
                Some("POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"),
                Some("POLYGON ((20 20, 20 30, 30 30, 30 20, 20 20))"),
            ],
            &WKB_GEOMETRY,
        );
        let build_batch0 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(build_geom_batch0.clone())])
                .unwrap();
        let evaluated_build0 = EvaluatedBatch {
            batch: build_batch0,
            geom_array: EvaluatedGeometryArray::try_new(build_geom_batch0, &WKB_GEOMETRY).unwrap(),
        };

        // Batch 1
        let build_geom_batch1 = create_array(
            &[Some("POLYGON ((40 40, 40 50, 50 50, 50 40, 40 40))")],
            &WKB_GEOMETRY,
        );
        let build_batch1 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(build_geom_batch1.clone())])
                .unwrap();
        let evaluated_build1 = EvaluatedBatch {
            batch: build_batch1,
            geom_array: EvaluatedGeometryArray::try_new(build_geom_batch1, &WKB_GEOMETRY).unwrap(),
        };

        // Build the multi-batch index using the helper function we created previously
        let indexed_batches = vec![evaluated_build0, evaluated_build1];
        let index = build_index_from_vec(Box::new(builder), indexed_batches, schema.clone()).await;

        // --- Probe Side: One Batch of Points ---
        let probe_geoms = &[
            Some("POINT (5 5)"),     // Matches Batch 0, Row 0
            Some("POINT (100 100)"), // No match
            Some("POINT (25 25)"),   // Matches Batch 0, Row 1
            Some("POINT (45 45)"),   // Matches Batch 1, Row 0
        ];
        let probe_batch = create_probe_batch(probe_geoms);

        // --- Execute Query ---
        let mut build_batch_positions = Vec::new();
        let mut probe_indices = Vec::new();

        // Probing all 4 points
        let (query_metrics, next_idx) = index
            .query_batch(
                &probe_batch,
                0..4,
                usize::MAX,
                &mut build_batch_positions,
                &mut probe_indices,
            )
            .await
            .unwrap();

        // --- Assertions ---
        // We expect exactly 3 matches out of the 4 probe points
        assert_eq!(query_metrics.count, 3);
        assert_eq!(build_batch_positions.len(), 3);
        assert_eq!(probe_indices.len(), 3);

        // The probe indices should match the rows in our probe batch that successfully intersected
        assert_eq!(probe_indices, vec![0, 2, 3]);

        // The query processed all 4 probe points
        assert_eq!(next_idx, 4);
    }
}
