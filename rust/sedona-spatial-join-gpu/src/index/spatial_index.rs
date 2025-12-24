use crate::evaluated_batch::EvaluatedBatch;
use crate::index::ensure_binary_array;
use crate::operand_evaluator::OperandEvaluator;
use crate::utils::once_fut::{OnceAsync, OnceFut};
use crate::{operand_evaluator::create_operand_evaluator, spatial_predicate::SpatialPredicate};
use arrow::array::BooleanBufferBuilder;
use arrow_array::ArrayRef;
use datafusion_common::{DataFusionError, Result};
use geo_types::Rect;
use parking_lot::{Mutex, RwLock};
use sedona_common::SpatialJoinOptions;
use sedona_libgpuspatial::GpuSpatial;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{ready, Poll};

pub struct SpatialIndex {
    /// The spatial predicate evaluator for the spatial predicate.
    #[allow(dead_code)] // reserved for GPU-based distance evaluation
    pub(crate) evaluator: Arc<dyn OperandEvaluator>,
    /// Indexed batch containing evaluated geometry arrays. It contains the original record
    /// batches and geometry arrays obtained by evaluating the geometry expression on the build side.
    pub(crate) build_batch: EvaluatedBatch,
    /// GPU spatial object for performing GPU-accelerated spatial queries
    pub(crate) gpu_spatial: Arc<GpuSpatial>,
    /// Shared bitmap builders for visited left indices
    pub(crate) visited_left_side: Option<Mutex<BooleanBufferBuilder>>,
    /// Counter of running probe-threads, potentially able to update `bitmap`.
    /// Each time a probe thread finished probing the index, it will decrement the counter.
    /// The last finished probe thread will produce the extra output batches for unmatched
    /// build side when running left-outer joins. See also [`report_probe_completed`].
    pub(crate) probe_threads_counter: AtomicUsize,
}

impl SpatialIndex {
    pub fn new(
        evaluator: Arc<dyn OperandEvaluator>,
        build_batch: EvaluatedBatch,
        visited_left_side: Option<Mutex<BooleanBufferBuilder>>,
        gpu_spatial: Arc<GpuSpatial>,
        probe_threads_counter: AtomicUsize,
    ) -> Self {
        Self {
            evaluator,
            build_batch,
            gpu_spatial,
            visited_left_side,
            probe_threads_counter,
        }
    }

    pub fn new_empty(
        build_batch: EvaluatedBatch,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        probe_threads_counter: AtomicUsize,
    ) -> Result<Self> {
        let evaluator = create_operand_evaluator(&spatial_predicate, options.clone());
        Ok(Self {
            evaluator,
            build_batch,
            gpu_spatial: Arc::new(
                GpuSpatial::new().map_err(|e| DataFusionError::Execution(e.to_string()))?,
            ),
            visited_left_side: None,
            probe_threads_counter,
        })
    }

    /// Get the bitmaps for tracking visited left-side indices. The bitmaps will be updated
    /// by the spatial join stream when producing output batches during index probing phase.
    pub(crate) fn visited_left_side(&self) -> Option<&Mutex<BooleanBufferBuilder>> {
        self.visited_left_side.as_ref()
    }
    pub(crate) fn report_probe_completed(&self) -> bool {
        self.probe_threads_counter.fetch_sub(1, Ordering::Relaxed) == 1
    }

    pub(crate) fn filter(&self, probe_rects: &[Rect<f32>]) -> Result<(Vec<u32>, Vec<u32>)> {
        let gs = &self.gpu_spatial.as_ref();

        let (mut build_indices, mut probe_indices) = gs.probe(probe_rects).map_err(|e| {
            DataFusionError::Execution(format!("GPU spatial query failed: {:?}", e))
        })?;

        Ok((build_indices, probe_indices))
    }

    pub(crate) fn refine_loaded(
        &self,
        probe_geoms: &ArrayRef,
        predicate: &SpatialPredicate,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<()> {
        match predicate {
            SpatialPredicate::Relation(rel_p) => {
                let geoms = ensure_binary_array(probe_geoms)?;

                self.gpu_spatial
                    .refine_loaded(&geoms, rel_p.relation_type, build_indices, probe_indices)
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

    pub(crate) fn refine(
        &self,
        probe_geoms: &arrow_array::ArrayRef,
        predicate: &SpatialPredicate,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<()> {
        match predicate {
            SpatialPredicate::Relation(rel_p) => {
                let gs = &self.gpu_spatial.as_ref();
                let geoms = ensure_binary_array(probe_geoms)?;

                gs.refine(
                    &self.build_batch.geom_array.geometry_array,
                    &geoms,
                    rel_p.relation_type,
                    build_indices,
                    probe_indices,
                )
                .map_err(|e| {
                    DataFusionError::Execution(format!("GPU spatial refinement failed: {:?}", e))
                })?;
                Ok(())
            }
            _ => Err(DataFusionError::NotImplemented(
                "Only Relation predicate is supported for GPU spatial query".to_string(),
            )),
        }
    }
}
