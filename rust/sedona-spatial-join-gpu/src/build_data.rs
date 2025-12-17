use crate::config::GpuSpatialJoinConfig;
use arrow_array::RecordBatch;

/// Shared build-side data for GPU spatial join
#[derive(Clone)]
pub(crate) struct GpuBuildData {
    /// All left-side data concatenated into single batch
    pub(crate) left_batch: RecordBatch,

    /// Configuration (includes geometry column indices, predicate, etc)
    pub(crate) config: GpuSpatialJoinConfig,

    /// Total rows in left batch
    pub(crate) left_row_count: usize,
}

impl GpuBuildData {
    pub fn new(left_batch: RecordBatch, config: GpuSpatialJoinConfig) -> Self {
        let left_row_count = left_batch.num_rows();
        Self {
            left_batch,
            config,
            left_row_count,
        }
    }

    pub fn left_batch(&self) -> &RecordBatch {
        &self.left_batch
    }

    pub fn config(&self) -> &GpuSpatialJoinConfig {
        &self.config
    }
}
