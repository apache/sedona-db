pub(crate) mod build_side_collector;
pub(crate) mod spatial_index;
pub(crate) mod spatial_index_builder;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
pub(crate) use build_side_collector::{
    BuildPartition, BuildSideBatchesCollector, CollectBuildSideMetrics,
};
use datafusion_common::{DataFusionError, Result};
pub use spatial_index::SpatialIndex;
pub use spatial_index_builder::{SpatialIndexBuilder, SpatialJoinBuildMetrics};
pub(crate) fn ensure_binary_array(array: &ArrayRef) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::BinaryView => {
            // OPTIMIZATION: Use Arrow's cast which is much faster than manual iteration
            use arrow::compute::cast;
            cast(array.as_ref(), &DataType::Binary).map_err(|e| {
                DataFusionError::Execution(format!(
                    "Arrow cast from BinaryView to Binary failed: {:?}",
                    e
                ))
            })
        }
        DataType::Binary | DataType::LargeBinary => {
            // Already in correct format
            Ok(array.clone())
        }
        _ => Err(DataFusionError::Execution(format!(
            "Expected Binary/BinaryView array, got {:?}",
            array.data_type()
        ))),
    }
}
