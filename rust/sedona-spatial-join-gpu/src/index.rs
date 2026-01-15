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
