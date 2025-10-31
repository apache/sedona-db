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

use arrow_array::{Array, ArrayRef, StructArray};
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_raster::array::{RasterRefImpl, RasterStructArray};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::datatypes::RASTER;

/// Helper for writing raster kernel implementations
///
/// The [RasterExecutor] provides a simplified interface for executing functions
/// on raster arrays, handling the common pattern of downcasting to StructArray,
/// creating raster iterators, and handling null values.
pub struct RasterExecutor<'a, 'b> {
    pub arg_types: &'a [SedonaType],
    pub args: &'b [ColumnarValue],
    num_iterations: usize,
}

impl<'a, 'b> RasterExecutor<'a, 'b> {
    /// Create a new [RasterExecutor]
    pub fn new(arg_types: &'a [SedonaType], args: &'b [ColumnarValue]) -> Self {
        Self {
            arg_types,
            args,
            num_iterations: Self::calc_num_iterations(args),
        }
    }

    /// Return the number of iterations that will be performed
    pub fn num_iterations(&self) -> usize {
        self.num_iterations
    }

    /// Execute a function by iterating over rasters in the first argument
    ///
    /// This handles the common pattern of:
    /// 1. Downcasting array to StructArray
    /// 2. Creating raster iterator
    /// 3. Iterating with null checks
    /// 4. Calling the provided function with each raster
    pub fn execute_raster_void<F>(&self, mut func: F) -> Result<()>
    where
        F: FnMut(usize, Option<RasterRefImpl<'_>>) -> Result<()>,
    {
        assert!(
            self.arg_types[0] == RASTER,
            "First argument must be a raster type"
        );
        let raster_array = match &self.args[0] {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(_) => {
                return Err(DataFusionError::NotImplemented(
                    "Scalar raster input not yet supported".to_string(),
                ));
            }
        };

        // Downcast to StructArray (rasters are stored as structs)
        let raster_struct = raster_array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                DataFusionError::Internal("Expected StructArray for raster data".to_string())
            })?;

        // Create raster iterator
        let raster_array = RasterStructArray::new(raster_struct);

        // Iterate through each raster in the array
        for i in 0..self.num_iterations {
            if raster_array.is_null(i) {
                func(i, None)?;
                continue;
            }
            let raster = raster_array.get(i).ok_or_else(|| {
                DataFusionError::Internal(format!("Failed to get raster at index {}", i))
            })?;

            func(i, Some(raster))?;
        }

        Ok(())
    }

    /// Finish an [ArrayRef] output as the appropriate [ColumnarValue]
    ///
    /// Converts the output into a [ColumnarValue::Scalar] if all arguments were scalars,
    /// or a [ColumnarValue::Array] otherwise.
    pub fn finish(&self, out: ArrayRef) -> Result<ColumnarValue> {
        for arg in self.args {
            match arg {
                // If any argument was an array, we return an array
                ColumnarValue::Array(_) => {
                    return Ok(ColumnarValue::Array(out));
                }
                ColumnarValue::Scalar(_) => {}
            }
        }

        // For all scalar arguments, we return a scalar
        Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(&out, 0)?))
    }

    /// Calculates the number of iterations that should happen based on the
    /// argument ColumnarValue types
    fn calc_num_iterations(args: &[ColumnarValue]) -> usize {
        for arg in args {
            match arg {
                // If any argument is an array, we have to iterate array.len() times
                ColumnarValue::Array(array) => {
                    return array.len();
                }
                ColumnarValue::Scalar(_) => {}
            }
        }

        // All scalars: we iterate once
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::UInt64Builder;
    use arrow_array::UInt64Array;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::rasters::sequential_rasters;
    use std::sync::Arc;

    #[test]
    fn test_raster_executor_execute_raster_void() {
        // 3 rasters, second one is null
        let rasters = sequential_rasters(3, Some(1)).unwrap();
        let args = [ColumnarValue::Array(Arc::new(rasters))];
        let arg_types = vec![RASTER];

        let executor = RasterExecutor::new(&arg_types, &args);
        assert_eq!(executor.num_iterations(), 3);

        let mut builder = UInt64Builder::with_capacity(executor.num_iterations());
        executor
            .execute_raster_void(|_i, raster_opt| {
                match raster_opt {
                    None => builder.append_null(),
                    Some(raster) => {
                        let width = raster.metadata().width();
                        builder.append_value(width);
                    }
                }
                Ok(())
            })
            .unwrap();

        let result = executor.finish(Arc::new(builder.finish())).unwrap();

        let width_array = match &result {
            ColumnarValue::Array(array) => array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("Expected UInt64Array"),
            ColumnarValue::Scalar(_) => panic!("Expected array, got scalar"),
        };

        assert_eq!(width_array.len(), 3);
        assert_eq!(width_array.value(0), 1);
        assert!(width_array.is_null(1));
        assert_eq!(width_array.value(2), 3);
    }
}
