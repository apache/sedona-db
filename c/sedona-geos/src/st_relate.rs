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

use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use datafusion_common::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_Relate implementation using GEOS
pub fn st_relate_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STRelate {})
}

#[derive(Debug)]
struct STRelate {}

impl SedonaScalarKernel for STRelate {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(arrow_schema::DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);

        // ST_Relate returns a 9-char DE-9IM string per row; 9 bytes * n rows
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), 9 * executor.num_iterations());

        executor.execute_wkb_wkb_void(|wkb1, wkb2| {
            match (wkb1, wkb2) {
                (Some(g1), Some(g2)) => {
                    let relate = g1
                        .relate(g2)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;

                    builder.append_value(relate);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}
