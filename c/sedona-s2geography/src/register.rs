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

use datafusion_common::Result;
use sedona_common::sedona_internal_err;
use sedona_expr::aggregate_udf::SedonaAccumulatorRef;
use sedona_expr::scalar_udf::ScalarKernelRef;
use sedona_functions::{st_analyze_agg, st_envelope_agg};
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};
use std::sync::{Arc, OnceLock};

static S2_SCALAR_KERNELS: OnceLock<Result<Vec<(String, ScalarKernelRef)>>> = OnceLock::new();

/// Initialize s2geography scalar kernels via extension ABI
///
/// This function is the entrypoint to S2Geography-based scalar kernels suitable for
/// adding to a FunctionSet.
pub fn scalar_kernels() -> Result<Vec<(&'static str, ScalarKernelRef)>> {
    match S2_SCALAR_KERNELS.get_or_init(init_scalar_kernels) {
        Ok(kernels) => Ok(kernels
            .iter()
            .map(|(name, kernel)| (name.as_str(), kernel.clone()))
            .collect()),
        Err(err) => sedona_internal_err!("Error initializing s2geography kernels: {err}"),
    }
}

fn init_scalar_kernels() -> Result<Vec<(String, ScalarKernelRef)>> {
    let kernels = crate::kernels::s2_scalar_kernels()?;
    Ok(kernels)
}

/// Returns aggregate kernels for s2geography functions
pub fn aggregate_kernels() -> Vec<(&'static str, Vec<SedonaAccumulatorRef>)> {
    vec![
        (
            "st_analyze_agg",
            st_analyze_agg::st_analyze_agg_impl_for::<crate::rect_bounder::WkbGeographyBounder>(
                ArgMatcher::new(
                    vec![ArgMatcher::is_geography()],
                    st_analyze_agg::output_sedona_type(),
                ),
            ),
        ),
        (
            "st_envelope_agg",
            vec![Arc::new(st_envelope_agg::STEnvelopeAgg::<
                crate::rect_bounder::WkbGeographyBounder,
            >::new(ArgMatcher::new(
                vec![ArgMatcher::is_geography()],
                WKB_GEOMETRY,
            )))],
        ),
    ]
}
