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
use std::{sync::Arc, vec};

use crate::executor::RasterExecutor;
use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_IsEmpty() scalar UDF implementation
///
/// Returns true when the raster has zero visible volume — some dimension
/// has length 0, so it holds no cells (e.g. the empty slabs a position
/// filter emits for non-matching chunks).
///
/// This is *visible-volume* emptiness (`Π shape() == 0`), not
/// byte-emptiness: a lazy / OutDb raster carries no pixel bytes but is
/// not empty — it has a real shape and resolves to pixels on demand.
pub fn rs_isempty_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_isempty",
        vec![Arc::new(RsIsEmpty {})],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsIsEmpty {}

impl SedonaScalarKernel for RsIsEmpty {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Boolean),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        executor.execute_raster_void(|_i, raster_opt| {
            match raster_opt {
                None => builder.append_null(),
                Some(raster) => builder.append_value(raster_is_empty(raster)?),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// A raster is empty when it carries no cells: it has no bands, or some
/// band has a zero-length dimension in its visible shape
/// (`Π shape() == 0`). Bands must agree on the spatial dims but may
/// differ on non-spatial ones, so every band is checked.
fn raster_is_empty(raster: &impl RasterRef) -> Result<bool> {
    let bands = raster.bands();
    if bands.is_empty() {
        return Ok(true);
    }
    // Band numbers are 1-based.
    for i in 1..=bands.len() {
        let band = bands.band(i)?;
        if band.shape().iter().product::<i64>() == 0 {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::BooleanArray;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::rasters::{generate_test_rasters, raster_from_single_band};
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_isempty_udf().into();
        assert_eq!(udf.name(), "rs_isempty");
    }

    #[test]
    fn udf_isempty() {
        let udf: ScalarUDF = rs_isempty_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        tester.assert_return_type(DataType::Boolean);

        // generate_test_rasters builds normal (non-empty) rasters, with a
        // null in the middle row.
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(BooleanArray::from(vec![Some(false), None, Some(false)]));
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert_array_equal(&result, &expected);

        // Null raster → null.
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Boolean(None));
    }

    #[test]
    fn udf_isempty_zero_volume() {
        let udf: ScalarUDF = rs_isempty_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        // A band with a zero-length dimension (here width 0) has zero
        // visible volume — RS_IsEmpty is true even though it isn't null.
        let empty = raster_from_single_band(0, 2, BandDataType::UInt8, &[], None);
        let result = tester.invoke_array(Arc::new(empty)).unwrap();
        let expected: Arc<dyn arrow_array::Array> = Arc::new(BooleanArray::from(vec![Some(true)]));
        assert_array_equal(&result, &expected);
    }
}
