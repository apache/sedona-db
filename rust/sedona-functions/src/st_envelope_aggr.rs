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

use crate::executor::WkbExecutor;
use crate::st_envelope::write_envelope;
use arrow_array::ArrayRef;
use arrow_schema::FieldRef;
use datafusion_common::{
    error::{DataFusionError, Result},
    ScalarValue,
};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, Accumulator, ColumnarValue, Documentation, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_expr::aggregate_udf::{SedonaAccumulator, SedonaAggregateUDF};
use sedona_geometry::{
    bounds::geo_traits_update_xy_bounds,
    interval::{Interval, IntervalTrait},
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

/// ST_Envelope_Aggr() aggregate UDF implementation
///
/// An implementation of envelope (bounding shape) calculation.
pub fn st_envelope_aggr_udf() -> SedonaAggregateUDF {
    SedonaAggregateUDF::new(
        "st_envelope_aggr",
        vec![Arc::new(STEnvelopeAggr {})],
        Volatility::Immutable,
        Some(st_envelope_aggr_doc()),
    )
}

fn st_envelope_aggr_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the entire envelope boundary of all geometries in geom",
        "ST_Envelope_Aggr (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("SELECT ST_Envelope_Aggr(ST_GeomFromWKT('MULTIPOINT (0 1, 10 11)'))")
    .build()
}

#[derive(Debug)]
struct STEnvelopeAggr {}

impl SedonaAccumulator for STEnvelopeAggr {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);
        matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(BoundsAccumulator2D::new(args[0].clone())))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![Arc::new(
            WKB_GEOMETRY.to_storage_field("envelope", true)?,
        )])
    }
}

#[derive(Debug)]
struct BoundsAccumulator2D {
    input_type: SedonaType,
    x: Interval,
    y: Interval,
}

impl BoundsAccumulator2D {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            x: Interval::empty(),
            y: Interval::empty(),
        }
    }

    // Create a WKB result based on the current state of the accumulator.
    fn make_wkb_result(&self) -> Result<Option<Vec<u8>>> {
        let mut wkb = Vec::new();
        let written = write_envelope(&self.x.into(), &self.y, &mut wkb)?;
        if written {
            Ok(Some(wkb))
        } else {
            Ok(None)
        }
    }

    // Check the input length for update methods.
    fn check_update_input_len(input: &[ArrayRef], expected: usize, context: &str) -> Result<()> {
        if input.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "No input arrays provided to accumulator in {context}"
            )));
        }
        if input.len() != expected {
            return sedona_internal_err!(
                "Unexpected input length in {} (expected {}, got {})",
                context,
                expected,
                input.len()
            );
        }
        Ok(())
    }

    // Execute the update operation for the accumulator.
    fn execute_update(&mut self, executor: WkbExecutor) -> Result<(), DataFusionError> {
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                geo_traits_update_xy_bounds(item, &mut self.x, &mut self.y)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            }
            Ok(())
        })?;
        Ok(())
    }
}

impl Accumulator for BoundsAccumulator2D {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        Self::check_update_input_len(values, 1, "update_batch")?;
        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &args);
        self.execute_update(executor)?;
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let wkb = self.make_wkb_result()?;
        Ok(ScalarValue::Binary(wkb))
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let wkb = self.make_wkb_result()?;
        Ok(vec![ScalarValue::Binary(wkb)])
    }

    fn size(&self) -> usize {
        size_of::<BoundsAccumulator2D>()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        Self::check_update_input_len(states, 1, "merge_batch")?;
        let array = &states[0];
        let args = [ColumnarValue::Array(array.clone())];
        let arg_types = [WKB_GEOMETRY.clone()];
        let executor = WkbExecutor::new(&arg_types, &args);
        self.execute_update(executor)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use datafusion_expr::AggregateUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{compare::assert_scalar_equal_wkb_geometry, testers::AggregateUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = st_envelope_aggr_udf().into();
        assert_eq!(udf.name(), "st_envelope_aggr");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester =
            AggregateUdfTester::new(st_envelope_aggr_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Finite input with nulls
        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON((0 1, 0 7, 6 7, 6 1, 0 1))"),
        );

        // Empty input
        assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(vec![]).unwrap(), None);

        // All coordinates empty
        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![Some("POINT EMPTY")]])
                .unwrap(),
            None,
        );

        // Degenerate output: point
        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![Some("POINT (0 1)")]])
                .unwrap(),
            Some("POINT (0 1)"),
        );

        // Degenerate output: vertical line
        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![Some("MULTIPOINT (0 2, 0 1)")]])
                .unwrap(),
            Some("LINESTRING (0 1, 0 2)"),
        );

        // Degenerate output: horizontal line
        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![Some("MULTIPOINT (1 1, 0 1)")]])
                .unwrap(),
            Some("LINESTRING (0 1, 1 1)"),
        );
    }
}
