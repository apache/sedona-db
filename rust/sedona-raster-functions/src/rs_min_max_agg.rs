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
use std::{mem::size_of_val, sync::Arc};

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::FieldRef;
use datafusion_common::{cast::as_struct_array, error::Result, exec_datafusion_err, ScalarValue};
use datafusion_expr::{Accumulator, ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef, SedonaAggregateUDF};
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::raster_min_max::{
    append_combined_single_band, extract_single_band, BandIndices, MinMaxOp,
};

/// RS_Min_Agg() aggregate UDF implementation
///
/// **Experimental.** Computes the pixel-wise minimum of one band across a
/// group, treating a band's nodata sentinel (when present) as "no value at
/// this pixel" rather than as a comparable number.
pub fn rs_min_agg_udf() -> SedonaAggregateUDF {
    build_min_max_agg_udf("rs_min_agg", MinMaxOp::Min)
}

/// RS_Max_Agg() aggregate UDF implementation
///
/// **Experimental.** Computes the pixel-wise maximum of one band across a
/// group, treating a band's nodata sentinel (when present) as "no value at
/// this pixel" rather than as a comparable number.
pub fn rs_max_agg_udf() -> SedonaAggregateUDF {
    build_min_max_agg_udf("rs_max_agg", MinMaxOp::Max)
}

fn build_min_max_agg_udf(name: &str, op: MinMaxOp) -> SedonaAggregateUDF {
    SedonaAggregateUDF::new(
        name,
        vec![
            Arc::new(RasterMinMaxAgg::new(op)) as SedonaAccumulatorRef,
            Arc::new(RasterMinMaxAggWithBand::new(op)) as SedonaAccumulatorRef,
        ],
        Volatility::Immutable,
    )
}

fn state_fields_for_raster_arg(args: &[SedonaType]) -> Result<Vec<FieldRef>> {
    let Some(raster_type) = args.first() else {
        return sedona_internal_err!(
            "RS_Min_Agg/RS_Max_Agg state_fields: expected a raster argument"
        );
    };
    // The running combined raster (always single-band) is itself a valid
    // partial state (min/max combine is associative and idempotent), so
    // state is just the raster type again — this lets update_batch and
    // merge_batch share one implementation.
    Ok(vec![Arc::new(
        raster_type.to_storage_field("raster_state", true)?,
    )])
}

/// `RS_Min_Agg(raster)`/`RS_Max_Agg(raster)` — operates on band 1.
#[derive(Debug)]
struct RasterMinMaxAgg {
    op: MinMaxOp,
    matcher: ArgMatcher,
}

impl RasterMinMaxAgg {
    fn new(op: MinMaxOp) -> Self {
        Self {
            op,
            matcher: ArgMatcher::new(vec![ArgMatcher::is_raster()], SedonaType::Raster),
        }
    }
}

impl SedonaAccumulator for RasterMinMaxAgg {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn accumulator(
        &self,
        _args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(MinMaxAccumulator::new(self.op)))
    }

    fn state_fields(&self, args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        state_fields_for_raster_arg(args)
    }
}

/// `RS_Min_Agg(raster, band_index)`/`RS_Max_Agg(raster, band_index)` — 1-based,
/// same convention as `RS_BandPixelType`.
#[derive(Debug)]
struct RasterMinMaxAggWithBand {
    op: MinMaxOp,
    matcher: ArgMatcher,
}

impl RasterMinMaxAggWithBand {
    fn new(op: MinMaxOp) -> Self {
        Self {
            op,
            matcher: ArgMatcher::new(
                vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
                SedonaType::Raster,
            ),
        }
    }
}

impl SedonaAccumulator for RasterMinMaxAggWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn accumulator(
        &self,
        _args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(MinMaxAccumulator::new(self.op)))
    }

    fn state_fields(&self, args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        state_fields_for_raster_arg(args)
    }
}

/// Plain [Accumulator] backing both overloads of `RS_Min_Agg`/`RS_Max_Agg`.
///
/// `current` holds the running combined single-band raster, or `None`
/// before any non-null raster has been seen. `band_index` may vary row to
/// row: each row's raster contributes whichever of its own bands that row
/// specifies (an out-of-range index for a given row is skipped, like a
/// null input).
#[derive(Debug)]
struct MinMaxAccumulator {
    op: MinMaxOp,
    current: Option<StructArray>,
}

impl MinMaxAccumulator {
    fn new(op: MinMaxOp) -> Self {
        Self { op, current: None }
    }

    fn combine_array(&mut self, array: &ArrayRef, band_indices: &BandIndices) -> Result<()> {
        let struct_array = as_struct_array(array)?;
        let rasters = RasterStructArray::try_new(struct_array)
            .map_err(|e| exec_datafusion_err!("invalid raster array: {e}"))?;
        for i in 0..rasters.len() {
            if struct_array.is_null(i) {
                continue;
            }
            let incoming = rasters.get(i).map_err(|e| exec_datafusion_err!("{e}"))?;
            self.combine_one(&incoming, band_indices.get(i))?;
        }
        Ok(())
    }

    fn combine_one(&mut self, incoming: &dyn RasterRef, band_index: i32) -> Result<()> {
        let Some(extracted) = extract_single_band(incoming, band_index)? else {
            // band_index doesn't exist on this row's raster: skip it, same
            // as a null input row.
            return Ok(());
        };

        self.current = Some(match self.current.take() {
            None => extracted,
            Some(current_struct) => {
                let current_rasters = RasterStructArray::try_new(&current_struct)
                    .map_err(|e| exec_datafusion_err!("{e}"))?;
                let current_raster = current_rasters
                    .get(0)
                    .map_err(|e| exec_datafusion_err!("{e}"))?;
                let extracted_rasters = RasterStructArray::try_new(&extracted)
                    .map_err(|e| exec_datafusion_err!("{e}"))?;
                let extracted_raster = extracted_rasters
                    .get(0)
                    .map_err(|e| exec_datafusion_err!("{e}"))?;

                let mut builder = RasterBuilder::new(1);
                append_combined_single_band(
                    self.op,
                    &mut builder,
                    &current_raster,
                    &extracted_raster,
                )?;
                builder.finish().map_err(|e| exec_datafusion_err!("{e}"))?
            }
        });
        Ok(())
    }
}

impl Accumulator for MinMaxAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() || values.len() > 2 {
            return sedona_internal_err!(
                "RS_Min_Agg/RS_Max_Agg update_batch: expected 1 or 2 arguments, got {}",
                values.len()
            );
        }
        let band_arg = values.get(1).map(|a| ColumnarValue::Array(a.clone()));
        let band_indices = BandIndices::resolve(band_arg.as_ref(), values[0].len())?;
        self.combine_array(&values[0], &band_indices)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let struct_array = match &self.current {
            Some(s) => s.clone(),
            None => {
                let mut builder = RasterBuilder::new(1);
                builder
                    .append_null()
                    .map_err(|e| exec_datafusion_err!("{e}"))?;
                builder.finish().map_err(|e| exec_datafusion_err!("{e}"))?
            }
        };
        Ok(ScalarValue::Struct(Arc::new(struct_array)))
    }

    fn size(&self) -> usize {
        size_of_val(self)
            + self
                .current
                .as_ref()
                .map(|s| s.get_array_memory_size())
                .unwrap_or(0)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 1 {
            return sedona_internal_err!(
                "RS_Min_Agg/RS_Max_Agg merge_batch: expected 1 state field, got {}",
                states.len()
            );
        }
        // State rows are already single-band (produced by evaluate()/state()
        // above), so band 1 is always the right (and only) band.
        self.combine_array(&states[0], &BandIndices::Fixed(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::AggregateUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{assert_raster_scalar_equals, RasterSpec};
    use sedona_testing::testers::AggregateUdfTester;

    fn min_tester() -> AggregateUdfTester {
        let udaf: AggregateUDF = rs_min_agg_udf().into();
        AggregateUdfTester::new(udaf, vec![RASTER])
    }

    fn max_tester() -> AggregateUdfTester {
        let udaf: AggregateUDF = rs_max_agg_udf().into();
        AggregateUdfTester::new(udaf, vec![RASTER])
    }

    #[test]
    fn udf_names() {
        let min_udaf: AggregateUDF = rs_min_agg_udf().into();
        let max_udaf: AggregateUDF = rs_max_agg_udf().into();
        assert_eq!(min_udaf.name(), "rs_min_agg");
        assert_eq!(max_udaf.name(), "rs_max_agg");
    }

    #[test]
    fn min_agg_basic() {
        let tester = min_tester();
        let a = RasterSpec::d2(2, 1).band_values(&[5u8, 2u8]);
        let b = RasterSpec::d2(2, 1).band_values(&[3u8, 9u8]);

        let result = tester
            .aggregate_rasters(vec![vec![Some(a), Some(b)]])
            .unwrap();

        let expected = RasterSpec::d2(2, 1).band_values(&[3u8, 2u8]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn max_agg_basic() {
        let tester = max_tester();
        let a = RasterSpec::d2(2, 1).band_values(&[5u8, 2u8]);
        let b = RasterSpec::d2(2, 1).band_values(&[3u8, 9u8]);

        let result = tester
            .aggregate_rasters(vec![vec![Some(a), Some(b)]])
            .unwrap();

        let expected = RasterSpec::d2(2, 1).band_values(&[5u8, 9u8]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn min_agg_nodata_is_skipped() {
        let tester = min_tester();
        // Pixel 0: a is nodata, b=3 -> 3 wins. Pixel 1: both nodata -> stays nodata.
        let a = RasterSpec::d2(2, 1)
            .band_values(&[255u8, 255u8])
            .nodata(255u8);
        let b = RasterSpec::d2(2, 1)
            .band_values(&[3u8, 255u8])
            .nodata(255u8);

        let result = tester
            .aggregate_rasters(vec![vec![Some(a), Some(b)]])
            .unwrap();

        let expected = RasterSpec::d2(2, 1)
            .band_values(&[3u8, 255u8])
            .nodata(255u8);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn min_agg_across_separate_batches_merges_state() {
        // Each inner Vec is its own "batch": the tester runs update_batch per
        // batch on a fresh accumulator, then merge_batch across their states,
        // exercising the state()/merge_batch path rather than just update_batch.
        let tester = min_tester();
        let a = RasterSpec::d2(2, 1).band_values(&[5u8, 2u8]);
        let b = RasterSpec::d2(2, 1).band_values(&[3u8, 9u8]);
        let c = RasterSpec::d2(2, 1).band_values(&[4u8, 1u8]);

        let result = tester
            .aggregate_rasters(vec![vec![Some(a)], vec![Some(b)], vec![Some(c)]])
            .unwrap();

        let expected = RasterSpec::d2(2, 1).band_values(&[3u8, 1u8]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn min_agg_single_null_batch_returns_null_raster() {
        let tester = min_tester();
        let result = tester
            .aggregate_rasters(vec![vec![None::<RasterSpec>]])
            .unwrap();
        assert!(matches!(result, ScalarValue::Struct(s) if s.is_null(0)));
    }

    #[test]
    fn min_agg_mismatched_shape_errors() {
        let tester = min_tester();
        let a = RasterSpec::d2(2, 1).band_values(&[5u8, 2u8]);
        let b = RasterSpec::d2(1, 1).band_values(&[3u8]);

        let err = tester
            .aggregate_rasters(vec![vec![Some(a), Some(b)]])
            .unwrap_err();
        assert!(err.to_string().contains("different shapes"));
    }

    #[test]
    fn min_agg_with_explicit_band_selects_second_band() {
        // The (raster, band_index) overload isn't a shape the single-arg
        // AggregateUdfTester helpers support, so drive the Accumulator
        // directly instead.
        use arrow_array::Int32Array;
        use sedona_testing::raster_spec::raster_array;

        let a = RasterSpec::d2(1, 1)
            .band_values(&[5u8])
            .band_values(&[50u8]);
        let b = RasterSpec::d2(1, 1)
            .band_values(&[3u8])
            .band_values(&[90u8]);

        let raster_array: ArrayRef = Arc::new(raster_array([Some(a), Some(b)]));
        let band_index_array: ArrayRef = Arc::new(Int32Array::from(vec![2, 2]));

        let mut accumulator = MinMaxAccumulator::new(MinMaxOp::Min);
        accumulator
            .update_batch(&[raster_array, band_index_array])
            .unwrap();
        let result = accumulator.evaluate().unwrap();

        let expected = RasterSpec::d2(1, 1).band_values(&[50u8]);
        assert_raster_scalar_equals(&result, &expected);
    }
}
