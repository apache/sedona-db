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
use datafusion_common::{cast::as_struct_array, error::Result, exec_err, ScalarValue};
use datafusion_expr::{Accumulator, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef, SedonaAggregateUDF};
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::{RasterBuilder, RasterOverrides};
use sedona_raster::traits::{BandRef, RasterRef};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher, raster::BandDataType};

/// RS_Min_Agg() aggregate UDF implementation
///
/// Computes the pixel-wise minimum raster across a group, treating a band's
/// nodata sentinel (when present) as "no value at this pixel" rather than as
/// a comparable number.
pub fn rs_min_agg_udf() -> SedonaAggregateUDF {
    build_min_max_agg_udf("rs_min_agg", MinMaxOp::Min)
}

/// RS_Max_Agg() aggregate UDF implementation
///
/// Computes the pixel-wise maximum raster across a group, treating a band's
/// nodata sentinel (when present) as "no value at this pixel" rather than as
/// a comparable number.
pub fn rs_max_agg_udf() -> SedonaAggregateUDF {
    build_min_max_agg_udf("rs_max_agg", MinMaxOp::Max)
}

fn build_min_max_agg_udf(name: &str, op: MinMaxOp) -> SedonaAggregateUDF {
    SedonaAggregateUDF::new(
        name,
        Arc::new(RasterMinMaxAgg::new(op)) as SedonaAccumulatorRef,
        Volatility::Immutable,
    )
}

/// Which pixel wins a pairwise comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MinMaxOp {
    Min,
    Max,
}

impl MinMaxOp {
    /// True if `a` should be kept over `b`.
    fn a_wins<T: PartialOrd>(&self, a: T, b: T) -> bool {
        match self {
            MinMaxOp::Min => a <= b,
            MinMaxOp::Max => a >= b,
        }
    }
}

/// Shared [SedonaAccumulator] for `RS_Min_Agg`/`RS_Max_Agg`.
///
/// Scope (v1): single raster argument, plain [Accumulator] only (no
/// [datafusion_expr::GroupsAccumulator] — DataFusion falls back to one
/// accumulator per group, which is correct but not the fastest path for
/// many small groups; see the Linear "Raster Stats Agg" project for a
/// possible follow-up). Every raster combined within one group must share
/// band count, dimension names, shape, data type, and nodata value —
/// mismatches are reported as a clean execution error rather than silently
/// producing a wrong or partial result.
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
        if args.len() != 1 {
            return sedona_internal_err!(
                "RS_Min_Agg/RS_Max_Agg state_fields: expected 1 argument, got {}",
                args.len()
            );
        }
        // The running combined raster is itself a valid partial state (min/max
        // combine is associative and idempotent), so state is just the output
        // type again — this lets update_batch and merge_batch share one
        // implementation (both consume a raster-typed array).
        Ok(vec![Arc::new(
            args[0].to_storage_field("raster_state", true)?,
        )])
    }
}

/// Plain [Accumulator] backing both `RS_Min_Agg` and `RS_Max_Agg`.
///
/// `current` holds the running combined result as a single-row raster
/// `StructArray`, or `None` before any non-null raster has been seen.
#[derive(Debug)]
struct MinMaxAccumulator {
    op: MinMaxOp,
    current: Option<StructArray>,
}

impl MinMaxAccumulator {
    fn new(op: MinMaxOp) -> Self {
        Self { op, current: None }
    }

    /// Fold every non-null raster row of `array` into `self.current`. Used by
    /// both `update_batch` (raw input rows) and `merge_batch` (partial-state
    /// rows from other accumulators) since both are raster-typed arrays.
    fn combine_array(&mut self, array: &ArrayRef) -> Result<()> {
        let struct_array = as_struct_array(array)?;
        let rasters = RasterStructArray::try_new(struct_array)
            .map_err(|e| datafusion_common::exec_datafusion_err!("invalid raster array: {e}"))?;
        for i in 0..rasters.len() {
            if struct_array.is_null(i) {
                continue;
            }
            let incoming = rasters
                .get(i)
                .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
            self.combine_one(&incoming)?;
        }
        Ok(())
    }

    fn combine_one(&mut self, incoming: &dyn RasterRef) -> Result<()> {
        let merged = match self.current.take() {
            None => {
                let mut builder = RasterBuilder::new(1);
                builder
                    .copy_raster_from(incoming, RasterOverrides::default())
                    .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
                builder
                    .finish()
                    .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?
            }
            Some(current_struct) => {
                let current_rasters = RasterStructArray::try_new(&current_struct)
                    .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
                let current_raster = current_rasters
                    .get(0)
                    .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
                self.merge_rasters(&current_raster, incoming)?
            }
        };
        self.current = Some(merged);
        Ok(())
    }

    fn merge_rasters(&self, a: &dyn RasterRef, b: &dyn RasterRef) -> Result<StructArray> {
        if a.num_bands() != b.num_bands() {
            return exec_err!(
                "cannot combine rasters with different band counts ({} vs {}) in the same group",
                a.num_bands(),
                b.num_bands()
            );
        }

        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_from(a, RasterOverrides::default())
            .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
        for band_idx in 0..a.num_bands() {
            let band_a = a.band(band_idx)?;
            let band_b = b.band(band_idx)?;
            self.merge_band(
                &mut builder,
                a.band_name(band_idx),
                band_a.as_ref(),
                band_b.as_ref(),
            )?;
            builder
                .finish_band()
                .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
        }
        builder
            .finish_raster()
            .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
        builder
            .finish()
            .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))
    }

    fn merge_band(
        &self,
        builder: &mut RasterBuilder,
        band_name: Option<&str>,
        a: &dyn BandRef,
        b: &dyn BandRef,
    ) -> Result<()> {
        if a.data_type() != b.data_type() {
            return exec_err!(
                "cannot combine bands with different data types ({:?} vs {:?}) in the same group",
                a.data_type(),
                b.data_type()
            );
        }
        if a.shape() != b.shape() || a.dim_names() != b.dim_names() {
            return exec_err!(
                "cannot combine bands with different shapes ({:?} {:?} vs {:?} {:?}) in the same group",
                a.dim_names(),
                a.shape(),
                b.dim_names(),
                b.shape()
            );
        }
        if a.nodata() != b.nodata() {
            return exec_err!(
                "cannot combine bands with different nodata values in the same group"
            );
        }

        let nd_a = a.nd_buffer()?;
        let nd_b = b.nd_buffer()?;
        let bytes_a = nd_a.as_contiguous()?;
        let bytes_b = nd_b.as_contiguous()?;
        let combined = combine_bytes(self.op, a.data_type(), bytes_a, bytes_b, a.nodata())?;

        builder
            .start_band_nd(
                band_name,
                &a.dim_names(),
                a.shape(),
                a.data_type(),
                a.nodata(),
                None,
                None,
            )
            .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
        builder.band_data_writer().append_value(&combined);
        Ok(())
    }
}

/// Combine two same-shape, same-dtype contiguous band buffers pixel-wise.
///
/// A pixel equal to `nodata` (exact byte match — nodata is a sentinel value
/// stored inline, not a separate mask) is treated as absent: the other side
/// wins outright, and a pixel where both sides are nodata stays nodata.
fn combine_bytes(
    op: MinMaxOp,
    data_type: BandDataType,
    a: &[u8],
    b: &[u8],
    nodata: Option<&[u8]>,
) -> Result<Vec<u8>> {
    if a.len() != b.len() {
        return sedona_internal_err!(
            "mismatched band byte length combining rasters: {} vs {}",
            a.len(),
            b.len()
        );
    }

    macro_rules! combine_as {
        ($t:ty) => {{
            let elem = std::mem::size_of::<$t>();
            if a.len() % elem != 0 {
                return sedona_internal_err!(
                    "band byte length {} is not a multiple of element size {} for {:?}",
                    a.len(),
                    elem,
                    data_type
                );
            }
            let mut out = Vec::with_capacity(a.len());
            for start in (0..a.len()).step_by(elem) {
                let end = start + elem;
                let a_bytes = &a[start..end];
                let b_bytes = &b[start..end];
                let a_is_nodata = nodata.is_some_and(|nd| nd == a_bytes);
                let b_is_nodata = nodata.is_some_and(|nd| nd == b_bytes);
                let winner: &[u8] = if a_is_nodata && b_is_nodata {
                    a_bytes
                } else if a_is_nodata {
                    b_bytes
                } else if b_is_nodata {
                    a_bytes
                } else {
                    let a_val = <$t>::from_le_bytes(a_bytes.try_into().unwrap());
                    let b_val = <$t>::from_le_bytes(b_bytes.try_into().unwrap());
                    if op.a_wins(a_val, b_val) {
                        a_bytes
                    } else {
                        b_bytes
                    }
                };
                out.extend_from_slice(winner);
            }
            out
        }};
    }

    Ok(match data_type {
        BandDataType::UInt8 => combine_as!(u8),
        BandDataType::Int8 => combine_as!(i8),
        BandDataType::UInt16 => combine_as!(u16),
        BandDataType::Int16 => combine_as!(i16),
        BandDataType::UInt32 => combine_as!(u32),
        BandDataType::Int32 => combine_as!(i32),
        BandDataType::UInt64 => combine_as!(u64),
        BandDataType::Int64 => combine_as!(i64),
        BandDataType::Float32 => combine_as!(f32),
        BandDataType::Float64 => combine_as!(f64),
    })
}

impl Accumulator for MinMaxAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() != 1 {
            return sedona_internal_err!(
                "RS_Min_Agg/RS_Max_Agg update_batch: expected 1 argument, got {}",
                values.len()
            );
        }
        self.combine_array(&values[0])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let struct_array = match &self.current {
            Some(s) => s.clone(),
            None => {
                let mut builder = RasterBuilder::new(1);
                builder
                    .append_null()
                    .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
                builder
                    .finish()
                    .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?
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
        self.combine_array(&states[0])
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
}
