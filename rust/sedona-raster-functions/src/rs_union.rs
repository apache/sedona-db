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

//! `RS_Union` — the bands of several rasters, in order, as one raster.
//!
//! ```text
//! RS_Union(raster1, raster2[, raster3, ..., raster7])  -> Raster
//! ```
//!
//! Every band of `raster1`, then every band of `raster2`, and so on, each
//! keeping its own pixel type, nodata value and name. The rasters must share
//! their width and height; the result takes the first raster's georeference and
//! CRS. Any NULL raster gives a NULL result.
//!
//! No pixel is read: each band is carried over as it is (zero-copy for InDb
//! bands, by reference for OutDb bands), so the function needs no loading.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use datafusion_common::{Result, exec_err, internal_datafusion_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::{RasterBuilder, RasterOverrides};
use sedona_raster::traits::{BandOverrides, RasterRef};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::executor::RasterExecutor;

/// The most rasters one call can join, as in Sedona Spark.
const MAX_RASTERS: usize = 7;

/// `RS_Union()` scalar UDF — the bands of several rasters as one raster.
pub fn rs_union_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_union",
        (2..=MAX_RASTERS)
            .map(|num_rasters| Arc::new(RsUnion { num_rasters }) as _)
            .collect(),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsUnion {
    num_rasters: usize,
}

impl SedonaScalarKernel for RsUnion {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = (0..self.num_rasters)
            .map(|_| ArgMatcher::is_raster())
            .collect();
        ArgMatcher::new(matchers, SedonaType::Raster).match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let n = RasterExecutor::num_iterations_over(args);
        let arrays = args
            .iter()
            .map(|arg| arg.clone().into_array(n))
            .collect::<Result<Vec<ArrayRef>>>()?;
        let rasters = arrays
            .iter()
            .map(|array| {
                let array = array
                    .as_any()
                    .downcast_ref::<StructArray>()
                    .ok_or_else(|| internal_datafusion_err!("Expected StructArray for raster"))?;
                Ok(RasterStructArray::try_new(array)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut builder = RasterBuilder::new(n);
        for i in 0..n {
            if rasters.iter().any(|r| r.is_null(i)) {
                builder.append_null()?;
                continue;
            }
            let row = rasters
                .iter()
                .map(|r| r.get(i))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let row: Vec<&dyn RasterRef> = row.iter().map(|r| r as &dyn RasterRef).collect();
            union(&mut builder, &row)?;
        }

        RasterExecutor::finish_over(args, Arc::new(builder.finish()?))
    }
}

/// Append one raster holding every band of `rasters`, in order, under the first
/// raster's header.
fn union(builder: &mut RasterBuilder, rasters: &[&dyn RasterRef]) -> Result<()> {
    let first = rasters[0];
    let (width, height) = (first.width()?, first.height()?);
    for (k, raster) in rasters.iter().enumerate().skip(1) {
        let (w, h) = (raster.width()?, raster.height()?);
        if (w, h) != (width, height) {
            return exec_err!(
                "RS_Union: raster {} is {w} x {h}, but the first raster is {width} x {height}; \
                 every raster must share its width and height",
                k + 1
            );
        }
    }

    builder.start_raster_from(first, RasterOverrides::default())?;
    for raster in rasters {
        for band_idx in 0..raster.num_bands() {
            raster
                .band(band_idx)?
                .copy_into(builder, BandOverrides::default())?;
            builder.finish_band()?;
        }
    }
    builder.finish_raster()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::raster_spec::{
        RasterSpec, assert_raster_scalar_equals, assert_rasters_equal, raster_array,
    };
    use sedona_testing::testers::ScalarUdfTester;

    fn tester(num_rasters: usize) -> ScalarUdfTester {
        ScalarUdfTester::new(rs_union_udf().into(), vec![RASTER; num_rasters])
    }

    fn arrays(rows: Vec<Vec<Option<RasterSpec>>>) -> Vec<ArrayRef> {
        rows.into_iter()
            .map(|col| Arc::new(raster_array(col)) as ArrayRef)
            .collect()
    }

    /// A 3x2 raster at `origin_x`, with one UInt8 band valued `base..base+6`.
    fn uint8(base: u8, origin_x: f64) -> RasterSpec {
        RasterSpec::d2(3, 2)
            .transform([origin_x, 1.0, 0.0, 0.0, 0.0, -1.0])
            .band_values(&[base, base + 1, base + 2, base + 3, base + 4, base + 5])
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_union_udf().into();
        assert_eq!(udf.name(), "rs_union");
    }

    #[test]
    fn appends_bands_in_argument_order() {
        let first = uint8(10, 0.0).nodata(10u8);
        let second = RasterSpec::d2(3, 2)
            .band_values(&[0.5f64, 1.5, 2.5, 3.5, 4.5, 5.5])
            .name("b")
            .band_values(&[-1i16, -2, -3, -4, -5, -6])
            .nodata(-1i16);
        let result = tester(2)
            .invoke_arrays(arrays(vec![vec![Some(first)], vec![Some(second)]]))
            .unwrap();

        // Each band keeps its own type, nodata and name; the grid is the first
        // raster's.
        let expected = uint8(10, 0.0)
            .nodata(10u8)
            .band_values(&[0.5f64, 1.5, 2.5, 3.5, 4.5, 5.5])
            .name("b")
            .band_values(&[-1i16, -2, -3, -4, -5, -6])
            .nodata(-1i16);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn takes_the_first_rasters_georeference_and_crs() {
        let first = uint8(1, 100.0).crs(Some("EPSG:3857"));
        let second = uint8(20, -5.0).crs(None);
        let result = tester(2)
            .invoke_arrays(arrays(vec![vec![Some(first)], vec![Some(second)]]))
            .unwrap();
        let expected = uint8(1, 100.0)
            .crs(Some("EPSG:3857"))
            .band_values(&[20u8, 21, 22, 23, 24, 25]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn joins_up_to_seven_rasters() {
        let rows = (0..7u8).map(|k| vec![Some(uint8(k * 10, 0.0))]).collect();
        let result = tester(7).invoke_arrays(arrays(rows)).unwrap();
        let expected = (1..7u8).fold(uint8(0, 0.0), |spec, k| {
            spec.band_values(&[
                k * 10,
                k * 10 + 1,
                k * 10 + 2,
                k * 10 + 3,
                k * 10 + 4,
                k * 10 + 5,
            ])
        });
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn a_null_raster_gives_null() {
        let result = tester(3)
            .invoke_arrays(arrays(vec![
                vec![Some(uint8(1, 0.0)), None, Some(uint8(1, 0.0))],
                vec![
                    Some(uint8(2, 0.0)),
                    Some(uint8(2, 0.0)),
                    Some(uint8(2, 0.0)),
                ],
                vec![Some(uint8(3, 0.0)), Some(uint8(3, 0.0)), None],
            ]))
            .unwrap();
        let expected = uint8(1, 0.0)
            .band_values(&[2u8, 3, 4, 5, 6, 7])
            .band_values(&[3u8, 4, 5, 6, 7, 8]);
        assert_rasters_equal(&result, &[Some(expected), None, None]);
    }

    #[test]
    fn shape_mismatch_errors() {
        // Width alone differs.
        let other = RasterSpec::d2(2, 2).band_values(&[1u8, 2, 3, 4]);
        let err = tester(2)
            .invoke_arrays(arrays(vec![vec![Some(uint8(0, 0.0))], vec![Some(other)]]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("raster 2 is 2 x 2, but the first raster is 3 x 2"),
            "{err}"
        );
    }

    #[test]
    fn scalar_arguments_give_a_scalar() {
        let result = tester(2)
            .invoke(vec![
                ColumnarValue::Scalar(uint8(1, 0.0).scalar()),
                ColumnarValue::Scalar(uint8(2, 0.0).scalar()),
            ])
            .unwrap();
        let ColumnarValue::Scalar(scalar) = result else {
            panic!("expected a scalar result");
        };
        assert_raster_scalar_equals(&scalar, &uint8(1, 0.0).band_values(&[2u8, 3, 4, 5, 6, 7]));
    }

    #[test]
    fn outdb_bands_are_carried_by_reference() {
        let outdb = RasterSpec::d2(3, 2)
            .band(BandDataType::UInt8)
            .outdb("s3://bucket/r.tif", Some("geotiff"));
        let result = tester(2)
            .invoke_arrays(arrays(vec![vec![Some(uint8(1, 0.0))], vec![Some(outdb)]]))
            .unwrap();
        let expected = uint8(1, 0.0)
            .band(BandDataType::UInt8)
            .outdb("s3://bucket/r.tif", Some("geotiff"));
        assert_rasters_equal(&result, &[Some(expected)]);
    }
}
