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

//! `RS_Stack` — the bands of several rasters, in order, as one raster.
//!
//! ```text
//! RS_Stack(raster1[, raster2, ...])  -> Raster
//! ```
//!
//! Sedona Spark 1.9 calls this function `RS_Union`; it stacks bands and does
//! not merge grids the way a raster union does, hence the name.
//!
//! Every band of `raster1`, then every band of `raster2`, and so on, each
//! keeping its own pixel type, nodata value and name. Any number of rasters can
//! be stacked, but they must all be on one grid: the same width, height,
//! geotransform and CRS. A band's spatial dimensions take the first raster's
//! names (`lat`/`lon` bands stacked onto a `y`/`x` raster become `y`/`x`),
//! matched by role rather than position; a band with a non-spatial dimension
//! already bearing one of those names is an error. Any NULL raster gives a
//! NULL result.
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

use crate::crs_utils::resolve_crs;
use crate::executor::RasterExecutor;

/// `RS_Stack()` scalar UDF — the bands of several rasters as one raster.
pub fn rs_stack_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new("rs_stack", vec![Arc::new(RsStack)], Volatility::Immutable)
}

/// One kernel for any number of rasters (at least one).
#[derive(Debug)]
struct RsStack;

impl SedonaScalarKernel for RsStack {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // A raster matcher per argument, so each argument type-checks as any
        // raster argument does; with no arguments this expects one and fails.
        let matchers = (0..args.len().max(1))
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
            stack(&mut builder, &row)?;
        }

        RasterExecutor::finish_over(args, Arc::new(builder.finish()?))
    }
}

/// Append one raster holding every band of `rasters`, in order, under the first
/// raster's header.
fn stack(builder: &mut RasterBuilder, rasters: &[&dyn RasterRef]) -> Result<()> {
    let first = rasters[0];
    for (k, raster) in rasters.iter().enumerate().skip(1) {
        check_same_grid(first, *raster, k + 1)?;
    }

    let (x_dim, y_dim) = (first.x_dim(), first.y_dim());
    builder.start_raster_from(first, RasterOverrides::default())?;
    for (k, raster) in rasters.iter().enumerate() {
        let (from_x, from_y) = (raster.x_dim(), raster.y_dim());
        for band_idx in 0..raster.num_bands() {
            let band = raster.band(band_idx)?;
            // Rename the band's spatial dimensions to the first raster's,
            // matching each by role (x to x, y to y) so no order is assumed.
            // A non-spatial dimension already holding one of those names
            // would leave the band with two dimensions of the same name.
            let dim_names = band
                .dim_names()
                .into_iter()
                .map(|dim| match dim {
                    d if d == from_x => Ok(x_dim),
                    d if d == from_y => Ok(y_dim),
                    d if d == x_dim || d == y_dim => exec_err!(
                        "RS_Stack: band {} of raster {} has a non-spatial dimension named \
                         '{d}', the name of one of the first raster's spatial dimensions",
                        band_idx + 1,
                        k + 1
                    ),
                    d => Ok(d),
                })
                .collect::<Result<Vec<&str>>>()?;
            let renamed = dim_names != band.dim_names();
            band.copy_into(
                builder,
                BandOverrides {
                    dim_names: renamed.then_some(dim_names.as_slice()),
                    ..Default::default()
                },
            )?;
            builder.finish_band()?;
        }
    }
    builder.finish_raster()?;
    Ok(())
}

/// Error unless `raster` (the `k`th argument) is on the first raster's grid:
/// the same width, height, geotransform and CRS.
fn check_same_grid(first: &dyn RasterRef, raster: &dyn RasterRef, k: usize) -> Result<()> {
    let (width, height) = (first.width()?, first.height()?);
    let (w, h) = (raster.width()?, raster.height()?);
    if (w, h) != (width, height) {
        return exec_err!(
            "RS_Stack: raster {k} is {w} x {h}, but the first raster is {width} x {height}; \
             every raster must be on the same grid"
        );
    }
    if raster.transform() != first.transform() {
        return exec_err!(
            "RS_Stack: raster {k} has geotransform {:?}, but the first raster has {:?}; \
             every raster must be on the same grid",
            raster.transform(),
            first.transform()
        );
    }
    // Identical CRS strings are the common case; only differing strings are
    // parsed, to compare CRSes spelled differently (e.g. EPSG code vs PROJJSON).
    let same_crs = raster.crs() == first.crs()
        || match (resolve_crs(first.crs())?, resolve_crs(raster.crs())?) {
            (None, None) => true,
            (Some(a), Some(b)) => a.crs_equals(b.as_ref()),
            _ => false,
        };
    if !same_crs {
        return exec_err!(
            "RS_Stack: raster {k} has a different CRS than the first raster; every raster must \
             be on the same grid"
        );
    }
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
        ScalarUdfTester::new(rs_stack_udf().into(), vec![RASTER; num_rasters])
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
        let udf: ScalarUDF = rs_stack_udf().into();
        assert_eq!(udf.name(), "rs_stack");
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
    fn rasters_off_the_first_grid_error() {
        let stack_err = |second: RasterSpec| {
            tester(2)
                .invoke_arrays(arrays(vec![vec![Some(uint8(1, 0.0))], vec![Some(second)]]))
                .unwrap_err()
                .to_string()
        };
        // Shifted georeference.
        let err = stack_err(uint8(2, 5.0));
        assert!(err.contains("raster 2 has geotransform"), "{err}");
        // Different CRS, and a CRS on only one side.
        let err = stack_err(uint8(2, 0.0).crs(Some("EPSG:3857")));
        assert!(err.contains("raster 2 has a different CRS"), "{err}");
        let err = stack_err(uint8(2, 0.0).crs(None));
        assert!(err.contains("raster 2 has a different CRS"), "{err}");
    }

    #[test]
    fn spatial_dimensions_take_the_first_rasters_names() {
        // Jia's case: a lat/lon raster stacked onto a y/x one.
        let first = RasterSpec::nd(&["y", "x"], &[2, 3]).band_values(&[1u8, 2, 3, 4, 5, 6]);
        let second = RasterSpec::nd(&["lat", "lon"], &[2, 3]).band_values(&[7u8, 8, 9, 10, 11, 12]);
        let result = tester(2)
            .invoke_arrays(arrays(vec![vec![Some(first)], vec![Some(second)]]))
            .unwrap();
        let expected = RasterSpec::nd(&["y", "x"], &[2, 3])
            .band_values(&[1u8, 2, 3, 4, 5, 6])
            .band_values(&[7u8, 8, 9, 10, 11, 12]);
        assert_rasters_equal(&result, &[Some(expected)]);

        // Non-spatial dimensions keep their names.
        let first =
            RasterSpec::nd(&["time", "y", "x"], &[1, 2, 3]).band_values(&[1u8, 2, 3, 4, 5, 6]);
        let second = RasterSpec::nd(&["time", "lat", "lon"], &[1, 2, 3])
            .band_values(&[7u8, 8, 9, 10, 11, 12]);
        let result = tester(2)
            .invoke_arrays(arrays(vec![vec![Some(first)], vec![Some(second)]]))
            .unwrap();
        let expected = RasterSpec::nd(&["time", "y", "x"], &[1, 2, 3])
            .band_values(&[1u8, 2, 3, 4, 5, 6])
            .band_values(&[7u8, 8, 9, 10, 11, 12]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn non_spatial_dimension_named_like_a_spatial_one_errors() {
        // Renaming lat/lon to y/x would give this band two dimensions named x.
        let first = RasterSpec::nd(&["y", "x"], &[2, 3]).band_values(&[1u8, 2, 3, 4, 5, 6]);
        let second =
            RasterSpec::nd(&["x", "lat", "lon"], &[1, 2, 3]).band_values(&[7u8, 8, 9, 10, 11, 12]);
        let err = tester(2)
            .invoke_arrays(arrays(vec![vec![Some(first)], vec![Some(second)]]))
            .unwrap_err();
        assert!(
            err.to_string().contains(
                "band 1 of raster 2 has a non-spatial dimension named 'x', \
                 the name of one of the first raster's spatial dimensions"
            ),
            "{err}"
        );
    }

    #[test]
    fn stacks_any_number_of_rasters() {
        // Past Sedona Spark's limit of seven.
        let rows = (0..9u8).map(|k| vec![Some(uint8(k * 10, 0.0))]).collect();
        let result = tester(9).invoke_arrays(arrays(rows)).unwrap();
        let expected = (1..9u8).fold(uint8(0, 0.0), |spec, k| {
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

        // One raster stacks to itself.
        let result = tester(1)
            .invoke_arrays(arrays(vec![vec![Some(uint8(3, 0.0))]]))
            .unwrap();
        assert_rasters_equal(&result, &[Some(uint8(3, 0.0))]);
    }

    #[test]
    fn needs_at_least_one_raster() {
        assert_eq!(RsStack.return_type(&[]).unwrap(), None);
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
            err.contains("raster 2 is 2 x 2, but the first raster is 3 x 2; every raster must be on the same grid"),
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
