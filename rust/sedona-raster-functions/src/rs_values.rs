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

//! `RS_Values` — sample a raster's pixel value at each point of a MultiPoint.
//!
//! ```text
//! RS_Values(raster, points)        -> List<Double>  -- single-band rasters only
//! RS_Values(raster, points, band)  -> List<Double>
//! ```
//!
//! The plural companion of [`RS_Value`](crate::rs_value): where `RS_Value` takes
//! one Point and returns one Double, `RS_Values` takes a MultiPoint (a single
//! Point is also accepted) and returns a `List<Double>` — one element per
//! sub-point, in input order. Each element is `NULL` when its sub-point is empty,
//! out of bounds, or reads the band's nodata; the whole list is `NULL` when the
//! raster, geometry, or band is `NULL`. An empty MultiPoint yields an empty list.
//!
//! Sampling, CRS handling, and pixel decoding are shared with `RS_Value` via
//! [`crate::sampling`]; this module only adds the per-sub-point iteration and the
//! list-shaped output.
//!
//! Like `RS_Value`, the function is tagged [`NEEDS_PIXELS_METADATA_KEY`] so the
//! planner materialises the raster InDb before a kernel runs, and only 2-D
//! rasters are supported.

use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder};
use arrow_schema::{DataType, Field};
use datafusion_common::cast::as_int32_array;
use datafusion_common::{exec_datafusion_err, exec_err, Result};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{CoordTrait, GeometryTrait, GeometryType, MultiPointTrait, PointTrait};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_proj::transform::with_global_proj_engine;
use sedona_raster::affine_transformation::AffineMatrix;
use sedona_raster::traits::{NdBuffer, RasterRef};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::read_wkb;

use crate::crs_utils::resolve_crs;
use crate::executor::RasterExecutor;
use crate::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use crate::sampling::{
    default_band, int32_array_arg, next_band, read_pixel, reproject_wkb, xy_to_pixel,
};

/// The `List<Float64>` output type, matching what a default
/// `ListBuilder<Float64Builder>` produces (field "item", nullable).
fn list_float64_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Float64, true)))
}

/// `RS_Values()` scalar UDF — sample pixel values at each point of a MultiPoint.
pub fn rs_values_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_values",
        vec![
            Arc::new(RsValues { with_band: false }), // RS_Values(raster, points)
            Arc::new(RsValues { with_band: true }),  // RS_Values(raster, points, band)
        ],
        Volatility::Immutable,
    )
    // The kernels read pixel bytes, so the raster argument must be materialised
    // InDb first; the planner injects RS_EnsureLoaded based on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

/// Kernel for `RS_Values(raster, points[, band])`.
#[derive(Debug)]
struct RsValues {
    with_band: bool,
}

impl SedonaScalarKernel for RsValues {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![
            ArgMatcher::is_raster(),
            ArgMatcher::is_geometry_or_geography(),
        ];
        if self.with_band {
            matchers.push(ArgMatcher::is_integer());
        }
        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(list_float64_type()));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();
        let mut list_builder = ListBuilder::new(Float64Builder::new());

        // The optional band argument, materialised once as an Int32 array. Held
        // as an `ArrayRef` so the typed view below borrows it instead of cloning
        // the typed `Int32Array`.
        let band_arr = if self.with_band {
            Some(int32_array_arg(&args[2], num_iterations)?)
        } else {
            None
        };
        let band_array = band_arr.as_ref().map(|a| as_int32_array(a)).transpose()?;
        let mut band_iter = band_array.map(|a| a.iter());

        // Reprojecting the points into the raster CRS needs a PROJ engine.
        with_global_proj_engine(|engine| {
            executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, geom_crs| {
                // Advance the band column every row so it stays in lockstep with
                // the row index (a no-op when there is no band argument).
                let band_arg = next_band(&mut band_iter);
                let (raster, geom_wkb) = match (raster_opt, wkb_opt) {
                    (Some(raster), Some(geom_wkb)) => (raster, geom_wkb),
                    // A NULL raster or geometry yields a NULL row.
                    _ => {
                        list_builder.append_null();
                        return Ok(());
                    }
                };

                // Resolve the band to sample. An explicit band column drives it
                // (a NULL element yields a NULL row); with no band argument it
                // defaults to band 1, but only for a single-band raster — sampling
                // an unspecified band of a multiband raster is ambiguous, so it
                // errors rather than silently picking band 1.
                let band_num = if self.with_band {
                    match band_arg {
                        Some(band_num) => band_num,
                        None => {
                            list_builder.append_null();
                            return Ok(());
                        }
                    }
                } else {
                    default_band("RS_Values", raster.num_bands())?
                };

                // Resolve the band buffer, nodata, and affine transform once for
                // this row, then sample every sub-point against them.
                let raster_crs = resolve_crs(raster.crs())?;
                let band = raster
                    .bands()
                    .band(band_num)
                    .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
                if !band.is_spatial_2d() {
                    return exec_err!(
                        "RS_Values supports 2-D rasters only; band is not a 2-D (y, x) grid"
                    );
                }
                let buffer = band
                    .nd_buffer()
                    .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
                let nodata = band
                    .nodata_as_f64()
                    .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
                let affine = AffineMatrix::from_metadata(&raster.metadata());

                // Reproject the whole geometry into the raster CRS (a no-op that
                // borrows the original bytes when the CRSes match or are absent).
                let reprojected = reproject_wkb(geom_wkb, geom_crs, raster_crs.as_deref(), engine)?;
                let effective = reprojected.as_deref().unwrap_or(geom_wkb);

                sample_points_into_list(effective, &affine, &buffer, nodata, &mut list_builder)?;
                list_builder.append(true);
                Ok(())
            })
        })?;

        executor.finish(Arc::new(list_builder.finish()))
    }
}

/// Sample each sub-point of a Point/MultiPoint WKB into one list row.
///
/// The WKB is assumed to already be in the raster's CRS (reprojected by the
/// caller). Each sub-point contributes one list element: its sampled value, or
/// `NULL` when the sub-point is empty, non-finite, out of bounds, or nodata. A
/// non-Point/MultiPoint geometry is an error.
fn sample_points_into_list(
    wkb: &[u8],
    affine: &AffineMatrix,
    buffer: &NdBuffer,
    nodata: Option<f64>,
    list_builder: &mut ListBuilder<Float64Builder>,
) -> Result<()> {
    let geom = read_wkb(wkb).map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
    match geom.as_type() {
        GeometryType::Point(point) => {
            let xy = point.coord().map(|c| (c.x(), c.y()));
            append_sample(xy, affine, buffer, nodata, list_builder)?;
        }
        GeometryType::MultiPoint(multi_point) => {
            for point in multi_point.points() {
                let xy = point.coord().map(|c| (c.x(), c.y()));
                append_sample(xy, affine, buffer, nodata, list_builder)?;
            }
        }
        _ => return exec_err!("RS_Values expects a Point or MultiPoint geometry"),
    }
    Ok(())
}

/// Append one sampled value (or NULL) to the current list row for a sub-point's
/// `(x, y)` in the raster CRS. `None` coordinates (an empty sub-point) and
/// out-of-bounds/nodata pixels both append a NULL element.
fn append_sample(
    xy: Option<(f64, f64)>,
    affine: &AffineMatrix,
    buffer: &NdBuffer,
    nodata: Option<f64>,
    list_builder: &mut ListBuilder<Float64Builder>,
) -> Result<()> {
    let sample = match xy {
        Some((x, y)) => match xy_to_pixel(affine, x, y)? {
            Some((col, row)) => read_pixel(buffer, nodata, col, row)?,
            None => None,
        },
        None => None,
    };
    match sample {
        Some(value) => list_builder.values().append_value(value),
        None => list_builder.values().append_null(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Float64Array, Int32Array, ListArray};
    use datafusion_expr::ScalarUDF;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::{Edges, RASTER};
    use sedona_schema::raster::BandDataType;
    use sedona_testing::create::create_array as create_geom_array;
    use sedona_testing::raster_spec::RasterSpec;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    /// North-up 2x2 raster, origin (0, 10), 1x1 pixels, band values row-major
    /// `[1, 2, 3, 4]` (row0 = [1, 2], row1 = [3, 4]). Pixel (0, 0) covers world
    /// x∈[0,1), y∈[9,10) and holds 1; pixel (1, 1) holds 4.
    fn raster_2x2() -> RasterSpec {
        RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
    }

    /// Extract one list row as a `Vec<Option<f64>>`.
    fn row(result: &dyn Array, i: usize) -> Vec<Option<f64>> {
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        let values = list.value(i);
        let values = values.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..values.len())
            .map(|j| (!values.is_null(j)).then(|| values.value(j)))
            .collect()
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_values_udf().into();
        assert_eq!(udf.name(), "rs_values");
    }

    #[test]
    fn udf_marks_needs_pixels() {
        assert_eq!(
            rs_values_udf()
                .metadata()
                .get(NEEDS_PIXELS_METADATA_KEY)
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn return_type_is_list_float64() {
        let return_type = RsValues { with_band: false }
            .return_type(&[RASTER, SedonaType::Wkb(Edges::Planar, lnglat())])
            .unwrap();
        assert_eq!(return_type, Some(SedonaType::Arrow(list_float64_type())));
    }

    #[test]
    fn multipoint_samples_each_point_in_order() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        // Pixel (0,0)=1, pixel (1,1)=4, and a point far outside -> NULL element.
        let geoms = create_geom_array(
            &[Some("MULTIPOINT (0.5 9.5, 1.5 8.5, 100 100)")],
            &geom_type,
        );
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms])
            .unwrap();
        assert_eq!(row(&result, 0), vec![Some(1.0), Some(4.0), None]);
    }

    #[test]
    fn single_point_yields_one_element_list() {
        // A plain Point is accepted and produces a one-element list.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        let geoms = create_geom_array(&[Some("POINT (1.5 8.5)")], &geom_type);
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms])
            .unwrap();
        assert_eq!(row(&result, 0), vec![Some(4.0)]);
    }

    #[test]
    fn empty_multipoint_yields_empty_list() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        let geoms = create_geom_array(&[Some("MULTIPOINT EMPTY")], &geom_type);
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms])
            .unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(
            !list.is_null(0),
            "empty MultiPoint is an empty list, not NULL"
        );
        assert_eq!(row(&result, 0), Vec::<Option<f64>>::new());
    }

    #[test]
    fn nodata_element_is_null() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        // Band [1, 2, 3, 4] with nodata=4: sampling pixel (1,1) reads nodata.
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .nodata(4u8)
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5, 1.5 8.5)")], &geom_type);
        let result = tester.invoke_arrays(vec![Arc::new(raster), geoms]).unwrap();
        assert_eq!(row(&result, 0), vec![Some(1.0), None]);
    }

    #[test]
    fn null_geometry_yields_null_list() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        let geoms = create_geom_array(&[None], &geom_type);
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms])
            .unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_null(0));
    }

    #[test]
    fn null_band_element_yields_null_list() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                geom_type.clone(),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5)")], &geom_type);
        let bands: arrow_array::ArrayRef = Arc::new(Int32Array::from(vec![None::<i32>]));
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms, bands])
            .unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_null(0));
    }

    #[test]
    fn second_band_is_addressable() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                geom_type.clone(),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        // Band 1 [1,2,3,4], band 2 [10,20,30,40]; sample band 2 at pixel (0,0).
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5)")], &geom_type);
        let bands: arrow_array::ArrayRef = Arc::new(Int32Array::from(vec![Some(2)]));
        let result = tester
            .invoke_arrays(vec![Arc::new(raster), geoms, bands])
            .unwrap();
        assert_eq!(row(&result, 0), vec![Some(10.0)]);
    }

    #[test]
    fn default_band_requires_single_band_raster() {
        // With no band argument, a multiband raster is ambiguous and errors
        // rather than silently sampling band 1 (matches RS_SetBandNoDataValue).
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(raster), geoms])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("specify which band"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn non_point_geometry_errors() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        let geoms = create_geom_array(&[Some("LINESTRING (0 0, 1 1)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![
                Arc::new(generate_test_rasters(1, None).unwrap()),
                geoms,
            ])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("expects a Point or MultiPoint"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn crs_mismatch_errors() {
        // Raster has a CRS (generate_test_rasters sets OGC:CRS84), points do not.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, None);
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let rasters = generate_test_rasters(1, None).unwrap();
        let geoms = create_geom_array(&[Some("MULTIPOINT (2.1 2.6)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("raster has a CRS but the point does not"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn non_2d_band_errors() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, None);
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let raster = RasterSpec::nd(&["time", "y", "x"], &[2, 2, 1])
            .band(BandDataType::UInt8)
            .crs(None)
            .build();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0 0)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(raster), geoms])
            .unwrap_err()
            .to_string();
        assert!(err.contains("2-D"), "unexpected error: {err}");
    }
}
