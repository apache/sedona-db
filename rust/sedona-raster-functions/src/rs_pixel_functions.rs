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
use std::vec;

use crate::executor::RasterExecutor;
use arrow_array::builder::{BinaryBuilder, StringViewBuilder};
use arrow_array::{cast::AsArray, types::Int32Type};
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::item_crs::make_item_crs;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::{write_wkb_point, write_wkb_polygon, WKB_MIN_PROBABLE_BYTES};
use sedona_raster::affine_transformation::{to_world_coordinate, AffineMatrix};
use sedona_raster::traits::RasterRef;
use sedona_schema::datatypes::Edges;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

// ===========================================================================
// RS_PixelAsPoint
// ===========================================================================

/// RS_PixelAsPoint(raster, colX, rowY) scalar UDF implementation
///
/// Returns the upper-left corner of the specified pixel as a Point geometry.
/// The pixel coordinates are 1-based. Extrapolates for out-of-bounds coordinates.
pub fn rs_pixelaspoint_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_pixelaspoint",
        vec![Arc::new(RsPixelAsPoint {})],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsPixelAsPoint {}

impl SedonaScalarKernel for RsPixelAsPoint {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let out_type = SedonaType::new_item_crs(&SedonaType::Wkb(Edges::Planar, None))?;
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
            ],
            out_type,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let col_array = args[1].clone().into_array(executor.num_iterations())?;
        let col_array = col_array.as_primitive::<Int32Type>();
        let row_array = args[2].clone().into_array(executor.num_iterations())?;
        let row_array = row_array.as_primitive::<Int32Type>();

        let bytes_per_point = WKB_MIN_PROBABLE_BYTES;
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            executor.num_iterations() * bytes_per_point,
        );
        let mut crs_builder = StringViewBuilder::with_capacity(executor.num_iterations());

        let mut col_iter = col_array.iter();
        let mut row_iter = row_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let col_x = col_iter.next().unwrap();
            let row_y = row_iter.next().unwrap();
            match (raster_opt, col_x, row_y) {
                (Some(raster), Some(col_x), Some(row_y)) => {
                    // Convert to 0-based for the affine transform
                    let (wx, wy) =
                        to_world_coordinate(raster, (col_x - 1) as i64, (row_y - 1) as i64);

                    write_wkb_point(&mut builder, (wx, wy))
                        .map_err(|e| DataFusionError::External(e.into()))?;
                    builder.append_value([]);
                    crs_builder.append_value(raster.crs().unwrap_or("0"));
                }
                _ => {
                    builder.append_null();
                    crs_builder.append_null();
                }
            }
            Ok(())
        })?;

        let item_array = builder.finish();
        let item_result = executor.finish(Arc::new(item_array))?;
        let crs_array = crs_builder.finish();
        let crs_value = if matches!(item_result, ColumnarValue::Scalar(_)) {
            ColumnarValue::Scalar(ScalarValue::try_from_array(&crs_array, 0)?)
        } else {
            ColumnarValue::Array(Arc::new(crs_array))
        };

        make_item_crs(
            &SedonaType::Wkb(Edges::Planar, None),
            item_result,
            &crs_value,
            None,
        )
    }
}

// ===========================================================================
// RS_PixelAsCentroid
// ===========================================================================

/// RS_PixelAsCentroid(raster, colX, rowY) scalar UDF implementation
///
/// Returns the centroid of the specified pixel as a Point geometry.
/// The pixel coordinates are 1-based. Extrapolates for out-of-bounds coordinates.
pub fn rs_pixelascentroid_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_pixelascentroid",
        vec![Arc::new(RsPixelAsCentroid {})],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsPixelAsCentroid {}

impl SedonaScalarKernel for RsPixelAsCentroid {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let out_type = SedonaType::new_item_crs(&SedonaType::Wkb(Edges::Planar, None))?;
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
            ],
            out_type,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let col_array = args[1].clone().into_array(executor.num_iterations())?;
        let col_array = col_array.as_primitive::<Int32Type>();
        let row_array = args[2].clone().into_array(executor.num_iterations())?;
        let row_array = row_array.as_primitive::<Int32Type>();

        let bytes_per_point = WKB_MIN_PROBABLE_BYTES;
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            executor.num_iterations() * bytes_per_point,
        );
        let mut crs_builder = StringViewBuilder::with_capacity(executor.num_iterations());

        let mut col_iter = col_array.iter();
        let mut row_iter = row_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let col_x = col_iter.next().unwrap();
            let row_y = row_iter.next().unwrap();
            match (raster_opt, col_x, row_y) {
                (Some(raster), Some(col_x), Some(row_y)) => {
                    // Centroid: use 0.5 offset within the pixel
                    let grid_x = (col_x - 1) as f64 + 0.5;
                    let grid_y = (row_y - 1) as f64 + 0.5;

                    let affine = AffineMatrix::from_metadata(raster.metadata());
                    let (wx, wy) = affine.transform(grid_x, grid_y);

                    write_wkb_point(&mut builder, (wx, wy))
                        .map_err(|e| DataFusionError::External(e.into()))?;
                    builder.append_value([]);
                    crs_builder.append_value(raster.crs().unwrap_or("0"));
                }
                _ => {
                    builder.append_null();
                    crs_builder.append_null();
                }
            }
            Ok(())
        })?;

        let item_array = builder.finish();
        let item_result = executor.finish(Arc::new(item_array))?;
        let crs_array = crs_builder.finish();
        let crs_value = if matches!(item_result, ColumnarValue::Scalar(_)) {
            ColumnarValue::Scalar(ScalarValue::try_from_array(&crs_array, 0)?)
        } else {
            ColumnarValue::Array(Arc::new(crs_array))
        };

        make_item_crs(
            &SedonaType::Wkb(Edges::Planar, None),
            item_result,
            &crs_value,
            None,
        )
    }
}

// ===========================================================================
// RS_PixelAsPolygon
// ===========================================================================

/// RS_PixelAsPolygon(raster, colX, rowY) scalar UDF implementation
///
/// Returns the bounding polygon of the specified pixel.
/// The pixel coordinates are 1-based. Extrapolates for out-of-bounds coordinates.
pub fn rs_pixelaspolygon_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_pixelaspolygon",
        vec![Arc::new(RsPixelAsPolygon {})],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsPixelAsPolygon {}

impl SedonaScalarKernel for RsPixelAsPolygon {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let out_type = SedonaType::new_item_crs(&SedonaType::Wkb(Edges::Planar, None))?;
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
            ],
            out_type,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let col_array = args[1].clone().into_array(executor.num_iterations())?;
        let col_array = col_array.as_primitive::<Int32Type>();
        let row_array = args[2].clone().into_array(executor.num_iterations())?;
        let row_array = row_array.as_primitive::<Int32Type>();

        // 1 (byte order) + 4 (type) + 4 (num rings) + 4 (num points) + 80 (5 pts * 16)
        let bytes_per_poly = 93;
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            executor.num_iterations() * bytes_per_poly,
        );
        let mut crs_builder = StringViewBuilder::with_capacity(executor.num_iterations());

        let mut col_iter = col_array.iter();
        let mut row_iter = row_array.iter();
        executor.execute_raster_void(|_, raster_opt| {
            let col_x = col_iter.next().unwrap();
            let row_y = row_iter.next().unwrap();
            match (raster_opt, col_x, row_y) {
                (Some(raster), Some(col_x), Some(row_y)) => {
                    let col_x = col_x as i64;
                    let row_y = row_y as i64;

                    // 4 corners in 0-based grid coords, then transformed to world coords
                    // Upper-left: (colX-1, rowY-1), Upper-right: (colX, rowY-1)
                    // Lower-right: (colX, rowY), Lower-left: (colX-1, rowY)
                    let ul = to_world_coordinate(raster, col_x - 1, row_y - 1);
                    let ur = to_world_coordinate(raster, col_x, row_y - 1);
                    let lr = to_world_coordinate(raster, col_x, row_y);
                    let ll = to_world_coordinate(raster, col_x - 1, row_y);

                    write_wkb_polygon(&mut builder, [ul, ur, lr, ll, ul].into_iter())
                        .map_err(|e| DataFusionError::External(e.into()))?;
                    builder.append_value([]);
                    crs_builder.append_value(raster.crs().unwrap_or("0"));
                }
                _ => {
                    builder.append_null();
                    crs_builder.append_null();
                }
            }
            Ok(())
        })?;

        let item_array = builder.finish();
        let item_result = executor.finish(Arc::new(item_array))?;
        let crs_array = crs_builder.finish();
        let crs_value = if matches!(item_result, ColumnarValue::Scalar(_)) {
            ColumnarValue::Scalar(ScalarValue::try_from_array(&crs_array, 0)?)
        } else {
            ColumnarValue::Array(Arc::new(crs_array))
        };

        make_item_crs(
            &SedonaType::Wkb(Edges::Planar, None),
            item_result,
            &crs_value,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::{RASTER, WKB_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array_item_crs;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    // -----------------------------------------------------------------------
    // RS_PixelAsPoint tests
    // -----------------------------------------------------------------------

    #[test]
    fn udf_pixelaspoint_metadata() {
        let udf: ScalarUDF = rs_pixelaspoint_udf().into();
        assert_eq!(udf.name(), "rs_pixelaspoint");
    }

    #[test]
    fn udf_pixelaspoint_invoke() {
        let udf = rs_pixelaspoint_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(arrow_schema::DataType::Int32),
                SedonaType::Arrow(arrow_schema::DataType::Int32),
            ],
        );

        // Raster index 0: width=1, height=2, UL=(1,2), scale=(0.1,-0.2), skew=(0,0)
        // Pixel (1,1) -> grid(0,0) -> world(1.0, 2.0)
        // Raster index 1: null
        // Raster index 2: width=3, height=4, UL=(3,4), scale=(0.2,-0.4), skew=(0.06,0.08)
        // Pixel (1,1) -> grid(0,0) -> world(3.0, 4.0)
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let cols = Int32Array::from(vec![1, 1, 1]);
        let rows = Int32Array::from(vec![1, 1, 1]);

        let expected = &create_array_item_crs(
            &[Some("POINT (1.0 2.0)"), None, Some("POINT (3.0 4.0)")],
            [Some("OGC:CRS84"), None, Some("OGC:CRS84")],
            &WKB_GEOMETRY,
        );

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(cols), Arc::new(rows)])
            .unwrap();

        assert_array_equal(&result, expected);
    }

    #[test]
    fn udf_pixelaspoint_out_of_bounds_extrapolates() {
        let udf = rs_pixelaspoint_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(arrow_schema::DataType::Int32),
                SedonaType::Arrow(arrow_schema::DataType::Int32),
            ],
        );

        // Raster 0 has width=1, height=2, UL=(1,2), scale=(0.1,-0.2), skew=(0,0)
        // Pixel (2,1) is out of bounds (width=1), but should extrapolate:
        // grid(1,0) -> world(1 + 1*0.1, 2 + 0) = (1.1, 2.0)
        let rasters = generate_test_rasters(1, None).unwrap();
        let cols = Int32Array::from(vec![2]); // out of bounds (width=1)
        let rows = Int32Array::from(vec![1]);

        let expected = &create_array_item_crs(
            &[Some("POINT (1.1 2.0)")],
            [Some("OGC:CRS84")],
            &WKB_GEOMETRY,
        );

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(cols), Arc::new(rows)])
            .unwrap();

        assert_array_equal(&result, expected);
    }

    // -----------------------------------------------------------------------
    // RS_PixelAsCentroid tests
    // -----------------------------------------------------------------------

    #[test]
    fn udf_pixelascentroid_metadata() {
        let udf: ScalarUDF = rs_pixelascentroid_udf().into();
        assert_eq!(udf.name(), "rs_pixelascentroid");
    }

    #[test]
    fn udf_pixelascentroid_invoke() {
        let udf = rs_pixelascentroid_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(arrow_schema::DataType::Int32),
                SedonaType::Arrow(arrow_schema::DataType::Int32),
            ],
        );

        // Raster 0: width=1, height=2, UL=(1,2), scale=(0.1,-0.2), skew=(0,0)
        // Pixel (1,1) centroid: grid(0.5, 0.5)
        //   wx = 1.0 + 0.5*0.1 + 0.5*0.0 = 1.05
        //   wy = 2.0 + 0.5*0.0 + 0.5*(-0.2) = 1.9
        // Raster 2: width=3, height=4, UL=(3,4), scale=(0.2,-0.4), skew=(0.06,0.08)
        // Pixel (1,1) centroid: grid(0.5, 0.5)
        //   wx = 3.0 + 0.5*0.2 + 0.5*0.06 = 3.13
        //   wy = 4.0 + 0.5*0.08 + 0.5*(-0.4) = 3.84
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let cols = Int32Array::from(vec![1, 1, 1]);
        let rows = Int32Array::from(vec![1, 1, 1]);

        let expected = &create_array_item_crs(
            &[Some("POINT (1.05 1.9)"), None, Some("POINT (3.13 3.84)")],
            [Some("OGC:CRS84"), None, Some("OGC:CRS84")],
            &WKB_GEOMETRY,
        );

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(cols), Arc::new(rows)])
            .unwrap();

        assert_array_equal(&result, expected);
    }

    // -----------------------------------------------------------------------
    // RS_PixelAsPolygon tests
    // -----------------------------------------------------------------------

    #[test]
    fn udf_pixelaspolygon_metadata() {
        let udf: ScalarUDF = rs_pixelaspolygon_udf().into();
        assert_eq!(udf.name(), "rs_pixelaspolygon");
    }

    #[test]
    fn udf_pixelaspolygon_invoke() {
        let udf = rs_pixelaspolygon_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(arrow_schema::DataType::Int32),
                SedonaType::Arrow(arrow_schema::DataType::Int32),
            ],
        );

        // Raster 0: width=1, height=2, UL=(1,2), scale=(0.1,-0.2), skew=(0,0)
        // Pixel (1,1) corners in 0-based grid coords: (0,0),(1,0),(1,1),(0,1)
        //   (0,0) -> (1.0, 2.0)
        //   (1,0) -> (1.1, 2.0)
        //   (1,1) -> (1.1, 1.8)
        //   (0,1) -> (1.0, 1.8)
        let rasters = generate_test_rasters(1, None).unwrap();
        let cols = Int32Array::from(vec![1]);
        let rows = Int32Array::from(vec![1]);

        let expected = &create_array_item_crs(
            &[Some(
                "POLYGON ((1.0 2.0, 1.1 2.0, 1.1 1.8, 1.0 1.8, 1.0 2.0))",
            )],
            [Some("OGC:CRS84")],
            &WKB_GEOMETRY,
        );

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(cols), Arc::new(rows)])
            .unwrap();

        assert_array_equal(&result, expected);
    }
}
