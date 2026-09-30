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

//! `RS_MinConvexHull` — the footprint of a raster's data pixels.
//!
//! ```text
//! RS_MinConvexHull(raster)        -> Geometry  -- data in any band
//! RS_MinConvexHull(raster, band)  -> Geometry  -- data in one band
//! ```
//!
//! Returns the polygon over the smallest pixel-aligned rectangle of the grid
//! that holds every data (non-nodata) pixel, in world coordinates: the analogue
//! of `RS_ConvexHull` with the nodata margin trimmed off. Without a band a pixel
//! counts as data when any band holds data there. The result is `NULL` when no
//! pixel holds data.

use std::ops::ControlFlow;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::builder::{BinaryBuilder, StringViewBuilder};
use datafusion_common::cast::as_int32_array;
use datafusion_common::{Result, ScalarValue, exec_datafusion_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::item_crs::make_item_crs;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::types::Edges;
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_raster::traits::{NdBuffer, RasterRef};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::RasterExecutor;
use crate::footprint::write_footprint_wkb;
use crate::pixel_scan::{NodataMatcher, scan_pixels, spatial_2d_buffer};
use crate::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use crate::sampling::{int32_array_arg, resolve_band};

/// `RS_MinConvexHull()` scalar UDF — the footprint of a raster's data pixels.
pub fn rs_minconvexhull_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_minconvexhull",
        vec![
            Arc::new(RsMinConvexHull { with_band: false }), // RS_MinConvexHull(raster)
            Arc::new(RsMinConvexHull { with_band: true }),  // RS_MinConvexHull(raster, band)
        ],
        Volatility::Immutable,
    )
    // The kernel reads pixel bytes, so the raster argument must be materialised
    // InDb first; the planner injects RS_EnsureLoaded based on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsMinConvexHull {
    with_band: bool,
}

impl SedonaScalarKernel for RsMinConvexHull {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let out_type = SedonaType::new_item_crs(&SedonaType::Wkb(Edges::Planar, None))?;
        let mut matchers = vec![ArgMatcher::is_raster()];
        if self.with_band {
            matchers.push(ArgMatcher::is_integer());
        }
        ArgMatcher::new(matchers, out_type).match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();
        // 1 (byte order) + 4 (type) + 4 (num rings) + 4 (num points) + 80 (5 points * 16 bytes)
        let bytes_per_poly = 93;
        let mut builder =
            BinaryBuilder::with_capacity(num_iterations, num_iterations * bytes_per_poly);
        let mut crs_builder = StringViewBuilder::with_capacity(num_iterations);

        let band_arr = if self.with_band {
            Some(int32_array_arg(&args[1], num_iterations)?)
        } else {
            None
        };
        let band = band_arr.as_ref().map(|a| as_int32_array(a)).transpose()?;

        executor.execute_raster_void(|i, raster_opt| {
            let Some(raster) = raster_opt else {
                builder.append_null();
                crs_builder.append_null();
                return Ok(());
            };
            let bounds = match band {
                None => data_bounds(raster, 1..=raster.num_bands())?,
                Some(band) if band.is_null(i) => None,
                // Clamp a negative band to 0 so resolve_band rejects it as not
                // 1-based rather than wrapping it into a huge usize.
                Some(band) => data_bounds(raster, [band.value(i).max(0) as usize])?,
            };
            match bounds {
                Some(bounds) => {
                    write_footprint_wkb(bounds.corners(raster), &mut builder)?;
                    builder.append_value([]);
                    crs_builder.append_value(raster.crs().unwrap_or("0"));
                }
                None => {
                    builder.append_null();
                    crs_builder.append_null();
                }
            }
            Ok(())
        })?;

        let item_result = executor.finish(Arc::new(builder.finish()))?;
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

/// An inclusive, 0-based range of grid cells: columns `min_col..=max_col` of
/// rows `min_row..=max_row`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PixelBounds {
    min_col: i64,
    min_row: i64,
    max_col: i64,
    max_row: i64,
}

impl PixelBounds {
    fn single(col: i64, row: i64) -> Self {
        Self {
            min_col: col,
            min_row: row,
            max_col: col,
            max_row: row,
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            min_col: self.min_col.min(other.min_col),
            min_row: self.min_row.min(other.min_row),
            max_col: self.max_col.max(other.max_col),
            max_row: self.max_row.max(other.max_row),
        }
    }

    /// World coordinates of the outer corners of the cells, in footprint ring
    /// order: upper-left, upper-right, lower-right, lower-left. A cell's far
    /// corners are the near corners of the next column and row.
    fn corners(&self, raster: &dyn RasterRef) -> [(f64, f64); 4] {
        [
            to_world_coordinate(raster, self.min_col, self.min_row),
            to_world_coordinate(raster, self.max_col + 1, self.min_row),
            to_world_coordinate(raster, self.max_col + 1, self.max_row + 1),
            to_world_coordinate(raster, self.min_col, self.max_row + 1),
        ]
    }
}

const FUNC: &str = "RS_MinConvexHull";

/// The cells holding data in any of the 1-based bands `band_nums`, or `None`
/// when no pixel of those bands holds data.
fn data_bounds(
    raster: &dyn RasterRef,
    band_nums: impl IntoIterator<Item = usize>,
) -> Result<Option<PixelBounds>> {
    let mut bounds: Option<PixelBounds> = None;
    for band_num in band_nums {
        // Resolve every band, even once the grid is covered, so a bad band
        // errors regardless of what the earlier bands hold.
        let band = resolve_band(FUNC, raster, band_num)?;
        let buffer = spatial_2d_buffer(FUNC, band.as_ref())?;
        let nodata = NodataMatcher::for_band(FUNC, band.as_ref())?;
        let (height, width) = (buffer.shape[0], buffer.shape[1]);
        if height <= 0 || width <= 0 {
            continue;
        }
        let grid = PixelBounds::single(0, 0).union(PixelBounds::single(width - 1, height - 1));
        if bounds == Some(grid) {
            // The grid is already covered, so no band can widen it.
            continue;
        }
        let band_bounds = match nodata {
            // Without a nodata value every pixel is data.
            None => Some(grid),
            Some(nodata) => band_data_bounds(&buffer, &nodata)?,
        };
        if let Some(band_bounds) = band_bounds {
            bounds = Some(bounds.map_or(band_bounds, |b| b.union(band_bounds)));
        }
    }
    Ok(bounds)
}

/// The cells of one band holding data, or `None` when every pixel is nodata.
///
/// Scans inward from each edge and stops at the first data pixel: rows down
/// from the top, rows up from the bottom, then columns in from the left and
/// right over the rows in between. A band that is mostly data costs a handful
/// of pixels rather than a full scan; a band that is mostly nodata costs at
/// most about two full scans. The bottom, left and right scans read the band
/// through reversed and transposed views of the same buffer.
fn band_data_bounds(buffer: &NdBuffer, nodata: &NodataMatcher) -> Result<Option<PixelBounds>> {
    // The first data pixel of `view` in row-major order, as its view-space
    // (column, row).
    let first_data = |view: &NdBuffer| -> Result<Option<(i64, i64)>> {
        let mut hit = None;
        scan_pixels(FUNC, view, |col, row, pixel| {
            if nodata.matches(pixel) {
                ControlFlow::Continue(())
            } else {
                hit = Some((col, row));
                ControlFlow::Break(())
            }
        })?;
        Ok(hit)
    };

    // The top scan runs over the band's own view, so it also bounds-checks
    // every byte the derived views below can reach.
    let Some((_, min_row)) = first_data(buffer)? else {
        return Ok(None);
    };
    let (height, width) = (buffer.shape[0], buffer.shape[1]);
    let (row_stride, col_stride) = (buffer.strides[0], buffer.strides[1]);
    let offset = buffer.offset as i64;
    let view = |offset: i64, shape: [i64; 2], strides: [i64; 2]| NdBuffer {
        buffer: buffer.buffer,
        shape: shape.to_vec(),
        strides: strides.to_vec(),
        offset: offset as u64,
        data_type: buffer.data_type,
    };
    // A data pixel exists in row `min_row`, so every scan below hits one.
    let found = || exec_datafusion_err!("{FUNC}: data pixel vanished between scans");

    // Rows from the bottom, up to `min_row`.
    let bottom_up = view(
        offset + (height - 1) * row_stride,
        [height - min_row, width],
        [-row_stride, col_stride],
    );
    let (_, rows_from_bottom) = first_data(&bottom_up)?.ok_or_else(found)?;
    let max_row = height - 1 - rows_from_bottom;

    // Columns (as view rows) over rows `min_row..=max_row`, from each side.
    let rows = max_row - min_row + 1;
    let left_in = view(
        offset + min_row * row_stride,
        [width, rows],
        [col_stride, row_stride],
    );
    let (_, min_col) = first_data(&left_in)?.ok_or_else(found)?;
    let right_in = view(
        offset + min_row * row_stride + (width - 1) * col_stride,
        [width - min_col, rows],
        [-col_stride, row_stride],
    );
    let (_, cols_from_right) = first_data(&right_in)?.ok_or_else(found)?;

    Ok(Some(PixelBounds {
        min_col,
        min_row,
        max_col: width - 1 - cols_from_right,
        max_row,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::{RASTER, WKB_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array_item_crs;
    use sedona_testing::raster_spec::{RasterSpec, raster_array};
    use sedona_testing::testers::ScalarUdfTester;

    fn bounds(spec: RasterSpec, band_nums: &[usize]) -> Result<Option<PixelBounds>> {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        data_bounds(&rasters.get(0).unwrap(), band_nums.iter().copied())
    }

    fn cells(min_col: i64, min_row: i64, max_col: i64, max_row: i64) -> Option<PixelBounds> {
        Some(PixelBounds {
            min_col,
            min_row,
            max_col,
            max_row,
        })
    }

    /// A 4x3 band of nodata (0) with data at (col 1, row 0) and (col 2, row 1):
    ///
    /// ```text
    /// 0 5 0 0
    /// 0 0 7 0
    /// 0 0 0 0
    /// ```
    fn sparse_band() -> RasterSpec {
        RasterSpec::d2(4, 3)
            .band_values(&[0u8, 5, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0])
            .nodata(0u8)
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_minconvexhull_udf().into();
        assert_eq!(udf.name(), "rs_minconvexhull");
        assert_eq!(
            rs_minconvexhull_udf()
                .metadata()
                .get(NEEDS_PIXELS_METADATA_KEY)
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn trims_the_nodata_margin() {
        assert_eq!(bounds(sparse_band(), &[1]).unwrap(), cells(1, 0, 2, 1));
    }

    #[test]
    fn band_without_nodata_covers_the_grid() {
        let spec = RasterSpec::d2(4, 3).band_values(&[0u8; 12]);
        assert_eq!(bounds(spec, &[1]).unwrap(), cells(0, 0, 3, 2));
    }

    #[test]
    fn all_nodata_is_none() {
        let spec = RasterSpec::d2(2, 2).band_values(&[0u8; 4]).nodata(0u8);
        assert_eq!(bounds(spec, &[1]).unwrap(), None);
        // NaN pixels are nodata under a NaN nodata value.
        let spec = RasterSpec::d2(2, 1)
            .band_values(&[f32::NAN, f32::NAN])
            .nodata(f32::NAN);
        assert_eq!(bounds(spec, &[1]).unwrap(), None);
    }

    #[test]
    fn bands_union_their_data() {
        // Band 2 holds data only at (col 3, row 2).
        let spec = sparse_band()
            .band_values(&[0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9])
            .nodata(0u8);
        assert_eq!(bounds(spec.clone(), &[2]).unwrap(), cells(3, 2, 3, 2));
        assert_eq!(bounds(spec, &[1, 2]).unwrap(), cells(1, 0, 3, 2));
    }

    /// The edge-inward scan against a full scan over every data mask of a few
    /// small grids (all 2^12 masks of a 4x3 grid, plus strips).
    #[test]
    fn edge_scan_matches_a_full_scan() {
        for (width, height) in [(4usize, 3usize), (1, 5), (6, 1), (2, 2)] {
            let cells = width * height;
            for mask in 0u32..(1 << cells) {
                let pixels: Vec<u8> = (0..cells).map(|i| ((mask >> i) & 1) as u8).collect();
                let spec = RasterSpec::d2(width as i64, height as i64)
                    .band_values(&pixels)
                    .nodata(0u8);
                let expected = pixels
                    .iter()
                    .enumerate()
                    .filter(|&(_, &p)| p != 0)
                    .map(|(i, _)| PixelBounds::single((i % width) as i64, (i / width) as i64))
                    .reduce(PixelBounds::union);
                assert_eq!(
                    bounds(spec, &[1]).unwrap(),
                    expected,
                    "{width}x{height} mask {mask:b}"
                );
            }
        }
    }

    #[test]
    fn band_out_of_range_errors() {
        let err = bounds(sparse_band(), &[2]).unwrap_err().to_string();
        assert!(err.contains("RS_MinConvexHull"), "{err}");
        let err = bounds(sparse_band(), &[0]).unwrap_err().to_string();
        assert!(err.contains("1-based"), "{err}");
    }

    #[test]
    fn udf_invoke_places_cells_in_world_coordinates() {
        // Two-unit pixels from origin (10, 20), north-up: cells (1..=2, 0..=1)
        // span x 12..16 and y 20..16.
        let raster = sparse_band().bbox(10.0, 14.0, 18.0, 20.0);
        // Rotated grid: the corners follow the skewed axes, not the world axes.
        let skewed = sparse_band().transform([10.0, 2.0, 1.0, 20.0, 1.0, -2.0]);
        let all_nodata = RasterSpec::d2(1, 1).band_values(&[0u8]).nodata(0u8);
        let rasters = raster_array([Some(raster), Some(skewed), Some(all_nodata), None]);

        let udf: ScalarUDF = rs_minconvexhull_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let expected = create_array_item_crs(
            &[
                Some("POLYGON ((12 20, 16 20, 16 16, 12 16, 12 20))"),
                Some("POLYGON ((12 21, 16 23, 18 19, 14 17, 12 21))"),
                None,
                None,
            ],
            [Some("OGC:CRS84"), Some("OGC:CRS84"), None, None],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn udf_invoke_negative_band_errors() {
        let udf: ScalarUDF = rs_minconvexhull_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);
        let err = tester
            .invoke_arrays(vec![
                Arc::new(raster_array([Some(sparse_band())])),
                Arc::new(Int32Array::from(vec![-1])),
            ])
            .unwrap_err()
            .to_string();
        assert!(err.contains("1-based"), "{err}");
    }

    #[test]
    fn udf_invoke_with_band() {
        let spec = sparse_band()
            .band_values(&[0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9])
            .nodata(0u8);
        let rasters = raster_array([Some(spec.clone()), Some(spec)]);

        let udf: ScalarUDF = rs_minconvexhull_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(rasters),
                Arc::new(Int32Array::from(vec![Some(2), None])),
            ])
            .unwrap();
        // Default transform: unit pixels, north-up, origin (0, 0).
        let expected = create_array_item_crs(
            &[Some("POLYGON ((3 -2, 4 -2, 4 -3, 3 -3, 3 -2))"), None],
            [Some("OGC:CRS84"), None],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&result, &expected);
    }
}
