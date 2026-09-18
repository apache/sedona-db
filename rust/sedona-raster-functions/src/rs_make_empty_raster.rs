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

//! RS_MakeEmptyRaster — build an in-database raster from a grid definition.
//!
//! The grid can be given three ways, each with and without an explicit band
//! data type (default `float64`):
//!
//! - `(num_bands[, type], width, height, extent)` — the grid that covers the
//!   envelope of `extent` with `width` x `height` north-up pixels, in the
//!   geometry's CRS. The "abstract grid from ncol, nrow, bbox and crs" form.
//! - `(num_bands[, type], width, height, upper_left_x, upper_left_y,
//!   cell_size)` — square north-up pixels, no CRS.
//! - `(num_bands[, type], width, height, upper_left_x, upper_left_y, scale_x,
//!   scale_y, skew_x, skew_y, srid)` — the full affine form.
//!
//! The last two match Sedona Spark's `RS_MakeEmptyRaster` argument for
//! argument. Every band is zero-filled with no nodata value; `num_bands` may be
//! 0 for a bandless grid template.

use std::{collections::HashMap, sync::Arc};

use arrow_array::{Array, Float64Array, Int64Array, StringArray};
use arrow_buffer::{Buffer, MutableBuffer};
use arrow_schema::DataType;
use datafusion_common::{
    cast::{as_float64_array, as_int64_array, as_string_array},
    error::Result,
    exec_datafusion_err, exec_err,
};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::{
    item_crs::parse_item_crs_arg_type,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{bounds::wkb_bounds_xy, interval::IntervalTrait};
use sedona_raster::builder::RasterBuilder;
use sedona_schema::{
    crs::CachedSRIDToCrs, datatypes::SedonaType, matchers::ArgMatcher, raster::BandDataType,
};

use crate::{executor::RasterExecutor, pixel_type::parse_pixel_type};

/// RS_MakeEmptyRaster() scalar UDF implementation
pub fn rs_make_empty_raster_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_makeemptyraster",
        vec![
            Arc::new(RsMakeEmptyRaster::new(Grid::Extent, false)),
            Arc::new(RsMakeEmptyRaster::new(Grid::Extent, true)),
            Arc::new(RsMakeEmptyRaster::new(Grid::CellSize, false)),
            Arc::new(RsMakeEmptyRaster::new(Grid::CellSize, true)),
            Arc::new(RsMakeEmptyRaster::new(Grid::Affine, false)),
            Arc::new(RsMakeEmptyRaster::new(Grid::Affine, true)),
        ],
        Volatility::Immutable,
    )
}

/// How the grid's placement is specified after `num_bands[, type], width, height`.
#[derive(Debug, Clone, Copy)]
enum Grid {
    /// `extent: geometry` — envelope and CRS come from the geometry.
    Extent,
    /// `upper_left_x, upper_left_y, cell_size` — square pixels, no CRS.
    CellSize,
    /// `upper_left_x, upper_left_y, scale_x, scale_y, skew_x, skew_y, srid`.
    Affine,
}

#[derive(Debug)]
struct RsMakeEmptyRaster {
    grid: Grid,
    /// Whether a band data type string follows `num_bands`.
    typed: bool,
}

impl RsMakeEmptyRaster {
    fn new(grid: Grid, typed: bool) -> Self {
        Self { grid, typed }
    }

    /// Index of the first grid-placement argument (the one after `height`).
    fn grid_arg_index(&self) -> usize {
        if self.typed { 4 } else { 3 }
    }
}

impl SedonaScalarKernel for RsMakeEmptyRaster {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![ArgMatcher::is_integer()];
        if self.typed {
            matchers.push(ArgMatcher::is_string());
        }
        matchers.push(ArgMatcher::is_integer()); // width
        matchers.push(ArgMatcher::is_integer()); // height

        // A geometry extent may arrive as an item_crs struct (e.g. from
        // RS_Envelope); match on its item type like RS_AsRaster does.
        let mut arg_types = args.to_vec();
        match self.grid {
            Grid::Extent => {
                matchers.push(ArgMatcher::is_geometry());
                let idx = self.grid_arg_index();
                if let Some(extent_type) = arg_types.get(idx) {
                    let (item_type, _) = parse_item_crs_arg_type(extent_type)?;
                    arg_types[idx] = item_type;
                }
            }
            Grid::CellSize => {
                matchers.extend((0..3).map(|_| ArgMatcher::is_numeric()));
            }
            Grid::Affine => {
                matchers.extend((0..6).map(|_| ArgMatcher::is_numeric()));
                matchers.push(ArgMatcher::is_integer()); // srid
            }
        }

        ArgMatcher::new(matchers, SedonaType::Raster).match_args(&arg_types)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let n = executor.num_iterations();

        let num_bands = int_column(&args[0], n)?;
        let data_type = if self.typed {
            Some(string_column(&args[1], n)?)
        } else {
            None
        };
        let g = self.grid_arg_index();
        let width = int_column(&args[g - 2], n)?;
        let height = int_column(&args[g - 1], n)?;

        let mut placement = match self.grid {
            Grid::Extent => Placement::Extent(executor.make_geom_wkb_crs_accessor(g)?),
            Grid::CellSize => Placement::CellSize {
                upper_left_x: f64_column(&args[g], n)?,
                upper_left_y: f64_column(&args[g + 1], n)?,
                cell_size: f64_column(&args[g + 2], n)?,
            },
            Grid::Affine => Placement::Affine(Box::new(AffineColumns {
                upper_left_x: f64_column(&args[g], n)?,
                upper_left_y: f64_column(&args[g + 1], n)?,
                scale_x: f64_column(&args[g + 2], n)?,
                scale_y: f64_column(&args[g + 3], n)?,
                skew_x: f64_column(&args[g + 4], n)?,
                skew_y: f64_column(&args[g + 5], n)?,
                srid: int_column(&args[g + 6], n)?,
                srid_to_crs: CachedSRIDToCrs::new(),
            })),
        };

        let mut builder = RasterBuilder::new(n);
        // One zero buffer per distinct band byte length, shared zero-copy by
        // every band (and row) of that size: the output batch holds a single
        // block of zeros however many bands it describes.
        let mut zeros: HashMap<usize, Buffer> = HashMap::new();

        for i in 0..n {
            // Validate the type name before looking at the other arguments so
            // a bad literal errors even on rows whose grid is null.
            let band_type = match &data_type {
                Some(names) if names.is_null(i) => {
                    builder.append_null()?;
                    continue;
                }
                Some(names) => parse_pixel_type(names.value(i))?,
                None => BandDataType::Float64,
            };

            if num_bands.is_null(i) || width.is_null(i) || height.is_null(i) {
                builder.append_null()?;
                continue;
            }
            let (num_bands, width, height) = (num_bands.value(i), width.value(i), height.value(i));
            let Some(geom) = placement.geometry(i, width, height)? else {
                builder.append_null()?;
                continue;
            };

            let (num_bands, band_len) = validate_shape(num_bands, width, height, band_type)?;

            builder.start_raster_2d(
                width,
                height,
                geom.upper_left_x,
                geom.upper_left_y,
                geom.scale_x,
                geom.scale_y,
                geom.skew_x,
                geom.skew_y,
                geom.crs.as_deref(),
            )?;
            let buffer = zeros
                .entry(band_len)
                .or_insert_with(|| MutableBuffer::from_len_zeroed(band_len).into());
            for _ in 0..num_bands {
                builder.start_band_2d(band_type, None)?;
                builder.append_band_data_buffer(buffer, 0, band_len as u32)?;
                builder.finish_band()?;
            }
            builder.finish_raster()?;
        }

        executor.finish(Arc::new(builder.finish()?))
    }
}

/// A resolved grid placement for one output raster.
struct GridGeometry {
    upper_left_x: f64,
    upper_left_y: f64,
    scale_x: f64,
    scale_y: f64,
    skew_x: f64,
    skew_y: f64,
    crs: Option<String>,
}

/// Per-row accessors for the grid-placement arguments of one kernel form.
enum Placement {
    Extent(crate::executor::GeomWkbCrsAccessor),
    CellSize {
        upper_left_x: Float64Array,
        upper_left_y: Float64Array,
        cell_size: Float64Array,
    },
    Affine(Box<AffineColumns>),
}

/// The seven per-row columns of the affine form (boxed so the enum stays small).
struct AffineColumns {
    upper_left_x: Float64Array,
    upper_left_y: Float64Array,
    scale_x: Float64Array,
    scale_y: Float64Array,
    skew_x: Float64Array,
    skew_y: Float64Array,
    srid: Int64Array,
    srid_to_crs: CachedSRIDToCrs,
}

impl Placement {
    /// Resolve row `i`'s placement, or `None` when any of its inputs is null.
    fn geometry(&mut self, i: usize, width: i64, height: i64) -> Result<Option<GridGeometry>> {
        match self {
            Placement::Extent(accessor) => {
                let (maybe_wkb, crs) = accessor.get(i)?;
                let Some(wkb) = maybe_wkb else {
                    return Ok(None);
                };
                let (xmin, ymin, xmax, ymax) = extent_bounds(wkb)?;
                Ok(Some(GridGeometry {
                    upper_left_x: xmin,
                    upper_left_y: ymax,
                    scale_x: (xmax - xmin) / width as f64,
                    scale_y: -(ymax - ymin) / height as f64,
                    skew_x: 0.0,
                    skew_y: 0.0,
                    crs: crs.map(|c| c.to_crs_string()),
                }))
            }
            Placement::CellSize {
                upper_left_x,
                upper_left_y,
                cell_size,
            } => {
                if upper_left_x.is_null(i) || upper_left_y.is_null(i) || cell_size.is_null(i) {
                    return Ok(None);
                }
                let cell_size = cell_size.value(i);
                Ok(Some(GridGeometry {
                    upper_left_x: upper_left_x.value(i),
                    upper_left_y: upper_left_y.value(i),
                    scale_x: cell_size,
                    scale_y: -cell_size,
                    skew_x: 0.0,
                    skew_y: 0.0,
                    crs: None,
                }))
            }
            Placement::Affine(cols) => {
                if cols.upper_left_x.is_null(i)
                    || cols.upper_left_y.is_null(i)
                    || cols.scale_x.is_null(i)
                    || cols.scale_y.is_null(i)
                    || cols.skew_x.is_null(i)
                    || cols.skew_y.is_null(i)
                    || cols.srid.is_null(i)
                {
                    return Ok(None);
                }
                Ok(Some(GridGeometry {
                    upper_left_x: cols.upper_left_x.value(i),
                    upper_left_y: cols.upper_left_y.value(i),
                    scale_x: cols.scale_x.value(i),
                    scale_y: cols.scale_y.value(i),
                    skew_x: cols.skew_x.value(i),
                    skew_y: cols.skew_y.value(i),
                    crs: cols.srid_to_crs.get_crs(cols.srid.value(i))?,
                }))
            }
        }
    }
}

/// The `(xmin, ymin, xmax, ymax)` envelope of an extent geometry, which must
/// span a positive width and height for the pixel size to be defined.
fn extent_bounds(wkb: &[u8]) -> Result<(f64, f64, f64, f64)> {
    let bbox = wkb_bounds_xy(wkb)
        .map_err(|e| exec_datafusion_err!("RS_MakeEmptyRaster: invalid extent geometry: {e}"))?;
    if bbox.is_empty() {
        return exec_err!("RS_MakeEmptyRaster: extent geometry is empty");
    }
    let (xmin, xmax) = (bbox.x().lo(), bbox.x().hi());
    let (ymin, ymax) = (bbox.y().lo(), bbox.y().hi());
    if !(xmax > xmin && ymax > ymin) {
        return exec_err!(
            "RS_MakeEmptyRaster: extent must span a positive width and height, \
             got envelope [{xmin}, {ymin}, {xmax}, {ymax}]"
        );
    }
    Ok((xmin, ymin, xmax, ymax))
}

/// Check the band count and grid size, returning the band count and the byte
/// length of one band's pixel data.
fn validate_shape(
    num_bands: i64,
    width: i64,
    height: i64,
    band_type: BandDataType,
) -> Result<(usize, usize)> {
    if num_bands < 0 {
        return exec_err!("RS_MakeEmptyRaster: num_bands must be >= 0, got {num_bands}");
    }
    if width <= 0 || height <= 0 {
        return exec_err!(
            "RS_MakeEmptyRaster: width and height must be positive, got {width} x {height}"
        );
    }
    // Band data is a BinaryView value, whose length is a u32.
    let band_len = (width as u64)
        .checked_mul(height as u64)
        .and_then(|pixels| pixels.checked_mul(band_type.byte_size() as u64))
        .filter(|&bytes| bytes <= u32::MAX as u64)
        .ok_or_else(|| {
            exec_datafusion_err!(
                "RS_MakeEmptyRaster: a {width} x {height} band of {} pixels exceeds the \
                 4 GiB per-band limit",
                band_type.pixel_type_name()
            )
        })?;
    Ok((num_bands as usize, band_len as usize))
}

fn int_column(arg: &ColumnarValue, n: usize) -> Result<Int64Array> {
    let array = arg.clone().cast_to(&DataType::Int64, None)?.into_array(n)?;
    Ok(as_int64_array(&array)?.clone())
}

fn f64_column(arg: &ColumnarValue, n: usize) -> Result<Float64Array> {
    let array = arg
        .clone()
        .cast_to(&DataType::Float64, None)?
        .into_array(n)?;
    Ok(as_float64_array(&array)?.clone())
}

fn string_column(arg: &ColumnarValue, n: usize) -> Result<StringArray> {
    let array = arg.clone().cast_to(&DataType::Utf8, None)?.into_array(n)?;
    Ok(as_string_array(&array)?.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, BinaryViewArray, ListArray, StructArray};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::crs::{deserialize_crs, lnglat};
    use sedona_schema::datatypes::{Edges, WKB_GEOMETRY};
    use sedona_schema::raster::{band_indices, raster_indices};
    use sedona_testing::create::{create_scalar_item_crs, create_scalar_value};
    use sedona_testing::raster_spec::{
        RasterSpec, assert_raster_scalar_equals, assert_rasters_equal,
    };
    use sedona_testing::testers::ScalarUdfTester;

    fn i64_t() -> SedonaType {
        SedonaType::Arrow(DataType::Int64)
    }
    fn f64_t() -> SedonaType {
        SedonaType::Arrow(DataType::Float64)
    }
    fn utf8_t() -> SedonaType {
        SedonaType::Arrow(DataType::Utf8)
    }
    fn int(v: i64) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Int64(Some(v)))
    }
    fn num(v: f64) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Float64(Some(v)))
    }
    fn text(v: &str) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(v.to_string())))
    }
    fn tester(types: Vec<SedonaType>) -> ScalarUdfTester {
        ScalarUdfTester::new(rs_make_empty_raster_udf().into(), types)
    }
    fn invoke_scalar(tester: &ScalarUdfTester, args: Vec<ColumnarValue>) -> ScalarValue {
        match tester.invoke(args).unwrap() {
            ColumnarValue::Scalar(scalar) => scalar,
            ColumnarValue::Array(_) => panic!("expected a scalar result"),
        }
    }
    fn invoke_err(tester: &ScalarUdfTester, args: Vec<ColumnarValue>) -> String {
        tester.invoke(args).unwrap_err().to_string()
    }

    fn cell_size_types(typed: bool) -> Vec<SedonaType> {
        let mut types = vec![i64_t()];
        if typed {
            types.push(utf8_t());
        }
        types.extend([i64_t(), i64_t(), f64_t(), f64_t(), f64_t()]);
        types
    }
    fn affine_types(typed: bool) -> Vec<SedonaType> {
        let mut types = vec![i64_t()];
        if typed {
            types.push(utf8_t());
        }
        types.extend([i64_t(), i64_t()]);
        types.extend((0..6).map(|_| f64_t()));
        types.push(i64_t());
        types
    }
    fn extent_types(typed: bool, extent: SedonaType) -> Vec<SedonaType> {
        let mut types = vec![i64_t()];
        if typed {
            types.push(utf8_t());
        }
        types.extend([i64_t(), i64_t(), extent]);
        types
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_make_empty_raster_udf().into();
        assert_eq!(udf.name(), "rs_makeemptyraster");

        for types in [
            cell_size_types(false),
            cell_size_types(true),
            affine_types(false),
            affine_types(true),
            extent_types(false, WKB_GEOMETRY),
            extent_types(true, WKB_GEOMETRY),
        ] {
            let arity = types.len();
            assert_eq!(
                tester(types).return_type().unwrap(),
                SedonaType::Raster,
                "arity {arity}"
            );
        }
    }

    #[test]
    fn cell_size_form_defaults_to_float64_bands_without_crs() {
        let tester = tester(cell_size_types(false));
        let result = invoke_scalar(
            &tester,
            vec![int(2), int(4), int(3), num(10.0), num(20.0), num(2.5)],
        );

        let zeros = vec![0f64; 12];
        let expected = RasterSpec::d2(4, 3)
            .transform([10.0, 2.5, 0.0, 20.0, 0.0, -2.5])
            .crs(None)
            .band_values(&zeros)
            .band_values(&zeros);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn cell_size_form_with_band_type() {
        let tester = tester(cell_size_types(true));
        // Coordinates may be integers: the matcher is numeric, not float.
        let result = invoke_scalar(
            &tester,
            vec![
                int(1),
                text("B"),
                int(4),
                int(3),
                num(0.0),
                num(0.0),
                num(1.0),
            ],
        );

        let expected = RasterSpec::d2(4, 3)
            .transform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            .crs(None)
            .band_values(&[0u8; 12]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn affine_form_sets_skew_and_srid() {
        let tester = tester(affine_types(true));
        let result = invoke_scalar(
            &tester,
            vec![
                int(1),
                text("I"),
                int(5),
                int(4),
                num(100.0),
                num(200.0),
                num(2.0),
                num(-3.0),
                num(0.5),
                num(0.25),
                int(3857),
            ],
        );

        let expected = RasterSpec::d2(5, 4)
            .transform([100.0, 2.0, 0.5, 200.0, 0.25, -3.0])
            .crs(Some("EPSG:3857"))
            .band_values(&[0i32; 20]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn affine_form_srid_mapping() {
        let tester = tester(affine_types(false));
        let args = |srid: i64| {
            vec![
                int(1),
                int(2),
                int(2),
                num(0.0),
                num(0.0),
                num(1.0),
                num(-1.0),
                num(0.0),
                num(0.0),
                int(srid),
            ]
        };
        let base = RasterSpec::d2(2, 2)
            .transform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            .band_values(&[0f64; 4]);

        // SRID 0 is "no CRS", 4326 is the lnglat CRS, anything else EPSG:<srid>
        assert_raster_scalar_equals(&invoke_scalar(&tester, args(0)), &base.clone().crs(None));
        assert_raster_scalar_equals(
            &invoke_scalar(&tester, args(4326)),
            &base.clone().crs(Some(&lnglat().unwrap().to_crs_string())),
        );
        assert_raster_scalar_equals(
            &invoke_scalar(&tester, args(32610)),
            &base.crs(Some("EPSG:32610")),
        );
    }

    #[test]
    fn extent_form_takes_envelope_and_crs_from_geometry() {
        let geom_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());
        let tester = tester(extent_types(true, geom_type.clone()));

        // A 5 x 4 grid over [0, 10] x [0, 20]: 2-wide, 5-tall north-up pixels
        // anchored at the envelope's top-left corner.
        let expected = RasterSpec::d2(5, 4)
            .bbox(0.0, 0.0, 10.0, 20.0)
            .crs(Some("EPSG:3857"))
            .band_values(&[0u8; 20]);

        // The envelope is what matters, not the shape: a rectangle and a
        // triangle with the same bounds define the same grid.
        for wkt in [
            "POLYGON ((0 0, 10 0, 10 20, 0 20, 0 0))",
            "POLYGON ((0 0, 10 0, 0 20, 0 0))",
            "MULTIPOINT ((0 0), (10 20))",
        ] {
            let result = invoke_scalar(
                &tester,
                vec![
                    int(1),
                    text("uint8"),
                    int(5),
                    int(4),
                    create_scalar_value(Some(wkt), &geom_type),
                ],
            );
            assert_raster_scalar_equals(&result, &expected);
        }
    }

    #[test]
    fn extent_form_without_geometry_crs_has_no_crs() {
        let tester = tester(extent_types(false, WKB_GEOMETRY));
        let result = invoke_scalar(
            &tester,
            vec![
                int(1),
                int(2),
                int(2),
                create_scalar_value(Some("POLYGON ((1 1, 3 1, 3 5, 1 5, 1 1))"), &WKB_GEOMETRY),
            ],
        );

        let expected = RasterSpec::d2(2, 2)
            .bbox(1.0, 1.0, 3.0, 5.0)
            .crs(None)
            .band_values(&[0f64; 4]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn extent_form_accepts_item_crs_geometry() {
        // e.g. the output of RS_Envelope, whose CRS rides along per item
        let item_crs_type = SedonaType::new_item_crs(&WKB_GEOMETRY).unwrap();
        let tester = tester(extent_types(false, item_crs_type));
        let extent = ColumnarValue::Scalar(create_scalar_item_crs(
            Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"),
            Some("EPSG:32610"),
            &WKB_GEOMETRY,
        ));
        let result = invoke_scalar(&tester, vec![int(1), int(4), int(4), extent]);

        let expected = RasterSpec::d2(4, 4)
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:32610"))
            .band_values(&[0f64; 16]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn zero_bands_is_a_bandless_grid() {
        let tester = tester(cell_size_types(false));
        let result = invoke_scalar(
            &tester,
            vec![int(0), int(4), int(3), num(0.0), num(0.0), num(1.0)],
        );
        let expected = RasterSpec::d2(4, 3)
            .transform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            .crs(None);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn array_inputs_yield_one_raster_per_row_with_nulls_propagated() {
        let tester = tester(cell_size_types(false));
        let widths: ArrayRef = Arc::new(Int64Array::from(vec![Some(4), None, Some(2)]));
        let result = tester
            .invoke(vec![
                int(1),
                ColumnarValue::Array(widths),
                int(3),
                num(0.0),
                num(0.0),
                num(1.0),
            ])
            .unwrap();
        let ColumnarValue::Array(array) = result else {
            panic!("expected an array result");
        };

        let spec = |width: i64| {
            RasterSpec::d2(width, 3)
                .transform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
                .crs(None)
                .band_values(&vec![0f64; (width * 3) as usize])
        };
        assert_rasters_equal(&array, &[Some(spec(4)), None, Some(spec(2))]);
    }

    #[test]
    fn null_band_type_or_srid_yields_null_raster() {
        let tester_typed = tester(cell_size_types(true));
        let null_type = ColumnarValue::Scalar(ScalarValue::Utf8(None));
        let result = invoke_scalar(
            &tester_typed,
            vec![
                int(1),
                null_type,
                int(2),
                int(2),
                num(0.0),
                num(0.0),
                num(1.0),
            ],
        );
        assert!(result.is_null());

        let tester_affine = tester(affine_types(false));
        let null_srid = ColumnarValue::Scalar(ScalarValue::Int64(None));
        let result = invoke_scalar(
            &tester_affine,
            vec![
                int(1),
                int(2),
                int(2),
                num(0.0),
                num(0.0),
                num(1.0),
                num(-1.0),
                num(0.0),
                num(0.0),
                null_srid,
            ],
        );
        assert!(result.is_null());

        let tester_extent = tester(extent_types(false, WKB_GEOMETRY));
        let result = invoke_scalar(
            &tester_extent,
            vec![
                int(1),
                int(2),
                int(2),
                create_scalar_value(None, &WKB_GEOMETRY),
            ],
        );
        assert!(result.is_null());
    }

    #[test]
    fn invalid_arguments_error() {
        let tester_cell = tester(cell_size_types(true));
        let cell = |bands: i64, band_type: &str, width: i64, height: i64| {
            vec![
                int(bands),
                text(band_type),
                int(width),
                int(height),
                num(0.0),
                num(0.0),
                num(1.0),
            ]
        };

        let err = invoke_err(&tester_cell, cell(-1, "uint8", 2, 2));
        assert!(err.contains("num_bands must be >= 0"), "{err}");

        let err = invoke_err(&tester_cell, cell(1, "uint8", 0, 2));
        assert!(err.contains("width and height must be positive"), "{err}");

        let err = invoke_err(&tester_cell, cell(1, "complex128", 2, 2));
        assert!(err.contains("Unsupported pixelType"), "{err}");

        // 100k x 100k float64 is 80 GB per band: past the BinaryView limit
        let err = invoke_err(&tester_cell, cell(1, "float64", 100_000, 100_000));
        assert!(err.contains("4 GiB per-band limit"), "{err}");

        let tester_extent = tester(extent_types(false, WKB_GEOMETRY));
        let extent = |wkt: &str| {
            vec![
                int(1),
                int(2),
                int(2),
                create_scalar_value(Some(wkt), &WKB_GEOMETRY),
            ]
        };
        let err = invoke_err(&tester_extent, extent("POINT (1 1)"));
        assert!(err.contains("positive width and height"), "{err}");
        let err = invoke_err(&tester_extent, extent("LINESTRING (0 0, 0 5)"));
        assert!(err.contains("positive width and height"), "{err}");
        let err = invoke_err(&tester_extent, extent("POLYGON EMPTY"));
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn every_band_shares_one_block_of_zeros() {
        // Two rows of two 8x8 uint8 bands: 64 bytes each, past the inline
        // view size, yet the output carries a single data block.
        let tester = tester(cell_size_types(true));
        let widths: ArrayRef = Arc::new(Int64Array::from(vec![8, 8]));
        let result = tester
            .invoke(vec![
                int(2),
                text("uint8"),
                ColumnarValue::Array(widths),
                int(8),
                num(0.0),
                num(0.0),
                num(1.0),
            ])
            .unwrap();
        let ColumnarValue::Array(array) = result else {
            panic!("expected an array result");
        };

        let rasters = array.as_any().downcast_ref::<StructArray>().unwrap();
        let bands = rasters
            .column(raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let data = bands
            .column(band_indices::DATA)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();
        assert_eq!(data.len(), 4);
        assert_eq!(data.data_buffers().len(), 1);
        assert_eq!(data.data_buffers()[0].len(), 64);
    }
}
