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

//! RS_MakeRaster — build a one-band out-db raster reference from its
//! arguments alone.
//!
//! - `(path, band_data_type, width, height, extent[, band])` — the grid that
//!   covers the envelope of `extent` with `width` x `height` north-up pixels,
//!   in the geometry's CRS.
//! - `(path, band_data_type, width, height, upper_left_x, upper_left_y,
//!   scale_x, scale_y, skew_x, skew_y, crs[, band])` — the full affine form;
//!   `crs` is an integer SRID or a CRS string.
//!
//! The result is the raster `RS_FromPath(path)` would produce for that band,
//! but nothing is read from `path`: the grid, CRS and pixel type are taken on
//! trust, and the pixels are loaded lazily by the out-db loader when a function
//! needs them.

use std::sync::Arc;

use arrow_array::Array;
use datafusion_common::{config::ConfigOptions, error::Result, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::{
    item_crs::parse_item_crs_arg_type,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_raster::builder::RasterBuilder;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::{
    executor::RasterExecutor,
    grid_placement::{CrsColumn, Placement, int_column, string_column},
    pixel_type::parse_pixel_type,
};

/// Function name used to prefix error messages.
const NAME: &str = "RS_MakeRaster";

/// Index of the first grid-placement argument (the one after `height`).
const GRID_ARG: usize = 4;

/// RS_MakeRaster() scalar UDF implementation
pub fn rs_make_raster_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_makeraster",
        vec![
            Arc::new(RsMakeRaster { grid: Grid::Extent }),
            Arc::new(RsMakeRaster { grid: Grid::Srid }),
            Arc::new(RsMakeRaster { grid: Grid::Crs }),
        ],
        Volatility::Immutable,
    )
}

/// How the grid's placement is specified after `path, band_data_type, width,
/// height`.
#[derive(Debug, Clone, Copy)]
enum Grid {
    /// `extent: geometry` — envelope and CRS come from the geometry.
    Extent,
    /// `upper_left_x, upper_left_y, scale_x, scale_y, skew_x, skew_y, srid`.
    Srid,
    /// `upper_left_x, upper_left_y, scale_x, scale_y, skew_x, skew_y, crs`.
    Crs,
}

#[derive(Debug)]
struct RsMakeRaster {
    grid: Grid,
}

impl RsMakeRaster {
    /// Index of the optional trailing `band` argument.
    fn band_arg_index(&self) -> usize {
        match self.grid {
            Grid::Extent => GRID_ARG + 1,
            Grid::Srid | Grid::Crs => GRID_ARG + 7,
        }
    }
}

impl SedonaScalarKernel for RsMakeRaster {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![
            ArgMatcher::is_string(),  // path
            ArgMatcher::is_string(),  // band_data_type
            ArgMatcher::is_integer(), // width
            ArgMatcher::is_integer(), // height
        ];

        // A geometry extent may arrive as an item_crs struct (e.g. from
        // RS_Envelope); match on its item type like RS_MakeEmptyRaster does.
        let mut arg_types = args.to_vec();
        match self.grid {
            Grid::Extent => {
                matchers.push(ArgMatcher::is_geometry_or_geography());
                if let Some(extent_type) = arg_types.get(GRID_ARG) {
                    let (item_type, _) = parse_item_crs_arg_type(extent_type)?;
                    arg_types[GRID_ARG] = item_type;
                }
            }
            Grid::Srid => {
                matchers.extend((0..6).map(|_| ArgMatcher::is_numeric()));
                matchers.push(ArgMatcher::is_integer());
            }
            Grid::Crs => {
                matchers.extend((0..6).map(|_| ArgMatcher::is_numeric()));
                matchers.push(ArgMatcher::is_string());
            }
        }
        matchers.push(ArgMatcher::optional(ArgMatcher::is_integer())); // band

        ArgMatcher::new(matchers, SedonaType::Raster).match_args(&arg_types)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke(arg_types, args, None)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        self.invoke(arg_types, args, config_options)
    }
}

impl RsMakeRaster {
    fn invoke(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let n = executor.num_iterations();

        let path = string_column(&args[0], n)?;
        let data_type = string_column(&args[1], n)?;
        let width = int_column(&args[2], n)?;
        let height = int_column(&args[3], n)?;
        let band = match args.get(self.band_arg_index()) {
            Some(arg) => Some(int_column(arg, n)?),
            None => None,
        };

        let g = GRID_ARG;
        let mut placement = match self.grid {
            Grid::Extent => Placement::extent(&executor, arg_types, g, NAME, config_options)?,
            Grid::Srid => Placement::affine(args, g, n, CrsColumn::srid(&args[g + 6], n)?)?,
            Grid::Crs => Placement::affine(args, g, n, CrsColumn::definition(&args[g + 6], n)?)?,
        };

        let mut builder = RasterBuilder::new(n);
        for i in 0..n {
            // Validate the type name before looking at the other arguments so
            // a bad literal errors even on rows whose other arguments are null.
            if data_type.is_null(i) {
                builder.append_null()?;
                continue;
            }
            let band_type = parse_pixel_type(data_type.value(i))?;

            let band = match &band {
                Some(band) if band.is_null(i) => {
                    builder.append_null()?;
                    continue;
                }
                Some(band) => band.value(i),
                None => 1,
            };
            if path.is_null(i) || width.is_null(i) || height.is_null(i) {
                builder.append_null()?;
                continue;
            }
            let (width, height) = (width.value(i), height.value(i));
            let Some(geom) = placement.geometry(NAME, i, width, height)? else {
                builder.append_null()?;
                continue;
            };

            if width <= 0 || height <= 0 {
                return exec_err!(
                    "{NAME}: width and height must be positive, got {width} x {height}"
                );
            }
            let Some(band) = u32::try_from(band).ok().filter(|&b| b >= 1) else {
                return exec_err!(
                    "{NAME}: band must be a 1-based band index between 1 and {}, got {band}",
                    u32::MAX
                );
            };

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
            builder.append_outdb_band_2d(path.value(i), band, band_type, None)?;
            builder.finish_raster()?;
        }

        executor.finish(Arc::new(builder.finish()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, Int64Array, StringArray};
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use datafusion_expr::{ScalarUDF, lit};
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::{Edges, WKB_GEOMETRY};
    use sedona_schema::raster::BandDataType;
    use sedona_testing::create::create_scalar_item_crs;
    use sedona_testing::raster_spec::{
        RasterSpec, assert_raster_scalar_equals, assert_rasters_equal,
    };
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_make_raster_udf().into();
        assert_eq!(udf.name(), "rs_makeraster");

        // Every form, with and without the trailing band, named by its
        // arguments so a failure says which one.
        let forms = [
            (
                "path, bandType, width, height, extent",
                vec![
                    SedonaType::Arrow(DataType::Utf8),
                    SedonaType::Arrow(DataType::Utf8),
                    SedonaType::Arrow(DataType::Int64),
                    SedonaType::Arrow(DataType::Int64),
                    WKB_GEOMETRY,
                ],
            ),
            (
                "path, bandType, width, height, extent, band",
                vec![
                    SedonaType::Arrow(DataType::Utf8),
                    SedonaType::Arrow(DataType::Utf8),
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Arrow(DataType::Int32),
                    WKB_GEOMETRY,
                    SedonaType::Arrow(DataType::Int32),
                ],
            ),
            (
                "path, bandType, width, height, upperLeftX, upperLeftY, scaleX, scaleY, \
                 skewX, skewY, srid",
                vec![
                    SedonaType::Arrow(DataType::Utf8),
                    SedonaType::Arrow(DataType::Utf8),
                    SedonaType::Arrow(DataType::Int64),
                    SedonaType::Arrow(DataType::Int64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Int64),
                ],
            ),
            (
                "path, bandType, width, height, upperLeftX, upperLeftY, scaleX, scaleY, \
                 skewX, skewY, crs, band",
                vec![
                    SedonaType::Arrow(DataType::Utf8View),
                    SedonaType::Arrow(DataType::Utf8),
                    SedonaType::Arrow(DataType::Int64),
                    SedonaType::Arrow(DataType::Int64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Int64),
                    SedonaType::Arrow(DataType::Int64),
                    SedonaType::Arrow(DataType::Utf8),
                    SedonaType::Arrow(DataType::Int64),
                ],
            ),
        ];
        for (form, arg_types) in forms {
            let tester = ScalarUdfTester::new(udf.clone(), arg_types);
            assert_eq!(
                tester.return_type().unwrap(),
                SedonaType::Raster,
                "form: {form}"
            );
        }
    }

    #[test]
    fn affine_form_with_crs_string_and_band() {
        // RS_MakeRaster(path, bandType, width, height, upperLeftX, upperLeftY,
        // scaleX, scaleY, skewX, skewY, crs, band)
        let mut arg_types = vec![
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Int64),
            SedonaType::Arrow(DataType::Int64),
        ];
        arg_types.extend((0..6).map(|_| SedonaType::Arrow(DataType::Float64)));
        arg_types.push(SedonaType::Arrow(DataType::Utf8));
        arg_types.push(SedonaType::Arrow(DataType::Int64));
        let tester = ScalarUdfTester::new(rs_make_raster_udf().into(), arg_types);

        let result = tester
            .invoke_scalars(vec![
                lit("s3://bucket/scene.tif"),
                lit("uint16"),
                lit(5),
                lit(3),
                lit(500000.0),
                lit(4100000.0),
                lit(10.0),
                lit(-10.0),
                lit(0.5),
                lit(0.25),
                lit("EPSG:32627"),
                lit(2),
            ])
            .unwrap();
        let expected = RasterSpec::d2(5, 3)
            .transform([500000.0, 10.0, 0.5, 4100000.0, 0.25, -10.0])
            .crs(Some("EPSG:32627"))
            .band(BandDataType::UInt16)
            .outdb("s3://bucket/scene.tif#band=2", None);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn affine_form_srid_mapping_and_default_band() {
        // RS_MakeRaster(path, bandType, width, height, upperLeftX, upperLeftY,
        // scaleX, scaleY, skewX, skewY, srid)
        let mut arg_types = vec![
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Int64),
            SedonaType::Arrow(DataType::Int64),
        ];
        arg_types.extend((0..6).map(|_| SedonaType::Arrow(DataType::Float64)));
        arg_types.push(SedonaType::Arrow(DataType::Int64));
        let tester = ScalarUdfTester::new(rs_make_raster_udf().into(), arg_types);

        // 4326 maps to OGC:CRS84 and 0 to no CRS, as in RS_MakeEmptyRaster.
        for (srid, crs) in [
            (4326, Some("OGC:CRS84")),
            (3857, Some("EPSG:3857")),
            (0, None),
        ] {
            let result = tester
                .invoke_scalars(vec![
                    lit("/data/a.tif"),
                    lit("F"),
                    lit(2),
                    lit(2),
                    lit(0.0),
                    lit(2.0),
                    lit(1.0),
                    lit(-1.0),
                    lit(0.0),
                    lit(0.0),
                    lit(srid),
                ])
                .unwrap();
            let expected = RasterSpec::d2(2, 2)
                .bbox(0.0, 0.0, 2.0, 2.0)
                .crs(crs)
                .band(BandDataType::Float32)
                .outdb("/data/a.tif#band=1", None);
            assert_raster_scalar_equals(&result, &expected);
        }
    }

    #[test]
    fn crs_string_is_normalized() {
        // A PROJJSON/WKT definition is kept in full, an authority code
        // compactly, and "0" means no CRS: the same normalization RS_SetCRS
        // applies.
        let mut arg_types = vec![
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Int64),
            SedonaType::Arrow(DataType::Int64),
        ];
        arg_types.extend((0..6).map(|_| SedonaType::Arrow(DataType::Float64)));
        arg_types.push(SedonaType::Arrow(DataType::Utf8));
        let tester = ScalarUdfTester::new(rs_make_raster_udf().into(), arg_types);

        for (crs, expected_crs) in [("EPSG:4326", Some("EPSG:4326")), ("0", None)] {
            let result = tester
                .invoke_scalars(vec![
                    lit("/data/a.tif"),
                    lit("B"),
                    lit(2),
                    lit(2),
                    lit(0.0),
                    lit(2.0),
                    lit(1.0),
                    lit(-1.0),
                    lit(0.0),
                    lit(0.0),
                    lit(crs),
                ])
                .unwrap();
            let expected = RasterSpec::d2(2, 2)
                .bbox(0.0, 0.0, 2.0, 2.0)
                .crs(expected_crs)
                .band(BandDataType::UInt8)
                .outdb("/data/a.tif#band=1", None);
            assert_raster_scalar_equals(&result, &expected);
        }

        let err = tester
            .invoke_scalars(vec![
                lit("/data/a.tif"),
                lit("B"),
                lit(2),
                lit(2),
                lit(0.0),
                lit(2.0),
                lit(1.0),
                lit(-1.0),
                lit(0.0),
                lit(0.0),
                lit("not a crs"),
            ])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("RS_MakeRaster: invalid crs 'not a crs'"),
            "{err}"
        );
    }

    #[test]
    fn extent_form_takes_envelope_and_crs_from_geometry() {
        // RS_MakeRaster(path, bandType, width, height, extent, band)
        let geom_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());
        let tester = ScalarUdfTester::new(
            rs_make_raster_udf().into(),
            vec![
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Int64),
                geom_type,
                SedonaType::Arrow(DataType::Int64),
            ],
        );
        let result = tester
            .invoke_scalars(vec![
                lit("https://example.com/r.tif"),
                lit("int16"),
                lit(5),
                lit(4),
                lit("POLYGON ((0 0, 10 0, 10 20, 0 20, 0 0))"),
                lit(3),
            ])
            .unwrap();
        let expected = RasterSpec::d2(5, 4)
            .bbox(0.0, 0.0, 10.0, 20.0)
            .crs(Some("EPSG:3857"))
            .band(BandDataType::Int16)
            .outdb("https://example.com/r.tif#band=3", None);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn extent_form_accepts_item_crs_geometry() {
        // e.g. the output of RS_Envelope, whose CRS rides along per item
        let tester = ScalarUdfTester::new(
            rs_make_raster_udf().into(),
            vec![
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Int64),
                SedonaType::new_item_crs(&WKB_GEOMETRY).unwrap(),
            ],
        );
        let extent = create_scalar_item_crs(
            Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"),
            Some("EPSG:32610"),
            &WKB_GEOMETRY,
        );
        let result = tester
            .invoke_scalars(vec![
                lit("/data/a.tif"),
                lit("float64"),
                lit(4),
                lit(4),
                lit(extent),
            ])
            .unwrap();
        let expected = RasterSpec::d2(4, 4)
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:32610"))
            .band(BandDataType::Float64)
            .outdb("/data/a.tif#band=1", None);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn array_inputs_yield_one_raster_per_row_with_nulls_propagated() {
        // RS_MakeRaster(path, bandType, width, height, extent, band) with a
        // null in each of path, width, extent and band.
        let tester = ScalarUdfTester::new(
            rs_make_raster_udf().into(),
            vec![
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Int64),
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Int64),
            ],
        );
        let paths: ArrayRef = Arc::new(StringArray::from(vec![
            Some("/a.tif"),
            None,
            Some("/c.tif"),
            Some("/d.tif"),
            Some("/e.tif"),
        ]));
        let widths: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(4),
            Some(4),
            None,
            Some(4),
            Some(2),
        ]));
        let extents = sedona_testing::create::create_array(
            &[
                Some("POLYGON ((0 0, 4 0, 4 2, 0 2, 0 0))"),
                Some("POLYGON ((0 0, 4 0, 4 2, 0 2, 0 0))"),
                Some("POLYGON ((0 0, 4 0, 4 2, 0 2, 0 0))"),
                None,
                Some("POLYGON ((0 0, 4 0, 4 2, 0 2, 0 0))"),
            ],
            &WKB_GEOMETRY,
        );
        let bands: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(1),
            Some(1),
            Some(1),
            Some(1),
            None,
        ]));
        let result = tester
            .invoke(vec![
                ColumnarValue::Array(paths),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("B".to_string()))),
                ColumnarValue::Array(widths),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
                ColumnarValue::Array(extents),
                ColumnarValue::Array(bands),
            ])
            .unwrap();
        let ColumnarValue::Array(array) = result else {
            panic!("expected an array result");
        };
        let expected = RasterSpec::d2(4, 2)
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(None)
            .band(BandDataType::UInt8)
            .outdb("/a.tif#band=1", None);
        assert_rasters_equal(&array, &[Some(expected), None, None, None, None]);
    }

    #[test]
    fn null_band_type_yields_null_raster() {
        let tester = ScalarUdfTester::new(
            rs_make_raster_udf().into(),
            vec![
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Int64),
                WKB_GEOMETRY,
            ],
        );
        let result = tester
            .invoke_scalars(vec![
                lit("/a.tif"),
                lit(ScalarValue::Utf8(None)),
                lit(2),
                lit(2),
                lit("POLYGON ((0 0, 1 0, 1 1, 0 0))"),
            ])
            .unwrap();
        assert!(result.is_null(), "{result:?}");
    }

    #[test]
    fn invalid_arguments_error() {
        // RS_MakeRaster(path, bandType, width, height, extent, band)
        let tester = ScalarUdfTester::new(
            rs_make_raster_udf().into(),
            vec![
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Int64),
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Int64),
            ],
        );
        let extent_err = |band_type: &str, width: i64, height: i64, band: i64| {
            tester
                .invoke_scalars(vec![
                    lit("/a.tif"),
                    lit(band_type),
                    lit(width),
                    lit(height),
                    lit("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"),
                    lit(band),
                ])
                .unwrap_err()
                .to_string()
        };

        let err = extent_err("uint8", 0, 2, 1);
        assert!(err.contains("width and height must be positive"), "{err}");
        let err = extent_err("uint8", 2, -1, 1);
        assert!(err.contains("width and height must be positive"), "{err}");
        let err = extent_err("complex128", 2, 2, 1);
        assert!(err.contains("Unsupported pixelType"), "{err}");
        let err = extent_err("uint8", 2, 2, 0);
        assert!(err.contains("band must be a 1-based band index"), "{err}");
        let err = extent_err("uint8", 2, 2, 1 << 32);
        assert!(err.contains("band must be a 1-based band index"), "{err}");

        let err = tester
            .invoke_scalars(vec![
                lit("/a.tif"),
                lit("uint8"),
                lit(2),
                lit(2),
                lit("POINT (1 1)"),
                lit(1),
            ])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("RS_MakeRaster: extent must span a positive width and height"),
            "{err}"
        );
    }
}
