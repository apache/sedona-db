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
use std::{marker::PhantomData, sync::Arc, vec};

use crate::executor::WkbBytesExecutor;
use arrow_array::builder::BinaryBuilder;
use datafusion_common::{
    error::{DataFusionError, Result},
    exec_datafusion_err,
};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::GeometryTrait;
use sedona_common::sedona_internal_err;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{
    bounds::{WkbBounder2D, WkbGeometryBounder},
    interval::{Interval, IntervalTrait, WraparoundInterval},
    wkb_factory::{
        write_wkb_empty_point, write_wkb_geometrycollection_header, write_wkb_linestring,
        write_wkb_linestring_header, write_wkb_multilinestring, write_wkb_multilinestring_header,
        write_wkb_multipoint_header, write_wkb_multipolygon, write_wkb_multipolygon_header,
        write_wkb_point, write_wkb_polygon, write_wkb_polygon_header, WKB_MIN_PROBABLE_BYTES,
    },
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

/// ST_Envelope() scalar UDF implementation
///
/// An implementation of envelope (bounding shape) calculation.
pub fn st_envelope_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_envelope",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STEnvelope::<WkbGeometryBounder>::new(
            ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
        ))]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
pub struct STEnvelope<T> {
    matcher: ArgMatcher,
    _phantom: PhantomData<T>,
}

impl<T> STEnvelope<T> {
    /// Create a new ST_Envelope implementation with a specific ArgMatcher
    pub fn new(matcher: ArgMatcher) -> Self {
        Self {
            matcher,
            _phantom: Default::default(),
        }
    }
}

impl<T: WkbBounder2D + Default> SedonaScalarKernel for STEnvelope<T> {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbBytesExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let mut bounder = T::default();
        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    invoke_scalar(item, &mut bounder, &mut builder)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    wkb_value: &[u8],
    bounder: &mut impl WkbBounder2D,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    bounder.clear();
    bounder
        .update_wkb_bytes(wkb_value)
        .map_err(|e| exec_datafusion_err!("Error updating bounder: {e}"))?;
    let (x, y) = bounder.finish();
    let written = write_envelope(&x, &y, writer)?;

    let wkb = wkb::reader::read_wkb(wkb_value)
        .map_err(|e| exec_datafusion_err!("Failed to parse empty WKB: {e}"))?;

    if !written {
        let result = match wkb.as_type() {
            geo_traits::GeometryType::Point(_) => write_wkb_empty_point(writer, wkb.dim()),
            geo_traits::GeometryType::LineString(_) => {
                write_wkb_linestring_header(writer, wkb.dim(), 0)
            }
            geo_traits::GeometryType::Polygon(_) => write_wkb_polygon_header(writer, wkb.dim(), 0),
            geo_traits::GeometryType::MultiPoint(_) => {
                write_wkb_multipoint_header(writer, wkb.dim(), 0)
            }
            geo_traits::GeometryType::MultiLineString(_) => {
                write_wkb_multilinestring_header(writer, wkb.dim(), 0)
            }
            geo_traits::GeometryType::MultiPolygon(_) => {
                write_wkb_multipolygon_header(writer, wkb.dim(), 0)
            }
            geo_traits::GeometryType::GeometryCollection(_) => {
                write_wkb_geometrycollection_header(writer, wkb.dim(), 0)
            }
            _ => return sedona_internal_err!("Unsupported geometry type"),
        };

        if let Err(e) = result {
            return Err(DataFusionError::External(e.into()));
        }
    }

    Ok(())
}

/// Writes the WKB for an envelope of a geometry given its XY bounds
///
/// Returns true if the envelope was written, false otherwise. A return value of false
/// indicates an empty envelope allowing the caller to handle it appropriately.
///
/// When `x.is_wraparound()` is true (indicating the envelope crosses the antimeridian),
/// this writes a MULTIPOLYGON or MULTILINESTRING split at 180/-180.
pub fn write_envelope(
    x: &WraparoundInterval,
    y: &Interval,
    out: &mut impl std::io::Write,
) -> Result<bool> {
    if x.is_empty() && y.is_empty() {
        // Return false and let the caller determine how to handle an empty envelope
        return Ok(false);
    }

    // Check for wraparound case (crossing the antimeridian)
    if x.is_wraparound() {
        match (true, y.width() > 0.0) {
            // Wraparound with height: MULTIPOLYGON with two polygons split at 180/-180
            (true, true) => {
                let poly1 = vec![
                    (x.lo(), y.lo()),
                    (x.lo(), y.hi()),
                    (180.0, y.hi()),
                    (180.0, y.lo()),
                    (x.lo(), y.lo()),
                ];
                let poly2 = vec![
                    (-180.0, y.lo()),
                    (-180.0, y.hi()),
                    (x.hi(), y.hi()),
                    (x.hi(), y.lo()),
                    (-180.0, y.lo()),
                ];
                write_wkb_multipolygon(out, [poly1, poly2].into_iter())
                    .map_err(|e| DataFusionError::External(e.into()))?;
            }
            // Wraparound with no height: MULTILINESTRING with two horizontal lines
            (true, false) => {
                let line1 = vec![(x.lo(), y.lo()), (180.0, y.lo())];
                let line2 = vec![(-180.0, y.lo()), (x.hi(), y.lo())];
                write_wkb_multilinestring(out, [line1, line2].into_iter())
                    .map_err(|e| DataFusionError::External(e.into()))?;
            }
            _ => unreachable!(),
        }
    } else {
        match (x.width() > 0.0, y.width() > 0.0) {
            // Extent has height and width: return a polygon
            (true, true) => {
                write_wkb_polygon(
                    out,
                    [
                        (x.lo(), y.lo()),
                        (x.lo(), y.hi()),
                        (x.hi(), y.hi()),
                        (x.hi(), y.lo()),
                        (x.lo(), y.lo()),
                    ]
                    .into_iter(),
                )
                .map_err(|e| DataFusionError::External(e.into()))?;
            }
            (false, true) | (true, false) => {
                // Extent has only height or width: return a vertical or horizontal line
                write_wkb_linestring(out, [(x.lo(), y.lo()), (x.hi(), y.hi())].into_iter())
                    .map_err(|e| DataFusionError::External(e.into()))?;
            }
            (false, false) => {
                // Extent has no height or width: return a point
                write_wkb_point(out, (x.lo(), y.lo()))
                    .map_err(|e| DataFusionError::External(e.into()))?;
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal_wkb_geometry},
        create::create_array,
        testers::ScalarUdfTester,
    };

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_envelope_udf().into();
        assert_eq!(udf.name(), "st_envelope");
    }

    #[rstest]
    fn udf_invoke(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_envelope_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester.invoke_scalar("POINT (1 3)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 3)");

        let input_wkt = vec![
            None,
            Some("LINESTRING (1 2, 2 2)"),
            Some("LINESTRING (0 0, 1 3)"),
            Some("GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (-1 -1, 1 2))"),
            Some("POINT EMPTY"),
            Some("LINESTRING EMPTY"),
            Some("POLYGON EMPTY"),
            Some("MULTIPOINT EMPTY"),
            Some("MULTILINESTRING EMPTY"),
            Some("MULTIPOLYGON EMPTY"),
            Some("GEOMETRYCOLLECTION EMPTY"),
        ];
        let expected = create_array(
            &[
                None,
                Some("LINESTRING (1 2, 2 2)"),
                Some("POLYGON ((0 0, 0 3, 1 3, 1 0, 0 0))"),
                Some("POLYGON((-1 -1, -1 5, 5 5, 5 -1, -1 -1))"),
                Some("POINT EMPTY"),
                Some("LINESTRING EMPTY"),
                Some("POLYGON EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTILINESTRING EMPTY"),
                Some("MULTIPOLYGON EMPTY"),
                Some("GEOMETRYCOLLECTION EMPTY"),
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_envelope_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type);

        let result = tester.invoke_scalar("POINT (1 3)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 3)");
    }

    #[test]
    fn write_envelope_empty_returns_false() {
        let x = WraparoundInterval::empty();
        let y = Interval::empty();
        let mut buf = Vec::new();
        let result = write_envelope(&x, &y, &mut buf).unwrap();
        assert!(!result);
        assert!(buf.is_empty());
    }

    #[test]
    fn write_envelope_non_wraparound_polygon() {
        // x and y both have width: should produce a polygon
        let x = WraparoundInterval::new(0.0, 10.0);
        let y = Interval::new(0.0, 20.0);
        let mut buf = Vec::new();
        let result = write_envelope(&x, &y, &mut buf).unwrap();
        assert!(result);
        assert_scalar_equal_wkb_geometry(
            &ScalarValue::Binary(Some(buf)),
            Some("POLYGON((0 0,0 20,10 20,10 0,0 0))"),
        );
    }

    #[test]
    fn write_envelope_non_wraparound_horizontal_line() {
        // x has width, y has no height: horizontal line
        let x = WraparoundInterval::new(-5.0, 5.0);
        let y = Interval::new(10.0, 10.0);
        let mut buf = Vec::new();
        let result = write_envelope(&x, &y, &mut buf).unwrap();
        assert!(result);
        assert_scalar_equal_wkb_geometry(
            &ScalarValue::Binary(Some(buf)),
            Some("LINESTRING(-5 10,5 10)"),
        );
    }

    #[test]
    fn write_envelope_non_wraparound_vertical_line() {
        // x has no width, y has height: vertical line
        let x = WraparoundInterval::new(5.0, 5.0);
        let y = Interval::new(-10.0, 10.0);
        let mut buf = Vec::new();
        let result = write_envelope(&x, &y, &mut buf).unwrap();
        assert!(result);
        assert_scalar_equal_wkb_geometry(
            &ScalarValue::Binary(Some(buf)),
            Some("LINESTRING(5 -10,5 10)"),
        );
    }

    #[test]
    fn write_envelope_non_wraparound_point() {
        // x and y both have no width: point
        let x = WraparoundInterval::new(5.0, 5.0);
        let y = Interval::new(10.0, 10.0);
        let mut buf = Vec::new();
        let result = write_envelope(&x, &y, &mut buf).unwrap();
        assert!(result);
        assert_scalar_equal_wkb_geometry(&ScalarValue::Binary(Some(buf)), Some("POINT(5 10)"));
    }

    #[test]
    fn write_envelope_wraparound_multipolygon() {
        // Wraparound case with height: MULTIPOLYGON split at 180/-180
        // lo=170, hi=-170 means the interval wraps around the antimeridian
        let x = WraparoundInterval::new(170.0, -170.0);
        let y = Interval::new(-10.0, 10.0);
        let mut buf = Vec::new();
        let result = write_envelope(&x, &y, &mut buf).unwrap();
        assert!(result);
        assert_scalar_equal_wkb_geometry(
            &ScalarValue::Binary(Some(buf)),
            Some("MULTIPOLYGON(((170 -10,170 10,180 10,180 -10,170 -10)),((-180 -10,-180 10,-170 10,-170 -10,-180 -10)))"),
        );
    }

    #[test]
    fn write_envelope_wraparound_multilinestring() {
        // Wraparound case with no height: MULTILINESTRING split at 180/-180
        let x = WraparoundInterval::new(170.0, -170.0);
        let y = Interval::new(5.0, 5.0);
        let mut buf = Vec::new();
        let result = write_envelope(&x, &y, &mut buf).unwrap();
        assert!(result);
        assert_scalar_equal_wkb_geometry(
            &ScalarValue::Binary(Some(buf)),
            Some("MULTILINESTRING((170 5,180 5),(-180 5,-170 5))"),
        );
    }
}
