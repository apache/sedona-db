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
use std::{sync::Arc, vec};

use crate::executor::WkbExecutor;
use arrow_array::builder::BinaryBuilder;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::GeometryTrait;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::{
    bounds::geo_traits_bounds_xy,
    interval::{Interval, IntervalTrait, WraparoundInterval},
    wkb_factory::{
        write_wkb_empty_point, write_wkb_geometrycollection_header, write_wkb_linestring,
        write_wkb_linestring_header, write_wkb_multilinestring_header, write_wkb_multipoint_header,
        write_wkb_multipolygon_header, write_wkb_point, write_wkb_polygon,
        write_wkb_polygon_header, WKB_MIN_PROBABLE_BYTES,
    },
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::Wkb;

/// ST_Envelope() scalar UDF implementation
///
/// An implementation of envelope (bounding shape) calculation.
pub fn st_envelope_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_envelope",
        vec![Arc::new(STEnvelope {})],
        Volatility::Immutable,
        Some(st_envelope_doc()),
    )
}

fn st_envelope_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the envelope of a geometry",
        "ST_Envelope(A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_Envelope(ST_Point(1.0, 2.0))")
    .build()
}

#[derive(Debug)]
struct STEnvelope {}

impl SedonaScalarKernel for STEnvelope {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    invoke_scalar(&item, &mut builder)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(wkb: &Wkb, writer: &mut impl std::io::Write) -> Result<()> {
    let bounds = geo_traits_bounds_xy(wkb).map_err(|e| DataFusionError::External(e.into()))?;

    let written = write_envelope(bounds.x(), bounds.y(), writer)?;

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
            _ => {
                return Err(DataFusionError::Internal(
                    "Unsupported geometry type".to_string(),
                ))
            }
        };

        if let Err(e) = result {
            return Err(DataFusionError::External(e.into()));
        }
    }

    Ok(())
}

/// Writes the WKB for an envelope of a geometry given its XY bounds
/// Returns true if the envelope was written, false otherwise
/// A return value of false indicates an empty envelope allowing the caller to handle it appropriately
pub fn write_envelope(
    x: &WraparoundInterval,
    y: &Interval,
    out: &mut impl std::io::Write,
) -> Result<bool> {
    if x.is_empty() && y.is_empty() {
        // Return true and let the caller determine how to handle an empty envelope
        return Ok(false);
    }
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
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_envelope_udf().into();
        assert_eq!(udf.name(), "st_envelope");
        assert!(udf.documentation().is_some());
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
}
