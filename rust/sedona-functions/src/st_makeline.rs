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
use std::{io::Write, sync::Arc, vec};

use arrow_array::builder::BinaryBuilder;
use datafusion_common::error::Result;
use datafusion_common::exec_err;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{CoordTrait, GeometryTrait, LineStringTrait, MultiPointTrait, PointTrait};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::write_wkb_linestring_header;
use sedona_schema::datatypes::WKB_GEOGRAPHY;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::WkbExecutor;

/// ST_MakeLine() scalar UDF implementation
///
/// Native implementation to create geometries from coordinates.
/// See [`st_geogline_udf`] for the corresponding geography constructor.
pub fn st_makeline_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_makeline",
        vec![
            Arc::new(STMakeLine {
                out_type: WKB_GEOMETRY,
            }),
            Arc::new(STMakeLine {
                out_type: WKB_GEOGRAPHY,
            }),
        ],
        Volatility::Immutable,
        Some(doc()),
    )
}

fn doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Construct a line".to_string(),
        "ST_MakeLine (g1: Geometry or Geography, g2: Geometry or Geography)".to_string(),
    )
    .with_argument("g1", "Geometry or Geography: The first point or geometry")
    .with_argument("g2", "Geometry or Geography: The second point or geometry")
    .with_sql_example("SELECT ST_MakeLine(ST_Point(0, 1), ST_Point(2, 3)) as geom")
    .build()
}

#[derive(Debug)]
struct STMakeLine {
    out_type: SedonaType,
}

impl SedonaScalarKernel for STMakeLine {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let match_geom = ArgMatcher::is_geometry();
        let match_geog = ArgMatcher::is_geography();

        let arg_matchers = if match_geom.match_type(&self.out_type) {
            vec![match_geom.clone(), match_geom]
        } else if match_geog.match_type(&self.out_type) {
            vec![match_geog.clone(), match_geog]
        } else {
            return sedona_internal_err!("Unexpected ST_MakeLine() output");
        };

        let matcher = ArgMatcher::new(arg_matchers, self.out_type.clone());

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);

        let min_segment_bytes = 1 + 4 + 4 + 16 + 16;
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            min_segment_bytes * executor.num_iterations(),
        );

        let mut coords = Vec::new();

        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    invoke_scalar(lhs, rhs, &mut coords, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            };
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    g1: &impl GeometryTrait<T = f64>,
    g2: &impl GeometryTrait<T = f64>,
    coords: &mut Vec<f64>,
    out: &mut impl Write,
) -> Result<()> {
    if g1.dim() != g2.dim() {
        return exec_err!("Can't ST_MakeLine() with mismatched dimensions");
    }

    coords.clear();
    let coord_size = g1.dim().size();

    // Add the first item
    add_coords(g1, coords, coord_size, None)?;

    // If there were any coordinates in the second item, pull the last few items and
    // pass to add_coords() so it can deduplicate
    if coords.len() >= coord_size {
        let mut last_coord = [0.0; 4];
        last_coord[0..coord_size]
            .copy_from_slice(&coords[(coords.len() - coord_size)..coords.len()]);
        add_coords(g2, coords, coord_size, Some(&last_coord[0..coord_size]))?;
    } else {
        add_coords(g2, coords, coord_size, None)?;
    }

    let n_coords = coords.len() / coord_size;
    write_wkb_linestring_header(out, g1.dim(), n_coords)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    for ord in coords {
        out.write_all(&ord.to_le_bytes())?;
    }

    Ok(())
}

fn add_coords(
    geom: &impl GeometryTrait<T = f64>,
    coords: &mut Vec<f64>,
    coord_size: usize,
    last_coord: Option<&[f64]>,
) -> Result<()> {
    match geom.as_type() {
        geo_traits::GeometryType::Point(pt) => {
            if let Some(coord) = pt.coord() {
                for j in 0..coord_size {
                    coords.push(unsafe { coord.nth_unchecked(j) });
                }
            }
        }
        geo_traits::GeometryType::LineString(ls) => {
            for (i, coord) in ls.coords().enumerate() {
                // Deduplicate the first point of any appended linestring
                if i == 0 {
                    let mut tmp = Vec::new();
                    for j in 0..coord_size {
                        tmp.push(unsafe { coord.nth_unchecked(j) });
                    }

                    if last_coord != Some(tmp.as_slice()) {
                        for item in tmp {
                            coords.push(item);
                        }
                    }
                } else {
                    for j in 0..coord_size {
                        coords.push(unsafe { coord.nth_unchecked(j) });
                    }
                }
            }
        }
        geo_traits::GeometryType::MultiPoint(mp) => {
            for pt in mp.points() {
                add_coords(&pt, coords, coord_size, None)?;
            }
        }
        _ => {
            return exec_err!(
                "ST_MakeLine() only supports Point, LineString, and MultiPoint as input"
            )
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        testers::ScalarUdfTester,
        {compare::assert_array_equal, create::create_array},
    };

    #[test]
    fn udf_metadata() {
        let geom_from_point: ScalarUDF = st_makeline_udf().into();
        assert_eq!(geom_from_point.name(), "st_makeline");
        assert!(geom_from_point.documentation().is_some());
    }

    #[rstest]
    fn udf_invoke(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_makeline_udf().into(),
            vec![sedona_type.clone(), sedona_type.clone()],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        // Basic usage
        let result = tester
            .invoke_scalar_scalar("POINT (0 1)", "POINT (2 3)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 1, 2 3)");

        // Deduplicating the first point of a linestring
        let result = tester
            .invoke_scalar_scalar("POINT (0 1)", "LINESTRING (0 1, 2 3)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 1, 2 3)");

        // Two linestrings should work as well
        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 1, 2 3)", "LINESTRING (4 5, 6 7)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 1, 2 3, 4 5, 6 7)");

        // Also multipoints
        let result = tester
            .invoke_scalar_scalar("MULTIPOINT (0 1, 2 3)", "MULTIPOINT (4 5, 6 7)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 1, 2 3, 4 5, 6 7)");

        // Mismatched dimensions or unsupported types should error
        let err = tester
            .invoke_scalar_scalar("POINT (0 1)", "POINT Z (1 2 3)")
            .unwrap_err();
        assert_eq!(
            err.message(),
            "Can't ST_MakeLine() with mismatched dimensions"
        );

        let err = tester
            .invoke_scalar_scalar("POINT (0 1)", "POLYGON EMPTY")
            .unwrap_err();
        assert_eq!(
            err.message(),
            "ST_MakeLine() only supports Point, LineString, and MultiPoint as input"
        );

        // Arrays, nulls, and dimensions
        let array0 = create_array(
            &[
                Some("POINT (0 1)"),
                Some("POINT Z (0 1 2)"),
                Some("POINT M (0 1 3)"),
                Some("POINT ZM (0 1 2 3)"),
                Some("POINT (0 0)"),
                None,
                None,
            ],
            &sedona_type,
        );
        let array1 = create_array(
            &[
                Some("POINT (10 11)"),
                Some("POINT Z (10 11 12)"),
                Some("POINT M (10 11 13)"),
                Some("POINT ZM (10 11 12 13)"),
                None,
                Some("POINT (0 0)"),
                None,
            ],
            &sedona_type,
        );
        let expected = create_array(
            &[
                Some("LINESTRING (0 1, 10 11)"),
                Some("LINESTRING Z (0 1 2, 10 11 12)"),
                Some("LINESTRING M (0 1 3, 10 11 13)"),
                Some("LINESTRING ZM (0 1 2 3, 10 11 12 13)"),
                None,
                None,
                None,
            ],
            &WKB_GEOMETRY,
        );

        let result = tester.invoke_array_array(array0, array1).unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_invoke_geog(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_makeline_udf().into(),
            vec![sedona_type.clone(), sedona_type.clone()],
        );
        tester.assert_return_type(WKB_GEOGRAPHY);

        // Basic usage
        let result = tester
            .invoke_scalar_scalar("POINT (0 1)", "POINT (2 3)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 1, 2 3)");
    }
}
