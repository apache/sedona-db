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

use arrow_array::builder::BinaryBuilder;
use datafusion_common::{exec_datafusion_err, Result};
use datafusion_expr::ColumnarValue;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_functions::executor::WkbBytesExecutor;
use sedona_geometry::wkb_factory::{
    write_wkb_empty_point, write_wkb_multipolygon, write_wkb_point, write_wkb_polygon,
    WKB_MIN_PROBABLE_BYTES,
};
use sedona_geometry::{types::GeometryTypeId, wkb_header::WkbHeader};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::geography::{Geography, GeographyFactory};
use crate::rect_bounder::RectBounder;

/// Returns a vector of (function_name, kernel) tuples for the ST_Envelope function
/// Includes both the base kernel and ItemCrs-wrapped version
pub fn st_envelope_kernels() -> Vec<(String, ScalarKernelRef)> {
    ItemCrsKernel::wrap_impl(vec![Arc::new(STEnvelope {}) as ScalarKernelRef])
        .into_iter()
        .map(|kernel| ("st_envelope".to_string(), kernel))
        .collect()
}

#[derive(Debug)]
struct STEnvelope {}

impl SedonaScalarKernel for STEnvelope {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // ST_Envelope() always returns geometry, even for geography input
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOMETRY);
        matcher.match_args(args)
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
        let mut factory = GeographyFactory::new();
        let mut geog = Geography::new();
        let mut bounder = RectBounder::new();

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    // Check if input is a Point - if so, return a 2D copy directly
                    let header = WkbHeader::try_new(wkb)
                        .map_err(|e| exec_datafusion_err!("Error parsing WKB header: {e}"))?;

                    if header
                        .geometry_type_id()
                        .map_err(|e| exec_datafusion_err!("Error getting geometry type: {e}"))?
                        == GeometryTypeId::Point
                    {
                        invoke_scalar_point(&header, &mut builder)?;
                    } else {
                        factory
                            .init_from_wkb(wkb, &mut geog)
                            .map_err(|e| exec_datafusion_err!("Error parsing geography: {e}"))?;
                        invoke_scalar_bounds(&geog, &mut bounder, &mut builder)?;
                    }
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Handle Point input - return a 2D copy of the point
fn invoke_scalar_point(header: &WkbHeader, writer: &mut impl std::io::Write) -> Result<()> {
    let is_empty = header
        .is_empty()
        .map_err(|e| exec_datafusion_err!("Error checking if point is empty: {e}"))?;

    if is_empty {
        write_wkb_empty_point(writer, geo_traits::Dimensions::Xy)
            .map_err(|e| exec_datafusion_err!("Error writing empty point: {e}"))?;
    } else {
        let (x, y) = header.first_xy();
        write_wkb_point(writer, (x, y))
            .map_err(|e| exec_datafusion_err!("Error writing point: {e}"))?;
    }

    Ok(())
}

/// Handle non-Point input - compute bounding box
fn invoke_scalar_bounds(
    geog: &Geography,
    bounder: &mut RectBounder,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    // Clear the bounder for reuse
    bounder.clear();

    // Add the geography to the bounder
    bounder
        .bound(geog)
        .map_err(|e| exec_datafusion_err!("Error bounding geography: {e}"))?;

    // Get the bounding rectangle
    let Some((xmin, ymin, xmax, ymax)) = bounder
        .finish()
        .map_err(|e| exec_datafusion_err!("Error finishing bounds: {e}"))?
    else {
        // Empty geography - write an empty point
        write_wkb_empty_point(writer, geo_traits::Dimensions::Xy)
            .map_err(|e| exec_datafusion_err!("Error writing empty point: {e}"))?;
        return Ok(());
    };

    // Check for wraparound case (xmin > xmax means crossing the antimeridian)
    // For non-Point geographies, we always output a polygon or multipolygon
    // (geodesic calculations ensure non-degenerate bounds for linestrings/polygons)
    if xmin > xmax {
        // Wraparound case: MULTIPOLYGON with two polygons
        // One from xmin to 180, one from -180 to xmax
        let poly1 = vec![
            (xmin, ymin),
            (xmin, ymax),
            (180.0, ymax),
            (180.0, ymin),
            (xmin, ymin),
        ];
        let poly2 = vec![
            (-180.0, ymin),
            (-180.0, ymax),
            (xmax, ymax),
            (xmax, ymin),
            (-180.0, ymin),
        ];
        write_wkb_multipolygon(writer, [poly1, poly2].into_iter())
            .map_err(|e| exec_datafusion_err!("Error writing multipolygon: {e}"))?;
    } else {
        // Non-wraparound case: POLYGON
        write_wkb_polygon(
            writer,
            [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ]
            .into_iter(),
        )
        .map_err(|e| exec_datafusion_err!("Error writing polygon: {e}"))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_VIEW_GEOGRAPHY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    fn create_udf() -> SedonaScalarUDF {
        let kernels = st_envelope_kernels();
        SedonaScalarUDF::new(
            "st_envelope",
            kernels.into_iter().map(|(_, k)| k).collect(),
            datafusion_expr::Volatility::Immutable,
        )
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = create_udf().into();
        assert_eq!(udf.name(), "st_envelope");
    }

    #[rstest]
    fn udf_invoke_array(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(WKB_GEOMETRY);

        let input_wkt = vec![
            None,
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
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
                Some("POINT EMPTY"),
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOGRAPHY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type);

        let result = tester.invoke_scalar("POINT EMPTY").unwrap();
        tester.assert_scalar_result_equals(result, "POINT EMPTY");
    }
}
