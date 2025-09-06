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

use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::ColumnarValue;
use geo_generic_alg::{Euclidean, Length};
use sedona_expr::scalar_udf::{ArgMatcher, ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_functions::st_isempty::is_wkb_empty;
use sedona_schema::datatypes::SedonaType;
use wkb::reader::Wkb;

use crate::to_geo::item_to_geometry;

/// ST_Length() implementation using [EuclideanLength]
pub fn st_length_impl() -> ScalarKernelRef {
    Arc::new(STLength {})
}

#[derive(Debug)]
struct STLength {}

impl SedonaScalarKernel for STLength {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    builder.append_value(invoke_scalar(&wkb)?);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(wkb: &Wkb) -> Result<f64> {
    // Check if geometry is empty using st_isempty logic to avoid expensive conversion
    if is_wkb_empty(wkb)? {
        return Ok(0.0);
    }

    // Convert to geometry for length calculation only if not empty (fallback if direct doesn't work)
    // TODO: Optimize by directly calculating length from WKB after the geo-alg crate supports it
    let geom = item_to_geometry(wkb)?;

    use geo_generic_alg::Geometry;
    let length = match geom {
        Geometry::Line(line) => Euclidean.length(&line),
        Geometry::LineString(linestring) => Euclidean.length(&linestring),
        Geometry::MultiLineString(multilinestring) => Euclidean.length(&multilinestring),
        Geometry::Polygon(_polygon) => {
            // OGC standard: polygons have zero length (only line-like geometries have length)
            0.0
        }
        Geometry::MultiPolygon(_multipolygon) => {
            // OGC standard: polygons have zero length (only line-like geometries have length)
            0.0
        }
        Geometry::GeometryCollection(collection) => {
            // Recursively calculate length for all geometries in collection
            let mut total_length = 0.0;
            for geom in collection.iter() {
                total_length += invoke_scalar_for_geometry(geom)?;
            }
            total_length
        }
        // Points have zero length
        _ => 0.0,
    };

    Ok(length)
}

// Helper function for recursion in GeometryCollection
fn invoke_scalar_for_geometry(geom: &geo_generic_alg::Geometry<f64>) -> Result<f64> {
    use geo_generic_alg::Geometry;
    let length = match geom {
        Geometry::Line(line) => Euclidean.length(line),
        Geometry::LineString(linestring) => Euclidean.length(linestring),
        Geometry::MultiLineString(multilinestring) => Euclidean.length(multilinestring),
        Geometry::Polygon(_polygon) => {
            // OGC standard: polygons have zero length (only line-like geometries have length)
            0.0
        }
        Geometry::MultiPolygon(_multipolygon) => {
            // OGC standard: polygons have zero length (only line-like geometries have length)
            0.0
        }
        Geometry::GeometryCollection(collection) => {
            let mut total_length = 0.0;
            for geom in collection.iter() {
                total_length += invoke_scalar_for_geometry(geom)?;
            }
            total_length
        }
        _ => 0.0,
    };

    Ok(length)
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_functions::register::stubs::st_length_udf;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let mut udf = st_length_udf();
        udf.add_kernel(st_length_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Float64)
        );

        // Test with a line string
        assert_eq!(
            tester
                .invoke_wkb_scalar(Some("LINESTRING (0 0, 3 4)"))
                .unwrap(),
            ScalarValue::Float64(Some(5.0))
        );

        let input_wkt = vec![
            Some("POINT(1 2)"), // Point should have 0 length
            None,
            Some("LINESTRING (0 0, 3 4)"),      // Should have length 5
            Some("LINESTRING (0 0, 1 0, 1 1)"), // Should have length 2
        ];
        let expected: ArrayRef = create_array!(Float64, [Some(0.0), None, Some(5.0), Some(2.0)]);
        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
