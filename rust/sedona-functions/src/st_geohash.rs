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

use crate::executor::WkbExecutor;
use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::{cast::as_int64_array, error::Result};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::GeometryTrait;
use sedona_common::sedona_internal_datafusion_err;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{bounds::geo_traits_bounds_xy, interval::IntervalTrait};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// The base32 alphabet used by geohash encoding (Gustavo Niemeyer's specification)
const BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// The maximum number of geohash characters (matches Apache Sedona's
/// PointGeoHashEncoder, which caps precision at 20)
const MAX_PRECISION: i64 = 20;

/// ST_GeoHash() scalar UDF
///
/// Native implementation to compute the geohash of a geometry at the given
/// precision (number of base32 characters)
pub fn st_geohash_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geohash",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STGeoHash {})]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STGeoHash {}

impl SedonaScalarKernel for STGeoHash {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_integer()],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = StringBuilder::with_capacity(
            executor.num_iterations(),
            MAX_PRECISION as usize * executor.num_iterations(),
        );

        let precision_value = args[1]
            .cast_to(&DataType::Int64, None)?
            .to_array(executor.num_iterations())?;
        let precision_array = as_int64_array(&precision_value)?;
        let mut precision_iter = precision_array.iter();

        executor.execute_wkb_void(|maybe_wkb| {
            match (maybe_wkb, precision_iter.next().unwrap()) {
                (Some(wkb), Some(precision)) => match invoke_scalar(wkb, precision)? {
                    Some(geohash) => builder.append_value(geohash),
                    // Geometry was empty or outside the lon/lat bounds
                    None => builder.append_null(),
                },
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Compute the geohash of a geometry
///
/// Follows Apache Sedona's GeometryGeoHashEncoder.calculate(): the point that
/// is hashed is the center of the geometry's bounding box, and null is
/// returned when the bounding box is not fully contained in
/// [-180, 180] x [-90, 90]. Unlike Sedona's Java implementation (where an
/// empty geometry yields JTS' "null envelope" and thus an accidental hash of
/// (-0.5, -0.5)), empty geometries return null here.
fn invoke_scalar(geom: impl GeometryTrait<T = f64>, precision: i64) -> Result<Option<String>> {
    let bounds = geo_traits_bounds_xy(geom)
        .map_err(|e| sedona_internal_datafusion_err!("Error computing bounds: {e}"))?;
    let (x, y) = (bounds.x(), bounds.y());
    if x.is_empty() || y.is_empty() {
        return Ok(None);
    }

    // Longitude can take values in [-180, 180]; latitude can take values in [-90, 90]
    if x.lo() < -180.0 || y.lo() < -90.0 || x.hi() > 180.0 || y.hi() > 90.0 {
        return Ok(None);
    }

    let lon = x.lo() + (x.hi() - x.lo()) / 2.0;
    let lat = y.lo() + (y.hi() - y.lo()) / 2.0;

    Ok(Some(geohash_encode(lon, lat, precision)))
}

/// Encode a lon/lat pair as a geohash string with `precision` base32 characters
///
/// Non-positive precisions result in an empty string and precisions greater
/// than 20 are truncated to 20, matching Apache Sedona's PointGeoHashEncoder.
fn geohash_encode(lon: f64, lat: f64, precision: i64) -> String {
    if precision <= 0 {
        return String::new();
    }

    let precision = precision.min(MAX_PRECISION) as usize;
    let mut out = String::with_capacity(precision);

    let (mut lon_min, mut lon_max) = (-180.0_f64, 180.0_f64);
    let (mut lat_min, mut lat_max) = (-90.0_f64, 90.0_f64);
    let mut is_even = true;
    let mut bit = 0;
    let mut ch = 0_usize;

    while out.len() < precision {
        let (value, min, max) = if is_even {
            (lon, &mut lon_min, &mut lon_max)
        } else {
            (lat, &mut lat_min, &mut lat_max)
        };

        let mid = (*min + *max) / 2.0;
        if value >= mid {
            ch = (ch << 1) | 1;
            *min = mid;
        } else {
            ch <<= 1;
            *max = mid;
        }

        is_even = !is_even;
        bit += 1;
        if bit == 5 {
            out.push(BASE32[ch] as char);
            bit = 0;
            ch = 0;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{create::create_array as create_wkb_array, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_geohash_udf().into();
        assert_eq!(udf.name(), "st_geohash");
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(
            st_geohash_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(arrow_schema::DataType::Int64),
            ],
        );
        tester.assert_return_type(DataType::Utf8);

        // Expected values from Apache Sedona's
        // spark/common/src/test/scala/org/apache/sedona/sql/functions/geohash/TestStGeoHash.scala
        let result = tester
            .invoke_scalar_scalar("POINT (21.4234 52.0423)", ScalarValue::Int64(Some(10)))
            .unwrap();
        tester.assert_scalar_result_equals(result, "u3r0pd0037");

        // Null geometry
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Int64(Some(10)))
            .unwrap();
        assert!(result.is_null());

        // Null precision
        let result = tester
            .invoke_scalar_scalar("POINT (21.4234 52.0423)", ScalarValue::Int64(None))
            .unwrap();
        assert!(result.is_null());

        // Empty geometries have no bounding box to hash
        let result = tester
            .invoke_scalar_scalar("POINT EMPTY", ScalarValue::Int64(Some(10)))
            .unwrap();
        assert!(result.is_null());

        let result = tester
            .invoke_scalar_scalar("GEOMETRYCOLLECTION EMPTY", ScalarValue::Int64(Some(10)))
            .unwrap();
        assert!(result.is_null());

        // Non-point geometries hash the center of their bounding box. Expected
        // values from TestStGeoHash.scala "should return geohash" (precision 10)
        let input_wkt = create_wkb_array(
            &[
                Some("POINT (21.4234 52.0423)"),
                Some("LINESTRING (30 10, 10 30, 40 40)"),
                Some("POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))"),
                Some("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))"),
                Some("MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"),
                Some("GEOMETRYCOLLECTION (POINT (40 10), LINESTRING (10 10, 20 20, 10 40), POLYGON ((40 40, 20 45, 45 30, 40 40)))"),
                None,
            ],
            &sedona_type,
        );
        let precisions = arrow_array::create_array!(
            Int64,
            [
                Some(10),
                Some(10),
                Some(10),
                Some(10),
                Some(10),
                Some(10),
                Some(10)
            ]
        );
        let expected: ArrayRef = create_array!(
            Utf8,
            [
                Some("u3r0pd0037"),
                Some("ss3y0zh7w1"),
                Some("ssgs3y0zh7"),
                Some("ss3y0zh7w1"),
                Some("ss1b0bh2n0"),
                Some("ssgs3y0zh7"),
                None
            ]
        );
        assert_eq!(
            &tester.invoke_arrays(vec![input_wkt, precisions]).unwrap(),
            &expected
        );
    }

    #[rstest]
    fn udf_precision_bounds(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_geohash_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(arrow_schema::DataType::Int64),
            ],
        );

        // Precision is truncated to the maximum of 20; expected value from
        // TestStGeoHash.scala "should return geohash truncated to max value"
        let result = tester
            .invoke_scalar_scalar(
                "POINT (21.427834 52.042576573)",
                ScalarValue::Int64(Some(21)),
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "u3r0pd53bxrjdsrz4fzj");

        // Non-positive precision returns an empty string; from
        // TestStGeoHash.scala "should return empty string when precision is negative or equal 0"
        let result = tester
            .invoke_scalar_scalar(
                "POINT (21.427834 52.042576573)",
                ScalarValue::Int64(Some(0)),
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "");

        let result = tester
            .invoke_scalar_scalar(
                "POINT (21.427834 52.042576573)",
                ScalarValue::Int64(Some(-1)),
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "");
    }

    #[rstest]
    fn udf_coordinate_bounds(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_geohash_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(arrow_schema::DataType::Int64),
            ],
        );

        // Expected value from TestStGeoHash.scala
        // "should not return null for 90 < long < 180 (SEDONA-123)"
        let result = tester
            .invoke_scalar_scalar("POINT (120.0 50.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(result, "y8vk6wjr4et3");

        // Boundary cases of min/max lon/lat; expected values from
        // TestStGeoHash.scala "should return expected value for boundary case of min lat/long"
        // and "... of max lat/long"
        let result = tester
            .invoke_scalar_scalar("POINT (-180.0 -90.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(result, "000000000000");

        let result = tester
            .invoke_scalar_scalar("POINT (180.0 90.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(result, "zzzzzzzzzzzz");

        // Coordinates outside [-180, 180] x [-90, 90] return null; from
        // TestStGeoHash.scala "should return null when geometry contains invalid coordinates"
        for wkt in [
            "POINT (-190.0 50.0)",
            "POINT (190.0 50.0)",
            "POINT (50.0 -100.0)",
            "POINT (50.0 100.0)",
        ] {
            let result = tester
                .invoke_scalar_scalar(wkt, ScalarValue::Int64(Some(1)))
                .unwrap();
            assert!(result.is_null());
        }
    }
}
