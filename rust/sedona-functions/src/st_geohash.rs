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
use datafusion_common::{
    cast::as_int64_array,
    config::ConfigOptions,
    error::{DataFusionError, Result},
    exec_err,
};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{GeometryTrait, GeometryType};
use sedona_common::{option::SedonaOptions, sedona_internal_datafusion_err, sedona_internal_err};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{
    bounds::{WkbBounder2D, WkbBounder2DFactory},
    interval::{IntervalTrait, WraparoundInterval},
    types::Edges,
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

/// The base32 alphabet used by geohash encoding (Gustavo Niemeyer's specification)
const BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// The maximum number of geohash characters (matches Apache Sedona's
/// PointGeoHashEncoder, which caps precision at 20)
const MAX_PRECISION: i64 = 20;

/// ST_GeoHash() scalar UDF
///
/// Native implementation to compute the geohash of a geometry or geography.
/// The two-argument form hashes at the requested precision (number of base32
/// characters); the one-argument form hashes a point at [MAX_PRECISION].
pub fn st_geohash_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geohash",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STGeoHash {
                matcher: ArgMatcher::new(
                    vec![ArgMatcher::is_geometry_or_geography()],
                    SedonaType::Arrow(DataType::Utf8),
                ),
            }),
            Arc::new(STGeoHash {
                matcher: ArgMatcher::new(
                    vec![
                        ArgMatcher::is_geometry_or_geography(),
                        ArgMatcher::is_integer(),
                    ],
                    SedonaType::Arrow(DataType::Utf8),
                ),
            }),
        ]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STGeoHash {
    matcher: ArgMatcher,
}

impl SedonaScalarKernel for STGeoHash {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke_batch_from_args(arg_types, args, &SedonaType::Arrow(DataType::Utf8), 0, None)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = StringBuilder::with_capacity(
            executor.num_iterations(),
            MAX_PRECISION as usize * executor.num_iterations(),
        );

        // A bounder is a resettable accumulator, so the batch shares one
        // instance and clear()s it per row rather than allocating per row.
        let mut bounder = bounder_for_arg_type(&arg_types[0], config_options)?;

        if args.len() > 1 {
            append_geohash_with_precision(&executor, args, bounder.as_mut(), &mut builder)?;
        } else {
            append_point_geohash(&executor, bounder.as_mut(), &mut builder)?;
        }

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Resolve the bounder to use for an argument's edge type
///
/// Planar (geometry) arguments always resolve, falling back to the default
/// Cartesian bounder. Spherical (geography) arguments resolve only when a
/// spherical bounder has been registered on the session runtime, which
/// requires the s2geography-backed bounder; there is no planar fallback,
/// because planar bounds of a geography would silently be wrong.
fn bounder_for_arg_type(
    arg_type: &SedonaType,
    config_options: Option<&ConfigOptions>,
) -> Result<Box<dyn WkbBounder2D>> {
    let edges = match arg_type {
        SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => *edges,
        // A literal NULL argument (e.g. ST_GeoHash(NULL, 10)) keeps its Null
        // type: every row is null, so the bounder is never used and the choice
        // of edge type doesn't matter.
        SedonaType::Arrow(DataType::Null) => Edges::Planar,
        _ => {
            return sedona_internal_err!(
                "Expected geometry or geography argument but got {arg_type:?}"
            )
        }
    };

    let maybe_bounder = match config_options.and_then(|o| o.extensions.get::<SedonaOptions>()) {
        Some(options) => options
            .runtime
            .bounder_factory()
            .bounder_for_edge_type(edges),
        None => WkbBounder2DFactory::default().bounder_for_edge_type(edges),
    };

    maybe_bounder.ok_or_else(|| {
        DataFusionError::Execution(
            "ST_GeoHash() on a geography requires the s2geography-backed spherical bounder, \
             which is not registered in this session"
                .to_string(),
        )
    })
}

/// Append the geohash of each geometry at the precision given by the second argument
fn append_geohash_with_precision(
    executor: &WkbExecutor<'_, '_>,
    args: &[ColumnarValue],
    bounder: &mut dyn WkbBounder2D,
    builder: &mut StringBuilder,
) -> Result<()> {
    let precision_value = args[1]
        .cast_to(&DataType::Int64, None)?
        .to_array(executor.num_iterations())?;
    let precision_array = as_int64_array(&precision_value)?;
    let mut precision_iter = precision_array.iter();

    executor.execute_wkb_void(|maybe_wkb| {
        match (maybe_wkb, precision_iter.next().unwrap()) {
            (Some(wkb), Some(precision)) => match invoke_scalar(wkb, precision, bounder)? {
                Some(geohash) => builder.append_value(geohash),
                // Geometry was empty or outside the lon/lat bounds
                None => builder.append_null(),
            },
            _ => builder.append_null(),
        }
        Ok(())
    })
}

/// Append the geohash of each point at [MAX_PRECISION]
///
/// This is the one-argument overload. PostGIS' one-argument ST_GeoHash()
/// derives a precision from the extent of the geometry (the smallest cell that
/// contains it, or a level-20 cell for a point); only the point case is
/// implemented here, where the answer is unambiguous. Anything else errors so
/// that a precision must be stated rather than guessed.
fn append_point_geohash(
    executor: &WkbExecutor<'_, '_>,
    bounder: &mut dyn WkbBounder2D,
    builder: &mut StringBuilder,
) -> Result<()> {
    executor.execute_wkb_void(|maybe_wkb| {
        match maybe_wkb {
            Some(wkb) => {
                // Only a single POINT counts: a MULTIPOINT, even one holding
                // exactly one point, takes the non-point path. PostGIS' rule is
                // really about a zero-area bounding box, but "POINT only" is
                // simpler to state and to predict.
                if !matches!(wkb.as_type(), GeometryType::Point(_)) {
                    return exec_err!(
                        "ST_GeoHash(geometry) is only defined for POINT; pass a precision to \
                         hash the bounding box center of a non-point geometry"
                    );
                }

                match invoke_scalar(wkb, MAX_PRECISION, bounder)? {
                    Some(geohash) => builder.append_value(geohash),
                    // Point was empty or outside the lon/lat bounds
                    None => builder.append_null(),
                }
            }
            None => builder.append_null(),
        }
        Ok(())
    })
}

/// Compute the geohash of a geometry
///
/// Follows Apache Sedona's GeometryGeoHashEncoder.calculate(): the point that
/// is hashed is the center of the geometry's bounding box, and null is
/// returned when the bounding box is not fully contained in
/// [-180, 180] x [-90, 90]. Unlike Sedona's Java implementation (where an
/// empty geometry yields JTS' "null envelope" and thus an accidental hash of
/// (-0.5, -0.5)), empty geometries return null here.
///
/// The bounding box comes from `bounder`, which the caller resolved from the
/// argument's edge type, so a geography is bounded on the sphere rather than
/// in the plane.
fn invoke_scalar(
    geom: &Wkb,
    precision: i64,
    bounder: &mut dyn WkbBounder2D,
) -> Result<Option<String>> {
    bounder.clear();
    bounder
        .update_wkb_bytes(geom.buf())
        .map_err(|e| sedona_internal_datafusion_err!("Error computing bounds: {e}"))?;
    let (x, y) = bounder.finish();

    if x.is_empty() || y.is_empty() {
        return Ok(None);
    }

    // Longitude can take values in [-180, 180]; latitude can take values in [-90, 90].
    // Out-of-range coordinates yield null rather than an error, matching Apache
    // Sedona (GeometryGeoHashEncoder.calculate returns null here). PostGIS instead
    // raises "Geohash requires inputs in decimal degrees"; the divergence is
    // deliberate, since a Spark query that returns nulls for out-of-range input
    // should keep returning nulls here rather than start failing.
    if x.lo() < -180.0 || y.lo() < -90.0 || x.hi() > 180.0 || y.hi() > 90.0 {
        return Ok(None);
    }

    let lon = center_longitude(&x);
    let lat = y.lo() + (y.hi() - y.lo()) / 2.0;

    Ok(Some(geohash_encode(lon, lat, precision)))
}

/// The center of a longitude interval, in [-180, 180]
///
/// A spherical bounder can return an interval that crosses the antimeridian,
/// which is expressed as `lo > hi` (e.g. (170, -170) covers 20 degrees through
/// 180, not the 340 degrees through 0). For those, `lo + (hi - lo) / 2` walks
/// the wrong way around the sphere, so measure the width eastward from `lo`
/// and wrap the result back into [-180, 180].
fn center_longitude(x: &WraparoundInterval) -> f64 {
    if !x.is_wraparound() {
        return x.lo() + (x.hi() - x.lo()) / 2.0;
    }

    let center = x.lo() + ((x.hi() + 360.0) - x.lo()) / 2.0;
    if center > 180.0 {
        center - 360.0
    } else {
        center
    }
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
    use sedona_geometry::bounds::WkbGeometryBounder;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
        WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{create::create_array as create_wkb_array, testers::ScalarUdfTester};

    use super::*;

    /// A tester whose session has a spherical bounder registered
    ///
    /// sedona-functions does not depend on sedona-s2geography, so these tests
    /// stand in the Cartesian bounder for the spherical one. That keeps the
    /// expected values identical to the geometry cases while still exercising
    /// the geography path end to end: the geography argument matches a kernel,
    /// the bounder comes from the session runtime rather than being hard-coded,
    /// and its output drives the hash. The spherical-vs-planar difference in
    /// the bounds themselves is sedona-s2geography's contract, not this
    /// function's.
    fn tester_with_stand_in_spherical_bounder(arg_types: Vec<SedonaType>) -> ScalarUdfTester {
        let mut tester = ScalarUdfTester::new(st_geohash_udf().into(), arg_types);
        let options = tester.sedona_options_mut();
        options.runtime = options
            .runtime
            .with_bounder(Edges::Spherical, Arc::new(WkbGeometryBounder::default()))
            .unwrap();
        tester
    }

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

        // Empty geometries have no bounding box to hash. PostGIS agrees on all
        // of these (lwgeom_geohash() returns NULL when the gbox can't be
        // computed), which the Python comparison tests pin.
        for wkt in [
            "POINT EMPTY",
            "LINESTRING EMPTY",
            "POLYGON EMPTY",
            "MULTIPOINT EMPTY",
            "MULTILINESTRING EMPTY",
            "MULTIPOLYGON EMPTY",
            "GEOMETRYCOLLECTION EMPTY",
        ] {
            let result = tester
                .invoke_scalar_scalar(wkt, ScalarValue::Int64(Some(10)))
                .unwrap();
            assert!(result.is_null(), "expected null for {wkt}");
        }

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

    #[rstest]
    fn udf_one_arg(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_geohash_udf().into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Utf8);

        // A point with no precision hashes at the 20 character maximum, and
        // extends the precision 10 value pinned in udf() above.
        let result = tester.invoke_scalar("POINT (21.4234 52.0423)").unwrap();
        tester.assert_scalar_result_equals(result, "u3r0pd0037ugg6hm1kb1");

        // Western and southern hemispheres, extending the precision 9 value
        // pinned against PostGIS in the Python comparison tests
        let result = tester.invoke_scalar("POINT (-122.4194 37.7749)").unwrap();
        tester.assert_scalar_result_equals(result, "9q8yyk8ytpxr8wwhcg8j");

        // Null geometry
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        // An empty point has no bounding box to hash, as in the two argument form
        let result = tester.invoke_scalar("POINT EMPTY").unwrap();
        assert!(result.is_null());

        // Out of range coordinates return null, as in the two argument form
        let result = tester.invoke_scalar("POINT (-190.0 50.0)").unwrap();
        assert!(result.is_null());
    }

    #[rstest]
    fn udf_one_arg_requires_point(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_geohash_udf().into(), vec![sedona_type]);

        // Every non-point geometry errors rather than guessing a precision.
        // MULTIPOINT is included deliberately: a single-element MULTIPOINT is
        // not treated as a point.
        for wkt in [
            "LINESTRING (30 10, 10 30, 40 40)",
            "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10))",
            "MULTIPOINT ((10 40))",
            "GEOMETRYCOLLECTION (POINT (40 10))",
            // Empty non-points error too: the geometry type decides, not the bounds
            "LINESTRING EMPTY",
        ] {
            let err = tester.invoke_scalar(wkt).unwrap_err().to_string();
            assert!(
                err.contains("ST_GeoHash(geometry) is only defined for POINT"),
                "unexpected error for {wkt}: {err}"
            );
        }
    }

    #[rstest]
    fn udf_geography(
        #[values(
            WKB_GEOGRAPHY,
            WKB_VIEW_GEOGRAPHY,
            WKB_GEOGRAPHY_ITEM_CRS.clone()
        )]
        sedona_type: SedonaType,
    ) {
        let tester = tester_with_stand_in_spherical_bounder(vec![
            sedona_type.clone(),
            SedonaType::Arrow(DataType::Int64),
        ]);
        tester.assert_return_type(DataType::Utf8);

        let result = tester
            .invoke_scalar_scalar("POINT (21.4234 52.0423)", ScalarValue::Int64(Some(10)))
            .unwrap();
        tester.assert_scalar_result_equals(result, "u3r0pd0037");

        let result = tester
            .invoke_scalar_scalar("POINT EMPTY", ScalarValue::Int64(Some(10)))
            .unwrap();
        assert!(result.is_null());

        // The one argument overload applies to geography as well
        let tester = tester_with_stand_in_spherical_bounder(vec![sedona_type]);
        tester.assert_return_type(DataType::Utf8);

        let result = tester.invoke_scalar("POINT (21.4234 52.0423)").unwrap();
        tester.assert_scalar_result_equals(result, "u3r0pd0037ugg6hm1kb1");

        let err = tester
            .invoke_scalar("LINESTRING (30 10, 10 30, 40 40)")
            .unwrap_err()
            .to_string();
        assert!(err.contains("ST_GeoHash(geometry) is only defined for POINT"));
    }

    #[rstest]
    fn udf_geography_without_spherical_bounder(
        #[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType,
    ) {
        // A default session has no spherical bounder, so a geography argument
        // errors rather than silently falling back to planar bounds.
        let tester = ScalarUdfTester::new(
            st_geohash_udf().into(),
            vec![sedona_type, SedonaType::Arrow(DataType::Int64)],
        );

        let err = tester
            .invoke_scalar_scalar("POINT (21.4234 52.0423)", ScalarValue::Int64(Some(10)))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("requires the s2geography-backed spherical bounder"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn udf_untyped_null_argument() {
        // A literal NULL (SELECT ST_GeoHash(NULL, 10)) keeps its Null type all
        // the way into the kernel, so resolving a bounder has to tolerate an
        // argument type that names no edge type.
        let tester = ScalarUdfTester::new(
            st_geohash_udf().into(),
            vec![
                SedonaType::Arrow(DataType::Null),
                SedonaType::Arrow(DataType::Int64),
            ],
        );
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Int64(Some(10)))
            .unwrap();
        assert!(result.is_null());

        let tester = ScalarUdfTester::new(
            st_geohash_udf().into(),
            vec![SedonaType::Arrow(DataType::Null)],
        );
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());
    }

    #[test]
    fn center_longitude_handles_wraparound() {
        // A plain interval keeps the arithmetic mean
        assert_eq!(
            center_longitude(&WraparoundInterval::new(10.0, 30.0)),
            20.0_f64
        );

        // An interval crossing the antimeridian covers lo -> 180 -> hi, so its
        // center is the eastward midpoint, not the midpoint of [hi, lo]
        assert_eq!(
            center_longitude(&WraparoundInterval::new(170.0, -170.0)),
            180.0_f64
        );
        assert_eq!(
            center_longitude(&WraparoundInterval::new(175.0, -160.0)),
            -172.5_f64
        );
        assert_eq!(
            center_longitude(&WraparoundInterval::new(160.0, -170.0)),
            175.0_f64
        );
    }
}
