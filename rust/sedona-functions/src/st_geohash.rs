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
    interval::{Interval, IntervalTrait, WraparoundInterval},
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

        // The CRS is a property of the type, so whether longitudes may be
        // wrapped is decided once for the batch rather than per row.
        let wrap = longitude_wrap_for_arg_type(&arg_types[0])?;

        if args.len() > 1 {
            append_geohash_with_precision(&executor, args, bounder.as_mut(), wrap, &mut builder)?;
        } else {
            append_point_geohash(&executor, bounder.as_mut(), wrap, &mut builder)?;
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

/// Whether an out-of-range longitude may be wrapped back into [-180, 180]
///
/// Wrapping is only meaningful when the coordinates are known to be longitude
/// and latitude in degrees: 181 is then an unambiguous spelling of -179, and
/// hashing it is better than dropping the row. Applied to a projected CRS the
/// same arithmetic would turn a coordinate in metres into a plausible-looking
/// geohash for somewhere it has nothing to do with, so it stays off unless the
/// units are known.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LongitudeWrap {
    /// The argument is known to be in longitude/latitude degrees
    Enabled,
    /// The units are projected, or simply not known, so an out-of-range
    /// coordinate is not necessarily a wrapped longitude
    Disabled,
}

/// Decide whether an argument's coordinates are known to be lon/lat degrees
///
/// A geography is lon/lat by definition. A geometry qualifies only when its
/// type-level CRS resolves to geographic parameters (EPSG:4326, OGC:CRS84,
/// EPSG:4269, ...), which is the same test `st_setsrid` uses to decide whether
/// a CRS is usable as a geography.
///
/// An absent CRS does *not* qualify. Undeclared coordinates are the common case
/// for `ST_GeomFromText()` and could be anything, so they keep the pre-existing
/// null-on-out-of-range behavior rather than being silently reinterpreted.
/// Item-level CRS does not qualify either: [`ItemCrsKernel`] resolves the
/// per-row CRS outside this kernel and hands the inner kernel an item type with
/// no CRS attached, so there is nothing to inspect here.
fn longitude_wrap_for_arg_type(arg_type: &SedonaType) -> Result<LongitudeWrap> {
    let edges = match arg_type {
        SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => *edges,
        // A literal NULL argument: every row is null, so this is never consulted.
        _ => return Ok(LongitudeWrap::Disabled),
    };

    if edges == Edges::Spherical {
        return Ok(LongitudeWrap::Enabled);
    }

    match arg_type.crs() {
        Some(crs) if crs.geographic_params()?.is_some() => Ok(LongitudeWrap::Enabled),
        _ => Ok(LongitudeWrap::Disabled),
    }
}

/// Append the geohash of each geometry at the precision given by the second argument
fn append_geohash_with_precision(
    executor: &WkbExecutor<'_, '_>,
    args: &[ColumnarValue],
    bounder: &mut dyn WkbBounder2D,
    wrap: LongitudeWrap,
    builder: &mut StringBuilder,
) -> Result<()> {
    let precision_value = args[1]
        .cast_to(&DataType::Int64, None)?
        .to_array(executor.num_iterations())?;
    let precision_array = as_int64_array(&precision_value)?;
    let mut precision_iter = precision_array.iter();

    executor.execute_wkb_void(|maybe_wkb| {
        match (maybe_wkb, precision_iter.next().unwrap()) {
            (Some(wkb), Some(precision)) => match invoke_scalar(wkb, precision, bounder, wrap)? {
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
    wrap: LongitudeWrap,
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

                match invoke_scalar(wkb, MAX_PRECISION, bounder, wrap)? {
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
/// is hashed is the center of the geometry's bounding box. Unlike Sedona's Java
/// implementation (where an empty geometry yields JTS' "null envelope" and thus
/// an accidental hash of (-0.5, -0.5)), empty geometries return null here.
///
/// Out-of-range coordinates yield null rather than an error, matching Apache
/// Sedona (GeometryGeoHashEncoder.calculate returns null) rather than PostGIS'
/// `geometry` overload (which raises "Geohash requires inputs in decimal
/// degrees"). The divergence is deliberate: a Spark query that returns nulls
/// for out-of-range input should keep returning nulls here rather than start
/// failing partway through a large scan.
///
/// When `wrap` is [`LongitudeWrap::Enabled`], an out-of-range *longitude* is
/// first wrapped back into [-180, 180] instead of nulling the row, so a
/// longitude of 181 hashes as -179. Latitude is never wrapped; see
/// [`normalize_longitude`] and [`in_latitude_range`] for why the two axes are
/// treated differently.
///
/// The bounding box comes from `bounder`, which the caller resolved from the
/// argument's edge type, so a geography is bounded on the sphere rather than
/// in the plane.
fn invoke_scalar(
    geom: &Wkb,
    precision: i64,
    bounder: &mut dyn WkbBounder2D,
    wrap: LongitudeWrap,
) -> Result<Option<String>> {
    bounder.clear();
    bounder
        .update_wkb_bytes(geom.buf())
        .map_err(|e| sedona_internal_datafusion_err!("Error computing bounds: {e}"))?;
    let (x, y) = bounder.finish();

    if x.is_empty() || y.is_empty() {
        return Ok(None);
    }

    // Latitude can take values in [-90, 90]. Unlike longitude it is not cyclic,
    // so there is no reinterpretation of an out-of-range value that recovers a
    // real location, and it nulls out under both wrap settings.
    if !in_latitude_range(&y) {
        return Ok(None);
    }

    // Longitude can take values in [-180, 180].
    let Some(x) = normalize_longitude(&x, wrap) else {
        return Ok(None);
    };

    let lon = center_longitude(&x);
    let lat = y.lo() + (y.hi() - y.lo()) / 2.0;

    Ok(Some(geohash_encode(lon, lat, precision)))
}

/// Whether a latitude interval lies within [-90, 90]
///
/// Latitude is not cyclic -- the bounder hands it back as a plain [`Interval`]
/// rather than a [`WraparoundInterval`] for exactly that reason -- so a latitude
/// of 100 does not denote a real place the way a longitude of 190 does. PostGIS'
/// `geography` cast reflects it back over the pole (100 becomes 80) but leaves
/// the longitude alone, which lands on a different point than either reading of
/// the input; rather than reproduce that, an out-of-range latitude stays null.
fn in_latitude_range(y: &Interval) -> bool {
    y.lo() >= -90.0 && y.hi() <= 90.0
}

/// Bring a longitude interval into [-180, 180], or reject it
///
/// Returns `None` when the interval cannot be hashed: it is out of range and
/// wrapping is disabled, it is not finite, or it spans a full turn or more (in
/// which case no single longitude is its center).
///
/// Wrapping happens at the interval level rather than per coordinate, which
/// preserves information that per-coordinate wrapping destroys: the bounding
/// box of LINESTRING (179 0, 181 0) is 179..181, which wraps to the two-degree
/// interval 179..-179 centered on 180. Wrapping the coordinates first and
/// bounding afterwards would instead give 179 and -179 to a bounder that knows
/// nothing about wraparound, yielding the 358-degree interval -179..179 and a
/// center of 0 -- the opposite side of the planet.
fn normalize_longitude(x: &WraparoundInterval, wrap: LongitudeWrap) -> Option<WraparoundInterval> {
    // Already in range. This deliberately includes the wraparound intervals a
    // spherical bounder produces (lo > hi, e.g. 170..-170), where both bounds
    // are in range and the interval already means what it should.
    if x.lo() >= -180.0 && x.hi() <= 180.0 {
        return Some(*x);
    }

    if wrap == LongitudeWrap::Disabled {
        return None;
    }

    if !x.lo().is_finite() || !x.hi().is_finite() {
        return None;
    }

    // A box spanning 360 degrees or more covers every longitude, so wrapping it
    // would collapse it to an arbitrary point rather than find its center.
    if x.hi() - x.lo() >= 360.0 {
        return None;
    }

    Some(WraparoundInterval::new(
        wrap_longitude(x.lo()),
        wrap_longitude(x.hi()),
    ))
}

/// Wrap a single longitude into [-180, 180]
///
/// Matches the coercion PostGIS applies when a geometry is cast to `geography`
/// ("Coordinate values were coerced into range [-180 -90, 180 90] for
/// GEOGRAPHY"): 190 becomes -170, -190 becomes 170, and 541 becomes -179.
/// Values already in range are returned untouched so that the closed bounds
/// -180 and 180 keep their sign rather than folding onto each other.
fn wrap_longitude(lon: f64) -> f64 {
    if (-180.0..=180.0).contains(&lon) {
        return lon;
    }

    (lon + 180.0).rem_euclid(360.0) - 180.0
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
    use sedona_schema::crs::{deserialize_crs, lnglat};
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

    /// A geometry type tagged with a lon/lat CRS, which enables longitude wrapping
    fn lnglat_geometry() -> SedonaType {
        SedonaType::Wkb(Edges::Planar, lnglat())
    }

    /// A geometry type tagged with a projected CRS, which does not
    fn projected_geometry() -> SedonaType {
        SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap())
    }

    fn geohash_tester(sedona_type: SedonaType) -> ScalarUdfTester {
        ScalarUdfTester::new(
            st_geohash_udf().into(),
            vec![sedona_type, SedonaType::Arrow(DataType::Int64)],
        )
    }

    #[test]
    fn wrap_longitude_matches_postgis_geography_coercion() {
        // Values pinned against PostGIS 3.6, which reports "Coordinate values
        // were coerced into range [-180 -90, 180 90] for GEOGRAPHY" and then
        // hashes the coerced point:
        //   SELECT ST_AsText('POINT (190 50)'::geography)  -> POINT(-170 50)
        //   SELECT ST_AsText('POINT (-190 50)'::geography) -> POINT(170 50)
        //   SELECT ST_AsText('POINT (541 50)'::geography)  -> POINT(-179 50)
        assert_eq!(wrap_longitude(190.0), -170.0);
        assert_eq!(wrap_longitude(-190.0), 170.0);
        assert_eq!(wrap_longitude(541.0), -179.0);

        // In-range values are returned untouched, so the closed bounds keep
        // their sign instead of folding onto each other.
        assert_eq!(wrap_longitude(180.0), 180.0);
        assert_eq!(wrap_longitude(-180.0), -180.0);
        assert_eq!(wrap_longitude(0.0), 0.0);
    }

    #[test]
    fn normalize_longitude_only_wraps_when_enabled() {
        let out_of_range = WraparoundInterval::new(190.0, 190.0);

        assert_eq!(
            normalize_longitude(&out_of_range, LongitudeWrap::Enabled),
            Some(WraparoundInterval::new(-170.0, -170.0))
        );
        assert_eq!(
            normalize_longitude(&out_of_range, LongitudeWrap::Disabled),
            None
        );

        // An in-range interval is untouched under either setting, including the
        // wraparound intervals a spherical bounder produces.
        for wrap in [LongitudeWrap::Enabled, LongitudeWrap::Disabled] {
            let in_range = WraparoundInterval::new(10.0, 20.0);
            assert_eq!(normalize_longitude(&in_range, wrap), Some(in_range));

            let crosses_antimeridian = WraparoundInterval::new(170.0, -170.0);
            assert_eq!(
                normalize_longitude(&crosses_antimeridian, wrap),
                Some(crosses_antimeridian)
            );
        }
    }

    #[test]
    fn normalize_longitude_rejects_unhashable_intervals() {
        // A box spanning a full turn or more has no single center longitude.
        assert_eq!(
            normalize_longitude(&WraparoundInterval::new(0.0, 360.0), LongitudeWrap::Enabled),
            None
        );
        assert_eq!(
            normalize_longitude(
                &WraparoundInterval::new(-400.0, 400.0),
                LongitudeWrap::Enabled
            ),
            None
        );

        // Non-finite bounds have nothing to wrap into.
        assert_eq!(
            normalize_longitude(
                &WraparoundInterval::new(f64::NEG_INFINITY, f64::INFINITY),
                LongitudeWrap::Enabled
            ),
            None
        );
        assert_eq!(
            normalize_longitude(
                &WraparoundInterval::new(f64::INFINITY, f64::INFINITY),
                LongitudeWrap::Enabled
            ),
            None
        );
    }

    #[rstest]
    fn longitude_wrap_follows_the_crs(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let edges = match &sedona_type {
            SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => *edges,
            _ => unreachable!(),
        };

        // A known lon/lat CRS enables wrapping...
        assert_eq!(
            longitude_wrap_for_arg_type(&SedonaType::Wkb(edges, lnglat())).unwrap(),
            LongitudeWrap::Enabled
        );
        // ...as does NAD83, which geographic_params() also recognizes as degrees.
        assert_eq!(
            longitude_wrap_for_arg_type(&SedonaType::Wkb(
                edges,
                deserialize_crs("EPSG:4269").unwrap()
            ))
            .unwrap(),
            LongitudeWrap::Enabled
        );
        // A geography is lon/lat by definition, whatever its declared CRS.
        assert_eq!(
            longitude_wrap_for_arg_type(&WKB_GEOGRAPHY).unwrap(),
            LongitudeWrap::Enabled
        );

        // A projected CRS does not, because wrapping metres would produce a
        // plausible-looking hash for the wrong place.
        assert_eq!(
            longitude_wrap_for_arg_type(&SedonaType::Wkb(
                edges,
                deserialize_crs("EPSG:3857").unwrap()
            ))
            .unwrap(),
            LongitudeWrap::Disabled
        );
        // Neither does an absent CRS, which is the ST_GeomFromText() default.
        assert_eq!(
            longitude_wrap_for_arg_type(&sedona_type).unwrap(),
            LongitudeWrap::Disabled
        );
    }

    #[test]
    fn udf_wraps_longitude_for_lnglat_crs() {
        let tester = geohash_tester(lnglat_geometry());

        // 190 is the same meridian as -170, so the two hash identically rather
        // than the out-of-range one dropping to null. Pinned against PostGIS:
        //   SELECT ST_GeoHash('POINT (190 50)'::geography, 12)  -> b0zh7w1z0gs3
        //   SELECT ST_GeoHash('POINT (-170 50)'::geography, 12) -> b0zh7w1z0gs3
        let wrapped = tester
            .invoke_scalar_scalar("POINT (190.0 50.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(wrapped, "b0zh7w1z0gs3");

        let equivalent = tester
            .invoke_scalar_scalar("POINT (-170.0 50.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(equivalent, "b0zh7w1z0gs3");

        // -190 wraps the other way, to 170.
        //   SELECT ST_GeoHash('POINT (-190 50)'::geography, 12) -> zbbukqnpp5e9
        //   SELECT ST_GeoHash('POINT (170 50)'::geography, 12)  -> zbbukqnpp5e9
        let result = tester
            .invoke_scalar_scalar("POINT (-190.0 50.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(result, "zbbukqnpp5e9");

        // Multiple turns wrap too: 541 - 720 = -179.
        //   SELECT ST_GeoHash('POINT (541 50)'::geography, 12)  -> b0bsqy0pjew1
        //   SELECT ST_GeoHash('POINT (-179 50)'::geography, 12) -> b0bsqy0pjew1
        let result = tester
            .invoke_scalar_scalar("POINT (541.0 50.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(result, "b0bsqy0pjew1");
    }

    #[test]
    fn udf_wrapping_keeps_a_bbox_centered_across_the_antimeridian() {
        let tester = geohash_tester(lnglat_geometry());

        // The bounding box of this linestring is 179..181, whose center is the
        // antimeridian itself. Wrapping the interval (rather than the vertices)
        // preserves that: the equivalent in-range geometry hashes the same.
        //   SELECT ST_GeoHash(ST_GeomFromText('POINT (180 0)'), 12) -> xbpbpbpbpbpb
        let wrapped = tester
            .invoke_scalar_scalar(
                "LINESTRING (179.0 0.0, 181.0 0.0)",
                ScalarValue::Int64(Some(12)),
            )
            .unwrap();
        tester.assert_scalar_result_equals(wrapped, "xbpbpbpbpbpb");
    }

    #[test]
    fn udf_does_not_wrap_longitude_without_a_lnglat_crs() {
        // A projected CRS keeps the pre-existing null, because an out-of-range
        // easting is not a wrapped longitude.
        let tester = geohash_tester(projected_geometry());
        let result = tester
            .invoke_scalar_scalar("POINT (190.0 50.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));

        // An undeclared CRS does the same.
        let tester = geohash_tester(WKB_GEOMETRY);
        let result = tester
            .invoke_scalar_scalar("POINT (190.0 50.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));
    }

    #[test]
    fn udf_never_wraps_latitude() {
        // Latitude is not cyclic, so it stays null even where longitude wraps.
        // (PostGIS' geography cast would reflect 100 to 80 while leaving the
        // longitude alone, landing on a third point entirely.)
        let tester = geohash_tester(lnglat_geometry());
        for wkt in ["POINT (50.0 100.0)", "POINT (50.0 -100.0)"] {
            let result = tester
                .invoke_scalar_scalar(wkt, ScalarValue::Int64(Some(12)))
                .unwrap();
            tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));
        }

        // Out of range on both axes is null as well -- latitude is checked first
        // and there is no wrapping that rescues it.
        let result = tester
            .invoke_scalar_scalar("POINT (190.0 100.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));
    }

    #[test]
    fn udf_one_arg_wraps_longitude() {
        // The one-argument overload shares invoke_scalar(), so it wraps too.
        //   SELECT ST_GeoHash(ST_GeomFromText('POINT (-170 50)'), 20)
        //     -> b0zh7w1z0gs3y0zh7w1z
        let tester = ScalarUdfTester::new(st_geohash_udf().into(), vec![lnglat_geometry()]);
        let wrapped = tester.invoke_scalar("POINT (190.0 50.0)").unwrap();
        tester.assert_scalar_result_equals(wrapped, "b0zh7w1z0gs3y0zh7w1z");
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
