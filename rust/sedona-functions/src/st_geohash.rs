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

use std::{collections::HashMap, iter::zip, sync::Arc};

use crate::executor::WkbExecutor;
use arrow_array::{builder::StringBuilder, Array};
use arrow_schema::DataType;
use datafusion_common::{
    cast::{as_int64_array, as_string_view_array, as_struct_array},
    config::ConfigOptions,
    error::{DataFusionError, Result},
    exec_err, plan_err, ScalarValue,
};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{GeometryTrait, GeometryType};
use sedona_common::{option::SedonaOptions, sedona_internal_datafusion_err, sedona_internal_err};
use sedona_expr::{
    item_crs::parse_item_crs_arg_type,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{
    bounds::{WkbBounder2D, WkbBounder2DFactory},
    interval::{Interval, IntervalTrait, WraparoundInterval},
    types::Edges,
};
use sedona_schema::{
    crs::{deserialize_crs, lnglat},
    datatypes::SedonaType,
    matchers::ArgMatcher,
};
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
        // ST_GeoHash cannot use ItemCrsKernel::wrap_impl(): that wrapper strips
        // the CRS before calling the inner kernel, which is what makes it free
        // for functions like ST_Area() whose answer does not depend on the CRS.
        // This one rejects a non-WGS84 CRS, so it needs to see it. The sibling
        // kernels below are the same shape ST_SRID() and ST_CRS() use, one
        // matching the plain types and one matching item_crs.
        vec![
            Arc::new(STGeoHashItemCrs {
                matcher: ArgMatcher::new(
                    vec![ArgMatcher::is_item_crs()],
                    SedonaType::Arrow(DataType::Utf8),
                ),
            }) as _,
            Arc::new(STGeoHashItemCrs {
                matcher: ArgMatcher::new(
                    vec![ArgMatcher::is_item_crs(), ArgMatcher::is_integer()],
                    SedonaType::Arrow(DataType::Utf8),
                ),
            }) as _,
            Arc::new(STGeoHash {
                matcher: ArgMatcher::new(
                    vec![ArgMatcher::is_geometry_or_geography()],
                    SedonaType::Arrow(DataType::Utf8),
                ),
            }) as _,
            Arc::new(STGeoHash {
                matcher: ArgMatcher::new(
                    vec![
                        ArgMatcher::is_geometry_or_geography(),
                        ArgMatcher::is_integer(),
                    ],
                    SedonaType::Arrow(DataType::Utf8),
                ),
            }) as _,
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STGeoHash {
    matcher: ArgMatcher,
}

impl SedonaScalarKernel for STGeoHash {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let Some(out_type) = self.matcher.match_args(args)? else {
            return Ok(None);
        };

        // Checked only after the argument shapes match, so that a call with a
        // different arity falls through to the next kernel rather than erroring
        // out of kernel resolution here.
        ensure_wgs84_crs(&args[0])?;

        Ok(Some(out_type))
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
        let mut raw_bounder = raw_bounder_for_arg_type(&arg_types[0]);

        // The CRS is a property of the type, so this is decided once for the
        // batch rather than per row.
        let wrap = longitude_wrap_for_arg_type(&arg_types[0]);
        let mut next_wrap = move || Ok(wrap);

        if args.len() > 1 {
            append_geohash_with_precision(
                &executor,
                args,
                bounder.as_mut(),
                &mut raw_bounder,
                &mut next_wrap,
                &mut builder,
            )?;
        } else {
            append_point_geohash(
                &executor,
                bounder.as_mut(),
                &mut raw_bounder,
                &mut next_wrap,
                &mut builder,
            )?;
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

/// A planar bounder for range-checking the coordinates a spherical bounder hides
///
/// s2geography bounds a geography with an `S2LatLngRect`, whose latitudes are
/// constrained to [-90, 90] by construction, so an out-of-range latitude is
/// clamped inside S2 before [`WkbBounder2D::finish`] returns it: a geography at
/// latitude 100, 91 or 270 all come back as 90 and would otherwise encode as
/// the pole. Bounding the same bytes in the plane recovers what the input
/// actually said, so the domain check sees the original value.
///
/// Only the latitude check uses this. The centre still comes from the spherical
/// bounds, which is the point of bounding a geography on the sphere. Longitude
/// needs no such care: S2 normalizes it into [-180, 180], which is the same
/// wrapping this function applies to a geometry.
///
/// Returns `None` for a planar argument, whose bounder does no clamping and can
/// be range-checked directly.
fn raw_bounder_for_arg_type(arg_type: &SedonaType) -> Option<Box<dyn WkbBounder2D>> {
    let edges = match arg_type {
        SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => *edges,
        _ => return None,
    };

    match edges {
        Edges::Planar => None,
        _ => WkbBounder2DFactory::default().bounder_for_edge_type(Edges::Planar),
    }
}

/// Per-batch cache of the WGS84 verdict for each distinct CRS string
///
/// Deserializing a CRS is expensive and an item_crs column carries only a
/// handful of distinct values across many rows, so each is resolved once.
/// Mirrors `CachedCrsToSRIDMapping`, which exists for the same reason.
#[derive(Default)]
struct CachedWgs84Check {
    /// `None` marks a CRS that is not WGS84, i.e. one this function rejects
    cache: HashMap<String, Option<LongitudeWrap>>,
}

impl CachedWgs84Check {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
        }
    }

    /// Resolve one row's CRS into a wrap setting, or reject the row
    ///
    /// Applies exactly the rule [`ensure_wgs84_crs`] applies to a type-level
    /// CRS, so the same geometry is treated the same way whether its CRS is
    /// attached to the type or to the row.
    fn wrap_for(&mut self, maybe_crs: Option<&str>) -> Result<LongitudeWrap> {
        // No CRS on this row: accepted and assumed WGS84, but not wrapped,
        // matching an absent type-level CRS.
        let Some(crs_str) = maybe_crs else {
            return Ok(LongitudeWrap::Disabled);
        };

        if let Some(cached) = self.cache.get(crs_str) {
            return cached.map_or_else(|| non_wgs84_err(crs_str), Ok);
        }

        // `deserialize_crs` yields None for the "no CRS" sentinels ("0", "").
        let crs = deserialize_crs(crs_str)?;
        let verdict = if crs.is_none() {
            Some(LongitudeWrap::Disabled)
        } else if crs == lnglat() {
            Some(LongitudeWrap::Enabled)
        } else {
            None
        };

        self.cache.insert(crs_str.to_string(), verdict);
        verdict.map_or_else(|| non_wgs84_err(crs_str), Ok)
    }
}

fn non_wgs84_err<T>(name: &str) -> Result<T> {
    exec_err!(
        "ST_GeoHash() requires WGS84 longitude/latitude coordinates but a row has CRS \
         '{name}'. Use ST_Transform() to reproject to EPSG:4326 first."
    )
}

/// ST_GeoHash() over an item_crs argument, where each row carries its own CRS
///
/// The rule is the one [`ensure_wgs84_crs`] applies at the type level; only the
/// moment of detection differs. A type-level CRS is known while the query is
/// planned, so it is rejected before any row is read; a row-level CRS is not
/// known until the row is in hand, so it is rejected then. Erroring rather than
/// nulling keeps the two consistent, and matches how the crate already treats a
/// bad row-level CRS elsewhere (`ensure_crs_string_arrays_equal2` raises on the
/// first mismatched row).
///
/// Seeing the per-row CRS also means a row that declares WGS84 gets longitude
/// wrapping, so item-level and type-level WGS84 agree on the same input rather
/// than one nulling where the other wraps.
#[derive(Debug)]
struct STGeoHashItemCrs {
    matcher: ArgMatcher,
}

impl SedonaScalarKernel for STGeoHashItemCrs {
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
        let struct_array = match &args[0] {
            ColumnarValue::Array(array) => as_struct_array(array)?,
            ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => struct_array.as_ref(),
            ColumnarValue::Scalar(ScalarValue::Null) => {
                return Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)));
            }
            _ => return sedona_internal_err!("Unexpected input to ST_GeoHash()"),
        };

        let (item_type, _) = parse_item_crs_arg_type(&arg_types[0])?;
        let crs_array = as_string_view_array(struct_array.column(1))?;

        // Rebuild the arguments against the unwrapped item so the geometry path
        // below is the same one the type-level kernel takes. A scalar argument
        // stays scalar, so that `WkbExecutor` still sizes the batch from
        // whichever argument is an array.
        let item_arg = match &args[0] {
            ColumnarValue::Array(_) => ColumnarValue::Array(struct_array.column(0).clone()),
            _ => ColumnarValue::Scalar(ScalarValue::try_from_array(struct_array.column(0), 0)?),
        };
        let mut item_arg_types = vec![item_type.clone()];
        let mut item_args = vec![item_arg];
        for (arg_type, arg) in zip(&arg_types[1..], &args[1..]) {
            item_arg_types.push(arg_type.clone());
            item_args.push(arg.clone());
        }

        let executor = WkbExecutor::new(&item_arg_types, &item_args);
        let mut builder = StringBuilder::with_capacity(
            executor.num_iterations(),
            MAX_PRECISION as usize * executor.num_iterations(),
        );
        let mut bounder = bounder_for_arg_type(&item_type, config_options)?;
        let mut raw_bounder = raw_bounder_for_arg_type(&item_type);
        let mut checker = CachedWgs84Check::with_capacity(crs_array.len());

        // A scalar struct carries a single CRS that applies to every iteration;
        // an array carries one per row.
        let scalar_wrap = match &args[0] {
            ColumnarValue::Array(_) => None,
            _ => Some(checker.wrap_for(crs_array.iter().next().flatten())?),
        };
        let mut crs_iter = crs_array.iter();
        let mut next_wrap = move || match scalar_wrap {
            Some(wrap) => Ok(wrap),
            // Every row is validated, including one whose geometry is null, so
            // that a column fails the same way regardless of where its nulls
            // happen to fall.
            None => checker.wrap_for(crs_iter.next().flatten()),
        };

        if item_args.len() > 1 {
            append_geohash_with_precision(
                &executor,
                &item_args,
                bounder.as_mut(),
                &mut raw_bounder,
                &mut next_wrap,
                &mut builder,
            )?;
        } else {
            append_point_geohash(
                &executor,
                bounder.as_mut(),
                &mut raw_bounder,
                &mut next_wrap,
                &mut builder,
            )?;
        }

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Whether an out-of-range longitude may be wrapped back into [-180, 180]
///
/// [`ensure_wgs84_crs`] has already rejected any CRS that is not WGS84, so this
/// only separates coordinates *declared* to be WGS84 from coordinates carrying
/// no CRS at all.
#[derive(Debug, Clone, Copy, PartialEq)]
enum LongitudeWrap {
    /// The argument declares a WGS84 CRS, or is a geography
    Enabled,
    /// The argument carries no CRS, so its units are assumed but not known
    Disabled,
}

/// Decide whether an argument's longitudes may be wrapped
///
/// Wrapping requires a *declared* WGS84 CRS (or a geography, which carries one
/// by construction). An absent CRS is accepted by [`ensure_wgs84_crs`] but does
/// not earn wrapping: accepting it is passive -- we cannot know, so we do not
/// break the query -- whereas wrapping it is active, and would invent a location
/// for coordinates whose provenance is unknown.
///
/// This is also what keeps faith with Apache Sedona. Spark has no CRS concept,
/// so every geometry migrated from it arrives here undeclared; wrapping those
/// would change the out-of-range result from Spark's null to a hash, which is
/// precisely the parity this function is built to preserve.
///
/// Item-level CRS resolves to `None` here for the reasons given on
/// [`ensure_wgs84_crs`], so it does not wrap either.
fn longitude_wrap_for_arg_type(arg_type: &SedonaType) -> LongitudeWrap {
    let edges = match arg_type {
        SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => *edges,
        // A literal NULL argument: every row is null, so this is never consulted.
        _ => return LongitudeWrap::Disabled,
    };

    if edges == Edges::Spherical || arg_type.crs() == &lnglat() {
        LongitudeWrap::Enabled
    } else {
        LongitudeWrap::Disabled
    }
}

/// Ensure a geohash argument's coordinates are WGS84 longitude/latitude
///
/// A geohash is defined against the WGS84 datum, so this is the only CRS whose
/// coordinates it can encode correctly. Anything else is rejected rather than
/// hashed: a projected coordinate is in metres, not degrees, and one that
/// happens to land inside [-180, 180] x [-90, 90] would otherwise produce a
/// confident geohash for an unrelated place. EPSG:3857 POINT (10 20) is ten
/// metres east and twenty metres north of the origin, in the Gulf of Guinea,
/// but reads as 10 degrees east and 20 degrees north -- in Chad.
///
/// `EPSG:4326` and `OGC:CRS84` are the same CRS here (see `authority_codes_equal`),
/// so both are accepted. Other geographic CRSes are *not*, even though their units
/// are also degrees: NAD83 sits one to two metres from WGS84 and NAD27 up to a
/// hundred, which is many cells wide at high precision. Reproject them with
/// `ST_Transform()`.
///
/// An absent CRS is accepted and assumed to be WGS84. This is what the function
/// already assumes of undeclared coordinates -- the [-180, 180] x [-90, 90]
/// domain check is only meaningful under that reading -- and rejecting it would
/// break every `ST_GeomFromText()` call, which carries no CRS.
///
/// Item-level CRS is *not* checked: [`ItemCrsKernel`] resolves the per-row CRS
/// outside this kernel and hands the inner kernel an item type whose CRS has
/// been stripped to `None` (`parse_item_crs_arg_type_strip_crs`), which is
/// indistinguishable here from a genuinely absent one. Validating it would mean
/// changing that kernel's contract for every function built on it.
fn ensure_wgs84_crs(arg_type: &SedonaType) -> Result<()> {
    let crs = arg_type.crs();

    if crs.is_none() || crs == &lnglat() {
        return Ok(());
    }

    let name = crs
        .as_ref()
        .map(|crs| crs.to_crs_string())
        .unwrap_or_else(|| "unknown".to_string());

    plan_err!(
        "ST_GeoHash() requires WGS84 longitude/latitude coordinates but the argument has CRS \
         '{name}'. Use ST_Transform() to reproject to EPSG:4326 first."
    )
}

/// Append the geohash of each geometry at the precision given by the second argument
fn append_geohash_with_precision(
    executor: &WkbExecutor<'_, '_>,
    args: &[ColumnarValue],
    bounder: &mut dyn WkbBounder2D,
    raw_bounder: &mut Option<Box<dyn WkbBounder2D>>,
    next_wrap: &mut dyn FnMut() -> Result<LongitudeWrap>,
    builder: &mut StringBuilder,
) -> Result<()> {
    let precision_value = args[1]
        .cast_to(&DataType::Int64, None)?
        .to_array(executor.num_iterations())?;
    let precision_array = as_int64_array(&precision_value)?;
    let mut precision_iter = precision_array.iter();

    executor.execute_wkb_void(|maybe_wkb| {
        let wrap = next_wrap()?;
        match (maybe_wkb, precision_iter.next().unwrap()) {
            (Some(wkb), Some(precision)) => {
                match invoke_scalar(wkb, precision, bounder, raw_bounder, wrap)? {
                    Some(geohash) => builder.append_value(geohash),
                    // Geometry was empty or outside the lon/lat bounds
                    None => builder.append_null(),
                }
            }
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
    raw_bounder: &mut Option<Box<dyn WkbBounder2D>>,
    next_wrap: &mut dyn FnMut() -> Result<LongitudeWrap>,
    builder: &mut StringBuilder,
) -> Result<()> {
    executor.execute_wkb_void(|maybe_wkb| {
        let wrap = next_wrap()?;
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

                match invoke_scalar(wkb, MAX_PRECISION, bounder, raw_bounder, wrap)? {
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
/// wrapped back into [-180, 180] rather than nulling the row, so a longitude of
/// 181 hashes as -179. Latitude is never wrapped; see [`normalize_longitude`]
/// and [`in_latitude_range`] for why the two axes are treated differently.
///
/// The bounding box comes from `bounder`, which the caller resolved from the
/// argument's edge type, so a geography is bounded on the sphere rather than
/// in the plane.
fn invoke_scalar(
    geom: &Wkb,
    precision: i64,
    bounder: &mut dyn WkbBounder2D,
    raw_bounder: &mut Option<Box<dyn WkbBounder2D>>,
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
    //
    // The check runs against the raw planar bounds where they are available,
    // because a spherical bounder clamps latitude into range before returning
    // it; see [`raw_bounder_for_arg_type`].
    let raw_y = match raw_bounder.as_mut() {
        Some(raw_bounder) => {
            raw_bounder.clear();
            raw_bounder
                .update_wkb_bytes(geom.buf())
                .map_err(|e| sedona_internal_datafusion_err!("Error computing bounds: {e}"))?;
            let (_, raw_y) = raw_bounder.finish();
            raw_y
        }
        None => y,
    };

    if !in_latitude_range(&raw_y) {
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
    fn normalize_longitude_leaves_in_range_intervals_alone() {
        // In-range intervals are untouched under either setting, including the
        // wraparound intervals a spherical bounder produces (lo > hi).
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
    fn normalize_longitude_wraps_out_of_range_intervals() {
        let out_of_range = WraparoundInterval::new(190.0, 190.0);
        assert_eq!(
            normalize_longitude(&out_of_range, LongitudeWrap::Enabled),
            Some(WraparoundInterval::new(-170.0, -170.0))
        );
        // Without a declared WGS84 CRS the row stays null.
        assert_eq!(
            normalize_longitude(&out_of_range, LongitudeWrap::Disabled),
            None
        );
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
    fn wgs84_crs_is_required(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let edges = match &sedona_type {
            SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => *edges,
            _ => unreachable!(),
        };

        // Both spellings of WGS84 are accepted...
        for code in ["EPSG:4326", "OGC:CRS84"] {
            let crs = deserialize_crs(code).unwrap();
            ensure_wgs84_crs(&SedonaType::Wkb(edges, crs))
                .unwrap_or_else(|e| panic!("{code} should be accepted: {e}"));
        }
        // ...as is lnglat() itself, and an absent CRS.
        ensure_wgs84_crs(&SedonaType::Wkb(edges, lnglat())).unwrap();
        ensure_wgs84_crs(&sedona_type).unwrap();
        // A geography carries lnglat by construction.
        ensure_wgs84_crs(&WKB_GEOGRAPHY).unwrap();

        // A projected CRS is rejected, and so is a non-WGS84 geographic one:
        // NAD83's units are degrees but its datum is not WGS84.
        for code in ["EPSG:3857", "EPSG:26918", "EPSG:4269"] {
            let crs = deserialize_crs(code).unwrap();
            let err = ensure_wgs84_crs(&SedonaType::Wkb(edges, crs))
                .expect_err("{code} should be rejected");
            let msg = err.to_string();
            assert!(msg.contains("requires WGS84"), "unexpected message: {msg}");
            assert!(
                msg.contains("ST_Transform"),
                "message should suggest a fix: {msg}"
            );
        }
    }

    #[test]
    fn udf_rejects_non_wgs84_crs_at_plan_time() {
        // The rejection happens while resolving the kernel, so it surfaces
        // before any row is read rather than partway through a scan.
        let tester = geohash_tester(projected_geometry());
        let err = tester
            .return_type()
            .expect_err("a projected CRS should be rejected");
        assert!(err.to_string().contains("requires WGS84"), "{err}");
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
    fn udf_does_not_wrap_longitude_for_an_absent_crs() {
        // Accepted, but not wrapped: an undeclared geometry keeps Apache
        // Sedona's null so that a Spark query migrating here keeps its results.
        // ST_SetSRID(geom, 4326) opts into wrapping.
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

    /// Invoke ST_GeoHash over an item_crs array with one CRS per row
    fn invoke_item_crs(
        wkts: &[Option<&str>],
        crses: &[Option<&str>],
        precision: i64,
    ) -> Result<Vec<Option<String>>> {
        use arrow_array::cast::as_string_array;
        use sedona_testing::create::create_array_item_crs;

        let tester = ScalarUdfTester::new(
            st_geohash_udf().into(),
            vec![
                WKB_GEOMETRY_ITEM_CRS.clone(),
                SedonaType::Arrow(DataType::Int64),
            ],
        );
        let array = create_array_item_crs(wkts, crses.iter().copied(), &WKB_GEOMETRY);
        let out = tester.invoke_array_scalar(array, ScalarValue::Int64(Some(precision)))?;
        let out = as_string_array(&out);
        Ok((0..Array::len(out))
            .map(|i| {
                if Array::is_null(out, i) {
                    None
                } else {
                    Some(arrow_array::array::StringArray::value(out, i).to_string())
                }
            })
            .collect())
    }

    #[test]
    fn udf_item_crs_rejects_non_wgs84_rows() {
        // The same rule the type level applies, detected when the row arrives
        // rather than when the query is planned. Before this, a per-row
        // EPSG:3857 was hashed as though it were degrees: POINT (10 20) in
        // metres is in the Gulf of Guinea, but hashed as degrees it lands in
        // Chad.
        for crs in ["EPSG:3857", "EPSG:26918", "EPSG:4269"] {
            let err = invoke_item_crs(&[Some("POINT (10 20)")], &[Some(crs)], 10)
                .expect_err("{crs} should be rejected");
            let msg = err.to_string();
            assert!(msg.contains("requires WGS84"), "unexpected message: {msg}");
            assert!(msg.contains(crs), "message should name the CRS: {msg}");
        }

        // One bad row is enough, wherever it sits in the batch.
        let err = invoke_item_crs(
            &[Some("POINT (10 20)"), Some("POINT (10 20)")],
            &[Some("EPSG:4326"), Some("EPSG:3857")],
            10,
        )
        .expect_err("a mixed batch should be rejected");
        assert!(err.to_string().contains("requires WGS84"), "{err}");
    }

    #[test]
    fn udf_item_crs_accepts_wgs84_and_absent_rows() {
        // Both spellings of WGS84, plus the "no CRS" spellings: a null entry
        // and the "0" sentinel that deserialize_crs() resolves to no CRS.
        let out = invoke_item_crs(
            &[
                Some("POINT (10 20)"),
                Some("POINT (10 20)"),
                Some("POINT (10 20)"),
                Some("POINT (10 20)"),
            ],
            &[Some("EPSG:4326"), Some("OGC:CRS84"), None, Some("0")],
            10,
        )
        .unwrap();
        assert_eq!(out, vec![Some("s5x1g8cu2y".to_string()); 4]);
    }

    #[test]
    fn udf_item_crs_wraps_longitude_only_for_declared_wgs84_rows() {
        // The inconsistency this closes: with the CRS visible per row, an
        // item-level WGS84 row wraps exactly as a type-level WGS84 column does,
        // instead of nulling where the other wraps. A row with no CRS keeps
        // Apache Sedona's null.
        let out = invoke_item_crs(
            &[Some("POINT (190 50)"), Some("POINT (190 50)")],
            &[Some("EPSG:4326"), None],
            12,
        )
        .unwrap();
        assert_eq!(out, vec![Some("b0zh7w1z0gs3".to_string()), None]);

        // Which is the same answer the type-level kernel gives for that point.
        let tester = geohash_tester(lnglat_geometry());
        let type_level = tester
            .invoke_scalar_scalar("POINT (190.0 50.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(type_level, "b0zh7w1z0gs3");
    }

    #[test]
    fn udf_item_crs_rejects_a_bad_crs_on_a_null_geometry_row() {
        // A row is validated even when its geometry is null, so that a column
        // fails the same way regardless of where its nulls happen to fall.
        let err = invoke_item_crs(&[None], &[Some("EPSG:3857")], 10)
            .expect_err("a null geometry should not excuse a bad CRS");
        assert!(err.to_string().contains("requires WGS84"), "{err}");
    }

    #[test]
    fn udf_item_crs_caches_each_distinct_crs_once() {
        let mut checker = CachedWgs84Check::default();
        assert_eq!(
            checker.wrap_for(Some("EPSG:4326")).unwrap(),
            LongitudeWrap::Enabled
        );
        assert_eq!(checker.wrap_for(None).unwrap(), LongitudeWrap::Disabled);
        assert!(checker.wrap_for(Some("EPSG:3857")).is_err());

        // Repeats are served from the cache, including the rejection.
        assert_eq!(
            checker.wrap_for(Some("EPSG:4326")).unwrap(),
            LongitudeWrap::Enabled
        );
        assert!(checker.wrap_for(Some("EPSG:3857")).is_err());
        assert_eq!(checker.cache.len(), 2);
    }

    /// A stand-in for the s2geography bounder that reproduces its latitude clamp
    ///
    /// s2geography bounds into an `S2LatLngRect`, whose latitudes are
    /// constrained to [-90, 90], so an out-of-range latitude never reaches the
    /// domain check. sedona-functions cannot depend on sedona-s2geography, so
    /// this wrapper reproduces just that behavior: the real bounder is exercised
    /// by the Python tests, which run against an s2geography build in CI.
    #[derive(Debug, Default)]
    struct ClampingBounder {
        inner: WkbGeometryBounder,
    }

    impl WkbBounder2D for ClampingBounder {
        fn clear(&mut self) {
            self.inner.clear()
        }

        fn update_bounds(
            &mut self,
            x: WraparoundInterval,
            y: Interval,
        ) -> std::result::Result<(), sedona_geometry::error::SedonaGeometryError> {
            self.inner.update_bounds(x, y)
        }

        fn update_wkb_bytes(
            &mut self,
            wkb_value: &[u8],
        ) -> std::result::Result<(), sedona_geometry::error::SedonaGeometryError> {
            self.inner.update_wkb_bytes(wkb_value)
        }

        fn finish(&self) -> (WraparoundInterval, Interval) {
            let (x, y) = self.inner.finish();
            if y.is_empty() {
                return (x, y);
            }
            // The clamp S2 applies.
            (
                x,
                Interval::new(y.lo().clamp(-90.0, 90.0), y.hi().clamp(-90.0, 90.0)),
            )
        }

        fn expand_by_distance(
            &mut self,
            distance: f64,
            radius: Option<f64>,
        ) -> std::result::Result<(), sedona_geometry::error::SedonaGeometryError> {
            self.inner.expand_by_distance(distance, radius)
        }

        fn mem_used(&self) -> usize {
            self.inner.mem_used()
        }

        fn create_instance(&self) -> Box<dyn WkbBounder2D> {
            Box::new(Self::default())
        }
    }

    #[test]
    fn udf_geography_rejects_latitude_the_bounder_clamps() {
        // With a bounder that clamps like S2, an out-of-range latitude would
        // otherwise be invisible: it arrives as exactly 90 and encodes as the
        // pole. The raw planar bound taken alongside it recovers the input.
        let mut tester = ScalarUdfTester::new(
            st_geohash_udf().into(),
            vec![WKB_GEOGRAPHY, SedonaType::Arrow(DataType::Int64)],
        );
        let options = tester.sedona_options_mut();
        options.runtime = options
            .runtime
            .with_bounder(Edges::Spherical, Arc::new(ClampingBounder::default()))
            .unwrap();

        for wkt in [
            "POINT (50.0 100.0)",
            "POINT (50.0 91.0)",
            "POINT (50.0 -91.0)",
            "LINESTRING (50.0 80.0, 60.0 100.0)",
        ] {
            let result = tester
                .invoke_scalar_scalar(wkt, ScalarValue::Int64(Some(12)))
                .unwrap();
            tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));
        }

        // The genuine poles are in range and still encode.
        let pole = tester
            .invoke_scalar_scalar("POINT (50.0 90.0)", ScalarValue::Int64(Some(12)))
            .unwrap();
        tester.assert_scalar_result_equals(pole, "vpgxczbzuryp");
    }

    #[test]
    fn raw_bounder_is_only_resolved_for_spherical_arguments() {
        // A planar bounder does no clamping, so a geometry needs no second pass.
        assert!(raw_bounder_for_arg_type(&WKB_GEOMETRY).is_none());
        assert!(raw_bounder_for_arg_type(&WKB_VIEW_GEOMETRY).is_none());
        assert!(raw_bounder_for_arg_type(&WKB_GEOGRAPHY).is_some());
        assert!(raw_bounder_for_arg_type(&WKB_VIEW_GEOGRAPHY).is_some());
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
