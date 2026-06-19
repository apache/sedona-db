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
use datafusion_common::{
    exec_err, plan_datafusion_err, plan_err, DataFusionError, HashMap, Result,
};
use lru::LruCache;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::num::NonZeroUsize;
use std::str::FromStr;
use std::sync::Arc;

use serde_json::Value;

thread_local! {
    /// Thread-local LRU cache for CRS deserialization (one cache per thread)
    /// Keeping the cache small to avoid excessive memory usage in multi-threaded contexts
    /// with json values.
    static CRS_CACHE: RefCell<LruCache<String, Crs>> =
        // We should consider making the size of the cache configurable
        // given the memory vs cpu performance trade-offs.
        RefCell::new(LruCache::new(NonZeroUsize::new(50).unwrap()));
}

/// Deserialize a specific GeoArrow "crs" value
///
/// Recognizes three encodings: an `authority:code` string (e.g. `"EPSG:4326"`),
/// a PROJJSON object, and a WKT1/WKT2 CRS string (e.g. `PROJCS[...]` /
/// `PROJCRS[...]`). WKT is carried verbatim (so it can be handed to PROJ for
/// transforms) with a best-effort `authority:code` extracted from its trailing
/// `AUTHORITY[...]` / `ID[...]` tag. The `OGC:CRS84` / `EPSG:4326` lon/lat
/// special case is recognized for the Geography type.
pub fn deserialize_crs(crs_str: &str) -> Result<Crs> {
    if crs_str.is_empty() {
        return Ok(None);
    }

    // Check cache first
    let cached_result = CRS_CACHE.with(|cache| cache.borrow().peek(crs_str).cloned());

    if let Some(cached) = cached_result {
        return Ok(cached);
    }

    // Handle "0", "{AUTH}:{CODE}" authority codes (including the OGC:CRS84 /
    // EPSG:4326 lon/lat aliases, which are kept verbatim rather than folded),
    // a WKT1/WKT2 CRS string, then PROJJSON.
    let crs = if crs_str == "0" {
        None
    } else if AuthorityCode::is_authority_code(crs_str) {
        AuthorityCode::crs(crs_str)
    } else if looks_like_wkt(crs_str) {
        Some(Arc::new(WktCrs::new(crs_str.to_string()))
            as Arc<dyn CoordinateReferenceSystem + Send + Sync>)
    } else {
        // Try to parse as PROJJSON string
        let projjson: ProjJSON = crs_str.parse()?;
        Some(Arc::new(projjson) as Arc<dyn CoordinateReferenceSystem + Send + Sync>)
    };

    // Cache result
    CRS_CACHE.with(|cache| {
        cache.borrow_mut().put(crs_str.to_string(), crs.clone());
    });

    Ok(crs)
}

/// Deserialize a CRS from a serde_json::Value object
pub fn deserialize_crs_from_obj(crs_value: &serde_json::Value) -> Result<Crs> {
    if crs_value.is_null() {
        return Ok(None);
    }

    if let Some(crs_str) = crs_value.as_str() {
        if crs_str.is_empty() || crs_str == "0" {
            return Ok(None);
        }

        // Authority codes (including the OGC:CRS84 / EPSG:4326 lon/lat aliases)
        // are kept verbatim rather than folded.
        if AuthorityCode::is_authority_code(crs_str) {
            return Ok(AuthorityCode::crs(crs_str));
        }
    }

    let projjson = if let Some(string) = crs_value.as_str() {
        // Handle the geopandas bug where it exported stringified projjson.
        // This could in the future handle other stringified versions with some
        // auto detection logic.
        string.parse()?
    } else {
        // Handle the case where Value is already an object
        ProjJSON::try_new(crs_value.clone())?
    };

    Ok(Some(Arc::new(projjson)))
}

/// Translating CRS into integer SRID with a cache to avoid expensive CRS deserialization.
#[derive(Default)]
pub struct CachedCrsToSRIDMapping {
    cache: HashMap<String, u32>,
}

impl CachedCrsToSRIDMapping {
    /// Create a new CachedCrsToSRIDMapping with an empty cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Create a new CachedCrsToSRIDMapping with an optional initial capacity for the cache.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
        }
    }

    /// Get the SRID for a given CRS string, using the cache to avoid expensive deserialization where possible.
    /// Returns 0 for missing CRS or CRS that don't have an SRID. Errors if the CRS string is invalid or if the
    /// CRS can't be deserialized.
    pub fn get_srid(&mut self, maybe_crs: Option<&str>) -> Result<u32> {
        if let Some(crs_str) = maybe_crs {
            if let Some(srid) = self.cache.get(crs_str) {
                Ok(*srid)
            } else if let Some(crs) = deserialize_crs(crs_str)? {
                if let Some(srid) = crs.srid()? {
                    self.cache.insert(crs_str.to_string(), srid);
                    Ok(srid)
                } else {
                    exec_err!("Can't extract SRID from item-level CRS '{crs_str}'")
                }
            } else {
                Ok(0)
            }
        } else {
            Ok(0)
        }
    }
}

/// Cache for converting integer SRIDs to CRS strings.
///
/// Maps SRID integers to their CRS string representation with caching to avoid
/// repeated validation/deserialization of the same SRID:
/// - `0` → `None` (no CRS)
/// - `4326` → `Some("OGC:CRS84")`
/// - other → `Some("EPSG:{srid}")`, validated once via the caller-provided closure
#[derive(Default)]
pub struct CachedSRIDToCrs {
    cache: HashMap<i64, Option<String>>,
}

impl CachedSRIDToCrs {
    /// Create a new CachedSRIDToCrs with an empty cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Create a new CachedSRIDToCrs with a pre-allocated cache.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
        }
    }

    /// Get the CRS string for a given SRID, using the cache to avoid repeated validation.
    pub fn get_crs(&mut self, srid: i64) -> Result<Option<String>> {
        if let Some(cached) = self.cache.get(&srid) {
            return Ok(cached.clone());
        }

        let result = if srid == 0 {
            None
        } else if srid == 4326 {
            Some("OGC:CRS84".to_string())
        } else {
            let auth_code = format!("EPSG:{srid}");
            Some(auth_code)
        };

        self.cache.insert(srid, result.clone());
        Ok(result)
    }
}

/// Normalize a CRS string to the round-trippable definition stored on
/// geometry/raster CRSes.
///
/// Returns the [`CoordinateReferenceSystem::to_crs_string`] form — an
/// `authority:code` stays compact, while a PROJJSON/WKT definition is preserved
/// in full rather than collapsed to its embedded authority code, so no
/// information is lost on the way in. Returns `None` for `"0"`, `""`, or CRS
/// strings that deserialize to `None`.
///
/// Parsing is memoized by [`deserialize_crs`]'s thread-local cache, so this is
/// cheap to call per row. The "full definition" guarantee is semantic, not
/// byte-for-byte: a PROJJSON re-serializes from its parsed tree, so object keys
/// may reorder and whitespace normalize; it round-trips and compares equal.
pub fn normalize_crs(crs_str: &str) -> Result<Option<String>> {
    Ok(deserialize_crs(crs_str)?.map(|crs| crs.to_crs_string()))
}

/// Longitude/latitude CRS (WGS84)
///
/// A [`Crs`] that matches EPSG:4326 or OGC:CRS84.
pub fn lnglat() -> Crs {
    LngLat::crs()
}

/// Coordinate reference systems
///
/// From Sedona's perspective, we can have a missing Crs at the type level or
/// something that can resolve to a JSON value (which is what we need to export it
/// to GeoArrow), something that can resolve to a PROJJSON value (which is what we need
/// to export it to Parquet/Iceberg) and something with which we can check
/// equality (for binary operators).
pub type Crs = Option<Arc<dyn CoordinateReferenceSystem + Send + Sync>>;

/// Borrowed form of [`Crs`] — a reference to a CRS trait object.
///
/// Returned by `Crs::as_deref()` and used in function signatures that only
/// need to inspect a CRS without taking ownership.
pub type CrsRef<'a> = Option<&'a (dyn CoordinateReferenceSystem + Send + Sync)>;

impl Display for dyn CoordinateReferenceSystem + Send + Sync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Ok(Some(auth_code)) = self.to_authority_code() {
            write!(f, "{}", auth_code.to_lowercase())
        } else {
            // We can probably try harder to get compact output out of more
            // types of CRSes
            write!(f, "{{...}}")
        }
    }
}

impl PartialEq<dyn CoordinateReferenceSystem + Send + Sync>
    for dyn CoordinateReferenceSystem + Send + Sync
{
    fn eq(&self, other: &Self) -> bool {
        // `crs_equals` is the single source of truth for CRS equality (including
        // the lon/lat alias); the operator is a pure delegate.
        self.crs_equals(other)
    }
}

/// Coordinate reference system abstraction
///
/// A trait defining the minimum required properties of a concrete coordinate
/// reference system, allowing the details of this to be implemented elsewhere.
pub trait CoordinateReferenceSystem: Debug {
    /// Compute the representation of this Crs in the form required for JSON output
    ///
    /// The output must be valid JSON (e.g., arbitrary strings must be quoted).
    fn to_json(&self) -> String;

    /// Compute the representation of this Crs as a string in the form Authority:Code
    ///
    /// If there is no such representation, returns None.
    fn to_authority_code(&self) -> Result<Option<String>>;

    /// Compute CRS equality
    ///
    /// CRS equality is a relatively thorny topic and can be difficult to compute;
    /// however, this method should try to compare self and other on value (e.g.,
    /// comparing authority_code where possible).
    fn crs_equals(&self, other: &dyn CoordinateReferenceSystem) -> bool;

    /// Convert this CRS representation to an integer SRID if possible.
    ///
    /// For the purposes of this trait, an SRID is always equivalent to the
    /// authority_code `"EPSG:{srid}"`. Note that other SRID representations
    /// (e.g., GeoArrow, Parquet GEOMETRY/GEOGRAPHY) do not make any guarantees
    /// that an SRID comes from the EPSG authority.
    ///
    /// The default derives the SRID from [`to_authority_code`](Self::to_authority_code):
    /// the lon/lat alias maps to 4326, an `EPSG:{code}` authority parses to its
    /// code, and anything else is `None`. Implementers only need to override
    /// this if they can supply an SRID without an authority code.
    fn srid(&self) -> Result<Option<u32>> {
        let Some(auth_code) = self.to_authority_code()? else {
            return Ok(None);
        };
        if LngLat::is_authority_code_lnglat(&auth_code) {
            return Ok(LngLat::srid());
        }
        match auth_code.split_once(':') {
            Some((authority, code)) if authority.eq_ignore_ascii_case("EPSG") => {
                Ok(code.parse().ok())
            }
            _ => Ok(None),
        }
    }

    /// Compute a CRS string representation
    ///
    /// Unlike `to_json()`, arbitrary string values returned by this method should
    /// not be escaped. This is the representation expected as input to PROJ, GDAL,
    /// and Parquet GEOMETRY/GEOGRAPHY representations of CRS.
    fn to_crs_string(&self) -> String;

    /// For Geographic CRSes, return the geographic parameters
    ///
    /// For non-geographic CRSes, this must return None. For the purposes of this
    /// trait, a geographic CRS must be suitable for the Geography type (i.e.,
    /// units in degrees).
    fn geographic_params(&self) -> Result<Option<GeographicCrsParams>>;
}

/// Parameters describing a geographic CRS
///
/// These parameters are used for distance calculations on the ellipsoid or sphere
/// for functions that support it. Notably, this is used for calculations involving
/// distance in the geography type. Currently this only supports a single parameter
/// (average spherical radius) but can be expanded to support ellipsoidal parameters
/// when we have functions that support it.
pub struct GeographicCrsParams {
    spherical_radius_m: f64,
}

impl GeographicCrsParams {
    /// The average spherical radius in meters for use in spherical geographies
    ///
    /// Because geographies with spherical edges approximate distance on the sphere,
    /// this is the value that will be used to calculate distance in such a case.
    pub fn spherical_radius(&self) -> f64 {
        self.spherical_radius_m
    }
}

/// Concrete implementation of a default longitude/latitude coordinate reference system
#[derive(Debug)]
struct LngLat {}

impl LngLat {
    pub fn crs() -> Crs {
        Crs::Some(Arc::new(AuthorityCode {
            auth_code: "OGC:CRS84".to_string(),
        }))
    }

    pub fn is_authority_code_lnglat(string_value: &str) -> bool {
        string_value == "OGC:CRS84" || string_value == "EPSG:4326"
    }

    pub fn srid() -> Option<u32> {
        Some(4326)
    }
}

/// Implementation of an authority:code CoordinateReferenceSystem
#[derive(Debug)]
struct AuthorityCode {
    auth_code: String,
}

/// Implementation of an authority:code
impl AuthorityCode {
    /// Create a Crs from an authority:code string
    /// Example: "EPSG:4269"
    pub fn crs(auth_code: &str) -> Crs {
        let ac = if Self::validate_epsg_code(auth_code) {
            format!("EPSG:{auth_code}")
        } else {
            auth_code.to_uppercase()
        };
        Some(Arc::new(AuthorityCode { auth_code: ac }))
    }

    /// Check if a Value is an authority:code string
    /// Note: this can be expanded to more types in the future
    pub fn is_authority_code(auth_code: &str) -> bool {
        // Expecting <authority>:<code> format
        if let Some(colon_pos) = auth_code.find(':') {
            let authority = &auth_code[..colon_pos];
            let code = &auth_code[colon_pos + 1..];
            Self::validate_authority(authority) && Self::validate_code(code)
        } else {
            // No colon, check if it's a valid EPSG code
            Self::validate_epsg_code(auth_code)
        }
    }

    /// Validate the authority part of an authority:code string
    /// Note: this can be expanded in the future to support more authorities
    fn validate_authority(authority: &str) -> bool {
        authority.chars().all(|c| c.is_alphanumeric())
    }

    /// Validate that a code is alphanumeric for general authority codes.
    /// Note: EPSG codes specifically require numeric validation, which is handled by `validate_epsg_code`.
    fn validate_code(code: &str) -> bool {
        code.chars().all(|c| c.is_alphanumeric())
    }

    /// Validate that a code is likely to be an EPSG code (all numbers and not too long)
    ///
    /// The maximum length of 9 characters is chosen because, as of 2024, all official EPSG codes
    /// are positive integers with at most 7 digits (e.g., "4326", "3857"), and this limit provides
    /// a small buffer for potential future codes or extensions while preventing unreasonably long inputs.
    fn validate_epsg_code(code: &str) -> bool {
        code.chars().all(|c| c.is_numeric()) && code.len() <= 9
    }
}

/// Compare two authority codes, treating the lon/lat aliases
/// (`EPSG:4326` and `OGC:CRS84`) as equal. This keeps the aliases equivalent
/// for `crs_equals` now that `deserialize_crs` preserves them verbatim instead
/// of folding `EPSG:4326` to `OGC:CRS84`.
fn authority_codes_equal(a: &str, b: &str) -> bool {
    a == b || (LngLat::is_authority_code_lnglat(a) && LngLat::is_authority_code_lnglat(b))
}

/// Implementation of an authority:code CoordinateReferenceSystem
impl CoordinateReferenceSystem for AuthorityCode {
    /// Convert to a JSON string
    ///
    /// The lon/lat aliases (`EPSG:4326`/`OGC:CRS84`) canonicalize to `OGC:CRS84`
    /// here so CRS metadata stays axis-order-explicit for GeoParquet/GeoArrow,
    /// even though `to_crs_string` preserves the authority code verbatim.
    fn to_json(&self) -> String {
        let code = if LngLat::is_authority_code_lnglat(&self.auth_code) {
            "OGC:CRS84"
        } else {
            &self.auth_code
        };
        let mut result = String::with_capacity(code.len() + 2);
        result.push('"');
        result.push_str(code);
        result.push('"');
        result
    }

    /// Convert to an authority code string
    fn to_authority_code(&self) -> Result<Option<String>> {
        Ok(Some(self.to_crs_string()))
    }

    /// Check equality with another CoordinateReferenceSystem
    fn crs_equals(&self, other: &dyn CoordinateReferenceSystem) -> bool {
        match other.to_authority_code() {
            Ok(Some(other_ac)) => authority_codes_equal(&other_ac, &self.auth_code),
            _ => false,
        }
    }

    /// Convert to a CRS string
    fn to_crs_string(&self) -> String {
        self.auth_code.clone()
    }

    /// Hard-code support for several known spherical CRSes so we can support them
    /// in the geography type
    fn geographic_params(&self) -> Result<Option<GeographicCrsParams>> {
        Ok(geographic_params_for_authority(&self.auth_code))
    }
}

/// Spherical parameters for the handful of geographic CRSes the Geography type
/// supports, keyed by `authority:code`. Shared by [`AuthorityCode`] and [`WktCrs`]
/// (which resolves its authority code from the WKT first).
fn geographic_params_for_authority(auth_code: &str) -> Option<GeographicCrsParams> {
    match auth_code {
        // Default lnglat(). Here we use S2Earth::RadiusMeters() as a constant
        // for consistent results (if we averaged over the ensemble, distance results
        // may be subtly different between versions as the ensemble is updated).
        "OGC:CRS84" | "EPSG:4326" => Some(GeographicCrsParams {
            spherical_radius_m: 6371010.0,
        }),
        // NAD83
        // https://spatialreference.org/ref/epsg/4269/
        "OGC:CRS83" | "EPSG:4269" => {
            const A: f64 = 6378137.0;
            const INV_F: f64 = 298.257222101;
            const B: f64 = A * (1.0 - 1.0 / INV_F);
            Some(GeographicCrsParams {
                spherical_radius_m: (2.0 * A + B) / 3.0,
            })
        }
        // NAD27
        // Mean radius = (2a + b) / 3
        // https://spatialreference.org/ref/epsg/4267/
        "OGC:CRS27" | "EPSG:4267" => {
            const A: f64 = 6378206.4;
            const B: f64 = 6356583.8;
            Some(GeographicCrsParams {
                spherical_radius_m: (2.0 * A + B) / 3.0,
            })
        }
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProjJSON {
    value: Value,
}

impl FromStr for ProjJSON {
    type Err = DataFusionError;

    fn from_str(s: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(s)
            .map_err(|err| plan_datafusion_err!("Error deserializing PROJJSON Crs: {err}"))?;

        Self::try_new(value)
    }
}

impl ProjJSON {
    pub fn try_new(value: Value) -> Result<Self> {
        if !value.is_object() {
            return plan_err!("Can't create PROJJSON from non-object: {value}");
        }

        Ok(Self { value })
    }
}

impl CoordinateReferenceSystem for ProjJSON {
    fn to_json(&self) -> String {
        self.value.to_string()
    }

    fn to_authority_code(&self) -> Result<Option<String>> {
        if let Some(identifier) = self.value.get("id") {
            let maybe_authority = identifier.get("authority").map(|v| v.as_str());
            let maybe_code = identifier.get("code").map(|v| {
                if let Some(string) = v.as_str() {
                    Some(string.to_string())
                } else {
                    v.as_number().map(|number| number.to_string())
                }
            });
            if let (Some(Some(authority)), Some(Some(code))) = (maybe_authority, maybe_code) {
                return Ok(Some(format!("{authority}:{code}")));
            }
        }

        Ok(None)
    }

    fn crs_equals(&self, other: &dyn CoordinateReferenceSystem) -> bool {
        if let (Ok(Some(auth_code)), Ok(Some(other_auth_code))) =
            (self.to_authority_code(), other.to_authority_code())
        {
            authority_codes_equal(&auth_code, &other_auth_code)
        } else if let Ok(other_value) = serde_json::from_str::<Value>(&other.to_json()) {
            self.value == other_value
        } else {
            false
        }
    }

    fn to_crs_string(&self) -> String {
        self.to_json()
    }

    fn geographic_params(&self) -> Result<Option<GeographicCrsParams>> {
        let Value::Object(obj) = &self.value else {
            return Ok(None);
        };

        let Some(Value::String(crs_type)) = obj.get("type") else {
            return Ok(None);
        };

        // Only geographic CRSes have geographic params
        if crs_type != "GeographicCRS" {
            return Ok(None);
        }

        // Try datum first, then datum_ensemble
        let ellipsoid = if let Some(Value::Object(datum)) = obj.get("datum") {
            datum.get("ellipsoid")
        } else if let Some(Value::Object(datum_ensemble)) = obj.get("datum_ensemble") {
            datum_ensemble.get("ellipsoid")
        } else {
            return exec_err!(
                "PROJJSON GeographicCRS missing or malformed datum or datum_ensemble ellipsoid"
            );
        };

        let Some(Value::Object(ellipsoid)) = ellipsoid else {
            return exec_err!("PROJJSON ellipsoid is not an object");
        };

        // Handle spherical case (radius only)
        if let Some(Value::Number(radius)) = ellipsoid.get("radius") {
            let Some(r) = radius.as_f64() else {
                return exec_err!("PROJJSON radius can't be converted to f64");
            };
            return Ok(Some(GeographicCrsParams {
                spherical_radius_m: r,
            }));
        }

        // Get semi_major_axis (required for ellipsoidal case)
        let Some(Value::Number(semi_major)) = ellipsoid.get("semi_major_axis") else {
            return exec_err!("PROJJSON ellipsoid missing or malformed semi_major_axis");
        };
        let Some(a) = semi_major.as_f64() else {
            return exec_err!("PROJJSON semi_major_axis can't be converted to f64");
        };

        // Calculate semi_minor_axis from inverse_flattening or use directly
        let b = if let Some(Value::Number(inv_f)) = ellipsoid.get("inverse_flattening") {
            if let Some(inv_f) = inv_f.as_f64() {
                a * (1.0 - 1.0 / inv_f)
            } else {
                return exec_err!("PROJJSON inverse_flattening can't be converted to f64");
            }
        } else if let Some(Value::Number(semi_minor)) = ellipsoid.get("semi_minor_axis") {
            if let Some(b) = semi_minor.as_f64() {
                b
            } else {
                return exec_err!("PROJJSON semi_minor_axis can't be converted to f64");
            }
        } else {
            return exec_err!(
                "PROJJSON ellipsoid missing or malformed inverse_flattening/semi_minor_axis"
            );
        };

        // Mean radius = (2a + b) / 3
        Ok(Some(GeographicCrsParams {
            spherical_radius_m: (2.0 * a + b) / 3.0,
        }))
    }
}

/// WKT CRS keywords that begin a recognized WKT1/WKT2 coordinate reference
/// system, matched case-insensitively against the trimmed input.
const WKT_CRS_KEYWORDS: &[&str] = &[
    "PROJCRS",
    "PROJCS",
    "GEOGCRS",
    "GEOGCS",
    "GEODCRS",
    "GEODETICCRS",
    "GEOCCS",
    "BOUNDCRS",
    "COMPOUNDCRS",
    "COMPD_CS",
    "VERTCRS",
    "VERT_CS",
    "ENGCRS",
    "ENGINEERINGCRS",
    "PARAMETRICCRS",
    "TIMECRS",
    "LOCAL_CS",
    "FITTED_CS",
];

/// Heuristic check for a WKT1/WKT2 CRS string: the trimmed value begins with a
/// recognized CRS keyword followed (after optional whitespace) by `[` or `(`.
fn looks_like_wkt(s: &str) -> bool {
    let upper = s.trim_start().to_ascii_uppercase();
    WKT_CRS_KEYWORDS.iter().any(|kw| {
        upper
            .strip_prefix(kw)
            .is_some_and(|rest| rest.trim_start().starts_with(['[', '(']))
    })
}

/// A node in a WKT CRS string. WKT is JSON-ish: a keyword followed by a
/// bracketed, comma-separated list of children, interspersed with quoted
/// strings and bare tokens (numbers, axis directions, enum values) as leaves.
enum WktNode<'a> {
    /// A bare token — a number or unquoted keyword value, e.g. `6378137`.
    Unquoted(&'a str),
    /// A quoted string, e.g. `"WGS 84"`.
    Quoted(&'a str),
    /// `KEYWORD[child, child, ...]` (or `(...)`), e.g. `ID["EPSG",4326]`.
    Tagged(&'a str, Vec<WktNode<'a>>),
}

impl WktNode<'_> {
    /// The inner text of a leaf node (quoted string or bare token), or `None`
    /// for a tagged node.
    fn leaf_str(&self) -> Option<&str> {
        match self {
            WktNode::Quoted(s) | WktNode::Unquoted(s) => Some(s),
            WktNode::Tagged(..) => None,
        }
    }
}

/// Maximum WKT nesting depth the parser will descend into. Real CRS definitions
/// nest under a dozen levels deep; the cap exists only to bound stack usage on
/// pathological or malformed input (the WKT can arrive from untrusted file
/// metadata), past which authority extraction gives up.
const WKT_MAX_DEPTH: usize = 128;

/// Recursive-descent parser over a WKT string. Whitespace is insignificant
/// except inside quoted strings. Recursion is bounded by [`WKT_MAX_DEPTH`].
struct WktParser<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> WktParser<'a> {
    fn skip_ws(&mut self) {
        self.pos += self.s[self.pos..]
            .find(|c: char| !c.is_whitespace())
            .unwrap_or(self.s.len() - self.pos);
    }

    fn peek(&self) -> Option<char> {
        self.s[self.pos..].chars().next()
    }

    /// Parse a quoted string, consuming the surrounding quotes. A doubled `""`
    /// is the WKT escape for a literal quote and does not terminate the string.
    fn parse_quoted(&mut self) -> Result<WktNode<'a>, DataFusionError> {
        self.pos += 1; // opening quote
        let start = self.pos;
        loop {
            let rest = &self.s[self.pos..];
            let Some(q) = rest.find('"') else {
                return Err(DataFusionError::Execution(
                    "WKT CRS string has an unterminated quoted string".into(),
                ));
            };
            // A doubled "" is an escaped quote: skip both and keep scanning.
            if rest[q + 1..].starts_with('"') {
                self.pos += q + 2;
                continue;
            }
            let value = &self.s[start..self.pos + q];
            self.pos += q + 1; // closing quote
            return Ok(WktNode::Quoted(value));
        }
    }

    fn parse_node(&mut self, depth: usize) -> Result<WktNode<'a>, DataFusionError> {
        if depth > WKT_MAX_DEPTH {
            return Err(DataFusionError::Execution(format!(
                "WKT CRS string nests beyond {WKT_MAX_DEPTH} levels"
            )));
        }
        self.skip_ws();
        match self.peek() {
            Some('"') => return self.parse_quoted(),
            Some(_) => {}
            None => {
                return Err(DataFusionError::Execution(
                    "unexpected end of WKT CRS string".into(),
                ))
            }
        }
        // A keyword or bare token runs until a structural character.
        let start = self.pos;
        let word_len = self.s[self.pos..]
            .find(|c: char| c.is_whitespace() || "[](),\"".contains(c))
            .unwrap_or(self.s.len() - self.pos);
        let word = self.s[start..start + word_len].trim();
        self.pos += word_len;
        self.skip_ws();

        // A `[`/`(` makes this a tagged node; otherwise it is a bare token.
        match self.peek() {
            Some(open @ ('[' | '(')) => {
                let close = if open == '[' { ']' } else { ')' };
                self.pos += 1;
                let mut children = Vec::new();
                loop {
                    self.skip_ws();
                    match self.peek() {
                        Some(c) if c == close => {
                            self.pos += 1;
                            break;
                        }
                        Some(',') => {
                            self.pos += 1;
                        }
                        Some(_) => children.push(self.parse_node(depth + 1)?),
                        None => {
                            return Err(DataFusionError::Execution(
                                "WKT CRS string has an unterminated bracket".into(),
                            ))
                        }
                    }
                }
                Ok(WktNode::Tagged(word, children))
            }
            _ if word.is_empty() => Err(DataFusionError::Execution(
                "WKT CRS string has an empty token".into(),
            )),
            _ => Ok(WktNode::Unquoted(word)),
        }
    }
}

/// Best-effort `authority:code` from a WKT CRS string, read from the **CRS-level**
/// `AUTHORITY["auth","code"]` (WKT1) or `ID["auth",code]` (WKT2) tag — i.e. a
/// direct child of the root node. Authorities on nested datums, ellipsoids,
/// projection methods, parameters, or units are deliberately ignored: a custom
/// CRS with no top-level authority must not be misidentified by an inner code
/// (e.g. a Lambert Conformal whose only `ID`s are the `EPSG:9001` metre unit and
/// projection operation codes). `None` when the root carries no authority tag.
fn extract_authority_from_wkt(wkt: &str) -> Option<String> {
    let mut parser = WktParser { s: wkt, pos: 0 };
    // A malformed or too-deeply-nested WKT simply yields no authority — the
    // definition is still carried verbatim, it just has no SRID.
    let Ok(WktNode::Tagged(_, children)) = parser.parse_node(0) else {
        return None;
    };
    for child in &children {
        let WktNode::Tagged(keyword, args) = child else {
            continue;
        };
        if !keyword.eq_ignore_ascii_case("AUTHORITY") && !keyword.eq_ignore_ascii_case("ID") {
            continue;
        }
        let authority = args.first()?.leaf_str()?.trim();
        let code = args.get(1)?.leaf_str()?.trim();
        if authority.is_empty() || code.is_empty() {
            return None;
        }
        return Some(format!("{}:{}", authority.to_uppercase(), code));
    }
    None
}

/// True if `keyword` is a geographic/geodetic CRS root — one with a meaningful
/// ellipsoid for spherical geography. Projected, engineering, vertical, etc.
/// roots are excluded so they never resolve geographic parameters.
fn is_geographic_crs_keyword(keyword: &str) -> bool {
    ["GEOGCS", "GEOGCRS", "GEODCRS", "GEODETICCRS"]
        .iter()
        .any(|k| keyword.eq_ignore_ascii_case(k))
}

/// Depth-first search for the first `ELLIPSOID` (WKT2) or `SPHEROID` (WKT1) node.
fn find_ellipsoid<'a, 'n>(node: &'n WktNode<'a>) -> Option<&'n WktNode<'a>> {
    let WktNode::Tagged(keyword, children) = node else {
        return None;
    };
    if keyword.eq_ignore_ascii_case("ELLIPSOID") || keyword.eq_ignore_ascii_case("SPHEROID") {
        return Some(node);
    }
    children.iter().find_map(find_ellipsoid)
}

/// Best-effort geographic parameters parsed straight from a WKT definition, for
/// geographic CRSes whose authority code isn't in the curated lookup (including
/// authority-less ones). Reads the ellipsoid's semi-major axis and inverse
/// flattening and reduces them to a mean spherical radius the same way
/// [`geographic_params_for_authority`] does; a sphere (inverse flattening 0)
/// uses its radius directly. Returns `None` for non-geographic CRSes (e.g. a
/// projected CRS, even though it nests a geographic base) or WKT without a
/// parseable ellipsoid.
fn geographic_params_from_wkt(wkt: &str) -> Option<GeographicCrsParams> {
    let root = WktParser { s: wkt, pos: 0 }.parse_node(0).ok()?;
    let WktNode::Tagged(keyword, _) = &root else {
        return None;
    };
    if !is_geographic_crs_keyword(keyword) {
        return None;
    }

    let WktNode::Tagged(_, ellipsoid_args) = find_ellipsoid(&root)? else {
        return None;
    };
    let semi_major: f64 = ellipsoid_args.get(1)?.leaf_str()?.trim().parse().ok()?;
    let inverse_flattening: f64 = ellipsoid_args.get(2)?.leaf_str()?.trim().parse().ok()?;
    if !semi_major.is_finite() || semi_major <= 0.0 {
        return None;
    }

    let spherical_radius_m = if inverse_flattening == 0.0 {
        semi_major
    } else {
        let semi_minor = semi_major * (1.0 - 1.0 / inverse_flattening);
        (2.0 * semi_major + semi_minor) / 3.0
    };
    Some(GeographicCrsParams { spherical_radius_m })
}

/// A CRS given as a WKT1/WKT2 string.
///
/// The WKT is preserved verbatim (`to_crs_string`) so it can be handed to PROJ
/// for transforms, while an `authority:code` is extracted once at construction
/// for equality and SRID. Geographic parameters resolve from that authority
/// when it's known, otherwise straight from the WKT's own ellipsoid.
#[derive(Debug)]
struct WktCrs {
    wkt: String,
    authority_code: Option<String>,
}

impl WktCrs {
    fn new(wkt: String) -> Self {
        let authority_code = extract_authority_from_wkt(&wkt);
        Self {
            wkt,
            authority_code,
        }
    }
}

impl CoordinateReferenceSystem for WktCrs {
    /// Emit the WKT verbatim (as a JSON string), preserving the definition the
    /// same way [`ProjJSON::to_json`] does — a WKT carrying an embedded
    /// authority is **not** collapsed to that code, so the full definition
    /// survives a metadata round-trip and `ST_Crs` reads it back unchanged.
    /// (The GeoParquet writer resolves it to PROJJSON, or to the lon/lat
    /// default, via the CRS provider, like any other definition.)
    fn to_json(&self) -> String {
        Value::String(self.wkt.clone()).to_string()
    }

    fn to_authority_code(&self) -> Result<Option<String>> {
        Ok(self.authority_code.clone())
    }

    fn crs_equals(&self, other: &dyn CoordinateReferenceSystem) -> bool {
        if let (Ok(Some(a)), Ok(Some(b))) = (self.to_authority_code(), other.to_authority_code()) {
            return authority_codes_equal(&a, &b);
        }
        // No authority on one side: fall back to comparing raw definitions.
        self.to_crs_string() == other.to_crs_string()
    }

    // `srid()` uses the trait default (derived from `to_authority_code`).

    /// The verbatim WKT — the form PROJ/GDAL consume directly.
    fn to_crs_string(&self) -> String {
        self.wkt.clone()
    }

    fn geographic_params(&self) -> Result<Option<GeographicCrsParams>> {
        // Prefer the curated authority lookup (stable, version-independent
        // radii); otherwise derive the ellipsoid straight from the WKT so
        // authority-less geographic CRSes still work.
        Ok(self
            .authority_code
            .as_deref()
            .and_then(geographic_params_for_authority)
            .or_else(|| geographic_params_from_wkt(&self.wkt)))
    }
}

pub const OGC_CRS84_PROJJSON: &str = r#"{"$schema":"https://proj.org/schemas/v0.7/projjson.schema.json","type":"GeographicCRS","name":"WGS 84 (CRS84)","datum_ensemble":{"name":"World Geodetic System 1984 ensemble","members":[{"name":"World Geodetic System 1984 (Transit)","id":{"authority":"EPSG","code":1166}},{"name":"World Geodetic System 1984 (G730)","id":{"authority":"EPSG","code":1152}},{"name":"World Geodetic System 1984 (G873)","id":{"authority":"EPSG","code":1153}},{"name":"World Geodetic System 1984 (G1150)","id":{"authority":"EPSG","code":1154}},{"name":"World Geodetic System 1984 (G1674)","id":{"authority":"EPSG","code":1155}},{"name":"World Geodetic System 1984 (G1762)","id":{"authority":"EPSG","code":1156}},{"name":"World Geodetic System 1984 (G2139)","id":{"authority":"EPSG","code":1309}},{"name":"World Geodetic System 1984 (G2296)","id":{"authority":"EPSG","code":1383}}],"ellipsoid":{"name":"WGS 84","semi_major_axis":6378137,"inverse_flattening":298.257223563},"accuracy":"2.0","id":{"authority":"EPSG","code":6326}},"coordinate_system":{"subtype":"ellipsoidal","axis":[{"name":"Geodetic longitude","abbreviation":"Lon","direction":"east","unit":"degree"},{"name":"Geodetic latitude","abbreviation":"Lat","direction":"north","unit":"degree"}]},"scope":"Not known.","area":"World.","bbox":{"south_latitude":-90,"west_longitude":-180,"north_latitude":90,"east_longitude":180},"id":{"authority":"OGC","code":"CRS84"}}"#;

#[cfg(test)]
mod test {
    use super::*;
    /// Projected CRS
    const EPSG_6318_PROJJSON: &str = r#"{"$schema": "https://proj.org/schemas/v0.4/projjson.schema.json","type": "GeographicCRS","name": "NAD83(2011)","datum": {"type": "GeodeticReferenceFrame","name": "NAD83 (National Spatial Reference System 2011)","ellipsoid": {"name": "GRS 1980","semi_major_axis": 6378137,"inverse_flattening": 298.257222101}},"coordinate_system": {"subtype": "ellipsoidal","axis": [{"name": "Geodetic latitude","abbreviation": "Lat","direction": "north","unit": "degree"},{"name": "Geodetic longitude","abbreviation": "Lon","direction": "east","unit": "degree"}]},"scope": "Horizontal component of 3D system.","area": "Puerto Rico - onshore and offshore. United States (USA) onshore and offshore - Alabama; Alaska; Arizona; Arkansas; California; Colorado; Connecticut; Delaware; Florida; Georgia; Idaho; Illinois; Indiana; Iowa; Kansas; Kentucky; Louisiana; Maine; Maryland; Massachusetts; Michigan; Minnesota; Mississippi; Missouri; Montana; Nebraska; Nevada; New Hampshire; New Jersey; New Mexico; New York; North Carolina; North Dakota; Ohio; Oklahoma; Oregon; Pennsylvania; Rhode Island; South Carolina; South Dakota; Tennessee; Texas; Utah; Vermont; Virginia; Washington; West Virginia; Wisconsin; Wyoming. US Virgin Islands - onshore and offshore.", "bbox": {"south_latitude": 14.92,"west_longitude": 167.65,"north_latitude": 74.71,"east_longitude": -63.88},"id": {"authority": "EPSG","code": 6318}}"#;

    /// Mars geographic CRS (only defines 'radius')
    const IAU_49900_PROJJSON: &str = r#"{"$schema":"https://proj.org/schemas/v0.7/projjson.schema.json","type":"GeographicCRS","name":"Mars (2015) - Sphere / Ocentric","datum":{"type":"GeodeticReferenceFrame","name":"Mars (2015) - Sphere","anchor":"Viking 1 lander: 47.95137 W","ellipsoid":{"name":"Mars (2015) - Sphere","radius":3396190},"prime_meridian":{"name":"Reference Meridian","longitude":0}},"coordinate_system":{"subtype":"ellipsoidal","axis":[{"name":"Geodetic latitude","abbreviation":"Lat","direction":"north","unit":"degree"},{"name":"Geodetic longitude","abbreviation":"Lon","direction":"east","unit":"degree"}]},"id":{"authority":"IAU","code":49900,"version":2015},"remarks":"Use semi-major radius as sphere for interoperability. Source of IAU Coordinate systems: https://doi.org/10.1007/s10569-017-9805-5"}"#;

    /// NAD27 (defines semi_major/semi_minor instead of inverse flattening)
    const EPSG_4267_PROJJSON: &str = r#"{"$schema":"https://proj.org/schemas/v0.7/projjson.schema.json","type":"GeographicCRS","name":"NAD27","datum":{"type":"GeodeticReferenceFrame","name":"North American Datum 1927","ellipsoid":{"name":"Clarke 1866","semi_major_axis":6378206.4,"semi_minor_axis":6356583.8}},"coordinate_system":{"subtype":"ellipsoidal","axis":[{"name":"Geodetic latitude","abbreviation":"Lat","direction":"north","unit":"degree"},{"name":"Geodetic longitude","abbreviation":"Lon","direction":"east","unit":"degree"}]},"scope":"Geodesy.","area":"North and central America: Antigua and Barbuda - onshore. Bahamas - onshore plus offshore over internal continental shelf only. Belize - onshore. British Virgin Islands - onshore. Canada onshore - Alberta, British Columbia, Manitoba, New Brunswick, Newfoundland and Labrador, Northwest Territories, Nova Scotia, Nunavut, Ontario, Prince Edward Island, Quebec, Saskatchewan and Yukon - plus offshore east coast. Cuba - onshore and offshore. El Salvador - onshore. Guatemala - onshore. Honduras - onshore. Panama - onshore. Puerto Rico - onshore. Mexico - onshore plus offshore east coast. Nicaragua - onshore. United States (USA) onshore and offshore - Alabama, Alaska, Arizona, Arkansas, California, Colorado, Connecticut, Delaware, Florida, Georgia, Idaho, Illinois, Indiana, Iowa, Kansas, Kentucky, Louisiana, Maine, Maryland, Massachusetts, Michigan, Minnesota, Mississippi, Missouri, Montana, Nebraska, Nevada, New Hampshire, New Jersey, New Mexico, New York, North Carolina, North Dakota, Ohio, Oklahoma, Oregon, Pennsylvania, Rhode Island, South Carolina, South Dakota, Tennessee, Texas, Utah, Vermont, Virginia, Washington, West Virginia, Wisconsin and Wyoming - plus offshore . US Virgin Islands - onshore.","bbox":{"south_latitude":7.15,"west_longitude":167.65,"north_latitude":83.17,"east_longitude":-47.74},"id":{"authority":"EPSG","code":4267},"remarks":"Note: this CRS includes longitudes which are POSITIVE EAST. Replaced by NAD27(76) (code 4608) in Ontario, CGQ77 (code 4609) in Quebec, Mexican Datum of 1993 (code 4483) in Mexico, NAD83 (code 4269) in Canada (excl. Ontario & Quebec) & USA."}"#;

    #[test]
    fn deserialize() {
        // lnglat() crses
        assert_eq!(deserialize_crs("EPSG:4326").unwrap(), lnglat());
        assert_eq!(deserialize_crs("OGC:CRS84").unwrap(), lnglat());
        assert_eq!(deserialize_crs(OGC_CRS84_PROJJSON).unwrap(), lnglat());

        // None crses
        assert!(deserialize_crs("").unwrap().is_none());
        assert!(deserialize_crs("0").unwrap().is_none());

        // Authority:Code CRSes
        // Make sure we can deserialize a few common authorities
        assert_eq!(
            deserialize_crs("EPSG:3857")
                .unwrap()
                .unwrap()
                .to_authority_code()
                .unwrap(),
            Some("EPSG:3857".to_string())
        );

        assert_eq!(
            deserialize_crs("OGC:CRS27")
                .unwrap()
                .unwrap()
                .to_authority_code()
                .unwrap(),
            Some("OGC:CRS27".to_string())
        );

        assert_eq!(
            deserialize_crs("ESRI:102005")
                .unwrap()
                .unwrap()
                .to_authority_code()
                .unwrap(),
            Some("ESRI:102005".to_string())
        );

        // Check that we can deserialize lowercase authorities and that they compare
        // equal to upperdase authorities
        assert_eq!(
            deserialize_crs("epsg:3857").unwrap(),
            deserialize_crs("EPSG:3857").unwrap()
        );

        // Erroneous CRSes
        assert!(deserialize_crs("[]").is_err());

        let crs_value = serde_json::Value::String("EPSG:4326".to_string());
        assert_eq!(deserialize_crs_from_obj(&crs_value).unwrap(), lnglat());

        let crs_value = serde_json::Value::String("OGC:CRS84".to_string());
        assert_eq!(deserialize_crs_from_obj(&crs_value).unwrap(), lnglat());

        let projjson_value: Value = serde_json::from_str(OGC_CRS84_PROJJSON).unwrap();
        assert_eq!(deserialize_crs_from_obj(&projjson_value).unwrap(), lnglat());

        assert!(deserialize_crs_from_obj(&Value::Null).unwrap().is_none());
        assert!(deserialize_crs_from_obj(&serde_json::Value::String("[]".to_string())).is_err());
    }

    #[test]
    fn crs_projjson() {
        let projjson = OGC_CRS84_PROJJSON.parse::<ProjJSON>().unwrap();
        assert_eq!(projjson.to_authority_code().unwrap().unwrap(), "OGC:CRS84");
        assert_eq!(projjson.srid().unwrap(), Some(4326));
        assert_eq!(projjson.to_json(), projjson.to_crs_string());

        let json_value: Value = serde_json::from_str(OGC_CRS84_PROJJSON).unwrap();
        let json_value_roundtrip: Value = serde_json::from_str(&projjson.to_json()).unwrap();
        assert_eq!(json_value_roundtrip, json_value);

        assert!(projjson.crs_equals(LngLat::crs().unwrap().as_ref()));
        assert!(projjson.crs_equals(&projjson));

        let projjson_without_identifier = "{}".parse::<ProjJSON>().unwrap();
        assert!(projjson_without_identifier
            .to_authority_code()
            .unwrap()
            .is_none());
        assert!(!projjson.crs_equals(&projjson_without_identifier));

        let projjson = EPSG_6318_PROJJSON.parse::<ProjJSON>().unwrap();
        assert_eq!(projjson.to_authority_code().unwrap().unwrap(), "EPSG:6318");
        assert_eq!(projjson.srid().unwrap(), Some(6318));
    }

    #[test]
    fn crs_authority_code() {
        let auth_code = AuthorityCode {
            auth_code: "EPSG:4269".to_string(),
        };
        assert!(auth_code.crs_equals(&auth_code));
        assert!(!auth_code.crs_equals(LngLat::crs().unwrap().as_ref()));
        assert_eq!(auth_code.srid().unwrap(), Some(4269));
        assert_eq!(auth_code.to_crs_string(), "EPSG:4269");

        assert_eq!(
            auth_code.to_authority_code().unwrap(),
            Some("EPSG:4269".to_string())
        );

        let json_value = auth_code.to_json();
        assert_eq!(json_value, "\"EPSG:4269\"");

        let auth_code_crs = AuthorityCode::crs("EPSG:4269");
        assert_eq!(auth_code_crs, auth_code_crs);
        assert_ne!(auth_code_crs, Crs::None);

        assert!(AuthorityCode::is_authority_code("EPSG:4269"));

        let new_crs = deserialize_crs("EPSG:4269").unwrap().unwrap();
        assert_eq!(
            new_crs.to_authority_code().unwrap(),
            Some("EPSG:4269".to_string())
        );
        assert_eq!(new_crs.srid().unwrap(), Some(4269));

        // Ensure we can also just pass a code here
        let new_crs = deserialize_crs("4269").unwrap();
        assert_eq!(
            new_crs.clone().unwrap().to_authority_code().unwrap(),
            Some("EPSG:4269".to_string())
        );
        assert_eq!(new_crs.unwrap().srid().unwrap(), Some(4269));

        let new_crs = deserialize_crs("EPSG:4326").unwrap();
        assert_eq!(new_crs.clone().unwrap().srid().unwrap(), Some(4326));

        let crs_value = serde_json::Value::String("EPSG:4326".to_string());
        let new_crs = deserialize_crs_from_obj(&crs_value).unwrap();
        assert_eq!(new_crs.clone().unwrap().srid().unwrap(), Some(4326));

        let crs_value = serde_json::Value::String("4269".to_string());
        let new_crs = deserialize_crs_from_obj(&crs_value).unwrap();
        assert_eq!(new_crs.clone().unwrap().srid().unwrap(), Some(4269));
    }

    #[test]
    fn geographic_params() {
        let lnglat = AuthorityCode {
            auth_code: "OGC:CRS84".to_string(),
        };
        let params = lnglat.geographic_params().unwrap().unwrap();
        assert_eq!(params.spherical_radius(), 6371010.0);

        let auth_code_nad27 = AuthorityCode {
            auth_code: "EPSG:4267".to_string(),
        };
        let params = auth_code_nad27.geographic_params().unwrap().unwrap();
        assert_eq!(params.spherical_radius(), 6370998.866666667);

        let projjson_nad27 = ProjJSON::from_str(EPSG_4267_PROJJSON).unwrap();
        let params = projjson_nad27.geographic_params().unwrap().unwrap();
        assert_eq!(params.spherical_radius(), 6370998.866666667);

        let projjson_wgs84 = ProjJSON::from_str(OGC_CRS84_PROJJSON).unwrap();
        let params = projjson_wgs84.geographic_params().unwrap().unwrap();
        assert_eq!(params.spherical_radius(), 6371008.771415059);

        let projjson_mars = ProjJSON::from_str(IAU_49900_PROJJSON).unwrap();
        let params = projjson_mars.geographic_params().unwrap().unwrap();
        assert_eq!(params.spherical_radius(), 3396190.0);
    }

    #[test]
    fn geographic_params_invalid_projjson() {
        // Missing type field -> Ok(None)
        let no_type = ProjJSON::try_new(serde_json::json!({"name": "test"})).unwrap();
        assert!(no_type.geographic_params().unwrap().is_none());

        // Non-GeographicCRS type -> Ok(None)
        let projected = ProjJSON::try_new(serde_json::json!({
            "type": "ProjectedCRS",
            "name": "test"
        }))
        .unwrap();
        assert!(projected.geographic_params().unwrap().is_none());

        // GeographicCRS missing datum and datum_ensemble -> Error
        let no_datum = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "name": "test"
        }))
        .unwrap();
        assert!(no_datum.geographic_params().is_err());

        // datum is not an object -> Error (datum_ensemble fallback also fails)
        let datum_not_object = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "datum": "not an object"
        }))
        .unwrap();
        assert!(datum_not_object.geographic_params().is_err());

        // ellipsoid is not an object -> Error
        let ellipsoid_not_object = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "datum": {"ellipsoid": "not an object"}
        }))
        .unwrap();
        assert!(ellipsoid_not_object.geographic_params().is_err());

        // ellipsoid missing all axis definitions -> Error
        let ellipsoid_empty = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "datum": {"ellipsoid": {"name": "empty"}}
        }))
        .unwrap();
        assert!(ellipsoid_empty.geographic_params().is_err());

        // semi_major_axis is not a number -> Error
        let semi_major_not_number = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "datum": {"ellipsoid": {"semi_major_axis": "not a number"}}
        }))
        .unwrap();
        assert!(semi_major_not_number.geographic_params().is_err());

        // semi_major_axis present but no inverse_flattening or semi_minor_axis -> Error
        let missing_minor = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "datum": {"ellipsoid": {"semi_major_axis": 6378137}}
        }))
        .unwrap();
        assert!(missing_minor.geographic_params().is_err());

        // inverse_flattening is not a number -> Error
        let inv_f_not_number = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "datum": {"ellipsoid": {
                "semi_major_axis": 6378137,
                "inverse_flattening": "not a number"
            }}
        }))
        .unwrap();
        assert!(inv_f_not_number.geographic_params().is_err());

        // semi_minor_axis is not a number -> Error
        let semi_minor_not_number = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "datum": {"ellipsoid": {
                "semi_major_axis": 6378137,
                "semi_minor_axis": "not a number"
            }}
        }))
        .unwrap();
        assert!(semi_minor_not_number.geographic_params().is_err());

        // radius is not a number -> Error
        let radius_not_number = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "datum": {"ellipsoid": {"radius": "not a number"}}
        }))
        .unwrap();
        assert!(radius_not_number.geographic_params().is_err());

        // datum_ensemble path works when datum is missing
        let datum_ensemble = ProjJSON::try_new(serde_json::json!({
            "type": "GeographicCRS",
            "datum_ensemble": {"ellipsoid": {"radius": 6371000}}
        }))
        .unwrap();
        let params = datum_ensemble.geographic_params().unwrap().unwrap();
        assert_eq!(params.spherical_radius(), 6371000.0);
    }

    #[test]
    fn normalize_preserves_definition() {
        // Empty / "0" -> no CRS.
        assert_eq!(normalize_crs("0").unwrap(), None);
        assert_eq!(normalize_crs("").unwrap(), None);

        // An authority:code stays compact.
        assert_eq!(
            normalize_crs("EPSG:3857").unwrap().as_deref(),
            Some("EPSG:3857")
        );

        // The lon/lat alias is kept verbatim (not folded to OGC:CRS84); the two
        // still compare equal, but the user's input string is preserved.
        assert_eq!(
            normalize_crs("EPSG:4326").unwrap().as_deref(),
            Some("EPSG:4326")
        );
        assert_eq!(
            deserialize_crs("EPSG:4326").unwrap(),
            deserialize_crs("OGC:CRS84").unwrap()
        );

        // A PROJJSON carrying an embedded authority id is preserved in full
        // rather than collapsed to "EPSG:6318" — no information lost on input.
        let normalized = normalize_crs(EPSG_6318_PROJJSON).unwrap().unwrap();
        assert!(
            normalized.contains("GeographicCRS"),
            "expected full PROJJSON, got {normalized}"
        );
        assert_ne!(normalized, "EPSG:6318");
        // It round-trips back to the same CRS.
        assert_eq!(
            deserialize_crs(&normalized).unwrap(),
            deserialize_crs(EPSG_6318_PROJJSON).unwrap()
        );

        // A PROJJSON with no authority id: there's nothing to collapse to, so
        // the full definition is the only faithful output. Round-trip equality
        // here exercises ProjJSON::crs_equals' full-body comparison (not the
        // authority-code short circuit), proving the definition survives
        // re-serialization.
        const NO_ID_PROJJSON: &str = r#"{"type":"GeographicCRS","name":"Custom","datum":{"type":"GeodeticReferenceFrame","name":"Custom datum","ellipsoid":{"name":"Custom","semi_major_axis":6378137,"inverse_flattening":298.257223563}},"coordinate_system":{"subtype":"ellipsoidal","axis":[{"name":"Geodetic latitude","abbreviation":"Lat","direction":"north","unit":"degree"},{"name":"Geodetic longitude","abbreviation":"Lon","direction":"east","unit":"degree"}]}}"#;
        let normalized = normalize_crs(NO_ID_PROJJSON).unwrap().unwrap();
        let from_normalized = deserialize_crs(&normalized).unwrap().unwrap();
        let from_original = deserialize_crs(NO_ID_PROJJSON).unwrap().unwrap();
        assert!(from_normalized.to_authority_code().unwrap().is_none());
        assert!(from_normalized.crs_equals(from_original.as_ref()));
    }

    #[test]
    fn lnglat_alias_preserved_in_string_canonical_in_metadata() {
        // EPSG:4326 is preserved as the round-trippable string (to_crs_string),
        // but the JSON metadata form (to_json) canonicalizes to OGC:CRS84 so
        // GeoParquet/GeoArrow stay axis-order-explicit.
        let crs = deserialize_crs("EPSG:4326").unwrap().unwrap();
        assert_eq!(crs.to_crs_string(), "EPSG:4326");
        assert_eq!(crs.to_json(), r#""OGC:CRS84""#);

        // OGC:CRS84 is unchanged in both forms.
        let crs84 = deserialize_crs("OGC:CRS84").unwrap().unwrap();
        assert_eq!(crs84.to_crs_string(), "OGC:CRS84");
        assert_eq!(crs84.to_json(), r#""OGC:CRS84""#);

        // The two aliases still compare equal, and both report SRID 4326.
        assert!(crs.crs_equals(crs84.as_ref()));
        assert_eq!(crs.srid().unwrap(), Some(4326));

        // A non-lnglat authority code is untouched in both forms.
        let crs3857 = deserialize_crs("EPSG:3857").unwrap().unwrap();
        assert_eq!(crs3857.to_crs_string(), "EPSG:3857");
        assert_eq!(crs3857.to_json(), r#""EPSG:3857""#);
    }
}

#[cfg(test)]
mod wkt_tests {
    use super::*;

    // CRS-level AUTHORITY is `EPSG:3857`; inner datum/ellipsoid tags
    // (7030/6326/4326) must not win.
    const WKT1_PSEUDO_MERCATOR: &str = r#"PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],AUTHORITY["EPSG","3857"]]"#;
    // WKT2 with a bare (unquoted) CRS-level code; the nested ELLIPSOID must not
    // be mistaken for the CRS identifier.
    const WKT2_WGS84: &str = r#"GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],CS[ellipsoidal,2],ID["EPSG",4326]]"#;
    const WKT_NO_AUTHORITY: &str = r#"PROJCS["Custom",GEOGCS["Custom datum",DATUM["D",SPHEROID["S",6378137,298.257223563]]],PROJECTION["Mercator_1SP"]]"#;

    // A custom projected CRS with no CRS-level authority whose inner GEOGCS *does*
    // carry one (EPSG:4326). The projected CRS is not EPSG:4326 — the base
    // geographic CRS is — so no authority should be extracted.
    const WKT1_CUSTOM_LCC_INNER_AUTHORITY: &str = r#"PROJCS["Custom LCC",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],AUTHORITY["EPSG","4326"]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["standard_parallel_1",33],PARAMETER["standard_parallel_2",45],UNIT["metre",1]]"#;

    // A native HRRR Lambert Conformal CRS that cannot be described by a single
    // EPSG code. Its only `ID`s are nested unit/operation codes — `EPSG:9001`
    // (metre), `EPSG:8901` (Greenwich), the `EPSG:9801` projection method, and
    // its `EPSG:8801..8807` parameters — and the last of them (`EPSG:9001`) is
    // exactly what a naive "last tag" scan would wrongly return as the CRS.
    const WKT2_HRRR_LCC: &str = r#"PROJCRS["HRRR Lambert Conformal",BASEGEOGCRS["unknown",DATUM["unknown",ELLIPSOID["unknown",6371229,0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8901]]],CONVERSION["unknown",METHOD["Lambert Conic Conformal (1SP)",ID["EPSG",9801]],PARAMETER["Latitude of natural origin",38.5,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",-97.5,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",1,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"#;

    #[test]
    fn detects_wkt_strings() {
        assert!(looks_like_wkt(WKT1_PSEUDO_MERCATOR));
        assert!(looks_like_wkt(WKT2_WGS84));
        assert!(looks_like_wkt("  geogcs[\"x\"]")); // leading ws + lowercase
        assert!(!looks_like_wkt("EPSG:4326"));
        assert!(!looks_like_wkt(r#"{"type":"GeographicCRS"}"#));
        assert!(!looks_like_wkt(r#"PROJECTILE["not a crs"]"#));
    }

    #[test]
    fn extracts_crs_level_authority_not_inner_tags() {
        assert_eq!(
            extract_authority_from_wkt(WKT1_PSEUDO_MERCATOR).as_deref(),
            Some("EPSG:3857")
        );
    }

    #[test]
    fn extracts_wkt2_bare_code_and_ignores_ellipsoid() {
        assert_eq!(
            extract_authority_from_wkt(WKT2_WGS84).as_deref(),
            Some("EPSG:4326")
        );
    }

    #[test]
    fn wkt_without_authority_has_no_code() {
        assert_eq!(extract_authority_from_wkt(WKT_NO_AUTHORITY), None);
    }

    #[test]
    fn inner_authority_does_not_win_without_root_authority() {
        // The inner GEOGCS EPSG:4326 belongs to the base CRS, not the projected
        // CRS, so the custom LCC has no authority of its own.
        assert_eq!(
            extract_authority_from_wkt(WKT1_CUSTOM_LCC_INNER_AUTHORITY),
            None
        );
        // The HRRR LCC's only codes are nested unit/operation IDs; none is the
        // CRS, so a naive scan returning EPSG:9001 (metre) would be wrong.
        assert_eq!(extract_authority_from_wkt(WKT2_HRRR_LCC), None);
    }

    #[test]
    fn quoted_name_with_structural_chars_is_not_parsed_as_structure() {
        // The name carries `[`, `]`, `,`, a doubled-quote escape, and the literal
        // text `ID` — none of which is structure, since it is all inside a quoted
        // string. The real CRS-level authority must still win.
        let wkt = r#"PROJCS["Bob""s [ID] zone, N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],UNIT["metre",1],AUTHORITY["EPSG","32633"]]"#;
        assert_eq!(
            extract_authority_from_wkt(wkt).as_deref(),
            Some("EPSG:32633")
        );
    }

    #[test]
    fn unit_nested_authority_does_not_win() {
        // WKT1 analog of the HRRR trap: the only authority is the metre UNIT's
        // EPSG:9001, nested below the root, so no CRS authority is extracted.
        let wkt = r#"PROJCS["Custom TM",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]]],PROJECTION["Transverse_Mercator"],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]"#;
        assert_eq!(extract_authority_from_wkt(wkt), None);
    }

    #[test]
    fn parser_handles_whitespace_and_lowercase_keywords() {
        // Pretty-printed across lines, indented, and lowercase keywords.
        let wkt = "geogcrs[\"WGS 84\",\n    datum[\"World Geodetic System 1984\",\n        ellipsoid[\"WGS 84\",6378137,298.257223563]],\n    cs[ellipsoidal,2],\n    id[\"epsg\",4326]]";
        assert_eq!(
            extract_authority_from_wkt(wkt).as_deref(),
            Some("EPSG:4326")
        );
    }

    #[test]
    fn malformed_wkt_returns_none_without_panicking() {
        for input in [
            "",
            "EPSG:4326",
            "PROJCS",        // keyword, no bracket
            "PROJCS[",       // unterminated node
            "GEOGCRS[\"x\"", // unterminated, dangling child
            "GEOGCRS[ID[",   // unterminated nested node
            "PROJCS[\"unterminated name",
        ] {
            assert_eq!(extract_authority_from_wkt(input), None, "input: {input:?}");
        }
    }

    #[test]
    fn deeply_nested_wkt_is_bounded() {
        // Pathological nesting must not overflow the stack; extraction bails out
        // (and logs a warning) well before that, returning None.
        let deep = format!("GEOGCRS{}", "[A".repeat(WKT_MAX_DEPTH + 500));
        assert_eq!(extract_authority_from_wkt(&deep), None);
    }

    #[test]
    fn wkt2_id_with_trailing_version_is_extracted() {
        // A third `ID` argument (version) must not prevent extraction.
        let wkt = r#"GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],ID["EPSG",4326,"2024"]]"#;
        assert_eq!(
            extract_authority_from_wkt(wkt).as_deref(),
            Some("EPSG:4326")
        );
    }

    #[test]
    fn custom_lcc_crs_has_no_srid_and_carries_definition_through() {
        // A custom Lambert Conformal CRS that cannot be represented as an EPSG
        // code must not be misidentified: no authority, no SRID, no spurious
        // equality with the nested unit code, and the WKT survives verbatim.
        let crs = deserialize_crs(WKT2_HRRR_LCC).unwrap().unwrap();
        assert_eq!(crs.to_authority_code().unwrap(), None);
        assert_eq!(crs.srid().unwrap(), None);
        assert_eq!(crs.to_crs_string(), WKT2_HRRR_LCC);
        let metre_unit = deserialize_crs("EPSG:9001").unwrap().unwrap();
        assert!(!crs.crs_equals(metre_unit.as_ref()));
    }

    #[test]
    fn deserialize_routes_wkt_to_authority_and_preserves_definition() {
        let crs = deserialize_crs(WKT1_PSEUDO_MERCATOR).unwrap().unwrap();
        assert_eq!(
            crs.to_authority_code().unwrap().as_deref(),
            Some("EPSG:3857")
        );
        assert_eq!(crs.srid().unwrap(), Some(3857));
        // PROJ/GDAL get the verbatim WKT, not the authority code.
        assert_eq!(crs.to_crs_string(), WKT1_PSEUDO_MERCATOR);
    }

    #[test]
    fn wkt_without_authority_carries_definition_through() {
        let crs = deserialize_crs(WKT_NO_AUTHORITY).unwrap().unwrap();
        assert_eq!(crs.to_authority_code().unwrap(), None);
        assert_eq!(crs.srid().unwrap(), None);
        assert_eq!(crs.to_crs_string(), WKT_NO_AUTHORITY);
    }

    #[test]
    fn wkt_equals_matching_authority_code() {
        let wkt_crs = deserialize_crs(WKT1_PSEUDO_MERCATOR).unwrap().unwrap();
        let auth_crs = deserialize_crs("EPSG:3857").unwrap().unwrap();
        assert!(wkt_crs.crs_equals(auth_crs.as_ref()));
        assert!(auth_crs.crs_equals(wkt_crs.as_ref()));
    }

    #[test]
    fn wkt_geographic_params_resolve_via_authority() {
        let geographic = deserialize_crs(WKT2_WGS84).unwrap().unwrap();
        assert!(geographic.geographic_params().unwrap().is_some()); // EPSG:4326
        let projected = deserialize_crs(WKT1_PSEUDO_MERCATOR).unwrap().unwrap();
        assert!(projected.geographic_params().unwrap().is_none()); // EPSG:3857
    }

    #[test]
    fn wkt_geographic_params_parsed_from_ellipsoid_when_authority_unknown() {
        // An authority-less geographic WKT: params are read from its ellipsoid
        // (GRS 1980, a=6378137, 1/f=298.257222101) and reduced to a mean radius.
        let wkt = r#"GEOGCRS["unknown",DATUM["unknown",ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],CS[ellipsoidal,2],AXIS["lat",north],AXIS["lon",east]]"#;
        let crs = deserialize_crs(wkt).unwrap().unwrap();
        let params = crs.geographic_params().unwrap().expect("geographic params");
        let a = 6378137.0_f64;
        let b = a * (1.0 - 1.0 / 298.257222101);
        assert!((params.spherical_radius() - (2.0 * a + b) / 3.0).abs() < 1e-6);
    }

    #[test]
    fn wkt_sphere_radius_parsed_from_ellipsoid() {
        // A spherical ellipsoid (inverse flattening 0) uses its radius directly.
        let wkt = r#"GEOGCRS["sphere",DATUM["d",ELLIPSOID["sphere",6371229,0,LENGTHUNIT["metre",1]]],CS[ellipsoidal,2]]"#;
        let crs = deserialize_crs(wkt).unwrap().unwrap();
        let params = crs.geographic_params().unwrap().expect("geographic params");
        assert_eq!(params.spherical_radius(), 6371229.0);
    }

    #[test]
    fn projected_wkt_has_no_geographic_params() {
        // A projected CRS is never a geography, even though it nests a GEOGCS
        // with a SPHEROID — the parse fallback must reject it.
        let crs = deserialize_crs(WKT_NO_AUTHORITY).unwrap().unwrap();
        assert!(crs.geographic_params().unwrap().is_none());
    }

    #[test]
    fn wkt_to_json_preserves_definition() {
        // to_json emits the WKT verbatim (as a quoted JSON string), like
        // ProjJSON — a WKT carrying an embedded authority is NOT collapsed to
        // that code, so the full definition survives a metadata round-trip.
        let crs = deserialize_crs(WKT1_PSEUDO_MERCATOR).unwrap().unwrap();
        assert_eq!(
            crs.to_json(),
            Value::String(WKT1_PSEUDO_MERCATOR.to_string()).to_string()
        );
        let bare = deserialize_crs(WKT_NO_AUTHORITY).unwrap().unwrap();
        assert_eq!(
            bare.to_json(),
            Value::String(WKT_NO_AUTHORITY.to_string()).to_string()
        );
    }
}
