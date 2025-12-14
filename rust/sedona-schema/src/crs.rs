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
use datafusion_common::{DataFusionError, Result};
use lru::LruCache;
use std::fmt::{Debug, Display};
use std::num::NonZeroUsize;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use serde_json::Value;

/// LRU cache for CRS deserialization
static CRS_CACHE: OnceLock<Mutex<LruCache<String, Crs>>> = OnceLock::new();

fn get_crs_cache() -> &'static Mutex<LruCache<String, Crs>> {
    CRS_CACHE.get_or_init(|| {
        // Cache up to 256 CRS strings
        Mutex::new(LruCache::new(NonZeroUsize::new(256).unwrap()))
    })
}

/// Deserialize a specific GeoArrow "crs" value
///
/// Currently the only CRS types supported are PROJJSON and authority:code
/// where the authority and code are well-known lon/lat (WGS84) codes.
pub fn deserialize_crs(crs_str: &str) -> Result<Crs> {
    if crs_str.is_empty() {
        return Ok(None);
    }

    // Check cache first
    {
        let mut cache = get_crs_cache().lock().unwrap();
        if let Some(cached) = cache.get(crs_str) {
            return Ok(cached.clone());
        }
    }

    // Handle JSON strings "OGC:CRS84" and "EPSG:4326"
    let crs = if LngLat::is_str_lnglat(crs_str) {
        lnglat()
    } else if AuthorityCode::is_authority_code(crs_str) {
        AuthorityCode::crs(crs_str)
    } else {
        // Try to parse as PROJJSON string
        let projjson: ProjJSON = crs_str.parse()?;
        Some(Arc::new(projjson) as Arc<dyn CoordinateReferenceSystem + Send + Sync>)
    };

    // Cache the result
    {
        let mut cache = get_crs_cache().lock().unwrap();
        cache.put(crs_str.to_string(), crs.clone());
    }

    Ok(crs)
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
        if LngLat::is_lnglat(self) && LngLat::is_lnglat(other) {
            true
        } else {
            self.crs_equals(other)
        }
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
    fn srid(&self) -> Result<Option<u32>>;

    /// Compute a CRS string representation
    ///
    /// Unlike `to_json()`, arbitrary string values returned by this method should
    /// not be escaped. This is the representation expected as input to PROJ, GDAL,
    /// and Parquet GEOMETRY/GEOGRAPHY representations of CRS.
    fn to_crs_string(&self) -> String;
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

    pub fn is_lnglat(crs: &dyn CoordinateReferenceSystem) -> bool {
        if let Ok(Some(auth_code)) = crs.to_authority_code() {
            Self::is_authority_code_lnglat(&auth_code)
        } else {
            false
        }
    }

    pub fn is_str_lnglat(string_value: &str) -> bool {
        Self::is_authority_code_lnglat(string_value)
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
            format!("EPSG:{}", auth_code)
        } else {
            auth_code.to_string()
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
        matches!(authority, "EPSG" | "OGC")
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

/// Implementation of an authority:code CoordinateReferenceSystem
impl CoordinateReferenceSystem for AuthorityCode {
    /// Convert to a JSON string
    fn to_json(&self) -> String {
        let mut result = String::with_capacity(self.auth_code.len() + 2);
        result.push('"');
        result.push_str(&self.auth_code);
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
            Ok(Some(other_ac)) => other_ac == self.auth_code,
            _ => false,
        }
    }

    /// Get the SRID if authority is EPSG
    fn srid(&self) -> Result<Option<u32>> {
        if LngLat::is_authority_code_lnglat(&self.auth_code) {
            return Ok(LngLat::srid());
        }
        if let Some(pos) = self.auth_code.find(':') {
            if self.auth_code[..pos].eq_ignore_ascii_case("EPSG") {
                return Ok(self.auth_code[pos + 1..].parse::<u32>().ok());
            }
        }
        Ok(None)
    }

    /// Convert to a CRS string
    fn to_crs_string(&self) -> String {
        self.auth_code.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ProjJSON {
    value: Value,
}

impl FromStr for ProjJSON {
    type Err = DataFusionError;

    fn from_str(s: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(s).map_err(|err| {
            DataFusionError::Internal(format!("Error deserializing PROJJSON Crs: {err}"))
        })?;

        Self::try_new(value)
    }
}

impl ProjJSON {
    pub fn try_new(value: Value) -> Result<Self> {
        if !value.is_object() {
            return Err(DataFusionError::Internal(format!(
                "Can't create PROJJSON from non-object: {value}"
            )));
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
            auth_code == other_auth_code
        } else if let Ok(other_value) = serde_json::from_str::<Value>(&other.to_json()) {
            self.value == other_value
        } else {
            false
        }
    }

    fn srid(&self) -> Result<Option<u32>> {
        let authority_code_opt = self.to_authority_code()?;
        if let Some(authority_code) = authority_code_opt {
            if LngLat::is_authority_code_lnglat(&authority_code) {
                return Ok(LngLat::srid());
            } else if let Some(colon_pos) = authority_code.find(':') {
                if authority_code[..colon_pos].eq_ignore_ascii_case("EPSG") {
                    return Ok(authority_code[colon_pos + 1..].parse::<u32>().ok());
                }
            }
        }

        Ok(None)
    }

    fn to_crs_string(&self) -> String {
        self.to_json()
    }
}

pub const OGC_CRS84_PROJJSON: &str = r#"{"$schema":"https://proj.org/schemas/v0.7/projjson.schema.json","type":"GeographicCRS","name":"WGS 84 (CRS84)","datum_ensemble":{"name":"World Geodetic System 1984 ensemble","members":[{"name":"World Geodetic System 1984 (Transit)","id":{"authority":"EPSG","code":1166}},{"name":"World Geodetic System 1984 (G730)","id":{"authority":"EPSG","code":1152}},{"name":"World Geodetic System 1984 (G873)","id":{"authority":"EPSG","code":1153}},{"name":"World Geodetic System 1984 (G1150)","id":{"authority":"EPSG","code":1154}},{"name":"World Geodetic System 1984 (G1674)","id":{"authority":"EPSG","code":1155}},{"name":"World Geodetic System 1984 (G1762)","id":{"authority":"EPSG","code":1156}},{"name":"World Geodetic System 1984 (G2139)","id":{"authority":"EPSG","code":1309}},{"name":"World Geodetic System 1984 (G2296)","id":{"authority":"EPSG","code":1383}}],"ellipsoid":{"name":"WGS 84","semi_major_axis":6378137,"inverse_flattening":298.257223563},"accuracy":"2.0","id":{"authority":"EPSG","code":6326}},"coordinate_system":{"subtype":"ellipsoidal","axis":[{"name":"Geodetic longitude","abbreviation":"Lon","direction":"east","unit":"degree"},{"name":"Geodetic latitude","abbreviation":"Lat","direction":"north","unit":"degree"}]},"scope":"Not known.","area":"World.","bbox":{"south_latitude":-90,"west_longitude":-180,"north_latitude":90,"east_longitude":180},"id":{"authority":"OGC","code":"CRS84"}}"#;

#[cfg(test)]
mod test {
    use super::*;
    const EPSG_6318_PROJJSON: &str = r#"{"$schema": "https://proj.org/schemas/v0.4/projjson.schema.json","type": "GeographicCRS","name": "NAD83(2011)","datum": {"type": "GeodeticReferenceFrame","name": "NAD83 (National Spatial Reference System 2011)","ellipsoid": {"name": "GRS 1980","semi_major_axis": 6378137,"inverse_flattening": 298.257222101}},"coordinate_system": {"subtype": "ellipsoidal","axis": [{"name": "Geodetic latitude","abbreviation": "Lat","direction": "north","unit": "degree"},{"name": "Geodetic longitude","abbreviation": "Lon","direction": "east","unit": "degree"}]},"scope": "Horizontal component of 3D system.","area": "Puerto Rico - onshore and offshore. United States (USA) onshore and offshore - Alabama; Alaska; Arizona; Arkansas; California; Colorado; Connecticut; Delaware; Florida; Georgia; Idaho; Illinois; Indiana; Iowa; Kansas; Kentucky; Louisiana; Maine; Maryland; Massachusetts; Michigan; Minnesota; Mississippi; Missouri; Montana; Nebraska; Nevada; New Hampshire; New Jersey; New Mexico; New York; North Carolina; North Dakota; Ohio; Oklahoma; Oregon; Pennsylvania; Rhode Island; South Carolina; South Dakota; Tennessee; Texas; Utah; Vermont; Virginia; Washington; West Virginia; Wisconsin; Wyoming. US Virgin Islands - onshore and offshore.", "bbox": {"south_latitude": 14.92,"west_longitude": 167.65,"north_latitude": 74.71,"east_longitude": -63.88},"id": {"authority": "EPSG","code": 6318}}"#;

    #[test]
    fn deserialize() {
        assert_eq!(deserialize_crs("EPSG:4326").unwrap(), lnglat());

        assert_eq!(deserialize_crs("OGC:CRS84").unwrap(), lnglat());

        assert_eq!(deserialize_crs(OGC_CRS84_PROJJSON).unwrap(), lnglat());

        assert!(deserialize_crs("").unwrap().is_none());

        assert!(deserialize_crs("[]").is_err());
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
    }
}
