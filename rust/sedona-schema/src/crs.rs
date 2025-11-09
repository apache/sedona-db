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
use std::fmt::{Debug, Display};
use std::str::FromStr;
use std::sync::Arc;

use serde_json::Value;

/// Deserialize a specific GeoArrow "crs" value
///
/// Currently the only CRS types supported are PROJJSON and authority:code
/// where the authority and code are well-known lon/lat (WGS84) codes.
pub fn deserialize_crs(crs: &Value) -> Result<Crs> {
    if crs.is_null() {
        return Ok(None);
    }

    // Handle JSON strings "OGC:CRS84" and "EPSG:4326"
    if LngLat::is_value_lnglat(crs) {
        return Ok(lnglat());
    }

    if AuthorityCode::is_authority_code(crs) {
        return Ok(AuthorityCode::crs(crs.as_str().unwrap().to_string()));
    }

    let projjson = if let Some(string) = crs.as_str() {
        // Handle the geopandas bug where it exported stringified projjson.
        // This could in the future handle other stringified versions with some
        // auto detection logic.
        string.parse()?
    } else {
        // Handle the case where Value is already an object
        ProjJSON::try_new(crs.clone())?
    };

    Ok(Some(Arc::new(projjson)))
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
    fn to_json(&self) -> String;
    fn to_authority_code(&self) -> Result<Option<String>>;
    fn crs_equals(&self, other: &dyn CoordinateReferenceSystem) -> bool;
    fn srid(&self) -> Result<Option<u32>>;
}

/// Concrete implementation of a default longitude/latitude coordinate reference system
#[derive(Debug)]
struct LngLat {}

impl LngLat {
    pub fn crs() -> Crs {
        Crs::Some(Arc::new(AuthorityCode {
            authority: "OGC".to_string(),
            code: "CRS84".to_string(),
        }))
    }

    pub fn is_lnglat(crs: &dyn CoordinateReferenceSystem) -> bool {
        if let Ok(Some(auth_code)) = crs.to_authority_code() {
            Self::is_authority_code_lnglat(&auth_code)
        } else {
            false
        }
    }

    pub fn is_value_lnglat(value: &Value) -> bool {
        if let Some(string_value) = value.as_str() {
            Self::is_authority_code_lnglat(string_value)
        } else {
            false
        }
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
    authority: String,
    code: String,
}

/// Implementation of an authority:code
impl AuthorityCode {
    /// Create a Crs from an authority:code string
    /// Example: "EPSG:4269"
    pub fn crs(auth_code: String) -> Crs {
        let (authority, code) = Self::split_auth_code(&auth_code).unwrap();
        Crs::Some(Arc::new(AuthorityCode { authority, code }))
    }

    /// Check if a Value is an authority:code string
    /// Note: this can be expanded to more types in the future
    pub fn is_authority_code(value: &Value) -> bool {
        if let Some(string_value) = value.as_str() {
            match Self::split_auth_code(string_value) {
                Some((authority, code)) => {
                    return Self::validate_authority(&authority) && Self::validate_code(&code)
                }
                None => return false,
            }
        }

        false
    }

    /// Split an authority:code string into its components
    fn split_auth_code(auth_code: &str) -> Option<(String, String)> {
        let parts: Vec<&str> = auth_code.split(':').collect();
        if parts.len() == 2 {
            Some((parts[0].to_string(), parts[1].to_string()))
        } else if parts.len() == 1 && Self::validate_epsg_code(auth_code) {
            Some(("EPSG".to_string(), auth_code.to_string()))
        } else {
            None
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
        format!("\"{}:{}\"", self.authority, self.code)
    }

    /// Convert to an authority code string
    fn to_authority_code(&self) -> Result<Option<String>> {
        Ok(Some(format!("{}:{}", self.authority, self.code)))
    }

    /// Check equality with another CoordinateReferenceSystem
    fn crs_equals(&self, other: &dyn CoordinateReferenceSystem) -> bool {
        match (other.to_authority_code(), self.to_authority_code()) {
            (Ok(Some(other_ac)), Ok(Some(this_ac))) => other_ac == this_ac,
            (_, _) => false,
        }
    }

    /// Get the SRID if authority is EPSG
    fn srid(&self) -> Result<Option<u32>> {
        if self.authority.eq_ignore_ascii_case("EPSG") {
            Ok(self.code.parse::<u32>().ok())
        } else if LngLat::is_lnglat(self) {
            Ok(LngLat::srid())
        } else {
            Ok(None)
        }
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
            } else if let Some((_, code)) = AuthorityCode::split_auth_code(&authority_code) {
                return Ok(code.parse::<u32>().ok());
            }
        }

        Ok(None)
    }
}

pub const OGC_CRS84_PROJJSON: &str = r#"{"$schema":"https://proj.org/schemas/v0.7/projjson.schema.json","type":"GeographicCRS","name":"WGS 84 (CRS84)","datum_ensemble":{"name":"World Geodetic System 1984 ensemble","members":[{"name":"World Geodetic System 1984 (Transit)","id":{"authority":"EPSG","code":1166}},{"name":"World Geodetic System 1984 (G730)","id":{"authority":"EPSG","code":1152}},{"name":"World Geodetic System 1984 (G873)","id":{"authority":"EPSG","code":1153}},{"name":"World Geodetic System 1984 (G1150)","id":{"authority":"EPSG","code":1154}},{"name":"World Geodetic System 1984 (G1674)","id":{"authority":"EPSG","code":1155}},{"name":"World Geodetic System 1984 (G1762)","id":{"authority":"EPSG","code":1156}},{"name":"World Geodetic System 1984 (G2139)","id":{"authority":"EPSG","code":1309}},{"name":"World Geodetic System 1984 (G2296)","id":{"authority":"EPSG","code":1383}}],"ellipsoid":{"name":"WGS 84","semi_major_axis":6378137,"inverse_flattening":298.257223563},"accuracy":"2.0","id":{"authority":"EPSG","code":6326}},"coordinate_system":{"subtype":"ellipsoidal","axis":[{"name":"Geodetic longitude","abbreviation":"Lon","direction":"east","unit":"degree"},{"name":"Geodetic latitude","abbreviation":"Lat","direction":"north","unit":"degree"}]},"scope":"Not known.","area":"World.","bbox":{"south_latitude":-90,"west_longitude":-180,"north_latitude":90,"east_longitude":180},"id":{"authority":"OGC","code":"CRS84"}}"#;

#[cfg(test)]
mod test {
    use super::*;
    const EPSG_6318_PROJJSON: &str = r#"{"$schema": "https://proj.org/schemas/v0.4/projjson.schema.json","type": "GeographicCRS","name": "NAD83(2011)","datum": {"type": "GeodeticReferenceFrame","name": "NAD83 (National Spatial Reference System 2011)","ellipsoid": {"name": "GRS 1980","semi_major_axis": 6378137,"inverse_flattening": 298.257222101}},"coordinate_system": {"subtype": "ellipsoidal","axis": [{"name": "Geodetic latitude","abbreviation": "Lat","direction": "north","unit": "degree"},{"name": "Geodetic longitude","abbreviation": "Lon","direction": "east","unit": "degree"}]},"scope": "Horizontal component of 3D system.","area": "Puerto Rico - onshore and offshore. United States (USA) onshore and offshore - Alabama; Alaska; Arizona; Arkansas; California; Colorado; Connecticut; Delaware; Florida; Georgia; Idaho; Illinois; Indiana; Iowa; Kansas; Kentucky; Louisiana; Maine; Maryland; Massachusetts; Michigan; Minnesota; Mississippi; Missouri; Montana; Nebraska; Nevada; New Hampshire; New Jersey; New Mexico; New York; North Carolina; North Dakota; Ohio; Oklahoma; Oregon; Pennsylvania; Rhode Island; South Carolina; South Dakota; Tennessee; Texas; Utah; Vermont; Virginia; Washington; West Virginia; Wisconsin; Wyoming. US Virgin Islands - onshore and offshore.", "bbox": {"south_latitude": 14.92,"west_longitude": 167.65,"north_latitude": 74.71,"east_longitude": -63.88},"id": {"authority": "EPSG","code": 6318}}"#;

    #[test]
    fn deserialize() {
        let value_auth_code_4326: Value = serde_json::from_str(r#""EPSG:4326""#).unwrap();
        assert_eq!(deserialize_crs(&value_auth_code_4326).unwrap(), lnglat());

        let value_auth_code_crs84: Value = serde_json::from_str(r#""OGC:CRS84""#).unwrap();
        assert_eq!(deserialize_crs(&value_auth_code_crs84).unwrap(), lnglat());

        let value_projjson: Value = serde_json::from_str(OGC_CRS84_PROJJSON).unwrap();
        assert_eq!(deserialize_crs(&value_projjson).unwrap(), lnglat());

        let value_gross_unescaped = Value::String(OGC_CRS84_PROJJSON.to_string());
        assert_eq!(deserialize_crs(&value_gross_unescaped).unwrap(), lnglat());

        assert!(deserialize_crs(&Value::Null).unwrap().is_none());

        let value_unsupported: Value = serde_json::from_str("[]").unwrap();
        assert!(deserialize_crs(&value_unsupported).is_err());
    }

    #[test]
    fn crs_projjson() {
        let projjson = OGC_CRS84_PROJJSON.parse::<ProjJSON>().unwrap();
        assert_eq!(projjson.to_authority_code().unwrap().unwrap(), "OGC:CRS84");
        assert_eq!(projjson.srid().unwrap(), Some(4326));

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
            authority: "EPSG".to_string(),
            code: "4269".to_string(),
        };
        assert!(auth_code.crs_equals(&auth_code));
        assert!(!auth_code.crs_equals(LngLat::crs().unwrap().as_ref()));
        assert_eq!(auth_code.srid().unwrap(), Some(4269));

        assert_eq!(
            auth_code.to_authority_code().unwrap(),
            Some("EPSG:4269".to_string())
        );

        let json_value = auth_code.to_json();
        assert_eq!(json_value, "\"EPSG:4269\"");

        let auth_code_crs = AuthorityCode::crs("EPSG:4269".to_string());
        assert_eq!(auth_code_crs, auth_code_crs);
        assert_ne!(auth_code_crs, Crs::None);

        let auth_code_parsed: Result<Value, _> = serde_json::from_str(&json_value);
        assert!(AuthorityCode::is_authority_code(&auth_code_parsed.unwrap()));

        let value: Value = serde_json::from_str("\"EPSG:4269\"").unwrap();
        let new_crs = deserialize_crs(&value).unwrap().unwrap();
        assert_eq!(
            new_crs.to_authority_code().unwrap(),
            Some("EPSG:4269".to_string())
        );
        assert_eq!(new_crs.srid().unwrap(), Some(4269));

        // Ensure we can also just pass a code here
        let value: Value = serde_json::from_str("\"4269\"").unwrap();
        let new_crs = deserialize_crs(&value).unwrap();
        assert_eq!(
            new_crs.clone().unwrap().to_authority_code().unwrap(),
            Some("EPSG:4269".to_string())
        );
        assert_eq!(new_crs.unwrap().srid().unwrap(), Some(4269));

        let value: Value = serde_json::from_str("\"EPSG:4326\"").unwrap();
        let new_crs = deserialize_crs(&value).unwrap();
        assert_eq!(new_crs.clone().unwrap().srid().unwrap(), Some(4326));
    }
}
