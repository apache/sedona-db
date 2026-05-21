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

//! GeoZarr attribute parsing for CRS and affine transform.
//!
//! Reads the `proj:*` (CRS) and `spatial:*` (transform / spatial dim
//! mapping) attribute conventions from a Zarr group's attributes, mapping
//! them onto SedonaDB's per-raster `crs` and `transform` fields.
//!
//! Attributes live at the group level and are inherited by every array.
//! Per-array overrides are rejected by the group-constraint validator
//! (see `loader`).

use arrow_schema::ArrowError;

/// Per-group geo metadata distilled from `proj:*` / `spatial:*` attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupGeoMetadata {
    /// CRS string in PROJ or WKT format (whichever the group declared).
    /// `None` if no `proj:wkt2` / `proj:projjson` / `proj:epsg` attribute
    /// is present on the group.
    pub crs: Option<String>,
    /// Affine transform in GDAL GeoTransform order:
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`. `None`
    /// when no `spatial:transform` attribute is present.
    pub transform: Option<[f64; 6]>,
    /// Names of the spatial dimensions in the order the group declares
    /// them (typically `["y", "x"]`). `None` falls back to a 2-D default
    /// at construction time in the loader.
    pub spatial_dims: Option<Vec<String>>,
}

impl GroupGeoMetadata {
    /// Parse the group-level attributes object (the raw JSON map zarrs
    /// surfaces from a group) into a `GroupGeoMetadata`.
    ///
    /// Returns `Ok(default-empty)` when none of the conventional keys are
    /// present — geospatial metadata is optional; downstream fall-backs in
    /// the loader provide identity transforms when needed.
    pub fn from_attributes(
        attrs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, ArrowError> {
        let crs = parse_crs(attrs)?;
        let transform = parse_transform(attrs)?;
        let spatial_dims = parse_spatial_dims(attrs)?;
        Ok(Self {
            crs,
            transform,
            spatial_dims,
        })
    }
}

fn parse_crs(
    attrs: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<String>, ArrowError> {
    // GeoZarr `proj:` precedence — wkt2 wins over projjson wins over epsg
    // code. Match how downstream tools (e.g. xarray + rioxarray) resolve
    // multi-attribute groups: more specific representations override
    // numeric codes.
    if let Some(v) = attrs.get("proj:wkt2") {
        return Ok(Some(json_value_to_string(v, "proj:wkt2")?));
    }
    if let Some(v) = attrs.get("proj:projjson") {
        return Ok(Some(json_value_to_string(v, "proj:projjson")?));
    }
    if let Some(v) = attrs.get("proj:epsg") {
        let code = v.as_i64().ok_or_else(|| {
            ArrowError::InvalidArgumentError("proj:epsg attribute must be an integer".into())
        })?;
        return Ok(Some(format!("EPSG:{code}")));
    }
    Ok(None)
}

fn parse_transform(
    attrs: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<[f64; 6]>, ArrowError> {
    let Some(t) = attrs.get("spatial:transform") else {
        return Ok(None);
    };
    let arr = t.as_array().ok_or_else(|| {
        ArrowError::InvalidArgumentError("spatial:transform attribute must be a JSON array".into())
    })?;
    if arr.len() != 6 {
        return Err(ArrowError::InvalidArgumentError(format!(
            "spatial:transform must have 6 elements (GDAL GeoTransform order); got {}",
            arr.len()
        )));
    }
    let mut out = [0f64; 6];
    for (i, v) in arr.iter().enumerate() {
        out[i] = v.as_f64().ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("spatial:transform[{i}] must be a number"))
        })?;
    }
    Ok(Some(out))
}

fn parse_spatial_dims(
    attrs: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<Vec<String>>, ArrowError> {
    let Some(v) = attrs.get("spatial:dims") else {
        return Ok(None);
    };
    let arr = v.as_array().ok_or_else(|| {
        ArrowError::InvalidArgumentError(
            "spatial:dims attribute must be a JSON array of strings".into(),
        )
    })?;
    let dims = arr
        .iter()
        .map(|e| {
            e.as_str().map(String::from).ok_or_else(|| {
                ArrowError::InvalidArgumentError("spatial:dims entries must be strings".into())
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(dims))
}

fn json_value_to_string(v: &serde_json::Value, attr_name: &str) -> Result<String, ArrowError> {
    if let Some(s) = v.as_str() {
        return Ok(s.to_string());
    }
    if v.is_object() {
        return Ok(v.to_string());
    }
    Err(ArrowError::InvalidArgumentError(format!(
        "{attr_name} attribute must be a string or JSON object"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(json: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        json.as_object().unwrap().clone()
    }

    #[test]
    fn empty_attrs_parses_to_all_none() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({}))).unwrap();
        assert!(g.crs.is_none());
        assert!(g.transform.is_none());
        assert!(g.spatial_dims.is_none());
    }

    #[test]
    fn epsg_code_parses_to_epsg_string() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({"proj:epsg": 4326}))).unwrap();
        assert_eq!(g.crs.as_deref(), Some("EPSG:4326"));
    }

    #[test]
    fn wkt2_takes_precedence_over_epsg() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "proj:epsg": 4326,
            "proj:wkt2": "GEOGCRS[\"WGS 84\", ...]"
        })))
        .unwrap();
        assert!(g.crs.as_deref().unwrap().starts_with("GEOGCRS"));
    }

    #[test]
    fn projjson_object_serialises_to_string() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "proj:projjson": {"type": "GeographicCRS"}
        })))
        .unwrap();
        let crs = g.crs.unwrap();
        assert!(crs.contains("GeographicCRS"));
    }

    #[test]
    fn transform_parses_six_floats() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:transform": [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
        })))
        .unwrap();
        assert_eq!(g.transform, Some([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]));
    }

    #[test]
    fn transform_wrong_length_errors() {
        let err = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:transform": [0.0, 1.0, 0.0]
        })))
        .unwrap_err()
        .to_string();
        assert!(err.contains("6 elements"), "{err}");
    }

    #[test]
    fn spatial_dims_parses_string_list() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:dims": ["y", "x"]
        })))
        .unwrap();
        assert_eq!(g.spatial_dims, Some(vec!["y".to_string(), "x".to_string()]));
    }

    #[test]
    fn spatial_dims_non_string_errors() {
        let err = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:dims": ["y", 1]
        })))
        .unwrap_err()
        .to_string();
        assert!(err.contains("strings"), "{err}");
    }
}
