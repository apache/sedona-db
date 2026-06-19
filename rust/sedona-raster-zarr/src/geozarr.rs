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
    /// `None` if the group declares no CRS through either the GeoZarr
    /// `proj:*` convention (`proj:wkt2` / `proj:projjson` / `proj:code`, or
    /// the legacy `proj:epsg`) or the CF / rioxarray convention (`crs_wkt` /
    /// `spatial_ref`). The CF convention also commonly puts these attributes
    /// on a separate grid-mapping variable rather than the group; that case
    /// is resolved in the loader (see `crs_from_cf_attributes`).
    pub crs: Option<String>,
    /// Affine transform, stored in GDAL GeoTransform order:
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`. The source
    /// `spatial:transform` attribute is in affine order `[a, b, c, d, e, f]`
    /// and is reordered on parse. `None` when no `spatial:transform`
    /// attribute is present.
    pub transform: Option<[f64; 6]>,
    /// Names of the spatial dimensions in the order the group declares them
    /// (typically `["y", "x"]`), from `spatial:dimensions` (or the legacy
    /// `spatial:dims`). `None` falls back to a 2-D default at construction
    /// time in the loader.
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
    // GeoZarr `proj:` convention precedence — wkt2 wins over projjson wins
    // over the authority `proj:code`. Match how downstream tools (e.g.
    // xarray + rioxarray) resolve multi-attribute groups: more specific
    // representations override authority codes.
    if let Some(v) = attrs.get("proj:wkt2") {
        return Ok(Some(json_value_to_string(v, "proj:wkt2")?));
    }
    if let Some(v) = attrs.get("proj:projjson") {
        return Ok(Some(json_value_to_string(v, "proj:projjson")?));
    }
    if let Some(v) = attrs.get("proj:code") {
        // Authority string, e.g. "EPSG:4326".
        return Ok(Some(json_value_to_string(v, "proj:code")?));
    }
    // Legacy: the proposal-era `proj:epsg` integer, superseded by
    // `proj:code`. Still read so older data keeps working.
    if let Some(v) = attrs.get("proj:epsg") {
        log::warn!(
            "Zarr group uses the legacy `proj:epsg` attribute; \
             prefer `proj:code` (e.g. \"EPSG:4326\")"
        );
        let code = v.as_i64().ok_or_else(|| {
            ArrowError::InvalidArgumentError("proj:epsg attribute must be an integer".into())
        })?;
        return Ok(Some(format!("EPSG:{code}")));
    }
    // No GeoZarr `proj:*` key; fall back to the CF / rioxarray convention,
    // which much of the public Zarr corpus (xarray + rioxarray writers) uses.
    crs_from_cf_attributes(attrs)
}

/// Read a CRS from the CF / rioxarray attribute convention.
///
/// rioxarray writes the CRS as `crs_wkt` (WKT2) and, redundantly, as a
/// `spatial_ref` attribute holding the same WKT; some writers (and group
/// roots) put an `authority:code` string in `spatial_ref` instead. `crs_wkt`
/// wins over `spatial_ref` when both are present.
///
/// These attributes live either on the group (handled by `parse_crs`) or on
/// a separate grid-mapping variable (handled by the loader), so this is
/// factored out for both callers.
///
/// PROJ.4 strings (`proj4` / `proj4_params`) are intentionally not read: the
/// CRS layer accepts authority codes, WKT and PROJJSON, but not PROJ.4, so
/// surfacing one would only produce a downstream parse error.
pub fn crs_from_cf_attributes(
    attrs: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<String>, ArrowError> {
    if let Some(v) = attrs.get("crs_wkt") {
        return Ok(Some(json_value_to_string(v, "crs_wkt")?));
    }
    if let Some(v) = attrs.get("spatial_ref") {
        return Ok(Some(json_value_to_string(v, "spatial_ref")?));
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
            "spatial:transform must have 6 elements (affine order [a, b, c, d, e, f]); got {}",
            arr.len()
        )));
    }
    // The `spatial:` convention stores the affine `[a, b, c, d, e, f]` with
    //   x = a*col + b*row + c
    //   y = d*col + e*row + f
    // We carry transforms internally in GDAL GeoTransform order
    // `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`, so reorder:
    //   origin_x = c, scale_x = a, skew_x = b,
    //   origin_y = f, skew_y = d, scale_y = e.
    let mut aff = [0f64; 6];
    for (i, v) in arr.iter().enumerate() {
        aff[i] = v.as_f64().ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!("spatial:transform[{i}] must be a number"))
        })?;
    }
    let [a, b, c, d, e, f] = aff;
    Ok(Some([c, a, b, f, d, e]))
}

fn parse_spatial_dims(
    attrs: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<Vec<String>>, ArrowError> {
    // `spatial:dimensions` is the current convention key; `spatial:dims` is
    // the proposal-era name, still read for older data.
    let (key, v) = match attrs.get("spatial:dimensions") {
        Some(v) => ("spatial:dimensions", v),
        None => match attrs.get("spatial:dims") {
            Some(v) => {
                log::warn!(
                    "Zarr group uses the legacy `spatial:dims` attribute; \
                     prefer `spatial:dimensions`"
                );
                ("spatial:dims", v)
            }
            None => return Ok(None),
        },
    };
    let arr = v.as_array().ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!("{key} attribute must be a JSON array of strings"))
    })?;
    let dims = arr
        .iter()
        .map(|e| {
            e.as_str().map(String::from).ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!("{key} entries must be strings"))
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
    fn proj_code_parses_to_string() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({"proj:code": "EPSG:4326"}))).unwrap();
        assert_eq!(g.crs.as_deref(), Some("EPSG:4326"));
    }

    #[test]
    fn legacy_epsg_code_parses_to_epsg_string() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({"proj:epsg": 4326}))).unwrap();
        assert_eq!(g.crs.as_deref(), Some("EPSG:4326"));
    }

    #[test]
    fn code_takes_precedence_over_legacy_epsg() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "proj:epsg": 3857,
            "proj:code": "EPSG:4326"
        })))
        .unwrap();
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
    fn cf_crs_wkt_parses_to_wkt_string() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "crs_wkt": "PROJCS[\"unknown\",GEOGCS[\"unknown\", ...]]"
        })))
        .unwrap();
        assert!(g.crs.as_deref().unwrap().starts_with("PROJCS"));
    }

    #[test]
    fn cf_spatial_ref_authority_code_parses() {
        // rioxarray / CarbonPlan group roots put an authority code here.
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial_ref": "EPSG:3031",
            "proj4_params": "+proj=stere +lat_0=-90"
        })))
        .unwrap();
        assert_eq!(g.crs.as_deref(), Some("EPSG:3031"));
    }

    #[test]
    fn cf_crs_wkt_takes_precedence_over_spatial_ref() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "crs_wkt": "PROJCS[\"a\"]",
            "spatial_ref": "EPSG:3031"
        })))
        .unwrap();
        assert_eq!(g.crs.as_deref(), Some("PROJCS[\"a\"]"));
    }

    #[test]
    fn geozarr_proj_code_takes_precedence_over_cf() {
        // A group declaring both conventions resolves via GeoZarr `proj:*`.
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "proj:code": "EPSG:4326",
            "crs_wkt": "PROJCS[\"other\"]"
        })))
        .unwrap();
        assert_eq!(g.crs.as_deref(), Some("EPSG:4326"));
    }

    #[test]
    fn cf_proj4_alone_is_not_read() {
        // PROJ.4 is not accepted by the CRS layer, so a group that declares
        // only `proj4` stays CRS-less rather than surfacing an unusable string.
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "proj4": "+proj=stere +lat_0=-90 +lat_ts=-71"
        })))
        .unwrap();
        assert!(g.crs.is_none());
    }

    #[test]
    fn transform_affine_reorders_to_gdal() {
        // Affine [a, b, c, d, e, f] = [1, 0, 100, 0, -1, 200]: north-up,
        // origin (100, 200), 1×-1 pixels. Stored internally as GDAL order
        // [origin_x, scale_x, skew_x, origin_y, skew_y, scale_y].
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:transform": [1.0, 0.0, 100.0, 0.0, -1.0, 200.0]
        })))
        .unwrap();
        assert_eq!(g.transform, Some([100.0, 1.0, 0.0, 200.0, 0.0, -1.0]));
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
    fn spatial_dimensions_parses_string_list() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:dimensions": ["y", "x"]
        })))
        .unwrap();
        assert_eq!(g.spatial_dims, Some(vec!["y".to_string(), "x".to_string()]));
    }

    #[test]
    fn legacy_spatial_dims_parses_string_list() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:dims": ["y", "x"]
        })))
        .unwrap();
        assert_eq!(g.spatial_dims, Some(vec!["y".to_string(), "x".to_string()]));
    }

    #[test]
    fn dimensions_take_precedence_over_legacy_dims() {
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:dims": ["lat", "lon"],
            "spatial:dimensions": ["y", "x"]
        })))
        .unwrap();
        assert_eq!(g.spatial_dims, Some(vec!["y".to_string(), "x".to_string()]));
    }

    #[test]
    fn spatial_dimensions_non_string_errors() {
        let err = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:dimensions": ["y", 1]
        })))
        .unwrap_err()
        .to_string();
        assert!(err.contains("strings"), "{err}");
    }
}
