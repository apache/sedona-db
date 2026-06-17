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
    /// `None` if no `proj:wkt2` / `proj:projjson` / `proj:code` (or the
    /// legacy `proj:epsg`) attribute is present on the group.
    pub crs: Option<String>,
    /// Affine transform, stored in GDAL GeoTransform order:
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`. Parsed from the
    /// `spatial:transform` attribute (affine order `[a, b, c, d, e, f]`,
    /// reordered on parse). `None` when the group declares no explicit
    /// transform; the loader then derives one from `bbox` (below) or the
    /// spatial coordinate arrays, where the array shape is in hand.
    pub transform: Option<[f64; 6]>,
    /// Names of the spatial dimensions in the order the group declares them
    /// (typically `["y", "x"]`), from `spatial:dimensions` (or the legacy
    /// `spatial:dims`). `None` falls back to a 2-D default at construction
    /// time in the loader.
    pub spatial_dims: Option<Vec<String>>,
    /// Spatial bounding box `[xmin, ymin, xmax, ymax]` from `spatial:bbox`, used
    /// to derive a transform when no explicit `spatial:transform` is present.
    /// The grid shape is read from the array itself (not a separate
    /// `spatial:shape` attribute, which could drift from the real shape), so the
    /// loader supplies it to [`derive_transform_from_bbox`].
    pub bbox: Option<[f64; 4]>,
    /// `spatial:registration` — `"pixel"` or `"node"`; `None` defaults to
    /// `"pixel"`. Governs how `bbox` maps to the grid (outer edge vs. cell
    /// centers).
    pub registration: Option<String>,
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
        let bbox = parse_bbox(attrs);
        let registration = parse_registration(attrs)?;
        Ok(Self {
            crs,
            transform,
            spatial_dims,
            bbox,
            registration,
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
    Ok(None)
}

fn parse_transform(
    attrs: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<[f64; 6]>, ArrowError> {
    let Some(t) = attrs.get("spatial:transform") else {
        // No explicit transform; the loader derives one from `bbox` or the
        // coordinate arrays, where the array's own shape is available.
        return Ok(None);
    };
    let arr = parse_f64_array(t, "spatial:transform")?;
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
    let [a, b, c, d, e, f] = [arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]];
    Ok(Some([c, a, b, f, d, e]))
}

/// Parse the optional `spatial:bbox` attribute into `[xmin, ymin, xmax, ymax]`.
/// `None` when absent. The grid shape is *not* read from `spatial:shape`; the
/// loader supplies the array's own shape to [`derive_transform_from_bbox`].
///
/// A malformed `spatial:bbox` (not a 4-element numeric array) is treated as
/// absent — it warns and returns `None` rather than failing the load, so the
/// loader can fall back to coordinate arrays, mirroring how a bad coordinate
/// array is handled.
fn parse_bbox(attrs: &serde_json::Map<String, serde_json::Value>) -> Option<[f64; 4]> {
    let v = attrs.get("spatial:bbox")?;
    let bbox: Option<Vec<f64>> = v
        .as_array()
        .and_then(|a| a.iter().map(|e| e.as_f64()).collect());
    match bbox.as_deref() {
        Some([xmin, ymin, xmax, ymax]) => Some([*xmin, *ymin, *xmax, *ymax]),
        _ => {
            log::warn!(
                "Zarr group has a malformed `spatial:bbox` (expected a 4-element numeric array \
                 [xmin, ymin, xmax, ymax]); ignoring it"
            );
            None
        }
    }
}

/// Parse the optional `spatial:registration` attribute (a string). `Ok(None)`
/// when absent; [`derive_transform_from_bbox`] then defaults to `"pixel"`.
fn parse_registration(
    attrs: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<String>, ArrowError> {
    match attrs.get("spatial:registration") {
        Some(v) => Ok(Some(
            v.as_str()
                .ok_or_else(|| {
                    ArrowError::InvalidArgumentError("spatial:registration must be a string".into())
                })?
                .to_string(),
        )),
        None => Ok(None),
    }
}

/// Derive a GDAL-order transform from a `spatial:bbox` and the grid's `height`
/// and `width` — the array's *own* spatial dimensions, not a `spatial:shape`
/// attribute (which could drift from the real shape).
///
/// `bbox` is `[xmin, ymin, xmax, ymax]`. `registration` (default `"pixel"`) sets
/// how the bbox relates to the grid, per the GeoZarr spatial convention
/// (<https://github.com/zarr-conventions/spatial>): `"pixel"` means the bbox is
/// the grid's outer edge, spanning all `N` cells (`scale = extent / N`) with the
/// top-left corner on the bbox edge `(xmin, ymax)`; `"node"` means the bbox
/// endpoints are the *centers* of the border cells, so `N` centers span `N - 1`
/// intervals (`scale = extent / (N - 1)`) and the corner sits half a cell outside
/// the bbox. `scale_y` is negative (rows increase downward).
pub(crate) fn derive_transform_from_bbox(
    bbox: [f64; 4],
    height: u64,
    width: u64,
    registration: Option<&str>,
) -> Result<[f64; 6], ArrowError> {
    let [xmin, ymin, xmax, ymax] = bbox;
    // Reject a degenerate or inverted bbox: a zero span gives a non-invertible
    // (zero-scale) transform and a negative span isn't north-up. Mirrors the
    // coordinate-array path's strictness about a wrong scale.
    if !(xmax > xmin && ymax > ymin) {
        return Err(ArrowError::InvalidArgumentError(format!(
            "spatial:bbox must have xmin < xmax and ymin < ymax; got [{xmin}, {ymin}, {xmax}, {ymax}]"
        )));
    }
    // The array's spatial dims; a real raster axis is at least 1.
    if height == 0 || width == 0 {
        return Err(ArrowError::InvalidArgumentError(
            "raster spatial dimensions must be non-zero to derive a transform from spatial:bbox"
                .into(),
        ));
    }
    let (height, width) = (height as f64, width as f64);

    // cell-area registration is the conventional default
    let registration = registration.unwrap_or("pixel");
    // scale_y is negative throughout: rows increase downward.
    let (scale_x, scale_y, origin_x, origin_y) = match registration {
        // Pixel-registered: the bbox is the grid's outer edge, spanning all N
        // cells, so the top-left corner is the bbox edge.
        "pixel" => {
            let scale_x = (xmax - xmin) / width;
            let scale_y = (ymin - ymax) / height;
            (scale_x, scale_y, xmin, ymax)
        }
        // Node-registered: the bbox endpoints are the *centers* of the border
        // cells, so N centers span N-1 intervals and the footprint extends half
        // a cell beyond — the corner sits half a cell outside the bbox.
        "node" => {
            if width < 2.0 || height < 2.0 {
                return Err(ArrowError::InvalidArgumentError(
                    "node-registered grid must be at least 2 cells in each spatial dimension"
                        .into(),
                ));
            }
            let scale_x = (xmax - xmin) / (width - 1.0);
            let scale_y = (ymin - ymax) / (height - 1.0);
            (scale_x, scale_y, xmin - scale_x / 2.0, ymax - scale_y / 2.0)
        }
        other => {
            return Err(ArrowError::InvalidArgumentError(format!(
                "spatial:registration must be \"pixel\" or \"node\"; got {other:?}"
            )))
        }
    };

    // GDAL GeoTransform order; axis-aligned, so the skews are 0.
    Ok([origin_x, scale_x, 0.0, origin_y, 0.0, scale_y])
}

/// Parse a JSON value as an array of `f64`. Caller validates the length.
fn parse_f64_array(v: &serde_json::Value, attr_name: &str) -> Result<Vec<f64>, ArrowError> {
    let arr = v.as_array().ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!("{attr_name} attribute must be a JSON array"))
    })?;
    arr.iter()
        .enumerate()
        .map(|(i, e)| {
            e.as_f64().ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!("{attr_name}[{i}] must be a number"))
            })
        })
        .collect()
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
        assert!(g.bbox.is_none());
        assert!(g.registration.is_none());
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
    fn bbox_pixel_registration_derives_transform() {
        // The worked example from docs/research/zarr-geozarr.md: a bbox over a
        // 1000x1000 grid with "pixel" registration derives the same transform
        // the dataset also declares explicitly ([10, 0, 600000, 0, -10, 5700000]
        // affine). Height/width are the array's own dims, passed in by the loader.
        let t = derive_transform_from_bbox(
            [600000.0, 5690000.0, 610000.0, 5700000.0],
            1000,
            1000,
            Some("pixel"),
        )
        .unwrap();
        assert_eq!(t, [600000.0, 10.0, 0.0, 5700000.0, 0.0, -10.0]);
    }

    #[test]
    fn bbox_registration_defaults_to_pixel() {
        // `None` registration behaves as cell-area ("pixel").
        let t = derive_transform_from_bbox(
            [600000.0, 5690000.0, 610000.0, 5700000.0],
            1000,
            1000,
            None,
        )
        .unwrap();
        assert_eq!(t, [600000.0, 10.0, 0.0, 5700000.0, 0.0, -10.0]);
    }

    #[test]
    fn bbox_node_registration_uses_n_minus_1_intervals() {
        // "node": the bbox endpoints are cell centers, so 11 centers span 10
        // intervals -> scale = 100 / 10 = 10, and the corner sits half a cell
        // (5) outside the bbox: origin (-5, 105).
        let t = derive_transform_from_bbox([0.0, 0.0, 100.0, 100.0], 11, 11, Some("node")).unwrap();
        assert_eq!(t, [-5.0, 10.0, 0.0, 105.0, 0.0, -10.0]);
    }

    #[test]
    fn node_registration_requires_at_least_two_cells() {
        // A single-cell node grid can't define a spacing from its bbox alone.
        let err = derive_transform_from_bbox([0.0, 0.0, 100.0, 100.0], 1, 1, Some("node"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least 2"), "{err}");
    }

    #[test]
    fn unknown_registration_errors() {
        let err = derive_transform_from_bbox([0.0, 0.0, 100.0, 100.0], 10, 10, Some("corner"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("pixel") && err.contains("node"), "{err}");
    }

    #[test]
    fn degenerate_or_inverted_bbox_errors() {
        // Zero x-span (non-invertible) and inverted y (not north-up) are both
        // rejected, matching the coordinate-array path's strictness about a
        // wrong scale.
        for bbox in [
            [10.0, 0.0, 10.0, 5.0], // xmin == xmax
            [0.0, 5.0, 5.0, 0.0],   // ymax < ymin
        ] {
            let err = derive_transform_from_bbox(bbox, 10, 10, None)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("xmin < xmax") || err.contains("ymin < ymax"),
                "{err}"
            );
        }
    }

    #[test]
    fn explicit_transform_surfaced_alongside_bbox() {
        // from_attributes surfaces both the explicit transform and the bbox; the
        // loader prefers the explicit transform (see the fallback chain there).
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:transform": [1.0, 0.0, 100.0, 0.0, -1.0, 200.0],
            "spatial:bbox": [600000.0, 5690000.0, 610000.0, 5700000.0]
        })))
        .unwrap();
        assert_eq!(g.transform, Some([100.0, 1.0, 0.0, 200.0, 0.0, -1.0]));
        assert_eq!(g.bbox, Some([600000.0, 5690000.0, 610000.0, 5700000.0]));
    }

    #[test]
    fn bbox_parsed_without_shape() {
        // No `spatial:shape` is needed: the bbox and registration are surfaced
        // and the loader supplies the array's shape. transform stays None (the
        // loader derives it).
        let g = GroupGeoMetadata::from_attributes(&map(json!({
            "spatial:bbox": [600000.0, 5690000.0, 610000.0, 5700000.0],
            "spatial:registration": "node"
        })))
        .unwrap();
        assert!(g.transform.is_none());
        assert_eq!(g.bbox, Some([600000.0, 5690000.0, 610000.0, 5700000.0]));
        assert_eq!(g.registration.as_deref(), Some("node"));
    }

    #[test]
    fn malformed_bbox_is_ignored() {
        // A malformed spatial:bbox (wrong length, non-array, or a non-numeric
        // element) is treated as absent rather than failing the load — the
        // loader falls back to coordinate arrays. (It also logs a warning.)
        for bad in [
            json!([1.0, 2.0, 3.0]),      // wrong length
            json!("not-an-array"),       // not an array
            json!([1.0, 2.0, "x", 4.0]), // non-numeric element
        ] {
            let g =
                GroupGeoMetadata::from_attributes(&map(json!({ "spatial:bbox": bad }))).unwrap();
            assert!(g.bbox.is_none(), "expected bbox ignored for {bad:?}");
        }
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
