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

use std::sync::{Arc, LazyLock};

use datafusion_common::{exec_datafusion_err, DataFusionError, Result};
use sedona_geometry::transform::{transform, CrsEngine};
use sedona_proj::st_transform::with_global_proj_engine;
use sedona_schema::crs::{deserialize_crs, lnglat, CoordinateReferenceSystem};
use wkb::reader::read_wkb;

/// Cached default CRS (WGS84 longitude/latitude). Initialized once on first access.
static DEFAULT_CRS: LazyLock<Arc<dyn CoordinateReferenceSystem + Send + Sync>> =
    LazyLock::new(|| lnglat().expect("lnglat() should always succeed"));

/// Run a closure with the active CRS engine, abstracting away the engine choice.
///
/// We keep this API engine-agnostic to allow future engines beyond PROJ; the
/// current implementation uses PROJ via `with_global_proj_engine`.
pub fn with_crs_engine<R, F: FnMut(&dyn CrsEngine) -> Result<R>>(mut func: F) -> Result<R> {
    with_global_proj_engine(|engine| func(engine))
}

/// Resolve an optional CRS string to a concrete CRS object.
///
/// - If `crs_str` is `Some` and deserializes to a known CRS, that CRS is returned.
/// - Otherwise (None, empty, "0", etc.), the default WGS84 lnglat CRS is returned.
pub fn resolve_crs(
    crs_str: Option<&str>,
) -> Result<Arc<dyn CoordinateReferenceSystem + Send + Sync>> {
    if let Some(crs_str) = crs_str {
        let crs = deserialize_crs(crs_str)?;
        Ok(crs.unwrap_or_else(|| DEFAULT_CRS.clone()))
    } else {
        Ok(DEFAULT_CRS.clone())
    }
}

/// Return a reference to the default CRS (WGS84 longitude/latitude).
///
/// This is a zero-cost accessor backed by a `LazyLock` static — no allocation
/// or atomic ref-count increment on each call.
pub fn default_crs() -> &'static (dyn CoordinateReferenceSystem + Send + Sync) {
    DEFAULT_CRS.as_ref()
}

/// Transform a geometry encoded as WKB from one CRS to another.
///
/// This is a utility used by raster/spatial functions to reproject a geometry
/// without leaking PROJ engine details into call sites.
///
/// **Behavior**
/// - If `from_crs` and `to_crs` are equal, returns the original WKB (clone) without decoding.
/// - Otherwise, builds a PROJ pipeline and transforms all coordinates.
///
/// **Errors**
/// - Returns an error if WKB parsing fails, PROJ cannot build the CRS-to-CRS transform,
///   or if the coordinate transformation itself fails.
pub fn crs_transform_wkb(
    wkb: &[u8],
    from_crs: &dyn CoordinateReferenceSystem,
    to_crs: &dyn CoordinateReferenceSystem,
) -> Result<Vec<u8>> {
    // Fast-path: if the CRS is identical, avoid WKB decoding + transform setup.
    if from_crs.crs_equals(to_crs) {
        return Ok(wkb.to_vec());
    }

    // Build a PROJ pipeline for CRS->CRS transformation using the default engine.
    with_crs_engine(|engine| {
        let crs_transform = engine
            .get_transform_crs_to_crs(&from_crs.to_crs_string(), &to_crs.to_crs_string(), None, "")
            .map_err(|e| exec_datafusion_err!("CRS transform error: {}", e))?;
        let geom = read_wkb(wkb).map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut out = Vec::with_capacity(wkb.len());
        transform(geom, crs_transform.as_ref(), &mut out)
            .map_err(|e| exec_datafusion_err!("Transform error: {}", e))?;
        Ok(out)
    })
}

/// Transform a coordinate from one CRS to another, using the default CRS engine.
pub fn crs_transform_coord(coord: (f64, f64), from_crs: &str, to_crs: &str) -> Result<(f64, f64)> {
    with_crs_engine(|engine| {
        let trans = engine
            .get_transform_crs_to_crs(from_crs, to_crs, None, "")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut coord = coord;
        trans
            .transform_coord(&mut coord)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(coord)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_traits::{CoordTrait, GeometryTrait, GeometryType, PointTrait};
    use sedona_testing::create::make_wkb;

    /// A simple WKB point at (1.0, 2.0) used across all tests.
    fn sample_wkb() -> Vec<u8> {
        make_wkb("POINT (1.0 2.0)")
    }

    fn transform(wkb: &[u8], from: Option<&str>, to: Option<&str>) -> Result<Vec<u8>> {
        let from_crs = resolve_crs(from)?;
        let to_crs = resolve_crs(to)?;
        crs_transform_wkb(wkb, from_crs.as_ref(), to_crs.as_ref())
    }

    // -----------------------------------------------------------------------
    // Case 1: Both CRSes are empty / None / "0" (all combinations)
    //
    // All of these resolve to the default lnglat CRS. Both sides are equal,
    // so the fast-path returns the original WKB unchanged.
    // -----------------------------------------------------------------------

    #[test]
    fn both_none() {
        let wkb = sample_wkb();
        let result = transform(&wkb, None, None).unwrap();
        // Both default to lnglat() which are equal, so fast-path returns original.
        assert_eq!(result, wkb);
    }

    #[test]
    fn both_empty_string() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some(""), Some("")).unwrap();
        // Both resolve to lnglat, which are equal — fast-path returns original.
        assert_eq!(result, wkb);
    }

    #[test]
    fn both_zero() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("0"), Some("0")).unwrap();
        // Both resolve to lnglat, which are equal — fast-path returns original.
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_empty_to_zero() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some(""), Some("0")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_zero_to_empty() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("0"), Some("")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_none_to_empty() {
        let wkb = sample_wkb();
        let result = transform(&wkb, None, Some("")).unwrap();
        // from defaults to lnglat, "" also resolves to lnglat — equal, no transform.
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_none_to_zero() {
        let wkb = sample_wkb();
        let result = transform(&wkb, None, Some("0")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_empty_to_none() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some(""), None).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_zero_to_none() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("0"), None).unwrap();
        assert_eq!(result, wkb);
    }

    // -----------------------------------------------------------------------
    // Case 2: One is empty/None/"0", the other is a real (non-WGS84) CRS
    //
    // Empty/"0" resolves to the default lnglat CRS (WGS84). When paired with
    // a different CRS like EPSG:3857, an actual transformation occurs.
    // -----------------------------------------------------------------------

    #[test]
    fn from_empty_to_real_crs() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some(""), Some("EPSG:3857")).unwrap();
        assert_ne!(
            result, wkb,
            "Empty CRS defaults to lnglat; transform to 3857 should change coords"
        );
    }

    #[test]
    fn from_zero_to_real_crs() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("0"), Some("EPSG:3857")).unwrap();
        assert_ne!(
            result, wkb,
            "\"0\" CRS defaults to lnglat; transform to 3857 should change coords"
        );
    }

    #[test]
    fn from_real_crs_to_empty() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:3857"), Some("")).unwrap();
        assert_ne!(
            result, wkb,
            "Empty CRS defaults to lnglat; transform from 3857 should change coords"
        );
    }

    #[test]
    fn from_real_crs_to_zero() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:3857"), Some("0")).unwrap();
        assert_ne!(
            result, wkb,
            "\"0\" CRS defaults to lnglat; transform from 3857 should change coords"
        );
    }

    // -----------------------------------------------------------------------
    // Case 3: Both are real CRSes that are equivalent
    //
    // The fast-path (crs_equals) should detect equality and return the
    // original WKB without invoking the PROJ pipeline.
    // -----------------------------------------------------------------------

    #[test]
    fn both_epsg_4326() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:4326"), Some("EPSG:4326")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn both_ogc_crs84() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("OGC:CRS84"), Some("OGC:CRS84")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn epsg_4326_vs_ogc_crs84() {
        // EPSG:4326 and OGC:CRS84 are treated as equivalent lnglat CRSes.
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:4326"), Some("OGC:CRS84")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn ogc_crs84_vs_epsg_4326() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("OGC:CRS84"), Some("EPSG:4326")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn both_epsg_3857() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:3857"), Some("EPSG:3857")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn both_none_defaults_to_lnglat_equal() {
        // Both None -> both default to lnglat(). They are equal, so fast-path.
        let wkb = sample_wkb();
        let result = transform(&wkb, None, None).unwrap();
        assert_eq!(result, wkb);
    }

    // -----------------------------------------------------------------------
    // Case 3.5: One is empty/None/"0", the other is WGS84-equivalent
    //
    // Empty/"0"/None all resolve to lnglat (WGS84). When paired with an
    // explicit WGS84-equivalent CRS (EPSG:4326, OGC:CRS84), both sides
    // are equal, so the fast-path returns the original WKB unchanged.
    // -----------------------------------------------------------------------

    #[test]
    fn from_none_to_epsg_4326() {
        // from=None -> lnglat(), to="EPSG:4326" -> lnglat(). Both are Some and equal.
        let wkb = sample_wkb();
        let result = transform(&wkb, None, Some("EPSG:4326")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_epsg_4326_to_none() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:4326"), None).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_none_to_ogc_crs84() {
        let wkb = sample_wkb();
        let result = transform(&wkb, None, Some("OGC:CRS84")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_ogc_crs84_to_none() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("OGC:CRS84"), None).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_empty_to_epsg_4326() {
        // from="" resolves to lnglat, to="EPSG:4326" also resolves to lnglat. Equal, no transform.
        let wkb = sample_wkb();
        let result = transform(&wkb, Some(""), Some("EPSG:4326")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_epsg_4326_to_empty() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:4326"), Some("")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_zero_to_epsg_4326() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("0"), Some("EPSG:4326")).unwrap();
        assert_eq!(result, wkb);
    }

    #[test]
    fn from_epsg_4326_to_zero() {
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:4326"), Some("0")).unwrap();
        assert_eq!(result, wkb);
    }

    // -----------------------------------------------------------------------
    // Case 4: Both are real CRSes that are NOT equivalent
    //
    // An actual coordinate transformation should occur. The output WKB must
    // differ from the input.
    // -----------------------------------------------------------------------

    #[test]
    fn transform_4326_to_3857() {
        // EPSG:4326 (WGS84) -> EPSG:3857 (Web Mercator) should change the coordinates.
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:4326"), Some("EPSG:3857")).unwrap();
        assert_ne!(result, wkb, "Coordinates should change after reprojection");
    }

    #[test]
    fn transform_3857_to_4326() {
        // The reverse direction should also produce a different WKB.
        let wkb = sample_wkb();
        let result = transform(&wkb, Some("EPSG:3857"), Some("EPSG:4326")).unwrap();
        assert_ne!(result, wkb, "Coordinates should change after reprojection");
    }

    #[test]
    fn transform_none_to_3857() {
        // from=None defaults to lnglat(). This is equivalent to 4326->3857.
        let wkb = sample_wkb();
        let result_default = transform(&wkb, None, Some("EPSG:3857")).unwrap();
        let result_explicit = transform(&wkb, Some("EPSG:4326"), Some("EPSG:3857")).unwrap();
        assert_ne!(result_default, wkb);
        assert_eq!(
            result_default, result_explicit,
            "None source should behave identically to explicit EPSG:4326"
        );
    }

    #[test]
    fn transform_3857_to_none() {
        // to=None defaults to lnglat(). This is equivalent to 3857->4326.
        let wkb = sample_wkb();
        let result_default = transform(&wkb, Some("EPSG:3857"), None).unwrap();
        let result_explicit = transform(&wkb, Some("EPSG:3857"), Some("EPSG:4326")).unwrap();
        assert_ne!(result_default, wkb);
        assert_eq!(
            result_default, result_explicit,
            "None target should behave identically to explicit EPSG:4326"
        );
    }

    #[test]
    fn roundtrip_4326_3857_4326() {
        // Transform 4326 -> 3857 -> 4326 should approximately recover the original point.
        let wkb = sample_wkb();
        let projected = transform(&wkb, Some("EPSG:4326"), Some("EPSG:3857")).unwrap();
        let roundtripped = transform(&projected, Some("EPSG:3857"), Some("EPSG:4326")).unwrap();
        let wkb = wkb::reader::read_wkb(&roundtripped).unwrap();
        let GeometryType::Point(p) = wkb.as_type() else {
            panic!("Expected a Point geometry");
        };
        let coord = p.coord().unwrap();
        assert_eq!(coord.x().round(), 1.0);
        assert_eq!(coord.y().round(), 2.0);
    }
}
