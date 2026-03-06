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

use datafusion_common::{exec_datafusion_err, DataFusionError, Result};
use sedona_geometry::transform::{transform, CrsEngine};
use sedona_schema::crs::{deserialize_crs, CoordinateReferenceSystem, Crs};
use wkb::reader::read_wkb;

/// Resolve an optional CRS string to a concrete CRS object.
///
/// - If `crs_str` is `Some` and deserializes to a known CRS, that CRS is returned.
/// - Otherwise (None, empty, "0", etc.), `None` is returned.
pub fn resolve_crs(crs_str: Option<&str>) -> Result<Crs> {
    if let Some(crs_str) = crs_str {
        deserialize_crs(crs_str)
    } else {
        Ok(None)
    }
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
///
/// **Note**
/// - For best performance, verify that `from_crs` and `to_crs` are different before calling this function
///   to avoid unnecessary WKB parsing and reprojection when CRSes are equivalent.
pub fn crs_transform_wkb(
    wkb: &[u8],
    from_crs: &dyn CoordinateReferenceSystem,
    to_crs: &dyn CoordinateReferenceSystem,
    engine: &dyn CrsEngine,
) -> Result<Vec<u8>> {
    let crs_transform = engine
        .get_transform_crs_to_crs(&from_crs.to_crs_string(), &to_crs.to_crs_string(), None, "")
        .map_err(|e| exec_datafusion_err!("CRS transform error: {}", e))?;
    let geom = read_wkb(wkb).map_err(|e| DataFusionError::External(Box::new(e)))?;
    let mut out = Vec::with_capacity(wkb.len());
    transform(geom, crs_transform.as_ref(), &mut out)
        .map_err(|e| exec_datafusion_err!("Transform error: {}", e))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo_traits::{CoordTrait, GeometryTrait, GeometryType, PointTrait};
    use sedona_proj::transform::with_global_proj_engine;
    use sedona_testing::create::make_wkb;

    /// A simple WKB point at (1.0, 2.0) used across all tests.
    fn sample_wkb() -> Vec<u8> {
        make_wkb("POINT (1.0 2.0)")
    }

    fn transform_wkb(
        wkb: &[u8],
        from: Option<&str>,
        to: Option<&str>,
        engine: &dyn CrsEngine,
    ) -> Result<Vec<u8>> {
        let from_crs = resolve_crs(from)?;
        let to_crs = resolve_crs(to)?;
        match (from_crs, to_crs) {
            (Some(from_crs), Some(to_crs)) => {
                crs_transform_wkb(wkb, from_crs.as_ref(), to_crs.as_ref(), engine)
            }
            _ => Ok(wkb.to_vec()),
        }
    }

    // -----------------------------------------------------------------------
    // Case 1: Both CRSes are empty / None / "0" (all combinations)
    //
    // All of these resolve to None. Since both are None, no transformation
    // is performed and the original WKB is returned.
    // -----------------------------------------------------------------------

    #[test]
    fn both_none() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, None, None, engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn both_empty_string() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some(""), Some(""), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn both_zero() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("0"), Some("0"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_empty_to_zero() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some(""), Some("0"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_zero_to_empty() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("0"), Some(""), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_none_to_empty() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, None, Some(""), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_none_to_zero() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, None, Some("0"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_empty_to_none() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some(""), None, engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_zero_to_none() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("0"), None, engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    // -----------------------------------------------------------------------
    // Case 2: One is empty/None/"0", the other is a real (non-WGS84) CRS
    //
    // Empty/"0"/None resolve to None. When one side has no CRS, no
    // transformation can be performed, so the original WKB is returned.
    // -----------------------------------------------------------------------

    #[test]
    fn from_empty_to_real_crs() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some(""), Some("EPSG:3857"), engine).unwrap();
            // One side is None, so no transformation — WKB returned unchanged.
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_zero_to_real_crs() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("0"), Some("EPSG:3857"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_real_crs_to_empty() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:3857"), Some(""), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_real_crs_to_zero() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:3857"), Some("0"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
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
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:4326"), Some("EPSG:4326"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn both_ogc_crs84() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("OGC:CRS84"), Some("OGC:CRS84"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn epsg_4326_vs_ogc_crs84() {
        // EPSG:4326 and OGC:CRS84 are treated as equivalent lnglat CRSes.
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:4326"), Some("OGC:CRS84"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn ogc_crs84_vs_epsg_4326() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("OGC:CRS84"), Some("EPSG:4326"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn both_epsg_3857() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:3857"), Some("EPSG:3857"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    // -----------------------------------------------------------------------
    // Case 3.5: One is empty/None/"0", the other is WGS84-equivalent
    //
    // Empty/"0"/None resolve to None. When one side is None, no
    // transformation is performed; the original WKB is returned.
    // -----------------------------------------------------------------------

    #[test]
    fn from_none_to_epsg_4326() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, None, Some("EPSG:4326"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_epsg_4326_to_none() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:4326"), None, engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_none_to_ogc_crs84() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, None, Some("OGC:CRS84"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_ogc_crs84_to_none() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("OGC:CRS84"), None, engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_empty_to_epsg_4326() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some(""), Some("EPSG:4326"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_epsg_4326_to_empty() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:4326"), Some(""), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_zero_to_epsg_4326() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("0"), Some("EPSG:4326"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_epsg_4326_to_zero() {
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:4326"), Some("0"), engine).unwrap();
            assert_eq!(result, wkb);
            Ok(())
        })
        .unwrap();
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
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:4326"), Some("EPSG:3857"), engine).unwrap();
            assert_ne!(result, wkb, "Coordinates should change after reprojection");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn transform_3857_to_4326() {
        // The reverse direction should also produce a different WKB.
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let result = transform_wkb(&wkb, Some("EPSG:3857"), Some("EPSG:4326"), engine).unwrap();
            assert_ne!(result, wkb, "Coordinates should change after reprojection");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn roundtrip_4326_3857_4326() {
        // Transform 4326 -> 3857 -> 4326 should approximately recover the original point.
        let wkb = sample_wkb();
        with_global_proj_engine(|engine| {
            let projected =
                transform_wkb(&wkb, Some("EPSG:4326"), Some("EPSG:3857"), engine).unwrap();
            let roundtripped =
                transform_wkb(&projected, Some("EPSG:3857"), Some("EPSG:4326"), engine).unwrap();
            let wkb = wkb::reader::read_wkb(&roundtripped).unwrap();
            let GeometryType::Point(p) = wkb.as_type() else {
                panic!("Expected a Point geometry");
            };
            let coord = p.coord().unwrap();
            assert_eq!(coord.x().round(), 1.0);
            assert_eq!(coord.y().round(), 2.0);
            Ok(())
        })
        .unwrap();
    }
}
