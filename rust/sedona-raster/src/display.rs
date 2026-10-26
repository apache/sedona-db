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

use std::fmt;

use sedona_schema::crs::deserialize_crs;

use crate::affine_transformation::to_world_coordinate;
use crate::traits::RasterRef;

/// Wrapper for formatting a raster reference as a human-readable string.
///
/// # Format
///
/// Non-skewed rasters:
/// ```text
/// [WxH/nbands] @ [xmin ymin xmax ymax] / CRS
/// ```
///
/// Skewed rasters (includes skew parameters):
/// ```text
/// [WxH/nbands] @ [xmin ymin xmax ymax] skew=(skew_x, skew_y) / CRS
/// ```
///
/// With outdb bands:
/// ```text
/// [WxH/nbands] @ [xmin ymin xmax ymax] / CRS <outdb>
/// ```
///
/// Without CRS:
/// ```text
/// [WxH/nbands] @ [xmin ymin xmax ymax]
/// ```
///
/// # Examples
///
/// ```text
/// [64x32/3] @ [43.08 79.07 171.08 143.07] / OGC:CRS84
/// [3x4/1] @ [3 2.4 3.84 4.24] skew=(0.06, 0.08) / EPSG:2193
/// [10x10/1] @ [0 0 10 10] / OGC:CRS84 <outdb>
/// ```
pub struct RasterDisplay<'a>(pub &'a dyn RasterRef);

impl fmt::Display for RasterDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let raster = self.0;

        let width = raster.width().unwrap();
        let height = raster.height().unwrap();
        let nbands = raster.num_bands();

        // Compute axis-aligned bounding box from 4 corners in world coordinates.
        // This handles both skewed and non-skewed rasters correctly.
        let w = width;
        let h = height;
        let (ulx, uly) = to_world_coordinate(raster, 0, 0);
        let (urx, ury) = to_world_coordinate(raster, w, 0);
        let (lrx, lry) = to_world_coordinate(raster, w, h);
        let (llx, lly) = to_world_coordinate(raster, 0, h);

        let xmin = ulx.min(urx).min(lrx).min(llx);
        let xmax = ulx.max(urx).max(lrx).max(llx);
        let ymin = uly.min(ury).min(lry).min(lly);
        let ymax = uly.max(ury).max(lry).max(lly);

        let transform = raster.transform();
        let skew_x = transform[2];
        let skew_y = transform[4];
        let has_skew = skew_x != 0.0 || skew_y != 0.0;

        let has_outdb = (0..raster.num_bands())
            .filter_map(|i| raster.band(i).ok())
            .any(|band| !band.is_indb());

        // Write: [WxH/nbands] @ [xmin ymin xmax ymax]
        write!(
            f,
            "[{width}x{height}/{nbands}] @ [{xmin} {ymin} {xmax} {ymax}]"
        )?;

        // Conditionally append skew info when the raster is rotated/skewed
        if has_skew {
            write!(f, " skew=({skew_x}, {skew_y})")?;
        }

        // Append CRS if present, abbreviated to its authority code where possible
        if let Some(crs) = raster.crs() {
            write!(f, " / ")?;
            write_abbreviated_crs(f, crs)?;
        }

        if has_outdb {
            write!(f, " <outdb>")?;
        }

        Ok(())
    }
}

/// Write a compact form of a raster's CRS string
///
/// CRSes that can be parsed are written using the CRS `Display` implementation
/// (i.e., as their authority code where one is available, such as the `id` of a
/// PROJJSON object or the trailing `ID[...]`/`AUTHORITY[...]` of a WKT string,
/// or `{...}` otherwise).
fn write_abbreviated_crs(f: &mut fmt::Formatter<'_>, crs: &str) -> fmt::Result {
    match deserialize_crs(crs) {
        Ok(Some(parsed)) => write!(f, "{parsed}"),
        // Strings we can't parse as a CRS are written as-is unless they look
        // like (invalid) JSON, which may be arbitrarily long
        _ if crs.starts_with('{') => write!(f, "{{...}}"),
        _ => write!(f, "{crs}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::RasterStructArray;
    use sedona_schema::crs::OGC_CRS84_PROJJSON;
    use sedona_testing::rasters::{build_in_db_raster, generate_test_rasters};

    #[test]
    fn display_non_skewed_raster() {
        // i=0: w=1, h=2, scale=(0.1, -0.2), skew=(0, 0), CRS=OGC:CRS84
        // Bounds: xmin=1, ymin=1.6, xmax=1.1, ymax=2
        let rasters = generate_test_rasters(1, None).unwrap();
        let raster_array = RasterStructArray::try_new(&rasters).unwrap();
        let raster = raster_array.get(0).unwrap();

        let display = format!("{}", RasterDisplay(&raster));
        assert_eq!(display, "[1x2/1] @ [1 1.6 1.1 2] / ogc:crs84");
    }

    #[test]
    fn display_skewed_raster() {
        // i=2: w=3, h=4, scale=(0.2, -0.4), skew=(0.06, 0.08), CRS=OGC:CRS84
        // Corners: (3,4), (3.6,4.24), (3.84,2.64), (3.24,2.4)
        // AABB: xmin=3, ymin=2.4, xmax=3.84, ymax=4.24
        let rasters = generate_test_rasters(3, None).unwrap();
        let raster_array = RasterStructArray::try_new(&rasters).unwrap();
        let raster = raster_array.get(2).unwrap();

        let display = format!("{}", RasterDisplay(&raster));
        assert_eq!(
            display,
            "[3x4/1] @ [3 2.4 3.84 4.24] skew=(0.06, 0.08) / ogc:crs84"
        );
    }

    #[test]
    fn display_write_to_fmt_write() {
        // Verify RasterDisplay works with any fmt::Write target (e.g., String)
        let rasters = generate_test_rasters(1, None).unwrap();
        let raster_array = RasterStructArray::try_new(&rasters).unwrap();
        let raster = raster_array.get(0).unwrap();

        let mut buf = String::new();
        use std::fmt::Write;
        write!(buf, "{}", RasterDisplay(&raster)).unwrap();
        assert_eq!(buf, "[1x2/1] @ [1 1.6 1.1 2] / ogc:crs84");
    }

    fn display_with_crs(crs: Option<&str>) -> String {
        let rasters = build_in_db_raster(2, 3, [10.0, 1.0, 0.0, 20.0, 0.0, -1.0], crs, &[]);
        let raster_array = RasterStructArray::try_new(&rasters).unwrap();
        let raster = raster_array.get(0).unwrap();
        format!("{}", RasterDisplay(&raster))
    }

    #[test]
    fn display_crs_authority_code() {
        assert_eq!(
            display_with_crs(Some("EPSG:32618")),
            "[2x3/0] @ [10 17 12 20] / epsg:32618"
        );
        assert_eq!(display_with_crs(None), "[2x3/0] @ [10 17 12 20]");
    }

    #[test]
    fn display_crs_projjson_abbreviated() {
        assert_eq!(
            display_with_crs(Some(OGC_CRS84_PROJJSON)),
            "[2x3/0] @ [10 17 12 20] / ogc:crs84"
        );

        let nad83 = r#"{"type":"GeographicCRS","name":"NAD83","datum":{"type":"GeodeticReferenceFrame","name":"NAD83","ellipsoid":{"name":"GRS 1980","semi_major_axis":6378137,"inverse_flattening":298.257222101}},"coordinate_system":{"subtype":"ellipsoidal","axis":[{"name":"Geodetic latitude","abbreviation":"Lat","direction":"north","unit":"degree"},{"name":"Geodetic longitude","abbreviation":"Lon","direction":"east","unit":"degree"}]},"id":{"authority":"EPSG","code":4269}}"#;
        assert_eq!(
            display_with_crs(Some(nad83)),
            "[2x3/0] @ [10 17 12 20] / epsg:4269"
        );

        // PROJJSON without an id can't be abbreviated
        let no_id = r#"{"type":"GeographicCRS","name":"Unknown"}"#;
        assert_eq!(
            display_with_crs(Some(no_id)),
            "[2x3/0] @ [10 17 12 20] / {...}"
        );
    }

    #[test]
    fn display_crs_wkt_abbreviated() {
        let wkt = r#"GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east],ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",4326]]"#;
        assert_eq!(
            display_with_crs(Some(wkt)),
            "[2x3/0] @ [10 17 12 20] / epsg:4326"
        );

        let wkt_no_id = r#"GEOGCRS["Unknown",DATUM["Unknown",ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],AXIS["latitude",north],AXIS["longitude",east],ANGLEUNIT["degree",0.0174532925199433]]"#;
        assert_eq!(
            display_with_crs(Some(wkt_no_id)),
            "[2x3/0] @ [10 17 12 20] / {...}"
        );
    }
}
