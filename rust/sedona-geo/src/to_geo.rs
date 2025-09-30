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
use datafusion_common::{error::Result, not_impl_err, DataFusionError};
use geo_traits::{
    to_geo::{
        ToGeoLineString, ToGeoMultiLineString, ToGeoMultiPoint, ToGeoMultiPolygon, ToGeoPoint,
        ToGeoPolygon,
    },
    GeometryCollectionTrait, GeometryTrait,
    GeometryType::*,
};
use geo_types::Geometry;
use sedona_functions::executor::{GenericExecutor, GeometryFactory};

/// A [GenericExecutor] that iterates over [Geometry] objects
pub type GeoTypesExecutor<'a, 'b> =
    GenericExecutor<'a, 'b, GeoTypesGeometryFactory, GeoTypesGeometryFactory>;

/// A [GeometryFactory] for use with the [GenericExecutor] that iterates over [Geometry]
/// objects.
#[derive(Default)]
pub struct GeoTypesGeometryFactory {}

impl GeometryFactory for GeoTypesGeometryFactory {
    type Geom<'a> = Geometry;

    fn try_from_wkb<'a>(&self, wkb_bytes: &'a [u8]) -> Result<Self::Geom<'a>> {
        let wkb =
            wkb::reader::read_wkb(wkb_bytes).map_err(|e| DataFusionError::External(Box::new(e)))?;
        item_to_geometry(wkb)
    }
}

/// Convert a [GeometryTrait] into a [Geometry]
///
/// This implementation avoid issues with some versions of the Rust compiler in release mode.
/// Note that [Geometry] does not support all valid [GeometryTrait] objects (notably: the
/// empty point and a multipoint with an empty child).
///
/// This implementation does not currently support arbitrarily recursive GeometryCollections
/// (the recursion for which is the reason some versions of the Rust compiler fail to
/// compile the version of this function in the geo-traits crate). This implementation limits
/// the recursion to 1 level deep (e.g., GEOMETRYCOLLECTION (...)).
pub fn item_to_geometry(geo: impl GeometryTrait<T = f64>) -> Result<Geometry> {
    if let Some(geo) = to_geometry(geo) {
        Ok(geo)
    } else {
        not_impl_err!(
            "geo kernel implementation on {}, {}, or {} not supported",
            "MULTIPOINT with EMPTY child",
            "POINT EMPTY",
            "GEOMETRYCOLLECTION"
        )
    }
}

// GeometryCollection causes issues because it has a recursive definition and won't work
// with cargo run --release. Thus, we need our own version of this that works around this
// problem by processing GeometryCollection using a free function instead of relying
// on trait resolver.
// See also https://github.com/geoarrow/geoarrow-rs/pull/956.
fn to_geometry(item: impl GeometryTrait<T = f64>) -> Option<Geometry> {
    match item.as_type() {
        Point(geom) => geom.try_to_point().map(Geometry::Point),
        LineString(geom) => Some(Geometry::LineString(geom.to_line_string())),
        Polygon(geom) => Some(Geometry::Polygon(geom.to_polygon())),
        MultiPoint(geom) => geom.try_to_multi_point().map(Geometry::MultiPoint),
        MultiLineString(geom) => Some(Geometry::MultiLineString(geom.to_multi_line_string())),
        MultiPolygon(geom) => Some(Geometry::MultiPolygon(geom.to_multi_polygon())),
        GeometryCollection(geom) => geometry_collection_to_geometry(geom),
        _ => None,
    }
}

fn geometry_collection_to_geometry<GC: GeometryCollectionTrait<T = f64>>(
    geom: &GC,
) -> Option<Geometry> {
    let geometries = geom
        .geometries()
        .filter_map(|child| match child.as_type() {
            Point(geom) => geom.try_to_point().map(Geometry::Point),
            LineString(geom) => Some(Geometry::LineString(geom.to_line_string())),
            Polygon(geom) => Some(Geometry::Polygon(geom.to_polygon())),
            MultiPoint(geom) => geom.try_to_multi_point().map(Geometry::MultiPoint),
            MultiLineString(geom) => Some(Geometry::MultiLineString(geom.to_multi_line_string())),
            MultiPolygon(geom) => Some(Geometry::MultiPolygon(geom.to_multi_polygon())),
            GeometryCollection(geom) => geometry_collection_to_geometry(geom),
            _ => None,
        })
        .collect::<Vec<_>>();

    // If any child conversions failed, also return None
    if geometries.len() != geom.num_geometries() {
        return None;
    }

    Some(Geometry::GeometryCollection(geo_types::GeometryCollection(
        geometries,
    )))
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ColumnarValue;
    use geo_traits::to_geo::ToGeoGeometry;
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_array_storage;
    use std::str::FromStr;
    use wkt::Wkt;

    use super::*;

    #[test]
    fn unsupported() {
        let unsupported = Wkt::from_str("POINT EMPTY").unwrap();
        let err = item_to_geometry(unsupported).unwrap_err();
        assert!(err.message().starts_with("geo kernel implementation"));

        let unsupported = Wkt::from_str("GEOMETRYCOLLECTION (POINT EMPTY)").unwrap();
        let err = item_to_geometry(unsupported).unwrap_err();
        assert!(err.message().starts_with("geo kernel implementation"));
    }

    #[rstest]
    fn custom_to_geom(
        #[values(
            "POINT (0 1)",
            "LINESTRING (1 2, 3 4)",
            "POLYGON ((0 0, 1 0, 0 1, 0 0))",
            "MULTIPOINT (1 2, 3 4)",
            "MULTILINESTRING ((1 2, 3 4))",
            "MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)))",
            "GEOMETRYCOLLECTION(POINT (1 2))",
            "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION(POINT (1 2)))"
        )]
        wkt_value: &str,
    ) {
        let geom = Wkt::<f64>::from_str(wkt_value).unwrap();
        assert_eq!(geom.to_geometry(), to_geometry(geom).unwrap())
    }

    #[test]
    fn test_executor() {
        let items = vec![
            Some("POINT (0 1)"),
            Some("LINESTRING (1 2, 3 4)"),
            Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
            Some("MULTIPOINT (1 2, 3 4)"),
            Some("MULTILINESTRING ((1 2, 3 4))"),
            Some("MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)))"),
            Some("GEOMETRYCOLLECTION(POINT (1 2))"),
            Some("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION(POINT (1 2)))"),
            None,
        ];
        let args = vec![ColumnarValue::Array(create_array_storage(
            &items,
            &WKB_GEOMETRY,
        ))];

        let expected_items = items
            .iter()
            .map(|maybe_item| {
                maybe_item.map(|item| Wkt::<f64>::from_str(item).unwrap().to_geometry())
            })
            .collect::<Vec<_>>();

        let mut actual_items = Vec::new();
        let executor = GeoTypesExecutor::new(&[WKB_GEOMETRY], &args);
        executor
            .execute_wkb_void(|geo| {
                actual_items.push(geo);
                Ok(())
            })
            .unwrap();
        assert_eq!(actual_items, expected_items)
    }
}
