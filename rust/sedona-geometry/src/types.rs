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
use std::{fmt::Display, str::FromStr};

use geo_traits::{Dimensions, GeometryTrait};
use serde::{Deserialize, Serialize};
use serde_with::{DeserializeFromStr, SerializeDisplay};

use crate::error::SedonaGeometryError;

/// Geometry types
///
/// An enumerator for the set of natively supported geometry types without
/// considering [Dimensions]. See [GeometryTypeAndDimensions] for a struct to
/// track both.
///
/// This is named GeometryTypeId such that it does not conflict with
/// [geo_traits::GeometryType].
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Hash, Clone, Copy)]
pub enum GeometryTypeId {
    /// Unknown or mixed geometry type
    Geometry,
    /// Point geometry type
    Point,
    /// LineString geometry type
    LineString,
    /// Polygon geometry type
    Polygon,
    /// MultiPoint geometry type
    MultiPoint,
    /// MultiLineString geometry type
    MultiLineString,
    /// MultiPolygon geometry type
    MultiPolygon,
    /// GeometryCollection geometry type
    GeometryCollection,
}

impl GeometryTypeId {
    /// Construct a geometry type from a WKB type integer
    ///
    /// Parses the geometry type (not dimension) component of a WKB type code (e.g.,
    /// 1 for Point...7 for GeometryCollection).
    pub fn try_from_wkb_id(wkb_id: u32) -> Result<Self, SedonaGeometryError> {
        match wkb_id {
            0 => Ok(Self::Geometry),
            1 => Ok(Self::Point),
            2 => Ok(Self::LineString),
            3 => Ok(Self::Polygon),
            4 => Ok(Self::MultiPoint),
            5 => Ok(Self::MultiLineString),
            6 => Ok(Self::MultiPolygon),
            7 => Ok(Self::GeometryCollection),
            _ => Err(SedonaGeometryError::Invalid(format!(
                "Unknown geometry type identifier {wkb_id}"
            ))),
        }
    }

    /// WKB integer identifier
    ///
    /// The GeometryType portion of the WKB identifier (e.g., 1 for Point...7 for GeometryCollection).
    pub fn wkb_id(&self) -> u32 {
        match self {
            Self::Geometry => 0,
            Self::Point => 1,
            Self::LineString => 2,
            Self::Polygon => 3,
            Self::MultiPoint => 4,
            Self::MultiLineString => 5,
            Self::MultiPolygon => 6,
            Self::GeometryCollection => 7,
        }
    }

    /// GeoJSON/GeoParquet string identifier
    ///
    /// The identifier used by GeoJSON and GeoParquet to refer to this geometry type.
    /// Use [FromStr] to parse such a string back into a GeometryTypeId.
    pub fn geojson_id(&self) -> &'static str {
        match self {
            Self::Geometry => "Geometry",
            Self::Point => "Point",
            Self::LineString => "LineString",
            Self::Polygon => "Polygon",
            Self::MultiPoint => "MultiPoint",
            Self::MultiLineString => "MultiLineString",
            Self::MultiPolygon => "MultiPolygon",
            Self::GeometryCollection => "GeometryCollection",
        }
    }
}

impl FromStr for GeometryTypeId {
    type Err = SedonaGeometryError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value_lower = value.to_ascii_lowercase();
        match value_lower.as_str() {
            "geometry" => Ok(Self::Geometry),
            "point" => Ok(Self::Point),
            "linestring" => Ok(Self::LineString),
            "polygon" => Ok(Self::Polygon),
            "multipoint" => Ok(Self::MultiPoint),
            "multilinestring" => Ok(Self::MultiLineString),
            "multipolygon" => Ok(Self::MultiPolygon),
            "geometrycollection" => Ok(Self::GeometryCollection),
            _ => Err(SedonaGeometryError::Invalid(format!(
                "Invalid geometry type string: '{value}'"
            ))),
        }
    }
}

/// Geometry type and dimension
///
/// Combines a [GeometryTypeId] with [Dimensions] to handle cases where these concepts
/// are represented together (e.g., GeoParquet geometry types, WKB geometry type integers,
/// Parquet GeoStatistics). For sanity's sake, this combined concept is also frequently
/// just called "geometry type".
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, SerializeDisplay, DeserializeFromStr)]
pub struct GeometryTypeAndDimensions {
    geometry_type: GeometryTypeId,
    dimensions: Dimensions,
}

impl GeometryTypeAndDimensions {
    /// Create from [GeometryTypeId] and [Dimensions]
    pub fn new(geometry_type: GeometryTypeId, dimensions: Dimensions) -> Self {
        Self {
            geometry_type,
            dimensions,
        }
    }

    /// Create from [GeometryTrait]
    pub fn try_from_geom(geom: &impl GeometryTrait) -> Result<Self, SedonaGeometryError> {
        let dimensions = geom.dim();
        let geometry_type = match geom.as_type() {
            geo_traits::GeometryType::Point(_) => GeometryTypeId::Point,
            geo_traits::GeometryType::LineString(_) => GeometryTypeId::LineString,
            geo_traits::GeometryType::Polygon(_) => GeometryTypeId::Polygon,
            geo_traits::GeometryType::MultiPoint(_) => GeometryTypeId::MultiPoint,
            geo_traits::GeometryType::MultiLineString(_) => GeometryTypeId::MultiLineString,
            geo_traits::GeometryType::MultiPolygon(_) => GeometryTypeId::MultiPolygon,
            geo_traits::GeometryType::GeometryCollection(_) => GeometryTypeId::GeometryCollection,
            _ => {
                return Err(SedonaGeometryError::Invalid(
                    "Unsupported geometry type".to_string(),
                ))
            }
        };

        Ok(Self::new(geometry_type, dimensions))
    }

    /// The [GeometryTypeId]
    pub fn geometry_type(&self) -> GeometryTypeId {
        self.geometry_type
    }

    /// The [Dimensions]
    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    /// Create from an ISO WKB integer identifier (e.g., 1001 for Point Z)
    pub fn try_from_wkb_id(wkb_id: u32) -> Result<Self, SedonaGeometryError> {
        let dimensions = match wkb_id / 1000 {
            0 => Dimensions::Xy,
            1 => Dimensions::Xyz,
            2 => Dimensions::Xym,
            3 => Dimensions::Xyzm,
            _ => {
                return Err(SedonaGeometryError::Invalid(format!(
                    "Unknown dimensions in ISO WKB geometry type: {wkb_id}"
                )))
            }
        };

        let geometry_type = GeometryTypeId::try_from_wkb_id(wkb_id % 1000)?;
        Ok(Self {
            geometry_type,
            dimensions,
        })
    }

    /// ISO WKB integer identifier (e.g., 1001 for Point Z)
    pub fn wkb_id(&self) -> u32 {
        let dimensions_id = match self.dimensions {
            Dimensions::Xy => 0,
            Dimensions::Xyz => 1000,
            Dimensions::Xym => 2000,
            Dimensions::Xyzm => 3000,
            Dimensions::Unknown(n) => match n {
                2 => 0,
                3 => 1000,
                4 => 3000,
                _ => {
                    // Avoid a panic unless in debug mode
                    debug_assert!(false, "Unknown dimensions in GeometryTypeAndDimensions");
                    0
                }
            },
        };

        dimensions_id + self.geometry_type.wkb_id()
    }

    /// GeoJSON/GeoParquet identifier (e.g., Point Z, LineString, Polygon ZM)
    pub fn geojson_id(&self) -> String {
        self.to_string()
    }
}

impl From<(GeometryTypeId, Dimensions)> for GeometryTypeAndDimensions {
    fn from(value: (GeometryTypeId, Dimensions)) -> Self {
        Self {
            geometry_type: value.0,
            dimensions: value.1,
        }
    }
}

impl Display for GeometryTypeAndDimensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let suffix = match self.dimensions {
            Dimensions::Xy => "",
            Dimensions::Xyz => " Z",
            Dimensions::Xym => " M",
            Dimensions::Xyzm => " ZM",
            Dimensions::Unknown(_) => " Unknown",
        };

        f.write_str(self.geometry_type.geojson_id())?;
        f.write_str(suffix)
    }
}

impl FromStr for GeometryTypeAndDimensions {
    type Err = SedonaGeometryError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut parts = value.split_ascii_whitespace();
        let geometry_type = match parts.next() {
            Some(maybe_geometry_type) => GeometryTypeId::from_str(maybe_geometry_type)?,
            None => {
                return Err(SedonaGeometryError::Invalid(format!(
                    "Invalid geometry type string: '{value}'"
                )))
            }
        };

        let dimensions = match parts.next() {
            Some(maybe_dimensions) => match maybe_dimensions {
                "z" | "Z" => Dimensions::Xyz,
                "m" | "M" => Dimensions::Xym,
                "zm" | "ZM" => Dimensions::Xyzm,
                _ => {
                    return Err(SedonaGeometryError::Invalid(format!(
                        "invalid geometry type string: '{value}'"
                    )))
                }
            },
            None => Dimensions::Xy,
        };

        if parts.next().is_some() {
            return Err(SedonaGeometryError::Invalid(format!(
                "invalid geometry type string: '{value}'"
            )));
        }

        Ok(Self {
            geometry_type,
            dimensions,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use rstest::rstest;
    use Dimensions::*;
    use GeometryTypeId::*;

    #[rstest]
    fn geometry_type_wkb_id_roundtrip(
        #[values(
            (Geometry, 0),
            (Point, 1),
            (LineString, 2),
            (Polygon, 3),
            (MultiPoint, 4),
            (MultiLineString, 5),
            (MultiPolygon, 6),
            (GeometryCollection, 7)
        )]
        geometry_type_and_id: (GeometryTypeId, u32),
    ) {
        let (geometry_type, wkb_id) = geometry_type_and_id;
        assert_eq!(geometry_type.wkb_id(), wkb_id);
        assert_eq!(
            GeometryTypeId::try_from_wkb_id(wkb_id).unwrap(),
            geometry_type
        );
    }

    #[test]
    fn geometry_type_wkb_id_err() {
        let err = GeometryTypeId::try_from_wkb_id(17).unwrap_err();
        assert_eq!(err.to_string(), "Unknown geometry type identifier 17");
    }

    #[rstest]
    fn geometry_type_str_roundtrip(
        #[values(
            (Geometry, "Geometry"),
            (Point, "Point"),
            (LineString, "LineString"),
            (Polygon, "Polygon"),
            (MultiPoint, "MultiPoint"),
            (MultiLineString, "MultiLineString"),
            (MultiPolygon, "MultiPolygon"),
            (GeometryCollection, "GeometryCollection")
        )]
        geometry_type_and_str: (GeometryTypeId, &str),
    ) {
        let (geometry_type, string) = geometry_type_and_str;
        assert_eq!(geometry_type.geojson_id(), string);
        assert_eq!(GeometryTypeId::from_str(string).unwrap(), geometry_type);
    }

    #[test]
    fn geometry_type_str_err() {
        let err = GeometryTypeId::from_str("gazornenplat").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid geometry type string: 'gazornenplat'"
        );
    }

    #[rstest]
    fn geometry_type_dims_wkb_id_roundtrip(
        #[values(
            (Geometry, 0),
            (Point, 1),
            (LineString, 2),
            (Polygon, 3),
            (MultiPoint, 4),
            (MultiLineString, 5),
            (MultiPolygon, 6),
            (GeometryCollection, 7)
        )]
        geometry_type_and_id: (GeometryTypeId, u32),
        #[values(
            (Xy, 0),
            (Xyz, 1000),
            (Xym, 2000),
            (Xyzm, 3000),
        )]
        dimensions_and_id: (Dimensions, u32),
    ) {
        let (geometry_type, geometry_type_id) = geometry_type_and_id;
        let (dimensions, dimensions_id) = dimensions_and_id;

        let value = GeometryTypeAndDimensions::new(geometry_type, dimensions);
        assert_eq!(value.wkb_id(), dimensions_id + geometry_type_id);
        assert_eq!(
            GeometryTypeAndDimensions::try_from_wkb_id(dimensions_id + geometry_type_id).unwrap(),
            value
        );
    }

    #[test]
    fn geometry_type_dims_wkb_id_err() {
        let err = GeometryTypeAndDimensions::try_from_wkb_id(17).unwrap_err();
        assert_eq!(err.to_string(), "Unknown geometry type identifier 17");

        let err = GeometryTypeAndDimensions::try_from_wkb_id(4000).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unknown dimensions in ISO WKB geometry type: 4000"
        );
    }

    #[rstest]
    fn geometry_type_dims_str_roundtrip(
        #[values(
            (Geometry, "Geometry"),
            (Point, "Point"),
            (LineString, "LineString"),
            (Polygon, "Polygon"),
            (MultiPoint, "MultiPoint"),
            (MultiLineString, "MultiLineString"),
            (MultiPolygon, "MultiPolygon"),
            (GeometryCollection, "GeometryCollection")
        )]
        geometry_type_and_str: (GeometryTypeId, &str),
        #[values(
            (Xy, ""),
            (Xyz, " Z"),
            (Xym, " M"),
            (Xyzm, " ZM"),
        )]
        dimensions_and_suffix: (Dimensions, &str),
    ) {
        let (geometry_type, geometry_type_id) = geometry_type_and_str;
        let (dimensions, dimensions_id) = dimensions_and_suffix;
        let string_id = geometry_type_id.to_string() + dimensions_id;

        let value = GeometryTypeAndDimensions::new(geometry_type, dimensions);
        assert_eq!(value.geojson_id(), string_id);
        assert_eq!(
            GeometryTypeAndDimensions::from_str(string_id.as_str()).unwrap(),
            value
        );
    }
}
