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

/// A set containing [`GeometryTypeAndDimensions`] values
///
/// This set is conceptually similar to `HashSet<GeometryTypeAndDimensions>` but
/// uses a compact bitset representation for efficiency.
///
/// This set only supports the standard dimensions: XY, XYZ, XYM, and XYZM.
/// Unknown dimensions (other than these four standard types) are not supported
/// and will be rejected by [`insert`](Self::insert) or silently ignored by
/// [`insert_or_ignore`](Self::insert_or_ignore).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GeometryTypeAndDimensionsSet {
    /// Bitset encoding geometry types and dimensions.
    ///
    /// Uses bits 0-31 where each geometry type's WKB ID (0-7) is encoded
    /// at different offsets based on dimensions:
    /// - XY: bits 0-7
    /// - XYZ: bits 8-15
    /// - XYM: bits 16-23
    /// - XYZM: bits 24-31
    types: u32,
}

impl GeometryTypeAndDimensionsSet {
    #[inline]
    pub fn new() -> Self {
        Self { types: 0 }
    }

    /// Insert a geometry type and dimensions into the set.
    ///
    /// Returns an error if the dimensions are unknown (not one of XY, XYZ, XYM, or XYZM).
    /// Only the standard four dimension types are supported; attempting to insert
    /// a geometry with `Dimensions::Unknown(_)` will result in an error.
    #[inline]
    pub fn insert(
        &mut self,
        type_and_dim: &GeometryTypeAndDimensions,
    ) -> Result<(), SedonaGeometryError> {
        if let Dimensions::Unknown(n) = type_and_dim.dimensions() {
            return Err(SedonaGeometryError::Invalid(format!(
                "Unknown dimensions {} in GeometryTypeAndDimensionsSet::insert",
                n
            )));
        }
        self.insert_or_ignore(type_and_dim);
        Ok(())
    }

    /// Insert a geometry type and dimensions into the set, ignoring unknown dimensions.
    ///
    /// If the dimensions are unknown (not one of XY, XYZ, XYM, or XYZM), this method
    /// silently ignores the insertion without returning an error. This is useful when
    /// processing data that may contain unsupported dimension types that should be
    /// skipped rather than causing an error.
    #[inline]
    pub fn insert_or_ignore(&mut self, type_and_dim: &GeometryTypeAndDimensions) {
        let geom_shift = type_and_dim.geometry_type().wkb_id();
        // WKB ID must be < 8 to fit in the bitset layout (8 bits per dimension)
        if geom_shift >= 8 {
            panic!(
                "Invalid geometry type wkb_id {} in GeometryTypeAndDimensionsSet::insert_or_ignore",
                geom_shift
            );
        }
        let dim_shift = match type_and_dim.dimensions() {
            geo_traits::Dimensions::Unknown(_) => {
                // Ignore unknown dimensions
                return;
            }
            geo_traits::Dimensions::Xy => 0,
            geo_traits::Dimensions::Xyz => 8,
            geo_traits::Dimensions::Xym => 16,
            geo_traits::Dimensions::Xyzm => 24,
        };
        let bit_position = geom_shift + dim_shift;
        self.types |= 1 << bit_position;
    }

    /// Merge the given set into this set.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.types |= other.types;
    }

    /// Returns `true` if the set contains no geometry types.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.types == 0
    }

    /// Returns the number of geometry types in the set.
    #[inline]
    pub fn size(&self) -> usize {
        self.types.count_ones() as usize
    }

    /// Clears the set, removing all geometry types.
    #[inline]
    pub fn clear(&mut self) {
        self.types = 0;
    }

    /// Returns an iterator over the geometry types in the set.
    pub fn iter(&self) -> GeometryTypeSetIter {
        GeometryTypeSetIter {
            types: self.types,
            current_bit: 0,
        }
    }
}

/// Iterator over [`GeometryTypeAndDimensions`] values in a [`GeometryTypeAndDimensionsSet`]
pub struct GeometryTypeSetIter {
    types: u32,
    current_bit: u32,
}

impl Iterator for GeometryTypeSetIter {
    type Item = GeometryTypeAndDimensions;

    fn next(&mut self) -> Option<Self::Item> {
        // Find the next set bit
        while self.current_bit < 32 {
            let bit = self.current_bit;
            self.current_bit += 1;

            if (self.types & (1 << bit)) != 0 {
                // Decode the bit position into geometry type and dimensions
                let dim_shift = (bit / 8) * 8;
                let geom_shift = bit % 8;
                let dimensions = match dim_shift {
                    0 => Dimensions::Xy,
                    8 => Dimensions::Xyz,
                    16 => Dimensions::Xym,
                    24 => Dimensions::Xyzm,
                    _ => panic!(
                        "Invalid dimension bits at position {} in GeometryTypeAndDimensionsSet",
                        bit
                    ),
                };

                let geometry_type = GeometryTypeId::try_from_wkb_id(geom_shift)
                    .expect("Invalid geometry type wkb_id in GeometryTypeAndDimensionsSet");

                return Some(GeometryTypeAndDimensions::new(geometry_type, dimensions));
            }
        }

        None
    }
}

impl IntoIterator for &GeometryTypeAndDimensionsSet {
    type Item = GeometryTypeAndDimensions;
    type IntoIter = GeometryTypeSetIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// Serialize as a Vec to maintain compatibility with HashSet JSON format
impl Serialize for GeometryTypeAndDimensionsSet {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeSeq; // codespell:ignore ser
        let mut seq = serializer.serialize_seq(Some(self.size()))?;
        for item in self.iter() {
            seq.serialize_element(&item)?;
        }
        seq.end()
    }
}

// Deserialize from a Vec (which is what HashSet was serialized as)
impl<'de> Deserialize<'de> for GeometryTypeAndDimensionsSet {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        let items: Vec<GeometryTypeAndDimensions> = Vec::deserialize(deserializer)?;
        let mut set = GeometryTypeAndDimensionsSet::new();
        for item in items {
            set.insert(&item).map_err(D::Error::custom)?;
        }
        Ok(set)
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

    #[test]
    fn geometry_type_set_new_is_empty() {
        let set = GeometryTypeAndDimensionsSet::new();
        assert!(set.is_empty());
        assert_eq!(set.size(), 0);
        assert_eq!(set.iter().count(), 0);
    }

    #[test]
    fn geometry_type_set_insert_single() {
        let mut set = GeometryTypeAndDimensionsSet::new();
        let point_xy = GeometryTypeAndDimensions::new(Point, Xy);

        set.insert(&point_xy).unwrap();
        assert!(!set.is_empty());
        assert_eq!(set.size(), 1);

        let items: Vec<_> = set.iter().collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], point_xy);
    }

    #[test]
    fn geometry_type_set_insert_duplicate() {
        let mut set = GeometryTypeAndDimensionsSet::new();
        let point_xy = GeometryTypeAndDimensions::new(Point, Xy);

        set.insert(&point_xy).unwrap();
        set.insert(&point_xy).unwrap();
        set.insert(&point_xy).unwrap();

        assert_eq!(set.size(), 1);
        let items: Vec<_> = set.iter().collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], point_xy);
    }

    #[test]
    fn geometry_type_set_insert_all_types() {
        let mut set = GeometryTypeAndDimensionsSet::new();

        // Insert all geometry types with XY dimension
        for geom_type in [
            Geometry,
            Point,
            LineString,
            Polygon,
            MultiPoint,
            MultiLineString,
            MultiPolygon,
            GeometryCollection,
        ] {
            set.insert(&GeometryTypeAndDimensions::new(geom_type, Xy))
                .unwrap();
        }

        assert_eq!(set.size(), 8);
        let items: Vec<_> = set.iter().collect();
        assert_eq!(items.len(), 8);
    }

    #[test]
    fn geometry_type_set_insert_unknown_dimension() {
        let mut set = GeometryTypeAndDimensionsSet::new();
        let point_unknown = GeometryTypeAndDimensions::new(Point, Dimensions::Unknown(2));

        let result = set.insert(&point_unknown);

        // Unknown dimensions should return an error
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Unknown dimensions 2 in GeometryTypeAndDimensionsSet::insert"
        );
        assert!(set.is_empty());
    }

    #[test]
    fn geometry_type_set_clear() {
        let mut set = GeometryTypeAndDimensionsSet::new();
        let point_xy = GeometryTypeAndDimensions::new(Point, Xy);
        let linestring_xyz = GeometryTypeAndDimensions::new(LineString, Xyz);

        set.insert(&point_xy).unwrap();
        set.insert(&linestring_xyz).unwrap();
        assert!(!set.is_empty());
        assert_eq!(set.size(), 2);

        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.size(), 0);
        assert_eq!(set.iter().count(), 0);
    }

    #[test]
    fn geometry_type_set_merge() {
        let mut set1 = GeometryTypeAndDimensionsSet::new();
        let mut set2 = GeometryTypeAndDimensionsSet::new();

        let point_xy = GeometryTypeAndDimensions::new(Point, Xy);
        let linestring_xy = GeometryTypeAndDimensions::new(LineString, Xy);
        let polygon_xyz = GeometryTypeAndDimensions::new(Polygon, Xyz);

        set1.insert(&point_xy).unwrap();
        set1.insert(&linestring_xy).unwrap();

        set2.insert(&linestring_xy).unwrap(); // Duplicate
        set2.insert(&polygon_xyz).unwrap();

        set1.merge(&set2);

        assert_eq!(set1.size(), 3);
        let items: Vec<_> = set1.iter().collect();
        assert_eq!(items.len(), 3);
        assert!(items.contains(&point_xy));
        assert!(items.contains(&linestring_xy));
        assert!(items.contains(&polygon_xyz));
    }

    #[test]
    fn geometry_type_set_comprehensive() {
        let mut set = GeometryTypeAndDimensionsSet::new();

        // Add a mix of geometry types and dimensions
        let test_types = vec![
            GeometryTypeAndDimensions::new(Geometry, Xy),
            GeometryTypeAndDimensions::new(Point, Xy),
            GeometryTypeAndDimensions::new(LineString, Xyz),
            GeometryTypeAndDimensions::new(Polygon, Xym),
            GeometryTypeAndDimensions::new(MultiPoint, Xyzm),
            GeometryTypeAndDimensions::new(MultiLineString, Xy),
            GeometryTypeAndDimensions::new(MultiPolygon, Xyz),
            GeometryTypeAndDimensions::new(GeometryCollection, Xym),
            GeometryTypeAndDimensions::new(GeometryCollection, Xyzm),
        ];

        for type_and_dim in &test_types {
            set.insert(type_and_dim).unwrap();
        }

        assert_eq!(set.size(), test_types.len());
        let items: Vec<_> = set.iter().collect();
        assert_eq!(items.len(), test_types.len());

        for type_and_dim in &test_types {
            assert!(items.contains(type_and_dim));
        }
    }

    #[test]
    fn geometry_type_set_serde_empty() {
        let set = GeometryTypeAndDimensionsSet::new();

        // Serialize
        let json = serde_json::to_string(&set).unwrap();
        assert_eq!(json, "[]");

        // Deserialize
        let deserialized: GeometryTypeAndDimensionsSet = serde_json::from_str(&json).unwrap();
        assert!(deserialized.is_empty());
        assert_eq!(deserialized.size(), 0);
    }

    #[test]
    fn geometry_type_set_serde_single() {
        let mut set = GeometryTypeAndDimensionsSet::new();
        let point_xy = GeometryTypeAndDimensions::new(Point, Xy);
        set.insert(&point_xy).unwrap();

        // Serialize
        let json = serde_json::to_string(&set).unwrap();
        assert_eq!(json, "[\"Point\"]");

        // Deserialize
        let deserialized: GeometryTypeAndDimensionsSet = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.size(), 1);
        let items: Vec<_> = deserialized.iter().collect();
        assert_eq!(items[0], point_xy);
    }

    #[test]
    fn geometry_type_set_serde_multiple() {
        let mut set = GeometryTypeAndDimensionsSet::new();

        let test_types = vec![
            GeometryTypeAndDimensions::new(Point, Xy),
            GeometryTypeAndDimensions::new(LineString, Xyz),
            GeometryTypeAndDimensions::new(Polygon, Xyzm),
        ];

        for type_and_dim in &test_types {
            set.insert(type_and_dim).unwrap();
        }

        // Serialize
        let json = serde_json::to_string(&set).unwrap();
        assert_eq!(json, "[\"Point\",\"LineString Z\",\"Polygon ZM\"]");

        // Deserialize
        let deserialized: GeometryTypeAndDimensionsSet = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.size(), test_types.len());

        let items: Vec<_> = deserialized.iter().collect();
        for type_and_dim in &test_types {
            assert!(items.contains(type_and_dim));
        }
    }

    #[test]
    fn geometry_type_set_serde_roundtrip() {
        let mut set = GeometryTypeAndDimensionsSet::new();

        // Add all combinations of one geometry type with different dimensions
        set.insert(&GeometryTypeAndDimensions::new(Point, Xy))
            .unwrap();
        set.insert(&GeometryTypeAndDimensions::new(Point, Xyz))
            .unwrap();
        set.insert(&GeometryTypeAndDimensions::new(Point, Xym))
            .unwrap();
        set.insert(&GeometryTypeAndDimensions::new(Point, Xyzm))
            .unwrap();
        set.insert(&GeometryTypeAndDimensions::new(LineString, Xy))
            .unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&set).unwrap();
        assert_eq!(
            json,
            "[\"Point\",\"LineString\",\"Point Z\",\"Point M\",\"Point ZM\"]"
        );

        // Deserialize back
        let deserialized: GeometryTypeAndDimensionsSet = serde_json::from_str(&json).unwrap();

        // Verify the deserialized set matches the original
        assert_eq!(set.size(), deserialized.size());

        let original_items: Vec<_> = set.iter().collect();
        let deserialized_items: Vec<_> = deserialized.iter().collect();

        assert_eq!(original_items.len(), deserialized_items.len());
        for item in &original_items {
            assert!(deserialized_items.contains(item));
        }
    }
}
