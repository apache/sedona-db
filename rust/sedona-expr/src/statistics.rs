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
use std::str::FromStr;

use datafusion_common::{stats::Precision, ColumnStatistics, DataFusionError, Result, ScalarValue};
use sedona_geometry::interval::{Interval, IntervalTrait};
use sedona_geometry::{
    bounding_box::BoundingBox,
    types::{GeometryTypeAndDimensions, GeometryTypeAndDimensionsSet},
};
use serde::{Deserialize, Serialize};

/// Statistics specific to spatial data types
///
/// These statistics are an abstraction to provide sedonadb the ability to
/// perform generic pruning and optimization for datasources that have the
/// ability to provide this information. This may evolve to support more
/// fields; however, can currently express Parquet built-in GeoStatistics,
/// GeoParquet metadata, and GDAL OGR (via GetExtent() and GetGeomType()).
/// This struct can also represent partial or missing information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoStatistics {
    // Core spatial statistics for pruning
    bbox: Option<BoundingBox>, // The overall bounding box (min/max coordinates) containing all geometries
    geometry_types: Option<GeometryTypeAndDimensionsSet>, // Set of all geometry types and dimensions present

    // Extended statistics for analysis
    total_geometries: Option<i64>, // Total count of all geometries
    total_size_bytes: Option<i64>, // Total size of all geometries in bytes
    total_points: Option<i64>,     // Total number of points/vertices across all geometries

    // Type distribution counts
    puntal_count: Option<i64>,     // Count of point-type geometries
    lineal_count: Option<i64>,     // Count of line-type geometries
    polygonal_count: Option<i64>,  // Count of polygon-type geometries
    collection_count: Option<i64>, // Count of geometry collections

    // Envelope dimensions statistics
    total_envelope_width: Option<f64>, // Sum of all envelope widths (for calculating mean width)
    total_envelope_height: Option<f64>, // Sum of all envelope heights (for calculating mean height)
}

impl GeoStatistics {
    /// Create statistics representing unspecified information
    pub fn unspecified() -> Self {
        Self {
            bbox: None,
            geometry_types: None,
            total_geometries: None,
            total_size_bytes: None,
            total_points: None,
            puntal_count: None,
            lineal_count: None,
            polygonal_count: None,
            collection_count: None,
            total_envelope_width: None,
            total_envelope_height: None,
        }
    }

    /// Create statistics representing empty information (with zero values instead of None)
    pub fn empty() -> Self {
        Self {
            bbox: Some(BoundingBox::xy(Interval::empty(), Interval::empty())),
            geometry_types: Some(GeometryTypeAndDimensionsSet::new()), // Empty set of geometry types
            total_geometries: Some(0),                                 // Zero geometries
            total_size_bytes: Some(0),                                 // Zero bytes
            total_points: Some(0),                                     // Zero points
            puntal_count: Some(0),                                     // Zero point geometries
            lineal_count: Some(0),                                     // Zero line geometries
            polygonal_count: Some(0),                                  // Zero polygon geometries
            collection_count: Some(0),                                 // Zero collection geometries
            total_envelope_width: Some(0.0),                           // Zero width
            total_envelope_height: Some(0.0),                          // Zero height
        }
    }

    /// Update the bounding box and return self
    pub fn with_bbox(self, bbox: Option<BoundingBox>) -> Self {
        Self { bbox, ..self }
    }

    /// Update the geometry types and return self
    pub fn with_geometry_types(self, types: Option<GeometryTypeAndDimensionsSet>) -> Self {
        Self {
            geometry_types: types,
            ..self
        }
    }

    /// Get the bounding box if available
    pub fn bbox(&self) -> Option<&BoundingBox> {
        self.bbox.as_ref()
    }

    /// Get the geometry types if available
    pub fn geometry_types(&self) -> Option<&GeometryTypeAndDimensionsSet> {
        self.geometry_types.as_ref()
    }

    /// Get the total number of geometries if available
    pub fn total_geometries(&self) -> Option<i64> {
        self.total_geometries
    }

    /// Get the total size in bytes if available
    pub fn total_size_bytes(&self) -> Option<i64> {
        self.total_size_bytes
    }

    /// Get the total number of points if available
    pub fn total_points(&self) -> Option<i64> {
        self.total_points
    }

    /// Get the count of puntal geometries if available
    pub fn puntal_count(&self) -> Option<i64> {
        self.puntal_count
    }

    /// Get the count of lineal geometries if available
    pub fn lineal_count(&self) -> Option<i64> {
        self.lineal_count
    }

    /// Get the count of polygonal geometries if available
    pub fn polygonal_count(&self) -> Option<i64> {
        self.polygonal_count
    }

    /// Get the count of geometry collections if available
    pub fn collection_count(&self) -> Option<i64> {
        self.collection_count
    }

    /// Get the total envelope width if available
    pub fn total_envelope_width(&self) -> Option<f64> {
        self.total_envelope_width
    }

    /// Get the total envelope height if available
    pub fn total_envelope_height(&self) -> Option<f64> {
        self.total_envelope_height
    }

    /// Calculate the mean envelope width if possible
    pub fn mean_envelope_width(&self) -> Option<f64> {
        match (self.total_envelope_width, self.total_geometries) {
            (Some(width), Some(count)) if count > 0 => Some(width / count as f64),
            _ => None,
        }
    }

    /// Calculate the mean envelope height if possible
    pub fn mean_envelope_height(&self) -> Option<f64> {
        match (self.total_envelope_height, self.total_geometries) {
            (Some(height), Some(count)) if count > 0 => Some(height / count as f64),
            _ => None,
        }
    }

    /// Calculate the mean envelope area if possible
    pub fn mean_envelope_area(&self) -> Option<f64> {
        match (self.mean_envelope_width(), self.mean_envelope_height()) {
            (Some(width), Some(height)) => Some(width * height),
            _ => None,
        }
    }

    /// Calculate the mean size in bytes if possible
    pub fn mean_size_bytes(&self) -> Option<f64> {
        match (self.total_size_bytes, self.total_geometries) {
            (Some(bytes), Some(count)) if count > 0 => Some(bytes as f64 / count as f64),
            _ => None,
        }
    }

    /// Calculate the mean points per geometry if possible
    pub fn mean_points_per_geometry(&self) -> Option<f64> {
        match (self.total_points, self.total_geometries) {
            (Some(points), Some(count)) if count > 0 => Some(points as f64 / count as f64),
            _ => None,
        }
    }

    /// Update the total geometries count and return self
    pub fn with_total_geometries(self, count: i64) -> Self {
        Self {
            total_geometries: Some(count),
            ..self
        }
    }

    /// Update the total size in bytes and return self
    pub fn with_total_size_bytes(self, bytes: i64) -> Self {
        Self {
            total_size_bytes: Some(bytes),
            ..self
        }
    }

    /// Update the total points count and return self
    pub fn with_total_points(self, points: i64) -> Self {
        Self {
            total_points: Some(points),
            ..self
        }
    }

    /// Update the puntal geometries count and return self
    pub fn with_puntal_count(self, count: i64) -> Self {
        Self {
            puntal_count: Some(count),
            ..self
        }
    }

    /// Update the lineal geometries count and return self
    pub fn with_lineal_count(self, count: i64) -> Self {
        Self {
            lineal_count: Some(count),
            ..self
        }
    }

    /// Update the polygonal geometries count and return self
    pub fn with_polygonal_count(self, count: i64) -> Self {
        Self {
            polygonal_count: Some(count),
            ..self
        }
    }

    /// Update the collection geometries count and return self
    pub fn with_collection_count(self, count: i64) -> Self {
        Self {
            collection_count: Some(count),
            ..self
        }
    }

    /// Update the total envelope width and return self
    pub fn with_total_envelope_width(self, width: f64) -> Self {
        Self {
            total_envelope_width: Some(width),
            ..self
        }
    }

    /// Update the total envelope height and return self
    pub fn with_total_envelope_height(self, height: f64) -> Self {
        Self {
            total_envelope_height: Some(height),
            ..self
        }
    }

    /// Update this statistics object with another one
    pub fn merge(&mut self, other: &Self) {
        // Merge bounding boxes
        if let Some(other_bbox) = &other.bbox {
            match &mut self.bbox {
                Some(bbox) => bbox.update_box(other_bbox),
                None => self.bbox = Some(other_bbox.clone()),
            }
        }

        // Merge geometry types
        if let Some(other_types) = &other.geometry_types {
            match &mut self.geometry_types {
                Some(types) => {
                    types.merge(other_types);
                }
                None => self.geometry_types = Some(other_types.clone()),
            }
        }

        // Merge counts and totals
        self.total_geometries =
            Self::merge_option_add(self.total_geometries, other.total_geometries);
        self.total_size_bytes =
            Self::merge_option_add(self.total_size_bytes, other.total_size_bytes);
        self.total_points = Self::merge_option_add(self.total_points, other.total_points);

        // Merge type counts
        self.puntal_count = Self::merge_option_add(self.puntal_count, other.puntal_count);
        self.lineal_count = Self::merge_option_add(self.lineal_count, other.lineal_count);
        self.polygonal_count = Self::merge_option_add(self.polygonal_count, other.polygonal_count);
        self.collection_count =
            Self::merge_option_add(self.collection_count, other.collection_count);

        // Merge envelope dimensions
        self.total_envelope_width =
            Self::merge_option_add_f64(self.total_envelope_width, other.total_envelope_width);
        self.total_envelope_height =
            Self::merge_option_add_f64(self.total_envelope_height, other.total_envelope_height);
    }

    // Helper to merge two optional integers with addition
    fn merge_option_add(a: Option<i64>, b: Option<i64>) -> Option<i64> {
        match (a, b) {
            (Some(a_val), Some(b_val)) => Some(a_val + b_val),
            _ => None,
        }
    }

    // Helper to merge two optional floats with addition
    fn merge_option_add_f64(a: Option<f64>, b: Option<f64>) -> Option<f64> {
        match (a, b) {
            (Some(a_val), Some(b_val)) => Some(a_val + b_val),
            _ => None,
        }
    }

    /// Try to deserialize GeoStatistics from DataFusion [ColumnStatistics]
    ///
    /// Various DataFusion APIs operate on [ColumnStatistics], which do not support
    /// spatial statistics natively. This function attempts to reconstruct an object
    /// that was canonically serialized into one of these objects for transport through
    /// DataFusion internals.
    pub fn try_from_column_statistics(stats: &ColumnStatistics) -> Result<Option<Self>> {
        let scalar = match &stats.sum_value {
            Precision::Exact(value) => value,
            _ => {
                return Ok(None);
            }
        };

        if let ScalarValue::Binary(Some(serialized)) = scalar {
            serde_json::from_slice(serialized).map_err(|e| DataFusionError::External(Box::new(e)))
        } else {
            Ok(None)
        }
    }

    /// Serialize this object into a [ColumnStatistics]
    ///
    /// Canonically place this object into a [ColumnStatistics] for transport through
    /// DataFusion APIs.
    pub fn to_column_statistics(&self) -> Result<ColumnStatistics> {
        let serialized =
            serde_json::to_vec(self).map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(ColumnStatistics::new_unknown()
            .with_sum_value(Precision::Exact(ScalarValue::Binary(Some(serialized)))))
    }

    /// Add a definitive list of geometry type/dimension values specified as strings
    ///
    /// This accepts a list of strings like "point z" or "polygon zm". These strings
    /// are case insensitive.
    pub fn try_with_str_geometry_types(self, geometry_types: Option<&[&str]>) -> Result<Self> {
        match geometry_types {
            Some(strings) => {
                let mut new_geometry_types = GeometryTypeAndDimensionsSet::new();
                for string in strings {
                    let type_and_dim = GeometryTypeAndDimensions::from_str(string)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    new_geometry_types.insert_or_ignore(&type_and_dim);
                }

                Ok(Self {
                    geometry_types: Some(new_geometry_types),
                    ..self
                })
            }
            None => Ok(Self {
                geometry_types: None,
                ..self
            }),
        }
    }

    /// Convert this GeoStatistics to a ScalarValue for storage in DataFusion statistics
    pub fn to_scalar_value(&self) -> Result<ScalarValue> {
        // Serialize to JSON
        let serialized = serde_json::to_vec(self).map_err(|e| {
            DataFusionError::Internal(format!("Failed to serialize GeoStatistics: {e}"))
        })?;

        Ok(ScalarValue::Binary(Some(serialized)))
    }
}

#[cfg(test)]
mod test {
    use geo_traits::Dimensions;
    use sedona_geometry::types::GeometryTypeId;

    use super::*;

    #[test]
    fn unspecified() {
        let stats = GeoStatistics::unspecified();
        assert_eq!(stats.bbox(), None);
        assert_eq!(stats.geometry_types(), None);
        assert_eq!(stats.total_geometries(), None);
        assert_eq!(stats.total_size_bytes(), None);
        assert_eq!(stats.total_points(), None);
        assert_eq!(stats.puntal_count(), None);
        assert_eq!(stats.lineal_count(), None);
        assert_eq!(stats.polygonal_count(), None);
        assert_eq!(stats.collection_count(), None);
        assert_eq!(stats.total_envelope_width(), None);
        assert_eq!(stats.total_envelope_height(), None);

        let regular_stats = stats.to_column_statistics().unwrap();
        assert_eq!(
            GeoStatistics::try_from_column_statistics(&regular_stats)
                .unwrap()
                .unwrap(),
            stats
        );
    }

    #[test]
    fn specified_bbox() {
        let bbox = BoundingBox::xy((0.0, 1.0), (2.0, 3.0));
        // Test with_bbox
        let stats = GeoStatistics::empty().with_bbox(Some(bbox.clone()));
        assert_eq!(stats.bbox(), Some(&bbox));
        assert_eq!(
            stats.geometry_types(),
            Some(&GeometryTypeAndDimensionsSet::new())
        );

        let regular_stats = stats.to_column_statistics().unwrap();
        assert_eq!(
            GeoStatistics::try_from_column_statistics(&regular_stats)
                .unwrap()
                .unwrap(),
            stats
        );

        // Test with None
        let stats_with_none = GeoStatistics::empty().with_bbox(None);
        assert_eq!(stats_with_none.bbox(), None);
    }

    #[test]
    fn specified_geometry_types() {
        let mut types = GeometryTypeAndDimensionsSet::new();
        types
            .insert(&GeometryTypeAndDimensions::new(
                GeometryTypeId::Polygon,
                Dimensions::Xy,
            ))
            .unwrap();

        // Test with_geometry_types
        let stats = GeoStatistics::empty().with_geometry_types(Some(types.clone()));
        assert_eq!(stats.geometry_types(), Some(&types));
        assert_eq!(
            stats.bbox(),
            Some(&BoundingBox::xy(Interval::empty(), Interval::empty()))
        );

        let regular_stats = stats.to_column_statistics().unwrap();
        assert_eq!(
            GeoStatistics::try_from_column_statistics(&regular_stats)
                .unwrap()
                .unwrap(),
            stats
        );

        // Test with None
        let stats_with_none = GeoStatistics::empty().with_geometry_types(None);
        assert_eq!(stats_with_none.geometry_types(), None);
    }

    #[test]
    fn specified_geometry_types_by_name() {
        // Test try_with_str_geometry_types
        let stats = GeoStatistics::empty()
            .try_with_str_geometry_types(Some(&["polygon", "point"]))
            .unwrap();

        let mut expected_types = GeometryTypeAndDimensionsSet::new();
        expected_types
            .insert(&GeometryTypeAndDimensions::new(
                GeometryTypeId::Polygon,
                Dimensions::Xy,
            ))
            .unwrap();
        expected_types
            .insert(&GeometryTypeAndDimensions::new(
                GeometryTypeId::Point,
                Dimensions::Xy,
            ))
            .unwrap();

        assert_eq!(stats.geometry_types(), Some(&expected_types));
        assert_eq!(
            stats.bbox(),
            Some(&BoundingBox::xy(Interval::empty(), Interval::empty()))
        );

        // Test serialization
        let regular_stats = stats.to_column_statistics().unwrap();
        assert_eq!(
            GeoStatistics::try_from_column_statistics(&regular_stats)
                .unwrap()
                .unwrap(),
            stats
        );
    }

    #[test]
    fn from_non_geometry_stats() {
        // Can't make geo stats from unknown
        let stats = ColumnStatistics::new_unknown();
        assert!(GeoStatistics::try_from_column_statistics(&stats)
            .unwrap()
            .is_none());

        // Can't make geo stats from binary null
        let stats = ColumnStatistics::new_unknown()
            .with_sum_value(Precision::Exact(ScalarValue::Binary(None)));
        assert!(GeoStatistics::try_from_column_statistics(&stats)
            .unwrap()
            .is_none());

        // Can't make geo stats from binary null
        let stats = ColumnStatistics::new_unknown()
            .with_sum_value(Precision::Exact(ScalarValue::Binary(Some(vec![]))));
        let err = GeoStatistics::try_from_column_statistics(&stats).unwrap_err();
        assert_eq!(
            err.message(),
            "EOF while parsing a value at line 1 column 0"
        )
    }

    #[test]
    fn test_extended_stats() {
        // Use fluent API with with_* methods
        let stats = GeoStatistics::empty()
            .with_total_geometries(100)
            .with_total_size_bytes(10000)
            .with_total_points(5000)
            .with_puntal_count(20)
            .with_lineal_count(30)
            .with_polygonal_count(40)
            .with_collection_count(10)
            .with_total_envelope_width(500.0)
            .with_total_envelope_height(300.0);

        // Test getters
        assert_eq!(stats.total_geometries(), Some(100));
        assert_eq!(stats.total_size_bytes(), Some(10000));
        assert_eq!(stats.total_points(), Some(5000));
        assert_eq!(stats.puntal_count(), Some(20));
        assert_eq!(stats.lineal_count(), Some(30));
        assert_eq!(stats.polygonal_count(), Some(40));
        assert_eq!(stats.collection_count(), Some(10));
        assert_eq!(stats.total_envelope_width(), Some(500.0));
        assert_eq!(stats.total_envelope_height(), Some(300.0));

        // Test derived statistics
        assert_eq!(stats.mean_size_bytes(), Some(100.0));
        assert_eq!(stats.mean_points_per_geometry(), Some(50.0));
        assert_eq!(stats.mean_envelope_width(), Some(5.0));
        assert_eq!(stats.mean_envelope_height(), Some(3.0));
        assert_eq!(stats.mean_envelope_area(), Some(15.0));

        // Test serialization/deserialization via column statistics
        let column_stats = stats.to_column_statistics().unwrap();
        let deserialized = GeoStatistics::try_from_column_statistics(&column_stats)
            .unwrap()
            .unwrap();
        assert_eq!(deserialized, stats);
    }

    #[test]
    fn test_merge_extended_stats() {
        // Create statistics objects using fluent API
        let stats1 = GeoStatistics::empty()
            .with_total_geometries(50)
            .with_total_size_bytes(5000)
            .with_total_points(2500)
            .with_puntal_count(10)
            .with_lineal_count(15)
            .with_polygonal_count(20)
            .with_collection_count(5)
            .with_total_envelope_width(250.0)
            .with_total_envelope_height(150.0);

        let stats2 = GeoStatistics::empty()
            .with_total_geometries(50)
            .with_total_size_bytes(5000)
            .with_total_points(2500)
            .with_puntal_count(10)
            .with_lineal_count(15)
            .with_polygonal_count(20)
            .with_collection_count(5)
            .with_total_envelope_width(250.0)
            .with_total_envelope_height(150.0);

        // Now merge them
        let mut merged = stats1.clone();
        merged.merge(&stats2);

        // Check merged results
        assert_eq!(merged.total_geometries(), Some(100));
        assert_eq!(merged.total_size_bytes(), Some(10000));
        assert_eq!(merged.total_points(), Some(5000));
        assert_eq!(merged.puntal_count(), Some(20));
        assert_eq!(merged.lineal_count(), Some(30));
        assert_eq!(merged.polygonal_count(), Some(40));
        assert_eq!(merged.collection_count(), Some(10));
        assert_eq!(merged.total_envelope_width(), Some(500.0));
        assert_eq!(merged.total_envelope_height(), Some(300.0));

        // Test serialization/deserialization of merged stats
        let column_stats = merged.to_column_statistics().unwrap();
        let deserialized = GeoStatistics::try_from_column_statistics(&column_stats)
            .unwrap()
            .unwrap();
        assert_eq!(deserialized, merged);
    }

    #[test]
    fn test_partial_merge() {
        let stats1 = GeoStatistics::empty()
            .with_total_geometries(50)
            .with_total_size_bytes(5000);

        let stats2 = GeoStatistics::empty()
            .with_puntal_count(20)
            .with_lineal_count(30);

        let mut merged = stats1.clone();
        merged.merge(&stats2);

        // Check merged results
        assert_eq!(merged.total_geometries(), Some(50));
        assert_eq!(merged.total_size_bytes(), Some(5000));
        assert_eq!(merged.puntal_count(), Some(20));
        assert_eq!(merged.lineal_count(), Some(30));
        assert_eq!(merged.polygonal_count(), Some(0));

        // Test serialization/deserialization of partially merged stats
        let column_stats = merged.to_column_statistics().unwrap();
        let deserialized = GeoStatistics::try_from_column_statistics(&column_stats)
            .unwrap()
            .unwrap();
        assert_eq!(deserialized, merged);
    }
}
