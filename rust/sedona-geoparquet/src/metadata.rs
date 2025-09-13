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
/// Strongly-typed structs corresponding to the metadata provided by the GeoParquet specification.
///
/// This is a slightly modified version of geoarrow-rs/rust/geoarrow-geoparquet (modified
/// to remove the dependency on GeoArrow since we mostly don't need that here yet).
/// This should be synchronized with that crate when possible.
/// https://github.com/geoarrow/geoarrow-rs/blob/ad2d29ef90050c5cfcfa7dfc0b4a3e5d12e51bbe/rust/geoarrow-geoparquet/src/metadata.rs
use datafusion_common::Result;
use parquet::file::metadata::ParquetMetaData;
use sedona_expr::statistics::GeoStatistics;
use sedona_geometry::bounding_box::BoundingBox;
use sedona_geometry::interval::{Interval, IntervalTrait};
use sedona_geometry::types::GeometryTypeAndDimensions;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::fmt::Write;

use datafusion_common::DataFusionError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The actual encoding of the geometry in the Parquet file.
///
/// In contrast to the _user-specified API_, which is just "WKB" or "Native", here we need to know
/// the actual written encoding type so that we can save that in the metadata.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum GeoParquetColumnEncoding {
    /// Serialized Well-known Binary encoding
    WKB,
    /// Native Point encoding
    #[serde(rename = "point")]
    Point,
    /// Native LineString encoding
    #[serde(rename = "linestring")]
    LineString,
    /// Native Polygon encoding
    #[serde(rename = "polygon")]
    Polygon,
    /// Native MultiPoint encoding
    #[serde(rename = "multipoint")]
    MultiPoint,
    /// Native MultiLineString encoding
    #[serde(rename = "multilinestring")]
    MultiLineString,
    /// Native MultiPolygon encoding
    #[serde(rename = "multipolygon")]
    MultiPolygon,
}

impl Default for GeoParquetColumnEncoding {
    fn default() -> Self {
        Self::WKB
    }
}

impl Display for GeoParquetColumnEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use GeoParquetColumnEncoding::*;
        match self {
            WKB => write!(f, "WKB"),
            Point => write!(f, "point"),
            LineString => write!(f, "linestring"),
            Polygon => write!(f, "polygon"),
            MultiPoint => write!(f, "multipoint"),
            MultiLineString => write!(f, "multilinestring"),
            MultiPolygon => write!(f, "multipolygon"),
        }
    }
}

/// Bounding-box covering
///
/// Including a per-row bounding box can be useful for accelerating spatial queries by allowing
/// consumers to inspect row group and page index bounding box summary statistics. Furthermore a
/// bounding box may be used to avoid complex spatial operations by first checking for bounding box
/// overlaps. This field captures the column name and fields containing the bounding box of the
/// geometry for every row.
///
/// The format of the bbox encoding is
/// ```json
/// {
///     "xmin": ["column_name", "xmin"],
///     "ymin": ["column_name", "ymin"],
///     "xmax": ["column_name", "xmax"],
///     "ymax": ["column_name", "ymax"]
/// }
/// ```
///
/// The arrays represent Parquet schema paths for nested groups. In this example, column_name is a
/// Parquet group with fields xmin, ymin, xmax, ymax. The value in column_name MUST exist in the
/// Parquet file and meet the criteria in the Bounding Box Column definition. In order to constrain
/// this value to a single bounding group field, the second item in each element MUST be xmin,
/// ymin, etc. All values MUST use the same column name.
///
/// The value specified in this field should not be confused with the top-level bbox field which
/// contains the single bounding box of this geometry over the whole GeoParquet file.
///
/// Note: This technique to use the bounding box to improve spatial queries does not apply to
/// geometries that cross the antimeridian. Such geometries are unsupported by this method.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoParquetBboxCovering {
    /// The path in the Parquet schema of the column that contains the xmin
    pub xmin: Vec<String>,

    /// The path in the Parquet schema of the column that contains the ymin
    pub ymin: Vec<String>,

    /// The path in the Parquet schema of the column that contains the zmin
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zmin: Option<Vec<String>>,

    /// The path in the Parquet schema of the column that contains the xmax
    pub xmax: Vec<String>,

    /// The path in the Parquet schema of the column that contains the ymax
    pub ymax: Vec<String>,

    /// The path in the Parquet schema of the column that contains the zmax
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zmax: Option<Vec<String>>,
}

impl GeoParquetBboxCovering {
    /// Infer a bbox covering from a native geoarrow encoding
    ///
    /// Note: for now this infers 2D boxes only
    pub fn infer_from_native(
        column_name: &str,
        column_metadata: &GeoParquetColumnMetadata,
    ) -> Option<Self> {
        use GeoParquetColumnEncoding::*;
        let (x, y) = match column_metadata.encoding {
            WKB => return None,
            Point => {
                let x = vec![column_name.to_string(), "x".to_string()];
                let y = vec![column_name.to_string(), "y".to_string()];
                (x, y)
            }
            LineString => {
                let x = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "x".to_string(),
                ];
                let y = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "y".to_string(),
                ];
                (x, y)
            }
            Polygon => {
                let x = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "x".to_string(),
                ];
                let y = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "y".to_string(),
                ];
                (x, y)
            }
            MultiPoint => {
                let x = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "x".to_string(),
                ];
                let y = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "y".to_string(),
                ];
                (x, y)
            }
            MultiLineString => {
                let x = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "x".to_string(),
                ];
                let y = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "y".to_string(),
                ];
                (x, y)
            }
            MultiPolygon => {
                let x = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "x".to_string(),
                ];
                let y = vec![
                    column_name.to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "list".to_string(),
                    "element".to_string(),
                    "y".to_string(),
                ];
                (x, y)
            }
        };

        Some(Self {
            xmin: x.clone(),
            ymin: y.clone(),
            zmin: None,
            xmax: x,
            ymax: y,
            zmax: None,
        })
    }
}

/// Object containing bounding box column names to help accelerate spatial data retrieval
///
/// The covering field specifies optional simplified representations of each geometry. The keys of
/// the "covering" object MUST be a supported encoding. Currently the only supported encoding is
/// "bbox" which specifies the names of bounding box columns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoParquetCovering {
    /// Bounding-box covering
    pub bbox: GeoParquetBboxCovering,
}

/// Top-level GeoParquet file metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoParquetMetadata {
    /// The version identifier for the GeoParquet specification.
    pub version: String,

    /// The name of the "primary" geometry column. In cases where a GeoParquet file contains
    /// multiple geometry columns, the primary geometry may be used by default in geospatial
    /// operations.
    pub primary_column: String,

    /// Metadata about geometry columns. Each key is the name of a geometry column in the table.
    pub columns: HashMap<String, GeoParquetColumnMetadata>,
}

impl Default for GeoParquetMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            primary_column: Default::default(),
            columns: Default::default(),
        }
    }
}

/// GeoParquet column metadata
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct GeoParquetColumnMetadata {
    /// Name of the geometry encoding format. As of GeoParquet 1.1, `"WKB"`, `"point"`,
    /// `"linestring"`, `"polygon"`, `"multipoint"`, `"multilinestring"`, and `"multipolygon"` are
    /// supported.
    pub encoding: GeoParquetColumnEncoding,

    /// The geometry types of all geometries, or an empty array if they are not known.
    ///
    /// This field captures the geometry types of the geometries in the column, when known.
    /// Accepted geometry types are: `"Point"`, `"LineString"`, `"Polygon"`, `"MultiPoint"`,
    /// `"MultiLineString"`, `"MultiPolygon"`, `"GeometryCollection"`.
    ///
    /// In addition, the following rules are used:
    ///
    /// - In case of 3D geometries, a `" Z"` suffix gets added (e.g. `["Point Z"]`).
    /// - A list of multiple values indicates that multiple geometry types are present (e.g.
    ///   `["Polygon", "MultiPolygon"]`).
    /// - An empty array explicitly signals that the geometry types are not known.
    /// - The geometry types in the list must be unique (e.g. `["Point", "Point"]` is not valid).
    ///
    /// It is expected that this field is strictly correct. For example, if having both polygons
    /// and multipolygons, it is not sufficient to specify `["MultiPolygon"]`, but it is expected
    /// to specify `["Polygon", "MultiPolygon"]`. Or if having 3D points, it is not sufficient to
    /// specify `["Point"]`, but it is expected to list `["Point Z"]`.
    pub geometry_types: HashSet<GeometryTypeAndDimensions>,

    /// [PROJJSON](https://proj.org/specifications/projjson.html) object representing the
    /// Coordinate Reference System (CRS) of the geometry. If the field is not provided, the
    /// default CRS is [OGC:CRS84](https://www.opengis.net/def/crs/OGC/1.3/CRS84), which means the
    /// data in this column must be stored in longitude, latitude based on the WGS84 datum.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crs: Option<Value>,

    /// Winding order of exterior ring of polygons. If present must be `"counterclockwise"`;
    /// interior rings are wound in opposite order. If absent, no assertions are made regarding the
    /// winding order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orientation: Option<String>,

    /// Name of the coordinate system for the edges. Must be one of `"planar"` or `"spherical"`.
    /// The default value is `"planar"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edges: Option<String>,

    /// Bounding Box of the geometries in the file, formatted according to RFC 7946, section 5.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<Vec<f64>>,

    /// Coordinate epoch in case of a dynamic CRS, expressed as a decimal year.
    ///
    /// In a dynamic CRS, coordinates of a point on the surface of the Earth may change with time.
    /// To be unambiguous, the coordinates must always be qualified with the epoch at which they
    /// are valid.
    ///
    /// The optional epoch field allows to specify this in case the crs field defines a dynamic
    /// CRS. The coordinate epoch is expressed as a decimal year (e.g. `2021.47`). Currently, this
    /// specification only supports an epoch per column (and not per geometry).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<f64>,

    /// Object containing bounding box column names to help accelerate spatial data retrieval
    #[serde(skip_serializing_if = "Option::is_none")]
    pub covering: Option<GeoParquetCovering>,
}

impl GeoParquetMetadata {
    /// Construct a [`GeoParquetMetadata`] from a JSON string
    pub fn try_new(metadata: &str) -> Result<Self> {
        serde_json::from_str(metadata).map_err(|e| DataFusionError::Plan(e.to_string()))
    }

    /// Construct a [`GeoParquetMetadata`] from a [`ParquetMetaData`]
    pub fn try_from_parquet_metadata(metadata: &ParquetMetaData) -> Result<Option<Self>> {
        if let Some(kv) = metadata.file_metadata().key_value_metadata() {
            for item in kv {
                if item.key == "geo" && item.value.is_some() {
                    return Ok(Some(Self::try_new(item.value.as_ref().unwrap())?));
                }
            }
        }

        Ok(None)
    }

    /// Update a GeoParquetMetadata from another file's metadata
    ///
    /// This will expand the bounding box of each geometry column to include the bounding box
    /// defined in the other file's GeoParquet metadata
    pub fn try_update(&mut self, other: &GeoParquetMetadata) -> Result<()> {
        self.try_compatible_with(other)?;
        for (column_name, column_meta) in self.columns.iter_mut() {
            let other_column_meta = other.columns.get(column_name.as_str()).unwrap();

            column_meta.bbox = match (column_meta.bounding_box(), other_column_meta.bounding_box())
            {
                (Some(this_bbox), Some(other_bbox)) => {
                    let mut out = this_bbox.clone();
                    out.update_box(&other_bbox);
                    // For the purposes of this merging, we don't propagate Z bounds
                    Some(vec![out.x().lo(), out.x().hi(), out.y().lo(), out.y().hi()])
                }
                _ => None,
            };

            if column_meta.geometry_types.is_empty() || other_column_meta.geometry_types.is_empty()
            {
                column_meta.geometry_types.clear();
            } else {
                for item in &other_column_meta.geometry_types {
                    column_meta.geometry_types.insert(*item);
                }
            }
        }
        Ok(())
    }

    /// Assert that this metadata is compatible with another metadata instance, erroring if not
    pub fn try_compatible_with(&self, other: &GeoParquetMetadata) -> Result<()> {
        if self.version.as_str() != other.version.as_str() {
            return Err(DataFusionError::Plan(
                "Different GeoParquet versions".to_string(),
            ));
        }

        if self.primary_column.as_str() != other.primary_column.as_str() {
            return Err(DataFusionError::Plan(
                "Different GeoParquet primary columns".to_string(),
            ));
        }

        for key in self.columns.keys() {
            let left = self.columns.get(key).unwrap();
            let right = other.columns.get(key).ok_or(DataFusionError::Plan(format!(
                "Other GeoParquet metadata missing column {key}"
            )))?;

            if left.encoding != right.encoding {
                return Err(DataFusionError::Plan(format!(
                    "Different GeoParquet encodings for column {key}"
                )));
            }

            match (left.crs.as_ref(), right.crs.as_ref()) {
                (Some(left_crs), Some(right_crs)) => {
                    if left_crs != right_crs {
                        return Err(DataFusionError::Plan(format!(
                            "Different GeoParquet CRS for column {key}"
                        )));
                    }
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(DataFusionError::Plan(format!(
                        "Different GeoParquet CRS for column {key}"
                    )));
                }
                (None, None) => (),
            }
        }

        Ok(())
    }

    /// Get the bounding box covering for a geometry column
    ///
    /// If the desired column does not have covering metadata, if it is a native encoding its
    /// covering will be inferred.
    pub fn bbox_covering(
        &self,
        column_name: Option<&str>,
    ) -> Result<Option<GeoParquetBboxCovering>> {
        let column_name = column_name.unwrap_or(&self.primary_column);
        let column_meta = self
            .columns
            .get(column_name)
            .ok_or(DataFusionError::Plan(format!(
                "Column name {column_name} not found in metadata"
            )))?;
        if let Some(covering) = &column_meta.covering {
            Ok(Some(covering.bbox.clone()))
        } else {
            let inferred_covering =
                GeoParquetBboxCovering::infer_from_native(column_name, column_meta);
            Ok(inferred_covering)
        }
    }
}

impl GeoParquetColumnMetadata {
    pub fn to_geoarrow_metadata(&self) -> Result<String> {
        let geoarrow_crs = if let Some(crs) = &self.crs {
            serde_json::to_string(&crs).unwrap()
        } else {
            "\"OGC:CRS84\"".to_string()
        };

        let mut out = String::new();
        write!(out, r#"{{"crs": {geoarrow_crs}"#)?;

        if let Some(edges) = &self.edges {
            write!(out, r#", "edges": "{edges}""#)?;
        }
        // If `edges` is None, omit the field entirely.

        write!(out, "}}")?;
        Ok(out)
    }

    pub fn to_geo_statistics(&self) -> GeoStatistics {
        let stats = GeoStatistics::unspecified().with_bbox(self.bounding_box());
        if self.geometry_types.is_empty() {
            stats
        } else {
            let geometry_types = self.geometry_types.iter().cloned().collect::<Vec<_>>();
            stats.with_geometry_types(Some(&geometry_types))
        }
    }

    pub fn bounding_box(&self) -> Option<BoundingBox> {
        if let Some(bbox) = &self.bbox {
            match bbox.len() {
                4 => Some(BoundingBox::xy((bbox[0], bbox[2]), (bbox[1], bbox[3]))),
                6 => Some(BoundingBox::xyzm(
                    (bbox[0], bbox[3]),
                    (bbox[1], bbox[4]),
                    Some(Interval::new(bbox[2], bbox[5])),
                    None,
                )),
                _ => None,
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use geo_traits::Dimensions;
    use sedona_geometry::types::GeometryTypeId;

    use super::*;

    // We want to ensure that extra keys in future GeoParquet versions do not break
    // By default, serde allows and ignores unknown keys
    #[test]
    fn extra_keys_in_column_metadata() {
        let s = r#"{
            "encoding": "WKB",
            "geometry_types": ["Point"],
            "other_key": true
        }"#;
        let meta: GeoParquetColumnMetadata = serde_json::from_str(s).unwrap();
        assert_eq!(meta.encoding, GeoParquetColumnEncoding::WKB);
        assert_eq!(
            meta.geometry_types.iter().next().unwrap(),
            &GeometryTypeAndDimensions::new(GeometryTypeId::Point, Dimensions::Xy)
        );
    }
}
