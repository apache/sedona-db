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
use datafusion_common::{plan_err, Result};
use parquet::basic::{EdgeInterpolationAlgorithm, LogicalType};
use parquet::file::metadata::{KeyValue, ParquetMetaData};
use sedona_expr::statistics::GeoStatistics;
use sedona_geometry::bounding_box::BoundingBox;
use sedona_geometry::interval::{Interval, IntervalTrait};
use sedona_geometry::types::GeometryTypeAndDimensionsSet;
use std::collections::HashMap;
use std::fmt::Display;
use std::fmt::Write;

use datafusion_common::DataFusionError;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The actual encoding of the geometry in the Parquet file.
///
/// In contrast to the _user-specified API_, which is just "WKB" or "Native", here we need to know
/// the actual written encoding type so that we can save that in the metadata.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[allow(clippy::upper_case_acronyms)]
pub enum GeoParquetColumnEncoding {
    /// Serialized Well-known Binary encoding
    #[default]
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeoParquetCovering {
    /// Bounding-box covering
    pub bbox: GeoParquetBboxCovering,
}

impl GeoParquetCovering {
    pub fn bbox_struct_xy(struct_column_name: &str) -> Self {
        GeoParquetCovering {
            bbox: GeoParquetBboxCovering {
                xmin: vec![struct_column_name.to_string(), "xmin".to_string()],
                ymin: vec![struct_column_name.to_string(), "ymin".to_string()],
                zmin: None,
                xmax: vec![struct_column_name.to_string(), "xmax".to_string()],
                ymax: vec![struct_column_name.to_string(), "ymax".to_string()],
                zmax: None,
            },
        }
    }
}

/// Top-level GeoParquet file metadata
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
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
    ///
    /// Note: While the GeoParquet spec requires this field, some datasets in the wild (e.g.,
    /// Microsoft Building Footprints on Planetary Computer) omit it. We use `#[serde(default)]`
    /// to handle these out-of-spec files gracefully by defaulting to an empty set.
    #[serde(default)]
    pub geometry_types: GeometryTypeAndDimensionsSet,

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
    ///
    /// This constructor considers (1) the GeoParquet metadata in the key/value
    /// metadata and (2) Geometry/Geography types present in the Parquet schema.
    /// Specification of a column in the GeoParquet metadata takes precedence.
    pub fn try_from_parquet_metadata(metadata: &ParquetMetaData) -> Result<Option<Self>> {
        let schema = metadata.file_metadata().schema_descr().root_schema();
        let kv_metadata = metadata.file_metadata().key_value_metadata();
        Self::try_from_parquet_metadata_impl(schema, kv_metadata)
    }

    /// For testing, as it is easier to simulate the schema and key/value metadata
    /// than the whole ParquetMetaData.
    fn try_from_parquet_metadata_impl(
        root_schema: &parquet::schema::types::Type,
        kv_metadata: Option<&Vec<KeyValue>>,
    ) -> Result<Option<Self>> {
        let mut columns_from_schema = columns_from_parquet_schema(root_schema, kv_metadata)?;

        if let Some(value) = get_parquet_key_value("geo", kv_metadata) {
            // Values in the GeoParquet metadata take precedence over those from the
            // Parquet schema
            let mut out = Self::try_new(&value)?;
            for (k, v) in columns_from_schema.drain() {
                out.columns.entry(k).or_insert(v);
            }

            return Ok(Some(out));
        }

        // No geo metadata key, but we have geo columns from the schema
        if !columns_from_schema.is_empty() {
            // To keep metadata valid, ensure we set a primary column deterministically
            let mut column_names = columns_from_schema.keys().collect::<Vec<_>>();
            column_names.sort();
            let primary_column = column_names[0].to_string();

            Ok(Some(Self {
                version: "2.0.0".to_string(),
                columns: columns_from_schema,
                primary_column,
            }))
        } else {
            Ok(None)
        }
    }

    /// Replace any inferred metadata for the same column name with overrides
    pub fn override_columns(
        &mut self,
        overrides: &HashMap<String, GeoParquetColumnMetadata>,
    ) -> Result<()> {
        for (column_name, override_meta) in overrides {
            self.columns
                .insert(column_name.clone(), override_meta.clone());
        }

        Ok(())
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
                    column_meta
                        .geometry_types
                        .insert(&item)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
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
            stats.with_geometry_types(Some(self.geometry_types.clone()))
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

/// Collect column metadata from (top-level) Parquet Geometry/Geography columns
///
/// Converts embedded schema information into GeoParquet column metadata. Because
/// GeoParquet metadata does not support nested columns, this does not currently
/// support them either.
fn columns_from_parquet_schema(
    root_schema: &parquet::schema::types::Type,
    kv_metadata: Option<&Vec<KeyValue>>,
) -> Result<HashMap<String, GeoParquetColumnMetadata>> {
    let mut columns = HashMap::new();

    for field in root_schema.get_fields() {
        let column_metadata_opt =
            column_from_logical_type(field.get_basic_info().logical_type_ref(), kv_metadata)?;
        if let Some(column_metadata) = column_metadata_opt {
            let name = field.name().to_string();
            columns.insert(name, column_metadata);
        }
    }

    Ok(columns)
}

/// Convert a single LogicalType to GeoParquetColumnMetadata, if possible
///
/// Returns None for something that is not Geometry or Geography.
fn column_from_logical_type(
    logical_type: Option<&LogicalType>,
    kv_metadata: Option<&Vec<KeyValue>>,
) -> Result<Option<GeoParquetColumnMetadata>> {
    if let Some(logical_type) = logical_type {
        let mut column_metadata = GeoParquetColumnMetadata::default();

        match logical_type {
            LogicalType::Geometry { crs } => {
                column_metadata.crs = geoparquet_crs_from_logical_type(crs.as_ref(), kv_metadata);
                Ok(Some(column_metadata))
            }
            LogicalType::Geography { crs, algorithm } => {
                column_metadata.crs = geoparquet_crs_from_logical_type(crs.as_ref(), kv_metadata);

                let edges = match algorithm {
                    None | Some(EdgeInterpolationAlgorithm::SPHERICAL) => "spherical",
                    Some(EdgeInterpolationAlgorithm::VINCENTY) => "vincenty",
                    Some(EdgeInterpolationAlgorithm::ANDOYER) => "andoyer",
                    Some(EdgeInterpolationAlgorithm::THOMAS) => "thomas",
                    Some(EdgeInterpolationAlgorithm::KARNEY) => "karney",
                    Some(_) => {
                        return plan_err!(
                            "Unsupported edge interpolation algorithm in Parquet schema"
                        )
                    }
                };
                column_metadata.edges = Some(edges.to_string());
                Ok(Some(column_metadata))
            }
            _ => Ok(None),
        }
    } else {
        Ok(None)
    }
}

/// Parse a CRS from a Parquet logical type into one that will transfer to GeoParquet
/// and then to GeoArrow
///
/// This is identical to what will happen in the forthcoming Arrow release (packaged
/// with DataFusion 52), except this version also resolves projjson:xxx CRSes from
/// the key/value metadata (in Arrow this was hard because the conversion code was
/// not set up in a way that this was easy to do; here it is easy because we have
/// the whole ParquetMetadata).
fn geoparquet_crs_from_logical_type(
    crs: Option<&String>,
    kv_metadata: Option<&Vec<KeyValue>>,
) -> Option<Value> {
    if let Some(crs_str) = crs {
        // Treat an empty string the same as lon/lat. There is no concept of a "missing"
        // CRS in a Parquet LogicalType although this can be expressed with srid:0 in a
        // pinch.
        if crs_str.is_empty() {
            return None;
        }

        // Resolve projjson:some_key if possible. If this is not possible, the value that
        // will be passed on to the GeoParquet column metadata is the full string
        // "projjson:some_key".
        if let Some(crs_kv_key) = crs_str.strip_prefix("projjson:") {
            if let Some(crs_from_kv) = get_parquet_key_value(crs_kv_key, kv_metadata) {
                return Some(Value::String(crs_from_kv.to_string()));
            }
        }

        // Resolve srid:<int value> to "<int value>", which is accepted by SedonaDB internals
        // and is interpreted as an EPSG code. There is no guarantee that other implementations
        // will do this but it probably better than erroring for an unknown CRS.
        if let Some(srid_string) = crs_str.strip_prefix("srid:") {
            return Some(Value::String(srid_string.to_string()));
        }

        // Try to parse the output as JSON such that the resulting column metadata is closer
        // to what would have been in the GeoParquet metadata (e.g., where PROJJSON is parsed).
        if let Ok(value) = crs_str.parse::<Value>() {
            Some(value)
        } else {
            Some(Value::String(crs_str.to_string()))
        }
    } else {
        None
    }
}

/// Helper to get a key from the key/value metadata
fn get_parquet_key_value(key: &str, kv_metadata: Option<&Vec<KeyValue>>) -> Option<String> {
    if let Some(kv_metadata) = kv_metadata {
        for kv in kv_metadata {
            if kv.key == key {
                if let Some(value) = &kv.value {
                    return Some(value.to_string());
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use geo_traits::Dimensions;
    use parquet::basic::{Repetition, Type as PhysicalType};
    use parquet::schema::types::Type;
    use sedona_geometry::types::{GeometryTypeAndDimensions, GeometryTypeId};

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
            GeometryTypeAndDimensions::new(GeometryTypeId::Point, Dimensions::Xy)
        );
    }

    #[test]
    fn test_from_parquet_metadata_with_parquet_types() {
        let s = r#"{
            "version": "2.0.0",
            "primary_column": "geom_geoparquet",
            "columns": {
                "geom_geoparquet": {
                    "encoding": "WKB",
                    "crs": "geom_geoparquet_crs"
                }
            }
        }"#;

        let kv_metadata_with_geo_key = make_kv_metadata(&[("geo", s)]);

        let schema_no_parquet_geo = make_parquet_schema(&[("geom_geoparquet", None)]);
        let metadata = GeoParquetMetadata::try_from_parquet_metadata_impl(
            &schema_no_parquet_geo,
            kv_metadata_with_geo_key.as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(metadata.version, "2.0.0");
        assert_eq!(metadata.primary_column, "geom_geoparquet");
        assert_eq!(metadata.columns.len(), 1);
        assert!(metadata.columns.contains_key("geom_geoparquet"));
        assert_eq!(
            metadata.columns.get("geom_geoparquet").unwrap().crs,
            Some(Value::String("geom_geoparquet_crs".to_string()))
        );

        let schema_additional_parquet_geo = make_parquet_schema(&[
            ("geom_geoparquet", None),
            ("geom_parquet", Some(LogicalType::Geometry { crs: None })),
        ]);
        let metadata = GeoParquetMetadata::try_from_parquet_metadata_impl(
            &schema_additional_parquet_geo,
            kv_metadata_with_geo_key.as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(metadata.columns.len(), 2);
        assert!(metadata.columns.contains_key("geom_geoparquet"));
        assert!(metadata.columns.contains_key("geom_parquet"));

        let schema_overlapping_columns =
            make_parquet_schema(&[("geom_geoparquet", Some(LogicalType::Geometry { crs: None }))]);
        let metadata = GeoParquetMetadata::try_from_parquet_metadata_impl(
            &schema_overlapping_columns,
            kv_metadata_with_geo_key.as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(metadata.columns.len(), 1);
        assert!(metadata.columns.contains_key("geom_geoparquet"));
        // Ensure we use the CRS provided by the GeoParquet metadata instead of the CRS
        // provided by the type as a test that GeoParquet columns take precedence if both
        // are present.
        assert_eq!(
            metadata.columns.get("geom_geoparquet").unwrap().crs,
            Some(Value::String("geom_geoparquet_crs".to_string()))
        );

        let schema_only_parquet_geo =
            make_parquet_schema(&[("geom_parquet", Some(LogicalType::Geometry { crs: None }))]);
        let metadata = GeoParquetMetadata::try_from_parquet_metadata_impl(
            &schema_only_parquet_geo,
            None, // No key/value metadata
        )
        .unwrap()
        .unwrap();
        assert_eq!(metadata.columns.len(), 1);
        assert!(metadata.columns.contains_key("geom_parquet"));
    }

    #[test]
    fn test_column_from_logical_type() {
        let kv_metadata = make_kv_metadata(&[("some_projjson_key", "some_projjson_value")]);

        // A missing logical type annotation is never Geometry or Geography
        assert_eq!(
            column_from_logical_type(None, kv_metadata.as_ref()).unwrap(),
            None
        );

        // Logical type that is present but not Geometry or Geography should return None
        assert_eq!(
            column_from_logical_type(Some(&LogicalType::Uuid), kv_metadata.as_ref()).unwrap(),
            None
        );

        // Geometry logical type
        let metadata = column_from_logical_type(
            Some(&LogicalType::Geometry { crs: None }),
            kv_metadata.as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(metadata, GeoParquetColumnMetadata::default());

        // Ensure CRS is translated
        let metadata = column_from_logical_type(
            Some(&LogicalType::Geometry {
                crs: Some("projjson:some_projjson_key".to_string()),
            }),
            kv_metadata.as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            metadata.crs,
            Some(Value::String("some_projjson_value".to_string()))
        );

        // Geography logical type
        let metadata = column_from_logical_type(
            Some(&LogicalType::Geography {
                crs: None,
                algorithm: None,
            }),
            kv_metadata.as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(metadata.edges, Some("spherical".to_string()));

        // Ensure CRS is translated
        let metadata = column_from_logical_type(
            Some(&LogicalType::Geography {
                crs: Some("projjson:some_projjson_key".to_string()),
                algorithm: None,
            }),
            kv_metadata.as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            metadata.crs,
            Some(Value::String("some_projjson_value".to_string()))
        );

        // Ensure algorithm is translated
        let metadata = column_from_logical_type(
            Some(&LogicalType::Geography {
                crs: None,
                algorithm: Some(EdgeInterpolationAlgorithm::VINCENTY),
            }),
            kv_metadata.as_ref(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(metadata.edges, Some("vincenty".to_string()));
    }

    #[test]
    fn test_crs_from_logical_type() {
        let kv_metadata = make_kv_metadata(&[("some_projjson_key", "some_projjson_value")]);

        // None and "" both map to a GeoParquet default CRS
        assert_eq!(
            geoparquet_crs_from_logical_type(None, kv_metadata.as_ref()),
            None
        );
        assert_eq!(
            geoparquet_crs_from_logical_type(Some(&"".to_string()), kv_metadata.as_ref()),
            None
        );

        // projjson: string should resolve from the key/value metadata or be passed on
        // verbatim if it can't be resolved.
        assert_eq!(
            geoparquet_crs_from_logical_type(None, kv_metadata.as_ref()),
            None
        );
        assert_eq!(
            geoparquet_crs_from_logical_type(
                Some(&"projjson:some_projjson_key".to_string()),
                kv_metadata.as_ref()
            ),
            Some(Value::String("some_projjson_value".to_string()))
        );
        assert_eq!(
            geoparquet_crs_from_logical_type(
                Some(&"projjson:not_in_kv_metadata".to_string()),
                kv_metadata.as_ref()
            ),
            Some(Value::String("projjson:not_in_kv_metadata".to_string()))
        );

        // srid: string should have its prefix stripped
        assert_eq!(
            geoparquet_crs_from_logical_type(Some(&"srid:1234".to_string()), kv_metadata.as_ref()),
            Some(Value::String("1234".to_string()))
        );

        // strings should get passed through verbatim
        assert_eq!(
            geoparquet_crs_from_logical_type(Some(&"EPSG:1234".to_string()), kv_metadata.as_ref()),
            Some(Value::String("EPSG:1234".to_string()))
        );

        // Valid JSON should be parsed
        assert_eq!(
            geoparquet_crs_from_logical_type(
                Some(&"\"EPSG:1234\"".to_string()),
                kv_metadata.as_ref()
            ),
            Some(Value::String("EPSG:1234".to_string()))
        );
    }

    #[test]
    fn test_get_kv_metadata() {
        let kv_metadata = make_kv_metadata(&[("key", "value")]);
        assert_eq!(
            get_parquet_key_value("key", kv_metadata.as_ref()),
            Some("value".to_string())
        );
    }

    // Helper to make a Parquet schema. None here means Binary since it's the only primitive
    // type we need to test
    fn make_parquet_schema(fields: &[(&str, Option<LogicalType>)]) -> parquet::schema::types::Type {
        let fields = fields
            .iter()
            .map(|(name, logical_type)| {
                let mut builder = Type::primitive_type_builder(name, PhysicalType::BYTE_ARRAY)
                    .with_repetition(Repetition::OPTIONAL);
                if let Some(lt) = logical_type {
                    builder = builder.with_logical_type(Some(lt.clone()));
                }
                Arc::new(builder.build().unwrap())
            })
            .collect::<Vec<_>>();

        Type::group_type_builder("schema")
            .with_fields(fields)
            .build()
            .unwrap()
    }

    // Helper to simulate key/value metadata
    fn make_kv_metadata(pairs: &[(&str, &str)]) -> Option<Vec<KeyValue>> {
        Some(
            pairs
                .iter()
                .map(|(key, value)| KeyValue {
                    key: key.to_string(),
                    value: Some(value.to_string()),
                })
                .collect(),
        )
    }
}
