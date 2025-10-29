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
use arrow_schema::{DataType, Field};
use datafusion_common::error::{DataFusionError, Result};
use sedona_common::sedona_internal_err;
use serde_json::Value;
use std::fmt::{Debug, Display};
use std::sync::LazyLock;

use crate::crs::{deserialize_crs, Crs};
use crate::extension_type::ExtensionType;
use crate::raster::RasterSchema;

/// Data types supported by Sedona that resolve to a concrete Arrow DataType
#[derive(Debug, PartialEq, Clone)]
pub enum SedonaType {
    Arrow(DataType),
    Wkb(Edges, Crs),
    WkbView(Edges, Crs),
    Raster,
}

impl From<DataType> for SedonaType {
    fn from(value: DataType) -> Self {
        Self::Arrow(value)
    }
}

/// Edge interpolations
///
/// While at the logical level we refer to geometries and geographies, at the execution
/// layer we can reuse implementations for structural manipulation more efficiently if
/// we consider the edge interpolation as a parameter of the physical type. This maps to
/// the concept of "edges" in GeoArrow and "algorithm" in Parquet and Iceberg (where the
/// planar case would be resolved to a geometry instead of a geography).
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Edges {
    Planar,
    Spherical,
}

/// Sentinel for [`SedonaType::Wkb`] with planar edges
///
/// This constant is useful when defining type signatures as these ignore the Crs when
/// matching (and `SedonaType::Wkb(...)` is verbose)
pub const WKB_GEOMETRY: SedonaType = SedonaType::Wkb(Edges::Planar, Crs::None);

/// Sentinel for [`SedonaType::WkbView`] with planar edges
///
/// See [`WKB_GEOMETRY`]
pub const WKB_VIEW_GEOMETRY: SedonaType = SedonaType::WkbView(Edges::Planar, Crs::None);

/// Sentinel for [`SedonaType::Wkb`] with spherical edges
///
/// This constant is useful when defining type signatures as these ignore the Crs when
/// matching (and `SedonaType::Wkb(...)` is verbose)
pub const WKB_GEOGRAPHY: SedonaType = SedonaType::Wkb(Edges::Spherical, Crs::None);

/// Sentinel for [`SedonaType::WkbView`] with spherical edges
///
/// See [`WKB_GEOGRAPHY`]
pub const WKB_VIEW_GEOGRAPHY: SedonaType = SedonaType::WkbView(Edges::Spherical, Crs::None);

/// Sentinel for [`SedonaType::Raster`]
pub const RASTER: SedonaType = SedonaType::Raster;

/// Create a static value for the [`SedonaType::Raster`] that's initialized exactly once,
/// on first access
static RASTER_DATATYPE: LazyLock<DataType> =
    LazyLock::new(|| DataType::Struct(RasterSchema::fields()));

// Implementation details

impl SedonaType {
    /// Given a field as it would appear in an external Schema return the appropriate SedonaType
    pub fn from_storage_field(field: &Field) -> Result<SedonaType> {
        match ExtensionType::from_field(field) {
            Some(ext) => Self::from_extension_type(ext),
            None => Ok(Self::Arrow(field.data_type().clone())),
        }
    }

    /// Given an [`ExtensionType`], construct a SedonaType
    pub fn from_extension_type(extension: ExtensionType) -> Result<SedonaType> {
        let (edges, crs) = deserialize_edges_and_crs(&extension.extension_metadata)?;
        if extension.extension_name == "geoarrow.wkb" {
            sedona_type_wkb(edges, crs, extension.storage_type)
        } else if extension.extension_name == "sedona.raster" {
            if extension.storage_type == *RASTER_DATATYPE {
                Ok(RASTER)
            } else {
                sedona_internal_err!(
                    "Extension type sedona.raster has unexpected storage type: {}",
                    extension.storage_type
                )
            }
        } else {
            sedona_internal_err!(
                "Extension type not implemented: <{}>:{}",
                extension.extension_name,
                extension.storage_type
            )
        }
    }

    /// Construct a [`Field`] as it would appear in an external `RecordBatch`
    pub fn to_storage_field(&self, name: &str, nullable: bool) -> Result<Field> {
        self.extension_type().map_or(
            Ok(Field::new(name, self.storage_type().clone(), nullable)),
            |extension| Ok(extension.to_field(name, nullable)),
        )
    }

    /// Compute the storage [`DataType`] as it would appear in an external `RecordBatch`
    pub fn storage_type(&self) -> &DataType {
        match self {
            SedonaType::Arrow(data_type) => data_type,
            SedonaType::Wkb(_, _) => &DataType::Binary,
            SedonaType::WkbView(_, _) => &DataType::BinaryView,
            SedonaType::Raster => &RASTER_DATATYPE,
        }
    }

    /// Compute the extension name if this is an Arrow extension type or `None` otherwise
    pub fn extension_name(&self) -> Option<&'static str> {
        match self {
            SedonaType::Arrow(_) => None,
            SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _) => Some("geoarrow.wkb"),
            SedonaType::Raster => Some("sedona.raster"),
        }
    }

    /// Construct the [`ExtensionType`] that represents this type, if any
    pub fn extension_type(&self) -> Option<ExtensionType> {
        match self {
            SedonaType::Wkb(edges, crs) | SedonaType::WkbView(edges, crs) => {
                Some(ExtensionType::new(
                    self.extension_name().unwrap(),
                    self.storage_type().clone(),
                    Some(serialize_edges_and_crs(edges, crs)),
                ))
            }
            SedonaType::Raster => Some(ExtensionType::new(
                self.extension_name().unwrap(),
                self.storage_type().clone(),
                None,
            )),
            _ => None,
        }
    }

    /// The logical type name for this type
    ///
    /// The logical type name is used in tabular display and schema printing. Notably,
    /// it renders Wkb and WkbView as "geometry" or "geography" depending on the edge
    /// type. For Arrow types, this similarly strips the storage details (e.g.,
    /// both Utf8 and Utf8View types render as "utf8").
    pub fn logical_type_name(&self) -> String {
        match self {
            SedonaType::Wkb(Edges::Planar, _) | SedonaType::WkbView(Edges::Planar, _) => {
                "geometry".to_string()
            }
            SedonaType::Wkb(Edges::Spherical, _) | SedonaType::WkbView(Edges::Spherical, _) => {
                "geography".to_string()
            }
            SedonaType::Raster => "raster".to_string(),
            SedonaType::Arrow(data_type) => match data_type {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "utf8".to_string(),
                DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_) => "binary".to_string(),
                DataType::List(_)
                | DataType::LargeList(_)
                | DataType::ListView(_)
                | DataType::LargeListView(_)
                | DataType::FixedSizeList(_, _) => "list".to_string(),
                DataType::Dictionary(_, value_type) => {
                    SedonaType::Arrow(value_type.as_ref().clone()).logical_type_name()
                }
                DataType::RunEndEncoded(_, value_field) => {
                    match SedonaType::from_storage_field(value_field) {
                        Ok(value_sedona_type) => value_sedona_type.logical_type_name(),
                        Err(_) => format!("{value_field:?}"),
                    }
                }
                _ => {
                    let data_type_str = data_type.to_string();
                    if let Some(params_start) = data_type_str.find('(') {
                        data_type_str[0..params_start].to_string().to_lowercase()
                    } else {
                        data_type_str.to_lowercase()
                    }
                }
            },
        }
    }

    /// Returns True if another physical type matches this one for the purposes of dispatch
    ///
    /// For Arrow types this matches on type equality; for other type it matches on edges
    /// but not crs.
    pub fn match_signature(&self, other: &SedonaType) -> bool {
        match (self, other) {
            (SedonaType::Arrow(data_type), SedonaType::Arrow(other_data_type)) => {
                data_type == other_data_type
            }
            (SedonaType::Wkb(edges, _), SedonaType::Wkb(other_edges, _)) => edges == other_edges,
            (SedonaType::WkbView(edges, _), SedonaType::WkbView(other_edges, _)) => {
                edges == other_edges
            }
            (SedonaType::Raster, SedonaType::Raster) => true,
            _ => false,
        }
    }
}

// Implementation details for type serialization and display

impl Display for SedonaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SedonaType::Arrow(data_type) => Display::fmt(data_type, f),
            SedonaType::Wkb(edges, crs) => display_geometry("Wkb", edges, crs, f),
            SedonaType::WkbView(edges, crs) => display_geometry("WkbView", edges, crs, f),
            SedonaType::Raster => Display::fmt("Raster", f),
        }
    }
}

fn display_geometry(
    name: &str,
    edges: &Edges,
    crs: &Crs,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let mut params = Vec::new();

    if let Some(crs) = crs {
        params.push(crs.to_string());
    }

    match edges {
        Edges::Planar => {}
        Edges::Spherical => {
            params.push("Spherical".to_string());
        }
    }

    match params.len() {
        0 => write!(f, "{name}")?,
        1 => write!(f, "{name}({})", params[0])?,
        _ => write!(f, "{name}({})", params.join(", "))?,
    }

    Ok(())
}

// Implementation details for importing/exporting types from/to Arrow + metadata

/// Check a storage type for SedonaType::Wkb
fn sedona_type_wkb(edges: Edges, crs: Crs, storage_type: DataType) -> Result<SedonaType> {
    match storage_type {
        DataType::Binary => Ok(SedonaType::Wkb(edges, crs)),
        DataType::BinaryView => Ok(SedonaType::WkbView(edges, crs)),
        _ => sedona_internal_err!(
            "Expected Wkb type with Binary storage but got {}",
            storage_type
        ),
    }
}

/// Parse a GeoArrow metadata string
///
/// Deserializes the extension metadata from a GeoArrow extension type. See
/// https://geoarrow.org/extension-types.html for a full definition of the metadata
/// format.
fn deserialize_edges_and_crs(value: &Option<String>) -> Result<(Edges, Crs)> {
    match value {
        Some(val) => {
            if val.is_empty() || val == "{}" {
                return Ok((Edges::Planar, Crs::None));
            }

            let json_value: Value = serde_json::from_str(val).map_err(|err| {
                DataFusionError::Internal(format!("Error deserializing GeoArrow metadata: {err}"))
            })?;
            if !json_value.is_object() {
                return sedona_internal_err!(
                    "Expected GeoArrow metadata as JSON object but got {}",
                    val
                );
            }

            let edges = match json_value.get("edges") {
                Some(edges_value) => deserialize_edges(edges_value)?,
                None => Edges::Planar,
            };

            let crs = match json_value.get("crs") {
                Some(crs_value) => deserialize_crs(crs_value)?,
                None => Crs::None,
            };

            Ok((edges, crs))
        }
        None => Ok((Edges::Planar, Crs::None)),
    }
}

/// Create a GeoArrow metadata string
///
/// Deserializes the extension metadata from a GeoArrow extension type. See
/// https://geoarrow.org/extension-types.html for a full definition of the metadata
/// format.
fn serialize_edges_and_crs(edges: &Edges, crs: &Crs) -> String {
    let crs_component = crs
        .as_ref()
        .map(|crs| format!(r#""crs":{}"#, crs.to_json()));

    let edges_component = match edges {
        Edges::Planar => None,
        Edges::Spherical => Some(r#""edges":"spherical""#),
    };

    match (crs_component, edges_component) {
        (None, None) => "{}".to_string(),
        (None, Some(edges)) => format!("{{{edges}}}"),
        (Some(crs), None) => format!("{{{crs}}}"),
        (Some(crs), Some(edges)) => format!("{{{edges},{crs}}}"),
    }
}

/// Deserialize a specific GeoArrow "edges" value
fn deserialize_edges(edges: &Value) -> Result<Edges> {
    match edges.as_str() {
        Some(edges_str) => {
            if edges_str == "planar" {
                Ok(Edges::Planar)
            } else if edges_str == "spherical" {
                Ok(Edges::Spherical)
            } else {
                sedona_internal_err!("Unsupported edges value {}", edges_str)
            }
        }
        None => {
            sedona_internal_err!("Unsupported edges JSON type in metadata {}", edges)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::crs::lnglat;

    use super::*;

    #[test]
    fn sedona_type_arrow() {
        let sedona_type = SedonaType::Arrow(DataType::Int32);
        assert_eq!(sedona_type.storage_type(), &DataType::Int32);
        assert_eq!(sedona_type, SedonaType::Arrow(DataType::Int32));
        assert!(sedona_type.match_signature(&SedonaType::Arrow(DataType::Int32)));
        assert!(!sedona_type.match_signature(&SedonaType::Arrow(DataType::Utf8)));
    }

    #[test]
    fn sedona_type_wkb() {
        assert_eq!(WKB_GEOMETRY, WKB_GEOMETRY);
        assert_eq!(
            SedonaType::from_storage_field(&WKB_GEOMETRY.to_storage_field("", true).unwrap())
                .unwrap(),
            WKB_GEOMETRY
        );

        assert!(WKB_GEOMETRY.match_signature(&WKB_GEOMETRY));
    }

    #[test]
    fn sedona_type_wkb_view() {
        assert_eq!(WKB_VIEW_GEOMETRY.storage_type(), &DataType::BinaryView);
        assert_eq!(WKB_VIEW_GEOGRAPHY.storage_type(), &DataType::BinaryView);

        assert_eq!(WKB_VIEW_GEOMETRY, WKB_VIEW_GEOMETRY);
        assert_eq!(WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOGRAPHY);

        let storage_field = WKB_VIEW_GEOMETRY.to_storage_field("", true).unwrap();
        assert_eq!(
            SedonaType::from_storage_field(&storage_field).unwrap(),
            WKB_VIEW_GEOMETRY
        );
    }

    #[test]
    fn sedona_type_wkb_geography() {
        assert_eq!(WKB_GEOGRAPHY, WKB_GEOGRAPHY);
        assert_eq!(
            SedonaType::from_storage_field(&WKB_GEOGRAPHY.to_storage_field("", true).unwrap())
                .unwrap(),
            WKB_GEOGRAPHY
        );

        assert!(WKB_GEOGRAPHY.match_signature(&WKB_GEOGRAPHY));
        assert!(!WKB_GEOGRAPHY.match_signature(&WKB_GEOMETRY));
    }

    #[test]
    fn sedona_type_to_string() {
        assert_eq!(SedonaType::Arrow(DataType::Int32).to_string(), "Int32");
        assert_eq!(WKB_GEOMETRY.to_string(), "Wkb");
        assert_eq!(WKB_GEOGRAPHY.to_string(), "Wkb(Spherical)");
        assert_eq!(WKB_VIEW_GEOMETRY.to_string(), "WkbView");
        assert_eq!(WKB_VIEW_GEOGRAPHY.to_string(), "WkbView(Spherical)");
        assert_eq!(
            SedonaType::Wkb(Edges::Planar, lnglat()).to_string(),
            "Wkb(ogc:crs84)"
        );

        let projjson_value: Value = r#"{}"#.parse().unwrap();
        let projjson_crs = deserialize_crs(&projjson_value).unwrap();
        assert_eq!(
            SedonaType::Wkb(Edges::Planar, projjson_crs).to_string(),
            "Wkb({...})"
        );
        assert_eq!(RASTER.to_string(), "Raster");
    }

    #[test]
    fn sedona_logical_type_name() {
        assert_eq!(WKB_GEOMETRY.logical_type_name(), "geometry");
        assert_eq!(WKB_GEOGRAPHY.logical_type_name(), "geography");

        assert_eq!(
            SedonaType::Arrow(DataType::Int32).logical_type_name(),
            "int32"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::Utf8).logical_type_name(),
            "utf8"
        );
        assert_eq!(
            SedonaType::Arrow(DataType::Utf8View).logical_type_name(),
            "utf8"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::Binary).logical_type_name(),
            "binary"
        );
        assert_eq!(
            SedonaType::Arrow(DataType::BinaryView).logical_type_name(),
            "binary"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::Duration(arrow_schema::TimeUnit::Microsecond))
                .logical_type_name(),
            "duration"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::List(
                Field::new("item", DataType::Int32, true).into()
            ))
            .logical_type_name(),
            "list"
        );
        assert_eq!(
            SedonaType::Arrow(DataType::ListView(
                Field::new("item", DataType::Int32, true).into()
            ))
            .logical_type_name(),
            "list"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::Dictionary(
                Box::new(DataType::Int32),
                Box::new(DataType::Binary)
            ))
            .logical_type_name(),
            "binary"
        );

        assert_eq!(
            SedonaType::Arrow(DataType::RunEndEncoded(
                Field::new("ends", DataType::Int32, true).into(),
                Field::new("values", DataType::Binary, true).into()
            ))
            .logical_type_name(),
            "binary"
        );
    }

    #[test]
    fn geoarrow_serialize() {
        assert_eq!(serialize_edges_and_crs(&Edges::Planar, &Crs::None), "{}");
        assert_eq!(
            serialize_edges_and_crs(&Edges::Planar, &lnglat()),
            r#"{"crs":"OGC:CRS84"}"#
        );
        assert_eq!(
            serialize_edges_and_crs(&Edges::Spherical, &Crs::None),
            r#"{"edges":"spherical"}"#
        );
        assert_eq!(
            serialize_edges_and_crs(&Edges::Spherical, &lnglat()),
            r#"{"edges":"spherical","crs":"OGC:CRS84"}"#
        );
    }

    #[test]
    fn geoarrow_serialize_roundtrip() -> Result<()> {
        // Check configuration resulting in empty metadata
        assert_eq!(
            deserialize_edges_and_crs(&Some(serialize_edges_and_crs(&Edges::Planar, &Crs::None)))?,
            (Edges::Planar, Crs::None)
        );

        // Check configuration with non-empty metadata for both edges and crs
        assert_eq!(
            deserialize_edges_and_crs(&Some(serialize_edges_and_crs(
                &Edges::Spherical,
                &lnglat()
            )))?,
            (Edges::Spherical, lnglat())
        );

        Ok(())
    }

    #[test]
    fn geoarrow_deserialize_invalid() {
        let bad_json =
            ExtensionType::new("geoarrow.wkb", DataType::Binary, Some(r#"{"#.to_string()));
        assert!(SedonaType::from_extension_type(bad_json)
            .unwrap_err()
            .message()
            .contains("Error deserializing GeoArrow metadata"));

        let bad_type =
            ExtensionType::new("geoarrow.wkb", DataType::Binary, Some(r#"[]"#.to_string()));
        assert!(SedonaType::from_extension_type(bad_type)
            .unwrap_err()
            .message()
            .contains("Expected GeoArrow metadata as JSON object"));

        let bad_edges_type = ExtensionType::new(
            "geoarrow.wkb",
            DataType::Binary,
            Some(r#"{"edges": []}"#.to_string()),
        );
        assert!(SedonaType::from_extension_type(bad_edges_type)
            .unwrap_err()
            .message()
            .contains("Unsupported edges JSON type"));

        let bad_edges_value = ExtensionType::new(
            "geoarrow.wkb",
            DataType::Binary,
            Some(r#"{"edges": "gazornenplat"}"#.to_string()),
        );
        assert!(SedonaType::from_extension_type(bad_edges_value)
            .unwrap_err()
            .message()
            .contains("Unsupported edges value"));
    }
}
