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

use std::collections::HashMap;

use arrow_schema::{DataType, Schema};
use datafusion_common::{DFSchema, Result};

use crate::{datatypes::SedonaType, matchers::ArgMatcher};

pub trait SedonaSchema {
    /// Iterate over the fields of this schema as parsed [SedonaType]s
    fn sedona_types(&self) -> impl ExactSizeIterator<Item = Result<SedonaType>>;

    /// Return the indices of the columns that are geometry or geography
    fn geometry_column_indices(&self) -> Result<Vec<usize>>;

    /// Return the index of the column that should be considered the "primary" geometry
    ///
    /// This applies a heuritic to detect the "primary" geometry column for operations
    /// that need this information (e.g., creating a GeoPandas GeoDataFrame). The
    /// heuristic chooses (1) the column named "geometry", (2) the column name
    /// "geography", (3) the column named "geom", (4) the column named "geog",
    /// or (5) the first column with a geometry or geography data type.
    fn primary_geometry_column_index(&self) -> Result<Option<usize>>;

    /// Return the indices of top-level columns that are, or contain nested, a
    /// geometry or geography.
    ///
    /// Unlike [`Self::geometry_column_indices`], which only matches columns
    /// whose top-level type is geometry or geography, this descends into
    /// struct/list/map children so that nested geometry (for example the
    /// `List<Struct<path, geom>>` produced by `ST_Dump`) is also reported.
    fn geometry_column_indices_recursive(&self) -> Result<Vec<usize>> {
        let mut indices = Vec::new();
        for (i, sedona_type) in self.sedona_types().enumerate() {
            if sedona_type_contains_geometry(&sedona_type?)? {
                indices.push(i);
            }
        }
        Ok(indices)
    }
}

/// Return whether a parsed [SedonaType] is, or nests, a geometry or geography.
///
/// Descends into struct/list/map children, so nested geometry (for example the
/// `List<Struct<path, geom>>` produced by `ST_Dump`) is detected.
pub fn sedona_type_contains_geometry(sedona_type: &SedonaType) -> Result<bool> {
    match sedona_type {
        SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _) => Ok(true),
        SedonaType::Arrow(data_type) => data_type_contains_geometry(data_type),
        _ => Ok(false),
    }
}

fn data_type_contains_geometry(data_type: &DataType) -> Result<bool> {
    match data_type {
        DataType::Struct(fields) => {
            for field in fields {
                if sedona_type_contains_geometry(&SedonaType::from_storage_field(field)?)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        DataType::List(field) | DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            sedona_type_contains_geometry(&SedonaType::from_storage_field(field)?)
        }
        DataType::Map(field, _) => {
            sedona_type_contains_geometry(&SedonaType::from_storage_field(field)?)
        }
        _ => Ok(false),
    }
}

impl SedonaSchema for DFSchema {
    fn sedona_types(&self) -> impl ExactSizeIterator<Item = Result<SedonaType>> {
        let arrow_schema = self.as_arrow();
        <Schema as SedonaSchema>::sedona_types(arrow_schema)
    }

    fn geometry_column_indices(&self) -> Result<Vec<usize>> {
        let arrow_schema = self.as_arrow();
        <Schema as SedonaSchema>::geometry_column_indices(arrow_schema)
    }

    fn primary_geometry_column_index(&self) -> Result<Option<usize>> {
        let arrow_schema = self.as_arrow();
        <Schema as SedonaSchema>::primary_geometry_column_index(arrow_schema)
    }
}

impl SedonaSchema for Schema {
    fn sedona_types(&self) -> impl ExactSizeIterator<Item = Result<SedonaType>> {
        self.fields()
            .iter()
            .map(|f| SedonaType::from_storage_field(f))
    }

    fn geometry_column_indices(&self) -> Result<Vec<usize>> {
        let mut indices = Vec::new();
        let matcher = ArgMatcher::is_geometry_or_geography();
        for (i, sedona_type) in self.sedona_types().enumerate() {
            if matcher.match_type(&sedona_type?) {
                indices.push(i);
            }
        }

        Ok(indices)
    }

    fn primary_geometry_column_index(&self) -> Result<Option<usize>> {
        let indices = self.geometry_column_indices()?;
        let primary_index_opt =
            primary_geometry_column_from_names(indices.iter().map(|i| self.field(*i).name()));
        if let Some(primary_index) = primary_index_opt {
            Ok(Some(indices[primary_index]))
        } else {
            Ok(None)
        }
    }
}

/// Compute the primary geometry column given a list of geometry column names
///
/// This implementation powers [SedonaSchema::primary_geometry_column_index] and is
/// useful for applying a consistent heuristic when only a list of names are
/// available (e.g., GeoParquet metadata).
pub fn primary_geometry_column_from_names(
    column_names: impl DoubleEndedIterator<Item = impl AsRef<str>>,
) -> Option<usize> {
    let names_map = column_names
        .rev()
        .enumerate()
        .map(|(i, name)| (name.as_ref().to_lowercase(), i))
        .collect::<HashMap<_, _>>();

    if names_map.is_empty() {
        return None;
    }

    for special_name in ["geometry", "geography", "geom", "geog"] {
        if let Some(i) = names_map.get(special_name) {
            return Some(names_map.len() - *i - 1);
        }
    }

    Some(0)
}

#[cfg(test)]
mod test {
    use arrow_schema::{DataType, Field};

    use crate::datatypes::{WKB_GEOGRAPHY, WKB_GEOMETRY};

    use super::*;

    #[test]
    fn sedona_types() {
        let schema = Schema::new(vec![
            WKB_GEOGRAPHY.to_storage_field("geog", true).unwrap(),
            WKB_GEOMETRY.to_storage_field("geom", true).unwrap(),
            Field::new("one", DataType::Int32, true),
        ]);
        let df_schema: DFSchema = schema.clone().try_into().unwrap();

        let sedona_types = schema.sedona_types().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(
            sedona_types,
            vec![
                WKB_GEOGRAPHY,
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Int32)
            ]
        );

        let sedona_types = df_schema
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(
            sedona_types,
            vec![
                WKB_GEOGRAPHY,
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Int32)
            ]
        );
    }

    #[test]
    fn geometry_columns() {
        // No geometry column
        let schema = Schema::new(vec![Field::new("one", DataType::Int32, true)]);
        let df_schema: DFSchema = schema.clone().try_into().unwrap();
        assert!(schema.geometry_column_indices().unwrap().is_empty());
        assert!(schema.primary_geometry_column_index().unwrap().is_none());
        assert!(df_schema.geometry_column_indices().unwrap().is_empty());
        assert!(df_schema.primary_geometry_column_index().unwrap().is_none());

        // Should list geometry and geography but pick geom as the primary column
        let schema = Schema::new(vec![
            WKB_GEOGRAPHY.to_storage_field("geog", true).unwrap(),
            WKB_GEOMETRY.to_storage_field("geom", true).unwrap(),
        ]);
        assert_eq!(schema.geometry_column_indices().unwrap(), vec![0, 1]);
        assert_eq!(schema.primary_geometry_column_index().unwrap(), Some(1));

        // ...but should still detect a column without a special name
        let schema = Schema::new(vec![WKB_GEOMETRY
            .to_storage_field("name_not_special_cased", true)
            .unwrap()]);
        assert_eq!(schema.geometry_column_indices().unwrap(), vec![0]);
        assert_eq!(schema.primary_geometry_column_index().unwrap(), Some(0));
    }

    #[test]
    fn geometry_columns_recursive() {
        // A geometry nested inside a struct/list (e.g. ST_Dump output) is not a
        // top-level geometry column, but the recursive variant reports it.
        let nested = Field::new(
            "parts",
            DataType::List(
                Field::new_struct(
                    "item",
                    vec![
                        Field::new("path", DataType::Int32, true),
                        WKB_GEOMETRY.to_storage_field("geom", true).unwrap(),
                    ],
                    true,
                )
                .into(),
            ),
            true,
        );
        let schema = Schema::new(vec![Field::new("id", DataType::Int32, true), nested]);
        let df_schema: DFSchema = schema.clone().try_into().unwrap();

        // Top-level detection misses the nested geometry; recursive finds it.
        assert!(schema.geometry_column_indices().unwrap().is_empty());
        assert_eq!(schema.geometry_column_indices_recursive().unwrap(), vec![1]);
        assert_eq!(
            df_schema.geometry_column_indices_recursive().unwrap(),
            vec![1]
        );

        // A plain top-level geometry column is still reported.
        let schema = Schema::new(vec![WKB_GEOMETRY
            .to_storage_field("geometry", true)
            .unwrap()]);
        assert_eq!(schema.geometry_column_indices_recursive().unwrap(), vec![0]);
    }
}
