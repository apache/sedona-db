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

use arrow_schema::Schema;
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
}

impl SedonaSchema for DFSchema {
    fn sedona_types(&self) -> impl ExactSizeIterator<Item = Result<SedonaType>> {
        self.as_arrow().sedona_types()
    }

    fn geometry_column_indices(&self) -> Result<Vec<usize>> {
        self.as_arrow().geometry_column_indices()
    }

    fn primary_geometry_column_index(&self) -> Result<Option<usize>> {
        self.as_arrow().primary_geometry_column_index()
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
        if indices.is_empty() {
            return Ok(None);
        }

        let names_map = indices
            .iter()
            .rev()
            .map(|i| (self.field(*i).name().to_lowercase(), *i))
            .collect::<HashMap<_, _>>();

        for special_name in ["geometry", "geography", "geom", "geog"] {
            if let Some(i) = names_map.get(special_name) {
                return Ok(Some(*i));
            }
        }

        Ok(Some(indices[0]))
    }
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
}
