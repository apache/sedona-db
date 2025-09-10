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

use arrow_schema::{DataType, Field};

/// Parsed representation of an Arrow extension type
///
/// Because arrow-rs doesn't transport extension names or metadata alongside
/// a DataType, and DataFusion does not provide a built-in mechanism for
/// user-defined types we have to do a bit of wrapping and unwrapping to represent
/// them in a way that can be plugged in to user-defined functions. In particular,
/// we need to be able to:
///
/// - Declare function signatures in such a way that ST_Something(some_geometry)
///   can find the user-defined function implementation (or error if some_geometry
///   is not geometry).
/// - Declare an output type so that geometry can be recognized by the next ST
///   function.
///
/// Strictly speaking we don't need to use the Arrow extension type (i.e., name
/// and metadata) to do this; however, GeoArrow uses it and representing the types
/// in this way means we don't have to try very hard to integrate with geoarrow-rs
/// or geoarrow-c via FFI.
///
/// This wrapping/unwrapping can disappear when there is a built-in logical type
/// and/or DataFusion is better at propagating metadata through various pieces of
/// infrastructure.
#[derive(Debug, PartialEq)]
pub struct ExtensionType {
    pub extension_name: String,
    pub storage_type: DataType,
    pub extension_metadata: Option<String>,
}

impl ExtensionType {
    pub fn new(ext_name: &str, storage_type: DataType, extension_metadata: Option<String>) -> Self {
        let extension_name = ext_name.to_string();
        Self {
            extension_name,
            storage_type,
            extension_metadata,
        }
    }

    /// Wraps this ExtensionType as a Field whose data_type is the actual storage type
    ///
    /// This is how an Arrow extension type would be normally wrapped if it were a column
    /// in a RecordBatch.
    pub fn to_field(&self, name: &str, nullable: bool) -> Field {
        let mut field = Field::new(name, self.storage_type.clone(), nullable);
        let mut metadata = HashMap::from([(
            "ARROW:extension:name".to_string(),
            self.extension_name.clone(),
        )]);

        if let Some(extension_metadata) = &self.extension_metadata {
            metadata.insert(
                "ARROW:extension:metadata".to_string(),
                extension_metadata.clone(),
            );
        }

        field.set_metadata(metadata);
        field
    }

    /// Unwrap a Field into an ExtensionType if the field represents one
    ///
    /// Returns None if the field does not have Arrow extension metadata
    /// for the extension name. This is the inverse of to_field().
    pub fn from_field(field: &Field) -> Option<ExtensionType> {
        let metadata = field.metadata();

        metadata.get("ARROW:extension:name").map(|extension_name| {
            ExtensionType::new(
                extension_name,
                field.data_type().clone(),
                metadata.get("ARROW:extension:metadata").cloned(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_type_field() {
        let ext_type = ExtensionType::new("foofy", DataType::Binary, None);

        let field = ext_type.to_field("some name", true);
        assert_eq!(field.name(), "some name");
        assert_eq!(*field.data_type(), DataType::Binary);

        let metadata = field.metadata();
        assert_eq!(metadata.len(), 1);
        assert!(metadata.contains_key("ARROW:extension:name"));
        assert_eq!(metadata["ARROW:extension:name"], "foofy");
    }

    #[test]
    fn extension_type_field_with_metadata() {
        let ext_type = ExtensionType::new(
            "foofy",
            DataType::Binary,
            Some("foofy metadata".to_string()),
        );
        let field = ext_type.to_field("some name", true);
        let metadata = field.metadata();
        assert_eq!(metadata.len(), 2);
        assert!(metadata.contains_key("ARROW:extension:name"));
        assert_eq!(metadata["ARROW:extension:name"], "foofy");
        assert!(metadata.contains_key("ARROW:extension:metadata"));
        assert_eq!(metadata["ARROW:extension:metadata"], "foofy metadata");
    }
}
