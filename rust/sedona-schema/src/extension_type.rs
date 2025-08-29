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
use std::{collections::HashMap, sync::Arc};

use arrow_array::{ArrayRef, StructArray};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::error::Result;
use datafusion_common::internal_err;
use datafusion_common::scalar::ScalarValue;
use datafusion_expr::ColumnarValue;

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

    /// Wrap this ExtensionType as a Struct DataType
    ///
    /// This is the representation required internally until DataFusion can represent
    /// a non-standard Arrow type. This representation is a Struct that contains exactly
    /// one field whose name is the extension name and whose field metadata contains the
    /// extension name and metadata.
    pub fn to_data_type(&self) -> DataType {
        let field = self.to_field(&self.extension_name, true);
        DataType::Struct(Fields::from(vec![field]))
    }

    /// Wrap storage ColumnarValue as a StructArray
    pub fn wrap_arg(&self, arg: &ColumnarValue) -> Result<ColumnarValue> {
        match arg {
            ColumnarValue::Array(array) => {
                let array_out = self.wrap_array(array.clone())?;
                Ok(ColumnarValue::Array(array_out))
            }
            ColumnarValue::Scalar(scalar) => {
                let scalar_out = self.wrap_scalar(scalar)?;
                Ok(ColumnarValue::Scalar(scalar_out))
            }
        }
    }

    /// Wrap storage array as a StructArray
    pub fn wrap_array(&self, array: ArrayRef) -> Result<ArrayRef> {
        if array.data_type() != &self.storage_type {
            return internal_err!(
                "Type to wrap ({}) does not match storage type ({})",
                array.data_type(),
                &self.storage_type
            );
        }

        let array_data = array.to_data();
        let array_nulls = array_data.nulls();
        let wrapped = StructArray::new(
            vec![self.to_field(&self.extension_name, true)].into(),
            vec![array],
            array_nulls.cloned(),
        );

        Ok(Arc::new(wrapped))
    }

    /// Wrap storage scalar as a StructArray
    pub fn wrap_scalar(&self, scalar: &ScalarValue) -> Result<ScalarValue> {
        let array_in = scalar.to_array()?;
        let array_out = self.wrap_array(array_in)?;
        ScalarValue::try_from_array(&array_out, 0)
    }

    pub fn unwrap_arg(&self, arg: &ColumnarValue) -> Result<ColumnarValue> {
        match arg {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(self.unwrap_array(array)?)),
            ColumnarValue::Scalar(scalar) => Ok(ColumnarValue::Scalar(self.unwrap_scalar(scalar)?)),
        }
    }

    pub fn unwrap_array(&self, array: &ArrayRef) -> Result<ArrayRef> {
        let struct_array = StructArray::from(array.to_data());
        Ok(struct_array.column(0).clone())
    }

    pub fn unwrap_scalar(&self, scalar: &ScalarValue) -> Result<ScalarValue> {
        let array = scalar.to_array()?;
        let struct_array = StructArray::from(array.to_data());
        let array_out = struct_array.column(0).clone();
        ScalarValue::try_from_array(&array_out, 0)
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

    /// Unwrap a DataType that is potentially an extension type wrapped in a Struct
    ///
    /// Returns None if the storage type is not a Struct, if the Struct contains
    /// any number of fields != 1, if its only field is does not contain extension
    /// metadata, or if its extension name does not match the name of the struct.
    pub fn from_data_type(storage_type: &DataType) -> Option<ExtensionType> {
        if let DataType::Struct(fields) = storage_type {
            if fields.len() != 1 {
                return None;
            }

            let field = &fields[0];
            if let Some(extension_type) = ExtensionType::from_field(field) {
                if &extension_type.extension_name == field.name() {
                    return Some(extension_type);
                }
            }
        }

        None
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

    #[test]
    fn extension_type_struct() {
        let ext_type = ExtensionType::new(
            "foofy",
            DataType::Binary,
            Some("foofy metadata".to_string()),
        );
        let ext_struct = &ext_type.to_data_type();
        match ext_struct {
            DataType::Struct(fields) => {
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].name(), "foofy");
            }
            _ => panic!("not a struct"),
        }

        match ExtensionType::from_data_type(ext_struct) {
            Some(ext_type) => {
                assert_eq!(ext_type.extension_name, "foofy");
                assert_eq!(
                    ext_type.extension_metadata,
                    Some("foofy metadata".to_string())
                );
                assert_eq!(ext_type.storage_type, DataType::Binary);
            }
            None => panic!("unwrap did not detect valid extension type"),
        }
    }
}
