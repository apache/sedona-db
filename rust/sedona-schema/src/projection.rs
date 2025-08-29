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
use arrow_schema::{Field, Schema};

use crate::extension_type::ExtensionType;

/// Wrap a Schema possibly containing Extension Types
///
/// The resulting Schema will have all Extension types wrapped such that they
/// are propagated through operations that only supply a data type (e.g., UDF
/// execution). This is the projection that should be applied to input that
/// might contain extension types.
pub fn wrap_schema(schema: &Schema) -> Schema {
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|field| match ExtensionType::from_field(field) {
            Some(ext) => Field::new(field.name(), ext.to_data_type(), true).into(),
            None => field.clone(),
        })
        .collect();

    Schema::new(fields)
}

/// Unwrap a Schema that contains wrapped extension types
///
/// The resulting schema will have extension types represented with field metadata
/// instead of as wrapped structs. This is the projection that should be applied
/// when writing to output.
pub fn unwrap_schema(schema: &Schema) -> Schema {
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(
            |field| match ExtensionType::from_data_type(field.data_type()) {
                Some(ext) => ext.to_field(field.name(), true).into(),
                None => field.clone(),
            },
        )
        .collect();

    Schema::new(fields)
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;

    use super::*;

    /// An ExtensionType for tests
    pub fn geoarrow_wkt() -> ExtensionType {
        ExtensionType::new("geoarrow.wkt", DataType::Utf8, None)
    }

    #[test]
    fn schema_wrap_unwrap() {
        let schema_normal = Schema::new(vec![
            Field::new("field1", DataType::Boolean, true),
            geoarrow_wkt().to_field("field2", true),
        ]);

        let schema_wrapped = wrap_schema(&schema_normal);
        assert_eq!(schema_wrapped.field(0).name(), "field1");
        assert_eq!(*schema_wrapped.field(0).data_type(), DataType::Boolean);
        assert_eq!(schema_wrapped.field(1).name(), "field2");
        assert!(schema_wrapped.field(1).data_type().is_nested());

        let schema_unwrapped = unwrap_schema(&schema_wrapped);
        assert_eq!(schema_unwrapped, schema_normal);
    }
}
