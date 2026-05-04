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

//! A wrapper around [`Column`] that preserves field metadata in `return_field()`.
//!
//! DataFusion 52 has an issue where field metadata (like `ARROW:extension:name`)
//! is stripped when evaluating embedded projections in ParquetOpener. This happens
//! because `Column::return_field()` looks up fields from the input schema, which
//! comes from the parquet reader and lacks extension metadata.
//!
//! This wrapper stores the correct [`FieldRef`] with metadata and returns it
//! from `return_field()` regardless of the input schema, allowing projection
//! pushdown to work correctly while preserving GeoArrow extension metadata.

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, FieldRef, Schema};
use datafusion_common::Result;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::{expressions::Column, PhysicalExpr};
use sedona_common::sedona_internal_err;

/// A wrapper around [`Column`] that preserves field metadata in `return_field()`.
///
/// This expression delegates all operations to the inner [`Column`] except for
/// `return_field()`, which returns the stored [`FieldRef`] with correct metadata
/// instead of looking it up from the input schema.
#[derive(Clone, Debug)]
pub struct MetadataPreservingColumn {
    /// The inner Column expression that performs the actual column access
    inner: Arc<dyn PhysicalExpr>,
    /// The field with correct metadata to return from `return_field()`
    field: FieldRef,
}

impl Hash for MetadataPreservingColumn {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
        self.field.hash(state);
    }
}

impl PartialEq for MetadataPreservingColumn {
    fn eq(&self, other: &Self) -> bool {
        self.inner.as_ref().dyn_eq(other.inner.as_any()) && self.field == other.field
    }
}

impl Eq for MetadataPreservingColumn {}

impl MetadataPreservingColumn {
    /// Create a new MetadataPreservingColumn wrapping the given Column
    /// and using the provided field (with correct metadata) for `return_field()`.
    pub fn new(inner: Column, field: FieldRef) -> Self {
        Self {
            inner: Arc::new(inner),
            field,
        }
    }

    /// Access the wrapped expression
    pub fn inner(&self) -> &Arc<dyn PhysicalExpr> {
        &self.inner
    }
}

impl Display for MetadataPreservingColumn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetadataPreservingColumn({})", self.inner)
    }
}

impl PhysicalExpr for MetadataPreservingColumn {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        // Return data type from our stored field instead of looking up from input schema
        // This avoids index mismatch issues when the schema differs from the original
        Ok(self.field.data_type().clone())
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        // Return nullability from our stored field instead of looking up from input schema
        Ok(self.field.is_nullable())
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        // Delegate to inner Column
        self.inner.evaluate(batch)
    }

    fn return_field(&self, _input_schema: &Schema) -> Result<FieldRef> {
        // Return our stored field with correct metadata instead of looking up
        // from the input schema (which may lack extension metadata)
        Ok(Arc::clone(&self.field))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.inner]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if children.len() != 1 {
            return sedona_internal_err!("MetadataPreservingColumn expects exactly one child");
        }
        Ok(Arc::new(Self {
            inner: children[0].clone(),
            field: self.field.clone(),
        }))
    }

    fn fmt_sql(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt_sql(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_schema::{DataType, Field};
    use std::collections::HashMap;

    #[test]
    fn test_metadata_preserving_column_return_field() {
        // Create a schema without metadata (simulating parquet reader schema)
        let input_schema = Schema::new(vec![Field::new("geometry", DataType::Binary, true)]);

        // Create a field with extension metadata (the correct field we want to preserve)
        let mut metadata = HashMap::new();
        metadata.insert(
            "ARROW:extension:name".to_string(),
            "geoarrow.wkb".to_string(),
        );
        let correct_field: FieldRef =
            Arc::new(Field::new("geometry", DataType::Binary, true).with_metadata(metadata));

        // Create the wrapper
        let column = Column::new("geometry", 0);
        let wrapper = MetadataPreservingColumn::new(column, Arc::clone(&correct_field));

        // return_field should return the stored field with metadata, not the input schema's field
        let returned_field = wrapper.return_field(&input_schema).unwrap();
        assert!(returned_field
            .metadata()
            .contains_key("ARROW:extension:name"));
        assert_eq!(
            returned_field.metadata().get("ARROW:extension:name"),
            Some(&"geoarrow.wkb".to_string())
        );
    }

    #[test]
    fn test_metadata_preserving_column_evaluate() {
        let schema = Schema::new(vec![Field::new("col", DataType::Utf8, true)]);
        let data: StringArray = vec!["test"].into();
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(data)]).unwrap();

        let field: FieldRef = Arc::new(Field::new("col", DataType::Utf8, true));
        let column = Column::new("col", 0);
        let wrapper = MetadataPreservingColumn::new(column, field);

        // Evaluate should work correctly
        let result = wrapper.evaluate(&batch).unwrap();
        match result {
            ColumnarValue::Array(arr) => {
                assert_eq!(arr.len(), 1);
            }
            _ => panic!("Expected Array"),
        }
    }

    #[test]
    fn test_metadata_preserving_column_equality() {
        let field1: FieldRef = Arc::new(Field::new("col", DataType::Utf8, true));
        let field2: FieldRef = Arc::new(Field::new("col", DataType::Utf8, true));

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());
        let field3: FieldRef =
            Arc::new(Field::new("col", DataType::Utf8, true).with_metadata(metadata));

        let wrapper1 = MetadataPreservingColumn::new(Column::new("col", 0), field1);
        let wrapper2 = MetadataPreservingColumn::new(Column::new("col", 0), field2);
        let wrapper3 = MetadataPreservingColumn::new(Column::new("col", 0), field3);

        // Same column, same field (no metadata) -> equal
        assert_eq!(wrapper1, wrapper2);
        // Same column, different metadata -> not equal
        assert_ne!(wrapper1, wrapper3);
    }
}
