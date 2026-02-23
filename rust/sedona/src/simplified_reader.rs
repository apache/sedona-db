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

use std::iter::zip;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{ArrowError, DataType, FieldRef, Schema, SchemaRef, UnionFields};
use datafusion_common::datatype::DataTypeExt;

pub fn simplify_record_batch_reader(
    reader: Box<dyn RecordBatchReader + Send>,
) -> Box<dyn RecordBatchReader + Send> {
    let existing_schema = reader.schema();
    let new_schema = simplify_schema(&existing_schema);
    if new_schema == existing_schema {
        return reader;
    }

    let iter_schema = new_schema.clone();
    let reader_iter = reader.map(move |maybe_batch| {
        simplify_record_batch(&maybe_batch?, iter_schema.clone())
    });

    Box::new(RecordBatchIterator::new(reader_iter, new_schema))
}

fn simplify_schema(schema: &SchemaRef) -> SchemaRef {
    let new_fields = schema
        .fields()
        .iter()
        .cloned()
        .map(simplify_field)
        .collect::<Vec<_>>();
    let new_schema = Schema::new(new_fields).with_metadata(schema.metadata().clone());
    new_schema.into()
}

fn simplify_record_batch(
    batch: &RecordBatch,
    schema: SchemaRef,
) -> Result<RecordBatch, ArrowError> {
    let new_columns = zip(batch.columns(), schema.fields())
        .map(|(col, target)| simplify_array_storage(col, target))
        .collect::<Result<Vec<_>, ArrowError>>()?;
    RecordBatch::try_new(schema, new_columns)
}

fn simplify_array_storage(array: &ArrayRef, target: &FieldRef) -> Result<ArrayRef, ArrowError> {
    datafusion_common::arrow::compute::cast(array, target.data_type())
}

fn simplify_field(field: FieldRef) -> FieldRef {
    let new_type = match field.data_type() {
        DataType::BinaryView => DataType::Binary,
        DataType::Utf8View => DataType::Utf8,
        DataType::Dictionary(_key_type, value_type) => {
            simplify_field(value_type.clone().into_nullable_field_ref())
                .data_type()
                .clone()
        }
        DataType::RunEndEncoded(_run_ends, values) => {
            simplify_field(values.clone()).data_type().clone()
        }
        DataType::ListView(field) | DataType::List(field) => {
            DataType::List(simplify_field(field.clone()))
        }
        DataType::LargeListView(field) | DataType::LargeList(field) => {
            DataType::LargeList(simplify_field(field.clone()))
        }
        DataType::FixedSizeList(field, list_size) => {
            DataType::FixedSizeList(simplify_field(field.clone()), *list_size)
        }
        DataType::Struct(fields) => {
            DataType::Struct(fields.into_iter().cloned().map(simplify_field).collect())
        }
        DataType::Union(union_fields, union_mode) => {
            let new_fields = union_fields
                .iter()
                .map(|(_, field)| simplify_field(field.clone()))
                .collect::<Vec<_>>();
            let new_ids = union_fields.iter().map(|(idx, _)| idx).collect::<Vec<_>>();
            let new_union_fields = UnionFields::new(new_ids, new_fields);
            DataType::Union(new_union_fields, *union_mode)
        }
        DataType::Map(field, is_ordered) => {
            DataType::Map(simplify_field(field.clone()), *is_ordered)
        }
        _ => field.data_type().clone(),
    };

    let new_nullable = if let DataType::RunEndEncoded(_, values) = field.data_type() {
        field.is_nullable() || values.is_nullable()
    } else {
        field.is_nullable()
    };

    field
        .as_ref()
        .clone()
        .with_data_type(new_type)
        .with_nullable(new_nullable)
        .into()
}
