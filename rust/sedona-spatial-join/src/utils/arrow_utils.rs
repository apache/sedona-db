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

use std::sync::Arc;

use arrow::array::{Array, ArrayData, BinaryViewArray, ListArray, RecordBatch, StringViewArray};
use arrow_array::make_array;
use arrow_array::ArrayRef;
use arrow_array::StructArray;
use arrow_schema::SchemaRef;
use arrow_schema::{ArrowError, DataType};
use datafusion_common::Result;
use sedona_common::sedona_internal_err;

/// Checks if the schema contains any view types (Utf8View or BinaryView). Batches
/// with view types may need special handling (e.g. compaction) before spilling
/// or holding in memory for extended periods.
pub(crate) fn schema_contains_view_types(schema: &SchemaRef) -> bool {
    schema
        .flattened_fields()
        .iter()
        .any(|field| matches!(field.data_type(), DataType::Utf8View | DataType::BinaryView))
}

/// Reconstruct `batch` to organize the payload buffers of each `StringViewArray` and
/// `BinaryViewArray` in sequential order by calling `gc()` on them.
///
/// Note this is a workaround until <https://github.com/apache/arrow-rs/issues/7185> is
/// available.
///
/// # Rationale
///
/// The `interleave` kernel does not reconstruct the inner buffers of view arrays by default,
/// leading to non-sequential payload locations. A single payload buffer might be shared by
/// multiple `RecordBatch`es or multiple rows in the same batch might reference scattered
/// locations in a large buffer.
///
/// When writing each batch to disk, the writer has to write all referenced buffers. This
/// causes extra disk reads and writes, and potentially execution failure (e.g. No space left
/// on device).
///
/// # Example
///
/// Before interleaving:
/// batch1 -> buffer1 (large)
/// batch2 -> buffer2 (large)
///
/// interleaved_batch -> buffer1 (sparse access)
///                   -> buffer2 (sparse access)
///
/// Then when spilling the interleaved batch, the writer has to write both buffer1 and buffer2
/// entirely, even if only a few bytes are used.
pub(crate) fn compact_batch(batch: RecordBatch) -> Result<RecordBatch> {
    let mut new_columns: Vec<Arc<dyn Array>> = Vec::with_capacity(batch.num_columns());
    let mut arr_mutated = false;

    for array in batch.columns() {
        let (new_array, mutated) = compact_array(Arc::clone(array))?;
        new_columns.push(new_array);
        arr_mutated |= mutated;
    }

    if arr_mutated {
        Ok(RecordBatch::try_new(batch.schema(), new_columns)?)
    } else {
        Ok(batch)
    }
}

/// Recursively compacts view arrays in `array` by calling `gc()` on them.
/// Returns a tuple of the potentially new array and a boolean indicating
/// whether any compaction was performed.
pub(crate) fn compact_array(array: ArrayRef) -> Result<(ArrayRef, bool)> {
    if let Some(view_array) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok((Arc::new(view_array.gc()), true));
    }
    if let Some(view_array) = array.as_any().downcast_ref::<BinaryViewArray>() {
        return Ok((Arc::new(view_array.gc()), true));
    }

    // Fast path for non-nested arrays
    if !array.data_type().is_nested() {
        return Ok((array, false));
    }

    // Avoid ArrayData -> ArrayRef roundtrips for commonly used data types,
    // including StructArray and ListArray.

    if let Some(struct_array) = array.as_any().downcast_ref::<StructArray>() {
        let mut mutated = false;
        let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(struct_array.num_columns());
        for col in struct_array.columns() {
            let (new_col, col_mutated) = compact_array(Arc::clone(col))?;
            mutated |= col_mutated;
            new_columns.push(new_col);
        }

        if !mutated {
            return Ok((array, false));
        }

        let rebuilt = StructArray::new(
            struct_array.fields().clone(),
            new_columns,
            struct_array.nulls().cloned(),
        );
        return Ok((Arc::new(rebuilt), true));
    }

    if let Some(list_array) = array.as_any().downcast_ref::<ListArray>() {
        let (new_values, mutated) = compact_array(list_array.values().clone())?;
        if !mutated {
            return Ok((array, false));
        }

        let DataType::List(field) = list_array.data_type() else {
            // Defensive: this downcast should only succeed for DataType::List.
            return sedona_internal_err!(
                "ListArray has non-List data type: {:?}",
                list_array.data_type()
            );
        };

        let rebuilt = ListArray::new(
            Arc::clone(field),
            list_array.offsets().clone(),
            new_values,
            list_array.nulls().cloned(),
        );
        return Ok((Arc::new(rebuilt), true));
    }

    // For nested arrays (Map/Dictionary/etc.), recurse into children via ArrayData.
    let data = array.to_data();
    if data.child_data().is_empty() {
        return Ok((array, false));
    }

    let mut mutated = false;
    let mut new_child_data = Vec::with_capacity(data.child_data().len());
    for child in data.child_data().iter() {
        let child_array = make_array(child.clone());
        let (new_child_array, child_mutated) = compact_array(child_array)?;
        mutated |= child_mutated;
        new_child_data.push(new_child_array.to_data());
    }

    if !mutated {
        return Ok((array, false));
    }

    // Rebuild this array with identical buffers/nulls but replaced child_data.
    let mut builder = data.into_builder();
    builder = builder.child_data(new_child_data);
    let new_data = builder.build()?;
    Ok((make_array(new_data), true))
}

/// Estimate the in-memory size of a given RecordBatch. This function estimates the
/// size as if the underlying buffers were copied to somewhere else and not shared.
pub(crate) fn get_record_batch_memory_size(batch: &RecordBatch) -> Result<usize> {
    let mut total_size = 0;

    for array in batch.columns() {
        let array_data = array.to_data();
        total_size += get_array_data_memory_size(&array_data)?;
    }

    Ok(total_size)
}

/// Estimate the in-memory size of a given Arrow array. This function estimates the
/// size as if the underlying buffers were copied to somewhere else and not shared,
/// including the sizes of each BinaryView item (which is otherwise not counted by
/// `array_data.get_slice_memory_size()`).
pub(crate) fn get_array_memory_size(array: &ArrayRef) -> Result<usize> {
    let array_data = array.to_data();
    let size = get_array_data_memory_size(&array_data)?;
    Ok(size)
}

/// The maximum number of bytes that can be stored inline in a byte view.
///
/// See [`ByteView`] and [`GenericByteViewArray`] for more information on the
/// layout of the views.
///
/// [`GenericByteViewArray`]: https://docs.rs/arrow/latest/arrow/array/struct.GenericByteViewArray.html
pub const MAX_INLINE_VIEW_LEN: u32 = 12;

/// Compute the memory usage of `array_data` and its children recursively.
fn get_array_data_memory_size(array_data: &ArrayData) -> core::result::Result<usize, ArrowError> {
    // The `ArrayData::get_slice_memory_size` method does not account for the memory used by
    // the values of BinaryView/Utf8View arrays, so we need to compute that using
    // `get_binary_view_value_size` and add that to the total size.
    Ok(get_binary_view_value_size(array_data)? + array_data.get_slice_memory_size()?)
}

fn get_binary_view_value_size(array_data: &ArrayData) -> Result<usize, ArrowError> {
    let mut result: usize = 0;
    let array_data_type = array_data.data_type();

    if matches!(array_data_type, DataType::BinaryView | DataType::Utf8View) {
        // The views buffer contains length view structures with the following layout:
        // https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-view-layout
        //
        // * Short strings, length <= 12
        // | Bytes 0-3  | Bytes 4-15                            |
        // |------------|---------------------------------------|
        // | length     | data (padded with 0)                  |
        //
        // * Long strings, length > 12
        // | Bytes 0-3  | Bytes 4-7  | Bytes 8-11 | Bytes 12-15 |
        // |------------|------------|------------|-------------|
        // | length     | prefix     | buf. index | offset      |
        let views = &array_data.buffer::<u128>(0)[..array_data.len()];
        result = views
            .iter()
            .map(|v| {
                let len = *v as u32;
                if len > MAX_INLINE_VIEW_LEN {
                    len as usize
                } else {
                    0
                }
            })
            .sum();
    }

    // If this was not a BinaryView/Utf8View array, count the bytes of any BinaryView/Utf8View
    // children, taking into account the slice of this array that applies to the child.
    for child in array_data.child_data() {
        result += get_binary_view_value_size(child)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringViewBuilder;
    use arrow_array::builder::{BinaryViewBuilder, ListBuilder};
    use arrow_array::types::Int32Type;
    use arrow_array::{
        BinaryViewArray, BooleanArray, ListArray, StringArray, StringViewArray, StructArray,
    };
    use arrow_schema::{DataType, Field, Fields, Schema};
    use std::sync::Arc;

    fn make_schema(fields: Vec<Field>) -> SchemaRef {
        Arc::new(Schema::new(fields))
    }

    #[test]
    fn test_schema_contains_view_types_top_level() {
        let schema_ref = make_schema(vec![
            Field::new("a", DataType::Utf8View, true),
            Field::new("b", DataType::BinaryView, true),
        ]);

        assert!(schema_contains_view_types(&schema_ref));

        // Similar shape but without view types
        let schema_no_view = make_schema(vec![
            Field::new("a", DataType::Utf8, true),
            Field::new("b", DataType::Binary, true),
        ]);
        assert!(!schema_contains_view_types(&schema_no_view));
    }

    #[test]
    fn test_schema_contains_view_types_nested() {
        let nested = Field::new(
            "s",
            DataType::Struct(Fields::from(vec![Field::new(
                "v",
                DataType::Utf8View,
                true,
            )])),
            true,
        );

        let schema_ref = make_schema(vec![nested]);
        assert!(schema_contains_view_types(&schema_ref));

        // Nested struct without any view types
        let nested_no_view = Field::new(
            "s",
            DataType::Struct(Fields::from(vec![Field::new("v", DataType::Utf8, true)])),
            true,
        );
        let schema_no_view = make_schema(vec![nested_no_view]);
        assert!(!schema_contains_view_types(&schema_no_view));
    }

    #[test]
    fn test_string_view_array_memory_size() {
        let array = StringViewArray::from(vec![
            "short",                                               // Inline
            "Long string that is definitely longer than 12 bytes", // 51 bytes
        ]);
        let array_ref: ArrayRef = Arc::new(array);
        let size = get_array_memory_size(&array_ref).unwrap();
        // Views: 2 * 16 = 32 bytes
        // Data: 51 bytes
        // Total: 83 bytes
        assert_eq!(size, 83);
    }

    #[test]
    fn test_binary_view_array_memory_size() {
        let array = BinaryViewArray::from(vec![
            "short".as_bytes(),
            "Long string that is definitely longer than 12 bytes".as_bytes(),
        ]);
        let array_ref: ArrayRef = Arc::new(array);
        let size = get_array_memory_size(&array_ref).unwrap();
        assert_eq!(size, 83);
    }

    #[test]
    fn test_struct_array_with_view_memory_size() {
        let string_view_array = StringViewArray::from(vec![
            "short",
            "Long string that is definitely longer than 12 bytes",
        ]);
        let boolean_array = arrow_array::BooleanArray::from(vec![true, false]);

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("a", DataType::Utf8View, false)),
                Arc::new(string_view_array) as ArrayRef,
            ),
            (
                Arc::new(Field::new("b", DataType::Boolean, false)),
                Arc::new(boolean_array) as ArrayRef,
            ),
        ]);

        let array_ref: ArrayRef = Arc::new(struct_array);
        let size = get_array_memory_size(&array_ref).unwrap();
        // 83 (StringView) + 1 (Boolean values) = 84
        assert_eq!(size, 84);

        let size = get_array_memory_size(&array_ref.slice(0, 1)).unwrap();
        // 16 (StringView for one short element) + 1 (Boolean values) = 17
        assert_eq!(size, 17);

        let size = get_array_memory_size(&array_ref.slice(1, 1)).unwrap();
        // 67 (StringView for one long element) + 1 (Boolean values) = 68
        assert_eq!(size, 68);
    }

    #[test]
    fn test_sliced_view_array_memory_size() {
        let array = StringViewArray::from(vec![
            "short",
            "Long string that is definitely longer than 12 bytes",
            "Another long string to make buffer larger",
        ]);
        let sliced = array.slice(0, 2);
        let sliced_ref: ArrayRef = Arc::new(sliced);
        let size = get_array_memory_size(&sliced_ref).unwrap();
        // Views: 2 * 16 = 32
        // Data used: 51 ("Long string...")
        // Total: 83
        assert_eq!(size, 83);

        let size = get_array_memory_size(&sliced_ref.slice(1, 1)).unwrap();
        // Views: 1 * 16 = 16
        // Data used: 51 ("Long string...")
        // Total: 67
        assert_eq!(size, 67);

        // Empty slice
        let size = get_array_memory_size(&sliced_ref.slice(2, 0)).unwrap();
        assert_eq!(size, 0);
    }

    fn build_struct_with_list_of_view_and_list_of_i32(
    ) -> (ArrayRef, &'static [u8], &'static [u8], &'static [u8]) {
        let short: &'static [u8] = b"short";
        let long1: &'static [u8] = b"Long string that is definitely longer than 12 bytes";
        let long2: &'static [u8] = b"Another long string to make buffer larger";

        // Build List<BinaryView> with two list items:
        // 0: [short, long1]
        // 1: [long2]
        let mut bv_list_builder = ListBuilder::new(BinaryViewBuilder::new());
        bv_list_builder.values().append_value(short);
        bv_list_builder.values().append_value(long1);
        bv_list_builder.append(true);
        bv_list_builder.values().append_value(long2);
        bv_list_builder.append(true);
        let bv_list: ListArray = bv_list_builder.finish();
        let bv_list_data_type = bv_list.data_type().clone();

        // Build List<Int32> with two list items:
        // 0: [1, 2, 3]
        // 1: [4]
        let i32_list: ListArray = ListArray::from_iter_primitive::<Int32Type, _, _>([
            Some(vec![Some(1), Some(2), Some(3)]),
            Some(vec![Some(4)]),
        ]);
        let i32_list_data_type = i32_list.data_type().clone();

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("bv_list", bv_list_data_type, false)),
                Arc::new(bv_list) as ArrayRef,
            ),
            (
                Arc::new(Field::new("i32_list", i32_list_data_type, false)),
                Arc::new(i32_list) as ArrayRef,
            ),
        ]);

        (Arc::new(struct_array) as ArrayRef, short, long1, long2)
    }

    #[test]
    fn test_struct_array_with_list_of_view_and_list_of_i32_memory_size() {
        const VIEW_BYTES: usize = 16; // BinaryView/Utf8View: one u128 per value
        const OFFSET_BYTES: usize = 4; // ListArray offsets are i32
        const I32_BYTES: usize = 4;

        let (struct_ref, _short, long1, long2) = build_struct_with_list_of_view_and_list_of_i32();

        // Full array expected size:
        // - bv list offsets: 2 * 4
        // - bv views: 3 * 16
        // - bv long bytes: long1 + long2 (short is inline)
        // - i32 list offsets: 2 * 4
        // - i32 values: 4 * 4
        let expected_bv_full = 2 * OFFSET_BYTES + 3 * VIEW_BYTES + long1.len() + long2.len();
        let expected_i32_full = 2 * OFFSET_BYTES + 4 * I32_BYTES;
        assert_eq!(
            get_array_memory_size(&struct_ref).unwrap(),
            expected_bv_full + expected_i32_full
        );
    }

    #[test]
    #[ignore = "XFAIL: get_array_memory_size slice accounting for ListArray is incorrect/fragile"]
    fn test_struct_array_with_list_of_view_and_list_of_i32_memory_size_slices_xfail() {
        const VIEW_BYTES: usize = 16; // BinaryView/Utf8View: one u128 per value
        const OFFSET_BYTES: usize = 4; // ListArray offsets are i32
        const I32_BYTES: usize = 4;

        let (struct_ref, _short, long1, long2) = build_struct_with_list_of_view_and_list_of_i32();

        // Slice: first struct row only
        let slice0 = struct_ref.slice(0, 1);
        let expected_bv_slice0 = OFFSET_BYTES + 2 * VIEW_BYTES + long1.len();
        let expected_i32_slice0 = OFFSET_BYTES + 3 * I32_BYTES;
        assert_eq!(
            get_array_memory_size(&slice0).unwrap(),
            expected_bv_slice0 + expected_i32_slice0
        );

        // Slice: second struct row only
        let slice1 = struct_ref.slice(1, 1);
        let expected_bv_slice1 = OFFSET_BYTES + VIEW_BYTES + long2.len();
        let expected_i32_slice1 = OFFSET_BYTES + I32_BYTES;
        assert_eq!(
            get_array_memory_size(&slice1).unwrap(),
            expected_bv_slice1 + expected_i32_slice1
        );

        // Double slice should behave the same as slice(1, 1)
        let double_slice = struct_ref.slice(0, 2).slice(1, 1);
        assert_eq!(
            get_array_memory_size(&double_slice).unwrap(),
            expected_bv_slice1 + expected_i32_slice1
        );
    }

    #[test]
    fn test_compact_batch_recurses_into_struct() {
        let n = 256;
        let long = "x".repeat(2048);

        let mut builder = StringViewBuilder::with_capacity(n);
        for i in 0..n {
            builder.append_value(format!("batch0_{i}_{long}"));
        }
        let string_view_array: ArrayRef = Arc::new(builder.finish());
        let boolean_array: ArrayRef = Arc::new(BooleanArray::from(vec![true; n]));
        let struct_fields = vec![
            Arc::new(Field::new("a", DataType::Utf8View, false)),
            Arc::new(Field::new("b", DataType::Boolean, false)),
        ];
        let struct_array = StructArray::from(vec![
            (
                Arc::clone(&struct_fields[0]),
                Arc::clone(&string_view_array),
            ),
            (Arc::clone(&struct_fields[1]), Arc::clone(&boolean_array)),
        ]);

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "s",
            DataType::Struct(struct_fields.into()),
            false,
        )]));
        let batch0 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(struct_array) as ArrayRef],
        )
        .unwrap();
        let sliced = batch0.slice(0, 1);

        let before = sliced.get_array_memory_size();
        let compacted = compact_batch(sliced.clone()).unwrap();
        let after = compacted.get_array_memory_size();

        assert_eq!(sliced.schema(), compacted.schema());
        assert_eq!(sliced.num_rows(), compacted.num_rows());
        assert!(
            after < before,
            "expected compaction to reduce memory: before={before}, after={after}"
        );
    }

    #[test]
    fn test_compact_batch_without_view_returns_input_as_is() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("flag", DataType::Boolean, true),
        ]));

        let ids: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3]));
        let names: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("c")]));
        let flags: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)]));

        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, names, flags]).unwrap();
        let original = batch.clone();

        let compacted = compact_batch(batch).unwrap();

        // A no-op compaction should preserve the exact schema/column Arcs.
        assert!(Arc::ptr_eq(&original.schema(), &compacted.schema()));
        for i in 0..original.num_columns() {
            assert!(Arc::ptr_eq(original.column(i), compacted.column(i)));
        }
    }

    #[test]
    fn test_compact_array_compacts_struct_containnig_binary_view() {
        let i32_values = Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3]));
        let mut bv_builder = BinaryViewBuilder::new();
        bv_builder.append_value(b"short");
        bv_builder.append_value(b"Long string that is definitely longer than 12 bytes");
        bv_builder.append_value(b"Another long string to make buffer larger");
        let bv: BinaryViewArray = bv_builder.finish();
        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("a", DataType::Int32, false)),
                i32_values as ArrayRef,
            ),
            (
                Arc::new(Field::new("s", DataType::BinaryView, false)),
                Arc::new(bv),
            ),
        ]);

        let array: ArrayRef = Arc::new(struct_array);
        let slice = array.slice(0, 2);
        let before_size = slice.get_array_memory_size();

        let (compacted, mutated) = compact_array(Arc::new(slice)).unwrap();
        assert!(mutated);

        let after_size = compacted.get_array_memory_size();
        assert!(after_size < before_size);
    }

    #[test]
    fn test_compact_array_compacts_list_of_binary_view() {
        // Build a List<BinaryView> with many long values. Then slice the list so it contains
        // only one row; `compact_array` should compact the nested BinaryView values.
        let n = 256;
        let long = b"Long string that is definitely longer than 12 bytes";

        let mut bv_list_builder = ListBuilder::new(BinaryViewBuilder::new());
        for i in 0..n {
            bv_list_builder
                .values()
                .append_value([long, i.to_string().as_bytes()].concat());
            bv_list_builder.append(true);
        }
        let bv_list: ListArray = bv_list_builder.finish();
        let sliced: ArrayRef = Arc::new(bv_list.slice(0, 1));
        let before_size = sliced.get_array_memory_size();

        let (compacted, mutated) = compact_array(Arc::clone(&sliced)).unwrap();
        assert!(mutated);

        let after_size = compacted.get_array_memory_size();
        assert!(after_size <= before_size);
    }

    #[test]
    fn test_compact_array_list_without_view_is_noop() {
        let i32_list: ListArray = ListArray::from_iter_primitive::<Int32Type, _, _>([
            Some(vec![Some(1), Some(2), Some(3)]),
            Some(vec![Some(4)]),
        ]);

        let array: ArrayRef = Arc::new(i32_list);
        let (compacted, mutated) = compact_array(Arc::clone(&array)).unwrap();
        assert!(!mutated);
        assert!(Arc::ptr_eq(&array, &compacted));
    }

    #[test]
    fn test_compact_array_struct_without_view_is_noop() {
        let i32_values = Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3]));
        let bool_values = Arc::new(BooleanArray::from(vec![true, false, true]));
        let i32_list: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>([
            Some(vec![Some(1), Some(2)]),
            Some(vec![Some(3)]),
            None,
        ]));

        let struct_array = StructArray::from(vec![
            (
                Arc::new(Field::new("a", DataType::Int32, false)),
                i32_values as ArrayRef,
            ),
            (
                Arc::new(Field::new("b", DataType::Boolean, false)),
                bool_values as ArrayRef,
            ),
            (
                Arc::new(Field::new("c", i32_list.data_type().clone(), true)),
                i32_list,
            ),
        ]);

        let array: ArrayRef = Arc::new(struct_array);
        let (compacted, mutated) = compact_array(Arc::clone(&array)).unwrap();
        assert!(!mutated);
        assert!(Arc::ptr_eq(&array, &compacted));
    }
}
