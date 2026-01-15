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

use arrow::array::{Array, ArrayData, RecordBatch};
use arrow_array::ArrayRef;
use arrow_schema::{ArrowError, DataType};
use datafusion_common::Result;

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
    use arrow_array::builder::{BinaryViewBuilder, ListBuilder};
    use arrow_array::types::Int32Type;
    use arrow_array::{BinaryViewArray, ListArray, StringViewArray, StructArray};
    use arrow_schema::{DataType, Field};
    use std::sync::Arc;

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
        let expected_bv_slice0 = 1 * OFFSET_BYTES + 2 * VIEW_BYTES + long1.len();
        let expected_i32_slice0 = 1 * OFFSET_BYTES + 3 * I32_BYTES;
        assert_eq!(
            get_array_memory_size(&slice0).unwrap(),
            expected_bv_slice0 + expected_i32_slice0
        );

        // Slice: second struct row only
        let slice1 = struct_ref.slice(1, 1);
        let expected_bv_slice1 = 1 * OFFSET_BYTES + 1 * VIEW_BYTES + long2.len();
        let expected_i32_slice1 = 1 * OFFSET_BYTES + 1 * I32_BYTES;
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
}
