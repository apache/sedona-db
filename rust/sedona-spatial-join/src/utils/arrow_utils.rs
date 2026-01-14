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
        total_size += count_array_data_memory_size(&array_data)?;
    }

    Ok(total_size)
}

/// Estimate the in-memory size of a given Arrow array. This function estimates the
/// size as if the underlying buffers were copied to somewhere else and not shared.
pub(crate) fn get_array_memory_size(array: &ArrayRef) -> Result<usize> {
    let array_data = array.to_data();
    let size = count_array_data_memory_size(&array_data)?;
    Ok(size)
}

/// The maximum number of bytes that can be stored inline in a byte view.
///
/// See [`ByteView`] and [`GenericByteViewArray`] for more information on the
/// layout of the views.
///
/// [`GenericByteViewArray`]: https://docs.rs/arrow/latest/arrow/array/struct.GenericByteViewArray.html
pub const MAX_INLINE_VIEW_LEN: u32 = 12;

/// Count the memory usage of `array_data` and its children recursively.
fn count_array_data_memory_size(array_data: &ArrayData) -> core::result::Result<usize, ArrowError> {
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

    for child in array_data.child_data() {
        result += get_binary_view_value_size(child)?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BinaryViewArray, StringViewArray, StructArray};
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
    }
}
