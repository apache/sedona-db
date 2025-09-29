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
use std::{str::FromStr, sync::Arc};

use arrow_array::{ArrayRef, BinaryArray, BinaryViewArray};
use datafusion_common::ScalarValue;
use datafusion_expr::ColumnarValue;
use sedona_schema::datatypes::SedonaType;
use wkb::{writer::WriteOptions, Endianness};
use wkt::Wkt;

/// Create a [`ColumnarValue`] array from a sequence of WKT literals
///
/// Panics on invalid WKT or unsupported data type.
pub fn create_array_value(wkt_values: &[Option<&str>], data_type: &SedonaType) -> ColumnarValue {
    ColumnarValue::Array(create_array_storage(wkt_values, data_type))
}

/// Create a [`ColumnarValue`] scalar from a WKT literal
///
/// Panics on invalid WKT or unsupported data type.
pub fn create_scalar_value(wkt_value: Option<&str>, data_type: &SedonaType) -> ColumnarValue {
    ColumnarValue::Scalar(create_scalar_storage(wkt_value, data_type))
}

/// Create a [`ScalarValue`] from a WKT literal
///
/// Panics on invalid WKT or unsupported data type.
pub fn create_scalar(wkt_value: Option<&str>, data_type: &SedonaType) -> ScalarValue {
    create_scalar_storage(wkt_value, data_type)
}

/// Create an [`ArrayRef`] from a sequence of WKT literals
///
/// Panics on invalid WKT or unsupported data type.
pub fn create_array(wkt_values: &[Option<&str>], data_type: &SedonaType) -> ArrayRef {
    create_array_storage(wkt_values, data_type)
}

/// Create the storage [`ArrayRef`] from a sequence of WKT literals
///
/// Panics on invalid WKT or unsupported data type.
pub fn create_array_storage(wkt_values: &[Option<&str>], data_type: &SedonaType) -> ArrayRef {
    match data_type {
        SedonaType::Wkb(_, _) => Arc::new(make_wkb_array::<BinaryArray>(wkt_values)),
        SedonaType::WkbView(_, _) => Arc::new(make_wkb_array::<BinaryViewArray>(wkt_values)),
        _ => panic!("create_array_storage not implemented for {data_type:?}"),
    }
}

/// Create the storage [`ScalarValue`] from a WKT literal
///
/// Panics on invalid WKT or unsupported data type.
pub fn create_scalar_storage(wkt_value: Option<&str>, data_type: &SedonaType) -> ScalarValue {
    match data_type {
        SedonaType::Wkb(_, _) => ScalarValue::Binary(wkt_value.map(make_wkb)),
        SedonaType::WkbView(_, _) => ScalarValue::BinaryView(wkt_value.map(make_wkb)),
        _ => panic!("create_scalar_storage not implemented for {data_type:?}"),
    }
}

fn make_wkb_array<T>(wkt_values: &[Option<&str>]) -> T
where
    T: FromIterator<Option<Vec<u8>>>,
{
    wkt_values
        .iter()
        .map(|maybe_wkt| maybe_wkt.map(make_wkb))
        .collect()
}

/// Create a WKB from a WKT string.
pub fn make_wkb(wkt_value: &str) -> Vec<u8> {
    let geom = Wkt::<f64>::from_str(wkt_value).unwrap();
    let mut out: Vec<u8> = vec![];
    wkb::writer::write_geometry(
        &mut out,
        &geom,
        &WriteOptions {
            endianness: Endianness::LittleEndian,
        },
    )
    .unwrap();
    out
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use datafusion_common::cast::as_binary_array;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};

    use super::*;

    #[test]
    fn scalars() {
        let wkb_scalar = create_scalar_storage(Some("POINT (0 1)"), &WKB_GEOMETRY);
        assert_eq!(&wkb_scalar.data_type(), WKB_GEOMETRY.storage_type());
        assert!(create_scalar_storage(None, &WKB_GEOMETRY).is_null());

        let wkb_view_scalar = create_scalar_storage(Some("POINT (0 1)"), &WKB_VIEW_GEOMETRY);
        assert_eq!(
            &wkb_view_scalar.data_type(),
            WKB_VIEW_GEOMETRY.storage_type()
        );
        assert!(create_scalar_storage(None, &WKB_VIEW_GEOMETRY).is_null());
    }

    #[test]
    #[should_panic(expected = "create_scalar_storage not implemented")]
    fn scalar_storage_invalid() {
        create_scalar_storage(Some("POINT (0 1)"), &SedonaType::Arrow(DataType::Null));
    }

    #[test]
    fn arrays() {
        let wkb_array = create_array_storage(
            &[Some("POINT (0 1)"), None, Some("POINT (1 2)")],
            &WKB_GEOMETRY,
        );
        assert_eq!(wkb_array.data_type(), WKB_GEOMETRY.storage_type());
        assert_eq!(wkb_array.len(), 3);
        let wkb_binary_array = as_binary_array(&wkb_array).unwrap();
        assert_eq!(
            wkb_binary_array
                .iter()
                .map(|maybe_item| maybe_item.is_some())
                .collect::<Vec<bool>>(),
            vec![true, false, true]
        );

        let wkb_array = create_array_storage(
            &[Some("POINT (0 1)"), None, Some("POINT (1 2)")],
            &WKB_VIEW_GEOMETRY,
        );
        assert_eq!(wkb_array.data_type(), WKB_VIEW_GEOMETRY.storage_type());
    }

    #[test]
    #[should_panic(expected = "create_array_storage not implemented")]
    fn array_storage_invalid() {
        create_array_storage(&[Some("POINT (0 1)")], &SedonaType::Arrow(DataType::Null));
    }
}
