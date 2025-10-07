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

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::{
    cast::{as_binary_array, as_binary_view_array},
    ScalarValue,
};
use datafusion_expr::ColumnarValue;
use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY};

use crate::create::create_scalar;

/// Assert two [`ColumnarValue`]s are equal
///
/// Panics if the values' Scalar/Array status is different or if the content
/// is not equal. This can be used in place of `assert_eq!()` to generate reasonable
/// failure messages for geometry values where the default failure message would
/// otherwise be uninformative.
///
/// This is intended to be used with `create_array_value()` functions
/// for geometry types. It can be used with `create_array!()` and Arrow types as well;
/// however, `assert_eq!()` is usually sufficient for those cases.
pub fn assert_value_equal(actual: &ColumnarValue, expected: &ColumnarValue) {
    match (actual, expected) {
        (ColumnarValue::Array(actual_array), ColumnarValue::Array(expected_array)) => {
            assert_array_equal(actual_array, expected_array);
        }
        (ColumnarValue::Scalar(actual_scalar), ColumnarValue::Scalar(expected_scalar)) => {
            assert_scalar_equal(actual_scalar, expected_scalar);
        }
        (ColumnarValue::Array(_), ColumnarValue::Scalar(_)) => {
            panic!("ColumnarValues not equal: actual is Array, expected Scalar");
        }
        (ColumnarValue::Scalar(_), ColumnarValue::Array(_)) => {
            panic!("ColumnarValues not equal: actual is Scalar, expected Array");
        }
    }
}

/// Assert two [`ArrayRef`]s are equal
///
/// Panics if the values' length or types are different or if the content is otherwise not
/// equal. This can be used in place of `assert_eq!()` to generate reasonable
/// failure messages for geometry arrays where the default failure message would
/// otherwise be uninformative.
pub fn assert_array_equal(actual: &ArrayRef, expected: &ArrayRef) {
    let (actual_sedona, expected_sedona) = assert_type_equal(
        actual.data_type(),
        expected.data_type(),
        "actual Array",
        "expected Array",
    );

    if actual.len() != expected.len() {
        panic!(
            "Lengths not equal: actual Array has length {}, expected Array has length {}",
            actual.len(),
            expected.len()
        )
    }

    match (&actual_sedona, &expected_sedona) {
        (SedonaType::Arrow(_), SedonaType::Arrow(_)) => {
            assert_eq!(actual, expected)
        }

        (SedonaType::Wkb(_, _), SedonaType::Wkb(_, _)) => {
            assert_wkb_sequences_equal(
                as_binary_array(&actual).unwrap(),
                as_binary_array(&expected).unwrap(),
            );
        }
        (SedonaType::WkbView(_, _), SedonaType::WkbView(_, _)) => {
            assert_wkb_sequences_equal(
                as_binary_view_array(&actual).unwrap(),
                as_binary_view_array(&expected).unwrap(),
            );
        }
        (_, _) => {
            unreachable!()
        }
    }
}

/// Assert a [`ScalarValue`] is a WKB_GEOMETRY scalar corresponding to the given WKT
///
/// Panics if the values' are not equal, generating reasonable failure messages for geometry
/// arrays where the default failure message would otherwise be uninformative.
pub fn assert_scalar_equal_wkb_geometry(actual: &ScalarValue, expected_wkt: Option<&str>) {
    let expected = create_scalar(expected_wkt, &WKB_GEOMETRY);
    assert_eq!(actual.data_type(), DataType::Binary);
    assert_wkb_scalar_equal(actual, &expected, false);
}

/// Assert a [`ScalarValue`] is a WKB_GEOMETRY scalar corresponding to the given WKT. This function
/// compares the geometries topologically, so two geometries that are not byte-wise equal but are
/// topologically equal will be considered equal.
///
/// Panics if the values' are not topologically equal, generating reasonable failure messages for geometry
/// arrays where the default failure message would otherwise be uninformative.
#[cfg(feature = "geo")]
pub fn assert_scalar_equal_wkb_geometry_topologically(
    actual: &ScalarValue,
    expected_wkt: Option<&str>,
) {
    let expected = create_scalar(expected_wkt, &WKB_GEOMETRY);
    assert_eq!(actual.data_type(), DataType::Binary);
    assert_wkb_scalar_equal(actual, &expected, true);
}

/// Assert two [`ScalarValue`]s are equal
///
/// Panics if the values' are not equal, generating reasonable failure messages for geometry
/// arrays where the default failure message would otherwise be uninformative.
pub fn assert_scalar_equal(actual: &ScalarValue, expected: &ScalarValue) {
    let (actual_sedona, expected_sedona) = assert_type_equal(
        &actual.data_type(),
        &expected.data_type(),
        "actual ScalarValue",
        "expected ScalarValue",
    );

    match (&actual_sedona, &expected_sedona) {
        (SedonaType::Arrow(_), SedonaType::Arrow(_)) => assert_arrow_scalar_equal(actual, expected),
        (SedonaType::Wkb(_, _), SedonaType::Wkb(_, _))
        | (SedonaType::WkbView(_, _), SedonaType::WkbView(_, _)) => {
            assert_wkb_scalar_equal(actual, expected, false);
        }
        (_, _) => unreachable!(),
    }
}

fn assert_type_equal(
    actual: &DataType,
    expected: &DataType,
    actual_label: &str,
    expected_label: &str,
) -> (SedonaType, SedonaType) {
    let actual_sedona = SedonaType::Arrow(actual.clone());
    let expected_sedona = SedonaType::Arrow(expected.clone());
    if actual_sedona != expected_sedona {
        panic!(
            "{actual_label} != {expected_label}:\n{actual_label} has type {actual_sedona:?}, {expected_label} has type {expected_sedona:?}"
        );
    }

    (actual_sedona, expected_sedona)
}

fn assert_arrow_scalar_equal(actual: &ScalarValue, expected: &ScalarValue) {
    if actual != expected {
        panic!("Arrow ScalarValues not equal:\nactual is {actual:?}, expected {expected:?}")
    }
}

fn assert_wkb_sequences_equal<'a, 'b, TActual, TExpected>(actual: TActual, expected: TExpected)
where
    TActual: IntoIterator<Item = Option<&'a [u8]>>,
    TExpected: IntoIterator<Item = Option<&'b [u8]>>,
{
    for (i, (actual_item, expected_item)) in zip(actual, expected).enumerate() {
        let actual_label = format!("actual Array element #{i}");
        let expected_label = format!("expected Array element #{i}");
        assert_wkb_value_equal(
            actual_item,
            expected_item,
            &actual_label,
            &expected_label,
            false,
        );
    }
}

fn assert_wkb_scalar_equal(
    actual: &ScalarValue,
    expected: &ScalarValue,
    compare_topologically: bool,
) {
    match (actual, expected) {
        (ScalarValue::Binary(maybe_actual_wkb), ScalarValue::Binary(maybe_expected_wkb))
        | (
            ScalarValue::BinaryView(maybe_actual_wkb),
            ScalarValue::BinaryView(maybe_expected_wkb),
        ) => {
            assert_wkb_value_equal(
                maybe_actual_wkb.as_deref(),
                maybe_expected_wkb.as_deref(),
                "actual WKB scalar",
                "expected WKB scalar",
                compare_topologically,
            );
        }
        (_, _) => {
            unreachable!()
        }
    }
}

fn assert_wkb_value_equal(
    actual: Option<&[u8]>,
    expected: Option<&[u8]>,
    actual_label: &str,
    expected_label: &str,
    compare_topologically: bool,
) {
    match (actual, expected) {
        (None, None) => {}
        (None, Some(expected_wkb)) => {
            panic!(
                "{actual_label} != {expected_label}:\n{actual_label} is null, {expected_label} is {}",
                format_wkb(expected_wkb)
            )
        }
        (Some(actual_wkb), None) => {
            panic!(
                "{actual_label} != {expected_label}:\n{actual_label} is {}, {expected_label} is null",
                format_wkb(actual_wkb)
            )
        }
        (Some(actual_wkb), Some(expected_wkb)) => {
            // Quick test: if the binary of the WKB is the same, they are equal
            if actual_wkb != expected_wkb {
                let is_equals = if compare_topologically {
                    compare_wkb_topologically(expected_wkb, actual_wkb)
                } else {
                    false
                };

                if !is_equals {
                    let (actual_wkt, expected_wkt) =
                        (format_wkb(actual_wkb), format_wkb(expected_wkb));
                    panic!("{actual_label} != {expected_label}\n{actual_label}:\n  {actual_wkt}\n{expected_label}:\n  {expected_wkt}")
                }
            }
        }
    }
}

fn compare_wkb_topologically(
    #[allow(unused)] expected_wkb: &[u8],
    #[allow(unused)] actual_wkb: &[u8],
) -> bool {
    #[cfg(feature = "geo")]
    {
        use geo::Relate;
        use geo_traits::to_geo::ToGeoGeometry;
        use geo_traits::Dimensions;
        use geo_traits::GeometryTrait;

        let expected = wkb::reader::read_wkb(expected_wkb);
        let actual = wkb::reader::read_wkb(actual_wkb);
        match (expected, actual) {
            (Ok(expected_geom), Ok(actual_geom)) => {
                if expected_geom.dim() == Dimensions::Xy && actual_geom.dim() == Dimensions::Xy {
                    let expected_geom = expected_geom.to_geometry();
                    let actual_geom = actual_geom.to_geometry();
                    expected_geom.relate(&actual_geom).is_equal_topo()
                } else {
                    // geo crate does not support 3D/4D geometry operations, so we fall back to using the result
                    // of byte-wise comparison
                    false
                }
            }
            _ => false,
        }
    }
    #[cfg(not(feature = "geo"))]
    {
        panic!("Topological comparison requires the 'geo' feature to be enabled");
    }
}

fn format_wkb(value: &[u8]) -> String {
    if let Ok(geom) = wkb::reader::read_wkb(value) {
        let mut wkt = String::new();
        wkt::to_wkt::write_geometry(&mut wkt, &geom).unwrap();
        wkt
    } else {
        format!("Invalid WKB: {value:?}")
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::create_array;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};

    use crate::create::{create_array, create_array_value, create_scalar, create_scalar_value};

    use super::*;

    // For lower-level tests
    const POINT: [u8; 21] = [
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];

    #[test]
    fn values_equal() {
        assert_value_equal(
            &create_scalar_value(Some("POINT (0 1)"), &WKB_GEOMETRY),
            &create_scalar_value(Some("POINT (0 1)"), &WKB_GEOMETRY),
        );
        assert_value_equal(
            &create_array_value(&[Some("POINT (0 1)")], &WKB_GEOMETRY),
            &create_array_value(&[Some("POINT (0 1)")], &WKB_GEOMETRY),
        );
    }

    #[test]
    #[should_panic(expected = "ColumnarValues not equal: actual is Scalar, expected Array")]
    fn values_expected_scalar() {
        assert_value_equal(
            &create_scalar_value(None, &WKB_GEOMETRY),
            &create_array_value(&[], &WKB_GEOMETRY),
        );
    }

    #[test]
    #[should_panic(expected = "ColumnarValues not equal: actual is Array, expected Scalar")]
    fn values_expected_array() {
        assert_value_equal(
            &create_array_value(&[], &WKB_GEOMETRY),
            &create_scalar_value(None, &WKB_GEOMETRY),
        );
    }

    #[test]
    fn arrays_equal() {
        let arrow: ArrayRef = create_array!(Utf8, [Some("foofy"), None, Some("foofy2")]);
        let wkbs = [Some("POINT (0 1)"), None, Some("POINT (1 2)")];
        assert_array_equal(&arrow, &arrow);

        assert_array_equal(
            &create_array(&wkbs, &WKB_GEOMETRY),
            &create_array(&wkbs, &WKB_GEOMETRY),
        );

        assert_array_equal(
            &create_array(&wkbs, &WKB_VIEW_GEOMETRY),
            &create_array(&wkbs, &WKB_VIEW_GEOMETRY),
        );
    }

    #[test]
    #[should_panic(
        expected = "Lengths not equal: actual Array has length 1, expected Array has length 0"
    )]
    fn arrays_different_length() {
        assert_array_equal(
            &create_array(&[None], &WKB_GEOMETRY),
            &create_array(&[], &WKB_GEOMETRY),
        );
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed
  left: StringArray
[
  \"foofy\",
  null,
]
 right: StringArray
[
  null,
  \"foofy\",
]")]
    fn arrays_arrow_not_equal() {
        let lhs: ArrayRef = create_array!(Utf8, [Some("foofy"), None]);
        let rhs: ArrayRef = create_array!(Utf8, [None, Some("foofy")]);
        assert_array_equal(&lhs, &rhs);
    }

    #[test]
    fn scalars_equal() {
        assert_scalar_equal(
            &ScalarValue::Utf8(Some("foofy".to_string())),
            &ScalarValue::Utf8(Some("foofy".to_string())),
        );
        assert_scalar_equal(
            &create_scalar(Some("POINT (0 1)"), &WKB_GEOMETRY),
            &create_scalar(Some("POINT (0 1)"), &WKB_GEOMETRY),
        );
        assert_scalar_equal(
            &create_scalar(Some("POINT (0 1)"), &WKB_VIEW_GEOMETRY),
            &create_scalar(Some("POINT (0 1)"), &WKB_VIEW_GEOMETRY),
        );
    }

    #[test]
    #[should_panic(expected = "Arrow ScalarValues not equal:
actual is Utf8(\"foofy\"), expected Utf8(\"not foofy\")")]
    fn scalars_unequal_arrow() {
        assert_scalar_equal(
            &ScalarValue::Utf8(Some("foofy".to_string())),
            &ScalarValue::Utf8(Some("not foofy".to_string())),
        );
    }

    #[test]
    fn sequences_equal() {
        let sequence: Vec<Option<&[u8]>> = vec![Some(&POINT), None, Some(&[])];
        assert_wkb_sequences_equal(sequence.clone(), sequence);
    }

    #[test]
    #[should_panic(expected = "actual Array element #0 != expected Array element #0:
actual Array element #0 is POINT(1 2), expected Array element #0 is null")]
    fn sequences_with_difference() {
        let lhs: Vec<Option<&[u8]>> = vec![Some(&POINT), None, Some(&[])];
        let rhs: Vec<Option<&[u8]>> = vec![None, Some(&POINT), Some(&[])];
        assert_wkb_sequences_equal(lhs, rhs);
    }

    #[test]
    fn wkb_value_equal() {
        assert_wkb_value_equal(None, None, "lhs", "rhs", false);
        assert_wkb_value_equal(Some(&[]), Some(&[]), "lhs", "rhs", false);
    }

    #[test]
    #[should_panic(expected = "lhs != rhs:\nlhs is POINT(1 2), rhs is null")]
    fn wkb_value_expected_null() {
        assert_wkb_value_equal(Some(&POINT), None, "lhs", "rhs", false);
    }

    #[test]
    #[should_panic(expected = "lhs != rhs:\nlhs is null, rhs is POINT(1 2)")]
    fn wkb_value_actual_null() {
        assert_wkb_value_equal(None, Some(&POINT), "lhs", "rhs", false);
    }

    #[test]
    #[should_panic(expected = "lhs != rhs
lhs:
  Invalid WKB: []
rhs:
  POINT(1 2)")]
    fn wkb_value_values_not_equal() {
        assert_wkb_value_equal(Some(&[]), Some(&POINT), "lhs", "rhs", false);
    }

    #[cfg(feature = "geo")]
    #[test]
    fn wkb_value_equal_topologically() {
        use crate::create::make_wkb;
        assert_wkb_value_equal(Some(&POINT), Some(&POINT), "lhs", "rhs", true);
        let lhs = make_wkb("POLYGON ((0 0, 1 0, 0 1, 0 0))");
        let rhs = make_wkb("POLYGON ((0 0, 0 1, 1 0, 0 0))");
        assert_wkb_value_equal(Some(&lhs), Some(&rhs), "lhs", "rhs", true);
    }

    #[cfg(feature = "geo")]
    #[test]
    #[should_panic(expected = "lhs != rhs
lhs:
  POLYGON((0 0,1 0,0 1,0 0))
rhs:
  POLYGON((0 0,1 0,0 0))")]
    fn wkb_value_not_equal_topologically() {
        use crate::create::make_wkb;
        let lhs = make_wkb("POLYGON ((0 0, 1 0, 0 1, 0 0))");
        let rhs = make_wkb("POLYGON ((0 0, 1 0, 0 0))");
        assert_wkb_value_equal(Some(&lhs), Some(&rhs), "lhs", "rhs", true);
    }

    #[test]
    fn wkb_formatter() {
        assert_eq!(format_wkb(&POINT), "POINT(1 2)");
        assert_eq!(format_wkb(&[]), "Invalid WKB: []");
    }
}
