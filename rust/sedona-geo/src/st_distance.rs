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

use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_common::scalar::ScalarValue;
use datafusion_expr::ColumnarValue;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_functions::executor::WkbBytesExecutor;
use sedona_geo_generic_alg::line_measures::DistanceExt;
use sedona_geometry::error::SedonaGeometryError;
use sedona_geometry::wkb_header::WkbPointLayout;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::{read_wkb, Wkb};

/// ST_Distance() implementation using [DistanceExt]
pub fn st_distance_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STDistance {})
}

#[derive(Debug)]
struct STDistance {}

impl SedonaScalarKernel for STDistance {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbBytesExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        // Parse each *scalar* operand exactly once (capturing its Point xy if it
        // is one) so a scalar is never re-parsed per row, regardless of the
        // other operand's type. Array operands stay `None` and are read per row.
        let scalar0 = PreparedScalar::try_new(&args[0])?;
        let scalar1 = PreparedScalar::try_new(&args[1])?;

        executor.execute_wkb_wkb_void(|maybe_bytes0, maybe_bytes1| {
            match (maybe_bytes0, maybe_bytes1) {
                (Some(bytes0), Some(bytes1)) => {
                    let distance = distance(scalar0.as_ref(), bytes0, scalar1.as_ref(), bytes1)?;
                    builder.append_value(distance);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Distance between two operands, each either a parsed-once scalar or an array
/// element's raw WKB bytes.
fn distance(
    scalar0: Option<&PreparedScalar>,
    bytes0: &[u8],
    scalar1: Option<&PreparedScalar>,
    bytes1: &[u8],
) -> Result<f64> {
    // Point-to-point fast path. A scalar Point already knows its (x, y); an
    // array element only has its header inspected here (no coordinate read).
    // Resolving both before reading any array coordinate means a non-Point
    // operand never triggers a wasted coordinate read, and matched Point
    // coordinates are read without re-parsing the header.
    //
    // Skip it entirely when a scalar operand is a known non-Point: it can never
    // be half of a Point pair, so inspecting the array element's header every
    // row would be pure overhead.
    if !blocks_fast_path(scalar0) && !blocks_fast_path(scalar1) {
        if let (Some(head0), Some(head1)) = (
            PointHead::resolve(scalar0, bytes0),
            PointHead::resolve(scalar1, bytes1),
        ) {
            if let (Some((ax, ay)), Some((bx, by))) = (
                head0.read_xy(bytes0).ok().flatten(),
                head1.read_xy(bytes1).ok().flatten(),
            ) {
                let dx = ax - bx;
                let dy = ay - by;
                return Ok((dx * dx + dy * dy).sqrt());
            }
        }
    }

    // Any non-Point, POINT EMPTY, or malformed input falls back to the general
    // distance, reusing the parsed scalar(s) and parsing only the array bytes.
    // This preserves the exact semantics (and the same error on bad bytes).
    let owned0;
    let wkb0 = match scalar0 {
        Some(scalar) => &scalar.wkb,
        None => {
            owned0 = read_wkb(bytes0).map_err(|e| DataFusionError::External(Box::new(e)))?;
            &owned0
        }
    };
    let owned1;
    let wkb1 = match scalar1 {
        Some(scalar) => &scalar.wkb,
        None => {
            owned1 = read_wkb(bytes1).map_err(|e| DataFusionError::External(Box::new(e)))?;
            &owned1
        }
    };
    Ok(wkb0.distance_ext(wkb1))
}

/// True when `scalar` is a known non-Point operand, so the point-to-point fast
/// path can never apply to any row paired against it.
#[inline]
fn blocks_fast_path(scalar: Option<&PreparedScalar>) -> bool {
    matches!(scalar, Some(scalar) if scalar.xy.is_none())
}

/// A scalar geometry operand parsed once: its full [Wkb] and, if it is a finite
/// Point, its coordinates.
struct PreparedScalar<'a> {
    xy: Option<(f64, f64)>,
    wkb: Wkb<'a>,
}

impl<'a> PreparedScalar<'a> {
    /// Parses `arg` once if it is a scalar geometry; `None` for array arguments
    /// (read per row) and null/unrecognized scalars.
    fn try_new(arg: &'a ColumnarValue) -> Result<Option<Self>> {
        let ColumnarValue::Scalar(scalar) = arg else {
            return Ok(None);
        };
        let Some(bytes) = scalar_wkb_bytes(scalar) else {
            return Ok(None);
        };
        let wkb = read_wkb(bytes).map_err(|e| DataFusionError::External(Box::new(e)))?;
        let xy = WkbPointLayout::try_from_wkb(bytes)
            .ok()
            .flatten()
            .and_then(|layout| layout.read_xy(bytes).ok().flatten());
        Ok(Some(Self { xy, wkb }))
    }
}

/// A resolved Point operand whose coordinates can be read on demand: either a
/// scalar's known coordinates or an array element's header layout.
enum PointHead {
    Scalar(f64, f64),
    Array(WkbPointLayout),
}

impl PointHead {
    /// Resolves an operand to a Point, or `None` if it is not a finite Point.
    /// For an array operand this reads only the header — no coordinate.
    #[inline]
    fn resolve(scalar: Option<&PreparedScalar>, bytes: &[u8]) -> Option<Self> {
        match scalar {
            Some(scalar) => scalar.xy.map(|(x, y)| PointHead::Scalar(x, y)),
            None => WkbPointLayout::try_from_wkb(bytes)
                .ok()
                .flatten()
                .map(PointHead::Array),
        }
    }

    /// Reads the `(x, y)`, returning `Ok(None)` for `POINT EMPTY`.
    #[inline]
    fn read_xy(&self, bytes: &[u8]) -> Result<Option<(f64, f64)>, SedonaGeometryError> {
        match self {
            PointHead::Scalar(x, y) => Ok(Some((*x, *y))),
            PointHead::Array(layout) => layout.read_xy(bytes),
        }
    }
}

/// Extracts the raw WKB bytes from a scalar geometry value, if present.
fn scalar_wkb_bytes(scalar: &ScalarValue) -> Option<&[u8]> {
    match scalar {
        ScalarValue::Binary(maybe)
        | ScalarValue::BinaryView(maybe)
        | ScalarValue::LargeBinary(maybe) => maybe.as_deref(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, Float64Array};
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::{create_array, create_scalar};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        left_sedona_type: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        right_sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_distance", st_distance_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![left_sedona_type.clone(), right_sedona_type.clone()],
        );

        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Float64)
        );

        // Test distance between two points (3-4-5 triangle)
        let point_0_0 = create_scalar(Some("POINT (0 0)"), &left_sedona_type);
        let point_3_4 = create_scalar(Some("POINT (3 4)"), &right_sedona_type);

        let result = tester
            .invoke_scalar_scalar(point_0_0.clone(), point_3_4.clone())
            .unwrap();
        if let ScalarValue::Float64(Some(distance)) = result {
            assert!((distance - 5.0).abs() < 1e-10);
        } else {
            panic!("Expected Float64 result");
        }

        // Test with null values
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, point_3_4.clone())
            .unwrap();
        assert!(result.is_null());
        let result = tester
            .invoke_scalar_scalar(point_0_0.clone(), ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());
    }

    /// Exercises the array/scalar fast-path routing: a scalar operand is parsed
    /// once and reused per row, the point-to-point fast path matches the general
    /// distance, and mixed Point/non-Point inputs fall back correctly. Covered
    /// across storage encodings, including item-CRS (which arrives unwrapped as
    /// plain WKB, so the parse-once scalar handling still applies).
    #[rstest]
    fn array_scalar_routing(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] left: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] right: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_distance", st_distance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![left.clone(), right.clone()]);

        // Array of Points vs a scalar Point: the point-to-point fast path.
        let result = tester
            .invoke_wkb_array_scalar(
                vec![Some("POINT (3 4)"), None, Some("POINT (0 0)")],
                create_scalar(Some("POINT (0 0)"), &right),
            )
            .unwrap();
        let expected: ArrayRef = Arc::new(Float64Array::from(vec![Some(5.0), None, Some(0.0)]));
        assert_array_equal(&result, &expected);

        // Array of Polygons vs a scalar Point: the scalar is parsed once and
        // reused across rows; each row falls back to the general distance. The
        // scalar Point sits on a polygon vertex, so every distance is 0.
        let result = tester
            .invoke_wkb_array_scalar(
                vec![
                    Some("POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))"),
                    Some("POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))"),
                ],
                create_scalar(Some("POINT (10 10)"), &right),
            )
            .unwrap();
        let expected: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.0), Some(0.0)]));
        assert_array_equal(&result, &expected);

        // Scalar Point vs array of Points (mirror orientation).
        let result = tester
            .invoke_scalar_array(
                create_scalar(Some("POINT (0 0)"), &left),
                create_array(&[Some("POINT (3 4)"), Some("POINT (0 0)")], &right),
            )
            .unwrap();
        let expected: ArrayRef = Arc::new(Float64Array::from(vec![Some(5.0), Some(0.0)]));
        assert_array_equal(&result, &expected);

        // Array vs array with mixed Point / Polygon rows.
        let arg0 = create_array(
            &[
                Some("POINT (0 0)"),
                Some("POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))"),
            ],
            &left,
        );
        let arg1 = create_array(&[Some("POINT (3 4)"), Some("POINT (10 10)")], &right);
        let expected: ArrayRef = Arc::new(Float64Array::from(vec![Some(5.0), Some(0.0)]));
        assert_array_equal(&tester.invoke_array_array(arg0, arg1).unwrap(), &expected);
    }
}
