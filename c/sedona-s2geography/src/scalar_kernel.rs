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
use std::{iter::zip, sync::Arc};

use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY},
    matchers::{ArgMatcher, TypeMatcher},
};

use crate::s2geography::S2ScalarUDF;

/// Implementation of ST_Area() for geography using s2geography
pub fn st_area_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Area,
        vec![ArgMatcher::is_geography()],
        SedonaType::Arrow(DataType::Float64),
    )
}

/// Implementation of ST_Centroid() for geography using s2geography
pub fn st_centroid_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Centroid,
        vec![ArgMatcher::is_geography()],
        WKB_GEOGRAPHY,
    )
}

/// Implementation of ST_ClosestPoint() for geography using s2geography
pub fn st_closest_point_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::ClosestPoint,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        WKB_GEOGRAPHY,
    )
}

/// Implementation of ST_Contains() for geography using s2geography
pub fn st_contains_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Contains,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        SedonaType::Arrow(DataType::Boolean),
    )
}

/// Implementation of ST_ConvexHull() for geography using s2geography
pub fn st_convex_hull_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::ConvexHull,
        vec![ArgMatcher::is_geography()],
        WKB_GEOGRAPHY,
    )
}

/// Implementation of ST_Difference() for geography using s2geography
pub fn st_difference_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Difference,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        WKB_GEOGRAPHY,
    )
}

/// Implementation of ST_Distance() for geography using s2geography
pub fn st_distance_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Distance,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        SedonaType::Arrow(DataType::Float64),
    )
}

/// Implementation of ST_Equals() for geography using s2geography
pub fn st_equals_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Equals,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        SedonaType::Arrow(DataType::Boolean),
    )
}

/// Implementation of ST_Intersection() for geography using s2geography
pub fn st_intersection_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Intersection,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        WKB_GEOGRAPHY,
    )
}

/// Implementation of ST_Intersects() for geography using s2geography
pub fn st_intersects_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Intersects,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        SedonaType::Arrow(DataType::Boolean),
    )
}

/// Implementation of ST_Length() for geography using s2geography
pub fn st_length_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Length,
        vec![ArgMatcher::is_geography()],
        SedonaType::Arrow(DataType::Float64),
    )
}

/// Implementation of ST_LineInterpolatePoint() for geography using s2geography
pub fn st_line_interpolate_point_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::LineInterpolatePoint,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_numeric()],
        WKB_GEOGRAPHY,
    )
}

/// Implementation of ST_LineLocatePoint() for geography using s2geography
pub fn st_line_locate_point_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::LineLocatePoint,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        SedonaType::Arrow(DataType::Float64),
    )
}

/// Implementation of ST_MaxDistance() for geography using s2geography
pub fn st_max_distance_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::MaxDistance,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        SedonaType::Arrow(DataType::Float64),
    )
}

/// Implementation of ST_Perimeter() for geography using s2geography
pub fn st_perimeter_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Perimeter,
        vec![ArgMatcher::is_geography()],
        SedonaType::Arrow(DataType::Float64),
    )
}

/// Implementation of ST_ShortestLine() for geography using s2geography
pub fn st_shortest_line_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::ShortestLine,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        WKB_GEOGRAPHY,
    )
}

/// Implementation of ST_SymDifference() for geography using s2geography
pub fn st_sym_difference_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::SymDifference,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        WKB_GEOGRAPHY,
    )
}

/// Implementation of ST_Union() for geography using s2geography
pub fn st_union_impl() -> ScalarKernelRef {
    S2ScalarKernel::new_ref(
        S2ScalarUDF::Union,
        vec![ArgMatcher::is_geography(), ArgMatcher::is_geography()],
        WKB_GEOGRAPHY,
    )
}

#[derive(Debug)]
struct S2ScalarKernel {
    inner_factory: fn() -> S2ScalarUDF,
    matcher: ArgMatcher,
}

impl S2ScalarKernel {
    /// Creates a new reference to a S2ScalarKernel
    pub fn new_ref(
        inner_factory: fn() -> S2ScalarUDF,
        matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>>,
        out_type: SedonaType,
    ) -> ScalarKernelRef {
        Arc::new(Self {
            inner_factory,
            matcher: ArgMatcher::new(matchers, out_type),
        })
    }
}

impl SedonaScalarKernel for S2ScalarKernel {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let mut inner = (self.inner_factory)();

        let arg_types_if_null = self.matcher.types_if_null(arg_types)?;
        let args_casted_null = zip(args, &arg_types_if_null)
            .map(|(arg, type_if_null)| arg.cast_to(type_if_null.storage_type(), None))
            .collect::<Result<Vec<_>>>()?;

        // S2's scalar UDFs operate on fields with extension metadata
        let arg_fields = arg_types_if_null
            .iter()
            .map(|arg_type| arg_type.to_storage_field("", true))
            .collect::<Result<Vec<_>>>()?;

        // Initialize the UDF with a schema consisting of the output fields
        let out_ffi_schema = inner.init(arg_fields.into(), None)?;

        // Create arrays from each argument (scalars become arrays of size 1)
        let arg_arrays = args_casted_null
            .iter()
            .map(|arg| match arg {
                ColumnarValue::Array(array) => Ok(array.clone()),
                ColumnarValue::Scalar(_) => arg.to_array(1),
            })
            .collect::<Result<Vec<_>>>()?;

        // Execute the batch
        let out_ffi_array = inner.execute(&arg_arrays)?;

        // Create the ArrayRef
        let out_array_data = unsafe { arrow_array::ffi::from_ffi(out_ffi_array, &out_ffi_schema)? };
        let out_array = arrow_array::make_array(out_array_data);

        // Ensure scalar inputs map to scalar output
        for arg in args {
            if let ColumnarValue::Array(_) = arg {
                return Ok(ColumnarValue::Array(out_array));
            }
        }

        Ok(ScalarValue::try_from_array(&out_array, 0)?.into())
    }
}

#[cfg(test)]
mod test {

    use arrow_array::ArrayRef;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY};
    use sedona_testing::{
        create::{create_array, create_scalar},
        testers::ScalarUdfTester,
    };

    use super::*;

    #[rstest]
    fn unary_scalar_kernel(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_length", st_length_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Float64)
        );

        // Array -> Array
        let result = tester
            .invoke_wkb_array(vec![
                Some("POINT (0 1)"),
                Some("LINESTRING (0 0, 0 1)"),
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                None,
            ])
            .unwrap();

        let expected: ArrayRef = arrow_array::create_array!(
            Float64,
            [Some(0.0), Some(111195.10117748393), Some(0.0), None]
        );

        assert_eq!(&result, &expected);

        // Scalar -> Scalar
        let result = tester
            .invoke_wkb_scalar(Some("LINESTRING (0 0, 0 1)"))
            .unwrap();
        assert_eq!(result, ScalarValue::Float64(Some(111195.10117748393)));

        // Null scalar -> Null
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert_eq!(result, ScalarValue::Float64(None));
    }

    #[rstest]
    fn binary_scalar_kernel(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_intersects", st_intersects_impl());
        let tester =
            ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type.clone()]);
        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Boolean)
        );

        let point_array = create_array(
            &[Some("POINT (0.25 0.25)"), Some("POINT (10 10)"), None],
            &sedona_type,
        );
        let polygon_scalar = create_scalar(Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"), &sedona_type);
        let point_scalar = create_scalar(Some("POINT (0.25 0.25)"), &sedona_type);

        let expected: ArrayRef =
            arrow_array::create_array!(Boolean, [Some(true), Some(false), None]);

        // Array, Scalar -> Array
        let result = tester
            .invoke_array_scalar(point_array.clone(), polygon_scalar.clone())
            .unwrap();
        assert_eq!(&result, &expected);

        // Scalar, Array -> Array
        let result = tester
            .invoke_scalar_array(polygon_scalar.clone(), point_array.clone())
            .unwrap();
        assert_eq!(&result, &expected);

        // Scalar, Scalar -> Scalar
        let result = tester
            .invoke_scalar_scalar(polygon_scalar, point_scalar)
            .unwrap();
        assert_eq!(result, ScalarValue::Boolean(Some(true)));

        // Null scalars -> Null
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert_eq!(result, ScalarValue::Boolean(None));
    }

    #[test]
    fn area() {
        let udf = SedonaScalarUDF::from_kernel("st_area", st_area_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY]);

        tester.assert_return_type(DataType::Float64);
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, 6182489130.907195);
    }

    #[test]
    fn centroid() {
        let udf = SedonaScalarUDF::from_kernel("st_centroid", st_centroid_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY]);
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester.invoke_scalar("LINESTRING (0 0, 0 1)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 0.5)");
    }

    #[test]
    fn closest_point() {
        let udf = SedonaScalarUDF::from_kernel("st_closestpoint", st_closest_point_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", "POINT (-1 -1)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 0)");
    }

    #[test]
    fn contains() {
        let udf = SedonaScalarUDF::from_kernel("st_contains", st_contains_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(DataType::Boolean);
        let result = tester
            .invoke_scalar_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))", "POINT (0.25 0.25)")
            .unwrap();
        tester.assert_scalar_result_equals(result, true);
    }

    #[test]
    fn difference() {
        let udf = SedonaScalarUDF::from_kernel("st_difference", st_difference_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");
    }

    #[test]
    fn distance() {
        let udf = SedonaScalarUDF::from_kernel("st_distance", st_distance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(DataType::Float64);
        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 0.0);
    }

    #[test]
    fn equals() {
        let udf = SedonaScalarUDF::from_kernel("st_equals", st_equals_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(DataType::Boolean);
        let result1 = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result1, true);
        let result2 = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (0 1)")
            .unwrap();
        tester.assert_scalar_result_equals(result2, false);
    }

    #[test]
    fn intersection() {
        let udf = SedonaScalarUDF::from_kernel("st_intersection", st_intersection_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 0)");
    }

    #[test]
    fn intersects() {
        let udf = SedonaScalarUDF::from_kernel("st_intersects", st_intersects_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(DataType::Boolean);
        let result1 = tester
            .invoke_scalar_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))", "POINT (0.25 0.25)")
            .unwrap();
        tester.assert_scalar_result_equals(result1, true);
        let result2 = tester
            .invoke_scalar_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))", "POINT (-1 -1)")
            .unwrap();
        tester.assert_scalar_result_equals(result2, false);
    }

    #[test]
    fn length() {
        let udf = SedonaScalarUDF::from_kernel("st_length", st_length_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY]);
        tester.assert_return_type(DataType::Float64);
        let result = tester.invoke_scalar("LINESTRING (0 0, 0 1)").unwrap();
        tester.assert_scalar_result_equals(result, 111195.10117748393);
    }

    #[test]
    fn line_interpolate_point() {
        let udf = SedonaScalarUDF::from_kernel(
            "st_lineinterpolatepoint",
            st_line_interpolate_point_impl(),
        );
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOGRAPHY, SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", 0.5)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 0.5)");
    }

    #[test]
    fn line_locate_point() {
        let udf = SedonaScalarUDF::from_kernel("st_linelocatepoint", st_line_locate_point_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(DataType::Float64);
        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", "POINT (0 0.5)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 0.5);
    }

    #[test]
    fn max_distance() {
        let udf = SedonaScalarUDF::from_kernel("st_maxdistance", st_max_distance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(DataType::Float64);
        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (0 0, 0 1)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 111195.10117748393);
    }

    #[test]
    fn perimeter() {
        let udf = SedonaScalarUDF::from_kernel("st_perimeter", st_perimeter_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY]);
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, 379639.8304474758);
    }

    #[test]
    fn shortest_line() {
        let udf = SedonaScalarUDF::from_kernel("st_shortestline", st_shortest_line_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", "POINT (0 -1)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 0 -1)");
    }

    #[test]
    fn sym_difference() {
        let udf = SedonaScalarUDF::from_kernel("st_symdifference", st_sym_difference_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");
    }

    #[test]
    fn union() {
        let udf = SedonaScalarUDF::from_kernel("st_union", st_union_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 0)");
    }
}
