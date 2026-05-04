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

use std::ffi::c_int;
use std::ptr;

use crate::geography::Geography;
use crate::s2geog_call;
use crate::s2geog_check;
use crate::s2geography_c_bindgen::*;
use crate::utils::S2GeogCError;

/// S2Geography operator types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[repr(i32)]
pub enum OpType {
    /// Tests whether two geometries intersect
    Intersects = S2GEOGRAPHY_OP_INTERSECTS,
    /// Tests whether the first geometry contains the second
    Contains = S2GEOGRAPHY_OP_CONTAINS,
    /// Tests whether the first geometry is within the second
    Within = S2GEOGRAPHY_OP_WITHIN,
    /// Tests whether two geometries are equal
    Equals = S2GEOGRAPHY_OP_EQUALS,
    /// Tests whether two geometries are within some distance
    DWithin = S2GEOGRAPHY_OP_DISTANCE_WITHIN,
}

impl OpType {
    fn as_c_int(self) -> c_int {
        self as c_int
    }
}

/// S2Geography output types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[repr(i32)]
pub enum OutputType {
    Unknown = 0,
    Bool = S2GEOGRAPHY_OUTPUT_TYPE_BOOL,
}

/// Safe wrapper around S2GeogOp for evaluating spatial operations
///
/// This struct represents an operator that can be applied
/// to various combinations of arguments. If possible one
/// should use the kernels, which have lower overhead and operate more
/// directly on the input when it arrives as Arrow batches.
///
/// Operations are an abstraction that help amortize overhead when evaluating
/// the same operation many times and reduce the API surface of the s2geography
/// C API when the Arrow interface is not sufficient.
pub struct Op {
    ptr: *mut S2GeogOp,
}

impl Op {
    /// Create a new spatial predicate operator
    pub fn new(op_type: OpType) -> Self {
        let mut ptr: *mut S2GeogOp = ptr::null_mut();
        unsafe { s2geog_check!(S2GeogOpCreate(&mut ptr, op_type.as_c_int())) }.unwrap();
        Self { ptr }
    }

    /// Get the name of this operator
    pub fn name(&self) -> String {
        unsafe {
            let c_str = S2GeogOpName(self.ptr);
            if c_str.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(c_str)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }

    /// Get the output type of this operator
    pub fn output_type(&self) -> OutputType {
        let output_type_int = unsafe { S2GeogOpOutputType(self.ptr) };
        match output_type_int {
            S2GEOGRAPHY_OUTPUT_TYPE_BOOL => OutputType::Bool,
            _ => OutputType::Unknown,
        }
    }

    /// Evaluate a predicate on two geographies
    ///
    /// Evaluate an operation that accepts two geographies and returns a boolean
    pub fn eval_binary_predicate(
        &mut self,
        geog0: &Geography,
        geog1: &Geography,
    ) -> Result<bool, S2GeogCError> {
        unsafe {
            s2geog_call!(S2GeogOpEvalGeogGeog(
                self.ptr,
                geog0.as_ptr(),
                geog1.as_ptr()
            ))?;
            Ok(S2GeogOpGetInt(self.ptr) != 0)
        }
    }

    /// Evaluate a distance predicate on two geographies
    ///
    /// Evaluate an operation that accepts two geographies + a double and returns a boolean
    pub fn eval_binary_distance_predicate(
        &mut self,
        geog0: &Geography,
        geog1: &Geography,
        distance: f64,
    ) -> Result<bool, S2GeogCError> {
        unsafe {
            s2geog_call!(S2GeogOpEvalGeogGeogDouble(
                self.ptr,
                geog0.as_ptr(),
                geog1.as_ptr(),
                distance
            ))?;
            Ok(S2GeogOpGetInt(self.ptr) != 0)
        }
    }
}

impl Drop for Op {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                S2GeogOpDestroy(self.ptr);
            }
        }
    }
}

// Safety: Op contains only a pointer to C++ data that is thread-safe
// when accessed through its const methods
unsafe impl Send for Op {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geography::GeographyFactory;

    #[test]
    fn test_op_properties() {
        let op = Op::new(OpType::Intersects);
        assert_eq!(op.name(), "intersects");
        assert_eq!(op.output_type(), OutputType::Bool);
    }

    #[test]
    fn test_op_intersects() {
        let mut factory = GeographyFactory::new();
        let geog1 = factory
            .from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
            .unwrap();
        let geog2 = factory.from_wkt("POINT (5 5)").unwrap();

        let mut op = Op::new(OpType::Intersects);
        assert_eq!(op.name(), "intersects");
        assert!(op.eval_binary_predicate(&geog1, &geog2).unwrap());
    }

    #[test]
    fn test_op_contains() {
        let mut factory = GeographyFactory::new();
        let polygon = factory
            .from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
            .unwrap();
        let point_inside = factory.from_wkt("POINT (5 5)").unwrap();
        let point_outside = factory.from_wkt("POINT (20 20)").unwrap();

        let mut op = Op::new(OpType::Contains);
        assert!(op.eval_binary_predicate(&polygon, &point_inside).unwrap());
        assert!(!op.eval_binary_predicate(&polygon, &point_outside).unwrap());
    }

    #[test]
    fn test_op_within() {
        let mut factory = GeographyFactory::new();
        let polygon = factory
            .from_wkt("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))")
            .unwrap();
        let point_inside = factory.from_wkt("POINT (5 5)").unwrap();

        let mut op = Op::new(OpType::Within);
        assert!(op.eval_binary_predicate(&point_inside, &polygon).unwrap());
        assert!(!op.eval_binary_predicate(&polygon, &point_inside).unwrap());
    }

    #[test]
    fn test_op_equals() {
        let mut factory = GeographyFactory::new();
        let point1 = factory.from_wkt("POINT (5 5)").unwrap();
        let point2 = factory.from_wkt("POINT (5 5)").unwrap();
        let point3 = factory.from_wkt("POINT (10 10)").unwrap();

        let mut op = Op::new(OpType::Equals);
        assert!(op.eval_binary_predicate(&point1, &point2).unwrap());
        assert!(!op.eval_binary_predicate(&point1, &point3).unwrap());
    }

    #[test]
    fn test_op_dwithin() {
        let mut factory = GeographyFactory::new();
        let geog1 = factory.from_wkt("POINT (0 0)").unwrap();
        let geog2 = factory.from_wkt("POINT (0 1)").unwrap();

        let mut op = Op::new(OpType::DWithin);
        assert_eq!(op.name(), "distance_within");
        assert!(!op
            .eval_binary_distance_predicate(&geog1, &geog2, 100000.0)
            .unwrap());
        assert!(op
            .eval_binary_distance_predicate(&geog1, &geog2, 200000.0)
            .unwrap());
    }
}
