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

use datafusion_expr::Operator;
use datafusion_physical_expr::{expressions::BinaryExpr, PhysicalExpr, ScalarFunctionExpr};

/// Represents a parsed distance predicate with its constituent parts.
///
/// Distance predicates are spatial operations that determine whether two geometries
/// are within a specified distance of each other. This struct holds the parsed
/// components of such predicates for further processing.
///
/// ## Supported Distance Predicate Forms
///
/// This struct can represent the parsed components from any of these distance predicate forms:
///
/// 1. **Direct distance function**:
///    - `st_dwithin(geom1, geom2, distance)` - Returns true if geometries are within the distance
///
/// 2. **Distance comparison (left-to-right)**:
///    - `st_distance(geom1, geom2) <= distance` - Distance is less than or equal to threshold
///    - `st_distance(geom1, geom2) < distance` - Distance is strictly less than threshold
///
/// 3. **Distance comparison (right-to-left)**:
///    - `distance >= st_distance(geom1, geom2)` - Threshold is greater than or equal to distance
///    - `distance > st_distance(geom1, geom2)` - Threshold is strictly greater than distance
///
/// All forms are logically equivalent but may appear differently in SQL queries. The parser
/// normalizes them into this common structure for uniform processing.
pub struct ParsedDistancePredicate {
    /// The first geometry argument in the distance predicate
    pub arg0: Arc<dyn PhysicalExpr>,
    /// The second geometry argument in the distance predicate
    pub arg1: Arc<dyn PhysicalExpr>,
    /// The distance threshold argument (as a physical expression)
    pub arg_distance: Arc<dyn PhysicalExpr>,
}

/// Parses a physical expression to extract distance predicate components.
///
/// This function recognizes and parses distance predicates in spatial queries.
/// See [`ParsedDistancePredicate`] documentation for details on the supported
/// distance predicate forms.
///
/// # Arguments
///
/// * `expr` - A physical expression that potentially represents a distance predicate
///
/// # Returns
///
/// * `Some(ParsedDistancePredicate)` - If the expression is a recognized distance predicate,
///   returns the parsed components (two geometry arguments and the distance threshold)
/// * `None` - If the expression is not a distance predicate or cannot be parsed
///
/// # Examples
///
/// The function can parse expressions like:
/// - `st_dwithin(geometry_column, POINT(0 0), 100.0)`
/// - `st_distance(geom_a, geom_b) <= 50.0`
/// - `25.0 >= st_distance(geom_x, geom_y)`
pub fn parse_distance_predicate(expr: &Arc<dyn PhysicalExpr>) -> Option<ParsedDistancePredicate> {
    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = binary_expr.left();
        let right = binary_expr.right();
        let (st_distance_expr, distance_bound_expr) = match *binary_expr.op() {
            Operator::Lt | Operator::LtEq => (left, right),
            Operator::Gt | Operator::GtEq => (right, left),
            _ => return None,
        };

        if let Some(st_distance_expr) = st_distance_expr
            .as_any()
            .downcast_ref::<ScalarFunctionExpr>()
        {
            if st_distance_expr.fun().name() != "st_distance" {
                return None;
            }

            let args = st_distance_expr.args();
            assert!(args.len() >= 2);
            Some(ParsedDistancePredicate {
                arg0: Arc::clone(&args[0]),
                arg1: Arc::clone(&args[1]),
                arg_distance: Arc::clone(distance_bound_expr),
            })
        } else {
            None
        }
    } else if let Some(st_dwithin_expr) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        if st_dwithin_expr.fun().name() != "st_dwithin" {
            return None;
        }

        let args = st_dwithin_expr.args();
        assert!(args.len() >= 3);
        Some(ParsedDistancePredicate {
            arg0: Arc::clone(&args[0]),
            arg1: Arc::clone(&args[1]),
            arg_distance: Arc::clone(&args[2]),
        })
    } else {
        None
    }
}
