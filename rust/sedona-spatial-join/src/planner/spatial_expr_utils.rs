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

use std::collections::HashSet;
use std::sync::Arc;

use crate::spatial_predicate::{
    DistancePredicate, KNNPredicate, RelationPredicate, SpatialPredicate, SpatialRelationType,
};
use arrow_schema::Schema;
use datafusion_common::ScalarValue;
use datafusion_common::{
    tree_node::{Transformed, TreeNode},
    Column as LogicalColumn, JoinSide,
};
use datafusion_common::{DFSchema, HashMap, Result};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{Expr, Operator};
use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_physical_plan::joins::utils::ColumnIndex;
use datafusion_physical_plan::joins::utils::JoinFilter;
use sedona_common::sedona_internal_err;
use sedona_expr::utils::{parse_distance_predicate, ParsedDistancePredicate};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

/// Collect the names of spatial predicates appeared in expr. We assume that the given
/// `expr` evaluates to a boolean value and originates from a filter logical node.
pub(crate) fn collect_spatial_predicate_names(expr: &Expr) -> HashSet<String> {
    fn collect(expr: &Expr, acc: &mut HashSet<String>) {
        match expr {
            Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
                left, right, op, ..
            }) => match op {
                Operator::And => {
                    collect(left, acc);
                    collect(right, acc);
                }
                Operator::Lt | Operator::LtEq => {
                    if is_distance_expr(left) {
                        acc.insert("st_dwithin".to_string());
                    }
                }
                Operator::Gt | Operator::GtEq => {
                    if is_distance_expr(right) {
                        acc.insert("st_dwithin".to_string());
                    }
                }
                _ => (),
            },
            Expr::ScalarFunction(datafusion_expr::expr::ScalarFunction { func, .. }) => {
                let func_name = func.name().to_lowercase();
                if matches!(
                    func_name.as_str(),
                    "st_intersects"
                        | "st_contains"
                        | "st_within"
                        | "st_covers"
                        | "st_covered_by"
                        | "st_coveredby"
                        | "st_touches"
                        | "st_crosses"
                        | "st_overlaps"
                        | "st_equals"
                        | "st_dwithin"
                        | "st_knn"
                ) {
                    acc.insert(func_name);
                }
            }
            _ => (),
        }
    }

    fn is_distance_expr(expr: &Expr) -> bool {
        let Expr::ScalarFunction(datafusion_expr::expr::ScalarFunction { func, .. }) = expr else {
            return false;
        };
        func.name().to_lowercase() == "st_distance"
    }

    let mut acc = HashSet::new();
    collect(expr, &mut acc);
    acc
}

/// Transform the join filter to a spatial predicate and a remainder.
///
///   * The spatial predicate is a spatial predicate that is extracted from the join filter.
///   * The remainder is everything other than the spatial predicate.
///
/// The remainder may reference fewer columns than the original join filter. If that's the case,
/// the columns that are not referenced by the remainder will be pruned.
pub(crate) fn transform_join_filter(
    join_filter: &JoinFilter,
) -> Option<(SpatialPredicate, Option<JoinFilter>)> {
    let (spatial_predicate, remainder) =
        extract_spatial_predicate(join_filter.expression(), join_filter.column_indices())?;

    let remainder = remainder
        .as_ref()
        .map(|remainder| replace_join_filter_expr(remainder, join_filter));

    Some((spatial_predicate, remainder))
}

/// Extract the spatial predicate from the join filter. The extracted spatial predicate and the remaining filter
/// are returned. ST_KNN predicates are prioritized since they cannot be used as filters.
fn extract_spatial_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    column_indices: &[ColumnIndex],
) -> Option<(SpatialPredicate, Option<Arc<dyn PhysicalExpr>>)> {
    // First, scan the entire expression tree for ST_KNN predicates
    // ST_KNN must be the join condition since it cannot be a filter
    if let Some((knn_predicate, remainder)) =
        extract_knn_predicate_prioritized(expr, column_indices)
    {
        return Some((
            SpatialPredicate::KNearestNeighbors(knn_predicate),
            remainder,
        ));
    }

    // No ST_KNN found, proceed with normal extraction
    if let Some(scalar_fn) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        if let Some(relation_predicate) = match_relation_predicate(scalar_fn, column_indices) {
            return Some((SpatialPredicate::Relation(relation_predicate), None));
        }
    }

    if let Some(distance_predicate) = match_distance_predicate(expr, column_indices) {
        return Some((SpatialPredicate::Distance(distance_predicate), None));
    }

    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if !matches!(binary_expr.op(), Operator::And) {
            return None;
        }

        let left = binary_expr.left();
        let right = binary_expr.right();

        // Try to extract the spatial predicate from the left side
        if let Some((spatial_predicate, remainder)) =
            extract_spatial_predicate(left, column_indices)
        {
            let combined_remainder = match remainder {
                Some(remainder) => {
                    Arc::new(BinaryExpr::new(remainder, Operator::And, right.clone()))
                }
                None => right.clone(),
            };
            return Some((spatial_predicate, Some(combined_remainder)));
        }

        // Left side is not a spatial predicate, try to extract the spatial predicate from the right side
        if let Some((spatial_predicate, remainder)) =
            extract_spatial_predicate(right, column_indices)
        {
            let combined_remainder = match remainder {
                Some(remainder) => {
                    Arc::new(BinaryExpr::new(left.clone(), Operator::And, remainder))
                }
                None => left.clone(),
            };
            return Some((spatial_predicate, Some(combined_remainder)));
        }
    }

    None
}

/// Extract ST_KNN predicate from anywhere in the expression tree, collecting all other predicates as remainder
fn extract_knn_predicate_prioritized(
    expr: &Arc<dyn PhysicalExpr>,
    column_indices: &[ColumnIndex],
) -> Option<(KNNPredicate, Option<Arc<dyn PhysicalExpr>>)> {
    // Check if this expression itself is ST_KNN
    if let Some(scalar_fn) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        if let Some(knn_predicate) = match_knn_predicate(scalar_fn, column_indices) {
            return Some((knn_predicate, None));
        }
    }

    // If this is an AND expression, check both sides for ST_KNN
    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if matches!(binary_expr.op(), Operator::And) {
            let left = binary_expr.left();
            let right = binary_expr.right();

            // Check if left side contains ST_KNN
            if let Some((knn_predicate, left_remainder)) =
                extract_knn_predicate_prioritized(left, column_indices)
            {
                // ST_KNN found in left side, combine any left remainder with right side
                let combined_remainder = match left_remainder {
                    Some(remainder) => Some(Arc::new(BinaryExpr::new(
                        remainder,
                        Operator::And,
                        right.clone(),
                    )) as Arc<dyn PhysicalExpr>),
                    None => Some(right.clone()),
                };
                return Some((knn_predicate, combined_remainder));
            }

            // Check if right side contains ST_KNN
            if let Some((knn_predicate, right_remainder)) =
                extract_knn_predicate_prioritized(right, column_indices)
            {
                // ST_KNN found in right side, combine left side with any right remainder
                let combined_remainder = match right_remainder {
                    Some(remainder) => Some(Arc::new(BinaryExpr::new(
                        left.clone(),
                        Operator::And,
                        remainder,
                    )) as Arc<dyn PhysicalExpr>),
                    None => Some(left.clone()),
                };
                return Some((knn_predicate, combined_remainder));
            }
        }
    }

    None
}

/// Match the scalar function expression to a spatial relation predicate such as ST_Intersects(lhs.geom, rhs.geom).
/// The input arguments of the ST_ function should reference columns from different sides.
fn match_relation_predicate(
    scalar_fn: &ScalarFunctionExpr,
    column_indices: &[ColumnIndex],
) -> Option<RelationPredicate> {
    if let Some(relation_type) = SpatialRelationType::from_name(scalar_fn.fun().name()) {
        // Try to find the expressions that evaluates to the arguments of the spatial function
        let args = scalar_fn.args();
        assert!(args.len() >= 2);
        let arg0 = &args[0];
        let arg1 = &args[1];

        // Try to find the expressions that evaluates to the arguments of the spatial function
        let arg0_refs = collect_column_references(arg0, column_indices);
        let arg1_refs = collect_column_references(arg1, column_indices);

        let (arg0_side, arg1_side) = resolve_column_reference_sides(&arg0_refs, &arg1_refs)?;
        let arg0_reprojected =
            reproject_column_references_for_side(arg0, column_indices, arg0_side);
        let arg1_reprojected =
            reproject_column_references_for_side(arg1, column_indices, arg1_side);

        return match (arg0_side, arg1_side) {
            (JoinSide::Left, JoinSide::Right) => Some(RelationPredicate::new(
                arg0_reprojected,
                arg1_reprojected,
                relation_type,
            )),
            (JoinSide::Right, JoinSide::Left) => {
                // The spatial predicate needs to be inverted
                Some(RelationPredicate::new(
                    arg1_reprojected,
                    arg0_reprojected,
                    relation_type.invert(),
                ))
            }
            _ => None,
        };
    }
    None
}

/// Match the scalar function expression to a distance predicate such as ST_DWithin(geom1, geom2, distance)
/// or ST_Distance(geom1, geom2) <= distance.
/// The geometry input arguments of the ST_ function should reference columns from different sides.
/// The distance input argument should not reference columns from both sides simultaneously.
fn match_distance_predicate(
    expr: &Arc<dyn PhysicalExpr>,
    column_indices: &[ColumnIndex],
) -> Option<DistancePredicate> {
    let ParsedDistancePredicate {
        arg0,
        arg1,
        arg_distance,
    } = parse_distance_predicate(expr)?;

    // Try to find the expressions that evaluates to the arguments of the spatial function
    let arg0_refs = collect_column_references(&arg0, column_indices);
    let arg1_refs = collect_column_references(&arg1, column_indices);
    let arg_dist_refs = collect_column_references(&arg_distance, column_indices);

    let arg_dist_side = side_of_column_references(&arg_dist_refs)?;
    let (arg0_side, arg1_side) = resolve_column_reference_sides(&arg0_refs, &arg1_refs)?;

    let arg0_reprojected = reproject_column_references_for_side(&arg0, column_indices, arg0_side);
    let arg1_reprojected = reproject_column_references_for_side(&arg1, column_indices, arg1_side);
    let arg_dist_reprojected =
        reproject_column_references_for_side(&arg_distance, column_indices, arg_dist_side);

    match (arg0_side, arg1_side) {
        (JoinSide::Left, JoinSide::Right) => Some(DistancePredicate::new(
            arg0_reprojected,
            arg1_reprojected,
            arg_dist_reprojected,
            arg_dist_side,
        )),
        (JoinSide::Right, JoinSide::Left) => Some(DistancePredicate::new(
            arg1_reprojected,
            arg0_reprojected,
            arg_dist_reprojected,
            arg_dist_side,
        )),
        _ => None,
    }
}

/// Match the scalar function expression to a KNN predicate such as ST_KNN(geom1, geom2, k, use_spheroid).
/// The geometry input arguments of the ST_KNN function should reference columns from different sides.
/// The k and use_spheroid arguments must be literal values.
fn match_knn_predicate(
    scalar_fn: &ScalarFunctionExpr,
    column_indices: &[ColumnIndex],
) -> Option<KNNPredicate> {
    // Check if this is an ST_KNN function
    if scalar_fn.fun().name() != "st_knn" {
        return None;
    }

    let args = scalar_fn.args();

    if args.len() < 2 {
        return None;
    }

    let queries_geom = &args[0];
    let objects_geom = &args[1];

    let (k, use_spheroid) = match args.len() {
        2 => {
            // Apply default k (1) and use_spheroid (false)
            (1, false)
        }
        3 => {
            // Extract literal values for k and apply default use_spheroid (false)
            let k_expr = &args[2];
            let k = extract_literal_u32(k_expr)?;
            (k, false)
        }
        4 => {
            // Extract literal values for k and use_spheroid
            let k_expr = &args[2];
            let k = extract_literal_u32(k_expr)?;

            let use_spheroid_expr = &args[3];
            let use_spheroid = extract_literal_bool(use_spheroid_expr)?;
            (k, use_spheroid)
        }
        _ => return None,
    };

    // Collect column references for geometry arguments
    let queries_refs = collect_column_references(queries_geom, column_indices);
    let objects_refs = collect_column_references(objects_geom, column_indices);

    let (queries_side, objects_side) =
        resolve_column_reference_sides(&queries_refs, &objects_refs)?;

    // Reproject geometry arguments to their respective sides
    let queries_reprojected =
        reproject_column_references_for_side(queries_geom, column_indices, queries_side);
    let objects_reprojected =
        reproject_column_references_for_side(objects_geom, column_indices, objects_side);

    match (queries_side, objects_side) {
        (JoinSide::Left, JoinSide::Right) => {
            Some(KNNPredicate::new(
                queries_reprojected,
                objects_reprojected,
                k,
                use_spheroid,
                JoinSide::Left, // Probe side is left plan
            ))
        }
        (JoinSide::Right, JoinSide::Left) => {
            // Preserve the original query semantics: first argument is always probe, second is always build
            Some(KNNPredicate::new(
                queries_reprojected, // First argument (probe side)
                objects_reprojected, // Second argument (build side)
                k,
                use_spheroid,
                JoinSide::Right, // Probe side is right plan (since queries_side=Right)
            ))
        }
        _ => None,
    }
}

fn collect_column_references(
    expr: &Arc<dyn PhysicalExpr>,
    column_indices: &[ColumnIndex],
) -> Vec<ColumnIndex> {
    let mut collected_column_indices = Vec::with_capacity(column_indices.len());

    expr.apply(|node| {
        if let Some(column) = node.as_any().downcast_ref::<Column>() {
            let intermediate_index = column.index();
            let column_info = &column_indices[intermediate_index];
            collected_column_indices.push(column_info.clone());
        }

        Ok(datafusion_common::tree_node::TreeNodeRecursion::Continue)
    })
    .expect("Failed to collect column references");

    collected_column_indices
}

fn resolve_column_reference_sides(
    left_refs: &[ColumnIndex],
    right_refs: &[ColumnIndex],
) -> Option<(JoinSide, JoinSide)> {
    let left_side = side_of_column_references(left_refs)?;
    let right_side = side_of_column_references(right_refs)?;

    if left_side != right_side {
        Some((left_side, right_side))
    } else {
        None
    }
}

fn side_of_column_references(column_indices: &[ColumnIndex]) -> Option<JoinSide> {
    match column_indices.first() {
        Some(first) => {
            let first_side = first.side;
            if column_indices
                .iter()
                .all(|col_idx| col_idx.side == first_side)
            {
                Some(first_side)
            } else {
                // Referencing both sides simultaneously
                None
            }
        }
        None => Some(JoinSide::None),
    }
}

fn reproject_column_references(
    expr: &Arc<dyn PhysicalExpr>,
    index_map: &HashMap<usize, usize>,
) -> Arc<dyn PhysicalExpr> {
    expr.clone()
        .transform_down(|node| {
            // Check if this is a Column expression
            if let Some(column) = node.as_any().downcast_ref::<Column>() {
                let old_index = column.index();
                if let Some(&new_index) = index_map.get(&old_index) {
                    // Create a new Column with the mapped index
                    let new_column = Arc::new(Column::new(column.name(), new_index));
                    return Ok(Transformed::yes(new_column));
                }
            }

            // For all other expressions, continue with the default traversal
            Ok(Transformed::no(node))
        })
        .unwrap_or_else(|_| Transformed::no(expr.clone()))
        .data
}

fn reproject_column_references_for_side(
    expr: &Arc<dyn PhysicalExpr>,
    column_indices: &[ColumnIndex],
    side: JoinSide,
) -> Arc<dyn PhysicalExpr> {
    if side == JoinSide::None {
        return expr.clone();
    }

    let index_mapping: HashMap<usize, usize> = column_indices
        .iter()
        .enumerate()
        .filter_map(|(i, col_idx)| (col_idx.side == side).then_some((i, col_idx.index)))
        .collect();

    reproject_column_references(expr, &index_mapping)
}

/// Extract a literal u32 value from an expression.
/// Returns None if the expression is not a literal integer or if it's out of u32 range.
fn extract_literal_u32(expr: &Arc<dyn PhysicalExpr>) -> Option<u32> {
    let literal = expr.as_any().downcast_ref::<Literal>()?;
    match literal.value() {
        ScalarValue::UInt32(Some(val)) => Some(*val),
        ScalarValue::Int32(Some(val)) if *val >= 0 => Some(*val as u32),
        ScalarValue::Int64(Some(val)) if *val >= 0 && *val <= u32::MAX as i64 => Some(*val as u32),
        ScalarValue::UInt64(Some(val)) if *val <= u32::MAX as u64 => Some(*val as u32),
        _ => None,
    }
}

/// Extract a literal boolean value from an expression.
/// Returns None if the expression is not a literal boolean.
fn extract_literal_bool(expr: &Arc<dyn PhysicalExpr>) -> Option<bool> {
    let literal = expr.as_any().downcast_ref::<Literal>()?;
    match literal.value() {
        ScalarValue::Boolean(Some(val)) => Some(*val),
        _ => None,
    }
}

/// Replace the join filter expression with a new expression. The replaced join filter expression
/// may reference fewer columns than the original join filter expression. If that's the case,
/// the columns that are not referenced by the replaced join filter expression will be pruned.
fn replace_join_filter_expr(expr: &Arc<dyn PhysicalExpr>, join_filter: &JoinFilter) -> JoinFilter {
    let column_indices = join_filter.column_indices();
    let column_refs = collect_column_references(expr, column_indices);

    // column_refs could be a subset of column_indices. If that's the case, we can prune column_indices
    // to only include the columns that are referenced by the remainder.
    let referenced_columns: Vec<_> = column_indices
        .iter()
        .enumerate()
        .filter(|(_, col_idx)| column_refs.contains(col_idx))
        .collect();

    let pruned_column_indices: Vec<_> = referenced_columns
        .iter()
        .map(|(_, col_idx)| (*col_idx).clone())
        .collect();

    let column_index_mapping: HashMap<_, _> = referenced_columns
        .iter()
        .enumerate()
        .map(|(new_idx, (old_idx, _))| (*old_idx, new_idx))
        .collect();

    let project: Vec<_> = referenced_columns
        .iter()
        .map(|(old_idx, _)| *old_idx)
        .collect();

    let pruned_schema = join_filter
        .schema()
        .project(&project)
        .expect("Failed to project schema");
    let remainder_reprojected = reproject_column_references(expr, &column_index_mapping);
    JoinFilter::new(
        remainder_reprojected,
        pruned_column_indices,
        Arc::new(pruned_schema),
    )
}

pub(crate) fn is_spatial_predicate_supported(
    spatial_predicate: &SpatialPredicate,
    left_schema: &Schema,
    right_schema: &Schema,
) -> Result<bool> {
    /// Only spatial predicates working with planar geometry are supported for optimization.
    /// Geography (spherical) types are explicitly excluded and will not trigger optimized spatial joins.
    fn is_geometry_type_supported(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Result<bool> {
        let left_return_field = expr.return_field(schema)?;
        let sedona_type = SedonaType::from_storage_field(&left_return_field)?;
        let matcher = ArgMatcher::is_geometry();
        Ok(matcher.match_type(&sedona_type))
    }

    match spatial_predicate {
        SpatialPredicate::Relation(RelationPredicate { left, right, .. })
        | SpatialPredicate::Distance(DistancePredicate { left, right, .. }) => {
            Ok(is_geometry_type_supported(left, left_schema)?
                && is_geometry_type_supported(right, right_schema)?)
        }
        SpatialPredicate::KNearestNeighbors(KNNPredicate {
            left,
            right,
            probe_side,
            ..
        }) => {
            let (left, right) = match probe_side {
                JoinSide::Left => (left, right),
                JoinSide::Right => (right, left),
                _ => {
                    return sedona_internal_err!(
                        "Invalid probe side in KNN predicate: {:?}",
                        probe_side
                    )
                }
            };
            Ok(is_geometry_type_supported(left, left_schema)?
                && is_geometry_type_supported(right, right_schema)?)
        }
    }
}

/// Which side of the join is the query (probe) side for KNN.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KNNJoinQuerySide {
    Left,
    Right,
}

/// Find the query side of a KNN join by examining the first argument of the `ST_KNN` call
/// in the join filter expression.
///
/// The first argument of `ST_KNN(query_geom, object_geom, k, use_spheroid)` identifies the
/// query side. We collect all column references from that argument and check which child's
/// schema they belong to.
pub(crate) fn find_knn_query_side(
    filter_expr: &Expr,
    left_schema: &DFSchema,
    right_schema: &DFSchema,
) -> Option<KNNJoinQuerySide> {
    let knn_call = find_st_knn_call(filter_expr)?;

    // The first argument is the query geometry expression
    let query_geom_expr = knn_call.args.first()?;

    // Collect all column references from the query geometry argument
    let col_refs: HashSet<&LogicalColumn> = query_geom_expr.column_refs();

    if col_refs.is_empty() {
        return None;
    }

    let all_in_left = col_refs.iter().all(|col| left_schema.has_column(col));
    let all_in_right = col_refs.iter().all(|col| right_schema.has_column(col));

    match (all_in_left, all_in_right) {
        (true, false) => Some(KNNJoinQuerySide::Left),
        (false, true) => Some(KNNJoinQuerySide::Right),
        _ => None, // Ambiguous or references both sides
    }
}

/// Recursively find the `ST_KNN` scalar function call in an expression tree.
///
/// Searches through `AND` binary expressions to find the `ST_KNN` function call.
pub(crate) fn find_st_knn_call(expr: &Expr) -> Option<&ScalarFunction> {
    match expr {
        Expr::ScalarFunction(func) if func.func.name().eq_ignore_ascii_case("st_knn") => Some(func),
        Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr { left, right, op })
            if *op == Operator::And =>
        {
            find_st_knn_call(left).or_else(|| find_st_knn_call(right))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial_predicate::{SpatialPredicate, SpatialRelationType};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::config::ConfigOptions;
    use datafusion_common::{JoinSide, ScalarValue};
    use datafusion_expr::Operator;
    use datafusion_expr::{col, lit, ColumnarValue, Expr, ScalarUDF, SimpleScalarUDF};
    use datafusion_physical_expr::expressions::{BinaryExpr, Column, IsNotNullExpr, Literal};
    use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
    use datafusion_physical_plan::joins::utils::ColumnIndex;
    use datafusion_physical_plan::joins::utils::JoinFilter;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_GEOMETRY};
    use std::sync::Arc;

    // Helper function to create a test schema
    fn create_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("left_id", DataType::Int32, false), // index 0
            WKB_GEOMETRY.to_storage_field("left_geom", false).unwrap(), // index 1
            WKB_GEOMETRY.to_storage_field("right_geom", false).unwrap(), // index 2
            Field::new("right_distance", DataType::Float64, false), // index 3
        ]))
    }

    // Helper function to create test column indices for join filter
    fn create_test_column_indices() -> Vec<ColumnIndex> {
        vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            }, // left_id
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            }, // left_geom
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            }, // right_geom
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            }, // right_distance
        ]
    }

    // Helper to create dummy spatial UDFs for testing
    fn create_dummy_st_intersects_udf() -> Arc<ScalarUDF> {
        Arc::new(ScalarUDF::from(SimpleScalarUDF::new(
            "st_intersects",
            vec![
                WKB_GEOMETRY.storage_type().clone(),
                WKB_GEOMETRY.storage_type().clone(),
            ],
            DataType::Boolean,
            datafusion_expr::Volatility::Immutable,
            Arc::new(|_| Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))))),
        )))
    }

    fn create_dummy_st_dwithin_udf() -> Arc<ScalarUDF> {
        Arc::new(ScalarUDF::from(SimpleScalarUDF::new(
            "st_dwithin",
            vec![
                WKB_GEOMETRY.storage_type().clone(),
                WKB_GEOMETRY.storage_type().clone(),
                DataType::Float64,
            ],
            DataType::Boolean,
            datafusion_expr::Volatility::Immutable,
            Arc::new(|_| Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))))),
        )))
    }

    fn create_dummy_st_distance_udf() -> Arc<ScalarUDF> {
        Arc::new(ScalarUDF::from(SimpleScalarUDF::new(
            "st_distance",
            vec![
                WKB_GEOMETRY.storage_type().clone(),
                WKB_GEOMETRY.storage_type().clone(),
            ],
            DataType::Float64,
            datafusion_expr::Volatility::Immutable,
            Arc::new(|_| Ok(ColumnarValue::Scalar(ScalarValue::Float64(Some(100.0))))),
        )))
    }

    fn create_dummy_st_within_udf() -> Arc<ScalarUDF> {
        Arc::new(ScalarUDF::from(SimpleScalarUDF::new(
            "st_within",
            vec![
                WKB_GEOMETRY.storage_type().clone(),
                WKB_GEOMETRY.storage_type().clone(),
            ],
            DataType::Boolean,
            datafusion_expr::Volatility::Immutable,
            Arc::new(|_| Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))))),
        )))
    }

    // Helper to create spatial function expressions using the dummy UDFs
    fn create_spatial_function_expr(
        udf: Arc<ScalarUDF>,
        args: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Arc<ScalarFunctionExpr> {
        let return_type = udf.return_type(&[]).unwrap();
        let field = Arc::new(arrow::datatypes::Field::new("result", return_type, false));
        Arc::new(ScalarFunctionExpr::new(
            udf.name(),
            Arc::clone(&udf),
            args,
            field,
            Arc::new(ConfigOptions::default()),
        ))
    }

    #[test]
    fn test_collect_column_references() {
        let column_indices = create_test_column_indices();

        // Test single column reference
        let col_expr = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let refs = collect_column_references(&col_expr, &column_indices);

        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].index, 1);
        assert_eq!(refs[0].side, JoinSide::Left);

        // Test binary expression with columns from both sides
        let left_col = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_col = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let binary_expr =
            Arc::new(BinaryExpr::new(left_col, Operator::Eq, right_col)) as Arc<dyn PhysicalExpr>;

        let refs = collect_column_references(&binary_expr, &column_indices);
        assert_eq!(refs.len(), 2);

        // Should have one reference from each side
        let left_refs: Vec<_> = refs.iter().filter(|r| r.side == JoinSide::Left).collect();
        let right_refs: Vec<_> = refs.iter().filter(|r| r.side == JoinSide::Right).collect();
        assert_eq!(left_refs.len(), 1);
        assert_eq!(right_refs.len(), 1);

        // Test literal expression with no column references
        let literal_expr =
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))) as Arc<dyn PhysicalExpr>;
        let refs = collect_column_references(&literal_expr, &column_indices);
        assert_eq!(refs.len(), 0);
    }

    #[test]
    fn test_side_of_column_references() {
        // Test all left side
        let left_refs = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
        ];
        assert_eq!(side_of_column_references(&left_refs), Some(JoinSide::Left));

        // Test all right side
        let right_refs = vec![ColumnIndex {
            index: 0,
            side: JoinSide::Right,
        }];
        assert_eq!(
            side_of_column_references(&right_refs),
            Some(JoinSide::Right)
        );

        // Test mixed sides (should return None)
        let mixed_refs = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 0,
                side: JoinSide::Right,
            },
        ];
        assert_eq!(side_of_column_references(&mixed_refs), None);

        // Test empty (should return JoinSide::None)
        let empty_refs = vec![];
        assert_eq!(side_of_column_references(&empty_refs), Some(JoinSide::None));
    }

    #[test]
    fn test_resolve_column_reference_sides() {
        let left_refs = vec![ColumnIndex {
            index: 0,
            side: JoinSide::Left,
        }];
        let right_refs = vec![ColumnIndex {
            index: 0,
            side: JoinSide::Right,
        }];
        let mixed_refs = vec![
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ];

        // Test valid case - different sides
        let result = resolve_column_reference_sides(&left_refs, &right_refs);
        assert_eq!(result, Some((JoinSide::Left, JoinSide::Right)));
        let result = resolve_column_reference_sides(&right_refs, &left_refs);
        assert_eq!(result, Some((JoinSide::Right, JoinSide::Left)));

        // Test invalid case - same side
        let result = resolve_column_reference_sides(&left_refs, &left_refs);
        assert_eq!(result, None);
        let result = resolve_column_reference_sides(&right_refs, &right_refs);
        assert_eq!(result, None);

        // Test invalid case - mixed sides
        let result = resolve_column_reference_sides(&left_refs, &mixed_refs);
        assert_eq!(result, None);
        let result = resolve_column_reference_sides(&mixed_refs, &mixed_refs);
        assert_eq!(result, None);
    }

    #[test]
    fn test_reproject_column_references() {
        // Create a column expression
        let col_expr = Arc::new(Column::new("test_col", 2)) as Arc<dyn PhysicalExpr>;

        // Create index mapping: old index 2 -> new index 0
        let mut index_map = HashMap::new();
        index_map.insert(2, 0);

        let reprojected = reproject_column_references(&col_expr, &index_map);

        // Check that the column index was updated
        let reprojected_col = reprojected.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(reprojected_col.index(), 0);
        assert_eq!(reprojected_col.name(), "test_col");

        // Test expression with no mapping (should remain unchanged)
        let col_expr_unmapped = Arc::new(Column::new("other_col", 5)) as Arc<dyn PhysicalExpr>;
        let reprojected_unmapped = reproject_column_references(&col_expr_unmapped, &index_map);
        let unmapped_col = reprojected_unmapped
            .as_any()
            .downcast_ref::<Column>()
            .unwrap();
        assert_eq!(unmapped_col.index(), 5); // Should remain unchanged
    }

    #[test]
    fn test_reproject_column_reference_complex() {
        let udf = create_dummy_st_intersects_udf();
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            create_spatial_function_expr(
                udf,
                vec![
                    Arc::new(Column::new("left_geom", 1)),
                    Arc::new(Column::new("right_geom", 2)),
                ],
            ),
            Operator::And,
            Arc::new(BinaryExpr::new(
                Arc::new(IsNotNullExpr::new(Arc::new(Column::new("left_id", 0)))),
                Operator::And,
                Arc::new(IsNotNullExpr::new(Arc::new(Column::new(
                    "right_distance",
                    3,
                )))),
            )),
        ));

        // Create index mapping
        let mut index_map = HashMap::new();
        index_map.insert(1, 10);
        index_map.insert(2, 11);
        index_map.insert(3, 12);

        // Reproject the expression
        let reprojected = reproject_column_references(&expr, &index_map);
        let reprojected_col = reprojected.as_any().downcast_ref::<BinaryExpr>().unwrap();

        // The reprojected expression should be: ST_Intersects(left_geom[10], right_geom[11]) AND (IS NOT NULL(left_id[0]) AND IS NOT NULL(right_distance[12]))
        assert_eq!(reprojected_col.op(), &Operator::And);

        // Left side should be ST_Intersects
        let left_side = reprojected_col.left();
        let st_intersects = left_side
            .as_any()
            .downcast_ref::<ScalarFunctionExpr>()
            .unwrap();
        assert_eq!(st_intersects.fun().name(), "st_intersects");

        let st_intersects_args = st_intersects.args();
        assert_eq!(st_intersects_args.len(), 2);

        // First arg should be left_geom with index 10
        let left_geom_col = st_intersects_args[0]
            .as_any()
            .downcast_ref::<Column>()
            .unwrap();
        assert_eq!(left_geom_col.name(), "left_geom");
        assert_eq!(left_geom_col.index(), 10);

        // Second arg should be right_geom with index 11
        let right_geom_col = st_intersects_args[1]
            .as_any()
            .downcast_ref::<Column>()
            .unwrap();
        assert_eq!(right_geom_col.name(), "right_geom");
        assert_eq!(right_geom_col.index(), 11);

        // Right side should be nested AND expression
        let right_side = reprojected_col.right();
        let nested_and = right_side.as_any().downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(nested_and.op(), &Operator::And);

        // Left part of nested AND should be IS NOT NULL(left_id[0])
        let left_not_null = nested_and.left();
        let left_is_not_null = left_not_null
            .as_any()
            .downcast_ref::<IsNotNullExpr>()
            .unwrap();
        let left_id_col = left_is_not_null
            .arg()
            .as_any()
            .downcast_ref::<Column>()
            .unwrap();
        assert_eq!(left_id_col.name(), "left_id");
        assert_eq!(left_id_col.index(), 0); // Should remain 0 (not remapped)

        // Right part of nested AND should be IS NOT NULL(right_distance[12])
        let right_not_null = nested_and.right();
        let right_is_not_null = right_not_null
            .as_any()
            .downcast_ref::<IsNotNullExpr>()
            .unwrap();
        let right_distance_col = right_is_not_null
            .arg()
            .as_any()
            .downcast_ref::<Column>()
            .unwrap();
        assert_eq!(right_distance_col.name(), "right_distance");
        assert_eq!(right_distance_col.index(), 12); // Should be remapped from 3 to 12
    }

    #[test]
    fn test_reproject_column_references_for_left_side() {
        let column_indices = create_test_column_indices();

        // Create expression referencing left side column (intermediate index 1)
        let left_col_expr = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;

        // Reproject for left side
        let reprojected =
            reproject_column_references_for_side(&left_col_expr, &column_indices, JoinSide::Left);

        let reprojected_col = reprojected.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(reprojected_col.index(), 1); // Should map to original left side index
    }

    #[test]
    fn test_reproject_column_references_for_right_side() {
        let column_indices = create_test_column_indices();

        // Create expression referencing right side column (intermediate index 2)
        let right_col_expr = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;

        // Reproject for right side
        let reprojected =
            reproject_column_references_for_side(&right_col_expr, &column_indices, JoinSide::Right);

        let reprojected_col = reprojected.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(reprojected_col.index(), 0); // Should map to original right side index
    }

    #[test]
    fn test_reproject_column_references_for_none_side() {
        let column_indices = create_test_column_indices();

        let expr = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;

        // Test JoinSide::None (should return original expression)
        let none_reprojected =
            reproject_column_references_for_side(&expr, &column_indices, JoinSide::None);

        // Should be the same object
        assert!(Arc::ptr_eq(&expr, &none_reprojected));
    }

    #[test]
    fn test_match_relation_predicate_st_intersects() {
        let column_indices = create_test_column_indices();

        // Create ST_Intersects(left_geom, right_geom)
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;

        let st_intersects_udf = create_dummy_st_intersects_udf();
        let args = vec![left_geom, right_geom];
        let st_intersects = create_spatial_function_expr(st_intersects_udf, args);

        let predicate = match_relation_predicate(&st_intersects, &column_indices);
        assert!(predicate.is_some());

        let pred = predicate.unwrap();
        // Verify the relation type is Intersects
        assert_eq!(pred.relation_type, SpatialRelationType::Intersects);

        // Verify left argument is reprojected to left side (should reference index 1 on left side)
        let left_arg_col = pred.left.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_arg_col.index(), 1);
        assert_eq!(left_arg_col.name(), "left_geom");

        // Verify right argument is reprojected to right side (should reference index 0 on right side)
        let right_arg_col = pred.right.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(right_arg_col.index(), 0);
        assert_eq!(right_arg_col.name(), "right_geom");
    }

    #[test]
    fn test_match_relation_predicate_st_within_inverted() {
        let column_indices = create_test_column_indices();

        // Create ST_Within(right_geom, left_geom) - this should be inverted to left, right order
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;

        let st_within_udf = create_dummy_st_within_udf();
        let args = vec![right_geom, left_geom]; // Note: right, left order
        let st_within = create_spatial_function_expr(st_within_udf, args);

        let predicate = match_relation_predicate(&st_within, &column_indices);
        assert!(predicate.is_some());

        let pred = predicate.unwrap();
        // Verify the relation type is Contains (inverted from Within)
        assert_eq!(pred.relation_type, SpatialRelationType::Contains);

        // After inversion, left_arg should be the original left_geom
        let left_arg_col = pred.left.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_arg_col.index(), 1);
        assert_eq!(left_arg_col.name(), "left_geom");

        // After inversion, right_arg should be the original right_geom
        let right_arg_col = pred.right.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(right_arg_col.index(), 0);
        assert_eq!(right_arg_col.name(), "right_geom");
    }

    #[test]
    fn test_match_relation_predicate_same_side_fails() {
        let column_indices = create_test_column_indices();

        // Create ST_Intersects(left_geom, left_id) - both from same side, should fail
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;

        let st_intersects_udf = create_dummy_st_intersects_udf();
        let args = vec![left_geom, left_id];
        let st_intersects = create_spatial_function_expr(st_intersects_udf, args);

        let predicate = match_relation_predicate(&st_intersects, &column_indices);
        assert!(predicate.is_none()); // Should fail - both args from same side
    }

    #[test]
    fn test_match_distance_predicate_st_dwithin() {
        let column_indices = create_test_column_indices();

        // Create ST_DWithin(left_geom, right_geom, 1000.0)
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let distance =
            Arc::new(Literal::new(ScalarValue::Float64(Some(1000.0)))) as Arc<dyn PhysicalExpr>;

        let st_dwithin_udf = create_dummy_st_dwithin_udf();
        let args = vec![left_geom, right_geom, distance];
        let st_dwithin = create_spatial_function_expr(st_dwithin_udf, args);
        let st_dwithin_expr = st_dwithin as Arc<dyn PhysicalExpr>;

        let predicate = match_distance_predicate(&st_dwithin_expr, &column_indices);
        assert!(predicate.is_some());

        let pred = predicate.unwrap();
        // Verify left argument is reprojected to left side
        let left_arg_col = pred.left.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_arg_col.index(), 1);
        assert_eq!(left_arg_col.name(), "left_geom");

        // Verify right argument is reprojected to right side
        let right_arg_col = pred.right.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(right_arg_col.index(), 0);
        assert_eq!(right_arg_col.name(), "right_geom");

        // Verify distance is a literal with JoinSide::None
        assert_eq!(pred.distance_side, datafusion_common::JoinSide::None);
        let distance_literal = pred.distance.as_any().downcast_ref::<Literal>().unwrap();
        match distance_literal.value() {
            ScalarValue::Float64(Some(val)) => assert_eq!(val, &1000.0),
            _ => panic!("Expected Float64 literal"),
        }
    }

    #[test]
    fn test_match_distance_predicate_st_distance_comparison() {
        let column_indices = create_test_column_indices();

        // Create ST_Distance(left_geom, right_geom) <= 1000.0
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let distance =
            Arc::new(Literal::new(ScalarValue::Float64(Some(1000.0)))) as Arc<dyn PhysicalExpr>;

        let st_distance_udf = create_dummy_st_distance_udf();
        let st_distance_args = vec![left_geom, right_geom];
        let st_distance = create_spatial_function_expr(st_distance_udf, st_distance_args);
        let st_distance_expr = st_distance as Arc<dyn PhysicalExpr>;

        // Create <= comparison
        let comparison = Arc::new(BinaryExpr::new(st_distance_expr, Operator::LtEq, distance))
            as Arc<dyn PhysicalExpr>;

        let predicate = match_distance_predicate(&comparison, &column_indices);
        assert!(predicate.is_some());

        let pred = predicate.unwrap();
        // Verify left and right arguments are correctly reprojected
        let left_arg_col = pred.left.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_arg_col.index(), 1);
        assert_eq!(left_arg_col.name(), "left_geom");

        let right_arg_col = pred.right.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(right_arg_col.index(), 0);
        assert_eq!(right_arg_col.name(), "right_geom");

        // Verify distance is a literal with JoinSide::None
        assert_eq!(pred.distance_side, datafusion_common::JoinSide::None);
    }

    #[test]
    fn test_match_distance_predicate_with_column_distance() {
        let column_indices = create_test_column_indices();

        // Create ST_DWithin(left_geom, right_geom, right_distance) - distance from right side
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let right_distance = Arc::new(Column::new("right_distance", 3)) as Arc<dyn PhysicalExpr>;

        let st_dwithin_udf = create_dummy_st_dwithin_udf();
        let args = vec![left_geom, right_geom, right_distance];
        let st_dwithin = create_spatial_function_expr(st_dwithin_udf, args);
        let st_dwithin_expr = st_dwithin as Arc<dyn PhysicalExpr>;

        let predicate = match_distance_predicate(&st_dwithin_expr, &column_indices);
        assert!(predicate.is_some());

        let pred = predicate.unwrap();
        // Verify left and right geometry arguments
        let left_arg_col = pred.left.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_arg_col.index(), 1);
        assert_eq!(left_arg_col.name(), "left_geom");

        let right_arg_col = pred.right.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(right_arg_col.index(), 0);
        assert_eq!(right_arg_col.name(), "right_geom");

        // Verify distance comes from right side
        assert_eq!(pred.distance_side, datafusion_common::JoinSide::Right);
        let distance_col = pred.distance.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(distance_col.index(), 1); // Should be reprojected to right side index 1
        assert_eq!(distance_col.name(), "right_distance");
    }

    #[test]
    fn test_extract_spatial_predicate_simple() {
        let column_indices = create_test_column_indices();

        // Test simple ST_Intersects
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;

        let st_intersects_udf = create_dummy_st_intersects_udf();
        let args = vec![left_geom, right_geom];
        let st_intersects = create_spatial_function_expr(st_intersects_udf, args);
        let st_intersects_expr = st_intersects as Arc<dyn PhysicalExpr>;

        let result = extract_spatial_predicate(&st_intersects_expr, &column_indices);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        let SpatialPredicate::Relation(rel_pred) = spatial_pred else {
            panic!("Expected SpatialPredicate::Relation");
        };
        assert_eq!(
            rel_pred
                .left
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            1
        );
        assert_eq!(
            rel_pred
                .right
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            0
        );
        assert_eq!(rel_pred.relation_type, SpatialRelationType::Intersects);
        assert!(remainder.is_none()); // No remainder for simple predicate
    }

    #[test]
    fn test_extract_spatial_predicate_with_and() {
        let column_indices = create_test_column_indices();

        // Create ST_Intersects(left_geom, right_geom) AND left_id = 1
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let literal_one =
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))) as Arc<dyn PhysicalExpr>;

        let st_intersects_udf = create_dummy_st_intersects_udf();
        let st_intersects_args = vec![left_geom, right_geom];
        let st_intersects = create_spatial_function_expr(st_intersects_udf, st_intersects_args);
        let st_intersects_expr = st_intersects as Arc<dyn PhysicalExpr>;

        let id_filter =
            Arc::new(BinaryExpr::new(left_id, Operator::Eq, literal_one)) as Arc<dyn PhysicalExpr>;

        let and_expr = Arc::new(BinaryExpr::new(
            st_intersects_expr,
            Operator::And,
            id_filter,
        )) as Arc<dyn PhysicalExpr>;

        let result = extract_spatial_predicate(&and_expr, &column_indices);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        let SpatialPredicate::Relation(rel_pred) = spatial_pred else {
            panic!("Expected SpatialPredicate::Relation");
        };
        assert_eq!(
            rel_pred
                .left
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            1
        );
        assert_eq!(
            rel_pred
                .right
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            0
        );
        assert_eq!(rel_pred.relation_type, SpatialRelationType::Intersects);
        assert!(remainder.is_some()); // Should have remainder (the id filter)
        let remainder = remainder.unwrap();

        // Remainder should be: left_id = 1
        let remainder_binary = remainder.as_any().downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(remainder_binary.op(), &Operator::Eq);

        // Left side should be left_id column
        let left_side = remainder_binary.left();
        let left_col = left_side.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_col.name(), "left_id");
        assert_eq!(left_col.index(), 0);

        // Right side should be literal 1
        let right_side = remainder_binary.right();
        let literal = right_side.as_any().downcast_ref::<Literal>().unwrap();
        match literal.value() {
            ScalarValue::Int32(Some(val)) => assert_eq!(val, &1),
            _ => panic!("Expected Int32(1) literal"),
        }
    }

    #[test]
    fn test_extract_spatial_predicate_distance_in_and() {
        let column_indices = create_test_column_indices();

        // Create left_id = 1 AND ST_DWithin(left_geom, right_geom, 1000.0)
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let literal_one =
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))) as Arc<dyn PhysicalExpr>;
        let distance =
            Arc::new(Literal::new(ScalarValue::Float64(Some(1000.0)))) as Arc<dyn PhysicalExpr>;

        let id_filter =
            Arc::new(BinaryExpr::new(left_id, Operator::Eq, literal_one)) as Arc<dyn PhysicalExpr>;

        let st_dwithin_udf = create_dummy_st_dwithin_udf();
        let st_dwithin_args = vec![left_geom, right_geom, distance];
        let st_dwithin = create_spatial_function_expr(st_dwithin_udf, st_dwithin_args);
        let st_dwithin_expr = st_dwithin as Arc<dyn PhysicalExpr>;

        let and_expr = Arc::new(BinaryExpr::new(id_filter, Operator::And, st_dwithin_expr))
            as Arc<dyn PhysicalExpr>;

        let result = extract_spatial_predicate(&and_expr, &column_indices);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        let SpatialPredicate::Distance(dist_pred) = spatial_pred else {
            panic!("Expected SpatialPredicate::Distance");
        };
        assert_eq!(
            dist_pred
                .left
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            1
        );
        assert_eq!(
            dist_pred
                .right
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            0
        );
        assert_eq!(
            dist_pred
                .distance
                .as_any()
                .downcast_ref::<Literal>()
                .unwrap()
                .value(),
            &ScalarValue::Float64(Some(1000.0))
        );
        assert_eq!(dist_pred.distance_side, JoinSide::None);
        assert!(remainder.is_some()); // Should have remainder (the id filter)
        let remainder = remainder.unwrap();

        // Remainder should be: left_id = 1
        let remainder_binary = remainder.as_any().downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(remainder_binary.op(), &Operator::Eq);

        // Left side should be left_id column
        let left_side = remainder_binary.left();
        let left_col = left_side.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_col.name(), "left_id");
        assert_eq!(left_col.index(), 0);

        // Right side should be literal 1
        let right_side = remainder_binary.right();
        let literal = right_side.as_any().downcast_ref::<Literal>().unwrap();
        match literal.value() {
            ScalarValue::Int32(Some(val)) => assert_eq!(val, &1),
            _ => panic!("Expected Int32(1) literal"),
        }
    }

    #[test]
    fn test_extract_spatial_predicate_no_spatial() {
        let column_indices = create_test_column_indices();

        // Create non-spatial expression: left_id = right_distance
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let right_distance = Arc::new(Column::new("right_distance", 3)) as Arc<dyn PhysicalExpr>;

        let non_spatial = Arc::new(BinaryExpr::new(left_id, Operator::Eq, right_distance))
            as Arc<dyn PhysicalExpr>;

        let result = extract_spatial_predicate(&non_spatial, &column_indices);
        assert!(result.is_none()); // No spatial predicate found
    }

    #[test]
    fn test_replace_join_filter_expr() {
        let schema = create_test_schema();
        let column_indices = create_test_column_indices();

        // Create original join filter
        let dummy_expr =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))) as Arc<dyn PhysicalExpr>;
        let original_filter = JoinFilter::new(dummy_expr, column_indices.clone(), schema.clone());

        // Create new expression that only references some columns
        let left_id = Arc::new(Column::new("right_distance", 3)) as Arc<dyn PhysicalExpr>;
        let literal_one =
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))) as Arc<dyn PhysicalExpr>;
        let new_expr =
            Arc::new(BinaryExpr::new(left_id, Operator::Eq, literal_one)) as Arc<dyn PhysicalExpr>;

        let new_filter = replace_join_filter_expr(&new_expr, &original_filter);

        // The new filter should have fewer columns since it only references right_distance
        assert_eq!(new_filter.column_indices().len(), 1);
        assert_eq!(new_filter.schema().fields().len(), 1);
        assert_eq!(
            new_filter.column_indices()[0],
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            }
        );

        let expr = new_filter.expression();
        let binary_expr = expr.as_any().downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(binary_expr.op(), &Operator::Eq);
        assert_eq!(
            binary_expr
                .left()
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            0
        );
    }

    #[test]
    fn test_transform_join_filter_with_spatial_predicate() {
        let schema = create_test_schema();
        let column_indices = create_test_column_indices();

        // Create ST_Intersects expression
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;

        let st_intersects_udf = create_dummy_st_intersects_udf();
        let args = vec![left_geom, right_geom];
        let st_intersects = create_spatial_function_expr(st_intersects_udf, args);
        let st_intersects_expr = st_intersects as Arc<dyn PhysicalExpr>;

        let join_filter = JoinFilter::new(st_intersects_expr, column_indices, schema);

        let result = transform_join_filter(&join_filter);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        assert!(matches!(spatial_pred, SpatialPredicate::Relation(_)));
        assert!(remainder.is_none()); // No remainder for simple spatial predicate
    }

    #[test]
    fn test_transform_join_filter_with_spatial_and_non_spatial() {
        let schema = create_test_schema();
        let column_indices = create_test_column_indices();

        // Create ST_DWithin(left_geom, right_geom, 1000.0) AND left_id = 42
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let distance =
            Arc::new(Literal::new(ScalarValue::Float64(Some(1000.0)))) as Arc<dyn PhysicalExpr>;
        let literal_42 =
            Arc::new(Literal::new(ScalarValue::Int32(Some(42)))) as Arc<dyn PhysicalExpr>;

        let st_dwithin_udf = create_dummy_st_dwithin_udf();
        let st_dwithin_args = vec![left_geom, right_geom, distance];
        let st_dwithin = create_spatial_function_expr(st_dwithin_udf, st_dwithin_args);
        let st_dwithin_expr = st_dwithin as Arc<dyn PhysicalExpr>;

        let id_filter =
            Arc::new(BinaryExpr::new(left_id, Operator::Eq, literal_42)) as Arc<dyn PhysicalExpr>;

        let combined_expr = Arc::new(BinaryExpr::new(st_dwithin_expr, Operator::And, id_filter))
            as Arc<dyn PhysicalExpr>;

        let join_filter = JoinFilter::new(combined_expr, column_indices, schema);

        let result = transform_join_filter(&join_filter);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        assert!(matches!(spatial_pred, SpatialPredicate::Distance(_)));
        assert!(remainder.is_some()); // Should have remainder for the id filter

        // The remainder should have fewer columns since it only references left_id
        let remainder_filter = remainder.unwrap();
        assert!(remainder_filter.column_indices().len() < join_filter.column_indices().len());
    }

    #[test]
    fn test_complex_nested_spatial_and_filters() {
        let schema = create_test_schema();
        let column_indices = create_test_column_indices();

        // Create (left_id > 10 AND ST_Intersects(left_geom, right_geom)) AND right_distance < 500.0
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let right_distance = Arc::new(Column::new("right_distance", 3)) as Arc<dyn PhysicalExpr>;

        let literal_10 =
            Arc::new(Literal::new(ScalarValue::Int32(Some(10)))) as Arc<dyn PhysicalExpr>;
        let literal_500 =
            Arc::new(Literal::new(ScalarValue::Float64(Some(500.0)))) as Arc<dyn PhysicalExpr>;

        // Build left_id > 10
        let left_filter =
            Arc::new(BinaryExpr::new(left_id, Operator::Gt, literal_10)) as Arc<dyn PhysicalExpr>;

        // Build ST_Intersects(left_geom, right_geom)
        let st_intersects_udf = create_dummy_st_intersects_udf();
        let st_intersects_args = vec![left_geom, right_geom];
        let st_intersects = create_spatial_function_expr(st_intersects_udf, st_intersects_args);
        let st_intersects_expr = st_intersects as Arc<dyn PhysicalExpr>;

        // Build right_distance < 500.0
        let right_filter = Arc::new(BinaryExpr::new(right_distance, Operator::Lt, literal_500))
            as Arc<dyn PhysicalExpr>;

        // Combine: (left_id > 10 AND ST_Intersects(left_geom, right_geom))
        let inner_and = Arc::new(BinaryExpr::new(
            left_filter,
            Operator::And,
            st_intersects_expr,
        )) as Arc<dyn PhysicalExpr>;

        // Final: (left_id > 10 AND ST_Intersects(left_geom, right_geom)) AND right_distance < 500.0
        let complex_expr = Arc::new(BinaryExpr::new(inner_and, Operator::And, right_filter))
            as Arc<dyn PhysicalExpr>;

        let join_filter = JoinFilter::new(complex_expr, column_indices, schema);

        let result = transform_join_filter(&join_filter);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        assert!(matches!(spatial_pred, SpatialPredicate::Relation(_)));
        assert!(remainder.is_some()); // Should have remainder combining both non-spatial filters
        let remainder = remainder.unwrap();
        let binary_expr = remainder
            .expression()
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        assert_eq!(binary_expr.op(), &Operator::And);

        let left_expr = binary_expr
            .left()
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        assert_eq!(left_expr.op(), &Operator::Gt);
        let left_id_expr = left_expr.left().as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_id_expr.name(), "left_id");
        assert_eq!(left_id_expr.index(), 0);

        let right_expr = binary_expr
            .right()
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        assert_eq!(right_expr.op(), &Operator::Lt);
        let right_distance_expr = right_expr.left().as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(right_distance_expr.name(), "right_distance");
        assert_eq!(right_distance_expr.index(), 1);

        assert_eq!(remainder.column_indices().len(), 2);
        assert_eq!(
            remainder.column_indices()[0],
            ColumnIndex {
                index: 0,
                side: JoinSide::Left,
            }
        );
        assert_eq!(
            remainder.column_indices()[1],
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            }
        );
    }

    // Helper to create dummy ST_KNN UDF for testing
    fn create_dummy_st_knn_udf() -> Arc<ScalarUDF> {
        Arc::new(ScalarUDF::from(SimpleScalarUDF::new(
            "st_knn",
            vec![
                WKB_GEOMETRY.storage_type().clone(),
                WKB_GEOMETRY.storage_type().clone(),
                DataType::Int32,
                DataType::Boolean,
            ],
            DataType::Boolean,
            datafusion_expr::Volatility::Immutable,
            Arc::new(|_| Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))))),
        )))
    }

    #[test]
    fn test_match_knn_predicate_basic() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN(left_geom, right_geom, 5, false)
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        assert!(predicate.is_some());

        let pred = predicate.unwrap();
        // Verify left argument is reprojected to left side
        let left_arg_col = pred.left.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_arg_col.index(), 1);
        assert_eq!(left_arg_col.name(), "left_geom");

        // Verify right argument is reprojected to right side
        let right_arg_col = pred.right.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(right_arg_col.index(), 0);
        assert_eq!(right_arg_col.name(), "right_geom");

        // Verify k is literal value 5
        assert_eq!(pred.k, 5);

        // Verify use_spheroid is literal value false
        assert!(!pred.use_spheroid);
    }

    #[test]
    fn test_match_knn_predicate_inverted() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN(right_geom, left_geom, 3, false) - this should be inverted to left, right order
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![right_geom, left_geom, k_literal, use_spheroid_literal]; // Note: right, left order
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        assert!(predicate.is_some());

        let pred = predicate.unwrap();
        // After inversion, left_arg should be the original left_geom
        let left_arg_col = pred.left.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_arg_col.index(), 0);
        assert_eq!(left_arg_col.name(), "right_geom");

        // After inversion, right_arg should be the original right_geom
        let right_arg_col = pred.right.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(right_arg_col.index(), 1);
        assert_eq!(right_arg_col.name(), "left_geom");
    }

    #[test]
    fn test_match_knn_predicate_same_side_fails() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN(left_geom, left_id, 5, false) - both from same side, should fail
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, left_id, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        assert!(predicate.is_none()); // Should fail - both args from same side
    }

    #[test]
    fn test_match_knn_predicate_spheroid_true_accepted() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN(left_geom, right_geom, 5, true) - should be accepted with use_spheroid=true
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        assert!(predicate.is_some()); // Should succeed - use_spheroid=true is now supported

        let knn_pred = predicate.unwrap();
        assert_eq!(knn_pred.k, 5);
        assert!(knn_pred.use_spheroid); // Verify spheroid flag is set
    }

    #[test]
    fn test_extract_spatial_predicate_knn() {
        let column_indices = create_test_column_indices();

        // Test simple ST_KNN
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(10)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);
        let st_knn_expr = st_knn as Arc<dyn PhysicalExpr>;

        let result = extract_spatial_predicate(&st_knn_expr, &column_indices);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        let SpatialPredicate::KNearestNeighbors(knn_pred) = spatial_pred else {
            panic!("Expected SpatialPredicate::KNearestNeighbors");
        };
        assert_eq!(
            knn_pred
                .left
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            1
        );
        assert_eq!(
            knn_pred
                .right
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            0
        );
        assert!(remainder.is_none()); // No remainder for simple predicate
    }

    #[test]
    fn test_extract_spatial_predicate_knn_with_and() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN(left_geom, right_geom, 5, false) AND left_id = 1
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;
        let literal_one =
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let st_knn_args = vec![left_geom, right_geom, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, st_knn_args);
        let st_knn_expr = st_knn as Arc<dyn PhysicalExpr>;

        let id_filter =
            Arc::new(BinaryExpr::new(left_id, Operator::Eq, literal_one)) as Arc<dyn PhysicalExpr>;

        let and_expr = Arc::new(BinaryExpr::new(st_knn_expr, Operator::And, id_filter))
            as Arc<dyn PhysicalExpr>;

        let result = extract_spatial_predicate(&and_expr, &column_indices);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        let SpatialPredicate::KNearestNeighbors(knn_pred) = spatial_pred else {
            panic!("Expected SpatialPredicate::KNearestNeighbors");
        };
        assert_eq!(
            knn_pred
                .left
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            1
        );
        assert_eq!(
            knn_pred
                .right
                .as_any()
                .downcast_ref::<Column>()
                .unwrap()
                .index(),
            0
        );
        assert!(remainder.is_some()); // Should have remainder (the id filter)
        let remainder = remainder.unwrap();

        // Remainder should be: left_id = 1
        let remainder_binary = remainder.as_any().downcast_ref::<BinaryExpr>().unwrap();
        assert_eq!(remainder_binary.op(), &Operator::Eq);

        // Left side should be left_id column
        let left_side = remainder_binary.left();
        let left_col = left_side.as_any().downcast_ref::<Column>().unwrap();
        assert_eq!(left_col.name(), "left_id");
        assert_eq!(left_col.index(), 0);

        // Right side should be literal 1
        let right_side = remainder_binary.right();
        let literal = right_side.as_any().downcast_ref::<Literal>().unwrap();
        match literal.value() {
            ScalarValue::Int32(Some(val)) => assert_eq!(val, &1),
            _ => panic!("Expected Int32(1) literal"),
        }
    }

    #[test]
    fn test_transform_join_filter_with_knn_predicate() {
        let schema = create_test_schema();
        let column_indices = create_test_column_indices();

        // Create ST_KNN expression
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);
        let st_knn_expr = st_knn as Arc<dyn PhysicalExpr>;

        let join_filter = JoinFilter::new(st_knn_expr, column_indices, schema);

        let result = transform_join_filter(&join_filter);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        assert!(matches!(
            spatial_pred,
            SpatialPredicate::KNearestNeighbors(_)
        ));
        assert!(remainder.is_none()); // No remainder for simple spatial predicate
    }

    #[test]
    fn test_match_knn_predicate_insufficient_args() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN with only 3 arguments (insufficient - needs 4)
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_literal]; // Missing use_spheroid arg
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        assert!(predicate.is_none()); // Should fail due to insufficient arguments
    }

    #[test]
    fn test_match_knn_predicate_wrong_function_name() {
        let column_indices = create_test_column_indices();

        // Create a function that's not ST_KNN
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_intersects_udf = create_dummy_st_intersects_udf(); // Wrong function
        let args = vec![left_geom, right_geom, k_literal, use_spheroid_literal];
        let st_intersects = create_spatial_function_expr(st_intersects_udf, args);

        let predicate = match_knn_predicate(&st_intersects, &column_indices);
        assert!(predicate.is_none()); // Should fail due to wrong function name
    }

    #[test]
    fn test_match_knn_predicate_non_column_arguments() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN with literal geometry arguments (not column references)
        let left_literal = Arc::new(Literal::new(ScalarValue::Binary(Some(
            b"POINT(0 0)".to_vec(),
        )))) as Arc<dyn PhysicalExpr>;
        let right_literal = Arc::new(Literal::new(ScalarValue::Binary(Some(
            b"POINT(1 1)".to_vec(),
        )))) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_literal, right_literal, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        assert!(predicate.is_none()); // Should fail - geometry args are not column references
    }

    #[test]
    fn test_match_knn_predicate_complex_k_expressions() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN with complex k expression (column + literal)
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let literal_two =
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))) as Arc<dyn PhysicalExpr>;
        let k_expr = Arc::new(BinaryExpr::new(left_id, Operator::Plus, literal_two))
            as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_expr, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        assert!(predicate.is_none()); // Should fail - complex k expressions are no longer supported
    }

    #[test]
    fn test_match_knn_predicate_complex_use_spheroid_expressions() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN with complex use_spheroid expression (column comparison)
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let literal_one =
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_expr =
            Arc::new(BinaryExpr::new(left_id, Operator::Gt, literal_one)) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_literal, use_spheroid_expr];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        assert!(predicate.is_none()); // Should fail - complex use_spheroid expressions are no longer supported
    }

    #[test]
    fn test_match_knn_predicate_both_sides_in_k_expression() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN with k expression that references both sides
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let right_distance = Arc::new(Column::new("right_distance", 3)) as Arc<dyn PhysicalExpr>;
        // k expression that references both left and right sides
        let k_expr = Arc::new(BinaryExpr::new(left_id, Operator::Plus, right_distance))
            as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_expr, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        // Should fail because k expression references both sides, which is not allowed
        assert!(predicate.is_none());
    }

    #[test]
    fn test_extract_spatial_predicate_knn_with_multiple_clauses() {
        let column_indices = create_test_column_indices();

        // Create complex expression: ST_KNN(...) AND left_id > 0 AND right_distance < 100.0
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>;
        let right_distance = Arc::new(Column::new("right_distance", 3)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;
        let literal_zero =
            Arc::new(Literal::new(ScalarValue::Int32(Some(0)))) as Arc<dyn PhysicalExpr>;
        let literal_hundred =
            Arc::new(Literal::new(ScalarValue::Float64(Some(100.0)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let st_knn_args = vec![left_geom, right_geom, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, st_knn_args);
        let st_knn_expr = st_knn as Arc<dyn PhysicalExpr>;

        let id_filter =
            Arc::new(BinaryExpr::new(left_id, Operator::Gt, literal_zero)) as Arc<dyn PhysicalExpr>;
        let distance_filter = Arc::new(BinaryExpr::new(
            right_distance,
            Operator::Lt,
            literal_hundred,
        )) as Arc<dyn PhysicalExpr>;

        // Build: ST_KNN(...) AND left_id > 0 AND right_distance < 100.0
        let and1 = Arc::new(BinaryExpr::new(st_knn_expr, Operator::And, id_filter))
            as Arc<dyn PhysicalExpr>;
        let and2 = Arc::new(BinaryExpr::new(and1, Operator::And, distance_filter))
            as Arc<dyn PhysicalExpr>;

        let result = extract_spatial_predicate(&and2, &column_indices);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        let SpatialPredicate::KNearestNeighbors(knn_pred) = spatial_pred else {
            panic!("Expected SpatialPredicate::KNearestNeighbors");
        };

        // Verify KNN predicate parameters
        assert_eq!(knn_pred.k, 3); // k is a literal value
        assert!(!knn_pred.use_spheroid); // use_spheroid is a literal value

        // Should have remainder with both filter conditions
        assert!(remainder.is_some());
        let remainder_expr = remainder.unwrap();

        // Remainder should be: left_id > 0 AND right_distance < 100.0
        let remainder_and = remainder_expr
            .as_any()
            .downcast_ref::<BinaryExpr>()
            .unwrap();
        assert_eq!(remainder_and.op(), &Operator::And);
    }

    #[test]
    fn test_match_knn_predicate_nested_expressions() {
        let column_indices = create_test_column_indices();

        // Create ST_KNN with nested expressions for geometry arguments
        let left_geom_col = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom_col = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;

        // Wrap in IsNotNull expressions (common pattern)
        let left_geom = Arc::new(IsNotNullExpr::new(left_geom_col)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(IsNotNullExpr::new(right_geom_col)) as Arc<dyn PhysicalExpr>;

        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);

        let predicate = match_knn_predicate(&st_knn, &column_indices);
        assert!(predicate.is_some()); // Should succeed - nested expressions are allowed

        let pred = predicate.unwrap();
        // The wrapped columns should still be detected correctly
        let left_is_not_null = pred.left.as_any().downcast_ref::<IsNotNullExpr>().unwrap();
        let left_col = left_is_not_null
            .arg()
            .as_any()
            .downcast_ref::<Column>()
            .unwrap();
        assert_eq!(left_col.name(), "left_geom");
        assert_eq!(left_col.index(), 1); // reprojected index for left side

        let right_is_not_null = pred.right.as_any().downcast_ref::<IsNotNullExpr>().unwrap();
        let right_col = right_is_not_null
            .arg()
            .as_any()
            .downcast_ref::<Column>()
            .unwrap();
        assert_eq!(right_col.name(), "right_geom");
        assert_eq!(right_col.index(), 0); // reprojected index for right side
    }

    #[test]
    fn test_extract_spatial_predicate_knn_no_remainder() {
        let column_indices = create_test_column_indices();

        // Test ST_KNN as standalone predicate (no AND/OR combinations)
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let k_literal =
            Arc::new(Literal::new(ScalarValue::Int32(Some(7)))) as Arc<dyn PhysicalExpr>;
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, k_literal, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);
        let st_knn_expr = st_knn as Arc<dyn PhysicalExpr>;

        let result = extract_spatial_predicate(&st_knn_expr, &column_indices);
        assert!(result.is_some());

        let (spatial_pred, remainder) = result.unwrap();
        let SpatialPredicate::KNearestNeighbors(knn_pred) = spatial_pred else {
            panic!("Expected SpatialPredicate::KNearestNeighbors");
        };

        // Verify predicate details
        assert_eq!(knn_pred.k, 7); // literal k
        assert!(!knn_pred.use_spheroid); // literal use_spheroid

        // Should have no remainder for standalone KNN predicate
        assert!(remainder.is_none());
    }

    #[test]
    fn test_transform_join_filter_with_complex_knn_predicate() {
        let schema = create_test_schema();
        let column_indices = create_test_column_indices();

        // Create complex KNN expression with column-based k value
        let left_geom = Arc::new(Column::new("left_geom", 1)) as Arc<dyn PhysicalExpr>;
        let right_geom = Arc::new(Column::new("right_geom", 2)) as Arc<dyn PhysicalExpr>;
        let left_id = Arc::new(Column::new("left_id", 0)) as Arc<dyn PhysicalExpr>; // Use left_id as k
        let use_spheroid_literal =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false)))) as Arc<dyn PhysicalExpr>;

        let st_knn_udf = create_dummy_st_knn_udf();
        let args = vec![left_geom, right_geom, left_id, use_spheroid_literal];
        let st_knn = create_spatial_function_expr(st_knn_udf, args);
        let st_knn_expr = st_knn as Arc<dyn PhysicalExpr>;

        let join_filter = JoinFilter::new(st_knn_expr, column_indices, schema);

        let result = transform_join_filter(&join_filter);
        assert!(result.is_none()); // Should fail - k must be a literal value
    }

    #[test]
    fn test_is_spatial_predicate_supported() {
        // Planar geometry field
        let geom_field = WKB_GEOMETRY.to_storage_field("geom", false).unwrap();
        let schema = Arc::new(Schema::new(vec![geom_field.clone()]));
        let col_expr = Arc::new(Column::new("geom", 0)) as Arc<dyn PhysicalExpr>;
        let rel_pred = RelationPredicate::new(
            col_expr.clone(),
            col_expr.clone(),
            SpatialRelationType::Intersects,
        );
        let spatial_pred = SpatialPredicate::Relation(rel_pred);
        assert!(is_spatial_predicate_supported(&spatial_pred, &schema, &schema).unwrap());

        // Geography field (should NOT be supported)
        let geog_field = WKB_GEOGRAPHY.to_storage_field("geog", false).unwrap();
        let geog_schema = Arc::new(Schema::new(vec![geog_field.clone()]));
        let geog_col_expr = Arc::new(Column::new("geog", 0)) as Arc<dyn PhysicalExpr>;
        let rel_pred_geog = RelationPredicate::new(
            geog_col_expr.clone(),
            geog_col_expr.clone(),
            SpatialRelationType::Intersects,
        );
        let spatial_pred_geog = SpatialPredicate::Relation(rel_pred_geog);
        assert!(
            !is_spatial_predicate_supported(&spatial_pred_geog, &geog_schema, &geog_schema)
                .unwrap()
        );
    }

    #[test]
    fn test_is_knn_predicate_supported() {
        // ST_KNN(left, right)
        let left_schema = Arc::new(Schema::new(vec![WKB_GEOMETRY
            .to_storage_field("geom", false)
            .unwrap()]));
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            WKB_GEOMETRY.to_storage_field("geom", false).unwrap(),
        ]));
        let left_col_expr = Arc::new(Column::new("geom", 0)) as Arc<dyn PhysicalExpr>;
        let right_col_expr = Arc::new(Column::new("geom", 1)) as Arc<dyn PhysicalExpr>;
        let knn_pred = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            left_col_expr.clone(),
            right_col_expr.clone(),
            5,
            false,
            JoinSide::Left,
        ));
        assert!(is_spatial_predicate_supported(&knn_pred, &left_schema, &right_schema).unwrap());

        // ST_KNN(right, left)
        let knn_pred = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            right_col_expr.clone(),
            left_col_expr.clone(),
            5,
            false,
            JoinSide::Right,
        ));
        assert!(is_spatial_predicate_supported(&knn_pred, &left_schema, &right_schema).unwrap());

        // ST_KNN with geography (should NOT be supported)
        let left_geog_schema = Arc::new(Schema::new(vec![WKB_GEOGRAPHY
            .to_storage_field("geog", false)
            .unwrap()]));
        assert!(
            !is_spatial_predicate_supported(&knn_pred, &left_geog_schema, &right_schema).unwrap()
        );

        let right_geog_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            WKB_GEOGRAPHY.to_storage_field("geog", false).unwrap(),
        ]));
        assert!(
            !is_spatial_predicate_supported(&knn_pred, &left_schema, &right_geog_schema).unwrap()
        );
    }

    #[test]
    fn test_collect_spatial_predicate_names() {
        // ST_Intersects should be collected
        let st_intersects_udf = create_dummy_st_intersects_udf();
        let st_intersects_expr = Expr::ScalarFunction(datafusion_expr::expr::ScalarFunction {
            func: st_intersects_udf,
            args: vec![col("geom1"), col("geom2")],
        });
        let names = collect_spatial_predicate_names(&st_intersects_expr);
        assert_eq!(names, HashSet::from(["st_intersects".to_string()]));

        // ST_Distance(geom1, geom2) < 100 should be collected as st_dwithin
        let st_distance_udf = create_dummy_st_distance_udf();
        let st_distance_expr = Expr::ScalarFunction(datafusion_expr::expr::ScalarFunction {
            func: st_distance_udf,
            args: vec![col("geom1"), col("geom2")],
        });
        let distance_lt_expr = Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
            left: Box::new(st_distance_expr.clone()),
            op: Operator::Lt,
            right: Box::new(lit(100.0)),
        });
        let names = collect_spatial_predicate_names(&distance_lt_expr);
        assert_eq!(names, HashSet::from(["st_dwithin".to_string()]));

        // ST_Distance(geom1, geom2) > 100 should not be collected (wrong comparison direction)
        let distance_gt_expr = Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
            left: Box::new(st_distance_expr.clone()),
            op: Operator::Gt,
            right: Box::new(lit(100.0)),
        });
        let names = collect_spatial_predicate_names(&distance_gt_expr);
        assert!(names.is_empty());

        // AND expression: spatial predicate should be collected through conjunction
        let and_expr = Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
            left: Box::new(st_intersects_expr.clone()),
            op: Operator::And,
            right: Box::new(col("id").eq(lit(1))),
        });
        let names = collect_spatial_predicate_names(&and_expr);
        assert_eq!(names, HashSet::from(["st_intersects".to_string()]));

        // Non-spatial expressions should return empty set

        // Simple column comparison
        let non_spatial_expr = col("id").eq(lit(1));
        let names = collect_spatial_predicate_names(&non_spatial_expr);
        assert!(names.is_empty());

        // Not a spatial relationship function
        let non_st_func = Expr::ScalarFunction(datafusion_expr::expr::ScalarFunction {
            func: Arc::new(ScalarUDF::from(SimpleScalarUDF::new(
                "st_non_spatial_relation_func",
                vec![DataType::Int32],
                DataType::Boolean,
                datafusion_expr::Volatility::Immutable,
                Arc::new(|_| Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))))),
            ))),
            args: vec![col("id")],
        });
        let names = collect_spatial_predicate_names(&non_st_func);
        assert!(names.is_empty());

        // AND expression with no spatial predicates
        let non_spatial_and = Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
            left: Box::new(col("id").eq(lit(1))),
            op: Operator::And,
            right: Box::new(col("name").eq(lit("test"))),
        });
        let names = collect_spatial_predicate_names(&non_spatial_and);
        assert!(names.is_empty());
    }

    #[test]
    fn test_find_st_knn_call_simple() {
        let st_knn_udf = create_dummy_st_knn_udf();
        let expr = st_knn_udf.call(vec![col("l.geom"), col("r.geom"), lit(5i32), lit(false)]);

        let result = find_st_knn_call(&expr);
        assert!(result.is_some());
        assert_eq!(result.unwrap().func.name(), "st_knn");
    }

    #[test]
    fn test_find_st_knn_call_nested_in_and() {
        let st_knn_udf = create_dummy_st_knn_udf();
        let knn_expr = st_knn_udf.call(vec![col("l.geom"), col("r.geom"), lit(5i32), lit(false)]);

        // ST_KNN(...) AND l.id > 5
        let and_expr = Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
            left: Box::new(knn_expr),
            op: Operator::And,
            right: Box::new(col("l.id").gt(lit(5))),
        });

        let result = find_st_knn_call(&and_expr);
        assert!(result.is_some());
        assert_eq!(result.unwrap().func.name(), "st_knn");
    }

    #[test]
    fn test_find_st_knn_call_deeply_nested() {
        let st_knn_udf = create_dummy_st_knn_udf();
        let knn_expr = st_knn_udf.call(vec![col("l.geom"), col("r.geom"), lit(5i32), lit(false)]);

        // (l.id > 5 AND r.id > 3) AND ST_KNN(...)
        let inner_and = Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
            left: Box::new(col("l.id").gt(lit(5))),
            op: Operator::And,
            right: Box::new(col("r.id").gt(lit(3))),
        });
        let outer_and = Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
            left: Box::new(inner_and),
            op: Operator::And,
            right: Box::new(knn_expr),
        });

        let result = find_st_knn_call(&outer_and);
        assert!(result.is_some());
        assert_eq!(result.unwrap().func.name(), "st_knn");
    }

    #[test]
    fn test_find_st_knn_call_returns_none_for_non_knn() {
        let st_intersects_udf = create_dummy_st_intersects_udf();
        let expr = st_intersects_udf.call(vec![col("l.geom"), col("r.geom")]);

        assert!(find_st_knn_call(&expr).is_none());
    }

    #[test]
    fn test_find_st_knn_call_returns_none_for_non_and_binary() {
        let st_knn_udf = create_dummy_st_knn_udf();
        let knn_expr = st_knn_udf.call(vec![col("l.geom"), col("r.geom"), lit(5i32), lit(false)]);

        // ST_KNN(...) OR l.id > 5 — should NOT descend into OR
        let or_expr = Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
            left: Box::new(knn_expr),
            op: Operator::Or,
            right: Box::new(col("l.id").gt(lit(5))),
        });

        assert!(find_st_knn_call(&or_expr).is_none());
    }

    /// Helper: build a DFSchema with the given qualified table name and column names.
    fn make_df_schema(table: &str, columns: &[(&str, DataType)]) -> DFSchema {
        use arrow::datatypes::{Field, Schema};
        use datafusion_common::TableReference;

        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, dt)| Field::new(*name, dt.clone(), true))
            .collect();
        let schema = Schema::new(fields);
        DFSchema::try_from_qualified_schema(TableReference::bare(table), &schema).unwrap()
    }

    #[test]
    fn test_find_knn_query_side_left() {
        let left_schema =
            make_df_schema("l", &[("id", DataType::Int32), ("geom", DataType::Binary)]);
        let right_schema =
            make_df_schema("r", &[("id", DataType::Int32), ("geom", DataType::Binary)]);

        let st_knn_udf = create_dummy_st_knn_udf();
        // ST_KNN(l.geom, r.geom, 5, false)  →  query side = left
        let expr = st_knn_udf.call(vec![col("l.geom"), col("r.geom"), lit(5i32), lit(false)]);

        let result = find_knn_query_side(&expr, &left_schema, &right_schema);
        assert_eq!(result, Some(KNNJoinQuerySide::Left));
    }

    #[test]
    fn test_find_knn_query_side_right() {
        let left_schema =
            make_df_schema("l", &[("id", DataType::Int32), ("geom", DataType::Binary)]);
        let right_schema =
            make_df_schema("r", &[("id", DataType::Int32), ("geom", DataType::Binary)]);

        let st_knn_udf = create_dummy_st_knn_udf();
        // ST_KNN(r.geom, l.geom, 5, false)  →  query side = right
        let expr = st_knn_udf.call(vec![col("r.geom"), col("l.geom"), lit(5i32), lit(false)]);

        let result = find_knn_query_side(&expr, &left_schema, &right_schema);
        assert_eq!(result, Some(KNNJoinQuerySide::Right));
    }

    #[test]
    fn test_find_knn_query_side_nested_in_and() {
        let left_schema =
            make_df_schema("l", &[("id", DataType::Int32), ("geom", DataType::Binary)]);
        let right_schema =
            make_df_schema("r", &[("id", DataType::Int32), ("geom", DataType::Binary)]);

        let st_knn_udf = create_dummy_st_knn_udf();
        let knn_expr = st_knn_udf.call(vec![col("l.geom"), col("r.geom"), lit(5i32), lit(false)]);

        // l.id = r.id AND ST_KNN(l.geom, r.geom, 5, false)
        let and_expr = Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
            left: Box::new(col("l.id").eq(col("r.id"))),
            op: Operator::And,
            right: Box::new(knn_expr),
        });

        let result = find_knn_query_side(&and_expr, &left_schema, &right_schema);
        assert_eq!(result, Some(KNNJoinQuerySide::Left));
    }

    #[test]
    fn test_find_knn_query_side_no_knn_returns_none() {
        let left_schema =
            make_df_schema("l", &[("id", DataType::Int32), ("geom", DataType::Binary)]);
        let right_schema =
            make_df_schema("r", &[("id", DataType::Int32), ("geom", DataType::Binary)]);

        // Non-KNN expression
        let expr = col("l.id").eq(col("r.id"));

        let result = find_knn_query_side(&expr, &left_schema, &right_schema);
        assert_eq!(result, None);
    }

    #[test]
    fn test_find_knn_query_side_literal_first_arg_returns_none() {
        let left_schema =
            make_df_schema("l", &[("id", DataType::Int32), ("geom", DataType::Binary)]);
        let right_schema =
            make_df_schema("r", &[("id", DataType::Int32), ("geom", DataType::Binary)]);

        let st_knn_udf = create_dummy_st_knn_udf();
        // First argument is a literal, not a column reference
        let expr = st_knn_udf.call(vec![lit(42), col("r.geom"), lit(5i32), lit(false)]);

        let result = find_knn_query_side(&expr, &left_schema, &right_schema);
        assert_eq!(result, None);
    }
}
