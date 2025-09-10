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

use arrow_schema::{DataType, Schema};
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::Operator;
use datafusion_physical_expr::{
    expressions::{BinaryExpr, Column, Literal},
    PhysicalExpr, ScalarFunctionExpr,
};
use geo_traits::Dimensions;
use sedona_common::sedona_internal_err;
use sedona_geometry::{bounding_box::BoundingBox, bounds::wkb_bounds_xy, interval::IntervalTrait};
use sedona_schema::datatypes::SedonaType;

use crate::{
    statistics::GeoStatistics,
    utils::{parse_distance_predicate, ParsedDistancePredicate},
};

/// Simplified parsed spatial filter
///
/// This enumerator represents a parsed version of the [PhysicalExpr] provided as a
/// filter to an implementation of a table provider or file opener. This is intended
/// as a means by which to process an arbitrary PhysicalExpr against column statistics
/// to attempt pruning unnecessary files or parts of files specifically with respect
/// to a spatial filter (i.e., non-spatial filters we leave to an underlying
/// implementation).
#[derive(Debug)]
pub enum SpatialFilter {
    /// ST_Intersects(\<column\>, \<literal\>) or ST_Intersects(\<literal\>, \<column\>)
    Intersects(Column, BoundingBox),
    /// ST_CoveredBy(\<column\>, \<literal\>) or ST_CoveredBy(\<literal\>, \<column\>)
    CoveredBy(Column, BoundingBox),
    /// ST_HasZ(\<column\>)
    HasZ(Column),
    /// Logical AND
    And(Box<SpatialFilter>, Box<SpatialFilter>),
    /// Logical OR
    Or(Box<SpatialFilter>, Box<SpatialFilter>),
    /// A literal FALSE, which is never true
    LiteralFalse,
    /// An expression we don't know about, which we assume could be true
    Unknown,
}

impl SpatialFilter {
    /// Returns true if there is any chance the expression might be true
    ///
    /// In other words, returns false if and only if the expression is guaranteed
    /// to be false.
    pub fn evaluate(&self, table_stats: &[GeoStatistics]) -> bool {
        match self {
            SpatialFilter::Intersects(column, bounds) => {
                Self::evaluate_intersects_bbox(&table_stats[column.index()], bounds)
            }
            SpatialFilter::CoveredBy(column, bounds) => {
                Self::evaluate_covered_by_bbox(&table_stats[column.index()], bounds)
            }
            SpatialFilter::HasZ(column) => Self::evaluate_has_z(&table_stats[column.index()]),
            SpatialFilter::And(lhs, rhs) => Self::evaluate_and(lhs, rhs, table_stats),
            SpatialFilter::Or(lhs, rhs) => Self::evaluate_or(lhs, rhs, table_stats),
            SpatialFilter::LiteralFalse => false,
            SpatialFilter::Unknown => true,
        }
    }

    fn evaluate_intersects_bbox(column_stats: &GeoStatistics, bounds: &BoundingBox) -> bool {
        if let Some(bbox) = column_stats.bbox() {
            bbox.intersects(bounds)
        } else {
            true
        }
    }

    fn evaluate_covered_by_bbox(column_stats: &GeoStatistics, bounds: &BoundingBox) -> bool {
        if let Some(bbox) = column_stats.bbox() {
            bounds.contains(bbox)
        } else {
            true
        }
    }

    fn evaluate_has_z(column_stats: &GeoStatistics) -> bool {
        if let Some(bbox) = column_stats.bbox() {
            if let Some(z) = bbox.z() {
                if z.is_empty() {
                    return false;
                }
            }
        }

        if let Some(geometry_types) = column_stats.geometry_types() {
            for geometry_type in geometry_types {
                match geometry_type.dimensions() {
                    Dimensions::Xyz | Dimensions::Xyzm => return true,
                    _ => {}
                }
            }

            return false;
        }

        true
    }

    fn evaluate_and(lhs: &Self, rhs: &Self, table_stats: &[GeoStatistics]) -> bool {
        let maybe_lhs = lhs.evaluate(table_stats);
        let maybe_rhs = rhs.evaluate(table_stats);
        maybe_lhs && maybe_rhs
    }

    fn evaluate_or(lhs: &Self, rhs: &Self, table_stats: &[GeoStatistics]) -> bool {
        let maybe_lhs = lhs.evaluate(table_stats);
        let maybe_rhs = rhs.evaluate(table_stats);
        maybe_lhs || maybe_rhs
    }

    /// Construct a SpatialPredicate from a [PhysicalExpr]
    ///
    /// Parses expr to extract known expressions we can evaluate against statistics.
    pub fn try_from_expr(expr: &Arc<dyn PhysicalExpr>) -> Result<Self> {
        if let Some(spatial_filter) = Self::try_from_range_predicate(expr)? {
            Ok(spatial_filter)
        } else if let Some(spatial_filter) = Self::try_from_distance_predicate(expr)? {
            Ok(spatial_filter)
        } else if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
            match binary_expr.op() {
                Operator::And => Ok(Self::And(
                    Box::new(Self::try_from_expr(binary_expr.left())?),
                    Box::new(Self::try_from_expr(binary_expr.right())?),
                )),
                Operator::Or => Ok(Self::Or(
                    Box::new(Self::try_from_expr(binary_expr.left())?),
                    Box::new(Self::try_from_expr(binary_expr.right())?),
                )),
                // Not a binary expression we know about
                _ => Ok(Self::Unknown),
            }
        } else if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
            if let ScalarValue::Boolean(Some(value)) = literal.value() {
                match value {
                    true => Ok(Self::Unknown),
                    false => Ok(Self::LiteralFalse),
                }
            } else {
                // Not a literal we know about
                Ok(Self::Unknown)
            }
        } else {
            // Not an expression we know about
            Ok(Self::Unknown)
        }
    }

    fn try_from_range_predicate(expr: &Arc<dyn PhysicalExpr>) -> Result<Option<Self>> {
        let Some(scalar_fun) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() else {
            return Ok(None);
        };

        let raw_args = scalar_fun.args();
        let args = parse_args(raw_args);
        let fun_name = scalar_fun.fun().name();
        match fun_name {
            "st_intersects" | "st_equals" | "st_touches" => {
                if args.len() != 2 {
                    return sedona_internal_err!("unexpected argument count in filter evaluation");
                }

                match (&args[0], &args[1]) {
                    (ArgRef::Col(column), ArgRef::Lit(literal))
                    | (ArgRef::Lit(literal), ArgRef::Col(column)) => {
                        match literal_bounds(literal) {
                            Ok(literal_bounds) => {
                                Ok(Some(Self::Intersects(column.clone(), literal_bounds)))
                            }
                            Err(e) => Err(DataFusionError::External(Box::new(e))),
                        }
                    }
                    // Not between a literal and a column
                    _ => Ok(Some(Self::Unknown)),
                }
            }
            "st_within" | "st_covered_by" | "st_coveredby" => {
                if args.len() != 2 {
                    return sedona_internal_err!("unexpected argument count in filter evaluation");
                }

                match (&args[0], &args[1]) {
                    (ArgRef::Col(column), ArgRef::Lit(literal)) => {
                        // column within/covered_by literal -> CoveredBy filter
                        match literal_bounds(literal) {
                            Ok(literal_bounds) => {
                                Ok(Some(Self::CoveredBy(column.clone(), literal_bounds)))
                            }
                            Err(e) => Err(DataFusionError::External(Box::new(e))),
                        }
                    }
                    (ArgRef::Lit(literal), ArgRef::Col(column)) => {
                        // literal within/covered_by column -> Intersects filter
                        match literal_bounds(literal) {
                            Ok(literal_bounds) => {
                                Ok(Some(Self::Intersects(column.clone(), literal_bounds)))
                            }
                            Err(e) => Err(DataFusionError::External(Box::new(e))),
                        }
                    }
                    // Not between a literal and a column
                    _ => Ok(Some(Self::Unknown)),
                }
            }
            "st_contains" | "st_covers" => {
                if args.len() != 2 {
                    return sedona_internal_err!("unexpected argument count in filter evaluation");
                }

                match (&args[0], &args[1]) {
                    (ArgRef::Col(column), ArgRef::Lit(literal)) => {
                        // column contains/covers literal -> Intersects filter
                        // (column must potentially intersect literal to contain it)
                        match literal_bounds(literal) {
                            Ok(literal_bounds) => {
                                Ok(Some(Self::Intersects(column.clone(), literal_bounds)))
                            }
                            Err(e) => Err(DataFusionError::External(Box::new(e))),
                        }
                    }
                    (ArgRef::Lit(literal), ArgRef::Col(column)) => {
                        // literal contains/covers column -> CoveredBy filter
                        // (equivalent to st_within(column, literal))
                        match literal_bounds(literal) {
                            Ok(literal_bounds) => {
                                Ok(Some(Self::CoveredBy(column.clone(), literal_bounds)))
                            }
                            Err(e) => Err(DataFusionError::External(Box::new(e))),
                        }
                    }
                    // Not between a literal and a column
                    _ => Ok(Some(Self::Unknown)),
                }
            }
            "st_hasz" => {
                if args.len() != 1 {
                    return sedona_internal_err!("unexpected argument count in filter evaluation");
                }

                match &args[0] {
                    ArgRef::Col(column) => Ok(Some(Self::HasZ(column.clone()))),
                    _ => Ok(Some(Self::Unknown)),
                }
            }
            _ => Ok(None),
        }
    }

    fn try_from_distance_predicate(expr: &Arc<dyn PhysicalExpr>) -> Result<Option<Self>> {
        let Some(ParsedDistancePredicate {
            arg0,
            arg1,
            arg_distance,
        }) = parse_distance_predicate(expr)
        else {
            return Ok(None);
        };

        let raw_args = [arg0, arg1, arg_distance];
        let args = parse_args(&raw_args);

        match (&args[0], &args[1], &args[2]) {
            (ArgRef::Col(column), ArgRef::Lit(literal), ArgRef::Lit(distance))
            | (ArgRef::Lit(literal), ArgRef::Col(column), ArgRef::Lit(distance)) => {
                match (
                    literal_bounds(literal),
                    distance.value().cast_to(&DataType::Float64)?,
                ) {
                    (Ok(literal_bounds), distance_scalar_value) => {
                        let ScalarValue::Float64(Some(dist)) = distance_scalar_value else {
                            return Ok(None);
                        };
                        if dist.is_nan() || dist < 0.0 {
                            return Ok(None);
                        }
                        let expanded_bounds = literal_bounds.expand_by(dist);
                        Ok(Some(Self::Intersects(column.clone(), expanded_bounds)))
                    }
                    (Err(e), _) => Err(DataFusionError::External(Box::new(e))),
                }
            }
            // Not between a literal and a column
            _ => Ok(Some(Self::Unknown)),
        }
    }
}

/// Internal utility to help match physical expression types
enum ArgRef<'a> {
    Col(Column),
    Lit(&'a Literal),
    Other,
}

fn literal_bounds(literal: &Literal) -> Result<BoundingBox> {
    let literal_field = literal.return_field(&Schema::empty())?;
    let sedona_type = SedonaType::from_storage_field(&literal_field)?;
    match &sedona_type {
        SedonaType::Wkb(_, _) | SedonaType::WkbView(_, _) => match literal.value() {
            ScalarValue::Binary(maybe_vec) | ScalarValue::BinaryView(maybe_vec) => {
                if let Some(vec) = maybe_vec {
                    return wkb_bounds_xy(vec).map_err(|e| DataFusionError::External(Box::new(e)));
                }
            }
            _ => {}
        },
        _ => {}
    }

    sedona_internal_err!("Unexpected scalar type in filter expression ({literal:?})")
}

fn parse_args(args: &[Arc<dyn PhysicalExpr>]) -> Vec<ArgRef<'_>> {
    args.iter()
        .map(|arg| {
            if let Some(column) = arg.as_any().downcast_ref::<Column>() {
                ArgRef::Col(column.clone())
            } else if let Some(literal) = arg.as_any().downcast_ref::<Literal>() {
                ArgRef::Lit(literal)
            } else {
                ArgRef::Other
            }
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod test {

    use arrow_schema::{DataType, Field};
    use datafusion_expr::{ScalarUDF, Signature, SimpleScalarUDF, Volatility};
    use rstest::rstest;
    use sedona_geometry::{bounding_box::BoundingBox, interval::Interval};
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_scalar;

    use super::*;

    fn dummy_st_hasz() -> ScalarUDF {
        SimpleScalarUDF::new_with_signature(
            "st_hasz",
            Signature::any(2, Volatility::Immutable),
            DataType::Boolean,
            Arc::new(|_args| Ok(ScalarValue::Boolean(Some(true)).into())),
        )
        .into()
    }

    fn dummy_unrelated() -> ScalarUDF {
        SimpleScalarUDF::new_with_signature(
            "st_not_a_predicate",
            Signature::any(2, Volatility::Immutable),
            DataType::Boolean,
            Arc::new(|_args| Ok(ScalarValue::Boolean(Some(true)).into())),
        )
        .into()
    }

    fn create_dummy_spatial_function(name: &str, arg_count: usize) -> ScalarUDF {
        SimpleScalarUDF::new_with_signature(
            name,
            Signature::any(arg_count, Volatility::Immutable),
            DataType::Boolean,
            Arc::new(|_args| Ok(ScalarValue::Boolean(Some(true)).into())),
        )
        .into()
    }

    #[test]
    fn predicate_intersects() {
        let storage_field = WKB_GEOMETRY.to_storage_field("", true).unwrap();
        let literal = Literal::new_with_metadata(
            create_scalar(Some("POINT (1 2)"), &WKB_GEOMETRY),
            Some(storage_field.metadata().into()),
        );
        let bounds = literal_bounds(&literal).unwrap();

        let stats_no_info = [GeoStatistics::unspecified()];
        let stats_intersecting = [
            GeoStatistics::unspecified().with_bbox(Some(BoundingBox::xy((0.5, 1.5), (1.5, 2.5))))
        ];
        let col0 = Column::new("col0", 0);

        assert!(SpatialFilter::Intersects(col0.clone(), bounds.clone()).evaluate(&stats_no_info));
        assert!(
            SpatialFilter::Intersects(col0.clone(), bounds.clone()).evaluate(&stats_intersecting)
        );

        let stats_empty_bbox = [GeoStatistics::unspecified()
            .with_bbox(Some(BoundingBox::xy(Interval::empty(), Interval::empty())))];

        assert!(
            !SpatialFilter::Intersects(col0.clone(), bounds.clone()).evaluate(&stats_empty_bbox)
        );

        let unrelated_literal = Literal::new(ScalarValue::Null);

        let err = literal_bounds(&unrelated_literal).unwrap_err();
        assert!(err
            .message()
            .contains("Unexpected scalar type in filter expression"));
    }

    #[test]
    fn predicate_covered_by() {
        let storage_field = WKB_GEOMETRY.to_storage_field("", true).unwrap();
        let literal = Literal::new_with_metadata(
            create_scalar(Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"), &WKB_GEOMETRY),
            Some(storage_field.metadata().into()),
        );
        let bounds = literal_bounds(&literal).unwrap();

        let stats_no_info = [GeoStatistics::unspecified()];
        let stats_covered = [
            GeoStatistics::unspecified().with_bbox(Some(BoundingBox::xy((1.0, 1.0), (2.0, 2.0))))
        ];
        let stats_not_covered = [
            GeoStatistics::unspecified().with_bbox(Some(BoundingBox::xy((3.0, 3.0), (5.0, 5.0))))
        ];
        let col0 = Column::new("col0", 0);

        // CoveredBy should return true when column bbox is fully contained in literal bounds
        assert!(SpatialFilter::CoveredBy(col0.clone(), bounds.clone()).evaluate(&stats_no_info));
        assert!(SpatialFilter::CoveredBy(col0.clone(), bounds.clone()).evaluate(&stats_covered));
        assert!(
            !SpatialFilter::CoveredBy(col0.clone(), bounds.clone()).evaluate(&stats_not_covered)
        );
    }

    #[test]
    fn predicate_has_z() {
        let col0 = Column::new("col0", 0);
        let has_z = SpatialFilter::HasZ(col0.clone());

        let stats_z_geometry_types = [GeoStatistics::unspecified()
            .try_with_str_geometry_types(Some(&["POINT Z"]))
            .unwrap()];
        let stats_z_bbox = [
            GeoStatistics::unspecified().with_bbox(Some(BoundingBox::xyzm(
                (0, 1),
                (2, 3),
                Some((4, 5).into()),
                None,
            ))),
        ];
        let stats_no_info = [GeoStatistics::unspecified()];

        assert!(has_z.evaluate(&stats_z_geometry_types));
        assert!(has_z.evaluate(&stats_z_bbox));
        assert!(has_z.evaluate(&stats_no_info));

        let stats_no_z_geometry_types = [GeoStatistics::unspecified()
            .try_with_str_geometry_types(Some(&["POINT"]))
            .unwrap()];
        let stats_no_z_bbox = [
            GeoStatistics::unspecified().with_bbox(Some(BoundingBox::xyzm(
                (0, 1),
                (2, 3),
                Some(Interval::empty()),
                None,
            ))),
        ];

        assert!(!has_z.evaluate(&stats_no_z_geometry_types));
        assert!(!has_z.evaluate(&stats_no_z_bbox));
    }

    #[test]
    fn predicate_other() {
        assert!(!SpatialFilter::LiteralFalse.evaluate(&[]));
        assert!(SpatialFilter::Unknown.evaluate(&[]));

        assert!(SpatialFilter::And(
            Box::new(SpatialFilter::Unknown),
            Box::new(SpatialFilter::Unknown)
        )
        .evaluate(&[]));

        assert!(!SpatialFilter::And(
            Box::new(SpatialFilter::Unknown),
            Box::new(SpatialFilter::LiteralFalse)
        )
        .evaluate(&[]));

        assert!(SpatialFilter::Or(
            Box::new(SpatialFilter::Unknown),
            Box::new(SpatialFilter::Unknown)
        )
        .evaluate(&[]));

        assert!(SpatialFilter::Or(
            Box::new(SpatialFilter::Unknown),
            Box::new(SpatialFilter::LiteralFalse)
        )
        .evaluate(&[]));

        assert!(!SpatialFilter::Or(
            Box::new(SpatialFilter::LiteralFalse),
            Box::new(SpatialFilter::LiteralFalse)
        )
        .evaluate(&[]));
    }

    #[test]
    fn predicate_from_expr_errors() {
        let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Null));
        let unrelated = dummy_unrelated();

        // Not a scalar function
        assert!(matches!(
            SpatialFilter::try_from_expr(&literal).unwrap(),
            SpatialFilter::Unknown
        ));

        // Not a predicate
        let expr_no_args: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "intersects",
            Arc::new(unrelated),
            vec![],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        assert!(matches!(
            SpatialFilter::try_from_expr(&expr_no_args).unwrap(),
            SpatialFilter::Unknown
        ));
    }

    #[rstest]
    fn predicate_from_expr_commutative_functions(
        #[values("st_intersects", "st_equals", "st_touches")] func_name: &str,
    ) {
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("geometry", 0));
        let storage_field = WKB_GEOMETRY.to_storage_field("", true).unwrap();
        let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new_with_metadata(
            create_scalar(Some("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"), &WKB_GEOMETRY),
            Some(storage_field.metadata().into()),
        ));

        // Test functions that should result in Intersects filter
        let func = create_dummy_spatial_function(func_name, 2);
        let expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            func_name,
            Arc::new(func.clone()),
            vec![column.clone(), literal.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate = SpatialFilter::try_from_expr(&expr).unwrap();
        assert!(
            matches!(predicate, SpatialFilter::Intersects(_, _)),
            "Function {func_name} should produce Intersects filter"
        );

        // Test reversed argument order
        let expr_reversed: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            func_name,
            Arc::new(func),
            vec![literal.clone(), column.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate_reversed = SpatialFilter::try_from_expr(&expr_reversed).unwrap();
        assert!(
            matches!(predicate_reversed, SpatialFilter::Intersects(_, _)),
            "Function {func_name} with reversed args should produce Intersects filter"
        );
    }

    #[rstest]
    fn predicate_from_expr_within_covered_by_functions(
        #[values("st_within", "st_covered_by", "st_coveredby")] func_name: &str,
    ) {
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("geometry", 0));
        let storage_field = WKB_GEOMETRY.to_storage_field("", true).unwrap();
        let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new_with_metadata(
            create_scalar(Some("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"), &WKB_GEOMETRY),
            Some(storage_field.metadata().into()),
        ));

        // Test functions that should result in CoveredBy filter when column is first arg
        let func = create_dummy_spatial_function(func_name, 2);
        let expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            func_name,
            Arc::new(func.clone()),
            vec![column.clone(), literal.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate = SpatialFilter::try_from_expr(&expr).unwrap();
        assert!(
            matches!(predicate, SpatialFilter::CoveredBy(_, _)),
            "Function {func_name} should produce CoveredBy filter"
        );

        // Test reversed argument order: should be converted to Intersects filter
        let expr_reversed: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            func_name,
            Arc::new(func),
            vec![literal.clone(), column.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate_reversed = SpatialFilter::try_from_expr(&expr_reversed).unwrap();
        assert!(
            matches!(predicate_reversed, SpatialFilter::Intersects(_, _)),
            "Function {func_name} with reversed args should produce Intersects filter"
        );
    }

    #[rstest]
    fn predicate_from_expr_contains_covers_functions(
        #[values("st_contains", "st_covers")] func_name: &str,
    ) {
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("geometry", 0));
        let storage_field = WKB_GEOMETRY.to_storage_field("", true).unwrap();
        let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new_with_metadata(
            create_scalar(Some("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"), &WKB_GEOMETRY),
            Some(storage_field.metadata().into()),
        ));

        // Test functions that should result in Intersects filter when column is first arg
        // (column contains/covers literal -> column must intersect literal)
        let func = create_dummy_spatial_function(func_name, 2);
        let expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            func_name,
            Arc::new(func.clone()),
            vec![column.clone(), literal.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate = SpatialFilter::try_from_expr(&expr).unwrap();
        assert!(
            matches!(predicate, SpatialFilter::Intersects(_, _)),
            "Function {func_name} should produce Intersects filter"
        );

        // Test reversed argument order: should be converted to CoveredBy filter
        // (literal contains/covers column -> equivalent to st_within(column, literal))
        let expr_reversed: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            func_name,
            Arc::new(func),
            vec![literal.clone(), column.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate_reversed = SpatialFilter::try_from_expr(&expr_reversed).unwrap();
        assert!(
            matches!(predicate_reversed, SpatialFilter::CoveredBy(_, _)),
            "Function {func_name} with reversed args should produce CoveredBy filter"
        );
    }

    #[test]
    fn predicate_from_expr_distance_functions() {
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("geometry", 0));
        let storage_field = WKB_GEOMETRY.to_storage_field("", true).unwrap();
        let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new_with_metadata(
            create_scalar(Some("POINT (1 2)"), &WKB_GEOMETRY),
            Some(storage_field.metadata().into()),
        ));
        let distance_literal: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Float64(Some(100.0))));

        // Test ST_DWithin function
        let st_dwithin = create_dummy_spatial_function("st_dwithin", 3);
        let dwithin_expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "st_dwithin",
            Arc::new(st_dwithin.clone()),
            vec![column.clone(), literal.clone(), distance_literal.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate = SpatialFilter::try_from_expr(&dwithin_expr).unwrap();
        assert!(
            matches!(predicate, SpatialFilter::Intersects(_, _)),
            "ST_DWithin should produce Intersects filter with expanded bounds"
        );

        // Test ST_DWithin with reversed geometry arguments
        let dwithin_expr_reversed: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "st_dwithin",
            Arc::new(st_dwithin),
            vec![literal.clone(), column.clone(), distance_literal.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate_reversed = SpatialFilter::try_from_expr(&dwithin_expr_reversed).unwrap();
        assert!(
            matches!(predicate_reversed, SpatialFilter::Intersects(_, _)),
            "ST_DWithin with reversed args should produce Intersects filter"
        );

        // Test ST_Distance <= threshold
        let st_distance = create_dummy_spatial_function("st_distance", 2);
        let distance_expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "st_distance",
            Arc::new(st_distance.clone()),
            vec![column.clone(), literal.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let comparison_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            distance_expr.clone(),
            Operator::LtEq,
            distance_literal.clone(),
        ));
        let predicate = SpatialFilter::try_from_expr(&comparison_expr).unwrap();
        assert!(
            matches!(predicate, SpatialFilter::Intersects(_, _)),
            "ST_Distance <= threshold should produce Intersects filter"
        );

        // Test threshold >= ST_Distance
        let comparison_expr_reversed: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            distance_literal.clone(),
            Operator::GtEq,
            distance_expr.clone(),
        ));
        let predicate_reversed = SpatialFilter::try_from_expr(&comparison_expr_reversed).unwrap();
        assert!(
            matches!(predicate_reversed, SpatialFilter::Intersects(_, _)),
            "threshold >= ST_Distance should produce Intersects filter"
        );

        // Test with negative distance (should be treated as Unknown)
        let negative_distance: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Float64(Some(-10.0))));
        let st_dwithin = create_dummy_spatial_function("st_dwithin", 3);
        let dwithin_expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "st_dwithin",
            Arc::new(st_dwithin.clone()),
            vec![column.clone(), literal.clone(), negative_distance],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate = SpatialFilter::try_from_expr(&dwithin_expr).unwrap();
        assert!(
            matches!(predicate, SpatialFilter::Unknown),
            "Negative distance should result in Unknown filter"
        );

        // Test with NaN distance (should be treated as Unknown)
        let nan_distance: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Float64(Some(f64::NAN))));
        let dwithin_expr_nan: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "st_dwithin",
            Arc::new(st_dwithin),
            vec![column.clone(), literal.clone(), nan_distance],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate_nan = SpatialFilter::try_from_expr(&dwithin_expr_nan).unwrap();
        assert!(
            matches!(predicate_nan, SpatialFilter::Unknown),
            "NaN distance should result in Unknown filter"
        );
    }

    #[rstest]
    fn predicate_from_spatial_relation_function_errors(
        #[values(
            "st_intersects",
            "st_equals",
            "st_touches",
            "st_contains",
            "st_covers",
            "st_within",
            "st_covered_by",
            "st_coveredby"
        )]
        func_name: &str,
    ) {
        let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Null));
        let st_intersects = create_dummy_spatial_function(func_name, 2);

        // Wrong number of args
        let expr_no_args: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "intersects",
            Arc::new(st_intersects.clone()),
            vec![],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        assert!(SpatialFilter::try_from_expr(&expr_no_args)
            .unwrap_err()
            .message()
            .contains("unexpected argument count"));

        // Unsupported arg types
        let expr_wrong_types: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "intersects",
            Arc::new(st_intersects.clone()),
            vec![literal.clone(), literal.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        assert!(matches!(
            SpatialFilter::try_from_expr(&expr_wrong_types).unwrap(),
            SpatialFilter::Unknown
        ));
    }

    #[test]
    fn predicate_from_expr_has_z() {
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("geometry", 0));
        let has_z = dummy_st_hasz();

        let expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "has_z",
            Arc::new(has_z.clone()),
            vec![column.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate = SpatialFilter::try_from_expr(&expr).unwrap();
        assert!(matches!(predicate, SpatialFilter::HasZ(_)));
    }

    #[test]
    fn predicate_from_has_z_errors() {
        let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Null));
        let has_z = dummy_st_hasz();

        let expr_no_args: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "has_z",
            Arc::new(has_z.clone()),
            vec![],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        assert!(SpatialFilter::try_from_expr(&expr_no_args)
            .unwrap_err()
            .message()
            .contains("unexpected argument count"));

        // Wrong arg types
        let expr_wrong_types: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "intersects",
            Arc::new(has_z.clone()),
            vec![literal.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        assert!(matches!(
            SpatialFilter::try_from_expr(&expr_wrong_types).unwrap(),
            SpatialFilter::Unknown
        ));
    }

    #[test]
    fn predicate_from_binary() {
        let literal_false: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(false))));
        let literal_true: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Boolean(Some(true))));
        let binary_and: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            literal_false.clone(),
            Operator::And,
            literal_true.clone(),
        ));
        let binary_or: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            literal_false.clone(),
            Operator::Or,
            literal_true.clone(),
        ));

        if let SpatialFilter::And(lhs, rhs) = SpatialFilter::try_from_expr(&binary_and).unwrap() {
            assert!(matches!(*lhs, SpatialFilter::LiteralFalse));
            assert!(matches!(*rhs, SpatialFilter::Unknown));
        } else {
            panic!("Parse incorrect!")
        }

        if let SpatialFilter::Or(lhs, rhs) = SpatialFilter::try_from_expr(&binary_or).unwrap() {
            assert!(matches!(*lhs, SpatialFilter::LiteralFalse));
            assert!(matches!(*rhs, SpatialFilter::Unknown));
        } else {
            panic!("Parse incorrect!")
        }
    }
}
