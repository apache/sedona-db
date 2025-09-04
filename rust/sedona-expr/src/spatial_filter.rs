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

use arrow_schema::Schema;
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

use crate::statistics::GeoStatistics;

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
        if let Some(scalar_fun) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
            let raw_args = scalar_fun.args();
            let args = parse_args(raw_args);
            match scalar_fun.fun().name() {
                "st_intersects" => {
                    if args.len() != 2 {
                        return sedona_internal_err!(
                            "unexpected argument count in filter evaluation"
                        );
                    }

                    match (&args[0], &args[1]) {
                        (ArgRef::Col(column), ArgRef::Lit(literal))
                        | (ArgRef::Lit(literal), ArgRef::Col(column)) => {
                            match literal_bounds(literal) {
                                Ok(literal_bounds) => {
                                    Ok(Self::Intersects(column.clone(), literal_bounds))
                                }
                                Err(e) => Err(DataFusionError::External(Box::new(e))),
                            }
                        }
                        // Not between a literal and a column
                        _ => Ok(Self::Unknown),
                    }
                }
                "st_hasz" => {
                    if args.len() != 1 {
                        return sedona_internal_err!(
                            "unexpected argument count in filter evaluation"
                        );
                    }

                    match &args[0] {
                        ArgRef::Col(column) => Ok(Self::HasZ(column.clone())),
                        _ => Ok(Self::Unknown),
                    }
                }
                // Not a function we know about
                _ => Ok(Self::Unknown),
            }
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
    use sedona_geometry::{bounding_box::BoundingBox, interval::Interval};
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_scalar;

    use super::*;

    fn dummy_st_intersects() -> ScalarUDF {
        SimpleScalarUDF::new_with_signature(
            "st_intersects",
            Signature::any(2, Volatility::Immutable),
            DataType::Boolean,
            Arc::new(|_args| Ok(ScalarValue::Boolean(Some(true)).into())),
        )
        .into()
    }

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

    #[test]
    fn spatial_filters() {}

    #[test]
    fn predicate_intersects() {
        let literal = Literal::new(create_scalar(Some("POINT (1 2)"), &WKB_GEOMETRY));
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

    #[test]
    fn predicate_from_expr_intersects() {
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("geometry", 0));
        let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(create_scalar(
            Some("POINT (1 2)"),
            &WKB_GEOMETRY,
        )));
        let st_intersects = dummy_st_intersects();

        let expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "intersects",
            Arc::new(st_intersects.clone()),
            vec![column.clone(), literal.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate = SpatialFilter::try_from_expr(&expr).unwrap();
        assert!(matches!(predicate, SpatialFilter::Intersects(_, _)));

        let expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "intersects",
            Arc::new(st_intersects.clone()),
            vec![literal.clone(), column.clone()],
            Arc::new(Field::new("", DataType::Boolean, true)),
        ));
        let predicate = SpatialFilter::try_from_expr(&expr).unwrap();
        assert!(matches!(predicate, SpatialFilter::Intersects(_, _)))
    }

    #[test]
    fn predicate_from_intersects_errors() {
        let literal: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Null));
        let st_intersects = dummy_st_intersects();

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
