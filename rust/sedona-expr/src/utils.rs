use std::sync::Arc;

use datafusion_expr::Operator;
use datafusion_physical_expr::{expressions::BinaryExpr, PhysicalExpr, ScalarFunctionExpr};

pub struct ParsedDistancePredicate {
    pub arg0: Arc<dyn PhysicalExpr>,
    pub arg1: Arc<dyn PhysicalExpr>,
    pub arg_distance: Arc<dyn PhysicalExpr>,
}

pub fn parse_distance_predicate(expr: &Arc<dyn PhysicalExpr>) -> Option<ParsedDistancePredicate> {
    // There are 3 forms of distance predicates:
    // 1. st_dwithin(geom1, geom2, distance)
    // 2. st_distance(geom1, geom2) <= distance or st_distance(geom1, geom2) < distance
    // 3. distance >= st_distance(geom1, geom2) or distance > st_distance(geom1, geom2)
    if let Some(binary_expr) = expr.as_any().downcast_ref::<BinaryExpr>() {
        // handle case 2. and 3.
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
        // handle case 1.
        if st_dwithin_expr.fun().name() != "st_dwithin" {
            return None;
        }

        let args = st_dwithin_expr.args();
        assert!(args.len() >= 3);
        // Some((&args[0], &args[1], &args[2]))
        Some(ParsedDistancePredicate {
            arg0: Arc::clone(&args[0]),
            arg1: Arc::clone(&args[1]),
            arg_distance: Arc::clone(&args[2]),
        })
    } else {
        None
    }
}
