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

use datafusion_common::{JoinSide, Result};
use datafusion_physical_expr::projection::update_expr;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::projection::ProjectionExpr;
use sedona_common::sedona_internal_err;

pub trait SpatialPredicateTrait: Sized {
    fn swap_for_swapped_children(&self) -> Self;

    fn update_for_child_projections(
        &self,
        projected_left_exprs: &[ProjectionExpr],
        projected_right_exprs: &[ProjectionExpr],
    ) -> Result<Option<Self>>;
}

/// Spatial predicate is the join condition of a spatial join. It can be a distance predicate,
/// a relation predicate, or a KNN predicate.
#[derive(Debug, Clone)]
pub enum SpatialPredicate {
    Distance(DistancePredicate),
    Relation(RelationPredicate),
    KNearestNeighbors(KNNPredicate),
}

impl SpatialPredicate {
    /// Returns a semantically equivalent predicate for the case where the join children are swapped.
    ///
    /// This is used by `SpatialJoinExec::swap_inputs`.
    pub fn swap_for_swapped_children(&self) -> Self {
        SpatialPredicateTrait::swap_for_swapped_children(self)
    }

    pub fn update_for_child_projections(
        &self,
        projected_left_exprs: &[ProjectionExpr],
        projected_right_exprs: &[ProjectionExpr],
    ) -> Result<Option<Self>> {
        SpatialPredicateTrait::update_for_child_projections(
            self,
            projected_left_exprs,
            projected_right_exprs,
        )
    }
}

impl SpatialPredicateTrait for SpatialPredicate {
    fn swap_for_swapped_children(&self) -> Self {
        match self {
            SpatialPredicate::Relation(pred) => {
                SpatialPredicate::Relation(pred.swap_for_swapped_children())
            }
            SpatialPredicate::Distance(pred) => {
                SpatialPredicate::Distance(pred.swap_for_swapped_children())
            }
            SpatialPredicate::KNearestNeighbors(pred) => {
                SpatialPredicate::KNearestNeighbors(pred.swap_for_swapped_children())
            }
        }
    }

    fn update_for_child_projections(
        &self,
        projected_left_exprs: &[ProjectionExpr],
        projected_right_exprs: &[ProjectionExpr],
    ) -> Result<Option<Self>> {
        match self {
            SpatialPredicate::Relation(pred) => Ok(pred
                .update_for_child_projections(projected_left_exprs, projected_right_exprs)?
                .map(SpatialPredicate::Relation)),
            SpatialPredicate::Distance(pred) => Ok(pred
                .update_for_child_projections(projected_left_exprs, projected_right_exprs)?
                .map(SpatialPredicate::Distance)),
            SpatialPredicate::KNearestNeighbors(pred) => Ok(pred
                .update_for_child_projections(projected_left_exprs, projected_right_exprs)?
                .map(SpatialPredicate::KNearestNeighbors)),
        }
    }
}

impl SpatialPredicateTrait for RelationPredicate {
    fn swap_for_swapped_children(&self) -> Self {
        Self {
            left: Arc::clone(&self.right),
            right: Arc::clone(&self.left),
            relation_type: self.relation_type.invert(),
        }
    }

    fn update_for_child_projections(
        &self,
        projected_left_exprs: &[ProjectionExpr],
        projected_right_exprs: &[ProjectionExpr],
    ) -> Result<Option<Self>> {
        let Some(left) = update_expr(&self.left, projected_left_exprs, false)? else {
            return Ok(None);
        };
        let Some(right) = update_expr(&self.right, projected_right_exprs, false)? else {
            return Ok(None);
        };

        Ok(Some(Self {
            left,
            right,
            relation_type: self.relation_type,
        }))
    }
}

impl SpatialPredicateTrait for DistancePredicate {
    fn swap_for_swapped_children(&self) -> Self {
        Self {
            left: Arc::clone(&self.right),
            right: Arc::clone(&self.left),
            distance: Arc::clone(&self.distance),
            distance_side: self.distance_side.negate(),
        }
    }

    fn update_for_child_projections(
        &self,
        projected_left_exprs: &[ProjectionExpr],
        projected_right_exprs: &[ProjectionExpr],
    ) -> Result<Option<Self>> {
        let Some(left) = update_expr(&self.left, projected_left_exprs, false)? else {
            return Ok(None);
        };
        let Some(right) = update_expr(&self.right, projected_right_exprs, false)? else {
            return Ok(None);
        };

        let distance = match self.distance_side {
            JoinSide::Left => {
                let Some(distance) = update_expr(&self.distance, projected_left_exprs, false)?
                else {
                    return Ok(None);
                };
                distance
            }
            JoinSide::Right => {
                let Some(distance) = update_expr(&self.distance, projected_right_exprs, false)?
                else {
                    return Ok(None);
                };
                distance
            }
            JoinSide::None => Arc::clone(&self.distance),
        };

        Ok(Some(Self {
            left,
            right,
            distance,
            distance_side: self.distance_side,
        }))
    }
}

impl SpatialPredicateTrait for KNNPredicate {
    fn swap_for_swapped_children(&self) -> Self {
        Self {
            // Keep query/object expressions stable; only flip which child is considered probe.
            left: Arc::clone(&self.left),
            right: Arc::clone(&self.right),
            k: self.k,
            use_spheroid: self.use_spheroid,
            probe_side: self.probe_side.negate(),
        }
    }

    fn update_for_child_projections(
        &self,
        projected_left_exprs: &[ProjectionExpr],
        projected_right_exprs: &[ProjectionExpr],
    ) -> Result<Option<Self>> {
        let (probe_exprs, build_exprs) = match self.probe_side {
            JoinSide::Left => (projected_left_exprs, projected_right_exprs),
            JoinSide::Right => (projected_right_exprs, projected_left_exprs),
            JoinSide::None => {
                return sedona_internal_err!("KNN join requires explicit probe_side designation")
            }
        };

        let Some(left) = update_expr(&self.left, probe_exprs, false)? else {
            return Ok(None);
        };
        let Some(right) = update_expr(&self.right, build_exprs, false)? else {
            return Ok(None);
        };

        Ok(Some(Self {
            left,
            right,
            k: self.k,
            use_spheroid: self.use_spheroid,
            probe_side: self.probe_side,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion_common::ScalarValue;
    use datafusion_physical_expr::expressions::{Column, Literal};

    fn proj_col(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new(name, index))
    }

    fn proj_expr(expr: Arc<dyn PhysicalExpr>, alias: &str) -> ProjectionExpr {
        ProjectionExpr {
            expr,
            alias: alias.to_string(),
        }
    }

    fn assert_is_column(expr: &Arc<dyn PhysicalExpr>, name: &str, index: usize) {
        let col = expr
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column");
        assert_eq!(col.name(), name);
        assert_eq!(col.index(), index);
    }

    #[test]
    fn relation_rewrite_success() -> Result<()> {
        let on = SpatialPredicate::Relation(RelationPredicate {
            left: proj_col("a", 0),
            right: proj_col("x", 0),
            relation_type: SpatialRelationType::Intersects,
        });

        let projected_left_exprs = vec![proj_expr(proj_col("a", 0), "a_new")];
        let projected_right_exprs = vec![proj_expr(proj_col("x", 0), "x_new")];

        let Some(updated) =
            on.update_for_child_projections(&projected_left_exprs, &projected_right_exprs)?
        else {
            unreachable!("expected Some")
        };

        let SpatialPredicate::Relation(updated) = updated else {
            unreachable!("expected relation")
        };
        assert_is_column(&updated.left, "a_new", 0);
        assert_is_column(&updated.right, "x_new", 0);
        Ok(())
    }

    #[test]
    fn relation_rewrite_none_when_missing() -> Result<()> {
        let on = SpatialPredicate::Relation(RelationPredicate {
            left: proj_col("a", 1),
            right: proj_col("x", 0),
            relation_type: SpatialRelationType::Intersects,
        });

        let projected_left_exprs = vec![proj_expr(proj_col("a", 0), "a0")];
        let projected_right_exprs = vec![proj_expr(proj_col("x", 0), "x0")];

        assert!(on
            .update_for_child_projections(&projected_left_exprs, &projected_right_exprs)?
            .is_none());
        Ok(())
    }

    #[test]
    fn distance_rewrite_distance_side_left() -> Result<()> {
        let on = SpatialPredicate::Distance(DistancePredicate {
            left: proj_col("geom", 0),
            right: proj_col("geom", 0),
            distance: proj_col("dist", 1),
            distance_side: JoinSide::Left,
        });

        let projected_left_exprs = vec![
            proj_expr(proj_col("geom", 0), "geom_out"),
            proj_expr(proj_col("dist", 1), "dist_out"),
        ];
        let projected_right_exprs = vec![proj_expr(proj_col("geom", 0), "geom_r")];

        let Some(updated) =
            on.update_for_child_projections(&projected_left_exprs, &projected_right_exprs)?
        else {
            unreachable!("expected Some")
        };

        let SpatialPredicate::Distance(updated) = updated else {
            unreachable!("expected distance")
        };
        assert_is_column(&updated.left, "geom_out", 0);
        assert_is_column(&updated.right, "geom_r", 0);
        assert_is_column(&updated.distance, "dist_out", 1);
        assert_eq!(updated.distance_side, JoinSide::Left);
        Ok(())
    }

    #[test]
    fn distance_rewrite_distance_side_none_keeps_literal() -> Result<()> {
        let distance_lit: Arc<dyn PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Float64(Some(1.0))));

        let on = SpatialPredicate::Distance(DistancePredicate {
            left: proj_col("geom", 0),
            right: proj_col("geom", 0),
            distance: Arc::clone(&distance_lit),
            distance_side: JoinSide::None,
        });

        let projected_left_exprs = vec![proj_expr(proj_col("geom", 0), "geom_out")];
        let projected_right_exprs = vec![proj_expr(proj_col("geom", 0), "geom_r")];

        let Some(updated) =
            on.update_for_child_projections(&projected_left_exprs, &projected_right_exprs)?
        else {
            unreachable!("expected Some")
        };

        let SpatialPredicate::Distance(updated) = updated else {
            unreachable!("expected distance")
        };
        assert_is_column(&updated.left, "geom_out", 0);
        assert_is_column(&updated.right, "geom_r", 0);
        assert!(Arc::ptr_eq(&updated.distance, &distance_lit));
        assert_eq!(updated.distance_side, JoinSide::None);
        Ok(())
    }

    #[test]
    fn knn_rewrite_success_probe_left_and_right() -> Result<()> {
        let base = SpatialPredicate::KNearestNeighbors(KNNPredicate {
            left: proj_col("probe", 0),
            right: proj_col("build", 0),
            k: 10,
            use_spheroid: false,
            probe_side: JoinSide::Left,
        });

        let left_exprs = vec![proj_expr(proj_col("probe", 0), "probe_out")];
        let right_exprs = vec![proj_expr(proj_col("build", 0), "build_out")];

        let Some(updated) = base.update_for_child_projections(&left_exprs, &right_exprs)? else {
            unreachable!("expected Some")
        };
        let SpatialPredicate::KNearestNeighbors(updated) = updated else {
            unreachable!("expected knn")
        };
        assert_is_column(&updated.left, "probe_out", 0);
        assert_is_column(&updated.right, "build_out", 0);
        assert_eq!(updated.probe_side, JoinSide::Left);

        let base = SpatialPredicate::KNearestNeighbors(KNNPredicate {
            left: proj_col("probe", 0),
            right: proj_col("build", 0),
            k: 10,
            use_spheroid: false,
            probe_side: JoinSide::Right,
        });

        // For probe_side=Right: predicate.left (probe) is rewritten using right projections,
        // and predicate.right (build) is rewritten using left projections.
        let left_exprs = vec![proj_expr(proj_col("build", 0), "build_out_l")];
        let right_exprs = vec![proj_expr(proj_col("probe", 0), "probe_out_r")];

        let Some(updated) = base.update_for_child_projections(&left_exprs, &right_exprs)? else {
            unreachable!("expected Some")
        };
        let SpatialPredicate::KNearestNeighbors(updated) = updated else {
            unreachable!("expected knn")
        };
        assert_is_column(&updated.left, "probe_out_r", 0);
        assert_is_column(&updated.right, "build_out_l", 0);
        assert_eq!(updated.probe_side, JoinSide::Right);

        Ok(())
    }

    #[test]
    fn knn_rewrite_errors_on_none_probe_side() {
        let on = SpatialPredicate::KNearestNeighbors(KNNPredicate {
            left: proj_col("probe", 0),
            right: proj_col("build", 0),
            k: 10,
            use_spheroid: false,
            probe_side: JoinSide::None,
        });

        let left_exprs = vec![proj_expr(proj_col("probe", 0), "probe_out")];
        let right_exprs = vec![proj_expr(proj_col("build", 0), "build_out")];

        let err = on
            .update_for_child_projections(&left_exprs, &right_exprs)
            .expect_err("expected error");
        let msg = err.to_string();
        assert!(msg.contains("KNN join requires explicit probe_side designation"));
    }
}

impl std::fmt::Display for SpatialPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpatialPredicate::Distance(predicate) => write!(f, "{predicate}"),
            SpatialPredicate::Relation(predicate) => write!(f, "{predicate}"),
            SpatialPredicate::KNearestNeighbors(predicate) => write!(f, "{predicate}"),
        }
    }
}

/// Distance-based spatial join predicate.
///
/// This predicate represents a spatial join condition based on distance between geometries.
/// It is used to find pairs of geometries from left and right tables where the distance
/// between them is less than a specified threshold.
///
/// # Example SQL
/// ```sql
/// SELECT * FROM left_table l JOIN right_table r
/// ON ST_Distance(l.geom, r.geom) < 100.0
/// ```
///
/// # Fields
/// * `left` - Expression to evaluate the left side geometry
/// * `right` - Expression to evaluate the right side geometry
/// * `distance` - Expression to evaluate the distance threshold
/// * `distance_side` - Which side the distance expression belongs to (for column references)
#[derive(Debug, Clone)]
pub struct DistancePredicate {
    /// The expression for evaluating the geometry value on the left side. The expression
    /// should be evaluated directly on the left side batches.
    pub left: Arc<dyn PhysicalExpr>,
    /// The expression for evaluating the geometry value on the right side. The expression
    /// should be evaluated directly on the right side batches.
    pub right: Arc<dyn PhysicalExpr>,
    /// The expression for evaluating the distance value. The expression
    /// should be evaluated directly on the left or right side batches according to distance_side.
    pub distance: Arc<dyn PhysicalExpr>,
    /// The side of the distance expression. It could be JoinSide::None if the distance expression
    /// is not a column reference. The most common case is that the distance expression is a
    /// literal value.
    pub distance_side: JoinSide,
}

impl DistancePredicate {
    /// Creates a new distance predicate.
    ///
    /// # Arguments
    /// * `left` - Expression for the left side geometry
    /// * `right` - Expression for the right side geometry
    /// * `distance` - Expression for the distance threshold
    /// * `distance_side` - Which side (Left, Right, or None) the distance expression belongs to
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        right: Arc<dyn PhysicalExpr>,
        distance: Arc<dyn PhysicalExpr>,
        distance_side: JoinSide,
    ) -> Self {
        Self {
            left,
            right,
            distance,
            distance_side,
        }
    }
}

impl std::fmt::Display for DistancePredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ST_Distance({}, {}) < {}",
            self.left, self.right, self.distance
        )
    }
}

/// Spatial relation predicate for topological relationships.
///
/// This predicate represents a spatial join condition based on topological relationships
/// between geometries, such as intersects, contains, within, etc. It follows the
/// DE-9IM (Dimensionally Extended 9-Intersection Model) spatial relations.
///
/// # Example SQL
/// ```sql
/// SELECT * FROM buildings b JOIN parcels p
/// ON ST_Intersects(b.geometry, p.geometry)
/// ```
///
/// # Supported Relations
/// * `Intersects` - Geometries share at least one point
/// * `Contains` - Left geometry contains the right geometry
/// * `Within` - Left geometry is within the right geometry
/// * `Covers` - Left geometry covers the right geometry
/// * `CoveredBy` - Left geometry is covered by the right geometry
/// * `Touches` - Geometries touch at their boundaries
/// * `Crosses` - Geometries cross each other
/// * `Overlaps` - Geometries overlap
/// * `Equals` - Geometries are spatially equal
#[derive(Debug, Clone)]
pub struct RelationPredicate {
    /// The expression for evaluating the geometry value on the left side. The expression
    /// should be evaluated directly on the left side batches.
    pub left: Arc<dyn PhysicalExpr>,
    /// The expression for evaluating the geometry value on the right side. The expression
    /// should be evaluated directly on the right side batches.
    pub right: Arc<dyn PhysicalExpr>,
    /// The spatial relation type.
    pub relation_type: SpatialRelationType,
}

impl RelationPredicate {
    /// Creates a new spatial relation predicate.
    ///
    /// # Arguments
    /// * `left` - Expression for the left side geometry
    /// * `right` - Expression for the right side geometry
    /// * `relation_type` - The type of spatial relationship to test
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        right: Arc<dyn PhysicalExpr>,
        relation_type: SpatialRelationType,
    ) -> Self {
        Self {
            left,
            right,
            relation_type,
        }
    }
}

impl std::fmt::Display for RelationPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ST_{}({}, {})",
            self.relation_type, self.left, self.right
        )
    }
}

/// Type of spatial relation predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialRelationType {
    Intersects,
    Contains,
    Within,
    Covers,
    CoveredBy,
    Touches,
    Crosses,
    Overlaps,
    Equals,
}

impl SpatialRelationType {
    /// Converts a function name string to a SpatialRelationType.
    ///
    /// # Arguments
    /// * `name` - The spatial function name (e.g., "st_intersects", "st_contains")
    ///
    /// # Returns
    /// * `Some(SpatialRelationType)` if the name is recognized
    /// * `None` if the name is not a valid spatial relation function
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "st_intersects" => Some(SpatialRelationType::Intersects),
            "st_contains" => Some(SpatialRelationType::Contains),
            "st_within" => Some(SpatialRelationType::Within),
            "st_covers" => Some(SpatialRelationType::Covers),
            "st_coveredby" | "st_covered_by" => Some(SpatialRelationType::CoveredBy),
            "st_touches" => Some(SpatialRelationType::Touches),
            "st_crosses" => Some(SpatialRelationType::Crosses),
            "st_overlaps" => Some(SpatialRelationType::Overlaps),
            "st_equals" => Some(SpatialRelationType::Equals),
            _ => None,
        }
    }

    /// Returns the inverse spatial relation.
    ///
    /// Some spatial relations have natural inverses (e.g., Contains/Within),
    /// while others are symmetric (e.g., Intersects, Touches, Equals).
    ///
    /// # Returns
    /// The inverted spatial relation type
    pub fn invert(&self) -> Self {
        match self {
            SpatialRelationType::Intersects => SpatialRelationType::Intersects,
            SpatialRelationType::Covers => SpatialRelationType::CoveredBy,
            SpatialRelationType::CoveredBy => SpatialRelationType::Covers,
            SpatialRelationType::Contains => SpatialRelationType::Within,
            SpatialRelationType::Within => SpatialRelationType::Contains,
            SpatialRelationType::Touches => SpatialRelationType::Touches,
            SpatialRelationType::Crosses => SpatialRelationType::Crosses,
            SpatialRelationType::Overlaps => SpatialRelationType::Overlaps,
            SpatialRelationType::Equals => SpatialRelationType::Equals,
        }
    }
}

impl std::fmt::Display for SpatialRelationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpatialRelationType::Intersects => write!(f, "intersects"),
            SpatialRelationType::Contains => write!(f, "contains"),
            SpatialRelationType::Within => write!(f, "within"),
            SpatialRelationType::Covers => write!(f, "covers"),
            SpatialRelationType::CoveredBy => write!(f, "coveredby"),
            SpatialRelationType::Touches => write!(f, "touches"),
            SpatialRelationType::Crosses => write!(f, "crosses"),
            SpatialRelationType::Overlaps => write!(f, "overlaps"),
            SpatialRelationType::Equals => write!(f, "equals"),
        }
    }
}

/// K-Nearest Neighbors (KNN) spatial join predicate.
///
/// This predicate represents a spatial join that finds the k nearest neighbors
/// from the right side (object) table for each geometry in the left side (query) table.
/// It's commonly used for proximity analysis and spatial recommendations.
///
/// # Example SQL
/// ```sql
/// SELECT * FROM restaurants r
/// JOIN TABLE(ST_KNN(r.location, h.location, 5, false)) AS knn
/// ON r.id = knn.restaurant_id
/// ```
///
/// # Algorithm
/// For each geometry in the left (query) side:
/// 1. Find the k nearest geometries from the right (object) side
/// 2. Use spatial index for efficient nearest neighbor search
/// 3. Handle tie-breaking when multiple geometries have the same distance
///
/// # Performance Considerations
/// * Uses R-tree spatial index for efficient search
/// * Performance depends on k value and spatial distribution
/// * Tie-breaking may require additional distance calculations
///
/// # Limitations
/// * Currently only supports planar (Euclidean) distance calculations
/// * Spheroid distance (use_spheroid=true) is not yet implemented
#[derive(Debug, Clone)]
pub struct KNNPredicate {
    /// The expression for evaluating the geometry value on the left side (queries side).
    /// The expression should be evaluated directly on the left side batches.
    pub left: Arc<dyn PhysicalExpr>,
    /// The expression for evaluating the geometry value on the right side (object side).
    /// The expression should be evaluated directly on the right side batches.
    pub right: Arc<dyn PhysicalExpr>,
    /// The number of nearest neighbors to find (literal value).
    pub k: u32,
    /// Whether to use spheroid distance calculation or planar distance (literal value).
    /// Currently must be false as spheroid distance is not yet implemented.
    pub use_spheroid: bool,
    /// Which execution plan side (Left or Right) the probe expression belongs to.
    /// This is used to correctly assign build/probe plans in execution.
    pub probe_side: JoinSide,
}

impl KNNPredicate {
    /// Creates a new K-Nearest Neighbors predicate.
    ///
    /// # Arguments
    /// * `left` - Expression for the left side (query) geometry
    /// * `right` - Expression for the right side (object) geometry
    /// * `k` - Number of nearest neighbors to find (literal value)
    /// * `use_spheroid` - Whether to use spheroid distance (literal value, currently must be false)
    /// * `probe_side` - Which execution plan side the probe expression belongs to, cannot be None
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        right: Arc<dyn PhysicalExpr>,
        k: u32,
        use_spheroid: bool,
        probe_side: JoinSide,
    ) -> Self {
        assert!(matches!(probe_side, JoinSide::Left | JoinSide::Right));
        Self {
            left,
            right,
            k,
            use_spheroid,
            probe_side,
        }
    }
}

impl std::fmt::Display for KNNPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ST_KNN({}, {}, {}, {})",
            self.left, self.right, self.k, self.use_spheroid
        )
    }
}
