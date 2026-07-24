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
use datafusion_common::Result;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::ExecutionPlan;
use sedona_common::SedonaOptions;
use sedona_geometry::transform::CrsEngine;
use sedona_proj::transform::LazyProjEngine;
use sedona_query_planner::{
    spatial_join_physical_planner::{PlanSpatialJoinArgs, SpatialJoinPhysicalPlanner},
    spatial_predicate::{RelationPredicate, SpatialPredicate, SpatialRelationType},
};
use sedona_schema::{crs::Crs, datatypes::SedonaType, matchers::ArgMatcher};
use sedona_spatial_join::{physical_planner::repartition_probe_side, SpatialJoinExec};

use crate::join_provider::RasterJoinProvider;

/// [`SpatialJoinPhysicalPlanner`] implementation for raster/geometry spatial joins.
///
/// Handles `RS_Intersects`/`RS_Contains`/`RS_Within` predicates where exactly one
/// operand is a raster and the other is a planar geometry, producing an optimized
/// [`SpatialJoinExec`] backed by a [`RasterJoinProvider`]. The raster operand is
/// always placed on the probe side (see [`plan_spatial_join`]). All other
/// predicates (including raster/raster, for which there is no fixed common CRS to
/// compare in) are declined with `None` so they fall back to the nested-loop join.
///
/// [`plan_spatial_join`]: SpatialJoinPhysicalPlanner::plan_spatial_join
#[derive(Debug)]
pub struct RasterSpatialJoinPhysicalPlanner;

impl RasterSpatialJoinPhysicalPlanner {
    /// Create a new raster join planner.
    pub fn new() -> Self {
        Self
    }
}

impl Default for RasterSpatialJoinPhysicalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialJoinPhysicalPlanner for RasterSpatialJoinPhysicalPlanner {
    fn plan_spatial_join(
        &self,
        args: &PlanSpatialJoinArgs<'_>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(target_crs) = raster_geometry_target_crs(
            args.spatial_predicate,
            &args.physical_left.schema(),
            &args.physical_right.schema(),
        )?
        else {
            return Ok(None);
        };

        // Pin the raster operand to the probe (right) side, overriding the row-count
        // reordering heuristic (`should_swap_join_order`, which is deliberately not
        // consulted here).
        //
        // The build side is fully buffered and its ingest compacts view arrays
        // (`gc_view_arrays`), copying raster `BinaryView` band payloads into fresh
        // buffers and defeating the zero-copy design. The probe side streams one
        // batch at a time and does not compact, and output columns are assembled
        // with `arrow::compute::take`, which shares view-array data buffers. Placing
        // the raster on the probe side therefore keeps raster band buffers uncopied,
        // short-lived, and shared across the rows each raster fans out to. The
        // cardinality heuristic would put a low-cardinality raster input on the build
        // side, which is the worst case here, so it is bypassed for raster joins.
        //
        // The raster operand is pinned to the probe (streamed) side so its band
        // buffers are not fully buffered on the build side (where ingest gc-compacts
        // and copies view buffers) while passing through the join. Whether to pin
        // unconditionally, or only when the vector side is not the vastly larger
        // input (forcing a huge vector table onto the buffered build side can cost
        // more than copying a few raster payloads) — and whether to instead make
        // build-side ingest share view buffers so the statistics-based side
        // heuristic can stay intact — is an open question. See
        // https://github.com/apache/sedona-db/issues/1078.
        let raster_on_left =
            raster_operand_on_left(args.spatial_predicate, &args.physical_left.schema())?;
        let swap_raster_to_probe = raster_on_left && args.join_type.supports_swap();

        // Repartition the probe side when enabled, mirroring the default planner.
        // The raster operand ends up on the probe side, so `swap_raster_to_probe`
        // doubles as the swap decision `repartition_probe_side` uses to target the
        // pre-swap input that becomes the probe.
        let (physical_left, physical_right) = if args.join_options.repartition_probe_side {
            repartition_probe_side(
                args.physical_left.clone(),
                args.physical_right.clone(),
                args.spatial_predicate,
                swap_raster_to_probe,
            )?
        } else {
            (args.physical_left.clone(), args.physical_right.clone())
        };

        // Capture the session's CRS engine at plan time so raster footprint
        // reprojection honors a user-injected engine. Absent a SedonaOptions
        // extension, fall back to the process-global PROJ engine via
        // LazyProjEngine (the same default the session registers).
        let engine: Arc<dyn CrsEngine + Send + Sync> = args
            .options
            .extensions
            .get::<SedonaOptions>()
            .map(|opts| opts.runtime.crs_engine().clone())
            .unwrap_or_else(|| Arc::new(LazyProjEngine));

        let exec = SpatialJoinExec::try_new(
            physical_left,
            physical_right,
            args.spatial_predicate.clone(),
            args.remainder.cloned(),
            args.join_type,
            None,
            args.join_options,
        )?
        .with_spatial_join_provider(Arc::new(RasterJoinProvider::new(target_crs, engine)));

        if swap_raster_to_probe {
            // Move the raster from the build side onto the probe side.
            // `swap_inputs()` inverts the predicate (Contains<->Within), swaps the
            // join type (Left<->Right), reorders the output columns, and carries the
            // raster provider onto the rebuilt exec, so the raster-aware evaluator
            // survives the swap and lands on the final plan.
            exec.swap_inputs().map(Some)
        } else {
            Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
        }
    }
}

/// Return the target CRS (the geometry operand's CRS) when `spatial_predicate`
/// is a raster/geometry relation this planner can accelerate, or `None` when it
/// should fall through to another planner.
///
/// Handled: a `Relation` predicate of `Intersects`/`Contains`/`Within` with
/// exactly one raster operand and one planar-geometry operand. The returned
/// [`Crs`] is the geometry operand's schema CRS, which is `None` when the
/// geometry carries no CRS.
fn raster_geometry_target_crs(
    spatial_predicate: &SpatialPredicate,
    left_schema: &Schema,
    right_schema: &Schema,
) -> Result<Option<Crs>> {
    let SpatialPredicate::Relation(RelationPredicate {
        left,
        right,
        relation_type,
    }) = spatial_predicate
    else {
        return Ok(None);
    };

    if !matches!(
        relation_type,
        SpatialRelationType::Intersects
            | SpatialRelationType::Contains
            | SpatialRelationType::Within
    ) {
        return Ok(None);
    }

    let left_type = operand_sedona_type(left, left_schema)?;
    let right_type = operand_sedona_type(right, right_schema)?;

    let is_geometry = ArgMatcher::is_geometry();
    let target = match (
        matches!(left_type, SedonaType::Raster),
        matches!(right_type, SedonaType::Raster),
    ) {
        // Exactly one raster operand; the other must be a planar geometry.
        (true, false) if is_geometry.match_type(&right_type) => Some(right_type.crs().clone()),
        (false, true) if is_geometry.match_type(&left_type) => Some(left_type.crs().clone()),
        // raster/raster (no fixed common CRS), geometry/geometry, geography, or
        // item-crs operands are left to other planners / the nested-loop fallback.
        _ => None,
    };

    Ok(target)
}

/// Return `true` when the raster operand of `spatial_predicate` is the left
/// (build) input, resolving the left operand against `left_schema`.
///
/// This is only meaningful for a raster/geometry `Relation` predicate that
/// [`raster_geometry_target_crs`] has already accepted (exactly one operand is a
/// raster). Any other predicate shape reports `false`.
fn raster_operand_on_left(
    spatial_predicate: &SpatialPredicate,
    left_schema: &Schema,
) -> Result<bool> {
    let SpatialPredicate::Relation(RelationPredicate { left, .. }) = spatial_predicate else {
        return Ok(false);
    };
    Ok(matches!(
        operand_sedona_type(left, left_schema)?,
        SedonaType::Raster
    ))
}

/// Resolve the [`SedonaType`] an operand expression evaluates to against `schema`.
fn operand_sedona_type(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Result<SedonaType> {
    let return_field = expr.return_field(schema)?;
    SedonaType::from_storage_field(&return_field)
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_schema::Schema;
    use datafusion_physical_expr::{expressions::Column, PhysicalExpr};
    use sedona_geometry::types::Edges;
    use sedona_query_planner::spatial_predicate::{
        RelationPredicate, SpatialPredicate, SpatialRelationType,
    };
    use sedona_schema::crs::{deserialize_crs, lnglat};
    use sedona_schema::datatypes::{SedonaType, RASTER, WKB_GEOMETRY};

    use super::{raster_geometry_target_crs, raster_operand_on_left};

    fn schema_with(field_name: &str, sedona_type: &SedonaType) -> Arc<Schema> {
        Arc::new(Schema::new(vec![sedona_type
            .to_storage_field(field_name, true)
            .unwrap()]))
    }

    fn col(name: &str) -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new(name, 0))
    }

    fn relation(
        left: Arc<dyn PhysicalExpr>,
        right: Arc<dyn PhysicalExpr>,
        relation_type: SpatialRelationType,
    ) -> SpatialPredicate {
        SpatialPredicate::Relation(RelationPredicate::new(left, right, relation_type))
    }

    #[test]
    fn raster_geometry_resolves_geometry_crs_as_target() {
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let raster_schema = schema_with("raster", &RASTER);
        let geom_schema = schema_with("geom", &geom_type);

        // (raster, geometry): target is the geometry's CRS.
        let pred = relation(col("raster"), col("geom"), SpatialRelationType::Intersects);
        let target = raster_geometry_target_crs(&pred, &raster_schema, &geom_schema)
            .unwrap()
            .expect("raster/geometry should be handled");
        assert!(target
            .as_deref()
            .unwrap()
            .crs_equals(lnglat().as_deref().unwrap()));

        // (geometry, raster): still resolves to the geometry's CRS.
        let pred = relation(col("geom"), col("raster"), SpatialRelationType::Contains);
        let target = raster_geometry_target_crs(&pred, &geom_schema, &raster_schema)
            .unwrap()
            .expect("geometry/raster should be handled");
        assert!(target
            .as_deref()
            .unwrap()
            .crs_equals(lnglat().as_deref().unwrap()));
    }

    #[test]
    fn raster_operand_side_is_detected() {
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let raster_schema = schema_with("raster", &RASTER);
        let geom_schema = schema_with("geom", &geom_type);

        // (raster, geometry): the left operand resolves against the left (raster)
        // schema, so the raster is on the build side.
        let pred = relation(col("raster"), col("geom"), SpatialRelationType::Intersects);
        assert!(raster_operand_on_left(&pred, &raster_schema).unwrap());

        // (geometry, raster): the raster is on the probe side already.
        let pred = relation(col("geom"), col("raster"), SpatialRelationType::Intersects);
        assert!(!raster_operand_on_left(&pred, &geom_schema).unwrap());
    }

    #[test]
    fn crs_less_geometry_yields_crs_less_target() {
        let raster_schema = schema_with("raster", &RASTER);
        let geom_schema = schema_with("geom", &WKB_GEOMETRY);

        let pred = relation(col("raster"), col("geom"), SpatialRelationType::Within);
        let target = raster_geometry_target_crs(&pred, &raster_schema, &geom_schema)
            .unwrap()
            .expect("raster/geometry should be handled");
        assert!(
            target.is_none(),
            "CRS-less geometry means a CRS-less target"
        );
    }

    #[test]
    fn different_geometry_crs_is_carried_through() {
        let geom_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());
        let raster_schema = schema_with("raster", &RASTER);
        let geom_schema = schema_with("geom", &geom_type);

        let pred = relation(col("geom"), col("raster"), SpatialRelationType::Intersects);
        let target = raster_geometry_target_crs(&pred, &geom_schema, &raster_schema)
            .unwrap()
            .expect("raster/geometry should be handled");
        assert!(target
            .as_deref()
            .unwrap()
            .crs_equals(deserialize_crs("EPSG:3857").unwrap().as_deref().unwrap()));
    }

    #[test]
    fn raster_raster_is_declined() {
        let raster_schema = schema_with("raster", &RASTER);
        let pred = relation(
            col("raster"),
            col("raster"),
            SpatialRelationType::Intersects,
        );
        assert!(
            raster_geometry_target_crs(&pred, &raster_schema, &raster_schema)
                .unwrap()
                .is_none(),
            "raster/raster has no fixed common CRS and must be declined"
        );
    }

    #[test]
    fn geometry_geometry_is_declined() {
        let geom_schema = schema_with("geom", &WKB_GEOMETRY);
        let pred = relation(col("geom"), col("geom"), SpatialRelationType::Intersects);
        assert!(
            raster_geometry_target_crs(&pred, &geom_schema, &geom_schema)
                .unwrap()
                .is_none(),
            "pure geometry predicates are left to the default planner"
        );
    }

    #[test]
    fn unsupported_relation_type_is_declined() {
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let raster_schema = schema_with("raster", &RASTER);
        let geom_schema = schema_with("geom", &geom_type);

        // Equals is not an RS_ predicate.
        let pred = relation(col("raster"), col("geom"), SpatialRelationType::Equals);
        assert!(
            raster_geometry_target_crs(&pred, &raster_schema, &geom_schema)
                .unwrap()
                .is_none()
        );
    }
}
