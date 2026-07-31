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

//! Integration tests for raster/geometry spatial joins.
//!
//! The strict-parity tests run the same query two ways over the same data:
//!
//! * **Optimized** — the raster extension planner turns the `RS_*` join into a
//!   `SpatialJoinExec` that evaluates each raster into its (reprojected)
//!   footprint polygon.
//! * **Nested-loop** — with the spatial-join optimizer/planner absent, the join
//!   stays a `NestedLoopJoinExec` that evaluates the `RS_*` UDF kernel directly.
//!
//! When the rasters and the geometry share a CRS (as they do here — both are
//! lng/lat) neither side reprojects, and both refine with the same `tg` engine,
//! so the two row sets must be byte-identical. This asserts the accelerated path
//! does not silently diverge from the established kernel.
//!
//! `proj-sys` is enabled as a dev-dependency, so the `RS_*` kernel's global PROJ
//! engine initializes for the nested-loop side.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch, StructArray};
use arrow_schema::{Field, Schema};
use datafusion::{
    catalog::MemTable,
    execution::SessionStateBuilder,
    physical_plan::{displayable, ExecutionPlan},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_common::cast::as_int32_array;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::Result;
use rstest::rstest;
use sedona_common::option::SpatialJoinOptions;
use sedona_common::SedonaOptions;
use sedona_geometry::transform::CrsEngine;
use sedona_geometry::types::Edges;
use sedona_proj::transform::{with_global_proj_engine, LazyProjEngine};
use sedona_query_planner::{
    optimizer::register_spatial_join_logical_optimizer, query_planner::SedonaQueryPlanner,
};
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::footprint::{densify_footprint_ring, FOOTPRINT_POINTS_PER_EDGE};
use sedona_schema::crs::lnglat;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::raster::BandDataType;
use sedona_spatial_join::SpatialJoinExec;
use sedona_spatial_join_raster::physical_planner::RasterSpatialJoinPhysicalPlanner;
use sedona_testing::create::create_array;
use sedona_testing::raster_spec::RasterSpec;
use sedona_testing::rasters::generate_test_rasters;

/// Geometry type sharing the CRS that `generate_test_rasters` stamps on its
/// rasters (lng/lat), so the predicate compares in a common CRS.
fn geom_type() -> SedonaType {
    SedonaType::Wkb(Edges::Planar, lnglat())
}

/// Register a raster table `r(rid INT, raster RASTER)`.
///
/// `generate_test_rasters(3, Some(0))` yields: rid 0 a null raster, rid 1 a
/// footprint over x∈[2.0, 2.29] y∈[2.4, 3.08], rid 2 over x∈[3.0, 3.84]
/// y∈[2.4, 4.24]. All non-null rasters carry the lng/lat CRS.
fn register_raster_table(ctx: &SessionContext) -> Result<()> {
    let rasters = generate_test_rasters(3, Some(0)).unwrap();
    let rid = Int32Array::from(vec![0, 1, 2]);

    let schema = Arc::new(Schema::new(vec![
        Field::new("rid", arrow_schema::DataType::Int32, false),
        RASTER.to_storage_field("raster", true)?,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(rid), Arc::new(rasters)])?;
    let table = MemTable::try_new(schema, vec![vec![batch]])?;
    ctx.register_table("r", Arc::new(table))?;
    Ok(())
}

/// Register a geometry table `g(gid INT, geom GEOMETRY)` covering centers,
/// corners/edges, an enclosing polygon, a non-intersecting zone, and a null.
fn register_geom_table(ctx: &SessionContext) -> Result<()> {
    let geom_type = geom_type();
    let gid = Int32Array::from(vec![10, 20, 30, 40, 50, 60, 70]);
    let geom = create_array(
        &[
            Some("POINT (2.15 2.75)"),                       // inside raster 1
            Some("POINT (3.4 3.2)"),                         // inside raster 2
            Some("POINT (2.0 3.0)"),                         // on a raster-1 corner (boundary/edge)
            Some("POINT (0 0)"),                             // outside all rasters
            Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"), // encloses rasters 1 and 2
            Some("POLYGON ((2.0 2.4, 2.29 2.4, 2.29 3.08, 2.0 3.08, 2.0 2.4))"), // around raster 1
            None,                                            // null geometry never matches
        ],
        &geom_type,
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("gid", arrow_schema::DataType::Int32, false),
        geom_type.to_storage_field("geom", true)?,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(gid), geom])?;
    let table = MemTable::try_new(schema, vec![vec![batch]])?;
    ctx.register_table("g", Arc::new(table))?;
    Ok(())
}

/// Build a context with the raster UDFs registered but no tables. When
/// `optimized`, the spatial-join logical optimizer and the raster physical
/// planner are registered (and spatial join enabled), producing a
/// `SpatialJoinExec`. Otherwise the join stays a `NestedLoopJoinExec`.
fn build_context(optimized: bool) -> Result<SessionContext> {
    let mut session_config = SessionConfig::from_env()?.with_batch_size(16);
    session_config = session_config.with_option_extension(SedonaOptions::default());

    // Install a real CRS engine, mirroring the engine a normal session
    // registers. Both the accelerated join and the RS_* kernel read the engine
    // from the session options to reproject cross-CRS footprints/geometries.
    {
        let opts = session_config
            .options_mut()
            .extensions
            .get_mut::<SedonaOptions>()
            .unwrap();
        opts.runtime = opts.runtime.with_crs_engine(Arc::new(LazyProjEngine));
    }

    let mut state_builder = SessionStateBuilder::new();
    if optimized {
        state_builder = register_spatial_join_logical_optimizer(state_builder)?;
        state_builder = state_builder.with_query_planner(Arc::new(
            SedonaQueryPlanner::new().with_spatial_join_physical_planner(Arc::new(
                RasterSpatialJoinPhysicalPlanner::new(),
            )),
        ));
        let opts = session_config
            .options_mut()
            .extensions
            .get_mut::<SedonaOptions>()
            .unwrap();
        opts.spatial_join = SpatialJoinOptions::default();
    }

    let state = state_builder.with_config(session_config).build();
    let ctx = SessionContext::new_with_state(state);

    // Register the raster functions (RS_Intersects/RS_Contains/RS_Within).
    let function_set = sedona_raster_functions::register::default_function_set();
    function_set.scalar_udfs().for_each(|udf| {
        ctx.register_udf(udf.clone().into());
    });

    Ok(ctx)
}

/// Build a context with the same-CRS `r`/`g` tables registered.
fn setup_context(optimized: bool) -> Result<SessionContext> {
    let ctx = build_context(optimized)?;
    register_raster_table(&ctx)?;
    register_geom_table(&ctx)?;
    Ok(ctx)
}

fn count_spatial_join_execs(plan: &Arc<dyn ExecutionPlan>) -> Result<usize> {
    let mut count = 0;
    plan.apply(|node| {
        if node.as_any().downcast_ref::<SpatialJoinExec>().is_some() {
            count += 1;
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(count)
}

async fn run_query(optimized: bool, sql: &str) -> Result<RecordBatch> {
    let ctx = setup_context(optimized)?;
    let df = ctx.sql(sql).await?;
    let schema = df.schema().as_arrow().clone();
    let plan = df.clone().create_physical_plan().await?;

    let num_spatial_joins = count_spatial_join_execs(&plan)?;
    let physical_str = displayable(plan.as_ref()).indent(true).to_string();
    if optimized {
        assert_eq!(
            num_spatial_joins, 1,
            "expected an optimized SpatialJoinExec, got:\n{physical_str}"
        );
    } else {
        assert_eq!(
            num_spatial_joins, 0,
            "nested-loop baseline should not contain SpatialJoinExec, got:\n{physical_str}"
        );
        assert!(
            physical_str.contains("NestedLoopJoinExec"),
            "nested-loop baseline should use NestedLoopJoinExec, got:\n{physical_str}"
        );
    }

    let batches = df.collect().await?;
    Ok(datafusion::arrow::compute::concat_batches(
        &Arc::new(schema),
        &batches,
    )?)
}

/// Assert the optimized `SpatialJoinExec` returns byte-identical rows to the
/// nested-loop kernel for a same-CRS raster/geometry join. Covers all three
/// `RS_*` predicates, both operand orderings, and inner/left/right joins.
#[rstest]
#[tokio::test]
async fn same_crs_strict_parity(
    #[values("RS_Intersects", "RS_Contains", "RS_Within")] func: &str,
    #[values("r.raster, g.geom", "g.geom, r.raster")] operands: &str,
    #[values("INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN")] join_type: &str,
) -> Result<()> {
    let sql = format!(
        "SELECT r.rid, g.gid FROM r {join_type} g ON {func}({operands}) ORDER BY r.rid, g.gid"
    );

    let optimized = run_query(true, &sql).await?;
    let nested_loop = run_query(false, &sql).await?;

    assert_eq!(
        optimized, nested_loop,
        "optimized SpatialJoinExec must match the nested-loop kernel exactly for {sql}"
    );
    Ok(())
}

/// Anchor the parity comparison to hand-computed rows so a bug shared by both
/// paths cannot pass. `RS_Intersects(raster, geom)` inner join: raster 1 meets
/// its center point (gid 10), the enclosing polygon (gid 50), and its bbox
/// polygon (gid 60), plus the corner point (gid 30); raster 2 meets its center
/// (gid 20) and the enclosing polygon (gid 50). The null raster (rid 0) and the
/// null geometry (gid 70) never match.
#[tokio::test]
async fn intersects_produces_expected_rows() -> Result<()> {
    let sql = "SELECT r.rid, g.gid FROM r JOIN g ON RS_Intersects(r.raster, g.geom) \
               ORDER BY r.rid, g.gid";
    let result = run_query(true, sql).await?;

    datafusion::assert_batches_eq!(
        [
            "+-----+-----+",
            "| rid | gid |",
            "+-----+-----+",
            "| 1   | 10  |",
            "| 1   | 30  |",
            "| 1   | 50  |",
            "| 1   | 60  |",
            "| 2   | 20  |",
            "| 2   | 50  |",
            "+-----+-----+",
        ],
        &[result]
    );
    Ok(())
}

/// Cross-CRS: verify the accelerated path against an independent,
/// construction-based reference. The raster is skewed in EPSG:3857; the geometry
/// operand is lng/lat.
///
/// The accelerated footprint densifies each edge of the raster's four-corner quad
/// in the native CRS and reprojects every point into lng/lat, so the footprint
/// follows the curve of each reprojected edge rather than chording across it. The
/// reference builds that same densified, reprojected ring, then places test
/// points whose membership is known by construction: the centroid and the
/// halfway-to-each-corner points are strictly inside the footprint, while the
/// reflection of the centroid through each corner is strictly outside. These
/// points sit well away from the edges, so the expected result is exact (no
/// tolerance) no matter how finely the edges are modeled. The reference
/// reprojects in the same 3857 -> lng/lat direction as the accelerated path, so a
/// wrong transform direction places the accelerated footprint elsewhere and none
/// of the (correctly placed) inside points match — this validates transform
/// direction and the footprint/refine pipeline against PROJ (the same engine a
/// rasterio/geopandas reference would use).
#[tokio::test]
async fn cross_crs_matches_constructed_reference() -> Result<()> {
    let raster = build_skewed_raster_3857();

    // Footprint corners in the raster's native CRS (EPSG:3857). These are exact.
    let native_corners = skewed_raster_native_corners(&raster);

    // Densify the four-corner footprint in the native CRS and reproject every
    // point into lng/lat: this is exactly the footprint the accelerated path
    // builds (each edge follows its reprojected curve, not a straight chord).
    let lnglat_crs = lnglat().unwrap().to_crs_string();
    let ref_ring: Vec<(f64, f64)> = with_global_proj_engine(|engine| {
        let transform = engine
            .get_transform_crs_to_crs("EPSG:3857", &lnglat_crs, None, "")
            .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
        let mut ring = Vec::new();
        densify_footprint_ring(native_corners, &mut ring);
        for point in ring.iter_mut() {
            transform
                .transform_coord(point)
                .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
        }
        Ok(ring)
    })?;

    // The four reprojected corners are the densified ring's vertices at the start
    // of each edge. Test points are placed relative to these corners and the
    // centroid — deep inside or far outside — so their membership is exact
    // regardless of how finely the edges are modeled.
    let corner_stride = FOOTPRINT_POINTS_PER_EDGE + 1;
    let hull: [(f64, f64); 4] = [
        ref_ring[0],
        ref_ring[corner_stride],
        ref_ring[2 * corner_stride],
        ref_ring[3 * corner_stride],
    ];
    let cx = hull.iter().map(|c| c.0).sum::<f64>() / 4.0;
    let cy = hull.iter().map(|c| c.1).sum::<f64>() / 4.0;

    // lng/lat test points with membership known by construction against the
    // convex chord-quad `hull`.
    let mut projected: Vec<(i32, (f64, f64), bool)> = vec![(0, (cx, cy), true)];
    for (i, &(x, y)) in hull.iter().enumerate() {
        // Halfway from centroid to a vertex: strictly inside a convex polygon.
        projected.push((
            10 + i as i32,
            (cx + 0.5 * (x - cx), cy + 0.5 * (y - cy)),
            true,
        ));
        // Reflection of the centroid through a vertex: strictly outside.
        projected.push((
            20 + i as i32,
            (cx + 2.0 * (x - cx), cy + 2.0 * (y - cy)),
            false,
        ));
    }

    // Expected matches: the gids of the inside points.
    let mut expected_inside: Vec<i32> = projected
        .iter()
        .filter(|&&(_, _, inside)| inside)
        .map(|&(gid, _, _)| gid)
        .collect();
    expected_inside.sort_unstable();

    // Build the lng/lat geometry table.
    let geom_type = geom_type();
    let gids: Vec<i32> = projected.iter().map(|&(gid, _, _)| gid).collect();
    let wkts: Vec<String> = projected
        .iter()
        .map(|&(_, (lon, lat), _)| format!("POINT ({lon} {lat})"))
        .collect();
    let wkt_opts: Vec<Option<&str>> = wkts.iter().map(|w| Some(w.as_str())).collect();
    let geom = create_array(&wkt_opts, &geom_type);
    let geom_schema = Arc::new(Schema::new(vec![
        Field::new("gid", arrow_schema::DataType::Int32, false),
        geom_type.to_storage_field("geom", true)?,
    ]));
    let geom_batch = RecordBatch::try_new(
        geom_schema.clone(),
        vec![Arc::new(Int32Array::from(gids)), geom],
    )?;

    // Register the 3857 raster + lng/lat geometry tables in an optimized context.
    let ctx = build_context(true)?;
    let raster_schema = Arc::new(Schema::new(vec![RASTER.to_storage_field("raster", true)?]));
    let raster_batch = RecordBatch::try_new(raster_schema.clone(), vec![Arc::new(raster)])?;
    ctx.register_table(
        "r",
        Arc::new(MemTable::try_new(raster_schema, vec![vec![raster_batch]])?),
    )?;
    ctx.register_table(
        "g",
        Arc::new(MemTable::try_new(geom_schema, vec![vec![geom_batch]])?),
    )?;

    let sql = "SELECT g.gid FROM r JOIN g ON RS_Intersects(r.raster, g.geom) ORDER BY g.gid";
    let df = ctx.sql(sql).await?;
    let plan = df.clone().create_physical_plan().await?;
    assert_eq!(
        count_spatial_join_execs(&plan)?,
        1,
        "cross-CRS raster join should use the optimized SpatialJoinExec"
    );
    let batches = df.collect().await?;

    let mut got: Vec<i32> = Vec::new();
    for batch in &batches {
        let col = as_int32_array(batch.column(0))?;
        for i in 0..col.len() {
            got.push(col.value(i));
        }
    }
    got.sort_unstable();

    assert_eq!(
        got, expected_inside,
        "cross-CRS accelerated join must match the constructed reference"
    );

    // Sanity: we exercised both inside (5) and outside (4) points.
    assert_eq!(expected_inside.len(), 5);
    assert_eq!(got.len(), 5);
    assert_eq!(projected.len(), 9);
    Ok(())
}

/// A 4x4 raster in EPSG:3857 with skew, so its footprint is a non-axis-aligned
/// quadrilateral whose edges curve when reprojected to lng/lat.
fn build_skewed_raster_3857() -> StructArray {
    RasterSpec::d2(4, 4)
        .crs(Some("EPSG:3857"))
        // GDAL geotransform [origin_x, scale_x, skew_x, origin_y, skew_y, scale_y].
        .transform([0.0, 100_000.0, 30_000.0, 2_000_000.0, 20_000.0, -100_000.0])
        .band(BandDataType::UInt8)
        .build()
}

/// The four footprint corners of the single-row `raster` in its native CRS, in
/// ring order (upper-left, upper-right, lower-right, lower-left).
fn skewed_raster_native_corners(raster: &StructArray) -> [(f64, f64); 4] {
    let rasters = RasterStructArray::try_new(raster).unwrap();
    let r = rasters.get(0).unwrap();
    let w = r.width().unwrap();
    let h = r.height().unwrap();
    [
        to_world_coordinate(&r, 0, 0),
        to_world_coordinate(&r, w, 0),
        to_world_coordinate(&r, w, h),
        to_world_coordinate(&r, 0, h),
    ]
}

/// True if `p` lies inside (or on the boundary of) the convex quadrilateral
/// `quad`, tested by requiring the cross product of each directed edge with the
/// vector to `p` to keep a consistent sign.
fn point_in_convex_quad(quad: &[(f64, f64); 4], p: (f64, f64)) -> bool {
    let mut sign = 0.0f64;
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
        if cross != 0.0 {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// Densification catches a point in the sliver between a footprint edge's true
/// reprojected curve and the straight chord through its two reprojected corners.
///
/// Reprojecting the skewed EPSG:3857 raster into lng/lat bows at least one edge
/// outward: its true (curved) boundary bulges past the straight chord between the
/// two reprojected corners. A point placed just inside that curve is inside the
/// true footprint but outside the four-corner chord hull, so a join that only
/// reprojected the corners would miss it. Because the accelerated join densifies
/// each edge before reprojection, the footprint follows the curve and the point
/// matches — the accuracy gain densification buys.
#[tokio::test]
async fn densification_catches_curved_edge_sliver() -> Result<()> {
    let raster = build_skewed_raster_3857();
    let native_corners = skewed_raster_native_corners(&raster);

    // Reproject the four corners (the chord hull) and each edge's native midpoint
    // (a point on the true reprojected curve) into lng/lat.
    let lnglat_crs = lnglat().unwrap().to_crs_string();
    let mut hull = native_corners;
    let mut curve_mids = [(0.0, 0.0); 4];
    with_global_proj_engine(|engine| {
        let transform = engine
            .get_transform_crs_to_crs("EPSG:3857", &lnglat_crs, None, "")
            .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
        for corner in hull.iter_mut() {
            transform
                .transform_coord(corner)
                .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
        }
        for (e, mid) in curve_mids.iter_mut().enumerate() {
            let a = native_corners[e];
            let b = native_corners[(e + 1) % 4];
            let mut m = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
            transform
                .transform_coord(&mut m)
                .map_err(|e| datafusion_common::exec_datafusion_err!("{e}"))?;
            *mid = m;
        }
        Ok(())
    })?;

    let cx = hull.iter().map(|c| c.0).sum::<f64>() / 4.0;
    let cy = hull.iter().map(|c| c.1).sum::<f64>() / 4.0;

    // Find the edge whose true curve bows farthest outward (its midpoint lies
    // outside the chord hull) and build a point just inside that curve.
    let mut best: Option<(f64, (f64, f64))> = None; // (outward depth, test point)
    for e in 0..4 {
        let a = hull[e];
        let b = hull[(e + 1) % 4];
        let curve_mid = curve_mids[e];
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let len = (dx * dx + dy * dy).sqrt();
        // Unit left normal of the chord a -> b.
        let nx = -dy / len;
        let ny = dx / len;
        // Signed perpendicular offset of the curve midpoint and the centroid from
        // the chord line. Opposite signs => the curve bows away from the centroid,
        // i.e. outward past the chord hull.
        let perp_curve = (curve_mid.0 - a.0) * nx + (curve_mid.1 - a.1) * ny;
        let perp_centroid = (cx - a.0) * nx + (cy - a.1) * ny;
        if perp_curve == 0.0 || perp_curve.signum() == perp_centroid.signum() {
            continue; // straight or bows inward: no outward sliver on this edge
        }
        let depth = perp_curve.abs();
        // Step from the true curve halfway back toward the interior: still outside
        // the chord hull (< depth away), but inside the densified footprint (the
        // densified edge hugs the true curve to within ~depth/100).
        let inward = perp_centroid.signum();
        let test_pt = (
            curve_mid.0 + 0.5 * depth * nx * inward,
            curve_mid.1 + 0.5 * depth * ny * inward,
        );
        let better = match best {
            Some((d, _)) => depth > d,
            None => true,
        };
        if better {
            best = Some((depth, test_pt));
        }
    }

    let (_, test_pt) = best.expect("reprojecting the skewed raster must bow an edge outward");

    // The point is outside the four-corner chord hull: a join that reprojected
    // only the corners would miss it.
    assert!(
        !point_in_convex_quad(&hull, test_pt),
        "test point {test_pt:?} must lie outside the four-corner chord hull"
    );

    // Register the raster + the single sliver point in an optimized context.
    let ctx = build_context(true)?;
    let raster_schema = Arc::new(Schema::new(vec![RASTER.to_storage_field("raster", true)?]));
    let raster_batch = RecordBatch::try_new(raster_schema.clone(), vec![Arc::new(raster)])?;
    ctx.register_table(
        "r",
        Arc::new(MemTable::try_new(raster_schema, vec![vec![raster_batch]])?),
    )?;

    let geom_type = geom_type();
    let wkt = format!("POINT ({} {})", test_pt.0, test_pt.1);
    let geom = create_array(&[Some(wkt.as_str())], &geom_type);
    let geom_schema = Arc::new(Schema::new(vec![
        Field::new("gid", arrow_schema::DataType::Int32, false),
        geom_type.to_storage_field("geom", true)?,
    ]));
    let geom_batch = RecordBatch::try_new(
        geom_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1])), geom],
    )?;
    ctx.register_table(
        "g",
        Arc::new(MemTable::try_new(geom_schema, vec![vec![geom_batch]])?),
    )?;

    let sql = "SELECT g.gid FROM r JOIN g ON RS_Intersects(r.raster, g.geom)";
    let df = ctx.sql(sql).await?;
    let plan = df.clone().create_physical_plan().await?;
    assert_eq!(
        count_spatial_join_execs(&plan)?,
        1,
        "cross-CRS raster join should use the optimized SpatialJoinExec"
    );
    let batches = df.collect().await?;
    let matched: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        matched, 1,
        "the densified footprint must match the sliver point that the four-corner chord hull misses"
    );
    Ok(())
}
