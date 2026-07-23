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

//! End-to-end test for raster-geometry optimized spatial joins.
//!
//! A join predicated on `RS_Intersects(raster, geom)` is recognized as a spatial
//! predicate (so it becomes a `SpatialJoin` extension node) and is planned as the
//! optimized `SpatialJoinExec` by the raster extension planner, which evaluates
//! each raster into its footprint polygon rather than falling back to a
//! `NestedLoopJoinExec`.
//!
//! The row-correctness case is gated behind the `proj` feature because the raster
//! footprint evaluator (like the `RS_Intersects` kernel) initializes the global
//! PROJ engine (which needs `sedona-proj/proj-sys`); the plan-shape case only
//! plans and so runs without PROJ.

// The spatial-join planner/optimizer are only wired in when this feature is enabled.
#![cfg(feature = "spatial-join")]

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{Field, Schema};
use datafusion::catalog::MemTable;
use datafusion::physical_plan::displayable;
use sedona::context::SedonaContext;
use sedona_geometry::types::Edges;
use sedona_schema::crs::lnglat;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_testing::create::create_array;
use sedona_testing::rasters::generate_test_rasters;

/// Geometry type sharing the CRS that `generate_test_rasters` stamps on its rasters
/// (lng/lat), so the predicate compares in a common CRS without reprojection.
fn geom_type() -> SedonaType {
    SedonaType::Wkb(Edges::Planar, lnglat())
}

/// Register a raster table `r(rid INT, raster RASTER)`.
///
/// `generate_test_rasters(3, Some(0))` yields: index 0 null, index 1 with a convex
/// hull around x∈[2.0, 2.29] y∈[2.4, 3.08], index 2 around x∈[3.0, 3.84] y∈[2.4, 4.24].
fn register_raster_table(ctx: &SedonaContext) {
    let rasters = generate_test_rasters(3, Some(0)).unwrap();
    let rid = Int32Array::from(vec![0, 1, 2]);

    let raster_field = RASTER.to_storage_field("raster", true).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("rid", arrow_schema::DataType::Int32, false),
        raster_field,
    ]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(rid), Arc::new(rasters)]).unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.ctx.register_table("r", Arc::new(table)).unwrap();
}

/// Register a geometry table `g(gid INT, geom GEOMETRY)`.
fn register_geom_table(ctx: &SedonaContext) {
    let geom_type = geom_type();
    let gid = Int32Array::from(vec![10, 20, 30]);
    let geom = create_array(
        &[
            Some("POINT (2.15 2.75)"),                       // inside raster 1 only
            Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"), // contains rasters 1 and 2
            Some("POINT (0 0)"),                             // outside all rasters
        ],
        &geom_type,
    );

    let geom_field = geom_type.to_storage_field("geom", true).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("gid", arrow_schema::DataType::Int32, false),
        geom_field,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(gid), geom]).unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.ctx.register_table("g", Arc::new(table)).unwrap();
}

const RASTER_JOIN_SQL: &str = "SELECT r.rid, g.gid \
     FROM r JOIN g ON RS_Intersects(r.raster, g.geom) \
     ORDER BY r.rid, g.gid";

/// The `RS_Intersects` join is recognized as a spatial predicate (so the optimized
/// logical plan contains the `SpatialJoin` extension node) and is planned as the
/// optimized `SpatialJoinExec` by the raster extension planner, not the
/// `NestedLoopJoinExec` fallback. Planning only, so no PROJ required.
#[tokio::test]
async fn raster_join_uses_optimized_spatial_join() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    register_raster_table(&ctx);
    register_geom_table(&ctx);

    let df = ctx.sql(RASTER_JOIN_SQL).await.unwrap();

    let optimized = df.clone().into_optimized_plan().unwrap();
    let logical_str = format!("{}", optimized.display_indent());
    assert!(
        logical_str.contains("SpatialJoin"),
        "expected a SpatialJoin extension node in the optimized logical plan, got:\n{logical_str}"
    );

    let physical = df.create_physical_plan().await.unwrap();
    let physical_str = displayable(physical.as_ref()).indent(true).to_string();
    assert!(
        physical_str.contains("SpatialJoinExec"),
        "raster join should use the optimized SpatialJoinExec, got:\n{physical_str}"
    );
    assert!(
        !physical_str.contains("NestedLoopJoinExec"),
        "raster join must not fall back to NestedLoopJoinExec, got:\n{physical_str}"
    );
}

/// The raster join executes correctly through the nested-loop fallback: only
/// (raster 1, point-in-1), (raster 1, big-polygon), and (raster 2, big-polygon)
/// intersect, and the null raster (rid 0) never matches.
#[cfg(feature = "proj")]
#[tokio::test]
async fn raster_join_produces_correct_rows() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    register_raster_table(&ctx);
    register_geom_table(&ctx);

    let batches = ctx
        .sql(RASTER_JOIN_SQL)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    datafusion::assert_batches_eq!(
        [
            "+-----+-----+",
            "| rid | gid |",
            "+-----+-----+",
            "| 1   | 10  |",
            "| 1   | 20  |",
            "| 2   | 20  |",
            "+-----+-----+",
        ],
        &batches
    );
}

/// An untagged (CRS-less) geometry joined against the WGS84 rasters no longer
/// errors: the missing CRS is assumed to be WGS84 (matching the `RS_*` kernel and
/// Sedona Spark), so the join returns exactly the rows of the CRS-tagged case
/// (`raster_join_produces_correct_rows`).
#[cfg(feature = "proj")]
#[tokio::test]
async fn untagged_geometry_join_assumes_wgs84() {
    use sedona_schema::datatypes::WKB_GEOMETRY;

    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    register_raster_table(&ctx);

    // Same geometries as register_geom_table, but the column carries no CRS.
    let gid = Int32Array::from(vec![10, 20, 30]);
    let geom = create_array(
        &[
            Some("POINT (2.15 2.75)"),                       // inside raster 1 only
            Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"), // contains rasters 1 and 2
            Some("POINT (0 0)"),                             // outside all rasters
        ],
        &WKB_GEOMETRY,
    );
    let geom_field = WKB_GEOMETRY.to_storage_field("geom", true).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("gid", arrow_schema::DataType::Int32, false),
        geom_field,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(gid), geom]).unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.ctx.register_table("g", Arc::new(table)).unwrap();

    let batches = ctx
        .sql(RASTER_JOIN_SQL)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    datafusion::assert_batches_eq!(
        [
            "+-----+-----+",
            "| rid | gid |",
            "+-----+-----+",
            "| 1   | 10  |",
            "| 1   | 20  |",
            "| 2   | 20  |",
            "+-----+-----+",
        ],
        &batches
    );
}

/// A geometry-only join is unaffected by the raster changes: it still uses the
/// optimized `SpatialJoinExec` (no regression to the existing ST_* path).
#[tokio::test]
async fn geometry_join_still_uses_optimized_spatial_join() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    register_geom_table(&ctx);

    // Self-join two geometry tables on ST_Intersects.
    let geom_type = geom_type();
    let gid = Int32Array::from(vec![100, 200]);
    let geom = create_array(
        &[
            Some("POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))"),
            Some("POINT (100 100)"),
        ],
        &geom_type,
    );
    let geom_field = geom_type.to_storage_field("geom", true).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("gid2", arrow_schema::DataType::Int32, false),
        geom_field,
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(gid), geom]).unwrap();
    let table = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.ctx.register_table("g2", Arc::new(table)).unwrap();

    let sql = "SELECT g.gid, g2.gid2 \
               FROM g JOIN g2 ON ST_Intersects(g.geom, g2.geom)";
    let df = ctx.sql(sql).await.unwrap();

    let physical = df.clone().create_physical_plan().await.unwrap();
    let physical_str = displayable(physical.as_ref()).indent(true).to_string();
    assert!(
        physical_str.contains("SpatialJoinExec"),
        "geometry join should use the optimized SpatialJoinExec, got:\n{physical_str}"
    );

    // Sanity: the join still executes and produces at least the expected overlap row.
    let batches = df.collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows >= 1, "expected at least one intersecting pair");
}
