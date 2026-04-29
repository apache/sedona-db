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

//! Integration tests for geography-based spatial joins.
//!
//! These tests verify that geography spatial joins work correctly,
//! particularly for edge cases like geometries crossing the antimeridian.

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{Field, Schema, SchemaRef};
use datafusion::{
    catalog::{MemTable, TableProvider},
    execution::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::Result;
use datafusion_expr::JoinType;
use geo_types::{Coord, Rect};
use rstest::rstest;
use sedona_common::option::{add_sedona_option_extension, SpatialJoinOptions};
use sedona_common::SedonaOptions;
use sedona_geometry::types::GeometryTypeId;
use sedona_query_planner::{
    optimizer::register_spatial_join_logical_optimizer, query_planner::SedonaQueryPlanner,
};
use sedona_schema::datatypes::WKB_GEOGRAPHY;
use sedona_spatial_join::SpatialJoinExec;
use sedona_spatial_join_geography::physical_planner::GeographySpatialJoinPhysicalPlanner;
use sedona_testing::create::create_array_storage;
use sedona_testing::datagen::RandomPartitionedDataBuilder;

type TestPartitions = (SchemaRef, Vec<Vec<RecordBatch>>);

/// Creates test data with bounding boxes that cross the antimeridian.
///
/// This is a key test case for geography joins - on a Euclidean plane, these
/// geometries would be disjoint, but on a sphere they should intersect.
///
/// - Left data (polygons): bounds [170, -10, 190, 10] - east of the antimeridian
/// - Right data (polygons): bounds [-190, -10, -170, 10] - west of the antimeridian
fn create_antimeridian_crossing_test_data() -> Result<(TestPartitions, TestPartitions)> {
    // Bounding boxes that cross the antimeridian
    // East side: 170°E to 190°E (equivalent to 170°W)
    let east_bounds = Rect::new(Coord { x: 170.0, y: -10.0 }, Coord { x: 190.0, y: 10.0 });
    // West side: -190° (equivalent to 170°E) to -170° (170°W)
    let west_bounds = Rect::new(
        Coord {
            x: -190.0,
            y: -10.0,
        },
        Coord { x: -170.0, y: 10.0 },
    );

    let left_data = RandomPartitionedDataBuilder::new()
        .seed(44)
        .num_partitions(2)
        .batches_per_partition(2)
        .rows_per_batch(25)
        .geometry_type(GeometryTypeId::Polygon)
        .sedona_type(WKB_GEOGRAPHY)
        .bounds(east_bounds)
        .size_range((0.1, 5.0))
        .null_rate(0.0)
        .build()?;

    let right_data = RandomPartitionedDataBuilder::new()
        .seed(542)
        .num_partitions(4)
        .batches_per_partition(2)
        .rows_per_batch(25)
        .geometry_type(GeometryTypeId::Polygon)
        .sedona_type(WKB_GEOGRAPHY)
        .bounds(west_bounds)
        .size_range((0.1, 5.0))
        .null_rate(0.0)
        .build()?;

    Ok((left_data, right_data))
}

fn setup_context(options: Option<SpatialJoinOptions>, batch_size: usize) -> Result<SessionContext> {
    let mut session_config = SessionConfig::from_env()?
        .with_information_schema(true)
        .with_batch_size(batch_size);
    session_config = add_sedona_option_extension(session_config);
    let mut state_builder = SessionStateBuilder::new();
    if let Some(options) = options {
        state_builder = register_spatial_join_logical_optimizer(state_builder)?;
        state_builder = state_builder.with_query_planner(Arc::new(
            SedonaQueryPlanner::new().with_spatial_join_physical_planner(Arc::new(
                GeographySpatialJoinPhysicalPlanner::new(),
            )),
        ));
        let opts = session_config
            .options_mut()
            .extensions
            .get_mut::<SedonaOptions>()
            .unwrap();
        opts.spatial_join = options;
    }
    let state = state_builder.with_config(session_config).build();
    let ctx = SessionContext::new_with_state(state);

    // Register the full function set (includes geography functions)
    let mut function_set = sedona_functions::register::default_function_set();

    function_set.scalar_udfs().for_each(|udf| {
        ctx.register_udf(udf.clone().into());
    });

    // Register s2geography kernels
    for (name, kernel) in sedona_s2geography::register::scalar_kernels()?.into_iter() {
        let udf = function_set.add_scalar_udf_impl(name, kernel)?;
        ctx.register_udf(udf.clone().into());
    }

    Ok(ctx)
}

fn collect_spatial_join_exec(
    plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
) -> Result<Vec<&SpatialJoinExec>> {
    let mut spatial_join_execs = Vec::new();
    plan.apply(|node| {
        if let Some(spatial_join_exec) = node.as_any().downcast_ref::<SpatialJoinExec>() {
            spatial_join_execs.push(spatial_join_exec);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(spatial_join_execs)
}

async fn run_spatial_join_query(
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    left_partitions: Vec<Vec<RecordBatch>>,
    right_partitions: Vec<Vec<RecordBatch>>,
    options: Option<SpatialJoinOptions>,
    batch_size: usize,
    sql: &str,
) -> Result<RecordBatch> {
    let mem_table_left: Arc<dyn TableProvider> =
        Arc::new(MemTable::try_new(left_schema.to_owned(), left_partitions)?);
    let mem_table_right: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
        right_schema.to_owned(),
        right_partitions,
    )?);

    let is_optimized_spatial_join = options.is_some();
    let ctx = setup_context(options, batch_size)?;
    ctx.register_table("L", Arc::clone(&mem_table_left))?;
    ctx.register_table("R", Arc::clone(&mem_table_right))?;
    let df = ctx.sql(sql).await?;
    let actual_schema = df.schema().as_arrow().clone();
    let plan = df.clone().create_physical_plan().await?;
    let spatial_join_execs = collect_spatial_join_exec(&plan)?;
    if is_optimized_spatial_join {
        assert_eq!(
            spatial_join_execs.len(),
            1,
            "Expected optimized spatial join for geography query"
        );
    } else {
        assert!(
            spatial_join_execs.is_empty(),
            "Should not have SpatialJoinExec when optimization is disabled"
        );
    }
    let result_batches = df.collect().await?;
    let result_batch = arrow::compute::concat_batches(&Arc::new(actual_schema), &result_batches)?;
    Ok(result_batch)
}

async fn test_spatial_join_query(
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    left_partitions: Vec<Vec<RecordBatch>>,
    right_partitions: Vec<Vec<RecordBatch>>,
    options: &SpatialJoinOptions,
    batch_size: usize,
    sql: &str,
) -> Result<RecordBatch> {
    // Run spatial join using optimized SpatialJoinExec
    let actual = run_spatial_join_query(
        left_schema,
        right_schema,
        left_partitions.clone(),
        right_partitions.clone(),
        Some(options.clone()),
        batch_size,
        sql,
    )
    .await?;

    // Run spatial join using NestedLoopJoinExec (no optimization)
    let expected = run_spatial_join_query(
        left_schema,
        right_schema,
        left_partitions,
        right_partitions,
        None,
        batch_size,
        sql,
    )
    .await?;

    // Both should produce the same result
    assert_eq!(
        expected, actual,
        "Optimized join should match nested loop join"
    );

    Ok(actual)
}

/// Test geography spatial join with antimeridian-crossing geometries.
///
/// This test verifies that geography joins correctly handle geometries that cross
/// the antimeridian (±180° longitude), which would be disjoint on a Euclidean plane
/// but should intersect on a sphere.
///
/// Corresponds to Python test: test_spatial_join_geog_matches_postgis
#[rstest]
#[tokio::test]
async fn test_spatial_join_geog_antimeridian_intersects(
    #[values("INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN")] join_type: &str,
) -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_antimeridian_crossing_test_data()?;

    let options = SpatialJoinOptions::default();
    let batch_size = 30;

    let sql = format!(
        "SELECT L.id l_id, R.id r_id FROM L {} R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
        join_type
    );

    let result = test_spatial_join_query(
        &left_schema,
        &right_schema,
        left_partitions,
        right_partitions,
        &options,
        batch_size,
        &sql,
    )
    .await?;

    // For antimeridian-crossing data with geography type, we should get non-empty results
    // because the geometries DO intersect on the sphere, even though they appear
    // disjoint on a flat map projection
    assert!(
        result.num_rows() > 0,
        "Geography join across antimeridian should produce non-empty results"
    );

    Ok(())
}

/// Test geography distance join with antimeridian-crossing geometries.
///
/// Tests ST_Distance predicate for geography joins.
#[rstest]
#[tokio::test]
async fn test_spatial_join_geog_antimeridian_distance(
    #[values("INNER JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN")] join_type: &str,
) -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_antimeridian_crossing_test_data()?;

    let options = SpatialJoinOptions::default();
    let batch_size = 30;

    // Distance threshold of 100000 meters (100km)
    let sql = format!(
        "SELECT L.id l_id, R.id r_id FROM L {} R ON ST_Distance(L.geometry, R.geometry) < 100000 ORDER BY l_id, r_id",
        join_type
    );

    let result = test_spatial_join_query(
        &left_schema,
        &right_schema,
        left_partitions,
        right_partitions,
        &options,
        batch_size,
        &sql,
    )
    .await?;

    // Should get results for points within 100km of polygons
    assert!(
        result.num_rows() > 0,
        "Geography distance join across antimeridian should produce non-empty results"
    );

    Ok(())
}

/// Test geography join with different join types.
#[rstest]
#[tokio::test]
async fn test_geography_join_types(
    #[values(JoinType::Inner, JoinType::Left, JoinType::Right)] join_type: JoinType,
) -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_antimeridian_crossing_test_data()?;

    let options = SpatialJoinOptions::default();
    let batch_size = 30;

    let sql = match join_type {
        JoinType::Inner => "SELECT L.id l_id, R.id r_id FROM L INNER JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
        JoinType::Left => "SELECT L.id l_id, R.id r_id FROM L LEFT JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
        JoinType::Right => "SELECT L.id l_id, R.id r_id FROM L RIGHT JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
        _ => unreachable!("Only testing Inner, Left, Right join types"),
    };

    let result = test_spatial_join_query(
        &left_schema,
        &right_schema,
        left_partitions,
        right_partitions,
        &options,
        batch_size,
        sql,
    )
    .await?;

    // All join types should produce results
    assert!(result.num_rows() > 0);

    Ok(())
}

/// Test geography join with ST_DWithin predicate.
#[tokio::test]
async fn test_geography_join_dwithin() -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_antimeridian_crossing_test_data()?;

    let options = SpatialJoinOptions::default();
    let batch_size = 30;

    // DWithin with 100km distance threshold
    let sql = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_DWithin(L.geometry, R.geometry, 100000) ORDER BY l_id, r_id";

    let result = test_spatial_join_query(
        &left_schema,
        &right_schema,
        left_partitions,
        right_partitions,
        &options,
        batch_size,
        sql,
    )
    .await?;

    // Should get results for geometries within 100km
    assert!(result.num_rows() > 0);

    Ok(())
}

/// Creates test data for geography corner cases: full-width polygons and polar triangles.
///
/// These are edge cases that stress the spatial index bounding calculations:
/// - Full-width polygons spanning 360 degrees longitude
/// - Polar triangles touching the north/south poles
/// - Polygons crossing the antimeridian with coordinates > 180 or < -180
///
/// Returns (lhs_data, rhs_data) where:
/// - lhs: Corner case fixtures (full-width polygons, polar triangles)
/// - rhs: Random antimeridian-crossing polygons
fn create_corner_case_test_data() -> Result<(TestPartitions, TestPartitions)> {
    // WKT fixtures for edge cases: full-width polygons and polar triangles
    let wkt_fixtures: Vec<Option<&str>> = vec![
        // Full-width polygons (spanning 360 degrees longitude, with intermediate points)
        Some("POLYGON((-180 -10, -180 10, -90 10, 0 10, 90 10, 180 10, 180 -10, 90 -10, 0 -10, -90 -10, -180 -10))"),
        Some("POLYGON((-180 20, -180 40, -90 40, 0 40, 90 40, 180 40, 180 20, 90 20, 0 20, -90 20, -180 20))"),
        Some("POLYGON((-180 -45, -90 -30, 0 -45, 90 -30, 180 -45, 90 -60, 0 -45, -90 -60, -180 -45))"),
        // Full width polygon shifted so that it crosses the antimeridian
        Some("POLYGON((0 -10, 0 10, 90 10, 180 10, 270 10, 360 10, 360 -10, 270 -10, 180 -10, 80 -10, 0 -10))"),
        // Whole planet
        Some("POLYGON((-180 -90, 0 -90, 90 -90, 180 -90, 180 0, 180 90, 90 90, 0 90, -90 90, -180 90, -180 0, -180 -90))"),
        // North pole triangles
        Some("POLYGON((0 70, -120 70, 120 70, 0 70))"),
        Some("POLYGON((0 80, 60 80, -60 80, 0 80))"),
        Some("POLYGON((0 85, 0 85, 90 85, 0 85))"),
        // South pole triangles
        Some("POLYGON((0 -70, -120 -70, 120 -70, 0 -70))"),
        Some("POLYGON((0 -80, 60 -80, -60 -80, 0 -80))"),
        Some("POLYGON((0 -85, 0 -85, 90 -85, 0 -85))"),
    ];

    // Create LHS from WKT fixtures
    let lhs_geometry = create_array_storage(&wkt_fixtures, &WKB_GEOGRAPHY);
    let lhs_ids: Vec<i32> = (0..wkt_fixtures.len() as i32).collect();
    let lhs_id_array = Arc::new(Int32Array::from(lhs_ids));

    let lhs_schema = Arc::new(Schema::new(vec![
        Field::new("id", arrow_schema::DataType::Int32, false),
        WKB_GEOGRAPHY.to_storage_field("geometry", true)?,
    ]));

    let lhs_batch = RecordBatch::try_new(lhs_schema.clone(), vec![lhs_id_array, lhs_geometry])?;

    // Create RHS from random antimeridian-crossing polygons
    // Bounds [160, -90, 200, 90] to get lots of antimeridian crossing geometries
    let antimeridian_bounds = Rect::new(Coord { x: 160.0, y: -90.0 }, Coord { x: 200.0, y: 90.0 });

    let (rhs_schema, rhs_partitions) = RandomPartitionedDataBuilder::new()
        .seed(4326)
        .num_partitions(2)
        .batches_per_partition(2)
        .rows_per_batch(125)
        .geometry_type(GeometryTypeId::Polygon)
        .sedona_type(WKB_GEOGRAPHY)
        .bounds(antimeridian_bounds)
        .size_range((1.0, 5.0))
        .polygon_hole_rate(0.5)
        .null_rate(0.0)
        .build()?;

    Ok((
        (lhs_schema, vec![vec![lhs_batch]]),
        (rhs_schema, rhs_partitions),
    ))
}

/// Test geography spatial join corner cases with ST_Intersects.
///
/// Tests that the optimized spatial join produces the same results as the
/// nested loop join for corner case geometries (full-width polygons, polar
/// triangles, antimeridian-crossing polygons).
#[rstest]
#[tokio::test]
async fn test_spatial_join_geography_corner_case_intersects(
    #[values(
        "ST_Intersects(L.geometry, R.geometry)",
        "ST_Intersects(R.geometry, L.geometry)"
    )]
    predicate: &str,
) -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_corner_case_test_data()?;

    let options = SpatialJoinOptions {
        spatial_join_reordering: false, // Test both sides explicitly
        ..Default::default()
    };
    let batch_size = 30;

    let sql = format!(
        "SELECT L.id l_id, R.id r_id FROM L INNER JOIN R ON {} ORDER BY l_id, r_id",
        predicate
    );

    let result = test_spatial_join_query(
        &left_schema,
        &right_schema,
        left_partitions,
        right_partitions,
        &options,
        batch_size,
        &sql,
    )
    .await?;

    // Should produce non-empty results
    assert!(
        result.num_rows() > 0,
        "Intersects join should produce results for corner case geometries"
    );

    Ok(())
}

/// Test geography spatial join corner cases with ST_Distance predicate.
///
/// Unlike ST_Intersects, the distance-based join with a large threshold should
/// produce matching results between optimized and fallback paths because the
/// bounding box expansion accounts for the distance threshold.
#[rstest]
#[tokio::test]
async fn test_spatial_join_geography_corner_case_distance(
    #[values(
        "ST_Distance(L.geometry, R.geometry) < 10000000",
        "ST_Distance(R.geometry, L.geometry) < 10000000"
    )]
    predicate: &str,
) -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_corner_case_test_data()?;

    let options = SpatialJoinOptions {
        spatial_join_reordering: false, // Test both sides explicitly
        ..Default::default()
    };
    let batch_size = 30;

    let sql = format!(
        "SELECT L.id l_id, R.id r_id FROM L INNER JOIN R ON {} ORDER BY l_id, r_id",
        predicate
    );

    // This should pass - distance joins should match between optimized and fallback
    let result = test_spatial_join_query(
        &left_schema,
        &right_schema,
        left_partitions,
        right_partitions,
        &options,
        batch_size,
        &sql,
    )
    .await?;

    // Should produce non-empty results
    assert!(
        result.num_rows() > 0,
        "Distance join should produce results for corner case geometries"
    );

    Ok(())
}
