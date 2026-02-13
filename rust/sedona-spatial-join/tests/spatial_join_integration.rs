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

use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{
    catalog::{MemTable, TableProvider},
    execution::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, JoinType};
use datafusion_physical_plan::joins::NestedLoopJoinExec;
use datafusion_physical_plan::ExecutionPlan;
use geo::{Distance, Euclidean};
use geo_types::{Coord, Rect};
use rstest::rstest;
use sedona_common::SedonaOptions;
use sedona_geo::to_geo::GeoTypesExecutor;
use sedona_geometry::types::GeometryTypeId;
use sedona_schema::datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY};
use sedona_spatial_join::{
    register_planner, spatial_predicate::RelationPredicate, SpatialJoinExec, SpatialPredicate,
};
use sedona_testing::datagen::RandomPartitionedDataBuilder;
use tokio::sync::OnceCell;

use sedona_common::{
    option::{add_sedona_option_extension, ExecutionMode, SpatialJoinOptions},
    NumSpatialPartitionsConfig, SpatialJoinDebugOptions, SpatialLibrary,
};

type TestPartitions = (SchemaRef, Vec<Vec<RecordBatch>>);

/// Creates standard test data with left (Polygon) and right (Point) partitions
fn create_default_test_data() -> Result<(TestPartitions, TestPartitions)> {
    create_test_data_with_size_range((1.0, 10.0), WKB_GEOMETRY)
}

/// Creates test data with custom size range
fn create_test_data_with_size_range(
    size_range: (f64, f64),
    sedona_type: SedonaType,
) -> Result<(TestPartitions, TestPartitions)> {
    let bounds = Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 100.0 });

    let left_data = RandomPartitionedDataBuilder::new()
        .seed(11584)
        .num_partitions(2)
        .batches_per_partition(2)
        .rows_per_batch(30)
        .geometry_type(GeometryTypeId::Polygon)
        .sedona_type(sedona_type.clone())
        .bounds(bounds)
        .size_range(size_range)
        .null_rate(0.1)
        .build()?;

    let right_data = RandomPartitionedDataBuilder::new()
        .seed(54843)
        .num_partitions(4)
        .batches_per_partition(4)
        .rows_per_batch(30)
        .geometry_type(GeometryTypeId::Point)
        .sedona_type(sedona_type)
        .bounds(bounds)
        .size_range(size_range)
        .null_rate(0.1)
        .build()?;

    Ok((left_data, right_data))
}

/// Creates test data with empty partitions inserted at beginning and end
fn create_test_data_with_empty_partitions() -> Result<(TestPartitions, TestPartitions)> {
    let (mut left_data, mut right_data) = create_default_test_data()?;

    // Add empty partitions
    left_data.1.insert(0, vec![]);
    left_data.1.push(vec![]);
    right_data.1.insert(0, vec![]);
    right_data.1.push(vec![]);

    Ok((left_data, right_data))
}

/// Creates test data for KNN join (Point-Point)
fn create_knn_test_data(
    size_range: (f64, f64),
    sedona_type: SedonaType,
) -> Result<(TestPartitions, TestPartitions)> {
    let bounds = Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 100.0, y: 100.0 });

    let left_data = RandomPartitionedDataBuilder::new()
        .seed(1)
        .num_partitions(2)
        .batches_per_partition(2)
        .rows_per_batch(30)
        .geometry_type(GeometryTypeId::Point)
        .sedona_type(sedona_type.clone())
        .bounds(bounds)
        .size_range(size_range)
        .null_rate(0.1)
        .build()?;

    let right_data = RandomPartitionedDataBuilder::new()
        .seed(2)
        .num_partitions(4)
        .batches_per_partition(4)
        .rows_per_batch(30)
        .geometry_type(GeometryTypeId::Point)
        .sedona_type(sedona_type)
        .bounds(bounds)
        .size_range(size_range)
        .null_rate(0.1)
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
        state_builder = register_planner(state_builder);
        let opts = session_config
            .options_mut()
            .extensions
            .get_mut::<SedonaOptions>()
            .unwrap();
        opts.spatial_join = options;
    }
    let state = state_builder.with_config(session_config).build();
    let ctx = SessionContext::new_with_state(state);

    let mut function_set = sedona_functions::register::default_function_set();
    let scalar_kernels = sedona_geos::register::scalar_kernels();

    function_set.scalar_udfs().for_each(|udf| {
        ctx.register_udf(udf.clone().into());
    });

    for (name, kernel) in scalar_kernels.into_iter() {
        let udf = function_set.add_scalar_udf_impl(name, kernel)?;
        ctx.register_udf(udf.clone().into());
    }

    Ok(ctx)
}

#[tokio::test]
async fn test_empty_data() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("dist", DataType::Float64, false),
        WKB_GEOMETRY.to_storage_field("geometry", true).unwrap(),
    ]));

    let test_data_vec = vec![vec![vec![]], vec![vec![], vec![]]];

    let options = SpatialJoinOptions::default();
    let ctx = setup_context(Some(options.clone()), 10)?;
    for test_data in test_data_vec {
        let left_partitions = test_data.clone();
        let right_partitions = test_data;

        let mem_table_left: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
            Arc::clone(&schema),
            left_partitions.clone(),
        )?);
        let mem_table_right: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
            Arc::clone(&schema),
            right_partitions.clone(),
        )?);

        ctx.deregister_table("L")?;
        ctx.deregister_table("R")?;
        ctx.register_table("L", Arc::clone(&mem_table_left))?;
        ctx.register_table("R", Arc::clone(&mem_table_right))?;

        let sql = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id";
        let df = ctx.sql(sql).await?;
        let result_batches = df.collect().await?;
        for result_batch in result_batches {
            assert_eq!(result_batch.num_rows(), 0);
        }
    }

    Ok(())
}

// Shared test data and expected results - computed only once across all parameterized test cases
// Using tokio::sync::OnceCell for async lazy initialization to avoid recomputing expensive
// test data generation and nested loop join results for each test parameter combination
static TEST_DATA: OnceCell<(TestPartitions, TestPartitions)> = OnceCell::const_new();
static RANGE_JOIN_EXPECTED_RESULTS: OnceCell<Vec<RecordBatch>> = OnceCell::const_new();
static DIST_JOIN_EXPECTED_RESULTS: OnceCell<Vec<RecordBatch>> = OnceCell::const_new();

const RANGE_JOIN_SQL1: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id";
const RANGE_JOIN_SQL2: &str =
    "SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY L.id, R.id";
const RANGE_JOIN_SQLS: &[&str] = &[RANGE_JOIN_SQL1, RANGE_JOIN_SQL2];

const DIST_JOIN_SQL1: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Distance(L.geometry, R.geometry) < 1.0 ORDER BY l_id, r_id";
const DIST_JOIN_SQL2: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Distance(L.geometry, R.geometry) < L.dist / 10.0 ORDER BY l_id, r_id";
const DIST_JOIN_SQL3: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Distance(L.geometry, R.geometry) < R.dist / 10.0 ORDER BY l_id, r_id";
const DIST_JOIN_SQL4: &str = "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_DWithin(L.geometry, R.geometry, 1.0) ORDER BY l_id, r_id";
const DIST_JOIN_SQLS: &[&str] = &[
    DIST_JOIN_SQL1,
    DIST_JOIN_SQL2,
    DIST_JOIN_SQL3,
    DIST_JOIN_SQL4,
];

/// Get test data, computing it only once
async fn get_default_test_data() -> &'static (TestPartitions, TestPartitions) {
    TEST_DATA
        .get_or_init(|| async { create_default_test_data().expect("Failed to create test data") })
        .await
}

/// Get expected results, computing them only once
async fn get_expected_range_join_results() -> &'static Vec<RecordBatch> {
    get_or_init_expected_join_results(&RANGE_JOIN_EXPECTED_RESULTS, RANGE_JOIN_SQLS).await
}

async fn get_expected_distance_join_results() -> &'static Vec<RecordBatch> {
    get_or_init_expected_join_results(&DIST_JOIN_EXPECTED_RESULTS, DIST_JOIN_SQLS).await
}

async fn get_or_init_expected_join_results<'a>(
    lazy_init_results: &'a OnceCell<Vec<RecordBatch>>,
    sql_queries: &[&str],
) -> &'a Vec<RecordBatch> {
    lazy_init_results
        .get_or_init(|| async {
            let test_data = get_default_test_data().await;
            let ((left_schema, left_partitions), (right_schema, right_partitions)) = test_data;

            let batch_size = 10;

            // Run nested loop join to get expected results
            let mut expected_results = Vec::with_capacity(sql_queries.len());

            for (i, sql) in sql_queries.iter().enumerate() {
                let result = run_spatial_join_query(
                    left_schema,
                    right_schema,
                    left_partitions.clone(),
                    right_partitions.clone(),
                    None,
                    batch_size,
                    sql,
                )
                .await
                .unwrap_or_else(|e| panic!("Failed to generate expected result {}: {}", i + 1, e));
                expected_results.push(result);
            }

            expected_results
        })
        .await
}

#[rstest]
#[tokio::test]
async fn test_range_join_with_conf(
    #[values(10, 30, 1000)] max_batch_size: usize,
    #[values(
        ExecutionMode::PrepareNone,
        ExecutionMode::PrepareBuild,
        ExecutionMode::PrepareProbe,
        ExecutionMode::Speculative(20)
    )]
    execution_mode: ExecutionMode,
    #[values(SpatialLibrary::Geo, SpatialLibrary::Geos, SpatialLibrary::Tg)]
    spatial_library: SpatialLibrary,
) -> Result<()> {
    let test_data = get_default_test_data().await;
    let expected_results = get_expected_range_join_results().await;
    let ((left_schema, left_partitions), (right_schema, right_partitions)) = test_data;

    let options = SpatialJoinOptions {
        spatial_library,
        execution_mode,
        ..Default::default()
    };
    for (idx, sql) in RANGE_JOIN_SQLS.iter().enumerate() {
        let actual_result = run_spatial_join_query(
            left_schema,
            right_schema,
            left_partitions.clone(),
            right_partitions.clone(),
            Some(options.clone()),
            max_batch_size,
            sql,
        )
        .await?;
        assert_eq!(&actual_result, &expected_results[idx]);
    }

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_distance_join_with_conf(
    #[values(30, 1000)] max_batch_size: usize,
    #[values(SpatialLibrary::Geo, SpatialLibrary::Geos, SpatialLibrary::Tg)]
    spatial_library: SpatialLibrary,
) -> Result<()> {
    let test_data = get_default_test_data().await;
    let expected_results = get_expected_distance_join_results().await;
    let ((left_schema, left_partitions), (right_schema, right_partitions)) = test_data;

    let options = SpatialJoinOptions {
        spatial_library,
        ..Default::default()
    };
    for (idx, sql) in DIST_JOIN_SQLS.iter().enumerate() {
        let actual_result = run_spatial_join_query(
            left_schema,
            right_schema,
            left_partitions.clone(),
            right_partitions.clone(),
            Some(options.clone()),
            max_batch_size,
            sql,
        )
        .await?;
        assert_eq!(&actual_result, &expected_results[idx]);
    }

    Ok(())
}

#[tokio::test]
async fn test_spatial_join_with_filter() -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_test_data_with_size_range((0.1, 10.0), WKB_GEOMETRY)?;

    for max_batch_size in [10, 30, 100] {
        let options = SpatialJoinOptions::default();
        test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
            "SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) AND L.dist < R.dist ORDER BY L.id, R.id").await?;
        test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
            "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) AND L.dist < R.dist ORDER BY l_id, r_id").await?;
        test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
            "SELECT L.id l_id, R.id r_id, L.dist l_dist, R.dist r_dist FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) AND L.dist < R.dist ORDER BY l_id, r_id").await?;
    }

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_spatial_join_swap_inputs_produces_same_plan(
    #[values(
        ("INNER", "INNER", "L.id, R.id"),
        ("LEFT", "RIGHT", "L.id, R.id"),
        ("RIGHT", "LEFT", "L.id, R.id"),
        ("FULL", "FULL", "L.id, R.id"),
        ("LEFT SEMI", "RIGHT SEMI", "L.id"),
        ("RIGHT SEMI", "LEFT SEMI", "R.id"),
        ("LEFT ANTI", "RIGHT ANTI", "L.id"),
        ("RIGHT ANTI", "LEFT ANTI", "R.id"),
    )]
    join_types: (&str, &str, &str),
) -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_test_data_with_size_range((0.1, 10.0), WKB_GEOMETRY)?;
    let options = SpatialJoinOptions::default();
    let batch_size = 100;

    let mem_table_left: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
        left_schema.clone(),
        left_partitions.clone(),
    )?);
    let mem_table_right: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
        right_schema.clone(),
        right_partitions.clone(),
    )?);

    let ctx = setup_context(Some(options.clone()), batch_size)?;
    ctx.register_table("L", mem_table_left.clone())?;
    ctx.register_table("R", mem_table_right.clone())?;

    // We use a Left Join as a template to create the plan, then modify it to Mark Join
    let sqls = [
        format!("SELECT {} FROM L {} JOIN R ON ST_Contains(L.geometry, R.geometry) AND L.dist < R.dist ORDER BY {}", join_types.2, join_types.0, join_types.2),
        format!("SELECT {} FROM R {} JOIN L ON ST_Within(R.geometry, L.geometry) AND L.dist < R.dist ORDER BY {}", join_types.2, join_types.1, join_types.2),
    ];
    let mut spatial_exec_plans = Vec::with_capacity(sqls.len());
    let mut results = Vec::with_capacity(sqls.len());
    for sql in sqls {
        let df = ctx.sql(&sql).await?;
        let plan = df.clone().create_physical_plan().await?;
        let spatial_join_execs = collect_spatial_join_exec(&plan)?;
        assert_eq!(spatial_join_execs.len(), 1);
        let original_exec = spatial_join_execs[0];
        spatial_exec_plans.push((sql, (*original_exec.join_type(), original_exec.on.clone())));
        let result_batches = df.collect().await?;
        results.push(result_batches);
    }

    // Verify that join types and predicates are the same, the smaller input is always swapped to the left (build) side
    let (join_type_0, predicate_0) = &spatial_exec_plans[0].1;
    let (join_type_1, predicate_1) = &spatial_exec_plans[1].1;
    assert_eq!(join_type_0, join_type_1);
    match (predicate_0, predicate_1) {
        (
            SpatialPredicate::Relation(RelationPredicate {
                relation_type: rel_0,
                ..
            }),
            SpatialPredicate::Relation(RelationPredicate {
                relation_type: rel_1,
                ..
            }),
        ) => {
            assert_eq!(rel_0, rel_1);
        }
        _ => panic!("Expected RelationPredicate"),
    }

    // Verify that results are the same
    assert_eq!(results[0], results[1]);

    Ok(())
}

#[tokio::test]
async fn test_range_join_with_empty_partitions() -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_test_data_with_empty_partitions()?;

    for max_batch_size in [10, 30, 1000] {
        let options = SpatialJoinOptions::default();
        test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
            "SELECT L.id l_id, R.id r_id FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id").await?;
        test_spatial_join_query(
            &left_schema,
            &right_schema,
            left_partitions.clone(),
            right_partitions.clone(),
            &options,
            max_batch_size,
            "SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY L.id, R.id",
        )
        .await?;
    }

    Ok(())
}

#[tokio::test]
async fn test_inner_join() -> Result<()> {
    let options = SpatialJoinOptions::default();
    test_with_join_types(JoinType::Inner, options, 30).await?;
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_left_joins(
    #[values(JoinType::Left, JoinType::LeftSemi, JoinType::LeftAnti)] join_type: JoinType,
) -> Result<()> {
    let options = SpatialJoinOptions::default();
    test_with_join_types(join_type, options, 30).await?;
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_right_joins(
    #[values(JoinType::Right, JoinType::RightSemi, JoinType::RightAnti)] join_type: JoinType,
) -> Result<()> {
    let options = SpatialJoinOptions::default();
    test_with_join_types(join_type, options, 30).await?;
    Ok(())
}

#[tokio::test]
async fn test_full_outer_join() -> Result<()> {
    let options = SpatialJoinOptions::default();
    test_with_join_types(JoinType::Full, options, 30).await?;
    Ok(())
}

#[tokio::test]
async fn test_geography_join_is_not_optimized() -> Result<()> {
    let options = SpatialJoinOptions::default();
    let ctx = setup_context(Some(options), 10)?;

    // Prepare geography tables
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_test_data_with_size_range((0.1, 10.0), WKB_GEOGRAPHY)?;
    let mem_table_left: Arc<dyn TableProvider> =
        Arc::new(MemTable::try_new(left_schema, left_partitions)?);
    let mem_table_right: Arc<dyn TableProvider> =
        Arc::new(MemTable::try_new(right_schema, right_partitions)?);
    ctx.register_table("L", mem_table_left)?;
    ctx.register_table("R", mem_table_right)?;

    // Execute geography join query
    let df = ctx
        .sql("SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry)")
        .await?;
    let plan = df.create_physical_plan().await?;

    // Verify that no SpatialJoinExec is present (geography join should not be optimized)
    let spatial_joins = collect_spatial_join_exec(&plan)?;
    assert!(
        spatial_joins.is_empty(),
        "Geography joins should not be optimized to SpatialJoinExec"
    );

    Ok(())
}

#[tokio::test]
async fn test_query_window_in_subquery() -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_test_data_with_size_range((50.0, 60.0), WKB_GEOMETRY)?;
    let options = SpatialJoinOptions::default();
    test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, 10,
            "SELECT id FROM L WHERE ST_Intersects(L.geometry, (SELECT R.geometry FROM R WHERE R.id = 1))").await?;
    Ok(())
}

#[tokio::test]
async fn test_parallel_refinement_for_large_candidate_set() -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_test_data_with_size_range((1.0, 50.0), WKB_GEOMETRY)?;

    for max_batch_size in [10, 30, 100] {
        let options = SpatialJoinOptions {
            parallel_refinement_chunk_size: 10,
            ..Default::default()
        };
        test_spatial_join_query(&left_schema, &right_schema, left_partitions.clone(), right_partitions.clone(), &options, max_batch_size,
            "SELECT * FROM L JOIN R ON ST_Intersects(L.geometry, R.geometry) AND L.dist < R.dist ORDER BY L.id, R.id").await?;
    }

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_spatial_partitioned_range_join(
    #[values(10, 30, 1000)] max_batch_size: usize,
    #[values(
        ExecutionMode::PrepareNone,
        ExecutionMode::PrepareBuild,
        ExecutionMode::PrepareProbe,
        ExecutionMode::Speculative(20)
    )]
    execution_mode: ExecutionMode,
    #[values(SpatialLibrary::Geo, SpatialLibrary::Geos, SpatialLibrary::Tg)]
    spatial_library: SpatialLibrary,
) -> Result<()> {
    let test_data = get_default_test_data().await;
    let expected_results = get_expected_range_join_results().await;
    let ((left_schema, left_partitions), (right_schema, right_partitions)) = test_data;

    let debug = SpatialJoinDebugOptions {
        num_spatial_partitions: NumSpatialPartitionsConfig::Fixed(4),
        force_spill: true,
        memory_for_intermittent_usage: None,
        ..Default::default()
    };
    let options = SpatialJoinOptions {
        spatial_library,
        execution_mode,
        debug,
        ..Default::default()
    };

    for (idx, sql) in RANGE_JOIN_SQLS.iter().enumerate() {
        let actual_result = run_spatial_join_query(
            left_schema,
            right_schema,
            left_partitions.clone(),
            right_partitions.clone(),
            Some(options.clone()),
            max_batch_size,
            sql,
        )
        .await?;
        assert_eq!(&actual_result, &expected_results[idx]);
    }

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_spatial_partitioned_outer_join(
    #[values(10, 30, 1000)] batch_size: usize,
    #[values(
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::LeftAnti,
        JoinType::RightSemi,
        JoinType::RightAnti
    )]
    join_type: JoinType,
) -> Result<()> {
    let debug = SpatialJoinDebugOptions {
        num_spatial_partitions: NumSpatialPartitionsConfig::Fixed(4),
        force_spill: true,
        memory_for_intermittent_usage: None,
        ..Default::default()
    };
    let options = SpatialJoinOptions {
        debug,
        ..Default::default()
    };

    test_with_join_types(join_type, options, batch_size).await?;
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_mark_joins(
    #[values(JoinType::LeftMark, JoinType::RightMark)] join_type: JoinType,
) -> Result<()> {
    let options = SpatialJoinOptions::default();
    test_mark_join(join_type, options, 10).await?;
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_spatial_partitioned_mark_joins(
    #[values(JoinType::LeftMark, JoinType::RightMark)] join_type: JoinType,
) -> Result<()> {
    let debug = SpatialJoinDebugOptions {
        num_spatial_partitions: NumSpatialPartitionsConfig::Fixed(4),
        force_spill: true,
        memory_for_intermittent_usage: None,
        ..Default::default()
    };
    let options = SpatialJoinOptions {
        debug,
        ..Default::default()
    };
    test_mark_join(join_type, options, 10).await?;
    Ok(())
}

async fn test_with_join_types(
    join_type: JoinType,
    options: SpatialJoinOptions,
    batch_size: usize,
) -> Result<RecordBatch> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_test_data_with_empty_partitions()?;

    let inner_sql = "SELECT L.id l_id, R.id r_id FROM L INNER JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id";
    let sql = match join_type {
        JoinType::Inner => inner_sql,
        JoinType::Left => "SELECT L.id l_id, R.id r_id FROM L LEFT JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
        JoinType::Right => "SELECT L.id l_id, R.id r_id FROM L RIGHT JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
        JoinType::Full => "SELECT L.id l_id, R.id r_id FROM L FULL OUTER JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id, r_id",
        JoinType::LeftSemi => "SELECT L.id l_id FROM L LEFT SEMI JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id",
        JoinType::RightSemi => "SELECT R.id r_id FROM L RIGHT SEMI JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY r_id",
        JoinType::LeftAnti => "SELECT L.id l_id FROM L LEFT ANTI JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY l_id",
        JoinType::RightAnti => "SELECT R.id r_id FROM L RIGHT ANTI JOIN R ON ST_Intersects(L.geometry, R.geometry) ORDER BY r_id",
        JoinType::LeftMark => {
            unreachable!("LeftMark is not directly supported in SQL, will be tested in other tests");
        }
        JoinType::RightMark => {
            unreachable!("RightMark is not directly supported in SQL, will be tested in other tests");
        }
    };

    let batches = test_spatial_join_query(
        &left_schema,
        &right_schema,
        left_partitions.clone(),
        right_partitions.clone(),
        &options,
        batch_size,
        sql,
    )
    .await?;

    if matches!(join_type, JoinType::Left | JoinType::Right | JoinType::Full) {
        // Make sure that we are effectively testing outer joins. If outer joins produces the same result as inner join,
        // it means that the test data is not suitable for testing outer joins.
        let inner_batches = run_spatial_join_query(
            &left_schema,
            &right_schema,
            left_partitions,
            right_partitions,
            Some(options),
            batch_size,
            inner_sql,
        )
        .await?;
        assert!(inner_batches.num_rows() < batches.num_rows());
    }

    Ok(batches)
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
    // Run spatial join using SpatialJoinExec
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

    // Run spatial join using NestedLoopJoinExec
    let expected = run_spatial_join_query(
        left_schema,
        right_schema,
        left_partitions.clone(),
        right_partitions.clone(),
        None,
        batch_size,
        sql,
    )
    .await?;

    // Should produce the same result
    assert!(expected.num_rows() > 0);
    assert_eq!(expected, actual);

    Ok(actual)
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
        assert_eq!(spatial_join_execs.len(), 1);
    } else {
        assert!(spatial_join_execs.is_empty());
    }
    let result_batches = df.collect().await?;
    let result_batch = arrow::compute::concat_batches(&Arc::new(actual_schema), &result_batches)?;
    Ok(result_batch)
}

fn collect_spatial_join_exec(plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<&SpatialJoinExec>> {
    let mut spatial_join_execs = Vec::new();
    plan.apply(|node| {
        if let Some(spatial_join_exec) = node.as_any().downcast_ref::<SpatialJoinExec>() {
            spatial_join_execs.push(spatial_join_exec);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(spatial_join_execs)
}

async fn test_mark_join(
    join_type: JoinType,
    options: SpatialJoinOptions,
    batch_size: usize,
) -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_test_data_with_size_range((0.1, 10.0), WKB_GEOMETRY)?;
    let mem_table_left: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
        left_schema.clone(),
        left_partitions.clone(),
    )?);
    let mem_table_right: Arc<dyn TableProvider> = Arc::new(MemTable::try_new(
        right_schema.clone(),
        right_partitions.clone(),
    )?);

    // We use a Left Join as a template to create the plan, then modify it to Mark Join
    let sql = "SELECT * FROM L LEFT JOIN R ON ST_Intersects(L.geometry, R.geometry)";

    // Create SpatialJoinExec plan
    let ctx = setup_context(Some(options.clone()), batch_size)?;
    ctx.register_table("L", mem_table_left.clone())?;
    ctx.register_table("R", mem_table_right.clone())?;
    let df = ctx.sql(sql).await?;
    let plan = df.create_physical_plan().await?;
    let spatial_join_execs = collect_spatial_join_exec(&plan)?;
    assert_eq!(spatial_join_execs.len(), 1);
    let original_exec = spatial_join_execs[0];
    let mark_exec = SpatialJoinExec::try_new(
        original_exec.left.clone(),
        original_exec.right.clone(),
        original_exec.on.clone(),
        original_exec.filter.clone(),
        &join_type,
        None,
        &options,
    )?;

    // Create NestedLoopJoinExec plan for comparison
    let ctx_no_opt = setup_context(None, batch_size)?;
    ctx_no_opt.register_table("L", mem_table_left)?;
    ctx_no_opt.register_table("R", mem_table_right)?;
    let df_no_opt = ctx_no_opt.sql(sql).await?;
    let plan_no_opt = df_no_opt.create_physical_plan().await?;
    fn collect_nlj_exec(plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<&NestedLoopJoinExec>> {
        let mut execs = Vec::new();
        plan.apply(|node| {
            if let Some(exec) = node.as_any().downcast_ref::<NestedLoopJoinExec>() {
                execs.push(exec);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(execs)
    }
    let nlj_execs = collect_nlj_exec(&plan_no_opt)?;
    assert_eq!(nlj_execs.len(), 1);
    let original_nlj = nlj_execs[0];
    let mark_nlj = NestedLoopJoinExec::try_new(
        original_nlj.children()[0].clone(),
        original_nlj.children()[1].clone(),
        original_nlj.filter().cloned(),
        &join_type,
        None,
    )?;

    async fn run_and_sort(
        plan: Arc<dyn ExecutionPlan>,
        ctx: &SessionContext,
    ) -> Result<RecordBatch> {
        let results = datafusion_physical_plan::collect(plan, ctx.task_ctx()).await?;
        let batch = arrow::compute::concat_batches(&results[0].schema(), &results)?;
        let sort_col = batch.column(0);
        let indices = arrow::compute::sort_to_indices(sort_col, None, None)?;
        let sorted_batch = arrow::compute::take_record_batch(&batch, &indices)?;
        Ok(sorted_batch)
    }

    // Run both Mark Join plans and compare results
    let mark_batch = run_and_sort(Arc::new(mark_exec), &ctx).await?;
    let mark_nlj_batch = run_and_sort(Arc::new(mark_nlj), &ctx_no_opt).await?;
    assert_eq!(mark_batch, mark_nlj_batch);

    Ok(())
}

fn extract_geoms_and_ids(partitions: &[Vec<RecordBatch>]) -> Vec<(i32, geo::Geometry<f64>)> {
    let mut result = Vec::new();
    for partition in partitions {
        for batch in partition {
            let id_idx = batch.schema().index_of("id").expect("Id column not found");
            let ids = batch
                .column(id_idx)
                .as_any()
                .downcast_ref::<arrow_array::Int32Array>()
                .expect("Column 'id' should be Int32");

            let geom_idx = batch
                .schema()
                .index_of("geometry")
                .expect("Geometry column not found");

            let geoms_col = batch.column(geom_idx);
            let geom_type = SedonaType::from_storage_field(batch.schema().field(geom_idx))
                .expect("Failed to get SedonaType from geometry field");
            let arg_types = [geom_type];
            let arg_values = [ColumnarValue::Array(Arc::clone(geoms_col))];

            let executor = GeoTypesExecutor::new(&arg_types, &arg_values);
            let mut id_iter = ids.iter();
            executor
                .execute_wkb_void(|maybe_geom| {
                    if let Some(id_opt) = id_iter.next() {
                        if let (Some(id), Some(geom)) = (id_opt, maybe_geom) {
                            result.push((id, geom))
                        }
                    }
                    Ok(())
                })
                .expect("Failed to extract geoms and ids from RecordBatch");
        }
    }
    result
}

fn compute_knn_ground_truth_with_pair_filter<F>(
    left_partitions: &[Vec<RecordBatch>],
    right_partitions: &[Vec<RecordBatch>],
    k: usize,
    keep_pair: F,
) -> Vec<(i32, i32, f64)>
where
    F: Fn(i32, i32) -> bool,
{
    // NOTE: This helper mirrors our KNN semantics used in execution:
    // - select top-K unfiltered candidates by distance (stable by r_id)
    // - then apply a cross-side predicate to decide which pairs to keep
    //   (can yield < K results per probe row)
    //
    // The predicate is intentionally *post* top-K selection.
    // (See `test_knn_join_with_filter_correctness`.)
    let left_data = extract_geoms_and_ids(left_partitions);
    let right_data = extract_geoms_and_ids(right_partitions);

    let mut results = Vec::new();

    for (l_id, l_geom) in left_data {
        let mut distances: Vec<(i32, f64)> = right_data
            .iter()
            .map(|(r_id, r_geom)| (*r_id, Euclidean.distance(&l_geom, r_geom)))
            .collect();

        // Sort by distance, then by ID for stability
        distances.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        // KNN semantics: pick top-K unfiltered, then optionally post-filter.
        for (r_id, dist) in distances.iter().take(k.min(distances.len())) {
            if keep_pair(l_id, *r_id) {
                results.push((l_id, *r_id, *dist));
            }
        }
    }

    // Sort results by L.id, R.id
    results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    results
}

#[rstest]
#[tokio::test]
async fn test_knn_join_correctness(
    #[values(true, false)] point_only: bool,
    #[values(1, 2, 3, 4)] num_partitions: usize,
    #[values(10, 30, 1000)] max_batch_size: usize,
) -> Result<()> {
    // Generate slightly larger data
    let ((left_schema, left_partitions), (right_schema, right_partitions)) = if point_only {
        create_knn_test_data((0.1, 10.0), WKB_GEOMETRY)?
    } else {
        create_default_test_data()?
    };

    // Use single partition to verify algorithm correctness first, avoiding partitioning issues
    let options = SpatialJoinOptions {
        debug: SpatialJoinDebugOptions {
            num_spatial_partitions: NumSpatialPartitionsConfig::Fixed(num_partitions),
            ..Default::default()
        },
        ..Default::default()
    };
    let k = 6;

    let sql1 = format!(
        "SELECT L.id, R.id, ST_Distance(L.geometry, R.geometry) FROM L JOIN R ON ST_KNN(L.geometry, R.geometry, {}, false) ORDER BY L.id, R.id",
        k
    );
    let expected1 = compute_knn_ground_truth_with_pair_filter(
        &left_partitions,
        &right_partitions,
        k,
        |_l_id, _r_id| true,
    )
    .into_iter()
    .map(|(l, r, _)| (l, r))
    .collect::<Vec<_>>();
    let sql2 = format!(
        "SELECT R.id, L.id, ST_Distance(L.geometry, R.geometry) FROM L JOIN R ON ST_KNN(R.geometry, L.geometry, {}, false) ORDER BY R.id, L.id",
        k
    );
    let expected2 = compute_knn_ground_truth_with_pair_filter(
        &right_partitions,
        &left_partitions,
        k,
        |_l_id, _r_id| true,
    )
    .into_iter()
    .map(|(l, r, _)| (l, r))
    .collect::<Vec<_>>();

    let sqls = [(&sql1, &expected1), (&sql2, &expected2)];

    for (sql, expected_results) in sqls {
        let batches = run_spatial_join_query(
            &left_schema,
            &right_schema,
            left_partitions.clone(),
            right_partitions.clone(),
            Some(options.clone()),
            max_batch_size,
            sql,
        )
        .await?;

        // Collect actual results
        let mut actual_results = Vec::new();
        let combined_batch = arrow::compute::concat_batches(&batches.schema(), &[batches])?;
        let l_ids = combined_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        let r_ids = combined_batch
            .column(1)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();

        for i in 0..combined_batch.num_rows() {
            actual_results.push((l_ids.value(i), r_ids.value(i)));
        }
        actual_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        assert_eq!(actual_results, *expected_results);
    }

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_knn_join_with_filter_correctness(
    #[values(1, 2, 3, 4)] num_partitions: usize,
    #[values(10, 30, 1000)] max_batch_size: usize,
) -> Result<()> {
    let ((left_schema, left_partitions), (right_schema, right_partitions)) =
        create_knn_test_data((0.1, 10.0), WKB_GEOMETRY)?;

    let options = SpatialJoinOptions {
        debug: SpatialJoinDebugOptions {
            num_spatial_partitions: NumSpatialPartitionsConfig::Fixed(num_partitions),
            ..Default::default()
        },
        ..Default::default()
    };

    let k = 3;
    let sql = format!(
        "SELECT L.id AS l_id, R.id AS r_id FROM L JOIN R ON ST_KNN(L.geometry, R.geometry, {}, false) AND (L.id % 7) = (R.id % 7)",
        k
    );

    let batches = run_spatial_join_query(
        &left_schema,
        &right_schema,
        left_partitions.clone(),
        right_partitions.clone(),
        Some(options),
        max_batch_size,
        &sql,
    )
    .await?;

    let mut actual_results = Vec::new();
    let combined_batch = arrow::compute::concat_batches(&batches.schema(), &[batches])?;
    let l_ids = combined_batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
        .unwrap();
    let r_ids = combined_batch
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
        .unwrap();

    for i in 0..combined_batch.num_rows() {
        actual_results.push((l_ids.value(i), r_ids.value(i)));
    }
    actual_results.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // Prove the test actually exercises the "< K rows after filtering" case.
    // Build a list of all probe-side IDs and count how many results each has.
    let all_left_ids: Vec<i32> = extract_geoms_and_ids(&left_partitions)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let mut per_left_counts: std::collections::HashMap<i32, usize> =
        std::collections::HashMap::new();
    for (l_id, _) in &actual_results {
        *per_left_counts.entry(*l_id).or_default() += 1;
    }
    let min_count = all_left_ids
        .iter()
        .map(|l_id| *per_left_counts.get(l_id).unwrap_or(&0))
        .min()
        .unwrap_or(0);
    assert!(
        min_count < k,
        "expected at least one probe row to produce < K rows after filtering; min_count={min_count}, k={k}"
    );

    let expected_results = compute_knn_ground_truth_with_pair_filter(
        &left_partitions,
        &right_partitions,
        k,
        |l_id, r_id| (l_id.rem_euclid(7)) == (r_id.rem_euclid(7)),
    )
    .into_iter()
    .map(|(l, r, _)| (l, r))
    .collect::<Vec<_>>();

    assert_eq!(actual_results, expected_results);

    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_knn_join_include_tie_breakers(
    #[values(1, 2, 3, 4)] num_partitions: usize,
    #[values(10, 100)] max_batch_size: usize,
) -> Result<()> {
    // Construct a larger dataset with *guaranteed* exact ties at the kth distance.
    //
    // For each probe point at (10*i, 0), we create two candidate points at (10*i-1, 0)
    // and (10*i+1, 0). Those two candidates are tied (distance = 1).
    // A third candidate at (10*i+2, 0) ensures there are also non-tied options.
    // Spacing by 10 keeps other probes' candidates far enough away that they never interfere.
    //
    // With k=1:
    // - knn_include_tie_breakers=false should return exactly 1 match per probe row.
    // - knn_include_tie_breakers=true should return 2 matches per probe row (both ties).
    //
    // The exact choice of which tied row is returned when tie-breakers are disabled is not
    // asserted (it is allowed to be either tied candidate).

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("wkt", DataType::Utf8, false),
    ]));

    let num_probe_rows: i32 = 120;
    let k = 1;

    let input_batches_left = 6;
    let input_batches_right = 6;

    fn make_batches(
        schema: SchemaRef,
        ids: Vec<i32>,
        wkts: Vec<String>,
        num_batches: usize,
    ) -> Result<Vec<RecordBatch>> {
        assert_eq!(ids.len(), wkts.len());
        let total = ids.len();
        let chunk = total.div_ceil(num_batches);

        let mut batches = Vec::new();
        for b in 0..num_batches {
            let start = b * chunk;
            if start >= total {
                break;
            }
            let end = ((b + 1) * chunk).min(total);
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(arrow_array::Int32Array::from(ids[start..end].to_vec())),
                    Arc::new(arrow_array::StringArray::from(
                        wkts[start..end]
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                    )),
                ],
            )?;
            batches.push(batch);
        }
        Ok(batches)
    }

    let mut left_ids = Vec::with_capacity(num_probe_rows as usize);
    let mut left_wkts = Vec::with_capacity(num_probe_rows as usize);

    let mut right_ids = Vec::with_capacity((num_probe_rows as usize) * 3);
    let mut right_wkts = Vec::with_capacity((num_probe_rows as usize) * 3);

    for i in 0..num_probe_rows {
        let cx = (i as i64) * 10;
        left_ids.push(i);
        left_wkts.push(format!("POINT ({cx} 0)"));

        // Two tied candidates at distance 1.
        let base = i * 10;
        right_ids.push(base + 1);
        right_wkts.push(format!("POINT ({x} 0)", x = cx - 1));

        right_ids.push(base + 2);
        right_wkts.push(format!("POINT ({x} 0)", x = cx + 1));

        // One non-tied candidate.
        right_ids.push(base + 3);
        right_wkts.push(format!("POINT ({x} 0)", x = cx + 2));
    }

    let left_batches = make_batches(schema.clone(), left_ids, left_wkts, input_batches_left)?;
    let right_batches = make_batches(schema.clone(), right_ids, right_wkts, input_batches_right)?;

    // Put each side into a single MemTable partition, but with multiple batches.
    // This ensures the build/probe collectors see 4–8 batches and the round-robin batch
    // partitioner has something to distribute.
    let left_partitions = vec![left_batches];
    let right_partitions = vec![right_batches];

    let sql = format!(
        "SELECT L.id AS l_id, R.id AS r_id \
            FROM L JOIN R \
            ON ST_KNN(ST_GeomFromWKT(L.wkt), ST_GeomFromWKT(R.wkt), {k}, false)"
    );

    let base_options = SpatialJoinOptions {
        debug: SpatialJoinDebugOptions {
            num_spatial_partitions: NumSpatialPartitionsConfig::Fixed(num_partitions),
            ..Default::default()
        },
        ..Default::default()
    };

    // Without tie-breakers: exactly 1 match per probe row.
    let out_no_ties = run_spatial_join_query(
        &schema,
        &schema,
        left_partitions.clone(),
        right_partitions.clone(),
        Some(SpatialJoinOptions {
            knn_include_tie_breakers: false,
            ..base_options.clone()
        }),
        max_batch_size,
        &sql,
    )
    .await?;
    let combined = arrow::compute::concat_batches(&out_no_ties.schema(), &[out_no_ties])?;

    let l_ids = combined
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
        .unwrap();
    let r_ids = combined
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
        .unwrap();

    let mut per_left: std::collections::HashMap<i32, Vec<i32>> = std::collections::HashMap::new();
    for i in 0..combined.num_rows() {
        per_left
            .entry(l_ids.value(i))
            .or_default()
            .push(r_ids.value(i));
    }

    assert_eq!(per_left.len() as i32, num_probe_rows);
    for l_id in 0..num_probe_rows {
        let r_list = per_left.get(&l_id).unwrap();
        assert_eq!(
            r_list.len(),
            1,
            "expected exactly 1 match for l_id={l_id} when tie-breakers are disabled"
        );
        let base = l_id * 10;
        let r_id = r_list[0];
        assert!(
            r_id == base + 1 || r_id == base + 2,
            "expected a tied nearest neighbor for l_id={l_id}, got r_id={r_id}"
        );
    }

    // With tie-breakers: exactly 2 matches per probe row (both tied candidates).
    let out_with_ties = run_spatial_join_query(
        &schema,
        &schema,
        left_partitions.clone(),
        right_partitions.clone(),
        Some(SpatialJoinOptions {
            knn_include_tie_breakers: true,
            ..base_options
        }),
        max_batch_size,
        &sql,
    )
    .await?;
    let combined = arrow::compute::concat_batches(&out_with_ties.schema(), &[out_with_ties])?;
    let l_ids = combined
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
        .unwrap();
    let r_ids = combined
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
        .unwrap();

    let mut per_left: std::collections::HashMap<i32, Vec<i32>> = std::collections::HashMap::new();
    for i in 0..combined.num_rows() {
        per_left
            .entry(l_ids.value(i))
            .or_default()
            .push(r_ids.value(i));
    }
    assert_eq!(per_left.len() as i32, num_probe_rows);
    for l_id in 0..num_probe_rows {
        let mut r_list = per_left.get(&l_id).unwrap().clone();
        r_list.sort();
        let base = l_id * 10;
        assert_eq!(
            r_list,
            vec![base + 1, base + 2],
            "expected both tied nearest neighbors for l_id={l_id}"
        );
    }

    Ok(())
}
