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

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion_common::Result;
use datafusion_expr::col;
use rstest::rstest;
use sedona::context::{SedonaContext, SedonaDataFrame, SedonaWriteOptions};
use sedona_functions::sd_order_lnglat::OrderLngLat;
use sedona_schema::datatypes::{
    SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_LARGE_GEOGRAPHY, WKB_LARGE_GEOMETRY,
    WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
};
use sedona_testing::create::create_array;

async fn context(data_type: SedonaType, interactive: bool) -> Result<SedonaContext> {
    let mut ctx = if interactive {
        SedonaContext::new_local_interactive().await?
    } else {
        SedonaContext::new()
    };
    // An easily verified spatial order that differs from WKB byte ordering,
    // without requiring the optional S2 native library.
    ctx.register_scalar_kernels(
        [(
            "sd_order",
            OrderLngLat::new(|(x, _)| if x.is_nan() { u64::MAX } else { x as u64 }),
        )]
        .into_iter(),
    )?;
    let geom = create_array(
        &[
            Some("POINT (3 0)"),
            Some("POINT (1 0)"),
            None,
            Some("POINT (2 0)"),
            Some("POINT (1 0)"),
            Some("POINT EMPTY"),
        ],
        &data_type,
    );
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        data_type.to_storage_field("geom", true)?,
        // Same bytes, deliberately without extension metadata.
        Field::new("bytes", geom.data_type().clone(), true),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5])),
            geom.clone(),
            geom,
        ],
    )?;
    ctx.ctx.register_batch("points", batch)?;
    Ok(ctx)
}

async fn ids(ctx: &SedonaContext, sql: &str) -> Result<Vec<i32>> {
    Ok(ctx
        .ctx
        .sql(sql)
        .await?
        .collect()
        .await?
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect())
}

#[rstest]
#[tokio::test]
async fn spatial_sort_types_and_options(
    #[values(
        WKB_GEOMETRY,
        WKB_LARGE_GEOMETRY,
        WKB_VIEW_GEOMETRY,
        WKB_GEOGRAPHY,
        WKB_LARGE_GEOGRAPHY,
        WKB_VIEW_GEOGRAPHY
    )]
    data_type: SedonaType,
    #[values(false, true)] interactive: bool,
) -> Result<()> {
    let ctx = context(data_type, interactive).await?;
    for (options, expected) in [
        ("ASC NULLS LAST", vec![1, 4, 3, 0, 5, 2]),
        ("ASC NULLS FIRST", vec![2, 1, 4, 3, 0, 5]),
        ("DESC NULLS LAST", vec![5, 0, 3, 1, 4, 2]),
        ("DESC NULLS FIRST", vec![2, 5, 0, 3, 1, 4]),
    ] {
        assert_eq!(
            ids(
                &ctx,
                &format!("SELECT id FROM points ORDER BY geom {options}, id")
            )
            .await?,
            expected
        );
        assert_eq!(
            ids(
                &ctx,
                &format!("SELECT id FROM points ORDER BY geom {options}, id LIMIT 3")
            )
            .await?,
            expected[..3]
        );
    }

    let df = ctx.ctx.table("points").await?.sort(vec![
        col("geom").sort(true, false),
        col("id").sort(true, false),
    ])?;
    let plan = df.clone().into_optimized_plan()?;
    assert!(plan
        .display_indent()
        .to_string()
        .contains("sd_order(points.geom)"));
    assert_eq!(plan.schema().as_ref(), df.schema());
    assert_eq!(ctx.ctx.state().optimize(&plan)?, plan);
    // A binary column is not a geometry, even if its values happen to be WKB.
    let binary = ctx
        .ctx
        .sql("SELECT id FROM points ORDER BY bytes")
        .await?
        .into_optimized_plan()?;
    assert!(!binary.display_indent().to_string().contains("sd_order"));
    Ok(())
}

#[rstest]
#[case("SELECT id FROM points ORDER BY {key}, id", "geom")]
#[case(
    "SELECT id FROM points ORDER BY {key} DESC NULLS FIRST, id DESC LIMIT 4",
    "geom"
)]
#[case("SELECT id FROM points ORDER BY {key}, id LIMIT 3 OFFSET 1", "geom")]
#[case(
    "SELECT id FROM (SELECT id, geom AS g FROM points) p ORDER BY {key}, id",
    "g"
)]
#[case("SELECT id FROM points ORDER BY {key}, id", "ST_Point(ST_X(geom), 0)")]
#[case("SELECT ARRAY_AGG(id ORDER BY {key}, id) AS ids FROM points", "geom")]
#[case(
    "SELECT id, ROW_NUMBER() OVER (ORDER BY {key}, id ROWS UNBOUNDED PRECEDING) AS n FROM points ORDER BY id",
    "geom"
)]
#[case(
    "SELECT id, RANK() OVER (ORDER BY {key} ROWS UNBOUNDED PRECEDING) AS n FROM points ORDER BY id",
    "geom"
)]
#[case("SELECT id, ARRAY_AGG(id) OVER (ORDER BY {key}, id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS ids FROM points ORDER BY id", "geom")]
#[case(
    "SELECT DISTINCT ON (geom) id FROM points ORDER BY geom, {key} DESC, id DESC",
    "ST_Point(ST_X(geom), 0)"
)]
#[case("SELECT id FROM points WHERE id IN (SELECT id FROM points ORDER BY {key}, id LIMIT 3) ORDER BY id", "geom")]
#[case(
    "SELECT id FROM (SELECT * FROM points UNION ALL SELECT * FROM points) p ORDER BY {key}, id",
    "geom"
)]
#[tokio::test]
async fn spatial_sort_sql(#[case] template: &str, #[case] key: &str) -> Result<()> {
    let ctx = context(WKB_GEOMETRY, false).await?;
    let implicit = template.replace("{key}", key);
    let explicit = template.replace("{key}", &format!("sd_order({key})"));
    let actual = ctx.ctx.sql(&implicit).await?;
    let actual = actual.collect().await?;
    let expected = ctx.ctx.sql(&explicit).await?.collect().await?;
    assert_eq!(actual, expected, "{implicit}");
    Ok(())
}

#[tokio::test]
async fn spatial_sort_preserves_expression_names() -> Result<()> {
    let ctx = context(WKB_GEOMETRY, false).await?;
    for sql in [
        "SELECT ARRAY_AGG(id ORDER BY geom) FROM points",
        "SELECT ROW_NUMBER() OVER (ORDER BY geom ROWS UNBOUNDED PRECEDING) FROM points",
    ] {
        let df = ctx.ctx.sql(sql).await?;
        let schema = df.schema().clone();
        let optimized = df.into_optimized_plan()?;
        assert_eq!(optimized.schema().as_arrow(), schema.as_arrow());
        assert!(optimized
            .display_indent()
            .to_string()
            .contains("sd_order(points.geom)"));
        ctx.ctx
            .execute_logical_plan(optimized)
            .await?
            .collect()
            .await?;
    }
    Ok(())
}

#[tokio::test]
async fn spatial_sort_writer() -> Result<()> {
    let ctx = context(WKB_GEOMETRY, true).await?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("sorted.parquet");
    let path = path.to_str().unwrap();
    ctx.ctx
        .table("points")
        .await?
        .write_geoparquet(
            &ctx,
            path,
            SedonaWriteOptions::new()
                .with_single_file_output(true)
                .with_sort_by(vec![
                    col("geom").sort(true, false),
                    col("id").sort(false, false),
                ]),
            None,
        )
        .await?;
    assert_eq!(
        ids(&ctx, &format!("SELECT id FROM '{path}'")).await?,
        vec![4, 1, 3, 0, 5, 2]
    );
    Ok(())
}

#[tokio::test]
async fn spatial_sort_identity_fallback() -> Result<()> {
    let ctx = context(WKB_GEOMETRY, false).await?;
    ctx.ctx
        .register_udf(sedona_functions::sd_order::sd_order_udf().into());
    let df = ctx
        .ctx
        .sql("SELECT id FROM points ORDER BY geom, id")
        .await?;
    let plan = df.clone().into_optimized_plan()?;
    assert!(!plan.display_indent().to_string().contains("sd_order"));
    assert_eq!(ctx.ctx.state().optimize(&plan)?, plan);
    assert_eq!(
        df.collect().await?,
        ctx.ctx
            .sql("SELECT id FROM points ORDER BY bytes, id")
            .await?
            .collect()
            .await?
    );
    Ok(())
}

#[tokio::test]
async fn raster_sort_keeps_storage_order() -> Result<()> {
    let ctx = context(WKB_GEOMETRY, false).await?;
    let query = "SELECT id, RS_MakeEmptyRaster(1, 1, 1, id, 0, 1) AS rast FROM points";
    let df = ctx
        .ctx
        .sql(&format!("SELECT id FROM ({query}) ORDER BY rast"))
        .await?;
    let plan = df.clone().into_optimized_plan()?;
    assert!(!plan.display_indent().to_string().contains("sd_order"));
    assert_eq!(
        df.collect().await?,
        ctx.ctx
            .sql(&format!("SELECT id FROM ({query}) ORDER BY sd_order(rast)"))
            .await?
            .collect()
            .await?
    );
    Ok(())
}
