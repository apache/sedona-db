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

//! Exercise tile streaming together with Sedona's raster loading rules.
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::catalog::MemTable;
use datafusion::physical_plan::{collect, displayable};
use sedona::context::SedonaContext;
use sedona_schema::{datatypes::SedonaType, raster::BandDataType};
use sedona_testing::raster_spec::RasterSpec;
use std::sync::Arc;

#[tokio::test]
async fn tile_streaming_composes_with_materialization_and_multiple_partitions() {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    ctx.sql("SET sedona.raster.max_batch_bytes = 12")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let raster = RasterSpec::d2(5, 3)
        .crs(None)
        .band(BandDataType::UInt8)
        .build();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        SedonaType::Raster.to_storage_field("rast", false).unwrap(),
    ]));
    let partitions = (0..2)
        .map(|id| {
            vec![RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![id])),
                    Arc::new(raster.clone()),
                ],
            )
            .unwrap()]
        })
        .collect();
    ctx.ctx
        .register_table(
            "r",
            Arc::new(MemTable::try_new(schema, partitions).unwrap()),
        )
        .unwrap();

    let sql = "SELECT id, t.x AS x, t.y AS y, RS_Width(t.tile) AS width
        FROM (SELECT id, UNNEST(RS_LazyTile(rast, 2, 2)) AS t FROM r)
        ORDER BY id, y, x";
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let text = displayable(plan.as_ref()).indent(true).to_string();
    assert!(text.contains("RasterTileExec"), "{text}");
    assert!(
        !text.contains("rs_ensureloaded"),
        "metadata-only tiling must stay lazy: {text}"
    );
    let batches = collect(plan, ctx.ctx.task_ctx()).await.unwrap();
    datafusion::assert_batches_eq!(
        [
            "+----+---+---+-------+",
            "| id | x | y | width |",
            "+----+---+---+-------+",
            "| 0  | 0 | 0 | 2     |",
            "| 0  | 1 | 0 | 2     |",
            "| 0  | 2 | 0 | 1     |",
            "| 0  | 0 | 1 | 2     |",
            "| 0  | 1 | 1 | 2     |",
            "| 0  | 2 | 1 | 1     |",
            "| 1  | 0 | 0 | 2     |",
            "| 1  | 1 | 0 | 2     |",
            "| 1  | 2 | 0 | 1     |",
            "| 1  | 0 | 1 | 2     |",
            "| 1  | 1 | 1 | 2     |",
            "| 1  | 2 | 1 | 1     |",
            "+----+---+---+-------+",
        ],
        &batches
    );

    let df = ctx
        .sql("SELECT UNNEST(RS_LazyTile(rast, 1, 1)) AS t FROM r LIMIT 1")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(displayable(plan.as_ref())
        .indent(true)
        .to_string()
        .contains("RasterTileExec"));
    let batches = collect(plan, ctx.ctx.task_ctx()).await.unwrap();
    assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use async_trait::async_trait;
use sedona_raster::array::RasterStructArray;
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest, RasterLoadResult};
use sedona_raster::traits::RasterRef;
use sedona_raster::view_entries::ViewEntries;
use std::sync::Mutex;

/// Window-reading backend with observable requests: value at (x, y) is y*5+x.
#[derive(Debug, Default)]
struct RecordingLoader {
    requests: Mutex<Vec<ViewEntries>>,
}

#[async_trait]
impl AsyncRasterLoader for RecordingLoader {
    fn name(&self) -> &str {
        "tile-test"
    }
    fn supports_format(&self, format: Option<&str>) -> bool {
        format == Some("tile-test")
    }
    async fn load(
        &self,
        requests: &[&RasterLoadRequest],
    ) -> Result<Vec<RasterLoadResult>, ArrowError> {
        requests
            .iter()
            .map(|req| {
                assert_eq!(req.uri, "missing://source#band=1");
                assert_eq!(req.source_shape, &[3, 5]);
                self.requests.lock().unwrap().push(req.view.clone());
                let v = req.view.as_slice();
                let mut values = Vec::new();
                for y in 0..v[0].steps {
                    for x in 0..v[1].steps {
                        values.push(
                            ((v[0].start + y * v[0].step) * 5 + v[1].start + x * v[1].step) as u8,
                        );
                    }
                }
                let shape = vec![v[0].steps, v[1].steps];
                Ok(RasterLoadResult {
                    bytes: Buffer::from_vec(values),
                    view: ViewEntries::identity_for_shape(&shape),
                    source_shape: shape,
                })
            })
            .collect()
    }
}

async fn lazy_context() -> (SedonaContext, Arc<RecordingLoader>) {
    let ctx = SedonaContext::new_local_interactive().await.unwrap();
    ctx.sql("SET sedona.raster.cache_max_bytes = 0")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let loader = Arc::new(RecordingLoader::default());
    ctx.register_raster_loader(loader.clone());
    let raster = RasterSpec::d2(5, 3)
        .crs(None)
        .band(BandDataType::UInt8)
        .name("values")
        .nodata(255u8)
        .outdb("missing://source#band=1", Some("tile-test"))
        .build();
    let schema = Arc::new(Schema::new(vec![SedonaType::Raster
        .to_storage_field("rast", false)
        .unwrap()]));
    ctx.ctx
        .register_batch(
            "r",
            RecordBatch::try_new(schema, vec![Arc::new(raster)]).unwrap(),
        )
        .unwrap();
    (ctx, loader)
}

#[tokio::test]
async fn scalar_lists_and_streamed_tiles_do_not_load_pixels() {
    use datafusion_common::cast::{as_list_array, as_struct_array};
    let (ctx, loader) = lazy_context().await;
    for args in ["2, 2", "make_array(1), 2, 2"] {
        let sql = format!("SELECT RS_LazyTile(rast, {args}) AS tiles FROM r");
        let df = ctx.sql(&sql).await.unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let text = displayable(plan.as_ref()).indent(true).to_string();
        assert!(!text.contains("rs_ensureloaded"), "{text}");
        let batches = collect(plan, ctx.ctx.task_ctx()).await.unwrap();
        let list = as_list_array(batches[0].column(0)).unwrap();
        assert_eq!(list.value_length(0), 6);
        let items = as_struct_array(list.values()).unwrap();
        let tiles = RasterStructArray::try_new(as_struct_array(items.column(2)).unwrap()).unwrap();
        let tile = tiles.get(4).unwrap();
        assert_eq!((tile.width().unwrap(), tile.height().unwrap()), (2, 1));
        let band = tile.band(0).unwrap();
        assert!(!band.is_indb());
        assert_eq!(band.raw_source_shape(), &[3, 5]);
        assert_eq!(band.name(), Some("values"));
        assert_eq!(band.nodata(), Some([255u8].as_slice()));
        assert_eq!(band.view().as_slice()[0].start, 2);
        assert_eq!(band.view().as_slice()[1].start, 2);
        assert_eq!(band.outdb_format(), Some("tile-test"));

        let sql = format!(
            "SELECT RS_Width(t.tile) FROM (SELECT UNNEST(RS_LazyTile(rast, {args})) t FROM r)"
        );
        let df = ctx.sql(&sql).await.unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        assert!(displayable(plan.as_ref())
            .indent(true)
            .to_string()
            .contains("RasterTileExec"));
        assert_eq!(
            collect(plan, ctx.ctx.task_ctx())
                .await
                .unwrap()
                .iter()
                .map(|b| b.num_rows())
                .sum::<usize>(),
            6
        );
    }
    assert!(loader.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn filter_before_loading_reads_only_the_selected_tile() {
    use datafusion_common::cast::as_struct_array;
    let (ctx, loader) = lazy_context().await;
    let sql = "SELECT RS_EnsureLoaded(t.tile) AS tile
        FROM (SELECT UNNEST(RS_LazyTile(rast, 2, 2)) t FROM r)
        WHERE t.x = 1 AND t.y = 0";
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let text = displayable(plan.as_ref()).indent(true).to_string();
    assert!(text.contains("RasterTileExec"), "{text}");
    assert!(text.contains("RasterBatchSplitExec"), "{text}");
    let batches = collect(plan, ctx.ctx.task_ctx()).await.unwrap();
    let batch = batches.iter().find(|b| b.num_rows() > 0).unwrap();
    let tiles = RasterStructArray::try_new(as_struct_array(batch.column(0)).unwrap()).unwrap();
    let tile = tiles.get(0).unwrap();
    let band = tile.band(0).unwrap();
    assert_eq!(
        band.nd_buffer().unwrap().as_contiguous().unwrap(),
        &[2, 3, 7, 8]
    );
    let requests = loader.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].as_slice()[0].steps, 2);
    assert_eq!(requests[0].as_slice()[1].start, 2);
    assert_eq!(requests[0].as_slice()[1].steps, 2);
}

#[tokio::test]
async fn retiling_composes_views_and_pixel_functions_load_downstream() {
    let (ctx, loader) = lazy_context().await;
    let sql = "SELECT RS_Value(u.tile, 1, 1, 1) AS value FROM (
        SELECT UNNEST(RS_LazyTile(t.tile, 1, 1)) u FROM (
            SELECT UNNEST(RS_LazyTile(rast, 3, 2)) t FROM r
        ) WHERE t.x = 1 AND t.y = 1
    ) WHERE u.x = 1";
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    datafusion::assert_batches_eq!(
        [
            "+-------+",
            "| value |",
            "+-------+",
            "| 14.0  |",
            "+-------+"
        ],
        &batches
    );
    let requests = loader.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let v = requests[0].as_slice();
    assert_eq!(
        (v[0].start, v[0].steps, v[1].start, v[1].steps),
        (2, 1, 4, 1)
    );
}

#[tokio::test]
async fn rs_tile_padding_loads_pixels() {
    use datafusion_common::cast::as_struct_array;
    for args in ["2, 2, true, 99", "make_array(1), 2, 2, true, 99"] {
        let (ctx, loader) = lazy_context().await;
        let sql = format!("SELECT t.tile FROM (SELECT UNNEST(RS_Tile(rast, {args})) t FROM r) WHERE t.x = 2 AND t.y = 1");
        let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
        let batch = batches.iter().find(|b| b.num_rows() > 0).unwrap();
        let tiles = RasterStructArray::try_new(as_struct_array(batch.column(0)).unwrap()).unwrap();
        let tile = tiles.get(0).unwrap();
        let band = tile.band(0).unwrap();
        assert!(band.is_indb());
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap(),
            &[14, 99, 99, 99]
        );
        let requests = loader.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].is_identity(&[3, 5]));
    }
}

#[tokio::test]
async fn padding_a_lazy_tile_from_the_full_source_cache_uses_its_view() {
    use datafusion_common::cast::as_struct_array;
    let (ctx, loader) = lazy_context().await;
    ctx.sql("SET sedona.raster.cache_max_bytes = 1024")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    // Warm the cache with the full source. The later load will share those bytes
    // with a non-contiguous 2x2 view instead of resolving the window in the loader.
    ctx.sql("SELECT RS_EnsureLoaded(rast) FROM r")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let sql = "SELECT u.tile FROM (
        SELECT UNNEST(RS_Tile(t.tile, 3, 3, true, 99)) u FROM (
            SELECT UNNEST(RS_LazyTile(rast, 2, 2)) t FROM r
        ) WHERE t.x = 1 AND t.y = 0
    )";
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let batch = batches.iter().find(|b| b.num_rows() > 0).unwrap();
    let tiles = RasterStructArray::try_new(as_struct_array(batch.column(0)).unwrap()).unwrap();
    let tile = tiles.get(0).unwrap();
    let band = tile.band(0).unwrap();
    assert_eq!(
        band.nd_buffer().unwrap().as_contiguous().unwrap(),
        &[2, 3, 99, 7, 8, 99, 99, 99, 99]
    );
    assert_eq!(
        loader.requests.lock().unwrap().len(),
        1,
        "reuse the warm source cache"
    );
}

#[tokio::test]
async fn lazy_tile_rejects_padding_and_tile_still_loads_pixels() {
    let (ctx, loader) = lazy_context().await;
    for args in [
        "2, 2, false",
        "2, 2, true, 99",
        "make_array(1), 2, 2, false",
    ] {
        assert!(ctx
            .sql(&format!("SELECT RS_LazyTile(rast, {args}) FROM r"))
            .await
            .is_err());
    }
    assert!(loader.requests.lock().unwrap().is_empty());
    let batches = ctx
        .sql("SELECT t.tile FROM (SELECT UNNEST(RS_Tile(rast, 2, 2)) t FROM r)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(loader.requests.lock().unwrap().len(), 1);
    let rasters = RasterStructArray::try_new(
        datafusion_common::cast::as_struct_array(batches[0].column(0)).unwrap(),
    )
    .unwrap();
    assert!(rasters.get(0).unwrap().band(0).unwrap().is_indb());
}
