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

//! Streaming execution of a single `UNNEST(RS_Tile(...))`.
//!
//! Arguments and sibling expressions are evaluated once by a projection.
//! A cursor then produces only a bounded run of tiles from
//! one row at a time. DataFusion's UnnestExec expands those short lists, retaining
//! its null/empty-list semantics. The intermediate lists are private to this
//! execution node and never contain the complete tile grid unless it fits.
//!
//! OutDb tiles are lazy windows; pixel consumers load them downstream. InDb
//! tiles are copied, and padding retains eager input loading. Source pushdown
//! is a separate optimization. Multiple zipped
//! unnests and scalar consumers of the list retain their existing execution.

use std::{fmt, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion};
use datafusion_common::{Constraints, Result, UnnestOptions, internal_err};
use datafusion_execution::TaskContext;
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use datafusion_physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::unnest::{ListUnnest, UnnestExec};
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::{StreamExt, TryStreamExt};
use sedona_common::option::{DEFAULT_RASTER_MAX_BATCH_BYTES, SedonaOptions};
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::datatypes::SedonaType;

use crate::rs_tile::{TileCursor, tile_batch};

/// Replace the list-producing projection and its single unnest with a cursor.
/// Runs before RasterMaterializationRule, so the argument projection it creates
/// receives the usual byte-bounded input loading and source distribution.
#[derive(Debug)]
pub struct RasterTileRule;

impl PhysicalOptimizerRule for RasterTileRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(unnest) = node.downcast_ref::<UnnestExec>() else {
                return Ok(Transformed::no(node));
            };
            let [list] = unnest.list_column_indices() else {
                return Ok(Transformed::no(node));
            };
            if list.depth != 1 || !unnest.struct_column_indices().is_empty() {
                return Ok(Transformed::no(node));
            }
            // DataFusion may merge partitions between the projection and unnest.
            // Expand per partition instead, then merge the emitted tile batches.
            let merge = unnest.input().downcast_ref::<CoalescePartitionsExec>();
            if merge.is_some_and(|merge| merge.fetch().is_some()) {
                return Ok(Transformed::no(node));
            }
            let input = merge.map_or(unnest.input(), |merge| merge.input());
            let Some(projection) = input.downcast_ref::<ProjectionExec>() else {
                return Ok(Transformed::no(node));
            };
            let tile_index = list.index_in_input_schema;
            let Some(tile_expr) = projection.expr().get(tile_index) else {
                return Ok(Transformed::no(node));
            };
            let Some(function) = tile_expr.expr.downcast_ref::<ScalarFunctionExpr>() else {
                return Ok(Transformed::no(node));
            };
            // Do not rewrite arbitrary user functions sharing the SQL name.
            if function.fun().name() != "rs_tile"
                || function
                    .fun()
                    .inner()
                    .downcast_ref::<SedonaScalarUDF>()
                    .is_none()
            {
                return Ok(Transformed::no(node));
            }
            let mut arguments = Vec::<ProjectionExpr>::new();
            let mut layout = Vec::<ProjectionExpr>::new();
            let mut tile_args = Vec::new();
            let mut passthrough = Vec::new();
            for (i, expression) in projection.expr().iter().enumerate() {
                if i == tile_index {
                    let mut columns = Vec::<Arc<dyn PhysicalExpr>>::new();
                    for arg in function.args() {
                        let index = arguments.len();
                        let name = format!("__tile_arg_{index}");
                        arguments.push(ProjectionExpr {
                            expr: Arc::clone(arg),
                            alias: name.clone(),
                        });
                        columns.push(Arc::new(Column::new(&name, index)));
                        tile_args.push(index);
                    }
                    layout.push(ProjectionExpr {
                        expr: Arc::clone(&expression.expr).with_new_children(columns)?,
                        alias: expression.alias.clone(),
                    });
                    passthrough.push(None);
                } else {
                    let index = arguments.len();
                    arguments.push(expression.clone());
                    layout.push(ProjectionExpr {
                        expr: Arc::new(Column::new(&expression.alias, index)),
                        alias: expression.alias.clone(),
                    });
                    passthrough.push(Some(index));
                }
            }
            let arguments = Arc::new(ProjectionExec::try_new(
                arguments,
                Arc::clone(projection.input()),
            )?);
            let arg_types = tile_args
                .iter()
                .map(|&index| SedonaType::from_storage_field(arguments.schema().field(index)))
                .collect::<Result<Vec<_>>>()?;
            let lists =
                TileListExec::try_new(arguments, layout, tile_args, arg_types, passthrough)?;
            let mut result: Arc<dyn ExecutionPlan> = Arc::new(RasterTileExec::try_new(
                lists,
                tile_index,
                unnest.schema(),
                unnest.options().clone(),
            )?);
            if merge.is_some() {
                result = Arc::new(CoalescePartitionsExec::new(result));
            }
            Ok(Transformed::yes(result))
        })
        .data()
    }

    fn name(&self) -> &str {
        "sedona.stream_raster_tiles"
    }
    fn schema_check(&self) -> bool {
        true
    }
}

/// Streaming tile expansion with the same output schema as UNNEST(RS_Tile).
#[derive(Debug)]
pub struct RasterTileExec {
    lists: Arc<TileListExec>,
    unnest: UnnestExec,
    tile_index: usize,
}

impl RasterTileExec {
    fn try_new(
        lists: TileListExec,
        tile_index: usize,
        schema: SchemaRef,
        options: UnnestOptions,
    ) -> Result<Self> {
        let lists = Arc::new(lists);
        let unnest = UnnestExec::new(
            lists.clone(),
            vec![ListUnnest {
                index_in_input_schema: tile_index,
                depth: 1,
            }],
            vec![],
            schema,
            options,
        )?;
        Ok(Self {
            lists,
            unnest,
            tile_index,
        })
    }
}

impl DisplayAs for RasterTileExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RasterTileExec: tile_column={}", self.tile_index)
    }
}

impl ExecutionPlan for RasterTileExec {
    fn name(&self) -> &str {
        "RasterTileExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        self.unnest.properties()
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![self.lists.projection.input()]
    }
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("RasterTileExec expects one input");
        }
        let lists = self.lists.with_input(children.remove(0))?;
        Ok(Arc::new(Self::try_new(
            lists,
            self.tile_index,
            self.schema(),
            self.unnest.options().clone(),
        )?))
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }
    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Unknown
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.unnest.execute(partition, context)
    }
    fn metrics(&self) -> Option<MetricsSet> {
        let mut metrics = self.lists.metrics.clone_inner();
        if let Some(unnest) = self.unnest.metrics() {
            for metric in unnest.iter() {
                metrics.push(Arc::clone(metric));
            }
        }
        Some(metrics)
    }
}

/// Private adapter: one input row becomes one or more short list values. Its
/// projection describes the column mapping for properties only; scalar RS_Tile
/// is never evaluated through that projection.
#[derive(Debug)]
struct TileListExec {
    projection: ProjectionExec,
    tile_args: Vec<usize>,
    arg_types: Vec<SedonaType>,
    passthrough: Vec<Option<usize>>,
    cache: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl TileListExec {
    fn try_new(
        input: Arc<dyn ExecutionPlan>,
        layout: Vec<ProjectionExpr>,
        tile_args: Vec<usize>,
        arg_types: Vec<SedonaType>,
        passthrough: Vec<Option<usize>>,
    ) -> Result<Self> {
        let projection = ProjectionExec::try_new(layout, input)?;
        let eq = projection
            .properties()
            .eq_properties
            .clone()
            .with_constraints(Constraints::default());
        let cache = Arc::new(
            projection
                .properties()
                .as_ref()
                .clone()
                .with_eq_properties(eq),
        );
        Ok(Self {
            projection,
            tile_args,
            arg_types,
            passthrough,
            cache,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
    fn with_input(&self, input: Arc<dyn ExecutionPlan>) -> Result<Self> {
        Self::try_new(
            input,
            self.projection.expr().to_vec(),
            self.tile_args.clone(),
            self.arg_types.clone(),
            self.passthrough.clone(),
        )
    }
}

impl DisplayAs for TileListExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TileListExec")
    }
}

impl ExecutionPlan for TileListExec {
    fn name(&self) -> &str {
        "TileListExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![self.projection.input()]
    }
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("TileListExec expects one input");
        }
        Ok(Arc::new(self.with_input(children.remove(0))?))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let max_tiles = context.session_config().batch_size().max(1);
        let bytes = context
            .session_config()
            .options()
            .extensions
            .get::<SedonaOptions>()
            .map(|opts| opts.raster.max_batch_bytes)
            .unwrap_or(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let max_bytes = if bytes == 0 { u64::MAX } else { bytes as u64 };
        let input = self.projection.input().execute(partition, context)?;
        let generated = MetricBuilder::new(&self.metrics).counter("tiles_generated", partition);
        let schema = self.schema();
        let tile_args = self.tile_args.clone();
        let arg_types = self.arg_types.clone();
        let passthrough = self.passthrough.clone();
        let cursor = TileCursor {
            next: 0,
            total: 0,
            max_tiles,
            max_bytes,
        };
        let stream = futures::stream::try_unfold(
            (input, None::<RecordBatch>, 0usize, cursor),
            move |(mut input, mut pending, mut row, mut cursor)| {
                let schema = Arc::clone(&schema);
                let tile_args = tile_args.clone();
                let arg_types = arg_types.clone();
                let passthrough = passthrough.clone();
                let generated = generated.clone();
                async move {
                    while pending.as_ref().is_none_or(|batch| row == batch.num_rows()) {
                        pending = input.try_next().await?;
                        row = 0;
                        if pending.is_none() {
                            return Ok(None);
                        }
                    }
                    let batch = pending.as_ref().unwrap();
                    let args = tile_args
                        .iter()
                        .map(|&i| ColumnarValue::Array(batch.column(i).slice(row, 1)))
                        .collect::<Vec<_>>();
                    let tiles = tile_batch(&arg_types, &args, &mut cursor)?.into_array(1)?;
                    let list = datafusion_common::cast::as_list_array(&tiles)?;
                    generated.add(list.values().len());
                    let columns = passthrough
                        .iter()
                        .map(|index| match index {
                            Some(i) => batch.column(*i).slice(row, 1),
                            None => Arc::clone(&tiles),
                        })
                        .collect();
                    let output = RecordBatch::try_new(schema, columns)?;
                    if cursor.next == cursor.total {
                        row += 1;
                        cursor.next = 0;
                        cursor.total = 0;
                    }
                    Ok(Some((output, (input, pending, row, cursor))))
                }
            },
        );
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream.boxed(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int64Array};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::arrow::compute::concat_batches;
    use datafusion::execution::context::{SessionConfig, SessionContext};
    use datafusion::execution::session_state::SessionStateBuilder;
    use datafusion_physical_plan::{collect, displayable};
    use sedona_common::option::RasterOptions;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::raster_spec::{RasterSpec, raster_array};

    fn context(
        optimized: bool,
        rows: usize,
        bytes: usize,
        specs: Vec<Option<RasterSpec>>,
    ) -> SessionContext {
        let config = SessionConfig::new()
            .with_batch_size(rows)
            .with_target_partitions(1)
            .with_option_extension(SedonaOptions {
                raster: RasterOptions {
                    max_batch_bytes: bytes,
                    ..Default::default()
                },
                ..Default::default()
            });
        let mut state = SessionStateBuilder::new()
            .with_default_features()
            .with_config(config);
        if optimized {
            state = state.with_physical_optimizer_rule(Arc::new(RasterTileRule));
        }
        let ctx = SessionContext::new_with_state(state.build());
        ctx.register_udf(crate::rs_tile_udf().into());
        let rasters = raster_array(specs);
        let ids = Int64Array::from_iter_values(0..rasters.len() as i64);
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            SedonaType::Raster.to_storage_field("rast", true).unwrap(),
        ]));
        ctx.register_batch(
            "rasters",
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(rasters)]).unwrap(),
        )
        .unwrap();
        ctx
    }

    fn source() -> RasterSpec {
        RasterSpec::d2(5, 3)
            .crs(None)
            .transform([10.0, 2.0, 0.3, 20.0, 0.2, -2.0])
            .band(BandDataType::UInt8)
            .band(BandDataType::Int16)
    }

    async fn plan(ctx: &SessionContext, sql: &str) -> Arc<dyn ExecutionPlan> {
        ctx.sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn all_overloads_match_scalar_tiles() {
        // Byte budget forces one tile even though the row limit allows three.
        for args in [
            "2, 2",
            "2, 2, false",
            "2, 2, true, 7",
            "make_array(2, 1, 2), 2, 2",
            "make_array(2), 2, 2, true",
            "make_array(2, 1), 2, 2, true, 7",
            "CASE WHEN id = 0 THEN 1 ELSE 3 END, 2",
            "CAST(NULL AS BIGINT), 2",
            "2, 2, CAST(NULL AS BOOLEAN)",
            "2, 2, true, CAST(NULL AS DOUBLE)",
            "CAST(NULL AS BIGINT[]), 2, 2",
            "CAST(make_array() AS BIGINT[]), 2, 2",
        ] {
            let specs = vec![Some(source()), None, Some(source())];
            let plain = context(false, 3, 12, specs.clone());
            let optimized = context(true, 3, 12, specs);
            let sql = format!(
                "SELECT id, id + 100 AS other, UNNEST(RS_Tile(rast, {args})) AS t FROM rasters"
            );
            let original = plan(&plain, &sql).await;
            let rewritten = plan(&optimized, &sql).await;
            assert!(
                displayable(rewritten.as_ref())
                    .indent(true)
                    .to_string()
                    .contains("RasterTileExec"),
                "{args}"
            );
            let schema = original.schema();
            let expected = collect(original, plain.task_ctx()).await.unwrap();
            let actual = collect(rewritten, optimized.task_ctx()).await.unwrap();
            let expected = concat_batches(&schema, &expected).unwrap();
            let actual = concat_batches(&schema, &actual).unwrap();
            assert_eq!(actual, expected, "{args}");
        }
    }

    #[tokio::test]
    async fn stacked_planes_and_empty_rasters_match() {
        for spec in [
            RasterSpec::nd(&["time", "y", "x"], &[2, 3, 5])
                .crs(None)
                .band(BandDataType::Float32),
            RasterSpec::d2(0, 0).crs(None).band(BandDataType::UInt8),
        ] {
            let plain = context(false, 3, 32, vec![Some(spec.clone())]);
            let optimized = context(true, 3, 32, vec![Some(spec)]);
            let sql = "SELECT UNNEST(RS_Tile(rast, 2, 2, true, 0)) AS t FROM rasters";
            let p = plan(&plain, sql).await;
            let o = plan(&optimized, sql).await;
            let schema = o.schema();
            let expected = collect(p, plain.task_ctx()).await.unwrap();
            let actual = collect(o, optimized.task_ctx()).await.unwrap();
            assert_eq!(
                concat_batches(&schema, &actual).unwrap(),
                concat_batches(&schema, &expected).unwrap()
            );
        }
    }

    #[tokio::test]
    async fn output_is_bounded_and_stops_when_dropped() {
        let ctx = context(true, 100, 12, vec![Some(source())]);
        let p = plan(
            &ctx,
            "SELECT UNNEST(RS_Tile(rast, 2, 2, true, 0)) AS t FROM rasters",
        )
        .await;
        let mut stream = p.execute(0, ctx.task_ctx()).unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(
            first.num_rows(),
            1,
            "two bands use 12 bytes per padded tile"
        );
        drop(stream);
        let mut generated = 0;
        p.apply(|node| {
            if let Some(metrics) = node.metrics()
                && let Some(metric) = metrics.sum_by_name("tiles_generated")
            {
                generated += metric.as_usize();
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        assert_eq!(
            generated, 1,
            "dropping the stream must leave the other five tiles unbuilt"
        );

        let ctx = context(true, 2, 0, vec![Some(source())]);
        let p = plan(&ctx, "SELECT UNNEST(RS_Tile(rast, 1, 1)) AS t FROM rasters").await;
        let batches = collect(p, ctx.task_ctx()).await.unwrap();
        assert!(batches.iter().all(|b| b.num_rows() <= 2));
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 15);
    }

    #[tokio::test]
    async fn limit_does_not_build_a_million_tile_list() {
        let ctx = context(
            true,
            2,
            0,
            vec![Some(
                RasterSpec::d2(1024, 1024)
                    .crs(None)
                    .band(BandDataType::UInt8),
            )],
        );
        let p = plan(
            &ctx,
            "SELECT UNNEST(RS_Tile(rast, 1, 1)) AS t FROM rasters LIMIT 1",
        )
        .await;
        let batches = collect(Arc::clone(&p), ctx.task_ctx()).await.unwrap();
        assert_eq!(batches.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        let mut generated = 0;
        p.apply(|node| {
            if let Some(metrics) = node.metrics()
                && let Some(metric) = metrics.sum_by_name("tiles_generated")
            {
                generated += metric.as_usize();
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .unwrap();
        assert_eq!(generated, 2, "only one bounded batch should be copied");
    }

    #[tokio::test]
    async fn scalar_lists_and_zipped_unnests_are_not_rewritten() {
        let ctx = context(true, 2, 12, vec![Some(source())]);
        for sql in [
            "SELECT RS_Tile(rast, 2, 2) FROM rasters",
            "SELECT UNNEST(RS_Tile(rast, 2, 2)), UNNEST(make_array(id, id + 1)) FROM rasters",
        ] {
            let p = plan(&ctx, sql).await;
            assert!(
                !displayable(p.as_ref())
                    .indent(true)
                    .to_string()
                    .contains("RasterTileExec")
            );
            collect(p, ctx.task_ctx()).await.unwrap();
        }
    }

    #[tokio::test]
    async fn invalid_arguments_still_error() {
        for args in [
            "0, 2",
            "2, -1",
            "make_array(3), 2, 2",
            "make_array(NULL), 2, 2",
            "2, 2, false, 0",
            "2, 2, true, 1000000",
        ] {
            for optimized in [false, true] {
                let ctx = context(optimized, 2, 12, vec![Some(source())]);
                let sql = format!("SELECT UNNEST(RS_Tile(rast, {args})) FROM rasters");
                let result = ctx.sql(&sql).await;
                let result = match result {
                    Ok(df) => df.collect().await,
                    Err(e) => Err(e),
                };
                assert!(result.is_err(), "{args}, optimized={optimized}");
            }
        }
    }
}
