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
use std::{any::Any, fmt::Debug, sync::Arc};

use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::TableFunctionImpl;
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::expressions::Column;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{Partitioning, PhysicalExpr, SendableRecordBatchStream};
use datafusion::{
    catalog::{Session, TableProvider},
    common::Result,
    datasource::TableType,
    physical_expr::EquivalenceProperties,
    physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties},
    prelude::Expr,
};
use datafusion_common::{plan_err, DataFusionError, ScalarValue};
use geo_types::Rect;
use sedona_geometry::types::GeometryTypeId;
use sedona_testing::datagen::RandomPartitionedDataBuilder;
use serde::{Deserialize, Serialize};

/// A table function that refers to a table of random geometries
///
/// This table function accepts one argument, which is a JSON-specified
/// key/value version of the options that can be specified to the
/// [RandomPartitionedDataBuilder]. See the [RandomGeometryProvider]
/// for more details.
#[derive(Debug, Default)]
pub struct RandomGeometryFunction {}

impl TableFunctionImpl for RandomGeometryFunction {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        if exprs.len() != 1 {
            return plan_err!(
                "sd_random_geometry() expected 1 argument but got {}",
                exprs.len()
            );
        }

        if let Expr::Literal(scalar, _) = &exprs[0] {
            if let ScalarValue::Utf8(Some(options_str)) = scalar.cast_to(&DataType::Utf8)? {
                let builder = RandomPartitionedDataBuilder::new();
                return Ok(Arc::new(RandomGeometryProvider::try_new(
                    builder,
                    Some(options_str),
                )?));
            }
        }

        plan_err!(
            "Expected literal in sd_random_geometry() but got {}",
            &exprs[0]
        )
    }
}

/// A table provider that generates random geometries using the [RandomPartitionedDataBuilder]
///
/// This table provider wraps the [RandomPartitionedDataBuilder], allowing it to be used in
/// queries. The random generation may not be stable between versions, so if using to generate
/// a fixture, save to a file and use the file as test or benchmark data.
#[derive(Debug)]
pub struct RandomGeometryProvider {
    builder: RandomPartitionedDataBuilder,
    num_partitions: usize,
    rows_per_batch: usize,
    target_rows: usize,
}

impl RandomGeometryProvider {
    /// Create a new [RandomGeometryProvider] from an existing builder and options
    ///
    /// The provided options override any options specified in the existing builder.
    /// Parameters that affect the number of partitions or rows have different defaults that
    /// always override that of the builder (unless manually specified in the option). This
    /// reflects the SQL use case where often the parameter that needs tweaking is the number
    /// of total rows, which in this case can be set with `{"target_rows": 2048}`. The number
    /// of total rows will always be a multiple of the batch size times the number of partitions,
    /// whose defaults to 1024 and 1, respectively.
    ///
    /// See the [RandomGeometryProvider] for other parameters that can be specified via JSON.
    fn try_new(mut builder: RandomPartitionedDataBuilder, options: Option<String>) -> Result<Self> {
        let options = if let Some(options_str) = options {
            match serde_json::from_str::<RandomGeometryFunctionOptions>(&options_str) {
                Ok(options) => Some(options),
                Err(e) => {
                    return plan_err!("Failed to parse options: {e}\nOption were: {options_str}")
                }
            }
        } else {
            None
        };

        let mut num_partitions = 1;
        let mut rows_per_batch = 1024;
        let mut target_rows = 1024;

        if let Some(options) = options {
            if let Some(opt_num_partitions) = options.num_partitions {
                num_partitions = opt_num_partitions;
            }

            if let Some(opt_rows_per_batch) = options.rows_per_batch {
                rows_per_batch = opt_rows_per_batch;
            }

            if let Some(opt_target_rows) = options.target_rows {
                target_rows = opt_target_rows;
            }
            if let Some(seed) = options.seed {
                builder = builder.seed(seed);
            }
            if let Some(null_rate) = options.null_rate {
                builder = builder.null_rate(null_rate);
            }
            if let Some(geom_type) = options.geom_type {
                builder = builder.geometry_type(geom_type);
            }
            if let Some(bounds) = options.bounds {
                let bounds = Rect::new((bounds.0, bounds.1), (bounds.2, bounds.3));
                builder = builder.bounds(bounds);
            }
            if let Some(size_range) = options.size_range {
                builder = builder.size_range(size_range);
            }
            if let Some(vertices_range) = options.vertices_per_linestring_range {
                builder = builder.vertices_per_linestring_range(vertices_range);
            }
            if let Some(empty_rate) = options.empty_rate {
                builder = builder.empty_rate(empty_rate);
            }
            if let Some(hole_rate) = options.polygon_hole_rate {
                builder = builder.polygon_hole_rate(hole_rate);
            }
            if let Some(parts_range) = options.num_parts_range {
                builder = builder.num_parts_range(parts_range);
            }
        }

        Ok(RandomGeometryProvider {
            builder,
            num_partitions,
            rows_per_batch,
            target_rows,
        })
    }
}

#[async_trait]
impl TableProvider for RandomGeometryProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.builder.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::View
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let builder = builder_with_partition_sizes(
            self.builder.clone(),
            self.rows_per_batch,
            self.num_partitions,
            self.target_rows,
        );

        let exec = Arc::new(RandomGeometryExec::new(builder));

        // We're required to handle the projection or we'll get an execution error
        if let Some(projection) = projection {
            let schema = self.schema();
            let exprs: Vec<_> = projection
                .iter()
                .map(|index| -> (Arc<dyn PhysicalExpr>, String) {
                    let name = schema.field(*index).name();
                    (Arc::new(Column::new(name, *index)), name.clone())
                })
                .collect();
            Ok(Arc::new(ProjectionExec::try_new(exprs, exec)?))
        } else {
            Ok(exec)
        }
    }
}

#[derive(Debug)]
struct RandomGeometryExec {
    builder: RandomPartitionedDataBuilder,
    properties: PlanProperties,
}

impl RandomGeometryExec {
    pub fn new(builder: RandomPartitionedDataBuilder) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(builder.schema().clone()),
            Partitioning::UnknownPartitioning(builder.num_partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            builder,
            properties,
        }
    }
}

impl DisplayAs for RandomGeometryExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RecordBatchReaderExec")
    }
}

impl ExecutionPlan for RandomGeometryExec {
    fn name(&self) -> &str {
        "RandomGeometryExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.builder.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let rng = RandomPartitionedDataBuilder::default_rng(self.builder.seed + partition as u64);
        let reader = self.builder.partition_reader(rng, partition);
        let iter = reader.map(|item| match item {
            Ok(batch) => Ok(batch),
            Err(e) => Err(DataFusionError::from(e)),
        });

        let stream = Box::pin(futures::stream::iter(iter));
        let record_batch_stream = RecordBatchStreamAdapter::new(self.schema(), stream);
        Ok(Box::pin(record_batch_stream))
    }
}

/// These options only exist as a mechanism to deserialize JSON options
///
/// See the [RandomPartitionedDataBuilder] for definitive documentation of these
/// values.
#[derive(Serialize, Deserialize, Default)]
struct RandomGeometryFunctionOptions {
    num_partitions: Option<usize>,
    rows_per_batch: Option<usize>,
    target_rows: Option<usize>,
    seed: Option<u64>,
    null_rate: Option<f64>,
    geom_type: Option<GeometryTypeId>,
    bounds: Option<(f64, f64, f64, f64)>,
    size_range: Option<(f64, f64)>,
    vertices_per_linestring_range: Option<(usize, usize)>,
    empty_rate: Option<f64>,
    polygon_hole_rate: Option<f64>,
    num_parts_range: Option<(usize, usize)>,
}

fn builder_with_partition_sizes(
    builder: RandomPartitionedDataBuilder,
    batch_size: usize,
    partitions: usize,
    target_rows: usize,
) -> RandomPartitionedDataBuilder {
    let rows_for_one_batch_per_partition = batch_size * partitions;
    let batches_per_partition = if target_rows % rows_for_one_batch_per_partition == 0 {
        target_rows / rows_for_one_batch_per_partition
    } else {
        target_rows / rows_for_one_batch_per_partition + 1
    };

    builder
        .rows_per_batch(batch_size)
        .num_partitions(partitions)
        .batches_per_partition(batches_per_partition)
}

#[cfg(test)]
mod test {

    use datafusion::prelude::SessionContext;

    use super::*;

    #[tokio::test]
    async fn provider() {
        let ctx = SessionContext::new();

        // Generate batches using the builder
        let builder = RandomPartitionedDataBuilder::new()
            .num_partitions(4)
            .batches_per_partition(2)
            .rows_per_batch(1024);
        let (expected_schema, expected_results) = builder.build().unwrap();
        assert_eq!(expected_results.len(), 4);
        assert_eq!(expected_results[0].len(), 2);
        assert_eq!(expected_results[0][0].num_rows(), 1024);

        // Generate batches using the provider
        let provider = RandomGeometryProvider::try_new(
            builder,
            Some(
                r#"{"target_rows": 8192, "num_partitions": 4, "rows_per_batch": 1024}"#.to_string(),
            ),
        )
        .unwrap();
        let df = ctx.read_table(Arc::new(provider)).unwrap();

        assert_eq!(df.clone().count().await.unwrap(), 8192);

        assert_eq!(*df.schema().as_arrow(), *expected_schema);
        let actual_results = df.collect_partitioned().await.unwrap();
        assert_eq!(actual_results, expected_results);

        // Generate batches using the table function
        ctx.register_udtf("sd_random_geometry", Arc::new(RandomGeometryFunction {}));
        let df = ctx
            .sql(r#"
        SELECT * FROM sd_random_geometry('{"target_rows": 8192, "num_partitions": 4, "rows_per_batch": 1024}')
            "#)
            .await
            .unwrap();

        assert_eq!(*df.schema().as_arrow(), *expected_schema);
        let actual_results = df.collect_partitioned().await.unwrap();
        assert_eq!(actual_results, expected_results);
    }

    #[tokio::test]
    async fn provider_rows() {
        let ctx = SessionContext::new();

        // When the batch size * num_partitions fits evenly into the target rows, we should get
        // an exact number of rows
        let provider = RandomGeometryProvider::try_new(
            RandomPartitionedDataBuilder::new(),
            Some(
                r#"{"target_rows": 8192, "num_partitions": 2, "rows_per_batch": 1024}"#.to_string(),
            ),
        )
        .unwrap();
        let df = ctx.read_table(Arc::new(provider)).unwrap();
        assert_eq!(df.count().await.unwrap(), 8192);

        // If the batch size * num_partitions doesn't fit evenly, we should have more rows
        // than target_rows
        let provider = RandomGeometryProvider::try_new(
            RandomPartitionedDataBuilder::new(),
            Some(
                r#"{"target_rows": 9000, "num_partitions": 2, "rows_per_batch": 1024}"#.to_string(),
            ),
        )
        .unwrap();
        let df = ctx.read_table(Arc::new(provider)).unwrap();
        assert_eq!(df.count().await.unwrap(), 8192 + (2 * 1024));
    }
}
