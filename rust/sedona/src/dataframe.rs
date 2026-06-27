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

use arrow_array::RecordBatch;
use async_trait::async_trait;
use datafusion::{
    dataframe::DataFrameWriteOptions,
    datasource::file_format::format_as_file_type,
    error::{DataFusionError, Result},
    prelude::DataFrame,
};
use datafusion_common::not_impl_err;
use datafusion_expr::{dml::InsertOp, LogicalPlan, LogicalPlanBuilder, SortExpr};
use sedona_geoparquet::{format::GeoParquetFormatFactory, options::TableGeoParquetOptions};

use crate::{
    context::SedonaContext,
    show::{show_batches, DisplayTableOptions},
};

/// Sedona-specific [`DataFrame`] actions
///
/// This trait, implemented for [`DataFrame`], extends the DataFrame API to make it
/// ergonomic to work with dataframes that contain geometry columns. Currently these
/// are limited to output functions, as geometry columns currently require special
/// handling when written or exported to an external system.
#[async_trait]
pub trait SedonaDataFrame {
    /// Build a table of the first `limit` results in this DataFrame
    ///
    /// This will limit and execute the query and build a table using [show_batches].
    async fn show_sedona<'a>(
        self,
        ctx: &SedonaContext,
        limit: Option<usize>,
        options: DisplayTableOptions<'a>,
    ) -> Result<String>;

    async fn write_geoparquet(
        self,
        ctx: &SedonaContext,
        path: &str,
        options: SedonaWriteOptions,
        writer_options: Option<TableGeoParquetOptions>,
    ) -> Result<Vec<RecordBatch>>;
}

#[async_trait]
impl SedonaDataFrame for DataFrame {
    async fn show_sedona<'a>(
        self,
        ctx: &SedonaContext,
        limit: Option<usize>,
        mut options: DisplayTableOptions<'a>,
    ) -> Result<String> {
        let df = if matches!(
            self.logical_plan(),
            LogicalPlan::Explain(_) | LogicalPlan::DescribeTable(_) | LogicalPlan::Analyze(_)
        ) {
            // Show multi-line output without truncation for plans like `EXPLAIN`
            options.max_row_height = usize::MAX;

            // We don't want to apply an additional .limit() to plans like `Explain`
            // as that will trigger an internal error: Unsupported logical plan: Explain must be root of the plan
            self
        } else {
            // Apply limit if specified
            self.limit(0, limit)?
        };

        let schema_without_qualifiers = df.schema().clone().strip_qualifiers();
        let schema = schema_without_qualifiers.as_arrow();
        let batches = df.collect().await?;
        let mut out = Vec::new();
        show_batches(ctx, &mut out, schema, batches, options)?;
        String::from_utf8(out).map_err(|e| DataFusionError::External(Box::new(e)))
    }

    async fn write_geoparquet(
        self,
        ctx: &SedonaContext,
        path: &str,
        options: SedonaWriteOptions,
        writer_options: Option<TableGeoParquetOptions>,
    ) -> Result<Vec<RecordBatch>, DataFusionError> {
        if options.insert_op != InsertOp::Append {
            return not_impl_err!(
                "{} is not implemented for DataFrame::write_geoparquet.",
                options.insert_op
            );
        }

        let format = if let Some(parquet_opts) = writer_options {
            Arc::new(GeoParquetFormatFactory::new_with_options(parquet_opts))
        } else {
            Arc::new(GeoParquetFormatFactory::new())
        };

        let file_type = format_as_file_type(format);

        let plan = if options.sort_by.is_empty() {
            self.into_unoptimized_plan()
        } else {
            LogicalPlanBuilder::from(self.into_unoptimized_plan())
                .sort(options.sort_by)?
                .build()?
        };

        let plan = LogicalPlanBuilder::copy_to(
            plan,
            path.into(),
            file_type,
            Default::default(),
            options.partition_by,
        )?
        .build()?;

        DataFrame::new(ctx.ctx.state(), plan).collect().await
    }
}

/// A Sedona-specific copy of [DataFrameWriteOptions]
///
/// This is needed because [DataFrameWriteOptions] has private fields, so we
/// can't use it in our interfaces. This object can be converted to a
/// [DataFrameWriteOptions] using `.into()`.
pub struct SedonaWriteOptions {
    /// Controls how new data should be written to the table, determining whether
    /// to append, overwrite, or replace existing data.
    pub insert_op: InsertOp,
    /// Controls if all partitions should be coalesced into a single output file
    /// Generally will have slower performance when set to true.
    pub single_file_output: bool,
    /// Sets which columns should be used for hive-style partitioned writes by name.
    /// Can be set to empty vec![] for non-partitioned writes.
    pub partition_by: Vec<String>,
    /// Sets which columns should be used for sorting the output by name.
    /// Can be set to empty vec![] for non-sorted writes.
    pub sort_by: Vec<SortExpr>,
}

impl From<SedonaWriteOptions> for DataFrameWriteOptions {
    fn from(value: SedonaWriteOptions) -> Self {
        DataFrameWriteOptions::new()
            .with_insert_operation(value.insert_op)
            .with_single_file_output(value.single_file_output)
            .with_partition_by(value.partition_by)
            .with_sort_by(value.sort_by)
    }
}

impl SedonaWriteOptions {
    /// Create a new SedonaWriteOptions with default values
    pub fn new() -> Self {
        SedonaWriteOptions {
            insert_op: InsertOp::Append,
            single_file_output: false,
            partition_by: vec![],
            sort_by: vec![],
        }
    }

    /// Set the insert operation
    pub fn with_insert_operation(mut self, insert_op: InsertOp) -> Self {
        self.insert_op = insert_op;
        self
    }

    /// Set the single_file_output value to true or false
    pub fn with_single_file_output(mut self, single_file_output: bool) -> Self {
        self.single_file_output = single_file_output;
        self
    }

    /// Sets the partition_by columns for output partitioning
    pub fn with_partition_by(mut self, partition_by: Vec<String>) -> Self {
        self.partition_by = partition_by;
        self
    }

    /// Sets the sort_by columns for output sorting
    pub fn with_sort_by(mut self, sort_by: Vec<SortExpr>) -> Self {
        self.sort_by = sort_by;
        self
    }
}

impl Default for SedonaWriteOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod test {
    use tempfile::tempdir;

    use crate::context::SedonaContext;

    use super::*;

    #[tokio::test]
    async fn show() {
        let ctx = SedonaContext::new();
        let tbl = ctx
            .sql("SELECT 1 as one")
            .await
            .unwrap()
            .show_sedona(&ctx, None, DisplayTableOptions::default())
            .await
            .unwrap();

        #[rustfmt::skip]
        assert_eq!(
            tbl.lines().collect::<Vec<_>>(),
            vec![
                "+-----+",
                "| one |",
                "+-----+",
                "|   1 |",
                "+-----+"
            ]
        );
    }

    #[tokio::test]
    async fn show_explain() {
        let ctx = SedonaContext::new();
        for limit in [None, Some(10)] {
            let tbl = ctx
                .sql("EXPLAIN SELECT 1 as one")
                .await
                .unwrap()
                .show_sedona(&ctx, limit, DisplayTableOptions::default())
                .await
                .unwrap();

            #[rustfmt::skip]
            assert_eq!(
                tbl.lines().collect::<Vec<_>>(),
                vec![
                    "+---------------+---------------------------------+",
                    "|   plan_type   |               plan              |",
                    "+---------------+---------------------------------+",
                    "| logical_plan  | Projection: Int64(1) AS one     |",
                    "|               |   EmptyRelation: rows=1         |",
                    "| physical_plan | ProjectionExec: expr=[1 as one] |",
                    "|               |   PlaceholderRowExec            |",
                    "|               |                                 |",
                    "+---------------+---------------------------------+",
                ]
            );
        }
    }

    #[tokio::test]
    async fn write_geoparquet() {
        let tmpdir = tempdir().unwrap();
        let tmp_parquet = tmpdir.path().join("tmp.parquet");
        let ctx = SedonaContext::new();
        ctx.sql("SELECT 1 as one")
            .await
            .unwrap()
            .write_parquet(
                &tmp_parquet.to_string_lossy(),
                DataFrameWriteOptions::default(),
                None,
            )
            .await
            .unwrap();
    }
}
