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

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::datasource::{MemTable, TableType};
use datafusion::error::Result;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_expr::{Expr, JoinType};
use sedona::context::{SedonaContext, SedonaDataFrame};
use sedona::show::{show_batches, DisplayTableOptions};

#[derive(Debug)]
struct PreviewTable(MemTable);

#[async_trait]
impl TableProvider for PreviewTable {
    fn schema(&self) -> SchemaRef {
        self.0.schema()
    }

    fn table_type(&self) -> TableType {
        self.0.table_type()
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Only the fourth column is known to be hidden from the schema alone.
        assert_eq!(
            projection.map(Vec::as_slice),
            Some([0, 1, 2, 4, 5].as_slice())
        );
        self.0.scan(state, projection, filters, limit).await
    }
}

#[tokio::test]
async fn show_projects_hidden_columns() -> Result<()> {
    let text: ArrayRef = Arc::new(StringArray::from(vec!["hello", "world"]));
    let numbers: ArrayRef = Arc::new(Int64Array::from(vec![i64::MIN, 0]));
    let batch = RecordBatch::try_from_iter([
        ("first_col", text.clone()),
        ("second_col", numbers),
        ("third_col", text.clone()),
        ("fourth_col", text.clone()),
        ("fifth_col", text.clone()),
        ("last_col", text),
    ])?;
    let ctx = SedonaContext::new();
    let source = MemTable::try_new(batch.schema(), vec![vec![batch.clone()]])?;
    ctx.ctx
        .register_table("preview", Arc::new(PreviewTable(source)))?;
    let mut options = DisplayTableOptions::new();
    options.table_width = 59;

    // A wide number triggers more omissions after projection. The
    // cutoff and the resulting table should match the full batch path.
    let mut expected = Vec::new();
    show_batches(
        &ctx,
        &mut expected,
        &batch.schema(),
        vec![batch.slice(0, 1)],
        options.clone(),
    )?;
    let actual = ctx
        .ctx
        .table("preview")
        .await?
        .show_sedona(&ctx, Some(1), options)
        .await?;
    assert_eq!(actual, String::from_utf8(expected).unwrap());
    Ok(())
}

#[tokio::test]
async fn show_joins_with_repeated_column_names() -> Result<()> {
    let ctx = SedonaContext::new();
    let padding = (2..20)
        .map(|i| format!(", {i} AS c{i}"))
        .collect::<String>();
    for right_name in ["id", "lhs.id"] {
        let left = ctx
            .ctx
            .sql(&format!("SELECT 1 AS id{padding}"))
            .await?
            .alias("lhs")?;
        let right = ctx
            .ctx
            .sql(&format!("SELECT 20 AS \"{right_name}\""))
            .await?;
        let right = if right_name == "id" {
            right.alias("rhs")?
        } else {
            right
        };
        let joined = left.join(right, JoinType::Inner, &[], &[], None)?;
        let mut expected = Vec::new();
        let options = DisplayTableOptions::new();
        show_batches(
            &ctx,
            &mut expected,
            joined.schema().as_arrow(),
            joined.clone().collect().await?,
            options.clone(),
        )?;
        let actual = joined.show_sedona(&ctx, Some(1), options).await?;
        assert_eq!(actual, String::from_utf8(expected).unwrap());
    }
    Ok(())
}
