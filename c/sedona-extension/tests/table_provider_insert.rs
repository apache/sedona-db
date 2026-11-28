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

use arrow_array::{Int64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::Result;
use sedona_extension::runtime::RuntimeHandle;
use sedona_extension::table_provider::{ExportedTableProvider, ImportedTableProvider};

fn runtime() -> Arc<RuntimeHandle> {
    Arc::new(RuntimeHandle::new(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
    ))
}

fn check_insert(select: &str, expected: &[i64], use_async: bool) -> Result<()> {
    let consumer_runtime = runtime();
    let producer_runtime = consumer_runtime.clone();
    consumer_runtime.block_on(async {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        let target = Arc::new(MemTable::try_new(schema.clone(), vec![vec![]])?);
        let producer = SessionContext::new();
        producer.register_table("target", target.clone())?;
        let exported = ExportedTableProvider::new(
            target,
            Arc::new(producer.state()),
            producer_runtime.clone(),
        );
        let imported = ImportedTableProvider::try_new(exported.into(), consumer_runtime.clone())?
            .with_async_execution(use_async);
        let consumer =
            SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));
        consumer.register_table("target", Arc::new(imported))?;
        let partitions = [vec![1, 2, 3], vec![4, 5, 6]]
            .into_iter()
            .map(|values| {
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))])
                    .map(|batch| vec![batch])
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        consumer.register_table("source", Arc::new(MemTable::try_new(schema, partitions)?))?;

        let insert = consumer
            .sql(&format!("INSERT INTO target {select}"))
            .await?
            .create_physical_plan()
            .await?;
        // Planning must not modify the destination.
        let before = producer
            .sql("SELECT * FROM target")
            .await?
            .collect()
            .await?;
        assert_eq!(before.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);

        let counts = datafusion_physical_plan::collect(insert, consumer.task_ctx()).await?;
        assert_eq!(counts.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        let count = counts[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(count.value(0), expected.len() as u64);
        let result = producer
            .sql("SELECT id FROM target ORDER BY id")
            .await?
            .collect()
            .await?;
        let actual: Vec<i64> = result
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(actual, expected, "{select}, async={use_async}");
        Ok(())
    })
}

#[test]
fn insert_into_select_all_partitions() -> Result<()> {
    for use_async in [false, true] {
        check_insert("SELECT id FROM source", &[1, 2, 3, 4, 5, 6], use_async)?;
    }
    Ok(())
}

#[test]
fn insert_into_select_aggregate() -> Result<()> {
    for use_async in [false, true] {
        check_insert("SELECT SUM(id) AS id FROM source", &[21], use_async)?;
    }
    Ok(())
}

#[test]
fn insert_into_select_ordered_limit() -> Result<()> {
    for use_async in [false, true] {
        check_insert(
            "SELECT id FROM source ORDER BY id DESC LIMIT 2",
            &[5, 6],
            use_async,
        )?;
    }
    Ok(())
}
