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

use std::{collections::HashMap, sync::Arc};

use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    config::TableOptions,
    datasource::listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    execution::{options::ReadOptions, SessionState},
    prelude::{SessionConfig, SessionContext},
};
use datafusion_common::{exec_err, Result};

use crate::{format::RecordBatchReaderFormat, spec::RecordBatchReaderFormatSpec};

#[derive(Debug, Clone)]
pub struct RecordBatchReaderTableOptions {
    spec: Arc<dyn RecordBatchReaderFormatSpec>,
    table_partition_cols: Vec<(String, DataType)>,
    options: HashMap<String, String>,
}

impl RecordBatchReaderTableOptions {
    pub fn new(spec: Arc<dyn RecordBatchReaderFormatSpec>) -> Self {
        Self {
            spec,
            table_partition_cols: Vec::new(),
            options: HashMap::new(),
        }
    }
}

#[async_trait]
impl ReadOptions<'_> for RecordBatchReaderTableOptions {
    fn to_listing_options(
        &self,
        config: &SessionConfig,
        table_options: TableOptions,
    ) -> ListingOptions {
        let format = RecordBatchReaderFormat::new(self.spec.with_table_options(&table_options));
        ListingOptions::new(Arc::new(format))
            .with_file_extension(self.spec.extension())
            .with_table_partition_cols(self.table_partition_cols.clone())
            .with_session_config_options(config)
    }

    async fn get_resolved_schema(
        &self,
        config: &SessionConfig,
        state: SessionState,
        table_path: ListingTableUrl,
    ) -> Result<SchemaRef> {
        self.to_listing_options(config, state.default_table_options())
            .infer_schema(&state, &table_path)
            .await
    }
}

/// Create a [ListingTable] of GeoParquet (or normal Parquet) files
pub async fn generic_listing_table(
    context: &SessionContext,
    table_paths: Vec<ListingTableUrl>,
    mut options: RecordBatchReaderTableOptions,
) -> Result<ListingTable> {
    let session_config = context.copied_config();

    options.spec = options.spec.with_options(&options.options)?;
    let listing_options =
        options.to_listing_options(&session_config, context.copied_table_options());

    let option_extension = listing_options.file_extension.clone();

    if table_paths.is_empty() {
        return exec_err!("No table paths were provided");
    }

    // check if the file extension matches the expected extension if one is provided
    if !option_extension.is_empty() {
        for path in &table_paths {
            let file_path = path.as_str();
            if !file_path.ends_with(option_extension.clone().as_str()) && !path.is_collection() {
                return exec_err!(
                        "File path '{file_path}' does not match the expected extension '{option_extension}'"
                    );
            }
        }
    }

    let resolved_schema = options
        .get_resolved_schema(&session_config, context.state(), table_paths[0].clone())
        .await?;
    let config = ListingTableConfig::new_with_multi_paths(table_paths)
        .with_listing_options(listing_options)
        .with_schema(resolved_schema);

    ListingTable::try_new(config)
}
