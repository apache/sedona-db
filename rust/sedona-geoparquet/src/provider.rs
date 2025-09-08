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

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::{
    config::TableOptions,
    datasource::{
        file_format::parquet::ParquetFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    execution::{options::ReadOptions, SessionState},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_common::{exec_err, Result};

use crate::format::GeoParquetFormat;

/// Create a [ListingTable] of GeoParquet (or normal Parquet) files
///
/// Because [ListingTable] implements `TableProvider`, this can be used to
/// implement geo-aware Parquet reading with interfaces that are otherwise
/// hard-coded to the built-in Parquet reader.
pub async fn geoparquet_listing_table(
    context: &SessionContext,
    table_paths: Vec<ListingTableUrl>,
    options: GeoParquetReadOptions<'_>,
) -> Result<ListingTable> {
    let session_config = context.copied_config();
    let listing_options =
        options.to_listing_options(&session_config, context.copied_table_options());

    let option_extension = listing_options.file_extension.clone();

    if table_paths.is_empty() {
        return exec_err!("No table paths were provided");
    }

    // check if the file extension matches the expected extension
    for path in &table_paths {
        let file_path = path.as_str();
        if !file_path.ends_with(option_extension.clone().as_str()) && !path.is_collection() {
            return exec_err!(
                    "File path '{file_path}' does not match the expected extension '{option_extension}'"
                );
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

/// GeoParquet read options
///
/// Currently is just a wrapper around [ParquetReadOptions] that sets the
/// correct file format when creating [ListingOptions].
#[derive(Default, Clone)]
pub struct GeoParquetReadOptions<'a> {
    inner: ParquetReadOptions<'a>,
}

impl GeoParquetReadOptions<'_> {
    /// Create a new GeoParquetReadOptions with default values
    pub fn new() -> Self {
        Default::default()
    }
}

#[async_trait]
impl ReadOptions<'_> for GeoParquetReadOptions<'_> {
    fn to_listing_options(
        &self,
        config: &SessionConfig,
        table_options: TableOptions,
    ) -> ListingOptions {
        let mut options = self.inner.to_listing_options(config, table_options);
        if let Some(parquet_format) = options.format.as_any().downcast_ref::<ParquetFormat>() {
            options.format = Arc::new(GeoParquetFormat::new(parquet_format));
            return options;
        }

        unreachable!("GeoParquetReadOptions with non-ParquetFormat ListingOptions");
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

#[cfg(test)]
mod test {

    use arrow_schema::DataType;
    use sedona_schema::{
        crs::lnglat,
        datatypes::{Edges, SedonaType},
    };
    use sedona_testing::data::geoarrow_data_dir;

    use super::*;

    #[tokio::test]
    async fn listing_table() {
        let ctx = SessionContext::new();
        let data_dir = geoarrow_data_dir().unwrap();
        let tab = geoparquet_listing_table(
            &ctx,
            vec![
                ListingTableUrl::parse(format!("{data_dir}/example/files/*_geo.parquet")).unwrap(),
            ],
            GeoParquetReadOptions::default(),
        )
        .await
        .unwrap();

        let df = ctx.read_table(Arc::new(tab)).unwrap();

        let sedona_types: Result<Vec<_>> = df
            .schema()
            .as_arrow()
            .fields()
            .iter()
            .map(|f| SedonaType::from_storage_field(f))
            .collect();
        let sedona_types = sedona_types.unwrap();
        assert_eq!(sedona_types.len(), 2);
        assert_eq!(sedona_types[0], SedonaType::Arrow(DataType::Utf8View));
        assert_eq!(
            sedona_types[1],
            SedonaType::WkbView(Edges::Planar, lnglat())
        );

        // Make sure all the rows show up!
        let batches = df.collect().await.unwrap();
        let mut total_size = 0;
        for batch in batches {
            total_size += batch.num_rows();
        }
        assert_eq!(total_size, 244);
    }

    #[tokio::test]
    async fn listing_table_errors() {
        let ctx = SessionContext::new();
        let err = geoparquet_listing_table(
            &ctx,
            Vec::<ListingTableUrl>::new(),
            GeoParquetReadOptions::default(),
        )
        .await
        .unwrap_err();
        assert_eq!(err.message(), "No table paths were provided");

        let err = geoparquet_listing_table(
            &ctx,
            vec![ListingTableUrl::parse("foofy.wrongextension").unwrap()],
            GeoParquetReadOptions::default(),
        )
        .await
        .unwrap_err();
        assert!(err
            .message()
            .ends_with("does not match the expected extension '.parquet'"));
    }
}
