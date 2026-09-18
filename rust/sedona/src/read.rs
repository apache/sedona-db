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

//! Shared implementation for reading URLs through registered file formats.

use std::{collections::HashMap, sync::Arc};

use arrow_schema::DataType;
use datafusion::{
    catalog::TableProvider,
    datasource::listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    prelude::SessionContext,
};
use datafusion_common::{exec_err, plan_err, DataFusionError, Result};

use crate::object_storage::ensure_object_store_registered_with_options;
use sedona_datasource::{format::ExternalFormatFactory, provider::external_table};
use sedona_geoparquet::provider::GeoParquetReadOptions;

/// Resolve paths and options into a table provider using the session's registered
/// [`FileFormatFactory`](datafusion::datasource::file_format::FileFormatFactory).
///
/// This is the common implementation behind [`SedonaContext::read`](crate::context::SedonaContext::read)
/// and SQL URL tables. External formats that opt into
/// [`list_single_object`](sedona_datasource::spec::ExternalFormatSpec::list_single_object)
/// are routed through the single-object provider; all other formats use a
/// [`ListingTable`].
pub(crate) async fn read_provider(
    context: &SessionContext,
    table_paths: Vec<ListingTableUrl>,
    options: &HashMap<String, String>,
    format: Option<&str>,
    partitioning: Option<Vec<(String, DataType)>>,
) -> Result<Arc<dyn TableProvider>> {
    if table_paths.is_empty() {
        return exec_err!("No table paths were provided");
    }

    // Preserve the detailed cloud-option validation (including typo
    // suggestions) provided by the existing read API for every format.
    GeoParquetReadOptions::from_table_options(options.clone()).map_err(DataFusionError::Plan)?;

    for path in &table_paths {
        ensure_object_store_registered_with_options(
            &mut context.state(),
            path.as_str(),
            Some(options),
        )
        .await?;
    }

    let (format, compression, listing_extension) = resolve_format(&table_paths, format)?;
    let state = context.state();
    let factory = state.get_file_format_factory(&format).ok_or_else(|| {
        datafusion_common::plan_datafusion_err!("No format registered for extension '{format}'")
    })?;

    if format == "csv" {
        if let Some(delimiter) = options
            .get("delimiter")
            .or_else(|| options.get("format.delimiter"))
        {
            if delimiter.len() != 1 {
                return plan_err!("CSV delimiter must be a single byte, got {delimiter:?}");
            }
        }
    }

    let reader_options = reader_options(options);
    if let Some(external) = factory.downcast_ref::<ExternalFormatFactory>() {
        let spec = external.spec().with_options(&reader_options)?;
        return external_table(spec, context, table_paths, false, partitioning).await;
    }

    let mut factory_options = HashMap::with_capacity(reader_options.len() + 1);
    for (key, value) in reader_options {
        // Dotted non-format keys are connection/table options. Object-store
        // registration consumed the known ones above; retain the historical
        // behavior of ignoring other connection options here rather than
        // passing them to a built-in file-format parser.
        if key.contains('.') && !key.starts_with("format.") {
            continue;
        }
        let key = if key.starts_with("format.") {
            key
        } else {
            format!("format.{key}")
        };
        factory_options.insert(key, value);
    }
    if let Some(compression) = &compression {
        factory_options
            .entry("format.compression".to_string())
            .or_insert_with(|| compression.clone());
    }

    let file_format = factory.create(&state, &factory_options)?;
    let session_config = context.copied_config();
    let mut listing_options = ListingOptions::new(file_format)
        .with_file_extension(listing_extension.unwrap_or_default())
        .with_session_config_options(&session_config);

    if let Some(partitioning) = partitioning {
        listing_options = listing_options.with_table_partition_cols(partitioning);
    } else if session_config
        .options()
        .execution
        .listing_table_factory_infer_partitions
    {
        let inferred = listing_options
            .infer_partitions(&state, &table_paths[0])
            .await?;
        if !inferred.is_empty() {
            listing_options = listing_options.with_table_partition_cols(
                inferred
                    .into_iter()
                    .map(|name| (name, DataType::Utf8View))
                    .collect(),
            );
        }
    }

    let config = ListingTableConfig::new_with_multi_paths(table_paths)
        .with_listing_options(listing_options)
        .infer_schema(&state)
        .await?;
    Ok(Arc::new(ListingTable::try_new(config)?))
}

fn reader_options(options: &HashMap<String, String>) -> HashMap<String, String> {
    options
        .iter()
        .filter(|(key, _)| {
            !["aws.", "azure.", "gcp.", "gcs.", "google.", "http."]
                .iter()
                .any(|prefix| key.starts_with(prefix))
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn resolve_format(
    table_paths: &[ListingTableUrl],
    requested: Option<&str>,
) -> Result<(String, Option<String>, Option<String>)> {
    if let Some(requested) = requested {
        let requested = requested.trim_start_matches('.').to_lowercase();
        if requested.is_empty() {
            return plan_err!("Format must not be empty");
        }

        // Keep extension filtering when the paths have the requested suffix,
        // but allow an explicit format to read extensionless or differently
        // named objects.
        let inferred = infer_path_format(table_paths).ok();
        if let Some((inferred_format, compression)) = inferred {
            if inferred_format == requested {
                let extension = match &compression {
                    Some(compression) => format!("{requested}.{compression}"),
                    None => requested.clone(),
                };
                return Ok((requested, compression, Some(extension)));
            }
        }
        return Ok((requested, None, None));
    }

    let (format, compression) = infer_path_format(table_paths)?;
    let extension = match &compression {
        Some(compression) => format!("{format}.{compression}"),
        None => format.clone(),
    };
    Ok((format, compression, Some(extension)))
}

fn infer_path_format(table_paths: &[ListingTableUrl]) -> Result<(String, Option<String>)> {
    let mut resolved: Option<(String, Option<String>)> = None;
    for path in table_paths {
        let path_without_query = path
            .as_str()
            .split(['?', '#'])
            .next()
            .unwrap_or(path.as_str())
            .trim_end_matches('/');
        let mut next =
            ListingTableConfig::infer_file_extension_and_compression_type(path_without_query)?;
        if next.0.is_empty() {
            return plan_err!(
                "Can't guess format from path without an extension: '{}'",
                path.as_str()
            );
        }
        next.0.make_ascii_lowercase();
        if let Some(current) = &resolved {
            if current != &next {
                return plan_err!(
                    "Can't guess format from paths with multiple extensions ('{}' and '{}')",
                    current.0,
                    next.0
                );
            }
        } else {
            resolved = Some(next);
        }
    }

    Ok(resolved.expect("non-empty paths checked by read_provider"))
}
