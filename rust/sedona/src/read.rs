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
    datasource::{
        file_format::FileFormatFactory,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
    },
    execution::SessionState,
    prelude::SessionContext,
};
use datafusion_common::{exec_err, plan_err, Result};

use crate::object_storage::{
    ensure_object_store_registered_with_options, register_table_options_extension_from_scheme,
};
use sedona_datasource::{format::ExternalFileFormat, provider::external_table};

/// A file format factory plus the path-derived details needed for listing.
pub(crate) struct ResolvedReadFormat {
    factory: Arc<dyn FileFormatFactory>,
    compression: Option<String>,
    listing_extension: Option<String>,
}

/// Resolve paths and options into a table provider using an already-selected
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
    format: ResolvedReadFormat,
    partitioning: Option<Vec<(String, DataType)>>,
) -> Result<Arc<dyn TableProvider>> {
    if table_paths.is_empty() {
        return exec_err!("No table paths were provided");
    }

    for path in &table_paths {
        register_table_options_extension_from_scheme(context, path.scheme());
        ensure_object_store_registered_with_options(
            &mut context.state(),
            path.as_str(),
            Some(options),
        )
        .await?;
    }

    let state = context.state();
    let mut factory_options = sql_style_options(options);
    if let Some(compression) = &format.compression {
        factory_options
            .entry("format.compression".to_string())
            .or_insert_with(|| compression.clone());
    }

    // This is the same key/value entry point used by DataFusion's
    // ListingTableFactory for CREATE EXTERNAL TABLE. Any format-specific
    // deserialization belongs to the selected factory, not this generic path.
    let file_format = format.factory.create(&state, &factory_options)?;
    if let Some(external) = file_format.downcast_ref::<ExternalFileFormat>() {
        return external_table(
            external.spec().clone(),
            context,
            table_paths,
            false,
            partitioning,
        )
        .await;
    }

    let session_config = context.copied_config();
    let mut listing_options = ListingOptions::new(file_format)
        .with_file_extension(format.listing_extension.unwrap_or_default())
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

fn sql_style_options(options: &HashMap<String, String>) -> HashMap<String, String> {
    options
        .iter()
        .filter_map(|(key, value)| {
            // Dotted non-format keys belong to connection/table option
            // namespaces. Object-store registration consumes the known ones;
            // retain the historical behavior of ignoring unknown namespaces
            // rather than passing them to a built-in file-format parser.
            if key.starts_with("format.") {
                Some((key.clone(), value.clone()))
            } else if key.contains('.') {
                None
            } else {
                Some((format!("format.{key}"), value.clone()))
            }
        })
        .collect()
}

pub(crate) fn resolve_read_format(
    state: &SessionState,
    table_paths: &[ListingTableUrl],
    requested: Option<Arc<dyn FileFormatFactory>>,
    check_extension: bool,
) -> Result<ResolvedReadFormat> {
    if table_paths.is_empty() {
        return exec_err!("No table paths were provided");
    }

    if let Some(factory) = requested {
        if !check_extension {
            return Ok(ResolvedReadFormat {
                factory,
                compression: None,
                listing_extension: None,
            });
        }
        let requested_extension = factory.get_ext().trim_start_matches('.').to_lowercase();

        // Keep extension filtering when the paths have the requested suffix,
        // but allow an explicit format to read extensionless or differently
        // named objects.
        let inferred = infer_path_format(table_paths).ok();
        if let Some((inferred_format, compression)) = inferred {
            if inferred_format == requested_extension {
                let extension = match &compression {
                    Some(compression) => format!("{requested_extension}.{compression}"),
                    None => requested_extension,
                };
                return Ok(ResolvedReadFormat {
                    factory,
                    compression,
                    listing_extension: Some(extension),
                });
            }
        }
        return Ok(ResolvedReadFormat {
            factory,
            compression: None,
            listing_extension: None,
        });
    }

    let (extension, compression) = infer_path_format(table_paths)?;
    let factory = state.get_file_format_factory(&extension).ok_or_else(|| {
        datafusion_common::plan_datafusion_err!("No format registered for extension '{extension}'")
    })?;
    let extension = match &compression {
        Some(compression) => format!("{extension}.{compression}"),
        None => extension,
    };
    Ok(ResolvedReadFormat {
        factory,
        compression,
        listing_extension: Some(extension),
    })
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

    Ok(resolved.expect("non-empty paths checked by resolve_read_format"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sql_style_options_excludes_table_option_namespaces() {
        let options = HashMap::from([
            ("batch_size".to_string(), "1024".to_string()),
            (
                "format.schema_force_view_types".to_string(),
                "true".to_string(),
            ),
            ("aws.region".to_string(), "us-west-2".to_string()),
            ("test.option".to_string(), "value".to_string()),
        ]);

        assert_eq!(
            sql_style_options(&options),
            HashMap::from([
                ("format.batch_size".to_string(), "1024".to_string()),
                (
                    "format.schema_force_view_types".to_string(),
                    "true".to_string()
                ),
            ])
        );
    }

    #[test]
    fn resolve_read_format_rejects_empty_paths() {
        let state = SessionContext::new().state();
        let error = match resolve_read_format(&state, &[], None, true) {
            Ok(_) => panic!("empty paths should fail"),
            Err(error) => error,
        };
        assert_eq!(
            error.strip_backtrace(),
            "Execution error: No table paths were provided"
        );
    }
}
