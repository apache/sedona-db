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

use datafusion_common::{error::DataFusionError, pruning::PrunableStatistics};
use datafusion_datasource::{
    file_stream::{FileOpenFuture, FileOpener},
    PartitionedFile,
};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_pruning::PruningPredicate;
use futures::StreamExt;

use sedona_expr::spatial_filter::SpatialFilter;
use sedona_geometry::bounding_box::BoundingBox;

use crate::laz::{
    options::LazTableOptions,
    reader::{LazFileReader, LazFileReaderFactory},
    schema::try_schema_from_header,
};

pub struct LazOpener {
    /// Column indexes in `table_schema` needed by the query
    pub projection: Arc<[usize]>,
    /// Optional limit on the number of rows to read
    pub limit: Option<usize>,
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Factory for instantiating laz reader
    pub laz_file_reader_factory: Arc<LazFileReaderFactory>,
    /// Table options
    pub options: LazTableOptions,
    /// Target batch size
    pub(crate) batch_size: usize,
}

impl FileOpener for LazOpener {
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture, DataFusionError> {
        let projection = self.projection.clone();
        let limit = self.limit;
        let batch_size = self.batch_size;

        let predicate = self.predicate.clone();

        let laz_reader: Box<LazFileReader> = self
            .laz_file_reader_factory
            .create_reader(file.clone(), self.options.clone())?;

        Ok(Box::pin(async move {
            let metadata = laz_reader.get_metadata().await?;
            let schema = Arc::new(try_schema_from_header(
                &metadata.header,
                laz_reader.options.point_encoding,
                laz_reader.options.extra_bytes,
            )?);

            let pruning_predicate = predicate.and_then(|physical_expr| {
                PruningPredicate::try_new(physical_expr, schema.clone()).ok()
            });

            // file pruning
            if let Some(pruning_predicate) = &pruning_predicate {
                // based on spatial filter
                let spatial_filter = SpatialFilter::try_from_expr(pruning_predicate.orig_expr())?;
                let bounds = metadata.header.bounds();
                let bbox = BoundingBox::xyzm(
                    (bounds.min.x, bounds.max.x),
                    (bounds.min.y, bounds.max.y),
                    Some((bounds.min.z, bounds.max.z).into()),
                    None,
                );
                if !spatial_filter.filter_bbox("geometry").intersects(&bbox) {
                    return Ok(futures::stream::empty().boxed());
                }
                // based on file statistics
                if let Some(statistics) = file.statistics {
                    let prunable_statistics = PrunableStatistics::new(vec![statistics], schema);
                    if let Ok(filter) = pruning_predicate.prune(&prunable_statistics) {
                        if !filter[0] {
                            return Ok(futures::stream::empty().boxed());
                        }
                    }
                }
            }

            // map chunk table
            let chunk_table: Vec<_> = metadata
                .chunk_table
                .iter()
                .filter(|chunk_meta| {
                    file.range.as_ref().is_none_or(|range| {
                        let offset = chunk_meta.byte_range.start;
                        offset >= range.start as u64 && offset < range.end as u64
                    })
                })
                .cloned()
                .collect();

            let mut row_count = 0;

            let stream = async_stream::try_stream! {
                for chunk_meta in chunk_table.into_iter() {
                    // limit
                    if let Some(limit) = limit {
                        if row_count >= limit {
                            break;
                        }
                    }

                    // fetch batch
                    let record_batch = laz_reader.get_batch(&chunk_meta).await?;
                    let num_rows = record_batch.num_rows();
                    row_count += num_rows;

                    // project
                    let record_batch = record_batch
                        .project(&projection)
                        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

                    // adhere to target batch size
                    let mut offset = 0;

                    loop {
                        let length = batch_size.min(num_rows - offset);
                        yield record_batch.slice(offset, length);

                        offset += batch_size;
                        if offset >= num_rows {
                            break;
                        }
                    }
                }

            };

            Ok(Box::pin(stream) as _)
        }))
    }
}
