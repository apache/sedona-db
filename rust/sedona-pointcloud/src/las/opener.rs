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

use crate::{
    las::{
        reader::{LasFileReader, LasFileReaderFactory},
        schema::try_schema_from_header,
    },
    options::PointcloudOptions,
};

pub struct LasOpener {
    /// Column indexes in `table_schema` needed by the query
    pub projection: Arc<[usize]>,
    /// Optional limit on the number of rows to read
    pub limit: Option<usize>,
    /// Filter predicate for pruning
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Factory for instantiating LAS/LAZ reader
    pub file_reader_factory: Arc<LasFileReaderFactory>,
    /// Table options
    pub options: PointcloudOptions,
    /// Target batch size
    pub batch_size: usize,
    /// Target partition count
    pub partition_count: usize,
    /// Partition to read
    pub partition: usize,
}

impl FileOpener for LasOpener {
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture, DataFusionError> {
        let projection = self.projection.clone();
        let limit = self.limit;
        let batch_size = self.batch_size;
        let round_robin = self.options.round_robin_partitioning;
        let partition_count = self.partition_count;
        let partition = self.partition;

        let predicate = self.predicate.clone();

        let file_reader: Box<LasFileReader> = self
            .file_reader_factory
            .create_reader(file.clone(), self.options.clone())?;

        Ok(Box::pin(async move {
            let metadata = file_reader.get_metadata().await?;
            let schema = Arc::new(try_schema_from_header(
                &metadata.header,
                file_reader.options.geometry_encoding,
                file_reader.options.las.extra_bytes,
            )?);

            let pruning_predicate = predicate.and_then(|physical_expr| {
                PruningPredicate::try_new(physical_expr, schema.clone()).ok()
            });

            let spatial_filter = pruning_predicate
                .as_ref()
                .and_then(|p| SpatialFilter::try_from_expr(p.orig_expr()).ok());

            // file pruning
            if let Some(pruning_predicate) = &pruning_predicate {
                // based on spatial filter
                if let Some(spatial_filter) = &spatial_filter {
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

            // chunk pruning filter
            let chunk_filter_xyz = pruning_predicate.and_then(|predicate| {
                metadata
                    .statistics
                    .as_ref()
                    .and_then(|stats| predicate.prune(stats).ok())
            });

            let mut row_count = 0;

            let stream = async_stream::try_stream! {
                for (i, chunk_meta) in metadata.chunk_table.iter().enumerate() {
                    // round robin
                    if round_robin && i % partition_count != partition {
                        continue;
                    }

                    // limit
                    if let Some(limit) = limit {
                        if row_count >= limit {
                            break;
                        }
                    }

                    // byte range
                    if !file.range.as_ref().is_none_or(|range| {
                        let offset = chunk_meta.byte_range.start;
                        offset >= range.start as u64 && offset < range.end as u64
                    }) {
                        continue;
                    }

                    // pruning
                    if let Some(filter) = chunk_filter_xyz.as_ref() {
                        if !filter[i] {
                            continue;
                        }
                    }
                    if let (Some(spatial_filter), Some(stats)) = (spatial_filter.as_ref(), metadata.statistics.as_ref()) {
                        let bbox = stats.get_bbox(i).unwrap();
                        if !spatial_filter.filter_bbox("geometry").intersects(&bbox) {
                            continue;
                        }
                    }

                    // fetch batch
                    let record_batch = file_reader.get_batch(chunk_meta).await?;
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

#[cfg(test)]
mod tests {
    use sedona::context::SedonaContext;

    #[tokio::test]
    async fn las_statistics_pruning() {
        // file with two clusters, one at 0.5 one at 1.0
        let path = "tests/data/large.las";

        let ctx = SedonaContext::new_local_interactive().await.unwrap();

        // ensure no faulty chunk pruning
        ctx.sql("SET pointcloud.geometry_encoding = 'plain'")
            .await
            .unwrap();
        ctx.sql("SET pointcloud.collect_statistics = 'true'")
            .await
            .unwrap();

        let count = ctx
            .sql(&format!("SELECT * FROM \"{path}\" WHERE x < 0.7"))
            .await
            .unwrap()
            .count()
            .await
            .unwrap();
        assert_eq!(count, 50000);

        let count = ctx
            .sql(&format!("SELECT * FROM \"{path}\" WHERE y < 0.7"))
            .await
            .unwrap()
            .count()
            .await
            .unwrap();
        assert_eq!(count, 50000);

        ctx.sql("SET pointcloud.geometry_encoding = 'wkb'")
            .await
            .unwrap();
        let count = ctx
            .sql(&format!("SELECT * FROM \"{path}\" WHERE ST_Intersects(geometry, ST_GeomFromText('POLYGON ((0 0, 0.7 0, 0.7 0.7, 0 0.7, 0 0))'))"))
            .await
            .unwrap()
            .count()
            .await
            .unwrap();
        assert_eq!(count, 50000);
    }

    #[tokio::test]
    async fn laz_statistics_pruning() {
        // file with two clusters, one at 0.5 one at 1.0
        let path = "tests/data/large.laz";

        let ctx = SedonaContext::new_local_interactive().await.unwrap();

        // ensure no faulty chunk pruning
        ctx.sql("SET pointcloud.geometry_encoding = 'plain'")
            .await
            .unwrap();
        ctx.sql("SET pointcloud.collect_statistics = 'true'")
            .await
            .unwrap();

        let count = ctx
            .sql(&format!("SELECT * FROM \"{path}\" WHERE x < 0.7"))
            .await
            .unwrap()
            .count()
            .await
            .unwrap();
        assert_eq!(count, 50000);

        let count = ctx
            .sql(&format!("SELECT * FROM \"{path}\" WHERE y < 0.7"))
            .await
            .unwrap()
            .count()
            .await
            .unwrap();
        assert_eq!(count, 50000);

        ctx.sql("SET pointcloud.geometry_encoding = 'wkb'")
            .await
            .unwrap();
        let count = ctx
            .sql(&format!("SELECT * FROM \"{path}\" WHERE ST_Intersects(geometry, ST_GeomFromText('POLYGON ((0 0, 0.7 0, 0.7 0.7, 0 0.7, 0 0))'))"))
            .await
            .unwrap()
            .count()
            .await
            .unwrap();
        assert_eq!(count, 50000);
    }

    #[tokio::test]
    async fn round_robin_partitioning() {
        // file with two clusters, one at 0.5 one at 1.0
        let path = "tests/data/large.laz";

        let ctx = SedonaContext::new_local_interactive().await.unwrap();

        let result1 = ctx
            .sql(&format!("SELECT * FROM \"{path}\""))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        ctx.sql("SET pointcloud.round_robin_partitioning = 'true'")
            .await
            .unwrap();
        let result2 = ctx
            .sql(&format!("SELECT * FROM \"{path}\""))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(result1, result2);
    }
}
