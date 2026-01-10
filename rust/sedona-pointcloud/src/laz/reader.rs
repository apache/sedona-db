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

use std::{
    io::{Cursor, Read, Seek},
    sync::Arc,
};

use arrow_array::RecordBatch;
use datafusion_common::error::DataFusionError;
use datafusion_datasource::PartitionedFile;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use futures::{future::BoxFuture, FutureExt};
use las::{raw::Point as RawPoint, Point};
use laz::{
    record::{
        LayeredPointRecordDecompressor, RecordDecompressor, SequentialPointRecordDecompressor,
    },
    DecompressionSelection, LasZipError, LazItem,
};
use object_store::ObjectStore;

use crate::laz::{
    builder::RowBuilder,
    metadata::{ChunkMeta, LazMetadata, LazMetadataReader},
    options::LazTableOptions,
};

/// Laz file reader factory
#[derive(Debug)]
pub struct LazFileReaderFactory {
    store: Arc<dyn ObjectStore>,
    metadata_cache: Option<Arc<dyn FileMetadataCache>>,
}

impl LazFileReaderFactory {
    /// Create a new `LazFileReaderFactory`.
    pub fn new(
        store: Arc<dyn ObjectStore>,
        metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    ) -> Self {
        Self {
            store,
            metadata_cache,
        }
    }

    pub fn create_reader(
        &self,
        partitioned_file: PartitionedFile,
        options: LazTableOptions,
    ) -> Result<Box<LazFileReader>, DataFusionError> {
        Ok(Box::new(LazFileReader {
            partitioned_file,
            store: self.store.clone(),
            metadata_cache: self.metadata_cache.clone(),
            options,
        }))
    }
}

/// Reader for a laz file in object storage.
pub struct LazFileReader {
    partitioned_file: PartitionedFile,
    store: Arc<dyn ObjectStore>,
    metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    pub options: LazTableOptions,
}

impl LazFileReader {
    pub fn get_metadata<'a>(&'a self) -> BoxFuture<'a, Result<Arc<LazMetadata>, DataFusionError>> {
        let object_meta = self.partitioned_file.object_meta.clone();
        let metadata_cache = self.metadata_cache.clone();

        async move {
            LazMetadataReader::new(&self.store, &object_meta)
                .with_file_metadata_cache(metadata_cache)
                .with_options(self.options.clone())
                .fetch_metadata()
                .await
        }
        .boxed()
    }

    pub async fn get_batch(&self, chunk_meta: &ChunkMeta) -> Result<RecordBatch, DataFusionError> {
        let metadata = self.get_metadata().await?;
        let header = metadata.header.clone();

        // fetch bytes
        let location = &self.partitioned_file.object_meta.location;
        let bytes = self
            .store
            .get_range(location, chunk_meta.byte_range.clone())
            .await?;

        // laz decompressor
        let laz_vlr = header
            .laz_vlr()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let reader = Cursor::new(bytes);
        let mut decompressor = record_decompressor(laz_vlr.items(), reader)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // record batch builder
        let num_points = chunk_meta.num_points as usize;
        let mut builder = RowBuilder::new(num_points, header.clone())
            .with_point_encoding(self.options.point_encoding)
            .with_extra_attributes(metadata.extra_attributes.clone(), self.options.extra_bytes);

        // transform
        let format = header.point_format();
        let transforms = header.transforms();

        let out = vec![0; format.len() as usize];
        let mut buffer = Cursor::new(out);

        for _ in 0..chunk_meta.num_points {
            buffer.set_position(0);
            decompressor.decompress_next(buffer.get_mut())?;

            let point = RawPoint::read_from(&mut buffer, format)
                .map(|raw_point| Point::new(raw_point, transforms))
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            builder.append(point);
        }

        let struct_array = builder.finish()?;

        Ok(RecordBatch::from(struct_array))
    }
}

pub(super) fn record_decompressor<'a, R: Read + Seek + Send + Sync + 'a>(
    items: &Vec<LazItem>,
    input: R,
) -> Result<Box<dyn RecordDecompressor<R> + Send + Sync + 'a>, LasZipError> {
    let first_item = items
        .first()
        .expect("There should be at least one LazItem to be able to create a RecordDecompressor");

    let mut decompressor = match first_item.version() {
        1 | 2 => {
            let decompressor = SequentialPointRecordDecompressor::new(input);
            Box::new(decompressor) as Box<dyn RecordDecompressor<R> + Send + Sync>
        }
        3 | 4 => {
            let decompressor = LayeredPointRecordDecompressor::new(input);
            Box::new(decompressor) as Box<dyn RecordDecompressor<R> + Send + Sync>
        }
        _ => {
            return Err(LasZipError::UnsupportedLazItemVersion(
                first_item.item_type(),
                first_item.version(),
            ));
        }
    };

    decompressor.set_fields_from(items)?;
    decompressor.set_selection(DecompressionSelection::all());

    Ok(decompressor)
}
