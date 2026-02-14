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
    io::{Cursor, Read},
    ops::Range,
    sync::Arc,
};

use arrow_array::RecordBatch;
use bytes::Bytes;
use datafusion_common::error::DataFusionError;
use datafusion_datasource::PartitionedFile;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use futures::{future::BoxFuture, FutureExt};
use las::{raw::Point as RawPoint, Header, Point};
use laz::{
    record::{
        LayeredPointRecordDecompressor, RecordDecompressor, SequentialPointRecordDecompressor,
    },
    DecompressionSelection, LasZipError,
};
use object_store::ObjectStore;

use crate::{
    laz::{
        builder::RowBuilder,
        metadata::{ChunkMeta, LazMetadata, LazMetadataReader},
    },
    options::PointcloudOptions,
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
        options: PointcloudOptions,
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
    pub options: PointcloudOptions,
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
        let bytes = self.get_bytes(chunk_meta.byte_range.clone()).await?;

        // record batch builder
        let num_points = chunk_meta.num_points as usize;
        let mut builder = RowBuilder::new(num_points, header.clone())
            .with_geometry_encoding(self.options.geometry_encoding)
            .with_extra_attributes(
                metadata.extra_attributes.clone(),
                self.options.las.extra_bytes,
            );

        // parse points
        if header.laz_vlr().is_ok() {
            // laz decompressor
            let mut decompressor = record_decompressor(&header, bytes)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            let out = vec![0; header.point_format().len() as usize];
            let mut buffer = Cursor::new(out);

            for _ in 0..chunk_meta.num_points {
                buffer.set_position(0);
                decompressor.decompress_next(buffer.get_mut())?;
                let point = read_point(&mut buffer, &header)?;
                builder.append(point);
            }
        } else {
            let mut buffer = Cursor::new(bytes);

            for _ in 0..chunk_meta.num_points {
                let point = read_point(&mut buffer, &header)?;
                builder.append(point);
            }
        }

        let struct_array = builder.finish()?;

        Ok(RecordBatch::from(struct_array))
    }

    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes, object_store::Error> {
        let location = &self.partitioned_file.object_meta.location;
        self.store.get_range(location, range).await
    }
}

pub fn record_decompressor(
    header: &Header,
    bytes: Bytes,
) -> Result<Box<dyn RecordDecompressor<Cursor<Bytes>>>, las::Error> {
    let laz_vlr = header.laz_vlr()?;
    let reader = Cursor::new(bytes);

    let first_item = laz_vlr
        .items()
        .first()
        .expect("There should be at least one LazItem to be able to create a RecordDecompressor");

    let mut decompressor = match first_item.version() {
        1 | 2 => {
            let decompressor = SequentialPointRecordDecompressor::new(reader);
            Box::new(decompressor) as Box<dyn RecordDecompressor<Cursor<Bytes>>>
        }
        3 | 4 => {
            let decompressor = LayeredPointRecordDecompressor::new(reader);
            Box::new(decompressor) as Box<dyn RecordDecompressor<Cursor<Bytes>>>
        }
        _ => {
            return Err(LasZipError::UnsupportedLazItemVersion(
                first_item.item_type(),
                first_item.version(),
            )
            .into());
        }
    };

    decompressor.set_fields_from(laz_vlr.items())?;
    decompressor.set_selection(DecompressionSelection::all());

    Ok(decompressor)
}

pub(crate) fn read_point<R: Read>(buffer: R, header: &Header) -> Result<Point, DataFusionError> {
    RawPoint::read_from(buffer, header.point_format())
        .map(|raw_point| Point::new(raw_point, header.transforms()))
        .map_err(|e| DataFusionError::External(Box::new(e)))
}

#[cfg(test)]
mod tests {
    use std::{fs::File, sync::Arc};

    use datafusion_datasource::PartitionedFile;
    use las::{point::Format, Builder, Writer};
    use object_store::{local::LocalFileSystem, path::Path, ObjectStore};

    use crate::laz::reader::LazFileReaderFactory;

    #[tokio::test]
    async fn reader_basic_e2e() {
        let tmpdir = tempfile::tempdir().unwrap();

        // create laz file with one point
        let tmp_path = tmpdir.path().join("one.laz");
        let tmp_file = File::create(&tmp_path).unwrap();
        let mut builder = Builder::from((1, 4));
        builder.point_format = Format::new(0).unwrap();
        builder.point_format.is_compressed = true;
        let header = builder.into_header().unwrap();
        let mut writer = Writer::new(tmp_file, header).unwrap();
        writer.write_point(Default::default()).unwrap();
        writer.close().unwrap();

        // read batch with `LazFileReader`
        let store = LocalFileSystem::new();
        let location = Path::from_filesystem_path(tmp_path).unwrap();
        let object = store.head(&location).await.unwrap();

        let laz_file_reader = LazFileReaderFactory::new(Arc::new(store), None)
            .create_reader(
                PartitionedFile::new(location, object.size),
                Default::default(),
            )
            .unwrap();
        let metadata = laz_file_reader.get_metadata().await.unwrap();

        let batch = laz_file_reader
            .get_batch(&metadata.chunk_table[0])
            .await
            .unwrap();

        assert_eq!(batch.num_rows(), 1);
    }
}
