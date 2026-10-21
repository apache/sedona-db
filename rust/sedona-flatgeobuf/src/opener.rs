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

use crate::{metadata::Metadata, reader::decode_batch};
use arrow_schema::SchemaRef;
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use datafusion_datasource::FileRange;
use datafusion_datasource::{
    PartitionedFile,
    file_stream::{FileOpenFuture, FileOpener},
};
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_plan::{SendableRecordBatchStream, stream::RecordBatchStreamAdapter};
use futures::StreamExt;
use object_store::ObjectStore;
use std::sync::Arc;

pub(crate) struct FlatGeobufOpener {
    pub store: Arc<dyn ObjectStore>,
    pub cache: Option<Arc<dyn FileMetadataCache>>,
    pub format: crate::format::FlatGeobufFormat,
    pub projection: Vec<usize>,
    pub batch_size: usize,
    pub schema: SchemaRef,
}

impl FileOpener for FlatGeobufOpener {
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture> {
        let store = self.store.clone();
        let cache = self.cache.clone();
        let format = self.format.clone();
        let projection = self.projection.clone();
        let batch_size = self.batch_size;
        let schema = self.schema.clone();
        if batch_size == 0 {
            return exec_err!("FlatGeobuf batch size must be positive");
        }
        Ok(Box::pin(async move {
            let meta = crate::store_metadata::metadata(
                store.clone(),
                &file.object_meta,
                format.metadata_size_hint,
                cache.as_ref(),
            )
            .await?;
            if file.range.is_some() {
                return exec_err!("Native FlatGeobuf scans do not support byte ranges");
            }
            let output_schema = Arc::new(schema.project(&projection)?);
            let stream = async_stream::try_stream! {
                let mut reader = BufferedFeatures::new(store, file.object_meta, &meta);
                while let Some((bytes, offsets)) = reader.fetch_batch(batch_size).await? {
                    yield decode_batch(&meta, &schema, &projection, &bytes, &offsets, 0, &format.geometry_column_name)?;
                }
            };
            let stream: SendableRecordBatchStream =
                Box::pin(RecordBatchStreamAdapter::new(output_schema, stream));
            Ok(stream.boxed())
        }))
    }
}
/// Sequential framing and I/O are independent of Arrow conversion and row batch
/// size. Small batches reuse the same fetched buffer; large features may span
/// requests, but no request is issued for an individual prefix or RTree leaf.
struct BufferedFeatures {
    store: Arc<dyn ObjectStore>,
    object: object_store::ObjectMeta,
    buffer: bytes::BytesMut,
    next_fetch: u64,
    expected_count: usize,
    rows: usize,
}

impl BufferedFeatures {
    const READ_BYTES: u64 = 8 * 1024 * 1024;

    fn new(store: Arc<dyn ObjectStore>, object: object_store::ObjectMeta, meta: &Metadata) -> Self {
        Self {
            store,
            object,
            buffer: bytes::BytesMut::new(),
            next_fetch: meta.payload_begin,
            expected_count: meta.feature_count,
            rows: 0,
        }
    }

    async fn fill(&mut self, required: usize) -> Result<()> {
        while self.buffer.len() < required && self.next_fetch < self.object.size {
            let end = self
                .next_fetch
                .saturating_add(Self::READ_BYTES)
                .min(self.object.size);
            let bytes = crate::object_io::fetch_range(
                self.store.as_ref(),
                &self.object,
                self.next_fetch..end,
            )
            .await?;
            self.buffer.extend_from_slice(&bytes);
            self.next_fetch = end;
        }
        Ok(())
    }

    async fn next_frame(&mut self) -> Result<Option<bytes::Bytes>> {
        self.fill(4).await?;
        if self.buffer.is_empty() {
            if self.expected_count != 0 && self.rows != self.expected_count {
                return exec_err!("FlatGeobuf count does not match object");
            }
            return Ok(None);
        }
        let prefix = self
            .buffer
            .get(..4)
            .ok_or_else(|| exec_datafusion_err!("Truncated FlatGeobuf feature prefix"))?;
        let len = u32::from_le_bytes(prefix.try_into().unwrap()) as u64;
        let available = self.buffer.len() as u64 + self.object.size - self.next_fetch;
        if len == 0 || len + 4 > available {
            return exec_err!("Invalid or truncated FlatGeobuf feature payload");
        }
        let frame_len = usize::try_from(len + 4)
            .map_err(|_| exec_datafusion_err!("FlatGeobuf feature length overflow"))?;
        self.fill(frame_len).await?;
        self.rows += 1;
        if self.expected_count != 0 && self.rows > self.expected_count {
            return exec_err!("FlatGeobuf count does not match object");
        }
        Ok(Some(self.buffer.split_to(frame_len).freeze()))
    }

    async fn fetch_batch(&mut self, batch_size: usize) -> Result<Option<(Vec<u8>, Vec<u64>)>> {
        let mut bytes = Vec::new();
        let mut offsets = vec![0];
        while offsets.len() <= batch_size && bytes.len() < Self::READ_BYTES as usize {
            let Some(frame) = self.next_frame().await? else {
                break;
            };
            bytes.extend_from_slice(&frame);
            offsets.push(bytes.len() as u64);
        }
        if offsets.len() == 1 {
            Ok(None)
        } else {
            Ok(Some((bytes, offsets)))
        }
    }
}

/// Select features whose size-prefix starts are owned by this half-open byte range.
/// Indexed ranges may cut through payloads: ownership follows the prefix, and the
/// fetched payload is expanded to feature boundaries. Without an RTree only the
/// range owning byte zero reads rows, avoiding duplicate serial fallback scans.
pub(crate) fn feature_bounds(meta: &Metadata, range: Option<&FileRange>) -> Result<(usize, usize)> {
    let count = meta.offsets.len() - 1;
    let Some(range) = range else {
        return Ok((0, count));
    };
    if range.start < 0 || range.end < range.start {
        return exec_err!("Invalid FlatGeobuf file range");
    }
    if meta.indexed {
        let starts = &meta.offsets[..count];
        Ok((
            starts.partition_point(|p| *p < range.start as u64),
            starts.partition_point(|p| *p < range.end as u64),
        ))
    } else if range.start == 0 && range.end > 0 {
        Ok((0, count))
    } else {
        Ok((0, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::{ObjectStoreExt, memory::InMemory, path::Path};

    async fn framed_reader(bytes: Vec<u8>, expected_count: usize) -> BufferedFeatures {
        let store = Arc::new(InMemory::new());
        let path = Path::from("frames");
        store.put(&path, bytes.into()).await.unwrap();
        let object = store.head(&path).await.unwrap();
        BufferedFeatures {
            store,
            object,
            buffer: bytes::BytesMut::new(),
            next_fetch: 0,
            expected_count,
            rows: 0,
        }
    }

    #[tokio::test]
    async fn sequential_frames_cross_request_boundaries_and_reject_truncation() {
        let frame = |len: usize| {
            let mut b = (len as u32).to_le_bytes().to_vec();
            b.resize(len + 4, 7);
            b
        };
        // Next prefix straddles the 8 MiB read boundary; then a frame spans
        // multiple requests. Each frame is returned whole without loss/repeat.
        let frames = [
            frame(BufferedFeatures::READ_BYTES as usize - 6),
            frame(9),
            frame(BufferedFeatures::READ_BYTES as usize + 13),
        ];
        let mut reader = framed_reader(frames.concat(), 3).await;
        for expected in &frames {
            assert_eq!(
                reader.next_frame().await.unwrap().unwrap().as_ref(),
                expected
            );
        }
        assert!(reader.next_frame().await.unwrap().is_none());
        for bytes in [vec![1, 2], vec![0, 0, 0, 0], vec![9, 0, 0, 0, 1]] {
            assert!(framed_reader(bytes, 0).await.next_frame().await.is_err());
        }
        for count in [1, 3] {
            let mut reader = framed_reader([frame(8), frame(8)].concat(), count).await;
            assert!(reader.next_frame().await.unwrap().is_some());
            if count == 1 {
                assert!(reader.next_frame().await.is_err());
            } else {
                reader.next_frame().await.unwrap();
                assert!(reader.next_frame().await.is_err());
            }
        }
    }
}
