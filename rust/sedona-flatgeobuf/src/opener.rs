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

use arrow_schema::SchemaRef;
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use datafusion_datasource::{
    PartitionedFile,
    file_stream::{FileOpenFuture, FileOpener},
};
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use futures::StreamExt;
use object_store::ObjectStore;
use std::{io::Cursor, sync::Arc};

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
                &format.geometry_column_name,
                format.metadata_size_hint,
                cache.as_ref(),
            )
            .await?;
            if meta.schema.as_ref() != schema.as_ref() {
                return exec_err!("FlatGeobuf schema changed before reading");
            }
            let count = meta.offsets.len() - 1;
            let (lo, hi) = if let Some(range) = file.range {
                if range.start < 0 || range.end < range.start {
                    return exec_err!("Invalid FlatGeobuf file range");
                }
                if meta.indexed {
                    let starts = &meta.offsets[..count];
                    (
                        starts.partition_point(|p| *p < range.start as u64),
                        starts.partition_point(|p| *p < range.end as u64),
                    )
                } else if range.start == 0 && range.end > 0 {
                    (0, count)
                } else {
                    (0, 0)
                }
            } else {
                (0, count)
            };
            let stream = async_stream::try_stream! {
                let mut first=lo;
                while first<hi {
                    let last=first.saturating_add(batch_size).min(hi);
                    let feature_ranges=meta.offsets[first..=last].windows(2).map(|p|p[0]..p[1]).collect::<Vec<_>>();
                    let requests=crate::object_io::coalesce_ranges(&feature_ranges,file.object_meta.size,8*1024*1024)?;
                    for range in requests {
                        let begin=meta.offsets.binary_search(&range.start).map_err(|_|exec_datafusion_err!("Invalid planned feature start"))?;
                        let end=meta.offsets.binary_search(&range.end).map_err(|_|exec_datafusion_err!("Invalid planned feature end"))?;
                        let bytes=crate::object_io::fetch_range(store.as_ref(),&file.object_meta,range.clone()).await?;
                        validate_frames(&bytes,&meta.offsets[begin..=end],range.start)?;
                        let reader=crate::reader::decode(meta.clone(),Box::new(Cursor::new(bytes)),end-begin,projection.clone(),batch_size)?;
                        for batch in reader { yield batch.map_err(|e|datafusion_common::DataFusionError::ArrowError(Box::new(e),None))?; }
                    }
                    first=last;
                }
            };
            Ok(stream.boxed())
        }))
    }
}
fn validate_frames(bytes: &[u8], offsets: &[u64], begin: u64) -> Result<()> {
    for pair in offsets.windows(2) {
        let start = usize::try_from(pair[0] - begin)
            .map_err(|_| exec_datafusion_err!("FlatGeobuf offset overflow"))?;
        let prefix = bytes
            .get(start..start + 4)
            .ok_or_else(|| exec_datafusion_err!("Truncated FlatGeobuf feature prefix"))?;
        let len = u32::from_le_bytes(prefix.try_into().unwrap()) as u64;
        if len == 0 || 4 + len != pair[1] - pair[0] {
            return exec_err!("FlatGeobuf index does not match feature framing");
        }
    }
    Ok(())
}
