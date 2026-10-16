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

use crate::{
    metadata::Metadata,
    object_io::{fetch_header, fetch_range},
};
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use datafusion_execution::cache::cache_manager::{
    CachedFileMetadataEntry, FileMetadata, FileMetadataCache,
};
use flatgeobuf::{
    FgbReader,
    packed_r_tree::{NodeItem, PackedRTree},
};
use object_store::{ObjectMeta, ObjectStore};
use std::{any::Any, io::Cursor, sync::Arc};

#[derive(Debug)]
struct CachedMetadata {
    data: Arc<Metadata>,
    store: Arc<dyn ObjectStore>,
}
impl FileMetadata for CachedMetadata {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn extra_info(&self) -> std::collections::HashMap<String, String> {
        Default::default()
    }
    fn memory_size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.data.offsets.capacity() * 8
            + self.data.header.capacity()
            + self.data.types.capacity() * std::mem::size_of::<flatgeobuf::ColumnType>()
    }
}
pub(crate) async fn metadata(
    store: Arc<dyn ObjectStore>,
    object: &ObjectMeta,
    hint: usize,
    cache: Option<&Arc<dyn FileMetadataCache>>,
) -> Result<Arc<Metadata>> {
    if let Some(entry) = cache.and_then(|c| c.get(&object.location))
        && entry.is_valid_for(object)
        && entry.meta.e_tag == object.e_tag
        && entry.meta.version == object.version
        && let Some(cached) = entry
            .file_metadata
            .as_any()
            .downcast_ref::<CachedMetadata>()
        && Arc::ptr_eq(&cached.store, &store)
    {
        return Ok(cached.data.clone());
    }
    let header = fetch_header(store.as_ref(), object, hint).await?;
    let reader = FgbReader::open(Cursor::new(header.clone()))
        .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf header: {e}"))?;
    let (mut data, count, node_size) =
        Metadata::from_header(&reader.header(), header[8..].to_vec())?;
    let index_begin = header.len() as u64;
    data.offsets = if data.indexed {
        indexed_offsets(store.as_ref(), object, index_begin, count, node_size).await?
    } else {
        unindexed_offsets(store.as_ref(), object, index_begin, count).await?
    };
    let data = Arc::new(data);
    if let Some(cache) = cache {
        cache.put(
            &object.location,
            CachedFileMetadataEntry::new(
                object.clone(),
                Arc::new(CachedMetadata {
                    data: data.clone(),
                    store,
                }),
            ),
        );
    }
    Ok(data)
}
/// Only the RTree's leaf level is required for sequential feature range ownership.
async fn indexed_offsets(
    store: &dyn ObjectStore,
    object: &ObjectMeta,
    index_begin: u64,
    count: usize,
    node_size: u16,
) -> Result<Vec<u64>> {
    if node_size < 2 || count > usize::MAX / 80 || count as u64 > object.size / 40 {
        return exec_err!("Invalid FlatGeobuf index dimensions");
    }
    let begin = index_begin
        .checked_add(PackedRTree::index_size(count, node_size) as u64)
        .filter(|p| *p <= object.size)
        .ok_or_else(|| exec_datafusion_err!("FlatGeobuf index exceeds object"))?;
    let leaf_begin = begin - count as u64 * 40;
    let mut offsets = Vec::with_capacity(count + 1);
    // Fetch whole leaf records in bounded requests, not one request per feature.
    let mut pos = leaf_begin;
    while pos < begin {
        let end = (pos + 40 * 1638).min(begin);
        let bytes = fetch_range(store, object, pos..end).await?;
        let mut cursor = Cursor::new(bytes);
        for _ in 0..(end - pos) / 40 {
            let node = NodeItem::from_reader(&mut cursor)
                .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf leaf: {e}"))?;
            let offset = begin
                .checked_add(node.offset)
                .filter(|p| *p < object.size)
                .ok_or_else(|| exec_datafusion_err!("FlatGeobuf feature offset outside object"))?;
            if offsets.last().is_some_and(|p| *p >= offset) {
                return exec_err!("FlatGeobuf feature offsets not increasing");
            }
            offsets.push(offset);
        }
        pos = end;
    }
    if offsets.first().copied() != Some(begin) {
        return exec_err!("First FlatGeobuf feature offset is not zero");
    }
    offsets.push(object.size);
    Ok(offsets)
}
/// No RTree: discover boundaries from length prefixes. A bounded window covers
/// nearby prefixes; large payloads are skipped without retaining them in metadata.
async fn unindexed_offsets(
    store: &dyn ObjectStore,
    object: &ObjectMeta,
    begin: u64,
    count: usize,
) -> Result<Vec<u64>> {
    let mut offsets = vec![];
    let mut pos = begin;
    let mut window = bytes::Bytes::new();
    let mut window_begin = begin;
    while pos < object.size {
        if object.size - pos < 4 {
            return exec_err!("Truncated FlatGeobuf feature prefix");
        }
        if pos < window_begin || pos + 4 > window_begin + window.len() as u64 {
            window_begin = pos;
            window = fetch_range(
                store,
                object,
                pos..(pos.saturating_add(65536)).min(object.size),
            )
            .await?;
        }
        let i = (pos - window_begin) as usize;
        let len = u32::from_le_bytes(window[i..i + 4].try_into().unwrap()) as u64;
        let end = pos
            .checked_add(4 + len)
            .filter(|end| *end <= object.size)
            .ok_or_else(|| exec_datafusion_err!("Truncated FlatGeobuf feature payload"))?;
        if len == 0 {
            return exec_err!("Invalid zero-length FlatGeobuf feature");
        }
        offsets.push(pos);
        pos = end;
    }
    if count != 0 && count != offsets.len() {
        return exec_err!("FlatGeobuf count does not match object");
    }
    offsets.push(object.size);
    Ok(offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::{ObjectStoreExt, memory::InMemory, path::Path};

    fn empty_file() -> bytes::Bytes {
        let mut writer = flatgeobuf::FgbWriter::create_with_options(
            "empty",
            flatgeobuf::GeometryType::Point,
            flatgeobuf::FgbWriterOptions {
                write_index: false,
                ..Default::default()
            },
        )
        .unwrap();
        writer.add_column("id", flatgeobuf::ColumnType::Int, |_, _| {});
        let mut bytes = Vec::new();
        writer.write(&mut bytes).unwrap();
        bytes.into()
    }

    #[tokio::test]
    async fn cache_keys_include_object_version_and_store_with_schema_options_applied_after_fetch() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("empty.fgb");
        store.put(&path, empty_file().into()).await.unwrap();
        let object = store.head(&path).await.unwrap();
        let runtime = datafusion_execution::runtime_env::RuntimeEnv::default();
        let cache = runtime.cache_manager.get_file_metadata_cache();
        let first = metadata(store.clone(), &object, 12, Some(&cache))
            .await
            .unwrap();
        let again = metadata(store.clone(), &object, 65536, Some(&cache))
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        let renamed = metadata(store.clone(), &object, 12, Some(&cache))
            .await
            .unwrap();
        assert_eq!(
            renamed
                .infer_schema("shape")
                .unwrap()
                .fields()
                .last()
                .unwrap()
                .name(),
            "shape"
        );
        assert!(Arc::ptr_eq(&first, &renamed));
        let mut new_version = object.clone();
        new_version.e_tag = Some("changed-etag".into());
        new_version.version = Some("changed-version".into());
        let versioned = metadata(store.clone(), &new_version, 12, Some(&cache))
            .await
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &versioned));
        store.put(&path, empty_file().into()).await.unwrap();
        let updated = store.head(&path).await.unwrap();
        let replacement = metadata(store.clone(), &updated, 12, Some(&cache))
            .await
            .unwrap();
        assert!(!Arc::ptr_eq(&renamed, &replacement));
        let other: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        other.put(&path, empty_file().into()).await.unwrap();
        let different = metadata(other, &updated, 12, Some(&cache)).await.unwrap();
        assert!(!Arc::ptr_eq(&replacement, &different));
    }
}
