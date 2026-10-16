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

use bytes::Bytes;
use datafusion_common::{Result, exec_err};
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt};
use std::ops::Range;

/// Prefix reads include magic, the size prefix and the verified header buffer.
/// A too-small hint is followed by exactly the missing header bytes.
pub(crate) async fn fetch_header(
    store: &dyn ObjectStore,
    object: &ObjectMeta,
    hint: usize,
) -> Result<Bytes> {
    if object.size < 12 {
        return exec_err!("Truncated FlatGeobuf header prefix");
    }
    // Match the released decoder's header-size bound before any large fetch.
    const MAX_HEADER_BYTES: u64 = 10 * 1024 * 1024;
    let first_end = (hint.max(12) as u64)
        .min(object.size)
        .min(MAX_HEADER_BYTES + 12);
    let first = store.get_range(&object.location, 0..first_end).await?;
    if first.len() as u64 != first_end {
        return exec_err!("Truncated FlatGeobuf header range");
    }
    // The eighth byte is the minor version (GDAL writes 1). Match the
    // released decoder's compatibility check rather than requiring minor zero.
    if first[..3] != crate::metadata::MAGIC[..3]
        || first[4..7] != crate::metadata::MAGIC[4..7]
        || first[3] > flatgeobuf::VERSION
    {
        return exec_err!("Invalid FlatGeobuf magic");
    }
    let size = u32::from_le_bytes(first[8..12].try_into().unwrap()) as u64;
    let end = 12 + size;
    if !(8..=MAX_HEADER_BYTES).contains(&size) || end > object.size {
        return exec_err!("Invalid FlatGeobuf header length");
    }
    let result = if end <= first_end {
        first.slice(..end as usize)
    } else {
        let rest = store.get_range(&object.location, first_end..end).await?;
        if rest.len() as u64 != end - first_end {
            return exec_err!("Truncated FlatGeobuf header range");
        }
        let mut bytes = Vec::with_capacity(end as usize);
        bytes.extend_from_slice(&first);
        bytes.extend_from_slice(&rest);
        Bytes::from(bytes)
    };
    flatgeobuf::FgbReader::open(std::io::Cursor::new(result.clone()))
        .map_err(|e| datafusion_common::exec_datafusion_err!("Invalid FlatGeobuf header: {e}"))?;
    Ok(result)
}

/// Coalesce adjacent, disjoint feature ranges without reading gaps. Bound each
/// request's byte size; a single feature exceeding the target is kept intact.
pub(crate) fn coalesce_ranges(
    ranges: &[Range<u64>],
    file_size: u64,
    target_bytes: u64,
) -> Result<Vec<Range<u64>>> {
    if target_bytes == 0 {
        return exec_err!("FlatGeobuf range target must be positive");
    }
    let mut ranges = ranges.to_vec();
    for range in &ranges {
        if range.start > range.end || range.end > file_size {
            return exec_err!("Invalid FlatGeobuf object range");
        }
    }
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut result: Vec<Range<u64>> = vec![];
    for range in ranges {
        if range.is_empty() {
            continue;
        }
        if let Some(last) = result.last_mut() {
            let end = last.end.max(range.end);
            if range.start < last.end {
                return exec_err!("Overlapping FlatGeobuf feature ranges");
            }
            // The target is soft only for a single indivisible input feature.
            if range.start == last.end && end - last.start <= target_bytes {
                last.end = end;
                continue;
            }
        }
        result.push(range);
    }
    Ok(result)
}

/// Fetch exactly the requested range and reject short responses before decoding.
pub(crate) async fn fetch_range(
    store: &dyn ObjectStore,
    object: &ObjectMeta,
    range: Range<u64>,
) -> Result<Bytes> {
    if range.start > range.end || range.end > object.size {
        return exec_err!("Invalid FlatGeobuf object range");
    }
    let expected = range.end - range.start;
    let bytes = store.get_range(&object.location, range).await?;
    if bytes.len() as u64 != expected {
        return exec_err!("Truncated FlatGeobuf object range");
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::{memory::InMemory, path::Path};
    #[test]
    fn coalescing_preserves_coverage_and_bounds_requests() {
        let input = [12..20, 0..4, 4..8, 8..12, 20..24, 30..35];
        let out = coalesce_ranges(&input, 35, 12).unwrap();
        assert_eq!(out, [0..12, 12..24, 30..35]);
        let covered = |ranges: &[Range<u64>]| {
            ranges
                .iter()
                .flat_map(Clone::clone)
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(covered(&input), covered(&out));
        assert!(out.iter().all(|r| r.end - r.start <= 12));
        assert!(coalesce_ranges(&[0..8, 4..12], 12, 8).is_err());
        assert!(coalesce_ranges(&[std::ops::Range { start: 4, end: 2 }], 10, 4).is_err());
        assert!(coalesce_ranges(std::slice::from_ref(&(0..11)), 10, 4).is_err());
        assert!(coalesce_ranges(std::slice::from_ref(&(0..4)), 10, 0).is_err());
        assert!(
            coalesce_ranges(std::slice::from_ref(&(4..4)), 10, 4)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            coalesce_ranges(&[0..20, 20..24], 24, 8).unwrap(),
            [0..20, 20..24]
        );
    }
    #[tokio::test]
    async fn header_hint_refetch_on_generic_store() {
        let store = InMemory::new();
        let path = Path::from("test.fgb");
        let mut b = flatbuffers::FlatBufferBuilder::new();
        let name = b.create_string(&"long-name".repeat(20));
        let h = flatgeobuf::Header::create(
            &mut b,
            &flatgeobuf::HeaderArgs {
                name: Some(name),
                index_node_size: 0,
                ..Default::default()
            },
        );
        b.finish_size_prefixed(h, None);
        let mut bytes = crate::metadata::MAGIC.to_vec();
        bytes.extend_from_slice(b.finished_data());
        bytes[7] = 1; // GDAL minor version, also accepted by the released decoder.
        let header_len = bytes.len();
        bytes.extend_from_slice(&[99; 100]);
        store.put(&path, bytes.clone().into()).await.unwrap();
        let object = store.head(&path).await.unwrap();
        for hint in [0, 1, 12, header_len, header_len + 1000] {
            let result = fetch_header(&store, &object, hint).await.unwrap();
            assert_eq!(result.as_ref(), &bytes[..header_len]);
            assert!(flatgeobuf::FgbReader::open(std::io::Cursor::new(result)).is_ok());
        }
        store.put(&path, bytes[..15].to_vec().into()).await.unwrap();
        let object = store.head(&path).await.unwrap();
        assert!(fetch_header(&store, &object, 12).await.is_err());
        let mut malicious = crate::metadata::MAGIC.to_vec();
        malicious.extend_from_slice(&u32::MAX.to_le_bytes());
        store.put(&path, malicious.into()).await.unwrap();
        let mut object = store.head(&path).await.unwrap();
        object.size = u64::MAX;
        let error = fetch_header(&store, &object, 12).await.unwrap_err();
        assert!(error.to_string().contains("header length"));
    }
}
