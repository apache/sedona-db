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

use crate::metadata::{FileMetadata, MAGIC, Metadata, column_type, file_stamp};
use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion_common::{Result, ScalarValue, exec_datafusion_err, exec_err};
use flatgeobuf::{
    ColumnType, FallibleStreamingIterator, FeatureIter, FgbReader, Header, HeaderArgs, NotSeekable,
};
use sedona_datasource::spec::OpenReaderArgs;
use std::{
    io::{Cursor, Read, SeekFrom},
    sync::Arc,
};

type Stream = Box<dyn Read + Send>;
struct Reader {
    iter: FeatureIter<Stream, NotSeekable>,
    meta: Arc<Metadata>,
    projection: Vec<usize>,
    schema: SchemaRef,
    full_schema: SchemaRef,
    batch_size: usize,
    remaining: usize,
    failed: bool,
}
pub(crate) async fn open(
    meta: Arc<FileMetadata>,
    args: &OpenReaderArgs,
    geometry_column_name: &str,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    let full_schema = meta.infer_schema(geometry_column_name)?;
    if args
        .file_schema
        .as_ref()
        .is_some_and(|s| s.as_ref() != full_schema.as_ref())
    {
        return exec_err!("Requested FlatGeobuf schema differs from file schema");
    }
    let batch_size = args.batch_size.unwrap_or(8192);
    if batch_size == 0 {
        return exec_err!("FlatGeobuf batch size must be positive");
    }
    let projection = args
        .file_projection
        .clone()
        .unwrap_or_else(|| (0..full_schema.fields().len()).collect());
    full_schema.project(&projection)?;
    let (lo, hi) = feature_bounds(&meta, args)?;
    use tokio::io::AsyncSeekExt;
    let mut f = tokio::fs::File::open(&meta.path).await?;
    if file_stamp(&meta.path).await? != meta.stamp {
        return exec_err!("FlatGeobuf file changed before opening reader");
    }
    f.seek(SeekFrom::Start(meta.offsets[lo])).await?;
    let f = f.into_std().await;
    decode(
        Arc::new(meta.data.clone()),
        full_schema,
        Box::new(f.take(meta.offsets[hi] - meta.offsets[lo])),
        hi - lo,
        projection,
        batch_size,
    )
}

pub(crate) fn decode(
    meta: Arc<Metadata>,
    full_schema: SchemaRef,
    payload: Stream,
    count: usize,
    projection: Vec<usize>,
    batch_size: usize,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    if batch_size == 0 {
        return exec_err!("FlatGeobuf batch size must be positive");
    }
    let schema = Arc::new(full_schema.project(&projection)?);
    // A normalized unindexed header plus a bounded feature slice allows the
    // released crate's verified decoder to read a partition without a fork.
    let mut builder = flatbuffers::FlatBufferBuilder::new();
    let h = Header::create(
        &mut builder,
        &HeaderArgs {
            geometry_type: meta.geometry_type,
            has_z: meta.has_z,
            has_m: meta.has_m,
            features_count: count as u64,
            index_node_size: 0,
            ..Default::default()
        },
    );
    builder.finish_size_prefixed(h, None);
    let mut header = MAGIC.to_vec();
    header.extend_from_slice(builder.finished_data());
    let stream: Stream = Box::new(Cursor::new(header).chain(payload));
    let iter = FgbReader::open(stream)
        .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf partition header: {e}"))?
        .select_all_seq()
        .map_err(|e| exec_datafusion_err!("Cannot open FlatGeobuf partition: {e}"))?;
    Ok(Box::new(Reader {
        iter,
        meta,
        projection,
        schema,
        full_schema,
        batch_size,
        remaining: count,
        failed: false,
    }))
}
/// Assign features by their size-prefix start so byte cuts never split ownership.
/// The unindexed fallback assigns all rows only to the partition owning byte zero.
fn feature_bounds(meta: &FileMetadata, args: &OpenReaderArgs) -> Result<(usize, usize)> {
    let count = meta.offsets.len() - 1;
    let bounds = if let Some(r) = &args.src.range {
        if r.start < 0 || r.end < r.start {
            return exec_err!("Invalid FlatGeobuf byte range");
        }
        if !meta.indexed {
            if r.start == 0 && r.end > 0 {
                (0, count)
            } else {
                (0, 0)
            }
        } else {
            let starts = &meta.offsets[..count];
            (
                starts.partition_point(|p| *p < r.start as u64),
                starts.partition_point(|p| *p < r.end as u64),
            )
        }
    } else {
        (0, count)
    };
    Ok(bounds)
}
impl RecordBatchReader for Reader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
impl Iterator for Reader {
    type Item = std::result::Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.remaining == 0 {
            return None;
        }
        let result = self.read_batch();
        if result.is_err() {
            self.failed = true;
        }
        Some(result.map_err(|e| ArrowError::ExternalError(Box::new(e))))
    }
}
impl Reader {
    fn read_batch(&mut self) -> Result<RecordBatch> {
        let count = self.remaining.min(self.batch_size);
        let mut columns: Vec<Vec<ScalarValue>> = self
            .projection
            .iter()
            .map(|_| Vec::with_capacity(count))
            .collect();
        for _ in 0..count {
            let feature = self
                .iter
                .next()
                .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf feature: {e}"))?
                .ok_or_else(|| exec_datafusion_err!("Truncated FlatGeobuf partition"))?;
            if feature.fbs_feature().columns().is_some() {
                return exec_err!("FlatGeobuf feature-local schemas are not supported");
            }
            let mut row = self
                .projection
                .iter()
                .map(|i| ScalarValue::try_from(self.full_schema.field(*i).data_type()))
                .collect::<Result<Vec<_>>>()?;
            if self.projection.iter().any(|i| *i < self.meta.types.len())
                && let Some(props) = feature.fbs_feature().properties()
            {
                decode_properties(
                    props.bytes(),
                    &self.meta,
                    &self.full_schema,
                    &self.projection,
                    &mut row,
                )?;
            }
            if self.projection.contains(&self.meta.types.len()) {
                let value = if let Some(g) = feature.geometry() {
                    ScalarValue::Binary(Some(crate::geometry::to_wkb(
                        g,
                        self.meta.geometry_type,
                        self.meta.has_z,
                        self.meta.has_m,
                    )?))
                } else {
                    ScalarValue::Binary(None)
                };
                for (i, col) in self.projection.iter().enumerate() {
                    if *col == self.meta.types.len() {
                        row[i] = value.clone();
                    }
                }
            }
            for (i, v) in row.into_iter().enumerate() {
                columns[i].push(v);
            }
        }
        self.remaining -= count;
        let arrays = columns
            .into_iter()
            .map(ScalarValue::iter_to_array)
            .collect::<Result<Vec<ArrayRef>>>()?;
        Ok(RecordBatch::try_new_with_options(
            self.schema.clone(),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(count)),
        )?)
    }
}
fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    if bytes.len() < n {
        return exec_err!("Truncated FlatGeobuf property value");
    }
    let (value, rest) = bytes.split_at(n);
    *bytes = rest;
    Ok(value)
}
fn decode_properties(
    mut bytes: &[u8],
    meta: &Metadata,
    full_schema: &SchemaRef,
    projection: &[usize],
    row: &mut [ScalarValue],
) -> Result<()> {
    let mut seen = vec![false; meta.types.len()];
    while !bytes.is_empty() {
        let idx = u16::from_le_bytes(take(&mut bytes, 2)?.try_into().unwrap()) as usize;
        if idx >= meta.types.len() || seen[idx] {
            return exec_err!("Invalid or duplicate FlatGeobuf property index");
        }
        seen[idx] = true;
        let t = meta.types[idx];
        macro_rules! number {
            ($ty:ty,$variant:ident) => {{
                let raw = take(&mut bytes, std::mem::size_of::<$ty>())?;
                ScalarValue::$variant(Some(<$ty>::from_le_bytes(raw.try_into().unwrap())))
            }};
        }
        let value = match t {
            ColumnType::Byte => number!(i8, Int8),
            ColumnType::UByte => number!(u8, UInt8),
            ColumnType::Bool => {
                let v = take(&mut bytes, 1)?[0];
                if v > 1 {
                    return exec_err!("Invalid FlatGeobuf boolean");
                }
                ScalarValue::Boolean(Some(v != 0))
            }
            ColumnType::Short => number!(i16, Int16),
            ColumnType::UShort => number!(u16, UInt16),
            ColumnType::Int => number!(i32, Int32),
            ColumnType::UInt => number!(u32, UInt32),
            ColumnType::Long => number!(i64, Int64),
            ColumnType::ULong => number!(u64, UInt64),
            ColumnType::Float => number!(f32, Float32),
            ColumnType::Double => number!(f64, Float64),
            ColumnType::String | ColumnType::Json | ColumnType::DateTime | ColumnType::Binary => {
                let len = u32::from_le_bytes(take(&mut bytes, 4)?.try_into().unwrap()) as usize;
                let data = take(&mut bytes, len)?;
                if t == ColumnType::Binary {
                    ScalarValue::Binary(Some(data.to_vec()))
                } else {
                    let text = std::str::from_utf8(data)
                        .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf UTF-8: {e}"))?;
                    if t == ColumnType::DateTime {
                        let datetime = chrono::DateTime::parse_from_rfc3339(text).map_err(|e| {
                            exec_datafusion_err!("Invalid FlatGeobuf DateTime: {e}")
                        })?;
                        ScalarValue::TimestampMicrosecond(
                            Some(datetime.timestamp_micros()),
                            Some("UTC".into()),
                        )
                    } else {
                        ScalarValue::Utf8(Some(text.into()))
                    }
                }
            }
            _ => {
                column_type(t)?;
                unreachable!()
            }
        };
        for (i, col) in projection.iter().enumerate() {
            if *col == idx {
                row[i] = value.clone();
            }
        }
    }
    for (i, col) in projection.iter().enumerate() {
        if !full_schema.field(*col).is_nullable() && row[i].is_null() {
            return exec_err!(
                "Missing non-nullable FlatGeobuf property: {}",
                full_schema.field(*col).name()
            );
        }
    }
    Ok(())
}
