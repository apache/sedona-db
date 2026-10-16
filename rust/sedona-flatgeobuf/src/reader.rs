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
    converter::FlatGeobufArrowConverter,
    metadata::{FileMetadata, Metadata, file_stamp},
};
use arrow_array::{RecordBatch, RecordBatchOptions, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use flatgeobuf::{ColumnType, size_prefixed_root_as_feature};
use sedona_datasource::spec::OpenReaderArgs;
use std::{
    io::{Read, SeekFrom},
    sync::Arc,
};

/// Compatibility adapter for ExternalFormatSpec. Native scans use the async
/// object-store stream and the byte-slice decoder directly.
struct Reader {
    payload: Box<dyn Read + Send>,
    meta: Arc<Metadata>,
    full_schema: SchemaRef,
    projection: Vec<usize>,
    schema: SchemaRef,
    geometry_column_name: String,
    remaining: usize,
    next_feature: usize,
    batch_size: usize,
    failed: bool,
}
pub(crate) async fn open(
    meta: Arc<FileMetadata>,
    args: &OpenReaderArgs,
    geometry_column_name: &str,
) -> Result<Box<dyn RecordBatchReader + Send>> {
    let full_schema = args
        .file_schema
        .clone()
        .unwrap_or(meta.infer_schema(geometry_column_name)?);
    let batch_size = args.batch_size.unwrap_or(8192);
    if batch_size == 0 {
        return exec_err!("FlatGeobuf batch size must be positive");
    }
    let projection = args
        .file_projection
        .clone()
        .unwrap_or_else(|| (0..full_schema.fields().len()).collect());
    let bounds = crate::opener::feature_bounds(&meta.data, args.src.range.as_ref())?;
    use tokio::io::AsyncSeekExt;
    let mut file = tokio::fs::File::open(&meta.path).await?;
    if file_stamp(&meta.path).await? != meta.stamp {
        return exec_err!("FlatGeobuf file changed before opening reader");
    }
    file.seek(SeekFrom::Start(meta.offsets[bounds.0])).await?;
    let file = file.into_std().await;
    Ok(Box::new(Reader {
        payload: Box::new(file.take(meta.offsets[bounds.1] - meta.offsets[bounds.0])),
        meta: Arc::new(meta.data.clone()),
        schema: Arc::new(full_schema.project(&projection)?),
        full_schema,
        projection,
        geometry_column_name: geometry_column_name.into(),
        remaining: bounds.1 - bounds.0,
        next_feature: bounds.0,
        batch_size,
        failed: false,
    }))
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
        self.failed = result.is_err();
        Some(result.map_err(|e| ArrowError::ExternalError(Box::new(e))))
    }
}
impl Reader {
    fn read_batch(&mut self) -> Result<RecordBatch> {
        let count = self.remaining.min(self.batch_size);
        let mut payload = Vec::new();
        let mut offsets = vec![0];
        for _ in 0..count {
            let mut prefix = [0u8; 4];
            self.payload.read_exact(&mut prefix)?;
            let len = u32::from_le_bytes(prefix) as usize;
            if len == 0 {
                return exec_err!("Empty FlatGeobuf feature frame");
            }
            let start = payload.len();
            let feature = self.next_feature + offsets.len() - 1;
            if len as u64 + 4 != self.meta.offsets[feature + 1] - self.meta.offsets[feature] {
                return exec_err!("FlatGeobuf index does not match feature framing");
            }
            payload.extend_from_slice(&prefix);
            payload.resize(start + 4 + len, 0);
            self.payload.read_exact(&mut payload[start + 4..])?;
            offsets.push(payload.len() as u64);
        }
        let batch = decode_batch(
            &self.meta,
            &self.full_schema,
            &self.projection,
            &payload,
            &offsets,
            0,
            &self.geometry_column_name,
        )?;
        self.remaining -= count;
        self.next_feature += count;
        Ok(batch)
    }
}

/// Decode verified size-prefixed features from the fetched byte slice directly.
/// Output columns are matched by name, allowing reordered, missing and widened
/// table fields. Schema metadata is supplied by the planned table schema.
pub(crate) fn decode_batch(
    meta: &Metadata,
    full_schema: &SchemaRef,
    projection: &[usize],
    bytes: &[u8],
    offsets: &[u64],
    begin: u64,
    geometry_column_name: &str,
) -> Result<RecordBatch> {
    let schema = Arc::new(full_schema.project(projection)?);
    let source_schema = meta.infer_schema(geometry_column_name)?;
    // The output geometry name is configurable; identify it by its GeoArrow
    // extension rather than assuming it is the last table field.
    let positional = full_schema.fields().len() == source_schema.fields().len()
        && full_schema
            .fields()
            .iter()
            .zip(source_schema.fields())
            .all(|(a, b)| a.name() == b.name());
    let sources = projection
        .iter()
        .map(|i| {
            let field = full_schema.field(*i);
            if field
                .metadata()
                .get("ARROW:extension:name")
                .is_some_and(|v| v == "geoarrow.wkb")
            {
                let source = source_schema.field(meta.types.len());
                if field.metadata().get("ARROW:extension:metadata")
                    != source.metadata().get("ARROW:extension:metadata")
                    || field.data_type() != source.data_type()
                {
                    return exec_err!("Requested FlatGeobuf geometry/CRS differs from file schema");
                }
                Ok(Some(meta.types.len()))
            } else if positional {
                Ok(Some(*i))
            } else {
                Ok(source_schema
                    .fields()
                    .iter()
                    .position(|s| s.name() == field.name()))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let count = offsets.len().saturating_sub(1);
    let mut columns = schema
        .fields()
        .iter()
        .zip(&sources)
        .map(|(field, source)| {
            let ty = source.map(|i| {
                if i == meta.types.len() {
                    ColumnType::Binary
                } else {
                    meta.types[i]
                }
            });
            FlatGeobufArrowConverter::new(field.clone(), ty, count)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut seen = vec![false; meta.types.len()];
    let mut appended = vec![false; columns.len()];
    for pair in offsets.windows(2) {
        let start = usize::try_from(
            pair[0]
                .checked_sub(begin)
                .ok_or_else(|| exec_datafusion_err!("Invalid feature offset"))?,
        )
        .map_err(|_| exec_datafusion_err!("FlatGeobuf offset overflow"))?;
        let end = usize::try_from(
            pair[1]
                .checked_sub(begin)
                .ok_or_else(|| exec_datafusion_err!("Invalid feature offset"))?,
        )
        .map_err(|_| exec_datafusion_err!("FlatGeobuf offset overflow"))?;
        let frame = bytes
            .get(start..end)
            .ok_or_else(|| exec_datafusion_err!("Truncated FlatGeobuf feature"))?;
        if frame.len() < 4
            || u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize != frame.len() - 4
        {
            return exec_err!("FlatGeobuf index does not match feature framing");
        }
        let feature = size_prefixed_root_as_feature(frame)
            .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf feature: {e}"))?;
        if feature.columns().is_some() {
            return exec_err!("FlatGeobuf feature-local schemas are not supported");
        }
        seen.fill(false);
        appended.fill(false);
        if let Some(props) = feature.properties() {
            let mut remaining = props.bytes();
            while !remaining.is_empty() {
                let i = u16::from_le_bytes(take(&mut remaining, 2)?.try_into().unwrap()) as usize;
                if i >= meta.types.len() || seen[i] {
                    return exec_err!("Invalid or duplicate FlatGeobuf property index");
                }
                seen[i] = true;
                let ty = meta.types[i];
                let raw = property_bytes(&mut remaining, ty)?;
                for (j, source) in sources.iter().enumerate() {
                    if *source == Some(i) {
                        columns[j].append(ty, raw)?;
                        appended[j] = true;
                    }
                }
            }
        }
        if sources.contains(&Some(meta.types.len())) {
            let wkb = feature
                .geometry()
                .map(|g| crate::geometry::to_wkb(g, meta.geometry_type, meta.has_z, meta.has_m))
                .transpose()?;
            for (j, source) in sources.iter().enumerate() {
                if *source == Some(meta.types.len()) {
                    if let Some(wkb) = &wkb {
                        columns[j].append(ColumnType::Binary, wkb)?;
                    } else {
                        columns[j].append_null();
                    }
                    appended[j] = true;
                }
            }
        }
        for (c, appended) in columns.iter_mut().zip(&appended) {
            if !appended {
                c.append_null();
            }
        }
    }
    let arrays = columns
        .into_iter()
        .map(FlatGeobufArrowConverter::finish)
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new_with_options(
        schema,
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(count)),
    )?)
}
fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    if bytes.len() < n {
        return exec_err!("Truncated FlatGeobuf property value");
    }
    let (value, rest) = bytes.split_at(n);
    *bytes = rest;
    Ok(value)
}
fn property_bytes<'a>(bytes: &mut &'a [u8], ty: ColumnType) -> Result<&'a [u8]> {
    let len = match ty {
        ColumnType::Byte | ColumnType::UByte | ColumnType::Bool => 1,
        ColumnType::Short | ColumnType::UShort => 2,
        ColumnType::Int | ColumnType::UInt | ColumnType::Float => 4,
        ColumnType::Long | ColumnType::ULong | ColumnType::Double => 8,
        ColumnType::String | ColumnType::Json | ColumnType::DateTime | ColumnType::Binary => {
            u32::from_le_bytes(take(bytes, 4)?.try_into().unwrap()) as usize
        }
        _ => return exec_err!("Unsupported FlatGeobuf property type"),
    };
    take(bytes, len)
}
