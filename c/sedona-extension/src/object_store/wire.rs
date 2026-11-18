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

use std::io;
use std::sync::Arc;

use arrow_array::{Array, BinaryArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use chrono::{DateTime, Utc};
use object_store::{path::Path, Attribute, Attributes, GetOptions, GetRange, ObjectMeta};
use serde::{Deserialize, Serialize};

use super::bounded::{failure, invalid, unsupported};
use super::ffi::{OBJECT_STORE_MAX_BYTES, OBJECT_STORE_MAX_ROWS};

#[derive(Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Request {
    Get {
        location: String,
        options: ReadOptions,
    },
    List {
        prefix: Option<String>,
        offset: Option<String>,
        delimiter: bool,
    },
}
#[derive(Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReadOptions {
    pub if_match: Option<String>,
    pub if_none_match: Option<String>,
    pub if_modified_since: Option<String>,
    pub if_unmodified_since: Option<String>,
    pub range: Option<ReadRange>,
    pub version: Option<String>,
    pub head: bool,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ReadRange {
    Bounded { start: u64, end: u64 },
    Offset { offset: u64 },
    Suffix { length: u64 },
}
fn parse_time(value: &str) -> object_store::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|time| time.with_timezone(&Utc))
        .map_err(|e| invalid(e.to_string()))
}
impl TryFrom<GetOptions> for ReadOptions {
    type Error = object_store::Error;
    fn try_from(options: GetOptions) -> Result<Self, Self::Error> {
        if !options.extensions.is_empty() {
            return Err(unsupported("GetOptions extensions cannot cross the C ABI"));
        }
        Ok(Self {
            if_match: options.if_match,
            if_none_match: options.if_none_match,
            if_modified_since: options.if_modified_since.map(|time| time.to_rfc3339()),
            if_unmodified_since: options.if_unmodified_since.map(|time| time.to_rfc3339()),
            range: options.range.map(|range| match range {
                GetRange::Bounded(range) => ReadRange::Bounded {
                    start: range.start,
                    end: range.end,
                },
                GetRange::Offset(offset) => ReadRange::Offset { offset },
                GetRange::Suffix(length) => ReadRange::Suffix { length },
            }),
            version: options.version,
            head: options.head,
        })
    }
}
impl TryFrom<ReadOptions> for GetOptions {
    type Error = object_store::Error;
    fn try_from(options: ReadOptions) -> Result<Self, Self::Error> {
        let range = options.range.map(|range| match range {
            ReadRange::Bounded { start, end } => GetRange::Bounded(start..end),
            ReadRange::Offset { offset } => GetRange::Offset(offset),
            ReadRange::Suffix { length } => GetRange::Suffix(length),
        });
        if let Some(range) = &range {
            range.is_valid().map_err(|e| invalid(e.to_string()))?;
        }
        Ok(Self {
            if_match: options.if_match,
            if_none_match: options.if_none_match,
            if_modified_since: options
                .if_modified_since
                .as_deref()
                .map(parse_time)
                .transpose()?,
            if_unmodified_since: options
                .if_unmodified_since
                .as_deref()
                .map(parse_time)
                .transpose()?,
            range,
            version: options.version,
            head: options.head,
            ..Default::default()
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Metadata {
    pub location: String,
    pub last_modified: String,
    pub size: u64,
    pub e_tag: Option<String>,
    pub version: Option<String>,
}
impl From<ObjectMeta> for Metadata {
    fn from(meta: ObjectMeta) -> Self {
        Self {
            location: meta.location.to_string(),
            last_modified: meta.last_modified.to_rfc3339(),
            size: meta.size,
            e_tag: meta.e_tag,
            version: meta.version,
        }
    }
}
impl TryFrom<Metadata> for ObjectMeta {
    type Error = object_store::Error;
    fn try_from(meta: Metadata) -> Result<Self, Self::Error> {
        Ok(Self {
            location: Path::parse(meta.location)?,
            last_modified: parse_time(&meta.last_modified)?,
            size: meta.size,
            e_tag: meta.e_tag,
            version: meta.version,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireAttribute {
    ContentDisposition,
    ContentEncoding,
    ContentLanguage,
    ContentType,
    CacheControl,
    StorageClass,
    Metadata(String),
}
impl TryFrom<&Attribute> for WireAttribute {
    type Error = object_store::Error;
    fn try_from(attr: &Attribute) -> Result<Self, Self::Error> {
        Ok(match attr {
            Attribute::ContentDisposition => Self::ContentDisposition,
            Attribute::ContentEncoding => Self::ContentEncoding,
            Attribute::ContentLanguage => Self::ContentLanguage,
            Attribute::ContentType => Self::ContentType,
            Attribute::CacheControl => Self::CacheControl,
            Attribute::StorageClass => Self::StorageClass,
            Attribute::Metadata(key) => Self::Metadata(key.to_string()),
            _ => return Err(unsupported("unknown ObjectStore attribute")),
        })
    }
}
impl From<WireAttribute> for Attribute {
    fn from(attr: WireAttribute) -> Self {
        match attr {
            WireAttribute::ContentDisposition => Self::ContentDisposition,
            WireAttribute::ContentEncoding => Self::ContentEncoding,
            WireAttribute::ContentLanguage => Self::ContentLanguage,
            WireAttribute::ContentType => Self::ContentType,
            WireAttribute::CacheControl => Self::CacheControl,
            WireAttribute::StorageClass => Self::StorageClass,
            WireAttribute::Metadata(key) => Self::Metadata(key.into()),
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReadMetadata {
    pub meta: Metadata,
    pub range_start: u64,
    pub range_end: u64,
    pub attributes: Vec<(WireAttribute, String)>,
}
impl ReadMetadata {
    pub fn attributes(attrs: &Attributes) -> object_store::Result<Vec<(WireAttribute, String)>> {
        attrs
            .iter()
            .map(|(key, value)| Ok((key.try_into()?, value.to_string())))
            .collect()
    }
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ListEntry {
    Object { meta: Metadata },
    Prefix { path: String },
}

/// Storage codes are platform-independent; preserve them in both call and stream errors.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireError {
    pub code: i32,
    pub path: String,
    pub message: String,
}
impl WireError {
    pub fn from_error(error: object_store::Error) -> Self {
        use object_store::Error::*;
        let (code, path) = match &error {
            NotFound { path, .. } => (2, path.clone()),
            NotSupported { .. } | NotImplemented { .. } => (3, String::new()),
            AlreadyExists { path, .. } => (4, path.clone()),
            Precondition { path, .. } => (5, path.clone()),
            NotModified { path, .. } => (6, path.clone()),
            PermissionDenied { path, .. } => (7, path.clone()),
            Unauthenticated { path, .. } => (8, path.clone()),
            InvalidPath { .. } => (9, String::new()),
            Generic { source, .. } => (
                source
                    .downcast_ref::<io::Error>()
                    .map(|e| match e.kind() {
                        io::ErrorKind::Interrupted => 10,
                        io::ErrorKind::TimedOut => 11,
                        io::ErrorKind::WouldBlock => 12,
                        io::ErrorKind::InvalidData | io::ErrorKind::InvalidInput => 9,
                        _ => 1,
                    })
                    .unwrap_or(1),
                String::new(),
            ),
            _ => (1, String::new()),
        };
        Self {
            code,
            path,
            message: error.to_string(),
        }
    }
    pub fn into_error(self) -> object_store::Error {
        use object_store::Error::*;
        let path = self.path;
        let source = self.message.clone().into();
        match self.code {
            2 => NotFound { path, source },
            3 => NotSupported { source },
            4 => AlreadyExists { path, source },
            5 => Precondition { path, source },
            6 => NotModified { path, source },
            7 => PermissionDenied { path, source },
            8 => Unauthenticated { path, source },
            9 => failure(io::ErrorKind::InvalidData, self.message),
            10 => failure(io::ErrorKind::Interrupted, self.message),
            11 => failure(io::ErrorKind::TimedOut, self.message),
            12 => failure(io::ErrorKind::WouldBlock, self.message),
            _ => Generic {
                store: "SedonaCObjectStore",
                source,
            },
        }
    }
}

pub(crate) fn encode<T: Serialize>(value: &T) -> object_store::Result<Vec<u8>> {
    let bytes = serde_json::to_vec(value).map_err(|e| invalid(e.to_string()))?;
    if bytes.len() > OBJECT_STORE_MAX_BYTES {
        return Err(invalid("ObjectStore wire payload exceeds 1 MiB"));
    }
    Ok(bytes)
}
pub(crate) fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> object_store::Result<T> {
    if bytes.len() > OBJECT_STORE_MAX_BYTES {
        return Err(invalid("ObjectStore wire payload exceeds 1 MiB"));
    }
    serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))
}

pub(crate) fn schema(kind: &str) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("payload", DataType::Binary, true),
            Field::new("error", DataType::Utf8, true),
        ],
        [
            ("sedona.object_store.version".into(), "1".into()),
            ("sedona.object_store.kind".into(), kind.into()),
        ]
        .into(),
    ))
}
pub(crate) fn payload_batch(schema: SchemaRef, values: &[Vec<u8>]) -> RecordBatch {
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(BinaryArray::from_iter_values(
                values.iter().map(Vec::as_slice),
            )),
            Arc::new(StringArray::from(vec![None::<&str>; values.len()])),
        ],
    )
    .expect("fixed wire schema")
}
pub(crate) fn error_batch(schema: SchemaRef, error: object_store::Error) -> RecordBatch {
    let mut error = WireError::from_error(error);
    // Keep error messages bounded as well; never truncate in the middle of UTF-8.
    if error.message.len() > 8192 {
        let mut end = 8192;
        while !error.message.is_char_boundary(end) {
            end -= 1;
        }
        error.message.truncate(end);
    }
    let error = serde_json::to_string(&error).expect("serializable error");
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(BinaryArray::from(vec![None::<&[u8]>])),
            Arc::new(StringArray::from(vec![Some(error.as_str())])),
        ],
    )
    .expect("fixed wire schema")
}
/// Copy while still on the foreign executor: returning Arrow buffers would let a
/// later Bytes drop invoke a foreign release callback on the query thread.
pub(crate) fn decode_batch(
    batch: RecordBatch,
    expected: &SchemaRef,
) -> object_store::Result<Vec<Vec<u8>>> {
    if batch.schema() != *expected || batch.num_rows() > OBJECT_STORE_MAX_ROWS {
        return Err(invalid("invalid ObjectStore stream schema or batch size"));
    }
    let payload = batch
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| invalid("expected binary payload"))?;
    let errors = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| invalid("expected string errors"))?;
    let mut size = 0usize;
    let mut values = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        if payload.is_null(i) == errors.is_null(i) {
            return Err(invalid("expected exactly one payload or error"));
        }
        if !errors.is_null(i) {
            if batch.num_rows() != 1 {
                return Err(invalid("error batch must have one row"));
            }
            return Err(decode::<WireError>(errors.value(i).as_bytes())?.into_error());
        }
        size = size
            .checked_add(payload.value_length(i) as usize)
            .ok_or_else(|| invalid("payload size overflow"))?;
        if size > OBJECT_STORE_MAX_BYTES {
            return Err(invalid("stream batch exceeds 1 MiB"));
        }
        values.push(payload.value(i).to_vec());
    }
    Ok(values)
}
