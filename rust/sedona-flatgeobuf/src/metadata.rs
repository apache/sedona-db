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

use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use flatgeobuf::{
    ColumnType, FgbReader, GeometryType,
    packed_r_tree::{NodeItem, PackedRTree},
};
use sedona_datasource::spec::Object;
use sedona_schema::{
    crs::deserialize_crs,
    datatypes::{Edges, SedonaType},
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

pub(crate) const MAGIC: [u8; 8] = *b"fgb\x03fgb\0";
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct FileStamp {
    pub size: u64,
    modified: SystemTime,
}
pub(crate) fn file_stamp(path: &Path) -> Result<FileStamp> {
    let m = std::fs::metadata(path)?;
    Ok(FileStamp {
        size: m.len(),
        modified: m.modified()?,
    })
}
pub(crate) fn local_path(object: &Object) -> Result<PathBuf> {
    let value = object
        .to_url_string()
        .ok_or_else(|| exec_datafusion_err!("FlatGeobuf requires a local file URL"))?;
    let url =
        url::Url::parse(&value).map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf URL: {e}"))?;
    url.to_file_path()
        .map_err(|_| exec_datafusion_err!("FlatGeobuf currently supports only local file URLs"))
}
#[derive(Debug)]
pub(crate) struct FileMetadata {
    pub path: PathBuf,
    pub stamp: FileStamp,
    pub schema: SchemaRef,
    pub types: Vec<ColumnType>,
    pub geometry_type: GeometryType,
    pub has_z: bool,
    pub has_m: bool,
    pub indexed: bool,
    // Absolute starts, followed by the end of the feature section.
    pub offsets: Vec<u64>,
}
impl FileMetadata {
    pub fn read(path: &Path, stamp: FileStamp) -> Result<Self> {
        let mut file = File::open(path)?;
        let fgb = FgbReader::open(&mut file)
            .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf header: {e}"))?;
        let h = fgb.header();
        if h.has_t() || h.has_tm() {
            return exec_err!("FlatGeobuf T/TM dimensions are not supported");
        }
        if h.geometry_type().0 > GeometryType::GeometryCollection.0 {
            return exec_err!("FlatGeobuf curved geometry types are not supported");
        }
        let geometry_type = h.geometry_type();
        let has_z = h.has_z();
        let has_m = h.has_m();
        let count = usize::try_from(h.features_count())
            .map_err(|_| exec_datafusion_err!("FlatGeobuf count overflow"))?;
        let node_size = h.index_node_size();
        let indexed = node_size > 0 && count > 0;
        let mut fields = vec![];
        let mut types = vec![];
        if let Some(columns) = h.columns() {
            for col in columns {
                if col.name() == "geometry" || fields.iter().any(|f: &Field| f.name() == col.name())
                {
                    return exec_err!("Duplicate FlatGeobuf column name: {}", col.name());
                }
                let dt = column_type(col.type_())?;
                fields.push(Field::new(col.name(), dt, col.nullable()));
                types.push(col.type_());
            }
        }
        let crs = if let Some(c) = h.crs() {
            if let Some(wkt) = c.wkt().filter(|v| !v.is_empty()) {
                deserialize_crs(wkt)?
            } else if let Some(code) = c.code_string().filter(|v| !v.is_empty()) {
                deserialize_crs(&format!("{}:{code}", c.org().unwrap_or("EPSG")))?
            } else if c.code() != 0 {
                deserialize_crs(&format!("{}:{}", c.org().unwrap_or("EPSG"), c.code()))?
            } else {
                None
            }
        } else {
            None
        };
        fields.push(SedonaType::Wkb(Edges::Planar, crs).to_storage_field("geometry", true)?);
        drop(fgb);
        let index_begin = file.stream_position()?;
        let mut offsets = vec![];
        if indexed {
            if node_size < 2 || count > usize::MAX / 80 {
                return exec_err!("Invalid FlatGeobuf index dimensions");
            }
            // Reject impossible counts before index_size or vector allocation.
            if count as u64 > stamp.size / 40 {
                return exec_err!("FlatGeobuf index count exceeds file size");
            }
            let index_size = PackedRTree::index_size(count, node_size) as u64;
            let begin = index_begin
                .checked_add(index_size)
                .filter(|b| *b <= stamp.size)
                .ok_or_else(|| exec_datafusion_err!("FlatGeobuf index extends beyond file"))?;
            let leaf_begin = begin
                .checked_sub(count as u64 * 40)
                .ok_or_else(|| exec_datafusion_err!("Invalid FlatGeobuf index layout"))?;
            file.seek(SeekFrom::Start(leaf_begin))?;
            for _ in 0..count {
                let node = NodeItem::from_reader(&mut file)
                    .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf leaf: {e}"))?;
                let start = begin
                    .checked_add(node.offset)
                    .filter(|p| *p < stamp.size)
                    .ok_or_else(|| {
                        exec_datafusion_err!("FlatGeobuf feature offset outside file")
                    })?;
                if offsets.last().is_some_and(|p| *p >= start) {
                    return exec_err!("FlatGeobuf feature offsets are not increasing");
                }
                offsets.push(start);
            }
            if offsets.first().copied() != Some(begin) {
                return exec_err!("First FlatGeobuf feature offset is not zero");
            }
            offsets.push(stamp.size);
            for pair in offsets.windows(2) {
                validate_frame(&mut file, pair[0], pair[1])?;
            }
        } else {
            let mut pos = index_begin;
            while pos < stamp.size {
                offsets.push(pos);
                let end = frame_end(&mut file, pos)?;
                if end > stamp.size {
                    return exec_err!("Truncated FlatGeobuf feature payload");
                }
                pos = end;
            }
            if count != 0 && count != offsets.len() {
                return exec_err!("FlatGeobuf feature count does not match file");
            }
            offsets.push(stamp.size);
        }
        if file_stamp(path)? != stamp {
            return exec_err!("FlatGeobuf file changed while reading metadata");
        }
        Ok(Self {
            path: path.into(),
            stamp,
            schema: Arc::new(Schema::new(fields)),
            types,
            geometry_type,
            has_z,
            has_m,
            indexed,
            offsets,
        })
    }
}
fn frame_end(file: &mut File, start: u64) -> Result<u64> {
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = [0; 4];
    file.read_exact(&mut bytes)?;
    let size = u32::from_le_bytes(bytes) as u64;
    if size == 0 {
        return exec_err!("Invalid zero-length FlatGeobuf feature");
    }
    start
        .checked_add(4 + size)
        .ok_or_else(|| exec_datafusion_err!("FlatGeobuf feature size overflow"))
}
fn validate_frame(file: &mut File, start: u64, end: u64) -> Result<()> {
    if frame_end(file, start)? != end {
        return exec_err!("FlatGeobuf index does not match feature framing");
    }
    Ok(())
}
pub(crate) fn column_type(t: ColumnType) -> Result<DataType> {
    Ok(match t {
        ColumnType::Byte => DataType::Int8,
        ColumnType::UByte => DataType::UInt8,
        ColumnType::Bool => DataType::Boolean,
        ColumnType::Short => DataType::Int16,
        ColumnType::UShort => DataType::UInt16,
        ColumnType::Int => DataType::Int32,
        ColumnType::UInt => DataType::UInt32,
        ColumnType::Long => DataType::Int64,
        ColumnType::ULong => DataType::UInt64,
        ColumnType::Float => DataType::Float32,
        ColumnType::Double => DataType::Float64,
        ColumnType::String | ColumnType::Json | ColumnType::DateTime => DataType::Utf8,
        ColumnType::Binary => DataType::Binary,
        _ => return exec_err!("Unsupported FlatGeobuf column type: {t:?}"),
    })
}
