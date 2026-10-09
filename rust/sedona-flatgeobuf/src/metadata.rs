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

use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef, TimeUnit};
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use flatgeobuf::{Column, ColumnType, GeometryType, Header};
use sedona_datasource::spec::Object;
use sedona_schema::{
    crs::deserialize_crs,
    datatypes::{Edges, SedonaType},
};
use std::{
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
pub(crate) async fn file_stamp(path: &Path) -> Result<FileStamp> {
    let m = tokio::fs::metadata(path).await?;
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
    pub data: Metadata,
}
impl std::ops::Deref for FileMetadata {
    type Target = Metadata;
    fn deref(&self) -> &Metadata {
        &self.data
    }
}
#[derive(Debug, Clone)]
pub(crate) struct Metadata {
    pub header: Vec<u8>,
    pub types: Vec<ColumnType>,
    pub geometry_type: GeometryType,
    pub has_z: bool,
    pub has_m: bool,
    pub indexed: bool,
    // Absolute starts, followed by the end of the feature section.
    pub offsets: Vec<u64>,
}
impl FileMetadata {
    pub async fn read(path: &Path, stamp: FileStamp) -> Result<Self> {
        use object_store::{ObjectStoreExt, local::LocalFileSystem};
        let store = Arc::new(LocalFileSystem::new());
        let location = object_store::path::Path::from_filesystem_path(path)?;
        let object = store.head(&location).await?;
        let data = crate::store_metadata::metadata(store.clone(), &object, 65536, None).await?;
        // Preserve the local adapter's eager framing validation using byte slices.
        use futures::{StreamExt, TryStreamExt};
        futures::stream::iter(data.offsets.windows(2))
            .map(|pair: &[u64]| async {
                let prefix =
                    crate::object_io::fetch_range(store.as_ref(), &object, pair[0]..pair[0] + 4)
                        .await?;
                let size = u32::from_le_bytes(prefix.as_ref().try_into().unwrap()) as u64;
                if size == 0 || pair[1] - pair[0] != 4 + size {
                    return exec_err!("FlatGeobuf index does not match feature framing");
                }
                Ok(())
            })
            .boxed()
            .buffered(16)
            .try_collect::<Vec<_>>()
            .await?;
        if file_stamp(path).await? != stamp {
            return exec_err!("FlatGeobuf file changed while reading metadata");
        }
        Ok(Self {
            path: path.into(),
            stamp,
            data: data.as_ref().clone(),
        })
    }
}
impl Metadata {
    /// Derive Arrow fields from the retained FlatGeobuf header when required.
    /// Geometry naming is a scan option, not a property of cached file metadata.
    pub(crate) fn infer_schema(&self, geometry_column_name: &str) -> Result<SchemaRef> {
        let h = flatgeobuf::size_prefixed_root_as_header(&self.header)
            .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf header: {e}"))?;
        let mut fields = h
            .columns()
            .unwrap()
            .iter()
            .map(column_field)
            .collect::<Result<Vec<_>>>()?;
        fields.push(Arc::new(
            geometry_type(&h)?.to_storage_field(geometry_column_name, true)?,
        ));
        Ok(Arc::new(Schema::new(fields)))
    }

    pub(crate) fn from_header(h: &Header<'_>, header: Vec<u8>) -> Result<(Self, usize, u16)> {
        geometry_type(h)?;
        let geometry_type = h.geometry_type();
        let has_z = h.has_z();
        let has_m = h.has_m();
        let count = usize::try_from(h.features_count())
            .map_err(|_| exec_datafusion_err!("FlatGeobuf count overflow"))?;
        let node_size = h.index_node_size();
        let indexed = node_size > 0 && count > 0;
        let columns = h.columns().ok_or_else(||
            exec_datafusion_err!("FlatGeobuf missing header columns / feature-local schema inference is not supported"))?;
        let types = columns.iter().map(|col| col.type_()).collect();
        Ok((
            Self {
                header,
                types,
                geometry_type,
                has_z,
                has_m,
                indexed,
                offsets: vec![],
            },
            count,
            node_size,
        ))
    }
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
        ColumnType::String | ColumnType::Json => DataType::Utf8,
        ColumnType::DateTime => DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        ColumnType::Binary => DataType::Binary,
        _ => return exec_err!("Unsupported FlatGeobuf column type: {t:?}"),
    })
}

/// Convert a header column independently of DataFusion's column-name validation.
fn column_field(column: Column<'_>) -> Result<FieldRef> {
    let mut field = Field::new(
        column.name(),
        column_type(column.type_())?,
        column.nullable(),
    );
    if column.type_() == ColumnType::Json {
        field = field.with_metadata(
            [
                ("ARROW:extension:name".into(), "arrow.json".into()),
                ("ARROW:extension:metadata".into(), "".into()),
            ]
            .into(),
        );
    }
    Ok(Arc::new(field))
}

/// Validate supported dimensions and preserve the header's planar CRS.
fn geometry_type(header: &Header<'_>) -> Result<SedonaType> {
    if header.has_t() || header.has_tm() {
        return exec_err!("FlatGeobuf T/TM dimensions are not supported");
    }
    if header.geometry_type().0 > GeometryType::GeometryCollection.0 {
        return exec_err!("FlatGeobuf curved geometry types are not supported");
    }
    let crs = if let Some(c) = header.crs() {
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
    Ok(SedonaType::Wkb(Edges::Planar, crs))
}
