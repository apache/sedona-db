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
    any::Any,
    collections::HashMap,
    error::Error,
    io::{Cursor, Read},
    ops::Range,
    sync::Arc,
};

use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion_common::{
    error::DataFusionError, scalar::ScalarValue, stats::Precision, ColumnStatistics, Statistics,
};
use datafusion_execution::cache::cache_manager::{FileMetadata, FileMetadataCache};
use las::{
    raw::{Header as RawHeader, Vlr as RawVlr},
    Builder, Header, Vlr,
};
use laz::laszip::ChunkTable;
use object_store::{ObjectMeta, ObjectStore};

use crate::{laz::schema::try_schema_from_header, options::PointcloudOptions};

/// Laz chunk metadata
#[derive(Debug, Clone)]
pub struct ChunkMeta {
    pub num_points: u64,
    pub point_offset: u64,
    pub byte_range: Range<u64>,
}

/// Laz metadata
#[derive(Debug, Clone)]
pub struct LazMetadata {
    pub header: Arc<Header>,
    pub chunk_table: Vec<ChunkMeta>,
    pub extra_attributes: Arc<Vec<ExtraAttribute>>,
}

impl FileMetadata for LazMetadata {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn memory_size(&self) -> usize {
        std::mem::size_of::<Header>()
            + self
                .header
                .all_vlrs()
                .map(|vlr| vlr.data.len())
                .sum::<usize>()
            + self.chunk_table.capacity() * std::mem::size_of::<ChunkMeta>()
            + self.extra_attributes.capacity() * std::mem::size_of::<ExtraAttribute>()
    }

    fn extra_info(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Reader for laz file metadata in object storage.
pub struct LazMetadataReader<'a> {
    store: &'a dyn ObjectStore,
    object_meta: &'a ObjectMeta,
    file_metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    options: PointcloudOptions,
}

impl<'a> LazMetadataReader<'a> {
    pub fn new(store: &'a dyn ObjectStore, object_meta: &'a ObjectMeta) -> Self {
        Self {
            store,
            object_meta,
            file_metadata_cache: None,
            options: Default::default(),
        }
    }

    /// set file metadata cache
    pub fn with_file_metadata_cache(
        mut self,
        file_metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    ) -> Self {
        self.file_metadata_cache = file_metadata_cache;
        self
    }

    /// set table options
    pub fn with_options(mut self, options: PointcloudOptions) -> Self {
        self.options = options;
        self
    }

    /// Fetch header
    pub async fn fetch_header(&self) -> Result<Header, DataFusionError> {
        fetch_header(self.store, self.object_meta)
            .await
            .map_err(DataFusionError::External)
    }

    /// Fetch laz metadata from the remote object store
    pub async fn fetch_metadata(&self) -> Result<Arc<LazMetadata>, DataFusionError> {
        let Self {
            store,
            object_meta,
            file_metadata_cache,
            options: _,
        } = self;

        if let Some(las_file_metadata) = file_metadata_cache
            .as_ref()
            .and_then(|file_metadata_cache| file_metadata_cache.get(object_meta))
            .and_then(|file_metadata| {
                file_metadata
                    .as_any()
                    .downcast_ref::<LazMetadata>()
                    .map(|laz_file_metadata| Arc::new(laz_file_metadata.to_owned()))
            })
        {
            return Ok(las_file_metadata);
        }

        let header = self.fetch_header().await?;
        let extra_attributes = extra_bytes_attributes(&header)?;
        let chunk_table = chunk_table(*store, object_meta, &header).await?;

        let metadata = Arc::new(LazMetadata {
            header: Arc::new(header),
            chunk_table,
            extra_attributes: Arc::new(extra_attributes),
        });

        if let Some(file_metadata_cache) = file_metadata_cache {
            file_metadata_cache.put(object_meta, metadata.clone());
        }

        Ok(metadata)
    }

    /// Read and parse the schema of the laz file
    pub async fn fetch_schema(&mut self) -> Result<Schema, DataFusionError> {
        let metadata = self.fetch_metadata().await?;

        let schema = try_schema_from_header(
            &metadata.header,
            self.options.geometry_encoding,
            self.options.las.extra_bytes,
        )?;

        Ok(schema)
    }

    /// Fetch the metadata from the laz file via [`Self::fetch_metadata`] and extracts
    /// the statistics in the metadata
    pub async fn fetch_statistics(
        &self,
        table_schema: &SchemaRef,
    ) -> Result<Statistics, DataFusionError> {
        let metadata = self.fetch_metadata().await?;

        let mut statistics = Statistics::new_unknown(table_schema)
            .with_num_rows(Precision::Exact(metadata.header.number_of_points() as usize))
            .with_total_byte_size(Precision::Exact(
                metadata
                    .chunk_table
                    .iter()
                    .map(|meta| meta.byte_range.end - meta.byte_range.start)
                    .sum::<u64>() as usize,
            ));

        let bounds = metadata.header.bounds();
        for field in table_schema.fields() {
            let cs = match field.name().as_str() {
                "x" => ColumnStatistics::new_unknown()
                    .with_min_value(Precision::Exact(ScalarValue::Float64(Some(bounds.min.x))))
                    .with_max_value(Precision::Exact(ScalarValue::Float64(Some(bounds.max.x))))
                    .with_null_count(Precision::Exact(0)),
                "y" => ColumnStatistics::new_unknown()
                    .with_min_value(Precision::Exact(ScalarValue::Float64(Some(bounds.min.y))))
                    .with_max_value(Precision::Exact(ScalarValue::Float64(Some(bounds.max.y))))
                    .with_null_count(Precision::Exact(0)),
                "z" => ColumnStatistics::new_unknown()
                    .with_min_value(Precision::Exact(ScalarValue::Float64(Some(bounds.min.z))))
                    .with_max_value(Precision::Exact(ScalarValue::Float64(Some(bounds.max.z))))
                    .with_null_count(Precision::Exact(0)),
                _ => ColumnStatistics::new_unknown(),
            };

            statistics = statistics.add_column_statistics(cs);
        }

        Ok(statistics)
    }
}

pub(crate) async fn fetch_header(
    store: &(impl ObjectStore + ?Sized),
    object_meta: &ObjectMeta,
) -> Result<Header, Box<dyn Error + Send + Sync>> {
    let location = &object_meta.location;

    // Header
    let bytes = store.get_range(location, 0..375).await?;
    let reader = Cursor::new(bytes);
    let raw_header = RawHeader::read_from(reader)?;

    let header_size = raw_header.header_size as u64;
    let offset_to_point_data = raw_header.offset_to_point_data as u64;
    let num_vlr = raw_header.number_of_variable_length_records;
    let evlr = raw_header.evlr;

    let mut builder = Builder::new(raw_header)?;

    // VLRs
    let bytes = store
        .get_range(location, header_size..offset_to_point_data)
        .await?;
    let mut reader = Cursor::new(bytes);

    for _ in 0..num_vlr {
        let vlr = RawVlr::read_from(&mut reader, false).map(Vlr::new)?;
        builder.vlrs.push(vlr);
    }

    reader.read_to_end(&mut builder.vlr_padding)?;

    // EVLRs
    if let Some(evlr) = evlr {
        let mut start = evlr.start_of_first_evlr;

        for _ in 0..evlr.number_of_evlrs {
            let mut end = start + 60;

            let bytes = store.get_range(location, start..end).await?;

            end += u64::from_le_bytes(bytes[20..28].try_into()?);

            let bytes = store.get_range(location, start..end).await?;
            let mut reader = Cursor::new(bytes);
            let evlr = RawVlr::read_from(&mut reader, true).map(Vlr::new)?;

            builder.evlrs.push(evlr);

            start = end;
        }
    }

    Ok(builder.into_header()?)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtraAttribute {
    pub data_type: DataType,
    pub no_data: Option<[u8; 8]>,
    pub scale: Option<f64>,
    pub offset: Option<f64>,
}

pub(crate) fn extra_bytes_attributes(
    header: &Header,
) -> Result<Vec<ExtraAttribute>, Box<dyn Error + Send + Sync>> {
    let mut attributes = Vec::new();

    for vlr in header.all_vlrs() {
        if !(vlr.user_id == "LASF_Spec" && vlr.record_id == 4) {
            continue;
        }

        for bytes in vlr.data.chunks(192) {
            // data type
            let data_type = match bytes[2] {
                0 => DataType::FixedSizeBinary(bytes[3] as i32),
                1 => DataType::UInt8,
                2 => DataType::Int8,
                3 => DataType::UInt16,
                4 => DataType::Int16,
                5 => DataType::UInt32,
                6 => DataType::Int32,
                7 => DataType::UInt64,
                8 => DataType::Int64,
                9 => DataType::Float32,
                10 => DataType::Float64,
                11..=30 => return Err("deprecated extra bytes data type".into()),
                31..=255 => return Err("reserved extra butes data type".into()),
            };

            // no data
            let no_data = if bytes[2] != 0 && bytes[3] & 1 == 1 {
                Some(bytes[40..48].try_into().unwrap())
            } else {
                None
            };

            // scale
            let scale = if bytes[2] != 0 && bytes[3] >> 3 & 1 == 1 {
                Some(f64::from_le_bytes(bytes[112..120].try_into().unwrap()))
            } else {
                None
            };

            // offset
            let offset = if bytes[2] != 0 && bytes[3] >> 4 & 1 == 1 {
                Some(f64::from_le_bytes(bytes[136..144].try_into().unwrap()))
            } else {
                None
            };

            let attribute = ExtraAttribute {
                data_type,
                no_data,
                scale,
                offset,
            };

            attributes.push(attribute);
        }
    }

    Ok(attributes)
}

pub(crate) async fn chunk_table(
    store: &(impl ObjectStore + ?Sized),
    object_meta: &ObjectMeta,
    header: &Header,
) -> Result<Vec<ChunkMeta>, Box<dyn Error + Send + Sync>> {
    let num_points = header.number_of_points();
    let mut point_offset = 0;

    let vlr_len = header.vlrs().iter().map(|v| v.len(false)).sum::<usize>();
    let header_size = header.version().header_size() as usize + header.padding().len();
    let mut byte_offset = (header_size + vlr_len + header.vlr_padding().len()) as u64;

    let laz_vlr = header.laz_vlr()?;

    let ranges = [
        byte_offset..byte_offset + 8,
        object_meta.size - 8..object_meta.size,
    ];
    let bytes = store.get_ranges(&object_meta.location, &ranges).await?;
    let mut table_offset = None;

    let table_offset1 = i64::from_le_bytes(bytes[0].to_vec().try_into().unwrap()) as u64;
    let table_offset2 = i64::from_le_bytes(bytes[1].to_vec().try_into().unwrap()) as u64;

    if table_offset1 > byte_offset {
        table_offset = Some(table_offset1);
    } else if table_offset2 > byte_offset {
        table_offset = Some(table_offset2);
    }

    let Some(table_offset) = table_offset else {
        return Err("LAZ files without chunk table not supported (yet)".into());
    };

    if table_offset > object_meta.size {
        return Err("LAZ file chunk table position is missing/bad".into());
    }

    let bytes = store
        .get_range(&object_meta.location, table_offset..table_offset + 8)
        .await?;

    let num_chunks = u32::from_le_bytes(bytes[4..].to_vec().try_into().unwrap()) as u64;
    let range = table_offset..table_offset + 8 + 8 * num_chunks;
    let bytes = store.get_range(&object_meta.location, range).await?;

    let mut reader = Cursor::new(bytes);
    let variable_size = laz_vlr.uses_variable_size_chunks();
    let chunk_table = ChunkTable::read(&mut reader, variable_size)?;
    assert_eq!(chunk_table.len(), num_chunks as usize);

    let mut chunks = Vec::with_capacity(num_chunks as usize);
    let chunk_size = laz_vlr.chunk_size() as u64;
    byte_offset += 8;

    for chunk_table_entry in &chunk_table {
        let point_count = if variable_size {
            chunk_table_entry.point_count
        } else {
            chunk_size.min(num_points - point_offset)
        };

        let chunk = ChunkMeta {
            num_points: point_count,
            point_offset,
            byte_range: byte_offset..byte_offset + chunk_table_entry.byte_count,
        };
        chunks.push(chunk);
        point_offset += point_count;
        byte_offset += chunk_table_entry.byte_count;
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use las::{point::Format, Builder, Reader, Writer};
    use object_store::{local::LocalFileSystem, path::Path, ObjectStore};

    use crate::laz::metadata::LazMetadataReader;

    #[tokio::test]
    async fn header_basic_e2e() {
        let tmpdir = tempfile::tempdir().unwrap();

        let tmp_path = tmpdir.path().join("tmp.laz");
        let tmp_file = File::create(&tmp_path).unwrap();

        // create laz file
        let mut builder = Builder::from((1, 4));
        builder.point_format = Format::new(1).unwrap();
        builder.point_format.is_compressed = true;
        let header = builder.into_header().unwrap();
        let mut writer = Writer::new(tmp_file, header).unwrap();
        writer.close().unwrap();

        // read with `LazMetadataReader`
        let store = LocalFileSystem::new();
        let location = Path::from_filesystem_path(&tmp_path).unwrap();
        let object_meta = store.head(&location).await.unwrap();
        let metadata_reader = LazMetadataReader::new(&store, &object_meta);

        // read with las `Reader`
        let reader = Reader::from_path(&tmp_path).unwrap();

        assert_eq!(
            reader.header(),
            &metadata_reader.fetch_header().await.unwrap()
        );
    }
}
