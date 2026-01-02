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

use std::{any::Any, collections::HashMap, error::Error, io::Cursor, ops::Range, sync::Arc};

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

use crate::laz::{
    options::{LasExtraBytes, LasPointEncoding},
    schema::schema_from_header,
};

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
        // TODO: proper size
        std::mem::size_of_val(self)
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
    point_encoding: LasPointEncoding,
    extra_bytes: LasExtraBytes,
}

impl<'a> LazMetadataReader<'a> {
    pub fn new(store: &'a dyn ObjectStore, object_meta: &'a ObjectMeta) -> Self {
        Self {
            store,
            object_meta,
            file_metadata_cache: None,
            point_encoding: Default::default(),
            extra_bytes: Default::default(),
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

    /// set point encoding
    pub fn with_point_encoding(mut self, point_encoding: LasPointEncoding) -> Self {
        self.point_encoding = point_encoding;
        self
    }

    /// set extra bytes parsing strategy
    pub fn with_extra_bytes(mut self, extra_bytes: LasExtraBytes) -> Self {
        self.extra_bytes = extra_bytes;
        self
    }

    /// Fetch header
    pub async fn fetch_header(&self) -> Result<Header, DataFusionError> {
        fetch_header(self.store, self.object_meta, false)
            .await
            .map_err(DataFusionError::External)
    }

    /// Fetch laz metadata from the remote object store
    pub async fn fetch_metadata(&self) -> Result<Arc<LazMetadata>, DataFusionError> {
        let Self {
            store,
            object_meta,
            file_metadata_cache,
            point_encoding: _,
            extra_bytes: _,
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
        let extra_attributes = extra_bytes_attributes(&header);
        let chunk_table = chunk_table(*store, object_meta, &header).await.unwrap();

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

        let schema = schema_from_header(&metadata.header, self.point_encoding, self.extra_bytes);

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
    include_evlr: bool,
) -> Result<Header, Box<dyn Error + Send + Sync>> {
    let location = &object_meta.location;

    // Header
    let bytes = store.get_range(location, 0..375).await?;
    let reader = Cursor::new(bytes);
    let raw_header = RawHeader::read_from(reader)?;

    let header_size = raw_header.header_size as u64;
    let num_vlr = raw_header.number_of_variable_length_records;
    let offset_to_point_data = raw_header.offset_to_point_data as u64;
    let evlr = raw_header.evlr;

    // VLRs
    let bytes = store
        .get_range(location, header_size..offset_to_point_data)
        .await?;
    let mut reader = Cursor::new(bytes);

    let mut builder = Builder::new(raw_header)?;
    let mut position = 0;
    for _ in 0..num_vlr {
        let vlr = RawVlr::read_from(&mut reader, false).map(Vlr::new)?;
        position += vlr.len(false) as u64;
        builder.vlrs.push(vlr);
    }

    builder.padding = vec![0; (offset_to_point_data - (header_size + position)) as usize];

    // EVLRs
    if include_evlr {
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

pub(crate) fn extra_bytes_attributes(header: &Header) -> Vec<ExtraAttribute> {
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
                11..=30 => panic!("deprecated extra bytes data type"),
                31..=255 => panic!("reserved extra butes data type"),
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

    attributes
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

    let laz_vlr = header.laz_vlr().expect("LAZ file must contain laz vlr");

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
        unimplemented!("LAZ files without chunk table not supported (yet)")
    };

    let bytes = store
        .get_range(&object_meta.location, table_offset..table_offset + 8)
        .await?;

    let num_chunks = u32::from_le_bytes(bytes[4..].to_vec().try_into().unwrap()) as u64;

    let bytes = store
        .get_range(
            &object_meta.location,
            table_offset..table_offset + 8 + 8 * num_chunks,
        )
        .await?;

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
