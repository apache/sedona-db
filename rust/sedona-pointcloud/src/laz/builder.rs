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

use std::{fmt::Debug, sync::Arc};

use arrow_array::{
    builder::{
        ArrayBuilder, BinaryBuilder, BooleanBuilder, FixedSizeBinaryBuilder, Float32Builder,
        Float64Builder, Int16Builder, Int32Builder, Int64Builder, Int8Builder, UInt16Builder,
        UInt32Builder, UInt64Builder, UInt8Builder,
    },
    Array, ArrayRef, BooleanArray, FixedSizeBinaryArray, Float32Array, Float64Array, StructArray,
    UInt16Array, UInt8Array,
};
use arrow_buffer::ScalarBuffer;
use arrow_schema::{ArrowError, DataType};
use geoarrow_array::{
    array::{CoordBuffer, PointArray, SeparatedCoordBuffer},
    GeoArrowArray,
};
use geoarrow_schema::Dimension;
use las::{Header, Point};

use crate::laz::{
    metadata::ExtraAttribute,
    options::{LasExtraBytes, LasPointEncoding},
    schema::try_schema_from_header,
};

#[derive(Debug)]
pub struct RowBuilder {
    x: Float64Builder,
    y: Float64Builder,
    z: Float64Builder,
    intensity: UInt16Builder,
    return_number: UInt8Builder,
    number_of_returns: UInt8Builder,
    is_synthetic: BooleanBuilder,
    is_key_point: BooleanBuilder,
    is_withheld: BooleanBuilder,
    is_overlap: BooleanBuilder,
    scanner_channel: UInt8Builder,
    scan_direction: UInt8Builder,
    is_edge_of_flight_line: BooleanBuilder,
    classification: UInt8Builder,
    user_data: UInt8Builder,
    scan_angle: Float32Builder,
    point_source_id: UInt16Builder,
    gps_time: Float64Builder,
    red: UInt16Builder,
    green: UInt16Builder,
    blue: UInt16Builder,
    nir: UInt16Builder,
    extra: FixedSizeBinaryBuilder,
    header: Arc<Header>,
    point_encoding: LasPointEncoding,
    extra_bytes: LasExtraBytes,
    extra_attributes: Arc<Vec<ExtraAttribute>>,
}

impl RowBuilder {
    pub fn new(capacity: usize, header: Arc<Header>) -> Self {
        Self {
            x: Float64Array::builder(capacity),
            y: Float64Array::builder(capacity),
            z: Float64Array::builder(capacity),
            intensity: UInt16Array::builder(capacity),
            return_number: UInt8Array::builder(capacity),
            number_of_returns: UInt8Array::builder(capacity),
            is_synthetic: BooleanArray::builder(capacity),
            is_key_point: BooleanArray::builder(capacity),
            is_withheld: BooleanArray::builder(capacity),
            is_overlap: BooleanArray::builder(capacity),
            scanner_channel: UInt8Array::builder(capacity),
            scan_direction: UInt8Array::builder(capacity),
            is_edge_of_flight_line: BooleanArray::builder(capacity),
            classification: UInt8Array::builder(capacity),
            user_data: UInt8Array::builder(capacity),
            scan_angle: Float32Array::builder(capacity),
            point_source_id: UInt16Array::builder(capacity),
            gps_time: Float64Array::builder(capacity),
            red: UInt16Array::builder(capacity),
            green: UInt16Array::builder(capacity),
            blue: UInt16Array::builder(capacity),
            nir: UInt16Array::builder(capacity),
            extra: FixedSizeBinaryBuilder::with_capacity(
                capacity,
                header.point_format().extra_bytes as i32,
            ),

            header,
            point_encoding: Default::default(),
            extra_bytes: Default::default(),
            extra_attributes: Arc::new(Vec::new()),
        }
    }

    pub fn with_point_encoding(mut self, point_encoding: LasPointEncoding) -> Self {
        self.point_encoding = point_encoding;
        self
    }

    pub fn with_extra_attributes(
        mut self,
        attributes: Arc<Vec<ExtraAttribute>>,
        extra_bytes: LasExtraBytes,
    ) -> Self {
        self.extra_attributes = attributes;
        self.extra_bytes = extra_bytes;
        self
    }

    pub fn append(&mut self, p: Point) {
        self.x.append_value(p.x);
        self.y.append_value(p.y);
        self.z.append_value(p.z);
        self.intensity.append_option(Some(p.intensity));
        self.return_number.append_value(p.return_number);
        self.number_of_returns.append_value(p.number_of_returns);
        self.is_synthetic.append_value(p.is_synthetic);
        self.is_key_point.append_value(p.is_key_point);
        self.is_withheld.append_value(p.is_withheld);
        self.is_overlap.append_value(p.is_overlap);
        self.scanner_channel.append_value(p.scanner_channel);
        self.scan_direction.append_value(p.scan_direction as u8);
        self.is_edge_of_flight_line
            .append_value(p.is_edge_of_flight_line);
        self.classification.append_value(u8::from(p.classification));
        self.user_data.append_value(p.user_data);
        self.scan_angle.append_value(p.scan_angle);
        self.point_source_id.append_value(p.point_source_id);
        if self.header.point_format().has_gps_time {
            self.gps_time.append_value(p.gps_time.unwrap());
        }
        if self.header.point_format().has_color {
            let color = p.color.unwrap();
            self.red.append_value(color.red);
            self.green.append_value(color.green);
            self.blue.append_value(color.blue);
        }
        if self.header.point_format().has_nir {
            self.nir.append_value(p.nir.unwrap());
        }
        if self.header.point_format().extra_bytes > 0 {
            self.extra.append_value(p.extra_bytes).unwrap();
        }
    }

    /// Note: returns StructArray to allow nesting within another array if desired
    pub fn finish(&mut self) -> Result<StructArray, ArrowError> {
        let schema = try_schema_from_header(&self.header, self.point_encoding, self.extra_bytes)?;

        let mut columns = match self.point_encoding {
            LasPointEncoding::Plain => vec![
                Arc::new(self.x.finish()) as ArrayRef,
                Arc::new(self.y.finish()) as ArrayRef,
                Arc::new(self.z.finish()) as ArrayRef,
            ],
            LasPointEncoding::Wkb => {
                const POINT_SIZE: usize = 29;

                let n: usize = self.x.len();

                let mut builder = BinaryBuilder::with_capacity(n, n * POINT_SIZE);

                let x = self.x.finish();
                let y = self.y.finish();
                let z = self.z.finish();

                let mut wkb_bytes = [0_u8; POINT_SIZE];
                wkb_bytes[0] = 0x01; // Little-endian
                wkb_bytes[1..5].copy_from_slice(&[0xE9, 0x03, 0x00, 0x00]); // Point Z type (1001)

                for i in 0..n {
                    let x = unsafe { x.value_unchecked(i) };
                    let y = unsafe { y.value_unchecked(i) };
                    let z = unsafe { z.value_unchecked(i) };

                    wkb_bytes[5..13].copy_from_slice(x.to_le_bytes().as_slice());
                    wkb_bytes[13..21].copy_from_slice(y.to_le_bytes().as_slice());
                    wkb_bytes[21..29].copy_from_slice(z.to_le_bytes().as_slice());

                    builder.append_value(wkb_bytes);
                }

                vec![Arc::new(builder.finish()) as ArrayRef]
            }
            LasPointEncoding::Native => {
                let buffers = [
                    self.x.finish().into_parts().1,
                    self.y.finish().into_parts().1,
                    self.z.finish().into_parts().1,
                    ScalarBuffer::from(vec![]),
                ];
                let coords = CoordBuffer::Separated(SeparatedCoordBuffer::from_array(
                    buffers,
                    Dimension::XYZ,
                )?);
                let points = PointArray::new(coords, None, Default::default());
                vec![points.to_array_ref()]
            }
        };

        columns.extend([
            Arc::new(self.intensity.finish()) as ArrayRef,
            Arc::new(self.return_number.finish()) as ArrayRef,
            Arc::new(self.number_of_returns.finish()) as ArrayRef,
            Arc::new(self.is_synthetic.finish()) as ArrayRef,
            Arc::new(self.is_key_point.finish()) as ArrayRef,
            Arc::new(self.is_withheld.finish()) as ArrayRef,
            Arc::new(self.is_overlap.finish()) as ArrayRef,
            Arc::new(self.scanner_channel.finish()) as ArrayRef,
            Arc::new(self.scan_direction.finish()) as ArrayRef,
            Arc::new(self.is_edge_of_flight_line.finish()) as ArrayRef,
            Arc::new(self.classification.finish()) as ArrayRef,
            Arc::new(self.user_data.finish()) as ArrayRef,
            Arc::new(self.scan_angle.finish()) as ArrayRef,
            Arc::new(self.point_source_id.finish()) as ArrayRef,
        ]);
        if self.header.point_format().has_gps_time {
            columns.push(Arc::new(self.gps_time.finish()) as ArrayRef);
        }
        if self.header.point_format().has_color {
            columns.extend([
                Arc::new(self.red.finish()) as ArrayRef,
                Arc::new(self.green.finish()) as ArrayRef,
                Arc::new(self.blue.finish()) as ArrayRef,
            ]);
        }
        if self.header.point_format().has_nir {
            columns.push(Arc::new(self.nir.finish()) as ArrayRef);
        }

        // extra bytes
        let num_extra_bytes = self.header.point_format().extra_bytes as usize;
        if num_extra_bytes > 0 {
            match self.extra_bytes {
                LasExtraBytes::Typed => {
                    let extra = self.extra.finish();

                    let mut pos = 0;

                    for attribute in self.extra_attributes.iter() {
                        pos += build_attribute(attribute, pos, &extra, &mut columns)?;
                    }
                }
                LasExtraBytes::Blob => columns.push(Arc::new(self.extra.finish())),
                LasExtraBytes::Ignore => (),
            }
        }

        Ok(StructArray::new(schema.fields.to_owned(), columns, None))
    }
}

fn build_attribute(
    attribute: &ExtraAttribute,
    pos: usize,
    extra: &FixedSizeBinaryArray,
    columns: &mut Vec<ArrayRef>,
) -> Result<usize, ArrowError> {
    let scale = attribute.scale.unwrap_or(1.0);
    let offset = attribute.offset.unwrap_or(0.0);

    let width = if let DataType::FixedSizeBinary(width) = attribute.data_type {
        width as usize
    } else {
        attribute.data_type.primitive_width().unwrap()
    };

    let iter = extra.iter().map(|b| &b.unwrap()[pos..pos + width]);

    match &attribute.data_type {
        DataType::FixedSizeBinary(_) => {
            let data = FixedSizeBinaryArray::try_from_iter(iter)?;
            columns.push(Arc::new(data) as ArrayRef)
        }
        DataType::Int8 => {
            let mut builder = Int8Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(i64::from_le_bytes);

            for d in iter {
                let mut v = i8::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as i64 {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = (v as f64 * scale + offset) as i8;
                }
                builder.append_value(v)
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Int16 => {
            let mut builder = Int16Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(i64::from_le_bytes);

            for d in iter {
                let mut v = i16::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as i64 {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = (v as f64 * scale + offset) as i16;
                }
                builder.append_value(v)
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Int32 => {
            let mut builder = Int32Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(i64::from_le_bytes);

            for d in iter {
                let mut v = i32::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as i64 {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = (v as f64 * scale + offset) as i32;
                }
                builder.append_value(v)
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(i64::from_le_bytes);

            for d in iter {
                let mut v = i64::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = (v as f64 * scale + offset) as i64;
                }
                builder.append_value(v)
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::UInt8 => {
            let mut builder = UInt8Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(u64::from_le_bytes);

            for d in iter {
                let mut v = u8::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as u64 {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = (v as f64 * scale + offset) as u8;
                }
                builder.append_value(v)
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::UInt16 => {
            let mut builder = UInt16Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(u64::from_le_bytes);

            for d in iter {
                let mut v = u16::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as u64 {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = (v as f64 * scale + offset) as u16;
                }
                builder.append_value(v)
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::UInt32 => {
            let mut builder = UInt32Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(u64::from_le_bytes);

            for d in iter {
                let mut v = u32::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as u64 {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = (v as f64 * scale + offset) as u32;
                }
                builder.append_value(v)
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::UInt64 => {
            let mut builder = UInt64Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(u64::from_le_bytes);

            for d in iter {
                let mut v = u64::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = (v as f64 * scale + offset) as u64;
                }
                builder.append_value(v)
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Float32 => {
            let mut builder = Float32Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(f64::from_le_bytes);

            for d in iter {
                let mut v = f32::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as f64 {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = (v as f64 * scale + offset) as f32;
                }
                builder.append_value(v)
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(extra.len());
            let no_data = attribute.no_data.map(f64::from_le_bytes);

            for d in iter {
                let mut v = f64::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v {
                        builder.append_null();
                        continue;
                    }
                }
                if attribute.scale.is_some() || attribute.offset.is_some() {
                    v = v * scale + offset;
                }
                builder.append_value(v);
            }

            columns.push(Arc::new(builder.finish()) as ArrayRef)
        }

        dt => {
            return Err(ArrowError::ExternalError(
                format!("Unsupported data type for extra bytes: `{dt}`").into(),
            ))
        }
    }

    Ok(width)
}

#[cfg(test)]
mod tests {
    use std::{fs::File, sync::Arc};

    use arrow_array::{
        cast::AsArray,
        types::{
            Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
            UInt32Type, UInt64Type, UInt8Type,
        },
    };
    use datafusion_datasource::PartitionedFile;
    use las::{point::Format, Builder, Writer};
    use object_store::{local::LocalFileSystem, path::Path, ObjectStore};

    use crate::laz::{
        options::{LasExtraBytes, LazTableOptions},
        reader::LazFileReaderFactory,
    };

    #[tokio::test]
    async fn point_formats() {
        let tmpdir = tempfile::tempdir().unwrap();

        for format in 0..=10 {
            let tmp_path = tmpdir.path().join("format.laz");
            let tmp_file = File::create(&tmp_path).unwrap();

            // create laz file
            let mut builder = Builder::from((1, 4));
            builder.point_format = Format::new(format).unwrap();
            builder.point_format.is_compressed = true;
            let header = builder.into_header().unwrap();
            let mut writer = Writer::new(tmp_file, header).unwrap();
            writer.close().unwrap();

            // read batch with `LazFileReader`
            let store = LocalFileSystem::new();
            let location = Path::from_filesystem_path(tmp_path).unwrap();
            let object = store.head(&location).await.unwrap();

            let laz_file_reader = LazFileReaderFactory::new(Arc::new(store), None)
                .create_reader(
                    PartitionedFile::new(location, object.size),
                    LazTableOptions::default(),
                )
                .unwrap();
            let metadata = laz_file_reader.get_metadata().await.unwrap();

            let batch = laz_file_reader
                .get_batch(&metadata.chunk_table[0])
                .await
                .unwrap();

            match format {
                0 => assert_eq!(batch.num_columns(), 17),
                1 | 4 | 6 | 9 => assert_eq!(batch.num_columns(), 18),
                2 => assert_eq!(batch.num_columns(), 20),
                3 | 5 | 7 => assert_eq!(batch.num_columns(), 21),
                8 | 10 => assert_eq!(batch.num_columns(), 22),
                _ => unreachable!(),
            }
        }
    }

    #[tokio::test]
    async fn extra_attributes() {
        // file with extra attributes generated with `tests/data/generate.py`
        let extra_path = "tests/data/extra.laz";

        // read batch with `LazFileReader`
        let store = LocalFileSystem::new();
        let location = Path::from_filesystem_path(extra_path).unwrap();
        let object = store.head(&location).await.unwrap();

        let laz_file_reader = LazFileReaderFactory::new(Arc::new(store), None)
            .create_reader(
                PartitionedFile::new(location, object.size),
                LazTableOptions::default().with_extra_bytes(LasExtraBytes::Typed),
            )
            .unwrap();
        let metadata = laz_file_reader.get_metadata().await.unwrap();

        let batch = laz_file_reader
            .get_batch(&metadata.chunk_table[0])
            .await
            .unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 48);

        let schema = batch.schema();
        for field in schema.fields() {
            let name = field.name();

            match name.split_once('_') {
                Some((dt, kind)) => {
                    if !["plain", "scaled", "nodata"].contains(&kind) {
                        continue;
                    }

                    let array = batch.column_by_name(name).unwrap();

                    if kind == "nodata" {
                        assert_eq!(array.null_count(), 1);
                        continue;
                    }

                    assert_eq!(array.null_count(), 0);

                    match dt {
                        "uint8" => {
                            let array = array.as_primitive::<UInt8Type>();
                            assert_eq!(array.value(0), 21u8);
                        }
                        "int8" => {
                            let array = array.as_primitive::<Int8Type>();
                            assert_eq!(array.value(0), 21i8);
                        }
                        "uint16" => {
                            let array = array.as_primitive::<UInt16Type>();
                            assert_eq!(array.value(0), 21u16);
                        }
                        "int16" => {
                            let array = array.as_primitive::<Int16Type>();
                            assert_eq!(array.value(0), 21i16);
                        }
                        "uint32" => {
                            let array = array.as_primitive::<UInt32Type>();
                            assert_eq!(array.value(0), 21u32);
                        }
                        "int32" => {
                            let array = array.as_primitive::<Int32Type>();
                            assert_eq!(array.value(0), 21i32);
                        }
                        "uint64" => {
                            let array = array.as_primitive::<UInt64Type>();
                            assert_eq!(array.value(0), 21u64);
                        }
                        "int64" => {
                            let array = array.as_primitive::<Int64Type>();
                            assert_eq!(array.value(0), 21i64);
                        }
                        "float32" => {
                            let array = array.as_primitive::<Float32Type>();
                            assert_eq!(array.value(0), 21f32);
                        }
                        "float64" => {
                            let array = array.as_primitive::<Float64Type>();
                            assert_eq!(array.value(0), 21f64);
                        }
                        _ => unreachable!("unexpected data type"),
                    }
                }
                None => continue,
            }
        }
    }
}
