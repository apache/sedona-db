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

use crate::wkb::write_wkb_byte_order_marker;
use arrow_array::builder::BinaryBuilder;
use byteorder::{ByteOrder, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};
use wkt::types::Dimension;

pub(crate) fn get_polygon_marker(dimension: Dimension) -> u32 {
    match dimension {
        Dimension::XY => 3u32,
        Dimension::XYZ => 1003u32,
        Dimension::XYM => 2003u32,
        Dimension::XYZM => 3003u32,
    }
}

pub fn deserialize_polygon<IN: ByteOrder, OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
    metadata_reader: &mut Cursor<&[u8]>,
    dimension: Dimension,
) -> datafusion_common::Result<()> {
    let byte_type = get_polygon_marker(dimension);
    let number_of_rings = metadata_reader.read_u32::<IN>()?;

    write_wkb_byte_order_marker(builder)?;
    builder.write_u32::<OUT>(byte_type)?;
    builder.write_u32::<OUT>(number_of_rings)?;

    for _ in 0..number_of_rings {
        let ring_number_of_points = metadata_reader.read_u32::<IN>()?;
        builder.write_u32::<OUT>(ring_number_of_points)?;

        let mut buf = [0u8; 8];
        for _ in 0..ring_number_of_points * 2 {
            cursor.read_exact(&mut buf)?;
            _ = builder.write(&buf)?;
        }
    }

    Ok(())
}

pub(crate) fn deserialize_multipolygon<IN: ByteOrder, OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
    metadata_reader: &mut Cursor<&[u8]>,
    dimension: Dimension,
) -> datafusion_common::Result<()> {
    let byte_type = match dimension {
        Dimension::XY => 6u32,
        Dimension::XYZ => 1006u32,
        Dimension::XYM => 2006u32,
        Dimension::XYZM => 3006u32,
    };

    let number_of_points = cursor.read_u32::<IN>()?;
    let metadata_start_position = number_of_points * 8 * 2;
    metadata_reader.set_position(cursor.position() + (metadata_start_position) as u64);

    let number_of_geometries = metadata_reader.read_u32::<IN>()?;
    write_wkb_byte_order_marker(builder)?;
    builder.write_u32::<OUT>(byte_type)?;
    builder.write_u32::<OUT>(number_of_geometries)?;

    for _ in 0..number_of_geometries {
        deserialize_polygon::<IN, OUT>(builder, cursor, metadata_reader, dimension)?;
    }

    Ok(())
}

pub(crate) fn deserialize_empty_polygon<OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    dimension: Dimension,
) -> datafusion_common::Result<()> {
    let byte_type = match dimension {
        Dimension::XY => 3u32,
        Dimension::XYZ => 1003u32,
        Dimension::XYM => 2003u32,
        Dimension::XYZM => 3003u32,
    };

    write_wkb_byte_order_marker(builder)?;
    builder.write_u32::<OUT>(byte_type)?;
    builder.write_u32::<OUT>(0u32)?; // 0 rings

    Ok(())
}

pub fn serialize_polygon<OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
) -> datafusion_common::Result<()> {
    let number_of_rings = cursor.read_u32::<OUT>()?;

    let mut total_points = 0u32;
    let coordinates_vector = Vec::new();
    let mut coordinates_cursor = Cursor::new(coordinates_vector);
    let metadata_vector = Vec::new();
    let mut metadata_cursor = Cursor::new(metadata_vector);

    metadata_cursor.write_u32::<OUT>(number_of_rings)?;

    for _ in 0..number_of_rings {
        let number_of_points_in_ring = cursor.read_u32::<OUT>()?;
        metadata_cursor.write_u32::<OUT>(number_of_points_in_ring)?;

        total_points += number_of_points_in_ring;

        let mut buf = vec![0u8; (number_of_points_in_ring * 8 * 2) as usize];
        cursor.read_exact(&mut buf)?;
        _ = coordinates_cursor.write(&buf)?;
    }

    if total_points != 0 {
        builder.write_u32::<OUT>(total_points)?;

        _ = builder.write(coordinates_cursor.get_ref())?;
    }

    _ = builder.write(metadata_cursor.get_ref())?;

    Ok(())
}

pub fn serialize_multipolygon<OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
) -> datafusion_common::Result<()> {
    let number_of_polygons = cursor.read_u32::<OUT>()?;

    let mut total_points = 0u32;
    let coordinates_vector = Vec::new();
    let mut coordinates_cursor = Cursor::new(coordinates_vector);
    let metadata_vector = Vec::new();
    let mut metadata_cursor = Cursor::new(metadata_vector);

    metadata_cursor.write_u32::<OUT>(number_of_polygons)?;

    for _ in 0..number_of_polygons {
        let endianness_marker = cursor.read_u8()?;
        let _geometry_type = cursor.read_u32::<OUT>()?;
        if endianness_marker != 1 {
            return Err(datafusion_common::DataFusionError::Internal(
                "Invalid byte order in WKB".to_string(),
            ));
        }

        if _geometry_type != 3 {
            return Err(datafusion_common::DataFusionError::Internal(
                "Invalid geometry type in WKB".to_string(),
            ));
        }

        let number_of_rings = cursor.read_u32::<OUT>()?;
        metadata_cursor.write_u32::<OUT>(number_of_rings)?;

        for _ in 0..number_of_rings {
            let number_of_points_in_ring = cursor.read_u32::<OUT>()?;
            metadata_cursor.write_u32::<OUT>(number_of_points_in_ring)?;

            total_points += number_of_points_in_ring;

            let mut buf = vec![0u8; (number_of_points_in_ring * 8 * 2) as usize];
            cursor.read_exact(&mut buf)?;
            _ = coordinates_cursor.write(&buf)?;
        }
    }

    builder.write_u32::<OUT>(total_points)?;

    _ = builder.write(coordinates_cursor.get_ref())?;
    _ = builder.write(metadata_cursor.get_ref())?;

    Ok(())
}
