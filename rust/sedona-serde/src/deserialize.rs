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

use crate::linestring::{deserialize_linestring, deserialize_multilinestring};
use crate::point::{deserialize_multipoint, deserialize_point, deserialize_empty_point};
use crate::polygon::{deserialize_multipolygon, deserialize_polygon, deserialize_empty_polygon};
use crate::wkb::write_wkb_byte_order_marker;
use arrow_array::builder::BinaryBuilder;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use datafusion_common::error::DataFusionError;
use std::io::Cursor;
use wkt::types::Dimension;

pub fn deserialize(builder: &mut BinaryBuilder, bytes: &[u8]) -> datafusion_common::Result<()> {
    use std::io::Cursor;

    if bytes.len() < 8 {
        return Err(DataFusionError::Internal(
            "Sedona bytes are too short".to_string(),
        ));
    }

    let mut reader = Cursor::new(bytes);

    deserialize_geometry::<LittleEndian, LittleEndian>(builder, &mut reader, bytes)
}

pub fn deserialize_geometry<IN: ByteOrder, OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
    bytes: &[u8],
) -> datafusion_common::Result<()> {
    let preamble_byte = cursor.read_u8()?;

    let wkb_type = preamble_byte >> 4;

    let dimension = get_dimension((preamble_byte) >> 1);

    if dimension != Dimension::XY {
        return Err(DataFusionError::Execution(
            "Only 2D geometries (XY) are supported".to_string(),
        ));
    }

    let _has_srid = (preamble_byte & 0x01) != 0;

    cursor.set_position(cursor.position() + 3); // Skip 3 bytes

    match wkb_type {
        1 => {
            let number_of_coordinates = cursor.read_u32::<IN>()?;
            if number_of_coordinates == 0 {
                deserialize_empty_point::<OUT>(builder, dimension)?;
                return Ok(());
            }

            deserialize_point::<OUT>(builder, cursor, dimension)?;
        }
        2 => {
            deserialize_linestring::<IN, OUT>(builder, cursor, dimension)?;
        }
        3 => {
            let mut meta_data_reader = Cursor::new(bytes);

            let number_of_points = cursor.read_u32::<IN>()?;
            if number_of_points == 0 {
                deserialize_empty_polygon::<OUT>(builder, dimension)?;

                return Ok(());
            }

            let metadata_start_position = number_of_points * 8 * 2;
            meta_data_reader.set_position(cursor.position() + (metadata_start_position) as u64);

            deserialize_polygon::<IN, OUT>(builder, cursor, &mut meta_data_reader, dimension)?;
            cursor.set_position(meta_data_reader.position());
        }
        4 => {
            deserialize_multipoint::<IN, OUT>(builder, cursor, dimension)?;
        }
        5 => {
            let mut meta_data_reader = Cursor::new(bytes);
            deserialize_multilinestring::<IN, OUT>(builder, cursor, &mut meta_data_reader, dimension)?;
            cursor.set_position(meta_data_reader.position());
        }
        6 => {
            let mut meta_data_reader = Cursor::new(bytes);
            deserialize_multipolygon::<IN, OUT>(builder, cursor, &mut meta_data_reader, dimension)?;
            cursor.set_position(meta_data_reader.position());
        }
        7 => {
            let number_of_geometries = cursor.read_u32::<IN>()?;
            write_wkb_byte_order_marker(builder)?;
            builder.write_u32::<OUT>(get_byte_type_for_geometry_collection(dimension))?;

            builder.write_u32::<OUT>(number_of_geometries)?;

            for _i in 0..number_of_geometries {
                deserialize_geometry::<IN, OUT>(builder, cursor, bytes)?;
            }
        }
        _ => {
            return Err(DataFusionError::Execution(format!(
                "Unsupported geometry type: {}",
                wkb_type
            )))
        }
    }

    Ok(())
}

fn get_byte_type_for_geometry_collection(dimension: Dimension) -> u32 {
    match dimension {
        Dimension::XY => 7u32,
        Dimension::XYZ => 1007u32,
        Dimension::XYM => 2007u32,
        Dimension::XYZM => 3007u32,
    }
}

fn get_dimension(b: u8) -> Dimension {
    match b {
        1 => Dimension::XY,
        2 => Dimension::XYZ,
        3 => Dimension::XYM,
        4 => Dimension::XYZM,
        _ => Dimension::XY,
    }
}
