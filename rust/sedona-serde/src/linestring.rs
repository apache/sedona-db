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

fn get_linestring_marker(dimension: Dimension) -> u32 {
    match dimension {
        Dimension::XY => 2u32,
        Dimension::XYZ => 1002u32,
        Dimension::XYM => 2002u32,
        Dimension::XYZM => 3002u32,
    }
}

pub fn parse_linestring<IN: ByteOrder, OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
    dimension: Dimension,
) -> datafusion_common::Result<()> {
    let number_of_points = cursor.read_u32::<IN>()?;
    let byte_type = get_linestring_marker(dimension);

    write_wkb_byte_order_marker(builder)?;

    builder.write_u32::<OUT>(byte_type)?;

    builder.write_u32::<OUT>(number_of_points)?;

    let mut buf = [0u8; 8];
    for _ in 0..number_of_points * 2 {
        cursor.read_exact(&mut buf)?;
        _ = builder.write(&buf)?;
    }

    Ok(())
}

pub fn parse_multilinestring<IN: ByteOrder, OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
    metadata_reader: &mut Cursor<&[u8]>,
    dimension: Dimension,
) -> datafusion_common::Result<()> {
    let byte_type = match dimension {
        Dimension::XY => 5u32,
        Dimension::XYZ => 1005u32,
        Dimension::XYM => 2005u32,
        Dimension::XYZM => 3005u32,
    };

    let linestring_type = get_linestring_marker(dimension);

    write_wkb_byte_order_marker(builder)?;
    builder.write_u32::<OUT>(byte_type)?;

    let number_of_points = cursor.read_u32::<IN>()?;

    metadata_reader.set_position(cursor.position() + (number_of_points * 8 * 2) as u64);

    let number_of_geometries = metadata_reader.read_u32::<IN>()?;

    builder.write_u32::<OUT>(number_of_geometries)?;

    for _ in 0..number_of_geometries {
        let number_of_points_in_linestring = metadata_reader.read_u32::<IN>()?;
        write_wkb_byte_order_marker(builder)?;
        builder.write_u32::<OUT>(linestring_type)?;

        builder.write_u32::<OUT>(number_of_points_in_linestring)?;

        for _ in 0..number_of_points_in_linestring * 2 {
            let mut buf = [0u8; 8];
            cursor.read_exact(&mut buf)?;

            _ = builder.write(&buf)?;
        }
    }

    Ok(())
}
