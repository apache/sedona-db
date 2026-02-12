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
use datafusion_common::error::Result;
use std::io::{Cursor, Read, Write};
use wkt::types::Dimension;

const NAN_2X: [u8; 16] = [0, 0, 0, 0, 0, 0, 248, 127, 0, 0, 0, 0, 0, 0, 248, 127];

fn get_byte_type_for_point(dimension: Dimension) -> u32 {
    match dimension {
        Dimension::XY => 1u32,
        Dimension::XYZ => 1001u32,
        Dimension::XYM => 2001u32,
        Dimension::XYZM => 3001u32,
    }
}

pub fn deserialize_empty_point<OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    dimension: Dimension,
) -> Result<()> {
    write_wkb_byte_order_marker(builder)?;
    builder.write_u32::<OUT>(get_byte_type_for_point(dimension))?;

    builder.write_f64::<OUT>(f64::NAN)?; // X
    builder.write_f64::<OUT>(f64::NAN)?; // Y

    Ok(())
}

pub fn deserialize_point<OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
    dimension: Dimension,
) -> Result<()> {
    write_wkb_byte_order_marker(builder)?;
    builder.write_u32::<OUT>(get_byte_type_for_point(dimension))?;

    let mut buf = [0u8; 8];
    cursor.read_exact(&mut buf)?;

    _ = builder.write(&buf)?;

    cursor.read_exact(&mut buf)?;
    _ = builder.write(&buf)?;

    Ok(())
}

pub fn deserialize_multipoint<IN: ByteOrder, OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
    dimension: Dimension,
) -> Result<()> {
    let number_of_points = cursor.read_u32::<IN>()?;

    let byte_type = match dimension {
        Dimension::XY => 4u32,
        Dimension::XYZ => 1004u32,
        Dimension::XYM => 2004u32,
        Dimension::XYZM => 3004u32,
    };

    write_wkb_byte_order_marker(builder)?;
    builder.write_u32::<OUT>(byte_type)?;

    if number_of_points == 0 {
        builder.write_u32::<OUT>(0)?;
        return Ok(());
    }

    builder.write_u32::<OUT>(number_of_points)?;

    for _ in 0..number_of_points {
        deserialize_point::<OUT>(builder, cursor, dimension)?;
    }

    Ok(())
}

pub fn serialize_point<OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
) -> Result<()> {
    let mut buf = [0u8; 16];
    cursor.read_exact(&mut buf)?;
    if buf == NAN_2X {
        builder.write_u32::<OUT>(0)?; // no coordinates

        return Ok(());
    }

    builder.write_u32::<OUT>(1)?; // numCoordinates
    builder.write_all(&buf)?;

    Ok(())
}

pub fn serialize_multipoint<OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
) -> Result<()> {
    let number_of_points = cursor.read_u32::<OUT>()?;
    builder.write_u32::<OUT>(number_of_points)?; // numPoints
    for _ in 0..number_of_points {
        let endianness_marker = cursor.read_u8()?;
        let _geometry_type = cursor.read_u32::<OUT>()?;

        if _geometry_type != 1 {
            return Err(datafusion_common::DataFusionError::Internal(
                "Invalid geometry type in WKB".to_string(),
            ));
        }

        if endianness_marker != 1 {
            return Err(datafusion_common::DataFusionError::Internal(
                "Invalid byte order in WKB".to_string(),
            ));
        }

        let mut buf = [0u8; 16];
        cursor.read_exact(&mut buf)?;
        builder.write_all(&buf)?;
    }

    Ok(())
}
