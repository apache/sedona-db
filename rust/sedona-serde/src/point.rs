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

fn get_byte_type_for_point(dimension: Dimension) -> u32 {
    match dimension {
        Dimension::XY => 1u32,
        Dimension::XYZ => 1001u32,
        Dimension::XYM => 2001u32,
        Dimension::XYZM => 3001u32,
    }
}

pub fn write_empty_point<OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    dimension: Dimension,
) -> Result<()> {
    write_wkb_byte_order_marker(builder)?;
    builder.write_u32::<OUT>(get_byte_type_for_point(dimension))?;

    builder.write_f64::<OUT>(f64::NAN)?; // X
    builder.write_f64::<OUT>(f64::NAN)?; // Y

    Ok(())
}

pub fn parse_point<OUT: ByteOrder>(
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

pub fn parse_multipoint<IN: ByteOrder, OUT: ByteOrder>(
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
        parse_point::<OUT>(builder, cursor, dimension)?;
    }

    Ok(())
}
