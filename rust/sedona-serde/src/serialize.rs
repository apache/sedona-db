use crate::linestring::{serialize_linestring, serialize_multilinestring};
use crate::point::{serialize_multipoint, serialize_point};
use crate::polygon::{serialize_multipolygon, serialize_polygon};
use arrow_array::builder::BinaryBuilder;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use datafusion_common::DataFusionError;
use std::io::Cursor;
use wkb::reader::Wkb;
use wkt::types::Dimension;

pub fn serialize(
    wkb: &Wkb,
    builder: &mut BinaryBuilder,
    epsg_crs: Option<u32>,
) -> datafusion_common::Result<()> {
    use std::io::Cursor;
    let mut cursor = Cursor::new(wkb.buf());
    let byte_order = cursor.read_u8()?;

    if byte_order != 1 && byte_order != 0 {
        return Err(DataFusionError::Internal(
            "Invalid byte order in WKB".to_string(),
        ));
    }

    match byte_order {
        0 => Err(DataFusionError::Internal(
            "BigEndian WKB serialization not implemented".to_string(),
        )),
        1 => write_geometry::<LittleEndian, LittleEndian>(builder, &mut cursor, epsg_crs),
        _ => unreachable!(),
    }
}

pub fn write_geometry<IN: ByteOrder, OUT: ByteOrder>(
    builder: &mut BinaryBuilder,
    cursor: &mut Cursor<&[u8]>,
    epsg_crs: Option<u32>,
) -> datafusion_common::Result<()> {
    let geometry_type = cursor.read_u32::<IN>()?;
    verify_geometry_type(geometry_type)?;

    let wkb_byte = geometry_type as u8;

    let preamble_byte: u8 = (wkb_byte << 4)
        | (get_coordinate_type_value(Dimension::XY) << 1)
        | if epsg_crs.is_some() { 1 } else { 0 };

    builder.write_u8(preamble_byte)?;

    if let Some(srid) = epsg_crs {
        builder.write_u8(((srid >> 16) & 0xFF) as u8)?;
        builder.write_u8(((srid >> 8) & 0xFF) as u8)?;
        builder.write_u8((srid & 0xFF) as u8)?;
    } else {
        builder.write_u8(0)?;
        builder.write_u8(0)?;
        builder.write_u8(0)?;
    }

    match wkb_byte {
        1 => return serialize_point::<LittleEndian>(builder, cursor),
        2 => return serialize_linestring::<LittleEndian>(builder, cursor),
        3 => return serialize_polygon::<LittleEndian>(builder, cursor),
        4 => return serialize_multipoint::<LittleEndian>(builder, cursor),
        5 => return serialize_multilinestring::<LittleEndian>(builder, cursor),
        6 => return serialize_multipolygon::<LittleEndian>(builder, cursor),
        7 => {
            let number_of_geometries = cursor.read_u32::<IN>()?;
            builder.write_u32::<OUT>(number_of_geometries)?;
            for _ in 0..number_of_geometries {
                _ = cursor.read_u8()?;
                write_geometry::<IN, OUT>(builder, cursor, epsg_crs)?;
            }
        }
        _ => {
            return Err(DataFusionError::Internal(
                "Geometry type not supported yet".to_string(),
            ))
        }
    }

    Ok(())
}

fn verify_geometry_type(geometry_type: u32) -> datafusion_common::Result<()> {
    match geometry_type {
        1..=7 => Ok(()),
        _ => Err(DataFusionError::Internal(
            "Unsupported geometry type".to_string(),
        )),
    }
}

fn get_coordinate_type_value(dimension: Dimension) -> u8 {
    match dimension {
        Dimension::XY => 1,
        Dimension::XYZ => 2,
        Dimension::XYM => 3,
        Dimension::XYZM => 4,
    }
}
