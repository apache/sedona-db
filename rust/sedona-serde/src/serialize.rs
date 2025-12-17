use arrow_array::builder::BinaryBuilder;
use byteorder::LittleEndian;
use datafusion_common::DataFusionError;
use crate::deserialize::parse_geometry;

pub fn serialize(builder: &mut BinaryBuilder) -> datafusion_common::Result<()> {
    use std::io::Cursor;

    // let mut reader = Cursor::new(bytes);

    // parse_geometry::<LittleEndian, LittleEndian>(builder, &mut reader, bytes)
    Err(DataFusionError::NotImplemented(
        "Serialization is not yet implemented".to_string(),
    ))
}