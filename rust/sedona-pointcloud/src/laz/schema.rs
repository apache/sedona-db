use arrow_schema::{DataType, Field, Schema};
use las::Header;

use crate::laz::options::LasExtraBytes;

// Arrow schema for LAS points
pub fn schema_from_header(header: &Header, extra_bytes: LasExtraBytes) -> Schema {
    let mut fields = vec![
        Field::new("x", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
        Field::new("z", DataType::Float64, false),
        Field::new("intensity", DataType::UInt16, true),
        Field::new("return_number", DataType::UInt8, false),
        Field::new("number_of_returns", DataType::UInt8, false),
        Field::new("is_synthetic", DataType::Boolean, false),
        Field::new("is_key_point", DataType::Boolean, false),
        Field::new("is_withheld", DataType::Boolean, false),
        Field::new("is_overlap", DataType::Boolean, false),
        Field::new("scanner_channel", DataType::UInt8, false),
        Field::new("scan_direction", DataType::UInt8, false),
        Field::new("is_edge_of_flight_line", DataType::Boolean, false),
        Field::new("classification", DataType::UInt8, false),
        Field::new("user_data", DataType::UInt8, false),
        Field::new("scan_angle", DataType::Float32, false),
        Field::new("point_source_id", DataType::UInt16, false),
    ];
    if header.point_format().has_gps_time {
        fields.push(Field::new("gps_time", DataType::Float64, false));
    }
    if header.point_format().has_color {
        fields.extend([
            Field::new("red", DataType::UInt16, false),
            Field::new("green", DataType::UInt16, false),
            Field::new("blue", DataType::UInt16, false),
        ])
    }
    if header.point_format().has_nir {
        fields.push(Field::new("nir", DataType::UInt16, false));
    }

    // extra bytes
    if header.point_format().extra_bytes > 0 {
        match extra_bytes {
            LasExtraBytes::Typed => fields.extend(extra_bytes_fields(header)),
            LasExtraBytes::Blob => fields.push(Field::new(
                "extra_bytes",
                DataType::FixedSizeBinary(header.point_format().extra_bytes as i32),
                false,
            )),
            LasExtraBytes::Ignore => (),
        }
    }

    Schema::new(fields)
}

fn extra_bytes_fields(header: &Header) -> Vec<Field> {
    let mut fields = Vec::new();

    for vlr in header.all_vlrs() {
        if !(vlr.user_id == "LASF_Spec" && vlr.record_id == 4) {
            continue;
        }

        for bytes in vlr.data.chunks(192) {
            // name
            let name = std::str::from_utf8(&bytes[4..36])
                .unwrap()
                .trim_end_matches(char::from(0));

            // data type
            let data_type = if bytes[2] != 0 && (bytes[3] >> 3 & 1 == 1 || bytes[3] >> 4 & 1 == 1) {
                // if scaled and/or offset resolve to f64
                DataType::Float64
            } else {
                match bytes[2] {
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
                }
            };

            // nullability
            let nullable = if bytes[2] != 0 && bytes[3] & 1 == 1 {
                true // data bit is valid and set
            } else {
                false
            };

            fields.push(Field::new(name, data_type, nullable));
        }
    }

    fields
}
