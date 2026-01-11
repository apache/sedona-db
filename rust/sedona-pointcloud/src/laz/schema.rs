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

use std::sync::Arc;

use arrow_schema::{ArrowError, DataType, Field, Schema};
use geoarrow_schema::{CoordType, Crs, Dimension, Metadata, PointType, WkbType};
use las::Header;
use las_crs::{get_epsg_from_geotiff_crs, get_epsg_from_wkt_crs_bytes};

use crate::laz::options::{LasExtraBytes, LasPointEncoding};

// Arrow schema for LAS points
pub fn try_schema_from_header(
    header: &Header,
    point_encoding: LasPointEncoding,
    extra_bytes: LasExtraBytes,
) -> Result<Schema, ArrowError> {
    let epsg_crs = if header.has_wkt_crs() {
        header
            .get_wkt_crs_bytes()
            .and_then(|bytes| get_epsg_from_wkt_crs_bytes(bytes).ok())
    } else {
        header
            .get_geotiff_crs()
            .map(|gtc| gtc.and_then(|gtc| get_epsg_from_geotiff_crs(&gtc).ok()))
            .unwrap_or_default()
    };

    let crs = epsg_crs
        .map(|epsg_crs| Crs::from_authority_code(format!("EPSG:{}", epsg_crs.get_horizontal())))
        .unwrap_or_default();

    let mut fields = match point_encoding {
        LasPointEncoding::Plain => vec![
            Field::new("x", DataType::Float64, false),
            Field::new("y", DataType::Float64, false),
            Field::new("z", DataType::Float64, false),
        ],
        LasPointEncoding::Wkb => {
            let point_type = WkbType::new(Arc::new(Metadata::new(crs, None)));
            vec![Field::new("geometry", DataType::Binary, false).with_extension_type(point_type)]
        }
        LasPointEncoding::Native => {
            let point_type = PointType::new(Dimension::XYZ, Arc::new(Metadata::new(crs, None)))
                .with_coord_type(CoordType::Separated);
            vec![point_type.to_field("geometry", false)]
        }
    };
    fields.extend_from_slice(&[
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
    ]);
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
            LasExtraBytes::Typed => fields.extend(extra_bytes_fields(header)?),
            LasExtraBytes::Blob => fields.push(Field::new(
                "extra_bytes",
                DataType::FixedSizeBinary(header.point_format().extra_bytes as i32),
                false,
            )),
            LasExtraBytes::Ignore => (),
        }
    }

    Ok(Schema::new(fields))
}

fn extra_bytes_fields(header: &Header) -> Result<Vec<Field>, ArrowError> {
    let mut fields = Vec::new();

    for vlr in header.all_vlrs() {
        if !(vlr.user_id == "LASF_Spec" && vlr.record_id == 4) {
            continue;
        }

        for bytes in vlr.data.chunks(192) {
            // name
            let name = std::str::from_utf8(&bytes[4..36])?;
            let name = name.trim_end_matches(char::from(0));

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
                    11..=30 => {
                        return Err(ArrowError::ExternalError(
                            "deprecated extra bytes data type".into(),
                        ));
                    }
                    31..=255 => {
                        return Err(ArrowError::ExternalError(
                            "reserved extra bytes data type".into(),
                        ));
                    }
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

    Ok(fields)
}
