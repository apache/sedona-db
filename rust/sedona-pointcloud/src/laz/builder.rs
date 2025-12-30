use std::{fmt::Debug, sync::Arc};

use arrow_array::{
    builder::{
        BooleanBuilder, FixedSizeBinaryBuilder, Float32Builder, Float64Builder, Int16Builder,
        Int32Builder, Int64Builder, Int8Builder, UInt16Builder, UInt32Builder, UInt64Builder,
        UInt8Builder,
    },
    Array, ArrayRef, BooleanArray, FixedSizeBinaryArray, Float32Array, Float64Array, StructArray,
    UInt16Array, UInt8Array,
};
use arrow_schema::DataType;
use las::{Header, Point};

use crate::laz::{metadata::ExtraAttribute, options::LasExtraBytes, schema::schema_from_header};

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
    extra_bytes: FixedSizeBinaryBuilder,
    extra_attributes: Arc<Vec<ExtraAttribute>>,
    header: Arc<Header>,
}

impl RowBuilder {
    pub fn new(capacity: usize, header: Arc<Header>, attributes: Arc<Vec<ExtraAttribute>>) -> Self {
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
            extra_bytes: FixedSizeBinaryBuilder::with_capacity(
                capacity,
                header.point_format().extra_bytes as i32,
            ),
            extra_attributes: attributes,
            header,
        }
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
            self.extra_bytes.append_value(p.extra_bytes).unwrap();
        }
    }

    /// Note: returns StructArray to allow nesting within another array if desired
    pub fn finish(&mut self, extra: LasExtraBytes) -> StructArray {
        let mut columns = vec![
            Arc::new(self.x.finish()) as ArrayRef,
            Arc::new(self.y.finish()) as ArrayRef,
            Arc::new(self.z.finish()) as ArrayRef,
        ];

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
            match extra {
                LasExtraBytes::Typed => {
                    let extra_bytes = self.extra_bytes.finish();

                    let mut pos = 0;

                    for attribute in self.extra_attributes.iter() {
                        pos += build_attribute(
                            attribute,
                            num_extra_bytes,
                            pos,
                            &extra_bytes,
                            &mut columns,
                        );
                    }
                }
                LasExtraBytes::Blob => columns.push(Arc::new(self.extra_bytes.finish())),
                LasExtraBytes::Ignore => (),
            }
        }

        let schema = schema_from_header(&self.header, extra);
        StructArray::new(schema.fields.to_owned(), columns, None)
    }
}

fn build_attribute(
    attribute: &ExtraAttribute,
    num_extra_bytes: usize,
    pos: usize,
    extra_bytes: &FixedSizeBinaryArray,
    columns: &mut Vec<ArrayRef>,
) -> usize {
    let scale = attribute.scale.unwrap_or(1.0);
    let offset = attribute.offset.unwrap_or(0.0);

    let width = if let DataType::FixedSizeBinary(width) = attribute.data_type {
        width as usize
    } else {
        attribute.data_type.primitive_width().unwrap()
    };

    let iter = extra_bytes
        .value_data()
        .chunks(num_extra_bytes)
        .map(|b| &b[pos..pos + width]);

    match &attribute.data_type {
        DataType::FixedSizeBinary(_) => {
            let data = FixedSizeBinaryArray::try_from_iter(iter).unwrap();
            columns.push(Arc::new(data) as ArrayRef)
        }
        DataType::Int8 => {
            let no_data = attribute.no_data.map(i64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = d[0] as i8;
                if let Some(no_data) = no_data {
                    if no_data == v as i64 {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v as f64 * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = Int8Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }
        DataType::Int16 => {
            let no_data = attribute.no_data.map(i64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = i16::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as i64 {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v as f64 * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = Int16Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }
        DataType::Int32 => {
            let no_data = attribute.no_data.map(i64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = i32::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as i64 {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v as f64 * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = Int32Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }
        DataType::Int64 => {
            let no_data = attribute.no_data.map(i64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = i64::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v as f64 * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = Int64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }
        DataType::UInt8 => {
            let no_data = attribute.no_data.map(u64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = d[0];
                if let Some(no_data) = no_data {
                    if no_data == v as u64 {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v as f64 * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = UInt8Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }
        DataType::UInt16 => {
            let no_data = attribute.no_data.map(u64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = u16::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as u64 {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v as f64 * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = UInt16Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }
        DataType::UInt32 => {
            let no_data = attribute.no_data.map(u64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = u32::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as u64 {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v as f64 * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = UInt32Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }
        DataType::UInt64 => {
            let no_data = attribute.no_data.map(u64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = u64::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v as f64 * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = UInt64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }
        DataType::Float32 => {
            let no_data = attribute.no_data.map(f64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = f32::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v as f64 {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v as f64 * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = Float32Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }
        DataType::Float64 => {
            let no_data = attribute.no_data.map(f64::from_le_bytes);

            let iter = iter.map(|d| {
                let v = f64::from_le_bytes(d.try_into().unwrap());
                if let Some(no_data) = no_data {
                    if no_data == v {
                        return None;
                    }
                }
                Some(v)
            });

            if attribute.scale.is_some() || attribute.offset.is_some() {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v.map(|v| v * scale + offset));
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            } else {
                let mut builder = Float64Builder::with_capacity(extra_bytes.len());
                for v in iter {
                    builder.append_option(v);
                }
                columns.push(Arc::new(builder.finish()) as ArrayRef)
            }
        }

        dt => panic!("Unsupported data type for extra bytes: `{dt}`"),
    }

    width
}
