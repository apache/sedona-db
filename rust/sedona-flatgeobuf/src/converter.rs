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

use arrow_array::{ArrayRef, builder::*, new_null_array};
use arrow_schema::FieldRef;
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use flatgeobuf::ColumnType;

/// One typed buffer per projected column per batch. Finishing moves buffers into
/// Arrow; only a requested type conversion allocates a replacement array.
pub(crate) struct FlatGeobufArrowConverter {
    output: FieldRef,
    builder: Option<Box<dyn ArrayBuilder>>,
    rows: usize,
}
impl FlatGeobufArrowConverter {
    pub(crate) fn new(
        output: FieldRef,
        source: Option<ColumnType>,
        capacity: usize,
    ) -> Result<Self> {
        let builder = source
            .map(crate::metadata::column_type)
            .transpose()?
            .map(|t| make_builder(&t, capacity));
        Ok(Self {
            output,
            builder,
            rows: 0,
        })
    }
    pub(crate) fn append_null(&mut self) {
        if let Some(b) = self.builder.as_mut() {
            macro_rules! null {
                ($($ty:ty),*) => {$(if let Some(v)=b.as_any_mut().downcast_mut::<$ty>() {v.append_null();})*};
            }
            null!(
                Int8Builder,
                UInt8Builder,
                Int16Builder,
                UInt16Builder,
                Int32Builder,
                UInt32Builder,
                Int64Builder,
                UInt64Builder,
                Float32Builder,
                Float64Builder,
                BooleanBuilder,
                StringBuilder,
                BinaryBuilder,
                TimestampMicrosecondBuilder
            );
        }
        self.rows += 1;
    }
    pub(crate) fn append(&mut self, ty: ColumnType, bytes: &[u8]) -> Result<()> {
        let b = self
            .builder
            .as_mut()
            .ok_or_else(|| exec_datafusion_err!("Missing source column"))?;
        macro_rules! number {
            ($native:ty, $builder:ty) => {{
                let raw: [u8; std::mem::size_of::<$native>()] = bytes
                    .try_into()
                    .map_err(|_| exec_datafusion_err!("Invalid FlatGeobuf numeric width"))?;
                b.as_any_mut()
                    .downcast_mut::<$builder>()
                    .unwrap()
                    .append_value(<$native>::from_le_bytes(raw));
            }};
        }
        match ty {
            ColumnType::Byte => number!(i8, Int8Builder),
            ColumnType::UByte => number!(u8, UInt8Builder),
            ColumnType::Short => number!(i16, Int16Builder),
            ColumnType::UShort => number!(u16, UInt16Builder),
            ColumnType::Int => number!(i32, Int32Builder),
            ColumnType::UInt => number!(u32, UInt32Builder),
            ColumnType::Long => number!(i64, Int64Builder),
            ColumnType::ULong => number!(u64, UInt64Builder),
            ColumnType::Float => number!(f32, Float32Builder),
            ColumnType::Double => number!(f64, Float64Builder),
            ColumnType::Bool => {
                if bytes.len() != 1 || bytes[0] > 1 {
                    return exec_err!("Invalid FlatGeobuf boolean");
                }
                b.as_any_mut()
                    .downcast_mut::<BooleanBuilder>()
                    .unwrap()
                    .append_value(bytes[0] != 0);
            }
            ColumnType::Binary => b
                .as_any_mut()
                .downcast_mut::<BinaryBuilder>()
                .unwrap()
                .append_value(bytes),
            ColumnType::String | ColumnType::Json | ColumnType::DateTime => {
                let text = std::str::from_utf8(bytes)
                    .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf UTF-8: {e}"))?;
                if ty == ColumnType::DateTime {
                    let dt = chrono::DateTime::parse_from_rfc3339(text)
                        .map_err(|e| exec_datafusion_err!("Invalid FlatGeobuf DateTime: {e}"))?;
                    b.as_any_mut()
                        .downcast_mut::<TimestampMicrosecondBuilder>()
                        .unwrap()
                        .append_value(dt.timestamp_micros());
                } else {
                    b.as_any_mut()
                        .downcast_mut::<StringBuilder>()
                        .unwrap()
                        .append_value(text);
                }
            }
            _ => return exec_err!("Unsupported FlatGeobuf property type"),
        }
        self.rows += 1;
        Ok(())
    }
    pub(crate) fn finish(mut self) -> Result<ArrayRef> {
        let array = if let Some(b) = self.builder.as_mut() {
            let array = b.finish();
            if array.data_type() == self.output.data_type() {
                array
            } else {
                arrow_cast::cast_with_options(
                    &array,
                    self.output.data_type(),
                    &arrow_cast::CastOptions {
                        safe: false,
                        ..Default::default()
                    },
                )?
            }
        } else {
            new_null_array(self.output.data_type(), self.rows)
        };
        if !self.output.is_nullable() && array.null_count() > 0 {
            return exec_err!(
                "Missing non-nullable FlatGeobuf property: {}",
                self.output.name()
            );
        }
        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field};
    #[test]
    fn identity_and_numeric_widening_preserve_values_and_nulls() {
        let cases = [
            (
                ColumnType::Byte,
                DataType::Int8,
                (-7i8).to_le_bytes().to_vec(),
            ),
            (
                ColumnType::UByte,
                DataType::UInt8,
                7u8.to_le_bytes().to_vec(),
            ),
            (
                ColumnType::Short,
                DataType::Int16,
                (-7i16).to_le_bytes().to_vec(),
            ),
            (
                ColumnType::UShort,
                DataType::UInt16,
                7u16.to_le_bytes().to_vec(),
            ),
            (
                ColumnType::Int,
                DataType::Int32,
                (-7i32).to_le_bytes().to_vec(),
            ),
            (
                ColumnType::UInt,
                DataType::UInt32,
                7u32.to_le_bytes().to_vec(),
            ),
            (
                ColumnType::Long,
                DataType::Int64,
                (-7i64).to_le_bytes().to_vec(),
            ),
            (
                ColumnType::ULong,
                DataType::UInt64,
                7u64.to_le_bytes().to_vec(),
            ),
            (
                ColumnType::Float,
                DataType::Float32,
                7f32.to_le_bytes().to_vec(),
            ),
            (
                ColumnType::Double,
                DataType::Float64,
                7f64.to_le_bytes().to_vec(),
            ),
        ];
        for (ty, dt, value) in cases {
            for output in [dt, DataType::Float64] {
                let mut c = FlatGeobufArrowConverter::new(
                    std::sync::Arc::new(Field::new("v", output.clone(), true)),
                    Some(ty),
                    2,
                )
                .unwrap();
                c.append(ty, &value).unwrap();
                c.append_null();
                let a = c.finish().unwrap();
                assert_eq!(a.data_type(), &output);
                assert_eq!(a.null_count(), 1);
                let got = arrow_cast::cast(&a, &DataType::Float64).unwrap();
                let got = got
                    .as_any()
                    .downcast_ref::<arrow_array::Float64Array>()
                    .unwrap();
                assert_eq!(
                    got.value(0),
                    if [
                        ColumnType::Byte,
                        ColumnType::Short,
                        ColumnType::Int,
                        ColumnType::Long
                    ]
                    .contains(&ty)
                    {
                        -7.0
                    } else {
                        7.0
                    }
                );
            }
        }
    }
    #[test]
    fn variable_types_datetime_boolean_missing_and_errors() {
        for (ty, dt, raw) in [
            (ColumnType::Bool, DataType::Boolean, vec![1]),
            (ColumnType::String, DataType::Utf8, b"hello".to_vec()),
            (ColumnType::Json, DataType::Utf8, b"{\"x\":1}".to_vec()),
            (ColumnType::Binary, DataType::Binary, vec![0, 255]),
            (
                ColumnType::DateTime,
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
                b"1970-01-01T01:00:00+01:00".to_vec(),
            ),
        ] {
            let mut c = FlatGeobufArrowConverter::new(
                std::sync::Arc::new(Field::new("v", dt.clone(), true)),
                Some(ty),
                2,
            )
            .unwrap();
            c.append(ty, &raw).unwrap();
            c.append_null();
            let a = c.finish().unwrap();
            assert_eq!(a.null_count(), 1);
            match ty {
                ColumnType::Bool => assert!(
                    a.as_any()
                        .downcast_ref::<arrow_array::BooleanArray>()
                        .unwrap()
                        .value(0)
                ),
                ColumnType::Binary => assert_eq!(
                    a.as_any()
                        .downcast_ref::<arrow_array::BinaryArray>()
                        .unwrap()
                        .value(0),
                    raw
                ),
                ColumnType::DateTime => assert_eq!(
                    a.as_any()
                        .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
                        .unwrap()
                        .value(0),
                    0
                ),
                _ => assert_eq!(
                    a.as_any()
                        .downcast_ref::<arrow_array::StringArray>()
                        .unwrap()
                        .value(0)
                        .as_bytes(),
                    raw
                ),
            }
        }
        let field = std::sync::Arc::new(Field::new("missing", DataType::Int32, false));
        let mut c = FlatGeobufArrowConverter::new(field, None, 1).unwrap();
        c.append_null();
        assert!(c.finish().is_err());
        let field = std::sync::Arc::new(Field::new("small", DataType::Int8, true));
        let mut c = FlatGeobufArrowConverter::new(field, Some(ColumnType::Int), 1).unwrap();
        c.append(ColumnType::Int, &256i32.to_le_bytes()).unwrap();
        assert!(c.finish().is_err());
    }
}
