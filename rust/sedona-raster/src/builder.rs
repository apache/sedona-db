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

use arrow_array::{
    builder::{
        BinaryBuilder, BinaryViewBuilder, BooleanBuilder, Float64Builder, Int64Builder,
        StringBuilder, StringViewBuilder, UInt32Builder, UInt64Builder,
    },
    Array, ArrayRef, ListArray, StructArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::ArrowError;
use std::sync::Arc;

use sedona_schema::raster::BandDataType;
use sedona_schema::raster::RasterSchema;

use arrow_schema::DataType;

/// Builder for constructing N-D raster arrays.
///
/// # Usage
///
/// ```
/// use sedona_raster::builder::RasterBuilder;
/// use sedona_schema::raster::BandDataType;
///
/// let mut builder = RasterBuilder::new(1);
///
/// // 2D raster convenience: sets transform, x_dim="x", y_dim="y"
/// builder.start_raster_2d(100, 100, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, Some("EPSG:4326")).unwrap();
///
/// // 2D band convenience: sets dim_names=["y","x"], shape=[h,w], contiguous strides
/// builder.start_band_2d(BandDataType::UInt8, Some(&[0u8])).unwrap();
/// builder.band_data_writer().append_value(&vec![0u8; 10000]);
/// builder.finish_band().unwrap();
/// builder.finish_raster().unwrap();
///
/// let raster_array = builder.finish().unwrap();
/// ```
pub struct RasterBuilder {
    // Top-level raster fields
    crs: StringViewBuilder,
    transform_values: Float64Builder,
    transform_offsets: Vec<i32>,
    x_dim: StringViewBuilder,
    y_dim: StringViewBuilder,

    // Band fields (flattened across all bands)
    band_name: StringBuilder,
    band_dim_names_values: StringBuilder,
    band_dim_names_offsets: Vec<i32>,
    band_shape_values: UInt64Builder,
    band_shape_offsets: Vec<i32>,
    band_datatype: UInt32Builder,
    band_nodata: BinaryBuilder,
    band_strides_values: Int64Builder,
    band_strides_offsets: Vec<i32>,
    band_offset: UInt64Builder,
    band_outdb_uri: StringBuilder,
    band_data: BinaryViewBuilder,

    // List structure tracking
    band_offsets: Vec<i32>,  // Track where each raster's bands start/end
    current_band_count: i32, // Track bands in current raster

    // Current raster state (needed for start_band_2d)
    current_width: u64,
    current_height: u64,

    raster_validity: BooleanBuilder,
}

impl RasterBuilder {
    /// Create a new raster builder with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            crs: StringViewBuilder::with_capacity(capacity),
            transform_values: Float64Builder::with_capacity(capacity * 6),
            transform_offsets: vec![0],
            x_dim: StringViewBuilder::with_capacity(capacity),
            y_dim: StringViewBuilder::with_capacity(capacity),

            band_name: StringBuilder::with_capacity(capacity, capacity),
            band_dim_names_values: StringBuilder::with_capacity(capacity * 2, capacity * 4),
            band_dim_names_offsets: vec![0],
            band_shape_values: UInt64Builder::with_capacity(capacity * 2),
            band_shape_offsets: vec![0],
            band_datatype: UInt32Builder::with_capacity(capacity),
            band_nodata: BinaryBuilder::with_capacity(capacity, capacity),
            band_strides_values: Int64Builder::with_capacity(capacity * 2),
            band_strides_offsets: vec![0],
            band_offset: UInt64Builder::with_capacity(capacity),
            band_outdb_uri: StringBuilder::with_capacity(capacity, capacity),
            band_data: BinaryViewBuilder::with_capacity(capacity),

            band_offsets: vec![0],
            current_band_count: 0,
            current_width: 0,
            current_height: 0,

            raster_validity: BooleanBuilder::with_capacity(capacity),
        }
    }

    /// Start a new raster with explicit N-D parameters.
    ///
    /// `transform` must be a 6-element GDAL GeoTransform:
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`
    pub fn start_raster(
        &mut self,
        transform: &[f64; 6],
        x_dim: &str,
        y_dim: &str,
        crs: Option<&str>,
    ) -> Result<(), ArrowError> {
        // Transform
        for &v in transform {
            self.transform_values.append_value(v);
        }
        let next = *self.transform_offsets.last().unwrap() + 6;
        self.transform_offsets.push(next);

        // Spatial dim names
        self.x_dim.append_value(x_dim);
        self.y_dim.append_value(y_dim);

        // CRS
        match crs {
            Some(crs_data) => self.crs.append_value(crs_data),
            None => self.crs.append_null(),
        }

        self.current_band_count = 0;
        self.current_width = 0;
        self.current_height = 0;

        Ok(())
    }

    /// Convenience: start a 2D raster with the legacy 8-parameter interface.
    ///
    /// Sets `x_dim="x"`, `y_dim="y"`, and builds the 6-element GDAL transform
    /// from the individual parameters.
    pub fn start_raster_2d(
        &mut self,
        width: u64,
        height: u64,
        origin_x: f64,
        origin_y: f64,
        scale_x: f64,
        scale_y: f64,
        skew_x: f64,
        skew_y: f64,
        crs: Option<&str>,
    ) -> Result<(), ArrowError> {
        let transform = [origin_x, scale_x, skew_x, origin_y, skew_y, scale_y];
        self.start_raster(&transform, "x", "y", crs)?;
        self.current_width = width;
        self.current_height = height;
        Ok(())
    }

    /// Start a new band with explicit N-D parameters.
    pub fn start_band(
        &mut self,
        name: Option<&str>,
        dim_names: &[&str],
        shape: &[u64],
        data_type: BandDataType,
        nodata: Option<&[u8]>,
    ) -> Result<(), ArrowError> {
        // Name
        match name {
            Some(n) => self.band_name.append_value(n),
            None => self.band_name.append_null(),
        }

        // Dim names
        for dn in dim_names {
            self.band_dim_names_values.append_value(dn);
        }
        let next = *self.band_dim_names_offsets.last().unwrap() + dim_names.len() as i32;
        self.band_dim_names_offsets.push(next);

        // Shape
        for &s in shape {
            self.band_shape_values.append_value(s);
        }
        let next = *self.band_shape_offsets.last().unwrap() + shape.len() as i32;
        self.band_shape_offsets.push(next);

        // Data type
        self.band_datatype.append_value(data_type as u32);

        // Nodata
        match nodata {
            Some(nd) => self.band_nodata.append_value(nd),
            None => self.band_nodata.append_null(),
        }

        // Strides: standard C-order contiguous strides
        let elem_size = data_type.byte_size() as i64;
        let ndim = shape.len();
        let mut strides = vec![0i64; ndim];
        if ndim > 0 {
            strides[ndim - 1] = elem_size;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1] as i64;
            }
        }
        for &s in &strides {
            self.band_strides_values.append_value(s);
        }
        let next = *self.band_strides_offsets.last().unwrap() + ndim as i32;
        self.band_strides_offsets.push(next);

        // Offset (always 0 in Phase 1)
        self.band_offset.append_value(0);

        // OutDb URI (None for in-memory)
        self.band_outdb_uri.append_null();

        self.current_band_count += 1;

        Ok(())
    }

    /// Convenience: start a 2D band with `dim_names=["y","x"]` and `shape=[height, width]`.
    ///
    /// Must be called after `start_raster_2d` which sets the current width/height.
    pub fn start_band_2d(
        &mut self,
        data_type: BandDataType,
        nodata: Option<&[u8]>,
    ) -> Result<(), ArrowError> {
        self.start_band(
            None,
            &["y", "x"],
            &[self.current_height, self.current_width],
            data_type,
            nodata,
        )
    }

    /// Get direct access to the BinaryViewBuilder for writing the current band's data.
    pub fn band_data_writer(&mut self) -> &mut BinaryViewBuilder {
        &mut self.band_data
    }

    /// Finish writing the current band.
    pub fn finish_band(&mut self) -> Result<(), ArrowError> {
        Ok(())
    }

    /// Finish all bands for the current raster.
    pub fn finish_raster(&mut self) -> Result<(), ArrowError> {
        let next_offset = self.band_offsets.last().unwrap() + self.current_band_count;
        self.band_offsets.push(next_offset);
        self.raster_validity.append_value(true);
        Ok(())
    }

    /// Append a null raster.
    pub fn append_null(&mut self) -> Result<(), ArrowError> {
        // Transform: append 6 zeros
        for _ in 0..6 {
            self.transform_values.append_value(0.0);
        }
        let next = *self.transform_offsets.last().unwrap() + 6;
        self.transform_offsets.push(next);

        // Spatial dims: defaults
        self.x_dim.append_value("x");
        self.y_dim.append_value("y");

        // CRS: null
        self.crs.append_null();

        // No bands
        let current_offset = *self.band_offsets.last().unwrap();
        self.band_offsets.push(current_offset);

        // Mark null
        self.raster_validity.append_null();

        Ok(())
    }

    /// Finish building and return the constructed StructArray.
    pub fn finish(mut self) -> Result<StructArray, ArrowError> {
        // Build transform list
        let transform_values = self.transform_values.finish();
        let transform_offsets = OffsetBuffer::new(ScalarBuffer::from(self.transform_offsets));
        let DataType::List(transform_field) = RasterSchema::transform_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for transform".to_string(),
            ));
        };
        let transform_list = ListArray::new(
            transform_field,
            transform_offsets,
            Arc::new(transform_values),
            None,
        );

        // Build band dim_names nested list
        let dim_names_values = self.band_dim_names_values.finish();
        let dim_names_offsets = OffsetBuffer::new(ScalarBuffer::from(self.band_dim_names_offsets));
        let DataType::List(dim_names_field) = RasterSchema::dim_names_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for dim_names".to_string(),
            ));
        };
        let dim_names_list = ListArray::new(
            dim_names_field,
            dim_names_offsets,
            Arc::new(dim_names_values),
            None,
        );

        // Build band shape nested list
        let shape_values = self.band_shape_values.finish();
        let shape_offsets = OffsetBuffer::new(ScalarBuffer::from(self.band_shape_offsets));
        let DataType::List(shape_field) = RasterSchema::shape_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for shape".to_string(),
            ));
        };
        let shape_list = ListArray::new(shape_field, shape_offsets, Arc::new(shape_values), None);

        // Build band strides nested list
        let strides_values = self.band_strides_values.finish();
        let strides_offsets = OffsetBuffer::new(ScalarBuffer::from(self.band_strides_offsets));
        let DataType::List(strides_field) = RasterSchema::strides_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for strides".to_string(),
            ));
        };
        let strides_list = ListArray::new(
            strides_field,
            strides_offsets,
            Arc::new(strides_values),
            None,
        );

        // Build band struct
        let DataType::Struct(band_fields) = RasterSchema::band_type() else {
            return Err(ArrowError::SchemaError(
                "Expected struct type for band".to_string(),
            ));
        };

        let band_arrays: Vec<ArrayRef> = vec![
            Arc::new(self.band_name.finish()),
            Arc::new(dim_names_list),
            Arc::new(shape_list),
            Arc::new(self.band_datatype.finish()),
            Arc::new(self.band_nodata.finish()),
            Arc::new(strides_list),
            Arc::new(self.band_offset.finish()),
            Arc::new(self.band_outdb_uri.finish()),
            Arc::new(self.band_data.finish()),
        ];
        let band_struct = StructArray::new(band_fields, band_arrays, None);

        // Build bands list
        let DataType::List(bands_field) = RasterSchema::bands_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for bands".to_string(),
            ));
        };
        let band_list_offsets = OffsetBuffer::new(ScalarBuffer::from(self.band_offsets));
        let bands_list =
            ListArray::new(bands_field, band_list_offsets, Arc::new(band_struct), None);

        // Build top-level raster struct
        let raster_fields = RasterSchema::fields();
        let raster_arrays: Vec<ArrayRef> = vec![
            Arc::new(self.crs.finish()),
            Arc::new(transform_list),
            Arc::new(self.x_dim.finish()),
            Arc::new(self.y_dim.finish()),
            Arc::new(bands_list),
        ];

        let raster_validity_array = self.raster_validity.finish();
        let raster_nulls = raster_validity_array.nulls().cloned();

        Ok(StructArray::new(raster_fields, raster_arrays, raster_nulls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::RasterStructArray;
    use crate::traits::RasterRef;

    #[test]
    fn test_roundtrip_2d_raster() {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(
                10,
                20,
                100.0,
                200.0,
                1.0,
                -2.0,
                0.25,
                0.5,
                Some("EPSG:4326"),
            )
            .unwrap();
        builder
            .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
            .unwrap();
        builder.band_data_writer().append_value(&vec![1u8; 200]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        assert_eq!(rasters.len(), 1);

        let r = rasters.get(0).unwrap();
        assert_eq!(r.width(), Some(10));
        assert_eq!(r.height(), Some(20));
        assert_eq!(r.transform(), &[100.0, 1.0, 0.25, 200.0, 0.5, -2.0]);
        assert_eq!(r.x_dim(), "x");
        assert_eq!(r.y_dim(), "y");
        assert_eq!(r.crs(), Some("EPSG:4326"));
        assert_eq!(r.num_bands(), 1);

        let band = r.band(0).unwrap();
        assert_eq!(band.ndim(), 2);
        assert_eq!(band.dim_names(), vec!["y", "x"]);
        assert_eq!(band.shape(), &[20, 10]);
        assert_eq!(band.data_type(), BandDataType::UInt8);
        assert_eq!(band.nodata(), Some(&[255u8][..]));
        assert_eq!(band.contiguous_data().unwrap().len(), 200);
    }

    #[test]
    fn test_roundtrip_multi_band() {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(2, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();

        // Band 0: UInt8
        builder
            .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
            .unwrap();
        builder.band_data_writer().append_value(&[1u8, 2, 3, 4]);
        builder.finish_band().unwrap();

        // Band 1: Float32
        builder.start_band_2d(BandDataType::Float32, None).unwrap();
        let f32_data: Vec<u8> = [1.5f32, 2.5, 3.5, 4.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        builder.band_data_writer().append_value(&f32_data);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();

        assert_eq!(r.num_bands(), 2);

        let b0 = r.band(0).unwrap();
        assert_eq!(b0.data_type(), BandDataType::UInt8);
        assert_eq!(b0.nodata(), Some(&[255u8][..]));

        let b1 = r.band(1).unwrap();
        assert_eq!(b1.data_type(), BandDataType::Float32);
        assert_eq!(b1.nodata(), None);
    }

    #[test]
    fn test_null_raster() {
        let mut builder = RasterBuilder::new(2);
        builder
            .start_raster_2d(1, 1, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.band_data_writer().append_value(&[0u8]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        builder.append_null().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        assert_eq!(rasters.len(), 2);
        assert!(!rasters.is_null(0));
        assert!(rasters.is_null(1));
    }

    #[test]
    fn test_nd_band() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, "x", "y", None).unwrap();

        // 3D band: [time=3, y=4, x=5]
        builder
            .start_band(
                Some("temperature"),
                &["time", "y", "x"],
                &[3, 4, 5],
                BandDataType::Float32,
                None,
            )
            .unwrap();
        let data = vec![0u8; 3 * 4 * 5 * 4]; // 3*4*5 Float32 elements
        builder.band_data_writer().append_value(&data);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();

        assert_eq!(r.band_name(0), Some("temperature"));
        let band = r.band(0).unwrap();
        assert_eq!(band.ndim(), 3);
        assert_eq!(band.dim_names(), vec!["time", "y", "x"]);
        assert_eq!(band.shape(), &[3, 4, 5]);
        assert_eq!(band.dim_size("time"), Some(3));
        assert_eq!(band.dim_size("y"), Some(4));
        assert_eq!(band.dim_size("x"), Some(5));
        assert_eq!(band.dim_size("z"), None);

        // Verify strides are standard C-order: [4*5*4, 5*4, 4] = [80, 20, 4]
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[80, 20, 4]);
        assert_eq!(buf.offset, 0);
    }
}
