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

use arrow_array::{Array, ArrayRef, BinaryArray, BinaryViewArray, StringViewArray, StructArray};
use arrow_schema::DataType;
use datafusion_common::cast::{
    as_binary_array, as_binary_view_array, as_string_view_array, as_struct_array,
};
use datafusion_common::error::Result;
use datafusion_common::{exec_err, DataFusionError, ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use sedona_raster::array::{RasterRefImpl, RasterStructArray};
use sedona_schema::crs::deserialize_crs;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::datatypes::RASTER;

/// Helper for writing raster kernel implementations
///
/// The [RasterExecutor] provides a simplified interface for executing functions
/// on raster arrays, handling the common pattern of downcasting to StructArray,
/// creating raster iterators, and handling null values.
pub struct RasterExecutor<'a, 'b> {
    pub arg_types: &'a [SedonaType],
    pub args: &'b [ColumnarValue],
    num_iterations: usize,
}

#[derive(Clone)]
enum ItemWkbAccessor {
    Binary(BinaryArray),
    BinaryView(BinaryViewArray),
}

impl ItemWkbAccessor {
    #[inline]
    fn get(&self, i: usize) -> Option<&[u8]> {
        match self {
            Self::Binary(arr) => {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            }
            Self::BinaryView(arr) => {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            }
        }
    }
}

enum GeomWkbCrsAccessor {
    WkbArray {
        wkb: ItemWkbAccessor,
        static_crs: Option<String>,
    },
    WkbScalar {
        wkb: Option<Vec<u8>>,
        static_crs: Option<String>,
    },
    ItemCrsArray {
        struct_array: StructArray,
        item: ItemWkbAccessor,
        crs: StringViewArray,
        item_static_crs: Option<String>,
    },
    ItemCrsScalar {
        struct_array: StructArray,
        item: ItemWkbAccessor,
        crs: StringViewArray,
        item_static_crs: Option<String>,
    },
    Null,
}

impl GeomWkbCrsAccessor {
    #[inline]
    fn get(&self, i: usize) -> Result<(Option<&[u8]>, Option<&str>)> {
        match self {
            Self::Null => Ok((None, None)),
            Self::WkbArray { wkb, static_crs } => {
                let maybe_wkb = wkb.get(i);
                if maybe_wkb.is_none() {
                    return Ok((None, None));
                }
                Ok((maybe_wkb, static_crs.as_deref()))
            }
            Self::WkbScalar { wkb, static_crs } => {
                if wkb.is_none() {
                    return Ok((None, None));
                }
                let _ = i;
                Ok((wkb.as_deref(), static_crs.as_deref()))
            }
            Self::ItemCrsArray {
                struct_array,
                item,
                crs,
                item_static_crs,
            } => {
                if struct_array.is_null(i) {
                    return Ok((None, None));
                }

                let maybe_wkb = item.get(i);
                if maybe_wkb.is_none() {
                    return Ok((None, None));
                }

                let item_crs_str = if crs.is_null(i) {
                    None
                } else {
                    Some(crs.value(i))
                };
                let static_crs_str = item_static_crs.as_deref();
                let crs_out = resolve_item_crs(item_crs_str, static_crs_str)?;
                Ok((maybe_wkb, crs_out))
            }
            Self::ItemCrsScalar {
                struct_array,
                item,
                crs,
                item_static_crs,
            } => {
                if struct_array.is_null(0) {
                    return Ok((None, None));
                }

                let maybe_wkb = item.get(0);
                if maybe_wkb.is_none() {
                    return Ok((None, None));
                }

                let item_crs_str = if crs.is_null(0) {
                    None
                } else {
                    Some(crs.value(0))
                };
                let static_crs_str = item_static_crs.as_deref();
                let crs_out = resolve_item_crs(item_crs_str, static_crs_str)?;
                let _ = i;
                Ok((maybe_wkb, crs_out))
            }
        }
    }
}

fn resolve_item_crs<'a>(
    item_crs: Option<&'a str>,
    static_crs: Option<&'a str>,
) -> Result<Option<&'a str>> {
    match (item_crs, static_crs) {
        (None, None) => Ok(None),
        (Some(item), None) => Ok(Some(item)),
        (None, Some(st)) => Ok(Some(st)),
        (Some(item), Some(st)) => {
            if item == st {
                return Ok(Some(item));
            }

            let item_crs = deserialize_crs(item)?;
            let static_crs = deserialize_crs(st)?;
            if item_crs == static_crs {
                Ok(Some(item))
            } else {
                exec_err!("CRS values not equal: {item_crs:?} vs {static_crs:?}")
            }
        }
    }
}

fn crs_string_from_sedona_type(sedona_type: &SedonaType) -> Result<Option<String>> {
    match sedona_type {
        SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => {
            if let Some(crs) = crs {
                Ok(Some(
                    crs.to_authority_code()?.unwrap_or_else(|| crs.to_json()),
                ))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

fn is_item_crs_type(sedona_type: &SedonaType) -> bool {
    matches!(
        sedona_type,
        SedonaType::Arrow(DataType::Struct(fields))
            if fields.len() == 2 && fields[0].name() == "item" && fields[1].name() == "crs"
    )
}

impl<'a, 'b> RasterExecutor<'a, 'b> {
    /// Create a new [RasterExecutor]
    pub fn new(arg_types: &'a [SedonaType], args: &'b [ColumnarValue]) -> Self {
        Self {
            arg_types,
            args,
            num_iterations: Self::calc_num_iterations(args),
        }
    }

    /// Create a new [RasterExecutor] with an explicit number of iterations.
    ///
    /// This is useful when the executor is built from a subset of the original
    /// arguments (e.g. only raster + geometry) but the overall UDF should still
    /// iterate according to other array arguments.
    #[allow(dead_code)]
    pub fn new_with_num_iterations(
        arg_types: &'a [SedonaType],
        args: &'b [ColumnarValue],
        num_iterations: usize,
    ) -> Self {
        let has_any_array = args.iter().any(|a| matches!(a, ColumnarValue::Array(_)));
        Self {
            arg_types,
            args,
            num_iterations: if has_any_array {
                Self::calc_num_iterations(args)
            } else {
                num_iterations
            },
        }
    }

    /// Return the number of iterations that will be performed
    pub fn num_iterations(&self) -> usize {
        self.num_iterations
    }

    /// Execute a function by iterating over rasters in the first argument
    ///
    /// This handles the common pattern of:
    /// 1. Downcasting array to StructArray
    /// 2. Creating raster array
    /// 3. Iterating with null checks
    /// 4. Calling the provided function with each raster
    pub fn execute_raster_void<F>(&self, mut func: F) -> Result<()>
    where
        F: FnMut(usize, Option<&RasterRefImpl<'_>>) -> Result<()>,
    {
        if self.arg_types[0] != RASTER {
            return sedona_internal_err!("First argument must be a raster type");
        }

        match &self.args[0] {
            ColumnarValue::Array(array) => {
                // Downcast to StructArray (rasters are stored as structs)
                let raster_struct =
                    array
                        .as_any()
                        .downcast_ref::<StructArray>()
                        .ok_or_else(|| {
                            sedona_internal_datafusion_err!("Expected StructArray for raster data")
                        })?;

                let raster_array = RasterStructArray::new(raster_struct);

                // Iterate through each raster in the array
                for i in 0..self.num_iterations {
                    if raster_array.is_null(i) {
                        func(i, None)?;
                        continue;
                    }
                    let raster = raster_array.get(i)?;
                    func(i, Some(&raster))?;
                }

                Ok(())
            }
            ColumnarValue::Scalar(scalar_value) => match scalar_value {
                ScalarValue::Struct(arc_struct) => {
                    let raster_array = RasterStructArray::new(arc_struct.as_ref());
                    let raster_opt = if raster_array.is_null(0) {
                        None
                    } else {
                        Some(raster_array.get(0)?)
                    };
                    for i in 0..self.num_iterations {
                        func(i, raster_opt.as_ref())?;
                    }
                    Ok(())
                }
                ScalarValue::Null => {
                    for i in 0..self.num_iterations {
                        func(i, None)?;
                    }
                    Ok(())
                }
                _ => sedona_internal_err!("Expected Struct scalar for raster"),
            },
        }
    }

    /// Execute a function by iterating over two raster arguments in sync.
    ///
    /// Supports Array/Array, Array/Scalar, Scalar/Array, and Scalar/Scalar.
    /// If both inputs are arrays, their lengths must match `num_iterations`.
    pub fn execute_raster_raster_void<F>(&self, mut func: F) -> Result<()>
    where
        F: FnMut(usize, Option<&RasterRefImpl<'_>>, Option<&RasterRefImpl<'_>>) -> Result<()>,
    {
        if self.arg_types.first() != Some(&RASTER) || self.arg_types.get(1) != Some(&RASTER) {
            return sedona_internal_err!("Expected (raster, raster) argument types");
        }
        if self.args.len() < 2 {
            return sedona_internal_err!("Expected at least 2 arguments (raster, raster)");
        }

        match (&self.args[0], &self.args[1]) {
            (ColumnarValue::Array(a0), ColumnarValue::Array(a1)) => {
                let s0 = a0.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                    DataFusionError::Internal("Expected StructArray for raster data".to_string())
                })?;
                let s1 = a1.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                    DataFusionError::Internal("Expected StructArray for raster data".to_string())
                })?;

                let arr0 = RasterStructArray::new(s0);
                let arr1 = RasterStructArray::new(s1);
                if arr0.len() != self.num_iterations || arr1.len() != self.num_iterations {
                    return sedona_internal_err!(
                        "Expected arrays of length {} but got ({}, {})",
                        self.num_iterations,
                        arr0.len(),
                        arr1.len()
                    );
                }

                for i in 0..self.num_iterations {
                    let r0 = if arr0.is_null(i) {
                        None
                    } else {
                        Some(arr0.get(i)?)
                    };
                    let r1 = if arr1.is_null(i) {
                        None
                    } else {
                        Some(arr1.get(i)?)
                    };
                    func(i, r0.as_ref(), r1.as_ref())?;
                }
                Ok(())
            }
            (ColumnarValue::Array(a0), ColumnarValue::Scalar(sv1)) => {
                let s0 = a0.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                    DataFusionError::Internal("Expected StructArray for raster data".to_string())
                })?;
                let arr0 = RasterStructArray::new(s0);
                if arr0.len() != self.num_iterations {
                    return sedona_internal_err!(
                        "Expected array of length {} but got {}",
                        self.num_iterations,
                        arr0.len()
                    );
                }
                let r1 = match sv1 {
                    ScalarValue::Struct(arc_struct) => {
                        let arr1 = RasterStructArray::new(arc_struct.as_ref());
                        if arr1.is_null(0) {
                            None
                        } else {
                            Some(arr1.get(0)?)
                        }
                    }
                    ScalarValue::Null => None,
                    _ => {
                        return Err(DataFusionError::Internal(
                            "Expected Struct scalar for raster".to_string(),
                        ))
                    }
                };

                for i in 0..self.num_iterations {
                    let r0 = if arr0.is_null(i) {
                        None
                    } else {
                        Some(arr0.get(i)?)
                    };
                    func(i, r0.as_ref(), r1.as_ref())?;
                }
                Ok(())
            }
            (ColumnarValue::Scalar(sv0), ColumnarValue::Array(a1)) => {
                let s1 = a1.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                    DataFusionError::Internal("Expected StructArray for raster data".to_string())
                })?;
                let arr1 = RasterStructArray::new(s1);
                if arr1.len() != self.num_iterations {
                    return sedona_internal_err!(
                        "Expected array of length {} but got {}",
                        self.num_iterations,
                        arr1.len()
                    );
                }
                let r0 = match sv0 {
                    ScalarValue::Struct(arc_struct) => {
                        let arr0 = RasterStructArray::new(arc_struct.as_ref());
                        if arr0.is_null(0) {
                            None
                        } else {
                            Some(arr0.get(0)?)
                        }
                    }
                    ScalarValue::Null => None,
                    _ => {
                        return Err(DataFusionError::Internal(
                            "Expected Struct scalar for raster".to_string(),
                        ))
                    }
                };

                for i in 0..self.num_iterations {
                    let r1 = if arr1.is_null(i) {
                        None
                    } else {
                        Some(arr1.get(i)?)
                    };
                    func(i, r0.as_ref(), r1.as_ref())?;
                }
                Ok(())
            }
            (ColumnarValue::Scalar(sv0), ColumnarValue::Scalar(sv1)) => {
                let r0 = match sv0 {
                    ScalarValue::Struct(arc_struct) => {
                        let arr0 = RasterStructArray::new(arc_struct.as_ref());
                        if arr0.is_null(0) {
                            None
                        } else {
                            Some(arr0.get(0)?)
                        }
                    }
                    ScalarValue::Null => None,
                    _ => {
                        return Err(DataFusionError::Internal(
                            "Expected Struct scalar for raster".to_string(),
                        ))
                    }
                };
                let r1 = match sv1 {
                    ScalarValue::Struct(arc_struct) => {
                        let arr1 = RasterStructArray::new(arc_struct.as_ref());
                        if arr1.is_null(0) {
                            None
                        } else {
                            Some(arr1.get(0)?)
                        }
                    }
                    ScalarValue::Null => None,
                    _ => {
                        return Err(DataFusionError::Internal(
                            "Expected Struct scalar for raster".to_string(),
                        ))
                    }
                };

                for i in 0..self.num_iterations {
                    func(i, r0.as_ref(), r1.as_ref())?;
                }
                Ok(())
            }
        }
    }

    /// Execute a function by iterating over rasters and a geometry in sync.
    ///
    /// The geometry argument may be:
    /// - A WKB Binary or BinaryView array/scalar (with type-level CRS via [SedonaType])
    /// - An item-level CRS struct column: struct(item: wkb, crs: utf8view)
    ///
    /// The closure is invoked for each row with `(raster, wkb_bytes, crs_str)`.
    pub fn execute_raster_wkb_crs_void<F>(&self, mut func: F) -> Result<()>
    where
        F: FnMut(Option<&RasterRefImpl<'_>>, Option<&[u8]>, Option<&str>) -> Result<()>,
    {
        if self.arg_types.first() != Some(&RASTER) {
            return sedona_internal_err!("First argument must be a raster type");
        }
        if self.args.len() < 2 {
            return sedona_internal_err!("Expected at least 2 arguments (raster, geom)");
        }

        let geom_accessor = self.make_geom_wkb_crs_accessor(1)?;

        match &self.args[0] {
            ColumnarValue::Array(array) => {
                let raster_struct =
                    array
                        .as_any()
                        .downcast_ref::<StructArray>()
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "Expected StructArray for raster data".to_string(),
                            )
                        })?;
                let raster_array = RasterStructArray::new(raster_struct);

                for i in 0..self.num_iterations {
                    let (maybe_wkb, maybe_crs) = geom_accessor.get(i)?;
                    if raster_array.is_null(i) {
                        func(None, maybe_wkb, maybe_crs)?;
                        continue;
                    }
                    let raster = raster_array.get(i)?;
                    func(Some(&raster), maybe_wkb, maybe_crs)?;
                }

                Ok(())
            }
            ColumnarValue::Scalar(scalar_value) => match scalar_value {
                ScalarValue::Struct(arc_struct) => {
                    let raster_array = RasterStructArray::new(arc_struct.as_ref());
                    let raster_opt = if raster_array.is_null(0) {
                        None
                    } else {
                        Some(raster_array.get(0)?)
                    };
                    for i in 0..self.num_iterations {
                        let (maybe_wkb, maybe_crs) = geom_accessor.get(i)?;
                        func(raster_opt.as_ref(), maybe_wkb, maybe_crs)?;
                    }
                    Ok(())
                }
                ScalarValue::Null => {
                    for i in 0..self.num_iterations {
                        let (maybe_wkb, maybe_crs) = geom_accessor.get(i)?;
                        func(None, maybe_wkb, maybe_crs)?;
                    }
                    Ok(())
                }
                _ => Err(DataFusionError::Internal(
                    "Expected Struct scalar for raster".to_string(),
                )),
            },
        }
    }

    fn make_geom_wkb_crs_accessor(&self, arg_index: usize) -> Result<GeomWkbCrsAccessor> {
        let sedona_type = self
            .arg_types
            .get(arg_index)
            .ok_or_else(|| DataFusionError::Internal("Missing argument type".to_string()))?;
        let arg = self
            .args
            .get(arg_index)
            .ok_or_else(|| DataFusionError::Internal("Missing argument".to_string()))?;

        if is_item_crs_type(sedona_type) {
            let item_type = match sedona_type {
                SedonaType::Arrow(DataType::Struct(fields)) => {
                    SedonaType::from_storage_field(&fields[0])?
                }
                _ => return sedona_internal_err!("Unexpected item_crs type"),
            };
            let item_static_crs = crs_string_from_sedona_type(&item_type)?;

            match arg {
                ColumnarValue::Array(array) => {
                    let struct_array_ref = as_struct_array(array)?;
                    let item_col = struct_array_ref.column(0);
                    let crs_col = struct_array_ref.column(1);
                    let crs_array = as_string_view_array(crs_col)?.clone();
                    let item_accessor = match &item_type {
                        SedonaType::Wkb(_, _) => {
                            ItemWkbAccessor::Binary(as_binary_array(item_col)?.clone())
                        }
                        SedonaType::WkbView(_, _) => {
                            ItemWkbAccessor::BinaryView(as_binary_view_array(item_col)?.clone())
                        }
                        SedonaType::Arrow(DataType::Binary) => {
                            ItemWkbAccessor::Binary(as_binary_array(item_col)?.clone())
                        }
                        SedonaType::Arrow(DataType::BinaryView) => {
                            ItemWkbAccessor::BinaryView(as_binary_view_array(item_col)?.clone())
                        }
                        other => {
                            return sedona_internal_err!(
                                "Unsupported item_crs item field type: {other:?}"
                            );
                        }
                    };

                    Ok(GeomWkbCrsAccessor::ItemCrsArray {
                        struct_array: struct_array_ref.clone(),
                        item: item_accessor,
                        crs: crs_array,
                        item_static_crs,
                    })
                }
                ColumnarValue::Scalar(ScalarValue::Struct(struct_scalar)) => {
                    let struct_array_ref = struct_scalar.as_ref();
                    let item_col = struct_array_ref.column(0);
                    let crs_col = struct_array_ref.column(1);
                    let crs_array = as_string_view_array(crs_col)?.clone();
                    let item_accessor = match &item_type {
                        SedonaType::Wkb(_, _) => {
                            ItemWkbAccessor::Binary(as_binary_array(item_col)?.clone())
                        }
                        SedonaType::WkbView(_, _) => {
                            ItemWkbAccessor::BinaryView(as_binary_view_array(item_col)?.clone())
                        }
                        SedonaType::Arrow(DataType::Binary) => {
                            ItemWkbAccessor::Binary(as_binary_array(item_col)?.clone())
                        }
                        SedonaType::Arrow(DataType::BinaryView) => {
                            ItemWkbAccessor::BinaryView(as_binary_view_array(item_col)?.clone())
                        }
                        other => {
                            return sedona_internal_err!(
                                "Unsupported item_crs item field type: {other:?}"
                            );
                        }
                    };

                    Ok(GeomWkbCrsAccessor::ItemCrsScalar {
                        struct_array: struct_array_ref.clone(),
                        item: item_accessor,
                        crs: crs_array,
                        item_static_crs,
                    })
                }
                ColumnarValue::Scalar(ScalarValue::Null) => Ok(GeomWkbCrsAccessor::Null),
                other => {
                    sedona_internal_err!("Expected item_crs Struct scalar/array, got {other:?}")
                }
            }
        } else {
            let static_crs = crs_string_from_sedona_type(sedona_type)?;
            match arg {
                ColumnarValue::Array(array) => match sedona_type {
                    SedonaType::Wkb(_, _) | SedonaType::Arrow(DataType::Binary) => {
                        Ok(GeomWkbCrsAccessor::WkbArray {
                            wkb: ItemWkbAccessor::Binary(as_binary_array(array)?.clone()),
                            static_crs,
                        })
                    }
                    SedonaType::WkbView(_, _) | SedonaType::Arrow(DataType::BinaryView) => {
                        Ok(GeomWkbCrsAccessor::WkbArray {
                            wkb: ItemWkbAccessor::BinaryView(as_binary_view_array(array)?.clone()),
                            static_crs,
                        })
                    }
                    other => sedona_internal_err!("Unsupported geometry type: {other:?}"),
                },
                ColumnarValue::Scalar(scalar_value) => {
                    let maybe_bytes: Option<Vec<u8>> = match scalar_value {
                        ScalarValue::Binary(v)
                        | ScalarValue::BinaryView(v)
                        | ScalarValue::LargeBinary(v) => v.clone(),
                        ScalarValue::Null => None,
                        other => {
                            return sedona_internal_err!(
                                "Unsupported geometry scalar type: {other:?}"
                            )
                        }
                    };
                    Ok(GeomWkbCrsAccessor::WkbScalar {
                        wkb: maybe_bytes,
                        static_crs,
                    })
                }
            }
        }
    }

    /// Finish an [ArrayRef] output as the appropriate [ColumnarValue]
    ///
    /// Converts the output into a [ColumnarValue::Scalar] if all arguments were scalars,
    /// or a [ColumnarValue::Array] otherwise.
    pub fn finish(&self, out: ArrayRef) -> Result<ColumnarValue> {
        for arg in self.args {
            match arg {
                // If any argument was an array, we return an array
                ColumnarValue::Array(_) => {
                    return Ok(ColumnarValue::Array(out));
                }
                ColumnarValue::Scalar(_) => {}
            }
        }

        // All arguments are scalars, return a scalar
        Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(&out, 0)?))
    }

    /// Calculates the number of iterations that should happen based on the
    /// argument ColumnarValue types
    fn calc_num_iterations(args: &[ColumnarValue]) -> usize {
        for arg in args {
            match arg {
                // If any argument is an array, iterate array.len() times
                ColumnarValue::Array(array) => {
                    return array.len();
                }
                ColumnarValue::Scalar(_) => {}
            }
        }

        // All arguments are scalars, iterate once
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::UInt64Builder;
    use arrow_array::UInt64Array;
    use arrow_schema::Field;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::crs::{deserialize_crs, lnglat};
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::datatypes::{Edges, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::create::{create_array, create_array_item_crs};
    use sedona_testing::rasters::generate_test_rasters;
    use std::sync::Arc;

    #[test]
    fn test_raster_executor_execute_raster_void() {
        // 3 rasters, second one is null
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let args = [ColumnarValue::Array(Arc::new(rasters))];
        let arg_types = vec![RASTER];

        let executor = RasterExecutor::new(&arg_types, &args);
        assert_eq!(executor.num_iterations(), 3);

        let mut builder = UInt64Builder::with_capacity(executor.num_iterations());
        executor
            .execute_raster_void(|_i, raster_opt| {
                match raster_opt {
                    None => builder.append_null(),
                    Some(raster) => {
                        let width = raster.metadata().width();
                        builder.append_value(width);
                    }
                }
                Ok(())
            })
            .unwrap();

        let result = executor.finish(Arc::new(builder.finish())).unwrap();

        let width_array = match &result {
            ColumnarValue::Array(array) => array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("Expected UInt64Array"),
            ColumnarValue::Scalar(_) => panic!("Expected array, got scalar"),
        };

        assert_eq!(width_array.len(), 3);
        assert_eq!(width_array.value(0), 1);
        assert!(width_array.is_null(1));
        assert_eq!(width_array.value(2), 3);
    }

    #[test]
    fn test_raster_executor_scalar_input() {
        let rasters = generate_test_rasters(1, None).unwrap();
        let raster_struct = rasters.as_any().downcast_ref::<StructArray>().unwrap();
        let scalar_raster = ScalarValue::Struct(Arc::new(raster_struct.clone()));

        let args = [ColumnarValue::Scalar(scalar_raster)];
        let arg_types = vec![RASTER];

        let executor = RasterExecutor::new(&arg_types, &args);
        assert_eq!(executor.num_iterations(), 1);

        let mut builder = UInt64Builder::with_capacity(executor.num_iterations());
        executor
            .execute_raster_void(|_i, raster_opt| {
                match raster_opt {
                    None => builder.append_null(),
                    Some(raster) => {
                        let width = raster.metadata().width();
                        builder.append_value(width);
                    }
                }
                Ok(())
            })
            .unwrap();

        let result = executor.finish(Arc::new(builder.finish())).unwrap();

        // With scalar input, result should be a scalar
        let width_scalar = match &result {
            ColumnarValue::Scalar(scalar) => scalar,
            ColumnarValue::Array(_) => panic!("Expected scalar, got array"),
        };

        match width_scalar {
            ScalarValue::UInt64(Some(width)) => assert_eq!(*width, 1),
            _ => panic!("Expected UInt64 scalar"),
        }
    }

    #[test]
    fn test_raster_executor_null_scalar() {
        // Test with a null scalar
        let args = [ColumnarValue::Scalar(ScalarValue::Null)];
        let arg_types = vec![RASTER];

        let executor = RasterExecutor::new(&arg_types, &args);
        assert_eq!(executor.num_iterations(), 1);

        let mut builder = UInt64Builder::with_capacity(executor.num_iterations());
        executor
            .execute_raster_void(|_i, raster_opt| {
                match raster_opt {
                    None => builder.append_null(),
                    Some(raster) => {
                        let width = raster.metadata().width();
                        builder.append_value(width);
                    }
                }
                Ok(())
            })
            .unwrap();

        let result = executor.finish(Arc::new(builder.finish())).unwrap();

        // With null scalar input, result should be null scalar
        let width_scalar = match &result {
            ColumnarValue::Scalar(scalar) => scalar,
            ColumnarValue::Array(_) => panic!("Expected scalar, got array"),
        };

        assert_eq!(width_scalar, &ScalarValue::UInt64(None));
    }

    #[test]
    fn test_raster_executor_execute_raster_wkb_crs_void_static_crs() {
        let rasters = generate_test_rasters(2, None).unwrap();
        let raster_args = ColumnarValue::Array(Arc::new(rasters));

        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let geom_array = create_array(&[Some("POINT (0 0)"), None], &geom_type);
        let geom_args = ColumnarValue::Array(geom_array);

        let args = [raster_args, geom_args];
        let arg_types = vec![RASTER, geom_type];
        let executor = RasterExecutor::new(&arg_types, &args);

        let mut out_crs: Vec<Option<String>> = Vec::with_capacity(executor.num_iterations());
        let mut out_has_wkb: Vec<bool> = Vec::with_capacity(executor.num_iterations());
        executor
            .execute_raster_wkb_crs_void(|_raster, wkb, crs| {
                out_has_wkb.push(wkb.is_some());
                out_crs.push(crs.map(|s| s.to_string()));
                Ok(())
            })
            .unwrap();

        assert_eq!(out_has_wkb, vec![true, false]);
        assert!(out_crs[1].is_none());
        let parsed = deserialize_crs(out_crs[0].as_deref().unwrap()).unwrap();
        assert_eq!(parsed, deserialize_crs("EPSG:4326").unwrap());
    }

    #[test]
    fn test_raster_executor_execute_raster_wkb_crs_void_item_crs_row_level() {
        let rasters = generate_test_rasters(2, None).unwrap();
        let raster_args = ColumnarValue::Array(Arc::new(rasters));

        // Build a custom item_crs type (standard item type without static CRS)
        let item_field = WKB_GEOMETRY.to_storage_field("item", true).unwrap();
        let crs_field = Field::new("crs", DataType::Utf8View, true);
        let geom_type = SedonaType::Arrow(DataType::Struct(vec![item_field, crs_field].into()));

        let geom_array = create_array_item_crs(
            &[Some("POINT (0 0)"), Some("POINT (1 1)")],
            [Some("EPSG:4326"), None],
            &WKB_GEOMETRY,
        );
        let geom_args = ColumnarValue::Array(geom_array);

        let args = [raster_args, geom_args];
        let arg_types = vec![RASTER, geom_type];
        let executor = RasterExecutor::new(&arg_types, &args);

        let mut out_crs: Vec<Option<String>> = Vec::with_capacity(executor.num_iterations());
        executor
            .execute_raster_wkb_crs_void(|_raster, _wkb, crs| {
                out_crs.push(crs.map(|s| s.to_string()));
                Ok(())
            })
            .unwrap();

        assert_eq!(out_crs, vec![Some("EPSG:4326".to_string()), None]);
    }

    #[test]
    fn test_raster_executor_execute_raster_wkb_crs_void_item_crs_static_fallback() {
        let rasters = generate_test_rasters(1, None).unwrap();
        let raster_struct = rasters.as_any().downcast_ref::<StructArray>().unwrap();
        let scalar_raster = ScalarValue::Struct(Arc::new(raster_struct.clone()));

        let item_type_with_crs = SedonaType::Wkb(Edges::Planar, lnglat());
        let item_field = item_type_with_crs.to_storage_field("item", true).unwrap();
        let crs_field = Field::new("crs", DataType::Utf8View, true);
        let geom_type = SedonaType::Arrow(DataType::Struct(vec![item_field, crs_field].into()));

        let geom_array = create_array_item_crs(
            &[Some("POINT (0 0)"), Some("POINT (1 1)")],
            [None, None],
            &item_type_with_crs,
        );
        let geom_args = ColumnarValue::Array(geom_array);

        let args = [ColumnarValue::Scalar(scalar_raster), geom_args];
        let arg_types = vec![RASTER, geom_type];
        let executor = RasterExecutor::new(&arg_types, &args);
        assert_eq!(executor.num_iterations(), 2);

        let mut out_crs: Vec<Option<String>> = Vec::with_capacity(executor.num_iterations());
        executor
            .execute_raster_wkb_crs_void(|_raster, _wkb, crs| {
                out_crs.push(crs.map(|s| s.to_string()));
                Ok(())
            })
            .unwrap();

        let parsed0 = deserialize_crs(out_crs[0].as_deref().unwrap()).unwrap();
        let parsed1 = deserialize_crs(out_crs[1].as_deref().unwrap()).unwrap();
        assert_eq!(parsed0, deserialize_crs("EPSG:4326").unwrap());
        assert_eq!(parsed1, deserialize_crs("EPSG:4326").unwrap());
    }

    #[test]
    fn test_raster_executor_execute_raster_wkb_crs_void_item_crs_conflict_errors() {
        let rasters = generate_test_rasters(1, None).unwrap();
        let raster_args = ColumnarValue::Array(Arc::new(rasters));

        let item_type_with_crs = SedonaType::Wkb(Edges::Planar, lnglat());
        let item_field = item_type_with_crs.to_storage_field("item", true).unwrap();
        let crs_field = Field::new("crs", DataType::Utf8View, true);
        let geom_type = SedonaType::Arrow(DataType::Struct(vec![item_field, crs_field].into()));

        let geom_array = create_array_item_crs(
            &[Some("POINT (0 0)")],
            [Some("EPSG:3857")],
            &item_type_with_crs,
        );
        let geom_args = ColumnarValue::Array(geom_array);

        let args = [raster_args, geom_args];
        let arg_types = vec![RASTER, geom_type];
        let executor = RasterExecutor::new(&arg_types, &args);

        let err = executor
            .execute_raster_wkb_crs_void(|_raster, _wkb, _crs| Ok(()))
            .unwrap_err();
        assert!(err.message().contains("CRS values not equal"));
    }

    #[test]
    fn test_raster_executor_execute_raster_wkb_crs_void_wkb_view() {
        let rasters = generate_test_rasters(1, None).unwrap();
        let raster_args = ColumnarValue::Array(Arc::new(rasters));

        let geom_array = create_array(&[Some("POINT (0 0)")], &WKB_VIEW_GEOMETRY);
        let geom_args = ColumnarValue::Array(geom_array);

        let args = [raster_args, geom_args];
        let arg_types = vec![RASTER, WKB_VIEW_GEOMETRY];
        let executor = RasterExecutor::new(&arg_types, &args);

        executor
            .execute_raster_wkb_crs_void(|_raster, wkb, crs| {
                assert!(wkb.is_some());
                assert!(crs.is_none());
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn test_raster_executor_new_with_num_iterations_scalar_args() {
        let rasters = generate_test_rasters(1, None).unwrap();
        let raster_struct = rasters.as_any().downcast_ref::<StructArray>().unwrap();
        let scalar_raster = ScalarValue::Struct(Arc::new(raster_struct.clone()));
        let args = [ColumnarValue::Scalar(scalar_raster)];
        let arg_types = vec![RASTER];

        let executor = RasterExecutor::new_with_num_iterations(&arg_types, &args, 10);
        assert_eq!(executor.num_iterations(), 10);
    }
}
