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
use std::iter::zip;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::cast::{as_binary_array, as_binary_view_array, as_struct_array};
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::SedonaType;
use wkb::reader::Wkb;

/// Helper for writing general kernel implementations with geometry
///
/// The [GenericExecutor] wraps a set of arguments and their types and provides helpers
/// to make writing general compute functions less verbose. Broadly, kernel implementations
/// must consider multiple input data types (e.g., Wkb/WkbView or Float32/Float64) and
/// multiple combinations of Array or ScalarValue inputs. This executor is generic on
/// a [GeometryFactory] to support iterating over geometries from multiple libraries;
/// however, the most commonly used version is the [WkbExecutor].
///
/// The pattern supported by the [GenericExecutor] is:
///
/// - Create a [GenericExecutor] with `new()`
/// - Create an Arrow builder of the appropriate output type using
///   `with_capacity(executor.num_iterations()`
/// - Use `execute_wkb()` with a lambda whose contents appends to the builder
/// - Use `finish()` to build the output [ColumnarValue].
///
/// When all arguments are scalars, `execute_wkb()` will perform one iteration
/// and `finish()` will return a [ColumnarValue::Scalar]. Otherwise, the output will
/// be a [ColumnarValue::Array] built from `num_iterations()` iterations. This is true
/// even if a geometry scalar is passed with a non-geometry argument. Non-geometry
/// arrays are typically cheaper to cast to a concrete type (e.g., cast a numeric
/// to float64 or an integer to int64) and cheaper to access by element or iterator
/// compared to most geometry operations that require them.
///
/// The [GenericExecutor] is not built to be completely general and UDF implementers are
/// free to use other mechanisms to implement UDFs with geometry arguments. The balance
/// between optimizing iteration speed, minimizing dispatch overhead, and maximizing
/// readability of kernel implementations is difficult to achieve and future utilities
/// may provide alternatives optimized for a different balance than was chosen here.
///
/// This executor accepts two factories (supporting kernels accepting 0, 1, or 2
/// geometry arguments), which can also support implementations that wish to "prepare"
/// one side or the other (e.g., for binary predicates).
///
/// A critical optimization is iterating over two arguments where one argument is a
/// scalar. In this case, [GeometryFactory::try_from_wkb] is called exactly once on the
/// scalar side (e.g., whatever parsing needs to occur for the scalar only occurs once).
pub struct GenericExecutor<'a, 'b, Factory0, Factory1> {
    pub arg_types: &'a [SedonaType],
    pub args: &'b [ColumnarValue],
    num_iterations: usize,
    factory0: Factory0,
    factory1: Factory1,
}

/// Alias for an executor that iterates over geometries as [Wkb]
pub type WkbExecutor<'a, 'b> = GenericExecutor<'a, 'b, WkbGeometryFactory, WkbGeometryFactory>;

impl<'a, 'b, Factory0: GeometryFactory, Factory1: GeometryFactory>
    GenericExecutor<'a, 'b, Factory0, Factory1>
{
    /// Create a new [GenericExecutor]
    pub fn new(arg_types: &'a [SedonaType], args: &'b [ColumnarValue]) -> Self {
        Self {
            arg_types,
            args,
            num_iterations: Self::calc_num_iterations(args),
            factory0: Factory0::default(),
            factory1: Factory1::default(),
        }
    }

    /// Return the number of iterations that will be performed `execute_*()` methods
    ///
    /// If all arguments are [ColumnarValue::Scalar]s, this will be one iteration.
    /// Otherwise, it will be the length of the first array.
    pub fn num_iterations(&self) -> usize {
        self.num_iterations
    }

    /// Execute a function by iterating over [Wkb] scalars in the first argument
    ///
    /// Provides a mechanism to iterate over a geometry array by converting each to
    /// a [Wkb] scalar. For [SedonaType::Wkb] and [SedonaType::WkbView] arrays, this
    /// is the conversion that would normally happen. For other future supported geometry
    /// array types, this may incur a cast of item-wise conversion overhead.
    pub fn execute_wkb_void<F: FnMut(Option<Factory0::Geom<'b>>) -> Result<()>>(
        &self,
        mut func: F,
    ) -> Result<()> {
        let factory = Factory0::default();
        match &self.args[0] {
            ColumnarValue::Array(array) => {
                array.iter_with_factory(&factory, &self.arg_types[0], self.num_iterations, func)
            }
            ColumnarValue::Scalar(scalar_value) => {
                let wkb0 = scalar_value.scalar_from_factory(&self.factory0)?;
                func(wkb0)
            }
        }
    }

    /// Execute a binary geometry function by iterating over [Wkb] scalars in the
    /// first two arguments
    ///
    /// Provides a mechanism to iterate over two geometry arrays as pairs of [Wkb]
    /// scalars. [SedonaType::Wkb] and [SedonaType::WkbView] arrays are iterated over
    /// in place; however, future supported geometry array types may incur conversion
    /// overhead.
    pub fn execute_wkb_wkb_void<
        F: FnMut(Option<&Factory0::Geom<'b>>, Option<&Factory1::Geom<'b>>) -> Result<()>,
    >(
        &self,
        mut func: F,
    ) -> Result<()> {
        match (&self.args[0], &self.args[1]) {
            (ColumnarValue::Array(array0), ColumnarValue::Array(array1)) => iter_wkb_wkb_array(
                &self.factory0,
                &self.factory1,
                (&self.arg_types[0], &self.arg_types[1]),
                (array0, array1),
                func,
            ),
            (ColumnarValue::Array(array), ColumnarValue::Scalar(scalar_value)) => {
                let wkb1 = scalar_value.scalar_from_factory(&self.factory1)?;
                array.iter_with_factory(
                    &self.factory0,
                    &self.arg_types[0],
                    self.num_iterations(),
                    |wkb0| func(wkb0.as_ref(), wkb1.as_ref()),
                )
            }
            (ColumnarValue::Scalar(scalar_value), ColumnarValue::Array(array)) => {
                let wkb0 = scalar_value.scalar_from_factory(&self.factory0)?;
                array.iter_with_factory(
                    &self.factory1,
                    &self.arg_types[1],
                    self.num_iterations(),
                    |wkb1| func(wkb0.as_ref(), wkb1.as_ref()),
                )
            }
            (ColumnarValue::Scalar(scalar_value0), ColumnarValue::Scalar(scalar_value1)) => {
                let wkb0 = scalar_value0.scalar_from_factory(&self.factory0)?;
                let wkb1 = scalar_value1.scalar_from_factory(&self.factory1)?;
                func(wkb0.as_ref(), wkb1.as_ref())
            }
        }
    }

    /// Finish an [ArrayRef] output as the appropriate [ColumnarValue]
    ///
    /// Converts the output of `finish()`ing an Arrow builder into a
    /// [ColumnarValue::Scalar] if all arguments were scalars, or a
    /// [ColumnarValue::Array] otherwise.
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

        // For all scalar arguments, we return a scalar
        Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(&out, 0)?))
    }

    /// Calculates the number of iterations that should happen based on the
    /// argument ColumnarValue types
    fn calc_num_iterations(args: &[ColumnarValue]) -> usize {
        for arg in args {
            match arg {
                // If any argument is an array, we have to iterate array.len() times
                ColumnarValue::Array(array) => {
                    return array.len();
                }
                ColumnarValue::Scalar(_) => {}
            }
        }

        // For all scalar arguments, we iterate once
        1
    }
}

/// Factory object to help the [GenericExecutor] iterate over
/// various concrete geometry objects
pub trait GeometryFactory: Default {
    /// The concrete geometry type (e.g., [Wkb])
    ///
    /// Usually this is some type whose conversion from raw WKB bytes incurs some cost.
    /// The lifetime ensure that types that are a "view" of their input [ArrayRef] or
    /// [ScalarValue] can be used here; however, this type may also own its own data.
    type Geom<'a>;

    /// Parse bytes of WKB or EWKB into [GeometryFactory::Geom]
    fn try_from_wkb<'a>(&self, wkb_bytes: &'a [u8]) -> Result<Self::Geom<'a>>;

    /// Helper that calls [GeometryFactory::try_from_wkb] on an
    /// `Option<>`.
    fn try_from_maybe_wkb<'a>(
        &self,
        maybe_wkb_bytes: Option<&'a [u8]>,
    ) -> Result<Option<Self::Geom<'a>>> {
        match maybe_wkb_bytes {
            Some(wkb_bytes) => Ok(Some(self.try_from_wkb(wkb_bytes)?)),
            None => Ok(None),
        }
    }
}

/// A [GeometryFactory] whose geometry type is [Wkb]
///
/// Using this geometry factory iterates over items as references to [Wkb]
/// objects (which are the fastest objects when iterating over WKB input
/// that implement geo-traits).
#[derive(Default)]
pub struct WkbGeometryFactory {}

impl GeometryFactory for WkbGeometryFactory {
    type Geom<'a> = Wkb<'a>;

    fn try_from_wkb<'a>(&self, wkb_bytes: &'a [u8]) -> Result<Self::Geom<'a>> {
        wkb::reader::read_wkb(wkb_bytes).map_err(|e| DataFusionError::External(Box::new(e)))
    }
}

/// A [GeometryFactory] whose geometry type are raw WKB bytes
///
/// Using this geometry factory iterates over items as references to the raw underlying
/// bytes, which is useful for writing optimized kernels that do not need the full buffer to
/// be validated and/or parsed.
#[derive(Default)]
pub struct WkbBytesFactory {}

impl GeometryFactory for WkbBytesFactory {
    type Geom<'a> = &'a [u8];

    fn try_from_wkb<'a>(&self, wkb_bytes: &'a [u8]) -> Result<Self::Geom<'a>> {
        Ok(wkb_bytes)
    }
}

/// Alias for an executor that iterates over geometries in their raw [Wkb] bytes.
///
/// This [GenericExecutor] implementation provides more optimization opportunities,
/// but it requires additional manual processing of the raw [Wkb] bytes compared to
/// the [WkbExecutor].
pub type WkbBytesExecutor<'a, 'b> = GenericExecutor<'a, 'b, WkbBytesFactory, WkbBytesFactory>;

/// Trait for iterating over a container type as geometry scalars
///
/// Currently the only scalar type supported is [Wkb]; however, for future
/// geometry array types it may make sense to offer other scalar types over
/// which to iterate.
pub trait IterGeo {
    fn iter_as_wkb_bytes<'a, F: FnMut(Option<&'a [u8]>) -> Result<()>>(
        &'a self,
        sedona_type: &SedonaType,
        num_iterations: usize,
        func: F,
    ) -> Result<()>;

    fn iter_with_factory<
        'a,
        Factory: GeometryFactory,
        F: FnMut(Option<Factory::Geom<'a>>) -> Result<()>,
    >(
        &'a self,
        factory: &Factory,
        sedona_type: &SedonaType,
        num_iterations: usize,
        mut func: F,
    ) -> Result<()> {
        self.iter_as_wkb_bytes(
            sedona_type,
            num_iterations,
            |maybe_bytes| match maybe_bytes {
                Some(wkb_bytes) => {
                    let geom = factory.try_from_wkb(wkb_bytes)?;
                    func(Some(geom))
                }
                None => func(None),
            },
        )
    }

    /// Apply a function for each element of self as an optional [Wkb]
    ///
    /// The function will always be called num_iteration types to support
    /// efficient iteration over scalar containers (e.g., so that implementations
    /// can parse the Wkb once and reuse the object).
    fn iter_as_wkb<'a, F: FnMut(Option<Wkb<'a>>) -> Result<()>>(
        &'a self,
        sedona_type: &SedonaType,
        num_iterations: usize,
        func: F,
    ) -> Result<()> {
        let factory = WkbGeometryFactory {};
        self.iter_with_factory(&factory, sedona_type, num_iterations, func)
    }
}

/// Trait for obtaining geometry scalars from containers
///
/// Currently this is internal and only implemented for the [ScalarValue].
trait ScalarGeo {
    fn scalar_as_wkb_bytes(&self) -> Result<Option<&[u8]>>;

    fn scalar_from_factory<T: GeometryFactory>(&self, factory: &T) -> Result<Option<T::Geom<'_>>> {
        factory.try_from_maybe_wkb(self.scalar_as_wkb_bytes()?)
    }
}

impl IterGeo for ArrayRef {
    fn iter_as_wkb_bytes<'a, F: FnMut(Option<&'a [u8]>) -> Result<()>>(
        &'a self,
        sedona_type: &SedonaType,
        num_iterations: usize,
        mut func: F,
    ) -> Result<()> {
        if num_iterations != self.len() {
            return sedona_internal_err!(
                "Expected {num_iterations} items but got Array with {} items",
                self.len()
            );
        }

        match sedona_type {
            SedonaType::Arrow(DataType::Null) => {
                for _ in 0..num_iterations {
                    func(None)?;
                }

                Ok(())
            }
            SedonaType::Wkb(_, _) => iter_wkb_binary(as_binary_array(self)?, func),
            SedonaType::WkbView(_, _) => iter_wkb_binary(as_binary_view_array(self)?, func),
            SedonaType::Arrow(DataType::Struct(fields))
                if fields.iter().map(|f| f.name()).collect::<Vec<_>>() == vec!["item", "crs"] =>
            {
                let struct_array = as_struct_array(self)?;
                struct_array
                    .column(0)
                    .iter_as_wkb_bytes(sedona_type, num_iterations, func)
            }
            _ => {
                // We could cast here as a fallback, iterate and cast per-element, or
                // implement iter_as_something_else()/supports_iter_xxx() when more geo array types
                // are supported.
                sedona_internal_err!("Can't iterate over {:?} as Wkb", sedona_type)
            }
        }
    }
}

impl ScalarGeo for ScalarValue {
    fn scalar_as_wkb_bytes(&self) -> Result<Option<&[u8]>> {
        match self {
            ScalarValue::Binary(maybe_item)
            | ScalarValue::BinaryView(maybe_item)
            | ScalarValue::LargeBinary(maybe_item) => Ok(maybe_item.as_deref()),
            ScalarValue::Null => Ok(None),
            _ => sedona_internal_err!("Can't iterate over {:?} ScalarValue as &[u8]", self),
        }
    }
}

/// Helper to dispatch binary iteration over two arrays. The Scalar/Array,
/// Array/Scalar, and Scalar/Scalar case are handled using the unary iteration
/// infrastructure.
fn iter_wkb_wkb_array<
    'a,
    Factory0: GeometryFactory,
    Factory1: GeometryFactory,
    F: FnMut(Option<&Factory0::Geom<'a>>, Option<&Factory1::Geom<'a>>) -> Result<()>,
>(
    factory0: &Factory0,
    factory1: &Factory1,
    types: (&SedonaType, &SedonaType),
    arrays: (&'a ArrayRef, &'a ArrayRef),
    func: F,
) -> Result<()> {
    let (array0, array1) = arrays;
    match types {
        (SedonaType::Wkb(_, _), SedonaType::Wkb(_, _)) => iter_wkb_wkb_binary(
            factory0,
            factory1,
            as_binary_array(array0)?,
            as_binary_array(array1)?,
            func,
        ),
        (SedonaType::Wkb(_, _), SedonaType::WkbView(_, _)) => iter_wkb_wkb_binary(
            factory0,
            factory1,
            as_binary_array(array0)?,
            as_binary_view_array(array1)?,
            func,
        ),
        (SedonaType::WkbView(_, _), SedonaType::Wkb(_, _)) => iter_wkb_wkb_binary(
            factory0,
            factory1,
            as_binary_view_array(array0)?,
            as_binary_array(array1)?,
            func,
        ),
        (SedonaType::WkbView(_, _), SedonaType::WkbView(_, _)) => iter_wkb_wkb_binary(
            factory0,
            factory1,
            as_binary_view_array(array0)?,
            as_binary_view_array(array1)?,
            func,
        ),
        _ => {
            // We could do casting of one or both sides to support other cases as they
            // arise to manage the complexity/performance balance
            sedona_internal_err!(
                "Can't iterate over {:?} and {:?} arrays as a pair of Wkb scalars",
                types.0,
                types.1
            )
        }
    }
}

/// Generic function to iterate over a pair of optional bytes providers
/// (e.g., various concrete array types)
fn iter_wkb_wkb_binary<
    'a,
    Factory0: GeometryFactory,
    Factory1: GeometryFactory,
    T0: IntoIterator<Item = Option<&'a [u8]>>,
    T1: IntoIterator<Item = Option<&'a [u8]>>,
    F: FnMut(Option<&Factory0::Geom<'a>>, Option<&Factory1::Geom<'a>>) -> Result<()>,
>(
    factory0: &Factory0,
    factory1: &Factory1,
    iterable0: T0,
    iterable1: T1,
    mut func: F,
) -> Result<()> {
    for (item0, item1) in zip(iterable0, iterable1) {
        let wkb0 = factory0.try_from_maybe_wkb(item0)?;
        let wkb1 = factory1.try_from_maybe_wkb(item1)?;
        func(wkb0.as_ref(), wkb1.as_ref())?;
    }

    Ok(())
}

/// Generic function to iterate over a single provider of optional wkb byte slices
/// (e.g., concrete array types)
fn iter_wkb_binary<
    'a,
    T: IntoIterator<Item = Option<&'a [u8]>>,
    F: FnMut(Option<&'a [u8]>) -> Result<()>,
>(
    iterable: T,
    mut func: F,
) -> Result<()> {
    for item in iterable.into_iter() {
        func(item)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;
    use std::sync::Arc;

    use super::*;
    use arrow_array::{builder::BinaryBuilder, create_array};
    use arrow_schema::DataType;
    use datafusion_common::{cast::as_binary_view_array, scalar::ScalarValue};
    use datafusion_expr::ColumnarValue;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};

    const POINT: [u8; 21] = [
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];

    fn write_to_test_output(
        wkt_out: &mut String,
        i: usize,
        maybe_geom: Option<&Wkb>,
    ) -> Result<()> {
        write!(wkt_out, " {i}: ").unwrap();
        match maybe_geom {
            Some(geom) => wkt::to_wkt::write_geometry(wkt_out, &geom).unwrap(),
            None => write!(wkt_out, "None").unwrap(),
        };

        Ok(())
    }

    #[test]
    fn wkb_array() {
        let mut builder = BinaryBuilder::new();
        builder.append_value(POINT);
        builder.append_null();
        builder.append_value(POINT);
        let wkb_array = builder.finish();
        let wkb_array_ref: ArrayRef = Arc::new(wkb_array.clone());

        let mut out = Vec::new();
        iter_wkb_binary(&wkb_array, |x| {
            out.push(x.is_some());
            Ok(())
        })
        .unwrap();
        assert_eq!(out, vec![true, false, true]);

        let mut wkt_out = String::new();
        let mut i = 0;
        wkb_array_ref
            .iter_as_wkb(&WKB_GEOMETRY, 3, |maybe_geom| {
                write_to_test_output(&mut wkt_out, i, maybe_geom.as_ref()).unwrap();
                i += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(wkt_out, " 0: POINT(1 2) 1: None 2: POINT(1 2)");

        let wkb_view_array_ref = ColumnarValue::Array(wkb_array_ref)
            .cast_to(&DataType::BinaryView, None)
            .unwrap()
            .to_array(3)
            .unwrap();
        let wkb_view_array = as_binary_view_array(&wkb_view_array_ref).unwrap();

        let mut out = Vec::new();
        iter_wkb_binary(wkb_view_array, |x| {
            out.push(x.is_some());
            Ok(())
        })
        .unwrap();
        assert_eq!(out, vec![true, false, true]);

        let mut wkt_out = String::new();
        let mut i = 0;
        wkb_view_array_ref
            .iter_as_wkb(&WKB_VIEW_GEOMETRY, 3, |maybe_geom| {
                write_to_test_output(&mut wkt_out, i, maybe_geom.as_ref()).unwrap();
                i += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(wkt_out, " 0: POINT(1 2) 1: None 2: POINT(1 2)");
    }

    #[rstest]
    fn wkb_wkb_types(
        #[values(
            (WKB_GEOMETRY, WKB_GEOMETRY),
            (WKB_GEOMETRY, WKB_VIEW_GEOMETRY),
            (WKB_VIEW_GEOMETRY, WKB_GEOMETRY),
            (WKB_VIEW_GEOMETRY, WKB_VIEW_GEOMETRY))
        ]
        types: (SedonaType, SedonaType),
    ) {
        let (left_type, right_type) = types;

        let mut builder = BinaryBuilder::new();
        builder.append_value(POINT);
        builder.append_null();
        builder.append_value(POINT);
        builder.append_null();
        let binary_array0 = builder.finish();

        let mut builder = BinaryBuilder::new();
        builder.append_value(POINT);
        builder.append_value(POINT);
        builder.append_null();
        builder.append_null();
        let binary_array1 = builder.finish();

        let value0 = ColumnarValue::Array(Arc::new(binary_array0.clone()))
            .cast_to(left_type.storage_type(), None)
            .unwrap();

        let value1 = ColumnarValue::Array(Arc::new(binary_array1.clone()))
            .cast_to(right_type.storage_type(), None)
            .unwrap();

        let arg_types = [left_type, right_type];
        let args = [value0, value1];
        let executor = WkbExecutor::new(&arg_types, &args);

        let mut wkt_out = String::new();
        let mut i = 0;
        executor
            .execute_wkb_wkb_void(|maybe_geom0, maybe_geom1| {
                write_to_test_output(&mut wkt_out, i, maybe_geom0)?;
                write_to_test_output(&mut wkt_out, i, maybe_geom1)?;
                i += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(
            wkt_out,
            " 0: POINT(1 2) 0: POINT(1 2) 1: None 1: POINT(1 2) 2: POINT(1 2) 2: None 3: None 3: None"
        );
    }

    #[test]
    fn wkb_wkb_scalar_array() {
        let mut builder = BinaryBuilder::new();
        builder.append_value(POINT);
        builder.append_null();
        let wkb_array = builder.finish();
        let wkb_array_value = ColumnarValue::Array(Arc::new(wkb_array));
        let wkb_scalar_value = ColumnarValue::Scalar(ScalarValue::Binary(Some(POINT.to_vec())));
        let wkb_scalar_null_value = ColumnarValue::Scalar(ScalarValue::Binary(None));

        let arg_types = [WKB_GEOMETRY, WKB_GEOMETRY];
        let args = [wkb_array_value.clone(), wkb_scalar_value.clone()];
        let executor = WkbExecutor::new(&arg_types, &args);

        let mut wkt_out = String::new();
        let mut i = 0;
        executor
            .execute_wkb_wkb_void(|maybe_geom0, maybe_geom1| {
                write_to_test_output(&mut wkt_out, i, maybe_geom0)?;
                write_to_test_output(&mut wkt_out, i, maybe_geom1)?;
                i += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(
            wkt_out,
            " 0: POINT(1 2) 0: POINT(1 2) 1: None 1: POINT(1 2)"
        );

        let args = [wkb_array_value.clone(), wkb_scalar_value.clone()];
        let executor = WkbExecutor::new(&arg_types, &args);

        let mut wkt_out = String::new();
        let mut i = 0;
        executor
            .execute_wkb_wkb_void(|maybe_geom0, maybe_geom1| {
                write_to_test_output(&mut wkt_out, i, maybe_geom0)?;
                write_to_test_output(&mut wkt_out, i, maybe_geom1)?;
                i += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(
            wkt_out,
            " 0: POINT(1 2) 0: POINT(1 2) 1: None 1: POINT(1 2)"
        );

        let args = [wkb_array_value.clone(), wkb_array_value.clone()];
        let executor = WkbExecutor::new(&arg_types, &args);

        let mut wkt_out = String::new();
        let mut i = 0;
        executor
            .execute_wkb_wkb_void(|maybe_geom0, maybe_geom1| {
                write_to_test_output(&mut wkt_out, i, maybe_geom0)?;
                write_to_test_output(&mut wkt_out, i, maybe_geom1)?;
                i += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(wkt_out, " 0: POINT(1 2) 0: POINT(1 2) 1: None 1: None");

        let args = [wkb_scalar_null_value.clone(), wkb_scalar_value.clone()];
        let executor = WkbExecutor::new(&arg_types, &args);

        let mut wkt_out = String::new();
        let mut i = 0;
        executor
            .execute_wkb_wkb_void(|maybe_geom0, maybe_geom1| {
                write_to_test_output(&mut wkt_out, i, maybe_geom0)?;
                write_to_test_output(&mut wkt_out, i, maybe_geom1)?;
                i += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(wkt_out, " 0: None 0: POINT(1 2)");
    }

    #[test]
    fn wkb_array_errors() {
        let not_geometry_array: ArrayRef = create_array!(Int32, [0, 1, 2]);
        let err = not_geometry_array
            .iter_as_wkb(&SedonaType::Arrow(DataType::Int32), 3, |_| unreachable!())
            .unwrap_err();
        assert!(err.message().contains("Can't iterate over"));

        let err = not_geometry_array
            .iter_as_wkb(
                &SedonaType::Arrow(DataType::Int32),
                1000000,
                |_| unreachable!(),
            )
            .unwrap_err();
        assert!(err
            .message()
            .contains("Expected 1000000 items but got Array with 3 items"));

        let factory0 = WkbGeometryFactory::default();
        let factory1 = WkbGeometryFactory::default();
        let err = iter_wkb_wkb_array(
            &factory0,
            &factory1,
            (
                &SedonaType::Arrow(DataType::Int32),
                &SedonaType::Arrow(DataType::Int32),
            ),
            (&not_geometry_array, &not_geometry_array),
            |_, _| unreachable!(),
        )
        .unwrap_err();
        assert!(err.message().contains(
            "Can't iterate over Arrow(Int32) and Arrow(Int32) arrays as a pair of Wkb scalars"
        ));
    }

    #[test]
    fn wkb_scalar() {
        let factory = WkbGeometryFactory::default();
        assert!(ScalarValue::Binary(None)
            .scalar_from_factory(&factory)
            .unwrap()
            .is_none());

        let mut wkt_out = String::new();
        let binary_scalar = ScalarValue::Binary(Some(POINT.to_vec()));

        let wkb_item = binary_scalar
            .scalar_from_factory(&factory)
            .unwrap()
            .unwrap();
        wkt::to_wkt::write_geometry(&mut wkt_out, &wkb_item).unwrap();
        assert_eq!(wkt_out, "POINT(1 2)");
        drop(wkb_item);

        let mut wkt_out = String::new();
        let binary_scalar = ScalarValue::BinaryView(Some(POINT.to_vec()));

        let wkb_item = binary_scalar
            .scalar_from_factory(&factory)
            .unwrap()
            .unwrap();
        wkt::to_wkt::write_geometry(&mut wkt_out, &wkb_item).unwrap();
        assert_eq!(wkt_out, "POINT(1 2)");
        drop(wkb_item);

        let null_item = ScalarValue::Null.scalar_from_factory(&factory).unwrap();
        assert!(null_item.is_none());

        let err = ScalarValue::Binary(Some(vec![]))
            .scalar_from_factory(&factory)
            .unwrap_err();
        assert_eq!(err.message(), "failed to fill whole buffer");

        let err = ScalarValue::Date32(Some(1))
            .scalar_from_factory(&factory)
            .unwrap_err();
        assert!(err.message().contains("Can't iterate over"));
    }
}
