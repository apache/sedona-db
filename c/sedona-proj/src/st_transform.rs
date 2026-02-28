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
use crate::transform::{ProjCrsEngine, ProjCrsEngineBuilder};
use arrow_array::builder::{BinaryBuilder, StringViewBuilder};
use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::cast::{as_string_view_array, as_struct_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{exec_err, DataFusionError, Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use sedona_expr::item_crs::make_item_crs;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_geometry::transform::{transform, CachingCrsEngine, CrsEngine, CrsTransform};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::crs::{deserialize_crs, Crs};
use sedona_schema::datatypes::{Edges, SedonaType, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS};
use sedona_schema::matchers::ArgMatcher;
use std::cell::OnceCell;
use std::io::Write;
use std::iter::zip;
use std::sync::{Arc, RwLock};
use wkb::reader::Wkb;

/// ST_Transform() implementation using the proj crate
pub fn st_transform_impl() -> ScalarKernelRef {
    Arc::new(STTransform {})
}

#[derive(Debug)]
struct STTransform {}

impl SedonaScalarKernel for STTransform {
    fn return_type_from_args_and_scalars(
        &self,
        arg_types: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        let inputs = zip(arg_types, scalar_args)
            .map(|(arg_type, arg_scalar)| ArgInput::from_return_type_arg(arg_type, *arg_scalar))
            .collect::<Vec<_>>();

        if inputs.len() == 2 {
            match (inputs[0], inputs[1]) {
                // ScalarCrs output always returns a Wkb output type with concrete Crs
                (ArgInput::Geo(_), ArgInput::ScalarCrs(scalar_value))
                | (ArgInput::ItemCrs, ArgInput::ScalarCrs(scalar_value)) => {
                    Ok(Some(output_type_from_scalar_crs_value(scalar_value)?))
                }

                // Geo or ItemCrs with ArrayCrs output always return ItemCrs output
                (ArgInput::Geo(_), ArgInput::ArrayCrs)
                | (ArgInput::ItemCrs, ArgInput::ArrayCrs) => {
                    Ok(Some(WKB_GEOMETRY_ITEM_CRS.clone()))
                }
                _ => Ok(None),
            }
        } else if inputs.len() == 3 {
            match (inputs[0], inputs[1], inputs[2]) {
                // ScalarCrs output always returns a Wkb output type with concrete Crs
                (ArgInput::Geo(_), ArgInput::ScalarCrs(_), ArgInput::ScalarCrs(scalar_value))
                | (ArgInput::Geo(_), ArgInput::ArrayCrs, ArgInput::ScalarCrs(scalar_value))
                | (ArgInput::ItemCrs, ArgInput::ScalarCrs(_), ArgInput::ScalarCrs(scalar_value))
                | (ArgInput::ItemCrs, ArgInput::ArrayCrs, ArgInput::ScalarCrs(scalar_value)) => {
                    Ok(Some(output_type_from_scalar_crs_value(scalar_value)?))
                }

                // Geo or ItemCrs with ArrayCrs output always return ItemCrs output
                (ArgInput::Geo(_), ArgInput::ScalarCrs(_), ArgInput::ArrayCrs)
                | (ArgInput::Geo(_), ArgInput::ArrayCrs, ArgInput::ArrayCrs)
                | (ArgInput::ItemCrs, ArgInput::ScalarCrs(_), ArgInput::ArrayCrs)
                | (ArgInput::ItemCrs, ArgInput::ArrayCrs, ArgInput::ArrayCrs) => {
                    Ok(Some(WKB_GEOMETRY_ITEM_CRS.clone()))
                }
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let inputs = zip(arg_types, args)
            .map(|(arg_type, arg)| ArgInput::from_arg(arg_type, arg))
            .collect::<Vec<_>>();

        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        // Optimize the easy case, where we have exactly one transformation and there are no
        // null or missing CRSes to contend with.
        let from_index = inputs.len() - 2;
        let to_index = inputs.len() - 1;
        let (from, to) = (inputs[from_index], inputs[to_index]);
        if let (Some(from_constant), Some(to_constant)) = (from.crs_constant()?, to.crs_constant()?)
        {
            let maybe_from_crs = deserialize_crs(&from_constant)?;
            let maybe_to_crs = deserialize_crs(&to_constant)?;
            if let (Some(from_crs), Some(to_crs)) = (maybe_from_crs, maybe_to_crs) {
                with_global_proj_engine(|engine| {
                    let crs_transform = engine
                        .get_transform_crs_to_crs(
                            &from_crs.to_crs_string(),
                            &to_crs.to_crs_string(),
                            None,
                            "",
                        )
                        .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
                    executor.execute_wkb_void(|maybe_wkb| {
                        match maybe_wkb {
                            Some(wkb) => {
                                invoke_scalar(&wkb, crs_transform.as_ref(), &mut builder)?;
                                builder.append_value([]);
                            }
                            None => builder.append_null(),
                        }
                        Ok(())
                    })?;
                    Ok(())
                })?;
                return executor.finish(Arc::new(builder.finish()));
            }
        }

        // Iterate over pairs of CRS strings
        let from_crs_array = from.crs_array(&args[from_index], executor.num_iterations())?;
        let to_crs_array = to.crs_array(&args[to_index], executor.num_iterations())?;
        let from_crs_string_view_array = as_string_view_array(&from_crs_array)?;
        let to_crs_string_view_array = as_string_view_array(&to_crs_array)?;
        let mut crs_to_crs_iter = zip(from_crs_string_view_array, to_crs_string_view_array);

        // We might need to build an output array of sanitized CRS strings
        let mut maybe_crs_output = if matches!(to, ArgInput::ArrayCrs) {
            Some(StringViewBuilder::with_capacity(executor.num_iterations()))
        } else {
            None
        };

        with_global_proj_engine(|engine| {
            executor.execute_wkb_void(|maybe_wkb| {
                match (maybe_wkb, crs_to_crs_iter.next().unwrap()) {
                    (Some(wkb), (Some(from_crs_str), Some(to_crs_str))) => {
                        let maybe_from_crs = deserialize_crs(from_crs_str)?;
                        let maybe_to_crs = deserialize_crs(to_crs_str)?;

                        if let Some(crs_output) = &mut maybe_crs_output {
                            if let Some(to_crs) = &maybe_to_crs {
                                crs_output.append_value(to_crs.to_authority_code()?.unwrap_or_else(|| to_crs.to_crs_string()));
                            } else {
                                crs_output.append_null();
                            }
                        }

                        if maybe_from_crs == maybe_to_crs {
                            invoke_noop(&wkb, &mut builder)?;
                            builder.append_value([]);
                            return Ok(());
                        }

                        let crs_transform = match (maybe_from_crs, maybe_to_crs) {
                            (Some(from_crs), Some(to_crs)) => {
                                engine
                                .get_transform_crs_to_crs(&from_crs.to_crs_string(), &to_crs.to_crs_string(), None, "")
                                .map_err(|e| DataFusionError::Execution(format!("{e}")))?
                            },
                            _ => return exec_err!(
                                "Can't transform to or from an unset CRS. Do you need to call ST_SetSRID on the input?"
                            )
                        };

                        invoke_scalar(&wkb, crs_transform.as_ref(), &mut builder)?;
                        builder.append_value([]);
                    }
                    _ => {
                        if let Some(crs_output) = &mut maybe_crs_output {
                            crs_output.append_null();
                        }

                        builder.append_null()
                    },
                }
                Ok(())
            })?;
            Ok(())
        })?;

        let output_geometry = executor.finish(Arc::new(builder.finish()))?;
        if let Some(mut crs_output) = maybe_crs_output {
            let output_crs = executor.finish(Arc::new(crs_output.finish()))?;
            make_item_crs(&WKB_GEOMETRY, output_geometry, &output_crs, None)
        } else {
            Ok(output_geometry)
        }
    }

    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>, DataFusionError> {
        sedona_internal_err!("Return type should only be called with args")
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("invoke_batch should only be called with args")
    }
}

fn output_type_from_scalar_crs_value(scalar_arg: &ScalarValue) -> Result<SedonaType> {
    if let Some(crs_str) = parse_crs_from_scalar_crs_value(scalar_arg)? {
        Ok(SedonaType::Wkb(Edges::Planar, deserialize_crs(&crs_str)?))
    } else {
        Ok(WKB_GEOMETRY)
    }
}

fn parse_crs_from_scalar_crs_value(scalar_arg: &ScalarValue) -> Result<Option<String>> {
    if let ScalarValue::Utf8(maybe_to_crs_str) = scalar_arg.cast_to(&DataType::Utf8)? {
        if let Some(to_crs_str) = maybe_to_crs_str {
            Ok(Some(
                deserialize_crs(&to_crs_str)?
                    .map(|crs| crs.to_crs_string())
                    .unwrap_or("0".to_string()),
            ))
        } else {
            Ok(None)
        }
    } else {
        sedona_internal_err!("Expected scalar cast to utf8 to be a ScalarValue::Utf8")
    }
}

fn invoke_noop(wkb: &Wkb, builder: &mut impl Write) -> Result<()> {
    builder
        .write_all(wkb.buf())
        .map_err(DataFusionError::IoError)
}

fn invoke_scalar(wkb: &Wkb, trans: &dyn CrsTransform, builder: &mut impl Write) -> Result<()> {
    transform(wkb, trans, builder)
        .map_err(|err| DataFusionError::Execution(format!("Transform error: {err}")))?;
    Ok(())
}

/// Helper to label arguments because we have a lot argument types that are valid
#[derive(Debug, Clone, Copy)]
enum ArgInput<'a> {
    /// Geometry input. This currently only matches geometry and not geography
    /// because CRS support for geography is less clear at the moment. Must be
    /// the first argument (and not supported for other arguments).
    Geo(&'a Crs),
    /// Item-level CRS input. Must be the first argument if present (not supported
    /// for other arguments).
    ItemCrs,
    /// Scalar CRS input. Supported for second and third arguments. When present
    /// as the last argument (to), this forces type-level CRS output.
    ScalarCrs(&'a ScalarValue),
    /// Array CRS input. Supported for second and third arguments. When present
    /// as the last (to) argument, this forces Item CRS output.
    ArrayCrs,
    /// Sentinel for anything else
    Unsupported,
}

impl<'a> ArgInput<'a> {
    fn from_return_type_arg(arg_type: &'a SedonaType, scalar_arg: Option<&'a ScalarValue>) -> Self {
        if ArgMatcher::is_item_crs().match_type(arg_type) {
            Self::ItemCrs
        } else if ArgMatcher::is_numeric().match_type(arg_type)
            || ArgMatcher::is_string().match_type(arg_type)
        {
            if let Some(scalar_crs) = scalar_arg {
                Self::ScalarCrs(scalar_crs)
            } else {
                Self::ArrayCrs
            }
        } else {
            match arg_type {
                SedonaType::Wkb(Edges::Planar, crs) | SedonaType::WkbView(Edges::Planar, crs) => {
                    Self::Geo(crs)
                }
                _ => Self::Unsupported,
            }
        }
    }

    fn from_arg(arg_type: &'a SedonaType, arg: &'a ColumnarValue) -> Self {
        if ArgMatcher::is_item_crs().match_type(arg_type) {
            Self::ItemCrs
        } else if ArgMatcher::is_numeric().match_type(arg_type)
            || ArgMatcher::is_string().match_type(arg_type)
        {
            match arg {
                ColumnarValue::Array(_) => Self::ArrayCrs,
                ColumnarValue::Scalar(scalar_value) => Self::ScalarCrs(scalar_value),
            }
        } else {
            match arg_type {
                SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => Self::Geo(crs),
                _ => Self::Unsupported,
            }
        }
    }

    fn crs_constant(&self) -> Result<Option<String>> {
        match self {
            ArgInput::Geo(crs) => {
                let crs_str = if let Some(crs) = crs {
                    crs.to_crs_string()
                } else {
                    "0".to_string()
                };

                Ok(Some(crs_str))
            }
            ArgInput::ScalarCrs(scalar_value) => parse_crs_from_scalar_crs_value(scalar_value),
            _ => Ok(None),
        }
    }

    fn crs_array(&self, arg: &ColumnarValue, iterations: usize) -> Result<ArrayRef> {
        if let Some(crs_constant) = self.crs_constant()? {
            ScalarValue::Utf8View(Some(crs_constant)).to_array_of_size(iterations)
        } else if matches!(self, Self::ItemCrs) {
            match arg {
                ColumnarValue::Array(array) => {
                    let struct_array = as_struct_array(array)?;
                    Ok(struct_array.column(1).clone())
                }
                ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
                    Ok(struct_array.column(1).clone())
                }
                _ => sedona_internal_err!("Unexpected item_crs type"),
            }
        } else {
            arg.cast_to(&DataType::Utf8View, None)?
                .into_array(iterations)
        }
    }
}

/// Configure the global PROJ engine
///
/// Provides an opportunity for a calling application to provide the
/// [ProjCrsEngineBuilder] whose `build()` method will be used to create
/// a set of thread local [CrsEngine]s which in turn will perform the actual
/// computations. This provides an opportunity to configure locations of
/// various files in addition to network CDN access preferences.
///
/// This configuration can be set more than once; however, once the engines
/// are constructed they cannot currently be reconfigured. This code is structured
/// deliberately to ensure that if an error occurs creating an engine that the
/// configuration can be set again. Notably, this will occur if this crate was
/// built without proj-sys the first time somebody calls st_transform.
pub fn configure_global_proj_engine(builder: ProjCrsEngineBuilder) -> Result<()> {
    let mut global_builder = PROJ_ENGINE_BUILDER.try_write().map_err(|_| {
        DataFusionError::Configuration(
            "Failed to acquire write lock for global PROJ configuration".to_string(),
        )
    })?;
    global_builder.replace(builder);
    Ok(())
}

/// Do something with the global thread-local PROJ engine, creating it if it has not
/// already been created.
pub fn with_global_proj_engine<R, F: FnMut(&CachingCrsEngine<ProjCrsEngine>) -> Result<R>>(
    mut func: F,
) -> Result<R> {
    PROJ_ENGINE.with(|engine_cell| {
        // If there is already an engine, use it!
        if let Some(engine) = engine_cell.get() {
            return func(engine);
        }

        // Otherwise, attempt to get the builder
        let maybe_builder = PROJ_ENGINE_BUILDER.read().map_err(|_| {
            // Highly unlikely (can only occur when a panic occurred during set)
            sedona_internal_datafusion_err!(
                "Failed to acquire read lock for global PROJ configuration"
            )
        })?;

        // ...and build the engine. This will use a default configuration
        // (i.e., proj_sys or error) if the builder was never set.
        let proj_engine = maybe_builder
            .as_ref()
            .unwrap_or(&ProjCrsEngineBuilder::default())
            .build()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        engine_cell
            .set(CachingCrsEngine::new(proj_engine))
            .map_err(|_| sedona_internal_datafusion_err!("Failed to set cached PROJ transform"))?;
        func(engine_cell.get().unwrap())
    })
}

/// Global builder as a thread-safe RwLock. Normally set once on application start
/// or never set to use all default settings.
static PROJ_ENGINE_BUILDER: RwLock<Option<ProjCrsEngineBuilder>> =
    RwLock::<Option<ProjCrsEngineBuilder>>::new(None);

// CrsTransform backed by PROJ is not thread safe, so we define the cache as thread-local
// to avoid race conditions.
thread_local! {
    static PROJ_ENGINE: OnceCell<CachingCrsEngine<ProjCrsEngine>> = const {
        OnceCell::<CachingCrsEngine<ProjCrsEngine>>::new()
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::create_array;
    use arrow_array::ArrayRef;
    use arrow_schema::DataType;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::crs::lnglat;
    use sedona_schema::crs::Crs;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::create::create_array_item_crs;
    use sedona_testing::create::create_scalar;
    use sedona_testing::testers::ScalarUdfTester;

    const NAD83ZONE6PROJ: &str = "EPSG:2230";
    const WGS84: &str = "EPSG:4326";

    #[test]
    fn test_invoke_with_string() {
        let udf = SedonaScalarUDF::from_impl("st_transform", st_transform_impl());
        let geometry_input = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![geometry_input.clone(), SedonaType::Arrow(DataType::Utf8)],
        );

        // Return type with scalar to argument (returns type-level CRS)
        let expected_return_type = SedonaType::Wkb(Edges::Planar, get_crs(NAD83ZONE6PROJ));
        let return_type = tester
            .return_type_with_scalar_scalar(Option::<&str>::None, Some(NAD83ZONE6PROJ))
            .unwrap();
        assert_eq!(return_type, expected_return_type);

        // Return type with array to argument (returns item CRS)
        let return_type = tester.return_type().unwrap();
        assert_eq!(return_type, WKB_GEOMETRY_ITEM_CRS.clone());

        // Invoke with scalar to argument (returns type-level CRS)
        let expected_array = create_array(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            &expected_return_type,
        );
        let wkb = create_array(&[None, Some("POINT (79.3871 43.6426)")], &geometry_input);
        let result = tester.invoke_array_scalar(wkb, NAD83ZONE6PROJ).unwrap();
        assert_array_equal(&result, &expected_array);

        // Invoke with array to argument (returns item CRS)
        let expected_array = create_array_item_crs(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            [None, Some(NAD83ZONE6PROJ)],
            &WKB_GEOMETRY,
        );
        let wkb = create_array(&[None, Some("POINT (79.3871 43.6426)")], &geometry_input);
        let crs = create_array!(Utf8, [None, Some(NAD83ZONE6PROJ)]) as ArrayRef;
        let result = tester.invoke_array_array(wkb, crs).unwrap();
        assert_array_equal(&result, &expected_array);
    }

    #[test]
    fn test_invoke_with_srid() {
        let udf = SedonaScalarUDF::from_impl("st_transform", st_transform_impl());
        let geometry_input = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![geometry_input.clone(), SedonaType::Arrow(DataType::UInt32)],
        );

        // Return type with scalar to argument (returns type-level CRS)
        let expected_return_type = SedonaType::Wkb(Edges::Planar, get_crs(NAD83ZONE6PROJ));
        let return_type = tester
            .return_type_with_scalar_scalar(Option::<&str>::None, Some(2230))
            .unwrap();
        assert_eq!(return_type, expected_return_type);

        // Return type with array to argument (returns item CRS)
        let return_type = tester.return_type().unwrap();
        assert_eq!(return_type, WKB_GEOMETRY_ITEM_CRS.clone());

        // Invoke with scalar to argument (returns type-level CRS)
        let expected_array = create_array(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            &expected_return_type,
        );
        let wkb = create_array(&[None, Some("POINT (79.3871 43.6426)")], &geometry_input);
        let result = tester.invoke_array_scalar(wkb, 2230).unwrap();
        assert_array_equal(&result, &expected_array);

        // Invoke with array to argument (returns item CRS)
        let expected_array = create_array_item_crs(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            [None, Some(NAD83ZONE6PROJ)],
            &WKB_GEOMETRY,
        );
        let wkb = create_array(&[None, Some("POINT (79.3871 43.6426)")], &geometry_input);
        let crs = create_array!(Int32, [None, Some(2230)]) as ArrayRef;
        let result = tester.invoke_array_array(wkb, crs).unwrap();
        assert_array_equal(&result, &expected_array);
    }

    #[test]
    fn test_invoke_with_item_crs() {
        let udf = SedonaScalarUDF::from_impl("st_transform", st_transform_impl());
        let geometry_input = WKB_GEOMETRY_ITEM_CRS.clone();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![geometry_input.clone(), SedonaType::Arrow(DataType::Utf8)],
        );

        // Return type with scalar to argument (returns type-level CRS)
        // This is the same as for normal input
        let expected_return_type = SedonaType::Wkb(Edges::Planar, get_crs(NAD83ZONE6PROJ));
        let return_type = tester
            .return_type_with_scalar_scalar(Option::<&str>::None, Some(NAD83ZONE6PROJ))
            .unwrap();
        assert_eq!(return_type, expected_return_type);

        // Return type with array to argument (returns item CRS)
        // This is the same as for normal input
        let return_type = tester.return_type().unwrap();
        assert_eq!(return_type, WKB_GEOMETRY_ITEM_CRS.clone());

        // Invoke with scalar to argument (returns type-level CRS)
        let expected_array = create_array(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            &expected_return_type,
        );
        let array_in = create_array_item_crs(
            &[None, Some("POINT (79.3871 43.6426)")],
            [None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );
        let result = tester
            .invoke_array_scalar(array_in, NAD83ZONE6PROJ)
            .unwrap();
        assert_array_equal(&result, &expected_array);

        // Invoke with array to argument (returns item CRS)
        let expected_array = create_array_item_crs(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            [None, Some(NAD83ZONE6PROJ)],
            &WKB_GEOMETRY,
        );
        let array_in = create_array_item_crs(
            &[None, Some("POINT (79.3871 43.6426)")],
            [None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );
        let crs = create_array!(Utf8, [None, Some(NAD83ZONE6PROJ)]) as ArrayRef;
        let result = tester.invoke_array_array(array_in, crs).unwrap();
        assert_array_equal(&result, &expected_array);
    }

    #[rstest]
    fn test_invoke_source_arg() {
        let udf = SedonaScalarUDF::from_impl("st_transform", st_transform_impl());
        let geometry_input = WKB_GEOMETRY;
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                geometry_input.clone(),
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
            ],
        );

        // Return type with scalar to argument (returns type-level CRS)
        // This is the same as for normal input
        let expected_return_type = SedonaType::Wkb(Edges::Planar, get_crs(NAD83ZONE6PROJ));
        let return_type = tester
            .return_type_with_scalar_scalar_scalar(
                Option::<&str>::None,
                Option::<&str>::None,
                Some(NAD83ZONE6PROJ),
            )
            .unwrap();
        assert_eq!(return_type, expected_return_type);

        // Return type with array to argument (returns item CRS)
        // This is the same as for normal input
        let return_type = tester.return_type().unwrap();
        assert_eq!(return_type, WKB_GEOMETRY_ITEM_CRS.clone());

        // Invoke with scalar to argument (returns type-level CRS)
        let expected_array = create_array(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            &expected_return_type,
        );
        let array_in = create_array(&[None, Some("POINT (79.3871 43.6426)")], &geometry_input);
        let crs_from = create_array!(Utf8, [None, Some(WGS84)]) as ArrayRef;
        let result = tester
            .invoke_array_array_scalar(array_in, crs_from, NAD83ZONE6PROJ)
            .unwrap();
        assert_array_equal(&result, &expected_array);

        // Invoke with array to argument (returns item CRS)
        let expected_array = create_array_item_crs(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            [None, Some(NAD83ZONE6PROJ)],
            &WKB_GEOMETRY,
        );
        let array_in = create_array(&[None, Some("POINT (79.3871 43.6426)")], &WKB_GEOMETRY);
        let crs_from = create_array!(Utf8, [None, Some(WGS84)]) as ArrayRef;
        let crs_to = create_array!(Utf8, [None, Some(NAD83ZONE6PROJ)]) as ArrayRef;
        let result = tester
            .invoke_arrays(vec![array_in, crs_from, crs_to])
            .unwrap();
        assert_array_equal(&result, &expected_array);
    }

    #[test]
    fn test_invoke_null_crs_to() {
        let udf = SedonaScalarUDF::from_impl("st_transform", st_transform_impl());
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Utf8)],
        );

        // A null scalar CRS should generate WKB_GEOMETRY output with a type
        // level CRS that is unset; however, all the output will be null.
        let result = tester
            .invoke_scalar_scalar("POINT (0 1)", ScalarValue::Null)
            .unwrap();
        assert_eq!(result, create_scalar(None, &WKB_GEOMETRY));

        let expected_array = create_array(&[None, None, None], &WKB_GEOMETRY);
        let array_in = create_array(
            &[
                Some("POINT (0 1)"),
                Some("POINT (1 2)"),
                Some("POINT (2 3)"),
            ],
            &WKB_GEOMETRY,
        );
        let result = tester
            .invoke_array_scalar(array_in, ScalarValue::Null)
            .unwrap();
        assert_array_equal(&result, &expected_array);

        // This currently has a side effect of working even though there is not
        // valid transform from lnglat() to an unset CRS (because no transformations
        // will ever take place).
        let geometry_input = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![geometry_input, SedonaType::Arrow(DataType::Utf8)],
        );
        let result = tester
            .invoke_scalar_scalar("POINT (0 1)", ScalarValue::Null)
            .unwrap();
        assert_eq!(result, create_scalar(None, &WKB_GEOMETRY));
    }

    #[test]
    fn test_invoke_unset_crs_to() {
        let udf = SedonaScalarUDF::from_impl("st_transform", st_transform_impl());
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Int32)],
        );

        // A unset scalar CRS should generate WKB_GEOMETRY output with a type
        // level CRS that is unset. This transformation is only valid if the input
        // also has unset CRSes (and the result is a noop).
        let result = tester.invoke_scalar_scalar("POINT (0 1)", 0).unwrap();
        assert_eq!(result, create_scalar(Some("POINT (0 1)"), &WKB_GEOMETRY));

        let array_in = create_array(
            &[
                Some("POINT (0 1)"),
                Some("POINT (1 2)"),
                Some("POINT (2 3)"),
            ],
            &WKB_GEOMETRY,
        );
        let result = tester.invoke_array_scalar(array_in.clone(), 0).unwrap();
        assert_array_equal(&result, &array_in);

        // This should fail, because there is no valid transform between lnglat()
        // and an unset CRS.
        let geometry_input = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![geometry_input, SedonaType::Arrow(DataType::Int32)],
        );
        let err = tester.invoke_scalar_scalar("POINT (0 1)", 0).unwrap_err();
        assert_eq!(
            err.message(),
            "Can't transform to or from an unset CRS. Do you need to call ST_SetSRID on the input?"
        );
    }

    #[test]
    fn invalid_arg_types() {
        let udf = SedonaScalarUDF::from_impl("st_transform", st_transform_impl());

        // No args
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![]);
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "st_transform(): No kernel matching arguments"
        );

        // Too many args
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
            ],
        );
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "st_transform(utf8, utf8, utf8, utf8): No kernel matching arguments"
        );

        // First arg not geometry
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
            ],
        );
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "st_transform(utf8, utf8): No kernel matching arguments"
        );

        // Second arg not string or numeric
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Boolean)],
        );
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "st_transform(geometry, boolean): No kernel matching arguments"
        );

        // third arg not string or numeric
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Boolean),
            ],
        );
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "st_transform(geometry, utf8, boolean): No kernel matching arguments"
        );
    }

    fn get_crs(auth_code: &str) -> Crs {
        deserialize_crs(auth_code).unwrap()
    }
}
