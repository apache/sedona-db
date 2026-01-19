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
use arrow_array::builder::BinaryBuilder;
use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::{plan_err, DataFusionError, Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use geo_traits::to_geo::ToGeoGeometry;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_geometry::transform::{transform, CachingCrsEngine, CrsEngine, CrsTransform};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::crs::{CoordinateReferenceSystem, Crs, deserialize_crs};
use sedona_schema::datatypes::{Edges, SedonaType, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS};
use sedona_schema::matchers::ArgMatcher;
use std::cell::OnceCell;
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use wkb::reader::Wkb;

#[derive(Debug)]
struct STTransform {}

/// ST_Transform() implementation using the proj crate
pub fn st_transform_impl() -> ScalarKernelRef {
    Arc::new(STTransform {})
}



enum CrsInput<'a> {
    Crs(&'a Crs),
    Scalar(&'a ScalarValue),
    Array(Option<&'a ArrayRef>),
    ItemCrs(Option<&'a ArrayRef>),
    Unsupported(&'a SedonaType),
}

impl<'a> CrsInput<'a> {
    fn from_arg(arg_type: &'a SedonaType, scalar_arg: Option<&'a ScalarValue>) -> Self {
        if ArgMatcher::is_item_crs().match_type(arg_type) {
            Self::ItemCrs(None)
        } else if ArgMatcher::is_numeric().match_type(arg_type)
            || ArgMatcher::is_string().match_type(arg_type)
        {
            if let Some(scalar_crs) = scalar_arg {
                CrsInput::Scalar(scalar_crs)
            } else {
                CrsInput::Array(None)
            }
        } else {
            match arg_type {
                SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => CrsInput::Crs(crs),
                other => Self::Unsupported(other),
            }
        }
    }
}

enum STTransformImpl {
    NoOp,
    OutputIsAllNull,
    ScalarToScalar(String, String),
    ScalarToItemCrs(String),
    ItemCrsToScalar(String),
    ItemCrsToItemCrs,
}

impl STTransformImpl {
    fn try_from_types_and_scalar_args(
        arg_types: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<Self>> {
        if Self::is_transform_to(arg_types) {
            let crs_from_string = Self::parse_crs_from_type(&arg_types[0])?;

            if let Some(crs_to_scalar) = scalar_args[1] {
                if let Some(crs_to_string) = Self::parse_crs_from_scalar_value(crs_to_scalar)? {
                    Ok(Some(Self::ScalarToScalar(crs_from_string, crs_to_string)))
                } else {
                    Ok(Some(Self::OutputIsAllNull))
                }
            } else {
                Ok(Some(Self::ScalarToItemCrs(crs_from_string)))
            }
        } else if Self::is_transform_from_item_crs_to(arg_types) {
            if let Some(crs_to_scalar) = scalar_args[1] {
                if let Some(crs_to_string) = Self::parse_crs_from_scalar_value(crs_to_scalar)? {
                    Ok(Some(Self::ItemCrsToScalar(crs_to_string)))
                } else {
                    Ok(Some(Self::OutputIsAllNull))
                }
            } else {
                Ok(Some(Self::ItemCrsToItemCrs))
            }
        } else if Self::is_transform_from_to(arg_types) {
            todo!()
        } else {
            Ok(None)
        }
    }

    fn try_from_types_and_args(arg_types: &[SedonaType], args: &[ColumnarValue]) -> Result<Self> {
        let scalar_args = args
            .iter()
            .map(|arg| match arg {
                ColumnarValue::Array(_) => None,
                ColumnarValue::Scalar(scalar_value) => Some(scalar_value),
            })
            .collect::<Vec<_>>();

        if let Some(out) = Self::try_from_types_and_scalar_args(arg_types, &scalar_args)? {
            Ok(out)
        } else {
            sedona_internal_err!("ST_Transform invoked with in compatible arguments")
        }
    }

    fn is_transform_to(arg_types: &[SedonaType]) -> bool {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::or(vec![ArgMatcher::is_integer(), ArgMatcher::is_string()]),
            ],
            // We'll compute the actual type separately
            SedonaType::Arrow(DataType::Null),
        );

        matcher.matches(arg_types)
    }

    fn is_transform_from_to(arg_types: &[SedonaType]) -> bool {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::or(vec![ArgMatcher::is_integer(), ArgMatcher::is_string()]),
                ArgMatcher::or(vec![ArgMatcher::is_integer(), ArgMatcher::is_string()]),
            ],
            // We'll compute the actual type separately
            SedonaType::Arrow(DataType::Null),
        );

        matcher.matches(arg_types)
    }

    fn is_transform_from_item_crs_to(arg_types: &[SedonaType]) -> bool {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_item_crs(),
                ArgMatcher::or(vec![ArgMatcher::is_integer(), ArgMatcher::is_string()]),
            ],
            // We'll compute the actual type separately
            SedonaType::Arrow(DataType::Null),
        );

        matcher.matches(arg_types)
    }

    fn parse_crs_from_type(sedona_type: &SedonaType) -> Result<String> {
        match sedona_type {
            SedonaType::Wkb(_, Some(crs)) | SedonaType::WkbView(_, Some(crs)) => {
                Ok(crs.to_crs_string())
            }
            SedonaType::Wkb(_, None) | SedonaType::WkbView(_, None) => Ok("0".to_string()),
            _ => sedona_internal_err!("Can't parse CRS from SedonaType {sedona_type:?}"),
        }
    }

    fn parse_crs_from_scalar_value(scalar_arg: &ScalarValue) -> Result<Option<String>> {
        if let ScalarValue::Utf8(maybe_to_crs_str) = scalar_arg.cast_to(&DataType::Utf8)? {
            Ok(maybe_to_crs_str)
        } else {
            sedona_internal_err!("Expected scalar cast to utf8 to be a ScalarValue::Utf8")
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
pub(crate) fn with_global_proj_engine(
    mut func: impl FnMut(&CachingCrsEngine<ProjCrsEngine>) -> Result<()>,
) -> Result<()> {
    PROJ_ENGINE.with(|engine_cell| {
        // If there is already an engine, use it!
        if let Some(engine) = engine_cell.get() {
            return func(engine);
        }

        // Otherwise, attempt to get the builder
        let maybe_builder = PROJ_ENGINE_BUILDER.read().map_err(|_| {
            // Highly unlikely (can only occur when a panic occurred during set)
            DataFusionError::Internal(
                "Failed to acquire read lock for global PROJ configuration".to_string(),
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
            .map_err(|_| {
                DataFusionError::Internal("Failed to set cached PROJ transform".to_string())
            })?;
        func(engine_cell.get().unwrap())?;
        Ok(())
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

#[derive(Debug)]
struct STTransformTo {}

impl SedonaScalarKernel for STTransformTo {
    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>, DataFusionError> {
        sedona_internal_err!("Return type should only be called with args")
    }

    fn return_type_from_args_and_scalars(
        &self,
        arg_types: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::or(vec![ArgMatcher::is_integer(), ArgMatcher::is_string()]),
            ],
            // We'll compute the actual return type based on scalar arguments (if present) or
            // return item_crs (if not).
            SedonaType::Wkb(Edges::Planar, None),
        );

        if !matcher.matches(arg_types) {
            return Ok(None);
        }

        if let Some(to_crs_scalar) = scalar_args[1] {
            if let ScalarValue::Utf8(maybe_to_crs_str) = to_crs_scalar.cast_to(&DataType::Utf8)? {
                match maybe_to_crs_str {
                    Some(to_crs_str) => {
                        let to_crs = deserialize_crs(&to_crs_str)?;
                        Ok(Some(SedonaType::Wkb(Edges::Planar, to_crs)))
                    }
                    None => Ok(Some(WKB_GEOMETRY)),
                }
            } else {
                sedona_internal_err!("Expected scalar cast to utf8 to be a ScalarValue::Utf8")
            }
        } else {
            Ok(Some(WKB_GEOMETRY_ITEM_CRS.clone()))
        }
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("invoke_batch should only be called with args")
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
    ) -> Result<ColumnarValue> {
        todo!()
    }
}

#[derive(Debug)]
struct STTransformFromTo {}

impl SedonaScalarKernel for STTransformFromTo {
    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>, DataFusionError> {
        sedona_internal_err!("Return type should only be called with args")
    }

    fn return_type_from_args_and_scalars(
        &self,
        arg_types: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        todo!()
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("invoke_batch should only be called with args")
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
    ) -> Result<ColumnarValue> {
        todo!()
    }
}

fn parse_crs_from_columnar_value(scalar_arg: &ScalarValue) -> Result<Option<String>> {
    if let ScalarValue::Utf8(maybe_to_crs_str) = scalar_arg.cast_to(&DataType::Utf8)? {
        Ok(maybe_to_crs_str)
    } else {
        sedona_internal_err!("Expected scalar cast to utf8 to be a ScalarValue::Utf8")
    }
}

fn parse_crs_from_type(sedona_type: &SedonaType) -> Result<Option<String>> {
    match sedona_type {
        SedonaType::Wkb(_, Some(crs)) | SedonaType::WkbView(_, Some(crs)) => {
            Ok(Some(crs.to_crs_string()))
        }
        SedonaType::Wkb(_, None) | SedonaType::WkbView(_, None) => Ok(None),
        _ => sedona_internal_err!("Can't parse CRS from SedonaType {sedona_type:?}"),
    }
}

fn parse_crs_from_scalar_value(scalar_arg: &ScalarValue) -> Result<Option<String>> {
    if let ScalarValue::Utf8(maybe_to_crs_str) = scalar_arg.cast_to(&DataType::Utf8)? {
        Ok(maybe_to_crs_str)
    } else {
        sedona_internal_err!("Expected scalar cast to utf8 to be a ScalarValue::Utf8")
    }
}

struct TransformArgIndexes {
    wkb: usize,
    first_crs: usize,
    second_crs: Option<usize>,
    lenient: Option<usize>,
}

impl TransformArgIndexes {
    fn new() -> Self {
        Self {
            wkb: 0,
            first_crs: 1,
            second_crs: None,
            lenient: None,
        }
    }
}

fn define_arg_indexes(arg_types: &[SedonaType], indexes: &mut TransformArgIndexes) {
    indexes.wkb = 0;
    indexes.first_crs = 1;

    for (i, arg_type) in arg_types.iter().enumerate().skip(2) {
        if ArgMatcher::is_numeric().match_type(arg_type)
            || ArgMatcher::is_string().match_type(arg_type)
        {
            indexes.second_crs = Some(i);
        } else if *arg_type == SedonaType::Arrow(DataType::Boolean) {
            indexes.lenient = Some(i);
        }
    }
}

impl SedonaScalarKernel for STTransform {
    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>, DataFusionError> {
        sedona_internal_err!("Return type should only be called with args")
    }

    fn return_type_from_args_and_scalars(
        &self,
        arg_types: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::or(vec![ArgMatcher::is_numeric(), ArgMatcher::is_string()]),
                ArgMatcher::optional(ArgMatcher::or(vec![
                    ArgMatcher::is_numeric(),
                    ArgMatcher::is_string(),
                ])),
                ArgMatcher::optional(ArgMatcher::is_boolean()),
            ],
            SedonaType::Wkb(Edges::Planar, None),
        );

        if !matcher.matches(arg_types) {
            return Ok(None);
        }

        let mut indexes = TransformArgIndexes::new();
        define_arg_indexes(arg_types, &mut indexes);

        let scalar_arg_opt = if let Some(second_crs_index) = indexes.second_crs {
            scalar_args.get(second_crs_index).unwrap()
        } else {
            scalar_args.get(indexes.first_crs).unwrap()
        };

        let crs_str_opt = if let Some(scalar_crs) = scalar_arg_opt {
            to_crs_str(scalar_crs)
        } else {
            None
        };

        // If there is no CRS argument, we cannot determine the return type.
        match crs_str_opt {
            Some(to_crs) => {
                let crs = deserialize_crs(&to_crs)?;
                Ok(Some(SedonaType::Wkb(Edges::Planar, crs)))
            }
            _ => Ok(Some(SedonaType::Wkb(Edges::Planar, None))),
        }
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let mut indexes = TransformArgIndexes::new();
        define_arg_indexes(arg_types, &mut indexes);

        let first_crs = get_crs_str(args, indexes.first_crs).ok_or_else(|| {
            DataFusionError::Execution(
                "First CRS argument must be a string or numeric scalar".to_string(),
            )
        })?;

        let lenient = indexes
            .lenient
            .is_some_and(|i| get_scalar_bool(args, i).unwrap_or(false));

        let second_crs = if let Some(second_crs_index) = indexes.second_crs {
            get_crs_str(args, second_crs_index)
        } else {
            None
        };

        with_global_proj_engine(|engine| {
            let crs_from_geo = parse_crs_from_type(&arg_types[indexes.wkb])?;

            let transform = match &second_crs {
                Some(to_crs) => get_transform_crs_to_crs(engine, &first_crs, to_crs)?,
                None => get_transform_to_crs(engine, crs_from_geo, &first_crs, lenient)?,
            };

            executor.execute_wkb_void(|maybe_wkb| {
                match maybe_wkb {
                    Some(wkb) => invoke_scalar(&wkb, transform.as_ref(), &mut builder)?,
                    None => builder.append_null(),
                }

                Ok(())
            })?;

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn get_transform_to_crs(
    engine: &dyn CrsEngine,
    source_crs_opt: Option<String>,
    to_crs: &str,
    lenient: bool,
) -> Result<Rc<dyn CrsTransform>, DataFusionError> {
    let from_crs = match source_crs_opt {
        Some(crs) => crs,
        None if lenient => "EPSG:4326".to_string(),
        None => {
            return Err(DataFusionError::Execution(
                "Source CRS is required when transforming to a CRS".to_string(),
            ))
        }
    };
    get_transform_crs_to_crs(engine, &from_crs, to_crs)
}

fn get_transform_crs_to_crs(
    engine: &dyn CrsEngine,
    from_crs: &str,
    to_crs: &str,
) -> Result<Rc<dyn CrsTransform>, DataFusionError> {
    engine
        .get_transform_crs_to_crs(from_crs, to_crs, None, "")
        .map_err(|err| DataFusionError::Execution(format!("Transform error: {err}")))
}

fn invoke_scalar(wkb: &Wkb, trans: &dyn CrsTransform, builder: &mut BinaryBuilder) -> Result<()> {
    let geo_geom = wkb.to_geometry();
    transform(&geo_geom, trans, builder)
        .map_err(|err| DataFusionError::Execution(format!("Transform error: {err}")))?;
    builder.append_value([]);
    Ok(())
}

fn to_crs_str(scalar_arg: &ScalarValue) -> Option<String> {
    if let Ok(ScalarValue::Utf8(Some(crs))) = scalar_arg.cast_to(&DataType::Utf8) {
        if crs.chars().all(|c| c.is_ascii_digit()) {
            return Some(format!("EPSG:{crs}"));
        } else {
            return Some(crs);
        }
    }

    None
}

fn get_crs_str(args: &[ColumnarValue], index: usize) -> Option<String> {
    if let ColumnarValue::Scalar(scalar_crs) = &args[index] {
        return to_crs_str(scalar_crs);
    }
    None
}

fn get_scalar_bool(args: &[ColumnarValue], index: usize) -> Option<bool> {
    if let Some(ColumnarValue::Scalar(ScalarValue::Boolean(opt_bool))) = args.get(index) {
        *opt_bool
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use arrow_schema::{DataType, Field};
    use datafusion_common::config::ConfigOptions;
    use datafusion_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl};
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::crs::lnglat;
    use sedona_schema::crs::Crs;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::compare::assert_value_equal;
    use sedona_testing::{create::create_array, create::create_array_value};

    const NAD83ZONE6PROJ: &str = "EPSG:2230";
    const WGS84: &str = "EPSG:4326";

    #[rstest]
    fn invalid_arg_checks() {
        let udf: SedonaScalarUDF = SedonaScalarUDF::from_impl("st_transform", st_transform_impl());

        // No args
        let result = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &[],
            scalar_arguments: &[],
        });
        assert!(
            result.is_err()
                && result
                    .unwrap_err()
                    .to_string()
                    .contains("No kernel matching arguments")
        );

        // Too many args
        let arg_types = [
            WKB_GEOMETRY,
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Int32),
        ];
        let arg_fields: Vec<Arc<Field>> = arg_types
            .iter()
            .map(|arg_type| Arc::new(arg_type.to_storage_field("", true).unwrap()))
            .collect();
        let result = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &[None, None, None, None, None],
        });
        assert!(
            result.is_err()
                && result
                    .unwrap_err()
                    .to_string()
                    .contains("No kernel matching arguments")
        );

        // First arg not geometry
        let arg_types = [
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Utf8),
        ];
        let arg_fields: Vec<Arc<Field>> = arg_types
            .iter()
            .map(|arg_type| Arc::new(arg_type.to_storage_field("", true).unwrap()))
            .collect();
        let result = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &[None, None],
        });
        assert!(
            result.is_err()
                && result
                    .unwrap_err()
                    .to_string()
                    .contains("No kernel matching arguments")
        );

        // Second arg not string or numeric
        let arg_types = [WKB_GEOMETRY, SedonaType::Arrow(DataType::Boolean)];
        let arg_fields: Vec<Arc<Field>> = arg_types
            .iter()
            .map(|arg_type| Arc::new(arg_type.to_storage_field("", true).unwrap()))
            .collect();
        let result = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &[None, None],
        });
        assert!(
            result.is_err()
                && result
                    .unwrap_err()
                    .to_string()
                    .contains("No kernel matching arguments")
        );
    }

    #[rstest]
    fn test_invoke_batch_with_geo_crs() {
        // From-CRS pulled from sedona type
        let arg_types = [
            SedonaType::Wkb(Edges::Planar, lnglat()),
            SedonaType::Arrow(DataType::Utf8),
        ];

        let wkb = create_array(&[None, Some("POINT (79.3871 43.6426)")], &arg_types[0]);

        let scalar_args = vec![ScalarValue::Utf8(Some(NAD83ZONE6PROJ.to_string()))];

        let expected = create_array_value(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            &SedonaType::Wkb(Edges::Planar, get_crs(NAD83ZONE6PROJ)),
        );

        let (result_type, result_col) =
            invoke_udf_test(wkb, scalar_args, arg_types.to_vec()).unwrap();
        assert_value_equal(&result_col, &expected);
        assert_eq!(
            result_type,
            SedonaType::Wkb(Edges::Planar, get_crs(NAD83ZONE6PROJ))
        );
    }

    #[rstest]
    fn test_invoke_with_srids() {
        // Use an integer SRID for the to CRS
        let arg_types = [
            SedonaType::Wkb(Edges::Planar, lnglat()),
            SedonaType::Arrow(DataType::UInt32),
        ];

        let wkb = create_array(&[None, Some("POINT (79.3871 43.6426)")], &arg_types[0]);

        let scalar_args = vec![ScalarValue::UInt32(Some(2230))];

        let expected = create_array_value(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            &SedonaType::Wkb(Edges::Planar, get_crs(NAD83ZONE6PROJ)),
        );

        let (result_type, result_col) =
            invoke_udf_test(wkb, scalar_args, arg_types.to_vec()).unwrap();
        assert_value_equal(&result_col, &expected);
        assert_eq!(
            result_type,
            SedonaType::Wkb(Edges::Planar, get_crs(NAD83ZONE6PROJ))
        );
    }

    #[rstest]
    fn test_invoke_batch_with_lenient() {
        let arg_types = [
            WKB_GEOMETRY,
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Boolean),
        ];

        let wkb = create_array(&[None, Some("POINT (79.3871 43.6426)")], &WKB_GEOMETRY);
        let scalar_args = vec![
            ScalarValue::Utf8(Some(NAD83ZONE6PROJ.to_string())),
            ScalarValue::Boolean(Some(true)),
        ];

        let expected = create_array_value(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            &SedonaType::Wkb(Edges::Planar, Some(get_crs(NAD83ZONE6PROJ).unwrap())),
        );

        let (result_type, result_col) =
            invoke_udf_test(wkb, scalar_args, arg_types.to_vec()).unwrap();
        assert_value_equal(&result_col, &expected);
        assert_eq!(
            result_type,
            SedonaType::Wkb(Edges::Planar, Some(get_crs(NAD83ZONE6PROJ).unwrap()))
        );
    }

    #[rstest]
    fn test_invoke_batch_one_crs_no_lenient() {
        let arg_types = [WKB_GEOMETRY, SedonaType::Arrow(DataType::Utf8)];

        let wkb = create_array(&[None, Some("POINT (79.3871 43.6426)")], &WKB_GEOMETRY);
        let scalar_args = vec![ScalarValue::Utf8(Some(NAD83ZONE6PROJ.to_string()))];

        let err = invoke_udf_test(wkb, scalar_args, arg_types.to_vec());
        assert!(
            matches!(err, Err(DataFusionError::Execution(_))),
            "Expected an Execution error"
        );
    }

    #[rstest]
    fn test_invoke_batch_with_source_arg() {
        let arg_types = [
            WKB_GEOMETRY,
            SedonaType::Arrow(DataType::Utf8),
            SedonaType::Arrow(DataType::Utf8),
        ];

        let wkb = create_array(&[None, Some("POINT (79.3871 43.6426)")], &WKB_GEOMETRY);

        let scalar_args = vec![
            ScalarValue::Utf8(Some(WGS84.to_string())),
            ScalarValue::Utf8(Some(NAD83ZONE6PROJ.to_string())),
        ];

        // Note: would be nice to have an epsilon of tolerance when validating
        let expected = create_array_value(
            &[None, Some("POINT (-21508577.363421552 34067918.06097863)")],
            &SedonaType::Wkb(Edges::Planar, Some(get_crs(NAD83ZONE6PROJ).unwrap())),
        );

        let (result_type, result_col) =
            invoke_udf_test(wkb.clone(), scalar_args, arg_types.to_vec()).unwrap();
        assert_value_equal(&result_col, &expected);
        assert_eq!(
            result_type,
            SedonaType::Wkb(Edges::Planar, Some(get_crs(NAD83ZONE6PROJ).unwrap()))
        );

        // Test with integer SRIDs
        let arg_types = [
            WKB_GEOMETRY,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ];

        let scalar_args = vec![
            ScalarValue::Int32(Some(4326)),
            ScalarValue::Int32(Some(2230)),
        ];

        let (result_type, result_col) =
            invoke_udf_test(wkb, scalar_args, arg_types.to_vec()).unwrap();
        assert_value_equal(&result_col, &expected);
        assert_eq!(
            result_type,
            SedonaType::Wkb(Edges::Planar, Some(get_crs(NAD83ZONE6PROJ).unwrap()))
        );
    }

    fn get_crs(auth_code: &str) -> Crs {
        deserialize_crs(auth_code).unwrap()
    }

    fn invoke_udf_test(
        wkb: ArrayRef,
        scalar_args: Vec<ScalarValue>,
        arg_types: Vec<SedonaType>,
    ) -> Result<(SedonaType, ColumnarValue)> {
        let udf = SedonaScalarUDF::from_impl("st_transform", st_transform_impl());

        let arg_fields: Vec<Arc<Field>> = arg_types
            .into_iter()
            .map(|arg_type| Arc::new(arg_type.to_storage_field("", true).unwrap()))
            .collect();
        let row_count = wkb.len();

        let mut scalar_args_fields: Vec<Option<&ScalarValue>> = vec![None];
        let mut arg_vals: Vec<ColumnarValue> = vec![ColumnarValue::Array(Arc::new(wkb))];

        for scalar_arg in &scalar_args {
            scalar_args_fields.push(Some(scalar_arg));
            arg_vals.push(scalar_arg.clone().into());
        }

        let return_field_args = ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_args_fields,
        };

        let return_field = udf.return_field_from_args(return_field_args)?;
        let return_type = SedonaType::from_storage_field(&return_field)?;

        let args = ScalarFunctionArgs {
            args: arg_vals,
            arg_fields: arg_fields.to_vec(),
            number_rows: row_count,
            return_field,
            config_options: Arc::new(ConfigOptions::default()),
        };

        let value = udf.invoke_with_args(args)?;
        Ok((return_type, value))
    }
}
