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

//! Extensible scalar kernel properties and borrowed execution configuration.
use crate::{
    extension::{
        SedonaCConfigOptionsView, SedonaCError, SedonaCScalarKernelImpl, SedonaCScalarKernelV2,
    },
    scalar_kernel::{CScalarKernelImplWrapper, ExportedScalarKernelImpl},
    set_ffi_error,
    utils::{
        call_get_json_property_impl, cstr_from_ptr_or_empty, parse_json_c_args,
        write_json_property, write_utf8_property_schema,
    },
};
use arrow_array::{
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    new_empty_array,
};
use datafusion_common::{config::ConfigOptions, exec_err, DataFusionError, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, ScalarUDFImpl, Volatility};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;
use std::{
    collections::HashMap,
    ffi::{c_char, c_int, c_void},
    ptr::{null, null_mut},
    sync::Arc,
};

/// A kernel with the complete function-level declarations of its originating UDF.
pub struct ExportedScalarKernelV2 {
    kernel: ScalarKernelRef,
    name: String,
    metadata: HashMap<String, String>,
    volatility: Volatility,
    config: Arc<ConfigOptions>,
}
impl ExportedScalarKernelV2 {
    /// Export every overload with identical function-level declarations.
    pub fn from_udf(
        udf: &SedonaScalarUDF,
        config: Arc<ConfigOptions>,
    ) -> Vec<SedonaCScalarKernelV2> {
        udf.kernels()
            .iter()
            .map(|kernel| {
                Self {
                    kernel: kernel.clone(),
                    name: udf.name().into(),
                    metadata: udf.metadata().clone(),
                    volatility: udf.signature().volatility,
                    config: config.clone(),
                }
                .into()
            })
            .collect()
    }
}
impl From<ExportedScalarKernelV2> for SedonaCScalarKernelV2 {
    fn from(value: ExportedScalarKernelV2) -> Self {
        Self {
            get_property_schema: Some(property_schema),
            get_property: Some(property),
            new_impl: Some(new_impl),
            reserved: null_mut(),
            release: Some(release),
            private_data: Box::into_raw(Box::new(value)).cast(),
        }
    }
}
unsafe extern "C" fn property_schema(
    _: *const SedonaCScalarKernelV2,
    property: *const c_char,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    match cstr_from_ptr_or_empty(property).as_ref() {
        "function_name" | "volatility" | "metadata_keys" | "metadata" => {
            write_utf8_property_schema(out, err)
        }
        _ => unsupported(err, "Unknown scalar kernel property"),
    }
}
unsafe extern "C" fn property(
    raw: *const SedonaCScalarKernelV2,
    property: *const c_char,
    args: *const c_char,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    let this = &*((*raw).private_data as *const ExportedScalarKernelV2);
    match cstr_from_ptr_or_empty(property).as_ref() {
        "function_name" => write_json_property(&this.name, out, err),
        "volatility" => write_json_property(
            &match this.volatility {
                Volatility::Immutable => "immutable",
                Volatility::Stable => "stable",
                Volatility::Volatile => "volatile",
            },
            out,
            err,
        ),
        "metadata_keys" => {
            let mut keys = this.metadata.keys().collect::<Vec<_>>();
            keys.sort();
            write_json_property(&keys, out, err)
        }
        "metadata" => match parse_json_c_args::<String>(args) {
            Ok(key) => match this.metadata.get(&key) {
                Some(value) => write_json_property(value, out, err),
                None => unsupported(err, "Unknown metadata key"),
            },
            Err(e) => {
                set_ffi_error!(err, "{}", e);
                libc::EINVAL
            }
        },
        _ => unsupported(err, "Unknown scalar kernel property"),
    }
}
unsafe extern "C" fn new_impl(
    raw: *const SedonaCScalarKernelV2,
    config: *const SedonaCConfigOptionsView,
    out: *mut SedonaCScalarKernelImpl,
    err: *mut SedonaCError,
) -> c_int {
    let this = &*((*raw).private_data as *const ExportedScalarKernelV2);
    let config = if let Some(view) = config.as_ref() {
        match import_config(view, &this.config) {
            Ok(config) => Arc::new(config),
            Err(e) => {
                set_ffi_error!(err, "{}", e);
                return libc::EINVAL;
            }
        }
    } else {
        this.config.clone()
    };
    std::ptr::write(
        out,
        ExportedScalarKernelImpl::new(this.kernel.clone(), config).into(),
    );
    0
}
unsafe extern "C" fn release(raw: *mut SedonaCScalarKernelV2) {
    drop(Box::from_raw(
        (*raw).private_data as *mut ExportedScalarKernelV2,
    ));
    (*raw).private_data = null_mut();
    (*raw).release = None;
}

/// An imported V2 kernel. Declarations are read once, before registration.
pub struct ImportedScalarKernelV2 {
    inner: SedonaCScalarKernelV2,
    name: Option<String>,
    metadata: HashMap<String, String>,
    volatility: Volatility,
}
impl std::fmt::Debug for ImportedScalarKernelV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedScalarKernelV2")
            .field("name", &self.name)
            .finish()
    }
}
impl TryFrom<SedonaCScalarKernelV2> for ImportedScalarKernelV2 {
    type Error = DataFusionError;
    fn try_from(inner: SedonaCScalarKernelV2) -> Result<Self> {
        if inner.release.is_none()
            || inner.new_impl.is_none()
            || inner.get_property.is_none()
            || inner.get_property_schema.is_none()
        {
            return exec_err!("Invalid or released SedonaCScalarKernelV2");
        }
        let name = get_property(&inner, "function_name", None)?;
        let volatility: String = get_property(&inner, "volatility", None)?;
        let volatility = match volatility.as_str() {
            "immutable" => Volatility::Immutable,
            "stable" => Volatility::Stable,
            "volatile" => Volatility::Volatile,
            _ => return exec_err!("Invalid kernel volatility: {volatility}"),
        };
        let keys: Vec<String> = get_property(&inner, "metadata_keys", None)?;
        let mut metadata = HashMap::new();
        for key in keys {
            let value: String = get_property(&inner, "metadata", Some(&key))?;
            if metadata.insert(key.clone(), value).is_some() {
                return exec_err!("Duplicate kernel metadata key: {key}");
            }
        }
        Ok(Self {
            inner,
            name,
            metadata,
            volatility,
        })
    }
}
fn get_property<T: serde::de::DeserializeOwned>(
    inner: &SedonaCScalarKernelV2,
    property: &str,
    args: Option<&String>,
) -> Result<T> {
    call_get_json_property_impl(
        property,
        "SedonaCScalarKernelV2",
        args,
        |p, out, err| unsafe { inner.get_property_schema.unwrap()(inner, p, out, err) },
        |p, a, out, err| unsafe { inner.get_property.unwrap()(inner, p, a, out, err) },
    )
}
impl ImportedScalarKernelV2 {
    pub fn function_name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    fn implementation(
        &self,
        config: Option<&SedonaCConfigOptionsView>,
    ) -> Result<CScalarKernelImplWrapper> {
        let mut inner = SedonaCScalarKernelImpl::default();
        let mut err = SedonaCError::default();
        let status = unsafe {
            self.inner.new_impl.unwrap()(
                &self.inner,
                config.map_or(null(), |c| c),
                &mut inner,
                &mut err,
            )
        };
        if status != 0 {
            return exec_err!("SedonaCScalarKernelV2::new_impl failed ({status}): {err}");
        }
        if inner.release.is_none()
            || inner.init.is_none()
            || inner.execute.is_none()
            || inner.get_last_error.is_none()
        {
            return exec_err!("V2 factory returned an invalid implementation");
        }
        Ok(CScalarKernelImplWrapper { inner })
    }
}
impl SedonaScalarKernel for ImportedScalarKernelV2 {
    fn metadata(&self) -> Option<&HashMap<String, String>> {
        Some(&self.metadata)
    }
    fn volatility(&self) -> Option<Volatility> {
        Some(self.volatility)
    }
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.return_type_from_args_and_scalars(args, &vec![None; args.len()])
    }
    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        scalars: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        self.implementation(None)?.init(args, scalars)
    }
    fn invoke_batch(&self, _: &[SedonaType], _: &[ColumnarValue]) -> Result<ColumnarValue> {
        exec_err!("V2 kernel requires invoke_batch_from_args")
    }
    fn invoke_batch_from_args(
        &self,
        types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        rows: usize,
        config: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        if rows == 0 && args.iter().any(|a| matches!(a, ColumnarValue::Array(_))) {
            return Ok(ColumnarValue::Array(new_empty_array(
                return_type.storage_type(),
            )));
        }
        let view = config.map(config_view);
        // The view and its borrowed ConfigOptions outlive the implementation.
        let mut implementation = self.implementation(view.as_ref())?;
        let scalars = args
            .iter()
            .map(|a| match a {
                ColumnarValue::Scalar(v) => Some(v),
                _ => None,
            })
            .collect::<Vec<_>>();
        implementation.init(types, &scalars)?;
        let result = implementation.execute(args, return_type, rows)?;
        if args.iter().any(|a| matches!(a, ColumnarValue::Array(_))) {
            Ok(ColumnarValue::Array(result))
        } else if result.len() == 1 {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            exec_err!("Expected scalar result, got {} rows", result.len())
        }
    }
}
fn config_view(config: &ConfigOptions) -> SedonaCConfigOptionsView {
    SedonaCConfigOptionsView {
        get_property_schema: Some(config_schema),
        get_property: Some(config_property),
        get_extension: Some(config_extension),
        reserved: null_mut(),
        private_data: (config as *const ConfigOptions).cast(),
    }
}
unsafe extern "C" fn config_schema(
    _: *const SedonaCConfigOptionsView,
    property: *const c_char,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    if cstr_from_ptr_or_empty(property) == "option" {
        write_utf8_property_schema(out, err)
    } else {
        unsupported(err, "Unknown configuration property")
    }
}
unsafe extern "C" fn config_property(
    raw: *const SedonaCConfigOptionsView,
    property: *const c_char,
    args: *const c_char,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    if cstr_from_ptr_or_empty(property) != "option" {
        return unsupported(err, "Unknown configuration property");
    }
    let key = match parse_json_c_args::<String>(args) {
        Ok(key) => key,
        Err(e) => {
            set_ffi_error!(err, "{}", e);
            return libc::EINVAL;
        }
    };
    let config = &*((*raw).private_data as *const ConfigOptions);
    let value = config
        .entries()
        .into_iter()
        .find(|e| e.key == key)
        .and_then(|e| e.value);
    write_json_property(&value, out, err)
}
unsafe extern "C" fn config_extension(
    _: *const SedonaCConfigOptionsView,
    _: c_int,
    _: *mut c_void,
    _: usize,
    err: *mut SedonaCError,
) -> c_int {
    unsupported(err, "Unknown configuration extension")
}
unsafe fn unsupported(err: *mut SedonaCError, message: &str) -> c_int {
    set_ffi_error!(err, message);
    libc::ENOTSUP
}
// Overlay options understood by the producer, retaining producer-side typed
// extensions. Arbitrary Rust extension objects cannot cross this C ABI.
fn import_config(
    view: &SedonaCConfigOptionsView,
    defaults: &ConfigOptions,
) -> Result<ConfigOptions> {
    let (Some(schema), Some(property)) = (view.get_property_schema, view.get_property) else {
        return exec_err!("Invalid configuration view");
    };
    let mut config = defaults.clone();
    for entry in defaults.entries() {
        let value: Option<String> = call_get_json_property_impl(
            "option",
            "SedonaCConfigOptionsView",
            Some(&entry.key),
            |p, out, err| unsafe { schema(view, p, out, err) },
            |p, args, out, err| unsafe { property(view, p, args, out, err) },
        )?;
        if let Some(value) = value {
            config.set(&entry.key, &value)?;
        }
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;
    use sedona_expr::{function_set::FunctionSet, scalar_udf::SimpleSedonaScalarKernel};
    use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};
    use sedona_testing::{create::create_array, testers::ScalarUdfTester};

    fn udf() -> SedonaScalarUDF {
        SedonaScalarUDF::new(
            "identity",
            vec![SimpleSedonaScalarKernel::new_ref(
                ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
                Arc::new(|_, args| Ok(args[0].clone())),
            )],
            Volatility::Stable,
        )
        .with_metadata("needs_pixels", "true")
        .with_metadata("arbitrary.\"key", "unicode: 🦆\0value")
    }
    fn imported(udf: &SedonaScalarUDF) -> ScalarKernelRef {
        let raw = ExportedScalarKernelV2::from_udf(udf, Arc::new(ConfigOptions::new()))
            .pop()
            .unwrap();
        Arc::new(ImportedScalarKernelV2::try_from(raw).unwrap())
    }
    #[test]
    fn metadata_registration_and_execution_roundtrip() {
        let original = udf();
        let kernel = imported(&original);
        let mut functions = FunctionSet::new();
        let registered = functions
            .add_scalar_udf_impl("identity", vec![kernel])
            .unwrap();
        assert_eq!(registered.metadata(), original.metadata());
        assert_eq!(registered.signature().volatility, Volatility::Stable);
        let tester = ScalarUdfTester::new(registered.clone().into(), vec![WKB_GEOMETRY]);
        tester.assert_return_type(WKB_GEOMETRY);
        let result = tester.invoke_scalar("POINT (0 1)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 1)");
        for values in [vec![Some("POINT (0 1)"), None], vec![]] {
            let array = create_array(&values, &WKB_GEOMETRY);
            assert_eq!(&tester.invoke_array(array.clone()).unwrap(), &array);
        }
    }
    #[test]
    fn conflicts_do_not_mutate_registry() {
        let mut functions = FunctionSet::new();
        let original = udf();
        functions
            .add_scalar_udf_impl("identity", vec![imported(&original)])
            .unwrap();
        let conflicting = original.clone().with_metadata("another_key", "true");
        assert!(functions
            .add_scalar_udf_impl("identity", vec![imported(&conflicting)])
            .is_err());
        let different_volatility = SedonaScalarUDF::new(
            "identity",
            original.kernels().to_vec(),
            Volatility::Volatile,
        )
        .with_metadata("needs_pixels", "true")
        .with_metadata("arbitrary.\"key", "unicode: 🦆\0value");
        assert!(functions
            .add_scalar_udf_impl("identity", vec![imported(&different_volatility)])
            .is_err());
        assert_eq!(functions.scalar_udf("identity").unwrap().kernels().len(), 1);
        functions
            .add_scalar_udf_impl("identity", vec![imported(&original)])
            .unwrap();
        assert_eq!(functions.scalar_udf("identity").unwrap().kernels().len(), 2);
        assert!(SedonaScalarUDF::try_from_kernels(
            "identity",
            vec![imported(&original), imported(&conflicting)],
            None
        )
        .is_err());
    }
    #[test]
    fn property_errors_and_released_import() {
        let raw = ExportedScalarKernelV2::from_udf(&udf(), Arc::new(ConfigOptions::new()))
            .pop()
            .unwrap();
        let mut schema = FFI_ArrowSchema::empty();
        let mut out = FFI_ArrowArray::empty();
        let mut err = SedonaCError::default();
        unsafe {
            assert_eq!(
                raw.get_property_schema.unwrap()(&raw, c"unknown".as_ptr(), &mut schema, &mut err),
                libc::ENOTSUP
            );
            assert_eq!(
                raw.get_property.unwrap()(&raw, c"unknown".as_ptr(), null(), &mut out, &mut err),
                libc::ENOTSUP
            );
            assert_eq!(
                raw.get_property.unwrap()(
                    &raw,
                    c"metadata".as_ptr(),
                    c"\"missing\"".as_ptr(),
                    &mut out,
                    &mut err
                ),
                libc::ENOTSUP
            );
            assert_eq!(
                raw.get_property.unwrap()(
                    &raw,
                    c"metadata".as_ptr(),
                    c"[".as_ptr(),
                    &mut out,
                    &mut err
                ),
                libc::EINVAL
            );
        }
        assert!(err.to_string().contains("Failed to parse"));
        assert!(ImportedScalarKernelV2::try_from(SedonaCScalarKernelV2::default()).is_err());
    }
    #[derive(Debug)]
    struct BatchSize;
    impl SedonaScalarKernel for BatchSize {
        fn return_type(&self, _: &[SedonaType]) -> Result<Option<SedonaType>> {
            Ok(Some(SedonaType::Arrow(DataType::UInt64)))
        }
        fn invoke_batch(&self, _: &[SedonaType], _: &[ColumnarValue]) -> Result<ColumnarValue> {
            unreachable!()
        }
        fn invoke_batch_from_args(
            &self,
            _: &[SedonaType],
            _: &[ColumnarValue],
            _: &SedonaType,
            _: usize,
            config: Option<&ConfigOptions>,
        ) -> Result<ColumnarValue> {
            Ok(ColumnarValue::Scalar(ScalarValue::UInt64(
                config.map(|c| c.execution.batch_size as u64),
            )))
        }
    }
    #[test]
    fn caller_config_overrides_exporter_and_null_uses_exporter() {
        let original =
            SedonaScalarUDF::new("batch_size", vec![Arc::new(BatchSize)], Volatility::Stable);
        let mut defaults = ConfigOptions::new();
        defaults.execution.batch_size = 17;
        let raw = ExportedScalarKernelV2::from_udf(&original, Arc::new(defaults))
            .pop()
            .unwrap();
        let kernel = ImportedScalarKernelV2::try_from(raw).unwrap();
        let mut caller = ConfigOptions::new();
        caller.execution.batch_size = 23;
        for (config, expected) in [(None, 17), (Some(&caller), 23)] {
            let value = kernel
                .invoke_batch_from_args(&[], &[], &SedonaType::Arrow(DataType::UInt64), 1, config)
                .unwrap();
            assert!(
                matches!(value, ColumnarValue::Scalar(ScalarValue::UInt64(Some(v))) if v == expected)
            );
        }
        let view = config_view(&caller);
        let mut sentinel = 123u64;
        let mut err = SedonaCError::default();
        let status = unsafe {
            view.get_extension.unwrap()(&view, 999, (&mut sentinel as *mut u64).cast(), 8, &mut err)
        };
        assert_eq!(status, libc::ENOTSUP);
        assert_eq!(sentinel, 123);
    }
}
