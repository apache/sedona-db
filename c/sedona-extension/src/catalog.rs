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

//! Version-agnostic FFI wrappers for DataFusion's catalog hierarchy.

use std::ffi::{c_char, c_int, CString};
use std::fmt::{Debug, Formatter};
use std::ptr::null_mut;
use std::sync::Arc;

use arrow_array::ffi::FFI_ArrowArray;
use arrow_schema::ffi::FFI_ArrowSchema;
use async_trait::async_trait;
use datafusion_catalog::{
    CatalogProvider, CatalogProviderList, SchemaProvider, Session, TableProvider,
};
use datafusion_common::{not_impl_err, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::extension::{
    SedonaCCatalogProvider, SedonaCCatalogProviderList, SedonaCError, SedonaCSchemaProvider,
    SedonaCTableProvider,
};
use crate::runtime::RuntimeHandle;
use crate::set_ffi_error;
use crate::table_provider::{ExportedTableProvider, ImportedTableProvider};
use crate::utils::{
    call_get_json_property_impl, cstr_from_ptr_or_empty, parse_json_c_args, write_json_property,
    write_utf8_property_schema, ERRNO_OK,
};

/// Exports a [`CatalogProviderList`] through [`SedonaCCatalogProviderList`].
pub struct ExportedCatalogProviderList {
    inner: Arc<dyn CatalogProviderList>,
    session: Arc<dyn Session>,
    runtime: Arc<RuntimeHandle>,
}

impl Debug for ExportedCatalogProviderList {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedCatalogProviderList")
            .field("inner", &self.inner)
            .finish()
    }
}

impl ExportedCatalogProviderList {
    pub fn new(
        inner: Arc<dyn CatalogProviderList>,
        session: Arc<dyn Session>,
        runtime: Arc<RuntimeHandle>,
    ) -> Self {
        Self {
            inner,
            session,
            runtime,
        }
    }
}

impl From<ExportedCatalogProviderList> for SedonaCCatalogProviderList {
    fn from(value: ExportedCatalogProviderList) -> Self {
        Self {
            get_property_schema: Some(c_catalog_list_property_schema),
            get_property: Some(c_catalog_list_property),
            catalog: Some(c_catalog_list_catalog),
            register_catalog: Some(c_catalog_list_register),
            reserved: null_mut(),
            release: Some(c_catalog_list_release),
            private_data: Box::into_raw(Box::new(value)).cast(),
        }
    }
}

unsafe extern "C" fn c_catalog_list_property_schema(
    _self_: *const SedonaCCatalogProviderList,
    _property: *const c_char,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    write_utf8_property_schema(out, err)
}

unsafe extern "C" fn c_catalog_list_property(
    self_: *const SedonaCCatalogProviderList,
    property: *const c_char,
    _args: *const c_char,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedCatalogProviderList);
    match cstr_from_ptr_or_empty(property).as_ref() {
        "catalog_names" => write_json_property(&exported.inner.catalog_names(), out, err),
        property => {
            set_ffi_error!(err, "Unknown catalog list property: {}", property);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_catalog_list_catalog(
    self_: *const SedonaCCatalogProviderList,
    name: *const c_char,
    out: *mut SedonaCCatalogProvider,
    _err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedCatalogProviderList);
    let result = exported
        .inner
        .catalog(&cstr_from_ptr_or_empty(name))
        .map(|inner| {
            ExportedCatalogProvider::new(inner, exported.session.clone(), exported.runtime.clone())
                .into()
        })
        .unwrap_or_default();
    std::ptr::write(out, result);
    ERRNO_OK
}

unsafe extern "C" fn c_catalog_list_register(
    self_: *const SedonaCCatalogProviderList,
    name: *const c_char,
    catalog: *mut SedonaCCatalogProvider,
    out: *mut SedonaCCatalogProvider,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedCatalogProviderList);
    if catalog.is_null() {
        set_ffi_error!(err, "Catalog provider pointer is null");
        return libc::EINVAL;
    }
    // Move the provider out and invalidate the caller's copy, following the
    // Arrow C Data Interface ownership-transfer convention.
    let catalog = std::ptr::replace(catalog, SedonaCCatalogProvider::default());
    let imported = match ImportedCatalogProvider::try_new(
        catalog,
        exported.session.clone(),
        exported.runtime.clone(),
    ) {
        Ok(provider) => Arc::new(provider) as Arc<dyn CatalogProvider>,
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            return libc::EINVAL;
        }
    };
    let result = exported
        .inner
        .register_catalog(cstr_from_ptr_or_empty(name).into_owned(), imported)
        .map(|inner| {
            ExportedCatalogProvider::new(inner, exported.session.clone(), exported.runtime.clone())
                .into()
        })
        .unwrap_or_default();
    std::ptr::write(out, result);
    ERRNO_OK
}

unsafe extern "C" fn c_catalog_list_release(self_: *mut SedonaCCatalogProviderList) {
    let this = &mut *self_;
    if !this.private_data.is_null() {
        drop(Box::from_raw(
            this.private_data as *mut ExportedCatalogProviderList,
        ));
        this.private_data = null_mut();
    }
    this.release = None;
}

/// Imports a [`SedonaCCatalogProviderList`] as a DataFusion catalog list.
pub struct ImportedCatalogProviderList {
    inner: SedonaCCatalogProviderList,
    session: Arc<dyn Session>,
    runtime: Arc<RuntimeHandle>,
}

impl Debug for ImportedCatalogProviderList {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedCatalogProviderList").finish()
    }
}

impl ImportedCatalogProviderList {
    pub fn try_new(
        inner: SedonaCCatalogProviderList,
        session: Arc<dyn Session>,
        runtime: Arc<RuntimeHandle>,
    ) -> Result<Self> {
        if inner.release.is_none() {
            return sedona_common::sedona_internal_err!(
                "SedonaCCatalogProviderList does not have a release callback"
            );
        }
        if inner.get_property_schema.is_none()
            || inner.get_property.is_none()
            || inner.catalog.is_none()
        {
            return sedona_common::sedona_internal_err!(
                "SedonaCCatalogProviderList is missing a required callback"
            );
        }
        Ok(Self {
            inner,
            session,
            runtime,
        })
    }

    /// Register a catalog, preserving any error returned by the FFI callback.
    pub fn try_register_catalog(
        &self,
        name: String,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Result<Option<Arc<dyn CatalogProvider>>> {
        let Some(callback) = self.inner.register_catalog else {
            return not_impl_err!(
                "Registering catalogs is not supported by the foreign catalog list"
            );
        };
        let name = c_string(name)?;
        let mut input =
            ExportedCatalogProvider::new(catalog, self.session.clone(), self.runtime.clone())
                .into();
        let mut out = SedonaCCatalogProvider::default();
        let mut error = SedonaCError::default();
        let code =
            unsafe { callback(&self.inner, name.as_ptr(), &mut input, &mut out, &mut error) };
        if code != ERRNO_OK {
            return sedona_common::sedona_internal_err!("Failed to register catalog: {error}");
        }
        optional_catalog(out, self.session.clone(), self.runtime.clone())
    }

    /// Return catalog names, preserving any property error from FFI.
    pub fn try_catalog_names(&self) -> Result<Vec<String>> {
        let callback = self.inner.get_property.expect("validated in try_new");
        let schema_callback = self
            .inner
            .get_property_schema
            .expect("validated in try_new");
        call_get_json_property_impl(
            "catalog_names",
            "SedonaCCatalogProviderList",
            None::<&()>,
            |property, out, err| unsafe { schema_callback(&self.inner, property, out, err) },
            |property, args, out, err| unsafe { callback(&self.inner, property, args, out, err) },
        )
    }

    /// Look up a catalog, preserving any error returned by the FFI callback.
    pub fn try_catalog(&self, name: &str) -> Result<Option<Arc<dyn CatalogProvider>>> {
        let callback = self.inner.catalog.expect("validated in try_new");
        let name = c_string(name)?;
        let mut out = SedonaCCatalogProvider::default();
        let mut error = SedonaCError::default();
        let code = unsafe { callback(&self.inner, name.as_ptr(), &mut out, &mut error) };
        if code != ERRNO_OK {
            return sedona_common::sedona_internal_err!("Failed to get catalog: {error}");
        }
        optional_catalog(out, self.session.clone(), self.runtime.clone())
    }
}

impl CatalogProviderList for ImportedCatalogProviderList {
    fn register_catalog(
        &self,
        name: String,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Option<Arc<dyn CatalogProvider>> {
        self.try_register_catalog(name, catalog).ok().flatten()
    }

    fn catalog_names(&self) -> Vec<String> {
        self.try_catalog_names().unwrap_or_default()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn CatalogProvider>> {
        self.try_catalog(name).ok().flatten()
    }
}

/// Exports a [`CatalogProvider`] through [`SedonaCCatalogProvider`].
pub struct ExportedCatalogProvider {
    inner: Arc<dyn CatalogProvider>,
    session: Arc<dyn Session>,
    runtime: Arc<RuntimeHandle>,
}

impl Debug for ExportedCatalogProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedCatalogProvider")
            .field("inner", &self.inner)
            .finish()
    }
}

impl ExportedCatalogProvider {
    pub fn new(
        inner: Arc<dyn CatalogProvider>,
        session: Arc<dyn Session>,
        runtime: Arc<RuntimeHandle>,
    ) -> Self {
        Self {
            inner,
            session,
            runtime,
        }
    }
}

impl From<ExportedCatalogProvider> for SedonaCCatalogProvider {
    fn from(value: ExportedCatalogProvider) -> Self {
        Self {
            get_property_schema: Some(c_catalog_property_schema),
            get_property: Some(c_catalog_property),
            schema: Some(c_catalog_schema),
            register_schema: Some(c_catalog_register_schema),
            deregister_schema: Some(c_catalog_deregister_schema),
            reserved: null_mut(),
            release: Some(c_catalog_release),
            private_data: Box::into_raw(Box::new(value)).cast(),
        }
    }
}

unsafe extern "C" fn c_catalog_property_schema(
    _self_: *const SedonaCCatalogProvider,
    _property: *const c_char,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    write_utf8_property_schema(out, err)
}

unsafe extern "C" fn c_catalog_property(
    self_: *const SedonaCCatalogProvider,
    property: *const c_char,
    _args: *const c_char,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedCatalogProvider);
    match cstr_from_ptr_or_empty(property).as_ref() {
        "schema_names" => write_json_property(&exported.inner.schema_names(), out, err),
        property => {
            set_ffi_error!(err, "Unknown catalog property: {}", property);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_catalog_schema(
    self_: *const SedonaCCatalogProvider,
    name: *const c_char,
    out: *mut SedonaCSchemaProvider,
    _err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedCatalogProvider);
    let result = exported
        .inner
        .schema(&cstr_from_ptr_or_empty(name))
        .map(|inner| {
            ExportedSchemaProvider::new(inner, exported.session.clone(), exported.runtime.clone())
                .into()
        })
        .unwrap_or_default();
    std::ptr::write(out, result);
    ERRNO_OK
}

unsafe extern "C" fn c_catalog_register_schema(
    self_: *const SedonaCCatalogProvider,
    name: *const c_char,
    schema: *mut SedonaCSchemaProvider,
    out: *mut SedonaCSchemaProvider,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedCatalogProvider);
    if schema.is_null() {
        set_ffi_error!(err, "Schema provider pointer is null");
        return libc::EINVAL;
    }
    let schema = std::ptr::replace(schema, SedonaCSchemaProvider::default());
    let imported = match ImportedSchemaProvider::try_new(
        schema,
        exported.session.clone(),
        exported.runtime.clone(),
    ) {
        Ok(provider) => Arc::new(provider) as Arc<dyn SchemaProvider>,
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            return libc::EINVAL;
        }
    };
    match exported
        .inner
        .register_schema(&cstr_from_ptr_or_empty(name), imported)
    {
        Ok(result) => {
            let result = result
                .map(|inner| {
                    ExportedSchemaProvider::new(
                        inner,
                        exported.session.clone(),
                        exported.runtime.clone(),
                    )
                    .into()
                })
                .unwrap_or_default();
            std::ptr::write(out, result);
            ERRNO_OK
        }
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_catalog_deregister_schema(
    self_: *const SedonaCCatalogProvider,
    name: *const c_char,
    cascade: bool,
    out: *mut SedonaCSchemaProvider,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedCatalogProvider);
    match exported
        .inner
        .deregister_schema(&cstr_from_ptr_or_empty(name), cascade)
    {
        Ok(result) => {
            let result = result
                .map(|inner| {
                    ExportedSchemaProvider::new(
                        inner,
                        exported.session.clone(),
                        exported.runtime.clone(),
                    )
                    .into()
                })
                .unwrap_or_default();
            std::ptr::write(out, result);
            ERRNO_OK
        }
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_catalog_release(self_: *mut SedonaCCatalogProvider) {
    let this = &mut *self_;
    if !this.private_data.is_null() {
        drop(Box::from_raw(
            this.private_data as *mut ExportedCatalogProvider,
        ));
        this.private_data = null_mut();
    }
    this.release = None;
}

/// Imports a [`SedonaCCatalogProvider`] as a DataFusion catalog.
pub struct ImportedCatalogProvider {
    inner: SedonaCCatalogProvider,
    session: Arc<dyn Session>,
    runtime: Arc<RuntimeHandle>,
}

impl Debug for ImportedCatalogProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedCatalogProvider").finish()
    }
}

impl ImportedCatalogProvider {
    pub fn try_new(
        inner: SedonaCCatalogProvider,
        session: Arc<dyn Session>,
        runtime: Arc<RuntimeHandle>,
    ) -> Result<Self> {
        if inner.release.is_none() {
            return sedona_common::sedona_internal_err!(
                "SedonaCCatalogProvider does not have a release callback"
            );
        }
        if inner.get_property_schema.is_none()
            || inner.get_property.is_none()
            || inner.schema.is_none()
        {
            return sedona_common::sedona_internal_err!(
                "SedonaCCatalogProvider is missing a required callback"
            );
        }
        Ok(Self {
            inner,
            session,
            runtime,
        })
    }

    /// Return schema names, preserving any property error from FFI.
    pub fn try_schema_names(&self) -> Result<Vec<String>> {
        let callback = self.inner.get_property.expect("validated in try_new");
        let schema_callback = self
            .inner
            .get_property_schema
            .expect("validated in try_new");
        call_get_json_property_impl(
            "schema_names",
            "SedonaCCatalogProvider",
            None::<&()>,
            |property, out, err| unsafe { schema_callback(&self.inner, property, out, err) },
            |property, args, out, err| unsafe { callback(&self.inner, property, args, out, err) },
        )
    }

    /// Look up a schema, preserving any error returned by the FFI callback.
    pub fn try_schema(&self, name: &str) -> Result<Option<Arc<dyn SchemaProvider>>> {
        let callback = self.inner.schema.expect("validated in try_new");
        let name = c_string(name)?;
        let mut out = SedonaCSchemaProvider::default();
        let mut error = SedonaCError::default();
        let code = unsafe { callback(&self.inner, name.as_ptr(), &mut out, &mut error) };
        if code != ERRNO_OK {
            return sedona_common::sedona_internal_err!("Failed to get schema: {error}");
        }
        optional_schema(out, self.session.clone(), self.runtime.clone())
    }
}

impl CatalogProvider for ImportedCatalogProvider {
    fn schema_names(&self) -> Vec<String> {
        self.try_schema_names().unwrap_or_default()
    }
    fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
        self.try_schema(name).ok().flatten()
    }
    fn register_schema(
        &self,
        name: &str,
        schema: Arc<dyn SchemaProvider>,
    ) -> Result<Option<Arc<dyn SchemaProvider>>> {
        let Some(callback) = self.inner.register_schema else {
            return not_impl_err!("Registering schemas is not supported by the foreign catalog");
        };
        let name = c_string(name)?;
        let mut input =
            ExportedSchemaProvider::new(schema, self.session.clone(), self.runtime.clone()).into();
        let mut out = SedonaCSchemaProvider::default();
        let mut error = SedonaCError::default();
        let code =
            unsafe { callback(&self.inner, name.as_ptr(), &mut input, &mut out, &mut error) };
        if code != ERRNO_OK {
            return sedona_common::sedona_internal_err!("Failed to register schema: {error}");
        }
        optional_schema(out, self.session.clone(), self.runtime.clone())
    }
    fn deregister_schema(
        &self,
        name: &str,
        cascade: bool,
    ) -> Result<Option<Arc<dyn SchemaProvider>>> {
        let Some(callback) = self.inner.deregister_schema else {
            return not_impl_err!("Deregistering schemas is not supported by the foreign catalog");
        };
        let name = c_string(name)?;
        let mut out = SedonaCSchemaProvider::default();
        let mut error = SedonaCError::default();
        let code = unsafe { callback(&self.inner, name.as_ptr(), cascade, &mut out, &mut error) };
        if code != ERRNO_OK {
            return sedona_common::sedona_internal_err!("Failed to deregister schema: {error}");
        }
        optional_schema(out, self.session.clone(), self.runtime.clone())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TableExistArgs {
    name: String,
}

/// Exports a [`SchemaProvider`] through [`SedonaCSchemaProvider`].
pub struct ExportedSchemaProvider {
    inner: Arc<dyn SchemaProvider>,
    session: Arc<dyn Session>,
    runtime: Arc<RuntimeHandle>,
}

impl Debug for ExportedSchemaProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedSchemaProvider")
            .field("inner", &self.inner)
            .finish()
    }
}

impl ExportedSchemaProvider {
    pub fn new(
        inner: Arc<dyn SchemaProvider>,
        session: Arc<dyn Session>,
        runtime: Arc<RuntimeHandle>,
    ) -> Self {
        Self {
            inner,
            session,
            runtime,
        }
    }

    fn table(&self, name: String) -> Result<Option<Arc<dyn TableProvider>>> {
        let inner = self.inner.clone();
        let runtime = self.runtime.clone();
        std::thread::spawn(move || runtime.handle().block_on(inner.table(&name)))
            .join()
            .map_err(|error| {
                datafusion_common::DataFusionError::External(
                    format!("Table lookup thread panicked: {error:?}").into(),
                )
            })?
    }
}

impl From<ExportedSchemaProvider> for SedonaCSchemaProvider {
    fn from(value: ExportedSchemaProvider) -> Self {
        Self {
            get_property_schema: Some(c_schema_property_schema),
            get_property: Some(c_schema_property),
            table: Some(c_schema_table),
            register_table: Some(c_schema_register_table),
            deregister_table: Some(c_schema_deregister_table),
            reserved: null_mut(),
            release: Some(c_schema_release),
            private_data: Box::into_raw(Box::new(value)).cast(),
        }
    }
}

unsafe extern "C" fn c_schema_property_schema(
    _self_: *const SedonaCSchemaProvider,
    _property: *const c_char,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    write_utf8_property_schema(out, err)
}

unsafe extern "C" fn c_schema_property(
    self_: *const SedonaCSchemaProvider,
    property: *const c_char,
    args: *const c_char,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedSchemaProvider);
    match cstr_from_ptr_or_empty(property).as_ref() {
        "owner_name" => write_json_property(&exported.inner.owner_name(), out, err),
        "table_names" => write_json_property(&exported.inner.table_names(), out, err),
        "table_exist" => match parse_json_c_args::<TableExistArgs>(args) {
            Ok(args) => write_json_property(&exported.inner.table_exist(&args.name), out, err),
            Err(error) => {
                set_ffi_error!(err, "Failed to parse table_exist arguments: {}", error);
                libc::EINVAL
            }
        },
        property => {
            set_ffi_error!(err, "Unknown schema property: {}", property);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_schema_table(
    self_: *const SedonaCSchemaProvider,
    name: *const c_char,
    out: *mut SedonaCTableProvider,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedSchemaProvider);
    match exported.table(cstr_from_ptr_or_empty(name).into_owned()) {
        Ok(result) => {
            let result = result
                .map(|inner| {
                    ExportedTableProvider::new(
                        inner,
                        exported.session.clone(),
                        exported.runtime.clone(),
                    )
                    .into()
                })
                .unwrap_or_default();
            std::ptr::write(out, result);
            ERRNO_OK
        }
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_schema_register_table(
    self_: *const SedonaCSchemaProvider,
    name: *const c_char,
    table: *mut SedonaCTableProvider,
    out: *mut SedonaCTableProvider,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedSchemaProvider);
    if table.is_null() {
        set_ffi_error!(err, "Table provider pointer is null");
        return libc::EINVAL;
    }
    let table = std::ptr::replace(table, SedonaCTableProvider::default());
    let imported = match ImportedTableProvider::try_new(table) {
        Ok(provider) => Arc::new(provider) as Arc<dyn TableProvider>,
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            return libc::EINVAL;
        }
    };
    match exported
        .inner
        .register_table(cstr_from_ptr_or_empty(name).into_owned(), imported)
    {
        Ok(result) => {
            let result = result
                .map(|inner| {
                    ExportedTableProvider::new(
                        inner,
                        exported.session.clone(),
                        exported.runtime.clone(),
                    )
                    .into()
                })
                .unwrap_or_default();
            std::ptr::write(out, result);
            ERRNO_OK
        }
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_schema_deregister_table(
    self_: *const SedonaCSchemaProvider,
    name: *const c_char,
    out: *mut SedonaCTableProvider,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*self_).private_data as *const ExportedSchemaProvider);
    match exported
        .inner
        .deregister_table(&cstr_from_ptr_or_empty(name))
    {
        Ok(result) => {
            let result = result
                .map(|inner| {
                    ExportedTableProvider::new(
                        inner,
                        exported.session.clone(),
                        exported.runtime.clone(),
                    )
                    .into()
                })
                .unwrap_or_default();
            std::ptr::write(out, result);
            ERRNO_OK
        }
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_schema_release(self_: *mut SedonaCSchemaProvider) {
    let this = &mut *self_;
    if !this.private_data.is_null() {
        drop(Box::from_raw(
            this.private_data as *mut ExportedSchemaProvider,
        ));
        this.private_data = null_mut();
    }
    this.release = None;
}

/// Imports a [`SedonaCSchemaProvider`] as a DataFusion schema.
pub struct ImportedSchemaProvider {
    inner: SedonaCSchemaProvider,
    owner_name: Option<String>,
    session: Arc<dyn Session>,
    runtime: Arc<RuntimeHandle>,
}

impl Debug for ImportedSchemaProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedSchemaProvider").finish()
    }
}

impl ImportedSchemaProvider {
    pub fn try_new(
        inner: SedonaCSchemaProvider,
        session: Arc<dyn Session>,
        runtime: Arc<RuntimeHandle>,
    ) -> Result<Self> {
        if inner.release.is_none() {
            return sedona_common::sedona_internal_err!(
                "SedonaCSchemaProvider does not have a release callback"
            );
        }
        let Some(callback) = inner.get_property else {
            return sedona_common::sedona_internal_err!(
                "SedonaCSchemaProvider does not have get_property"
            );
        };
        if inner.get_property_schema.is_none() || inner.table.is_none() {
            return sedona_common::sedona_internal_err!(
                "SedonaCSchemaProvider is missing a required callback"
            );
        }
        let owner_name = call_get_json_property_impl(
            "owner_name",
            "SedonaCSchemaProvider",
            None::<&()>,
            |property, out, err| unsafe {
                inner.get_property_schema.expect("validated above")(&inner, property, out, err)
            },
            |property, args, out, err| unsafe { callback(&inner, property, args, out, err) },
        )?;
        Ok(Self {
            inner,
            owner_name,
            session,
            runtime,
        })
    }

    fn property<T: DeserializeOwned>(&self, property: &str) -> Result<T> {
        let callback = self.inner.get_property.expect("validated in try_new");
        let schema_callback = self
            .inner
            .get_property_schema
            .expect("validated in try_new");
        call_get_json_property_impl(
            property,
            "SedonaCSchemaProvider",
            None::<&()>,
            |property, out, err| unsafe { schema_callback(&self.inner, property, out, err) },
            |property, args, out, err| unsafe { callback(&self.inner, property, args, out, err) },
        )
    }

    /// Return table names, preserving any property error from FFI.
    pub fn try_table_names(&self) -> Result<Vec<String>> {
        self.property("table_names")
    }

    /// Test whether a table exists, preserving any property error from FFI.
    pub fn try_table_exist(&self, name: &str) -> Result<bool> {
        let callback = self.inner.get_property.expect("validated in try_new");
        let schema_callback = self
            .inner
            .get_property_schema
            .expect("validated in try_new");
        call_get_json_property_impl(
            "table_exist",
            "SedonaCSchemaProvider",
            Some(&TableExistArgs {
                name: name.to_owned(),
            }),
            |property, out, err| unsafe { schema_callback(&self.inner, property, out, err) },
            |property, args, out, err| unsafe { callback(&self.inner, property, args, out, err) },
        )
    }
}

#[async_trait]
impl SchemaProvider for ImportedSchemaProvider {
    fn owner_name(&self) -> Option<&str> {
        self.owner_name.as_deref()
    }
    fn table_names(&self) -> Vec<String> {
        self.try_table_names().unwrap_or_default()
    }
    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        let callback = self.inner.table.expect("validated in try_new");
        let name = c_string(name)?;
        let mut out = SedonaCTableProvider::default();
        let mut error = SedonaCError::default();
        let code = unsafe { callback(&self.inner, name.as_ptr(), &mut out, &mut error) };
        if code != ERRNO_OK {
            return sedona_common::sedona_internal_err!("Failed to get table: {error}");
        }
        optional_table(out)
    }
    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> Result<Option<Arc<dyn TableProvider>>> {
        let Some(callback) = self.inner.register_table else {
            return not_impl_err!("Registering tables is not supported by the foreign schema");
        };
        let name = c_string(name)?;
        let mut input =
            ExportedTableProvider::new(table, self.session.clone(), self.runtime.clone()).into();
        let mut out = SedonaCTableProvider::default();
        let mut error = SedonaCError::default();
        let code =
            unsafe { callback(&self.inner, name.as_ptr(), &mut input, &mut out, &mut error) };
        if code != ERRNO_OK {
            return sedona_common::sedona_internal_err!("Failed to register table: {error}");
        }
        optional_table(out)
    }
    fn deregister_table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        let Some(callback) = self.inner.deregister_table else {
            return not_impl_err!("Deregistering tables is not supported by the foreign schema");
        };
        let name = c_string(name)?;
        let mut out = SedonaCTableProvider::default();
        let mut error = SedonaCError::default();
        let code = unsafe { callback(&self.inner, name.as_ptr(), &mut out, &mut error) };
        if code != ERRNO_OK {
            return sedona_common::sedona_internal_err!("Failed to deregister table: {error}");
        }
        optional_table(out)
    }
    fn table_exist(&self, name: &str) -> bool {
        self.try_table_exist(name).unwrap_or(false)
    }
}

fn c_string(value: impl Into<Vec<u8>>) -> Result<CString> {
    CString::new(value).map_err(|error| {
        datafusion_common::DataFusionError::External(
            format!("Catalog name contains an interior NUL: {error}").into(),
        )
    })
}

fn optional_catalog(
    raw: SedonaCCatalogProvider,
    session: Arc<dyn Session>,
    runtime: Arc<RuntimeHandle>,
) -> Result<Option<Arc<dyn CatalogProvider>>> {
    if raw.release.is_none() {
        Ok(None)
    } else {
        Ok(Some(Arc::new(ImportedCatalogProvider::try_new(
            raw, session, runtime,
        )?)))
    }
}

fn optional_schema(
    raw: SedonaCSchemaProvider,
    session: Arc<dyn Session>,
    runtime: Arc<RuntimeHandle>,
) -> Result<Option<Arc<dyn SchemaProvider>>> {
    if raw.release.is_none() {
        Ok(None)
    } else {
        Ok(Some(Arc::new(ImportedSchemaProvider::try_new(
            raw, session, runtime,
        )?)))
    }
}

fn optional_table(raw: SedonaCTableProvider) -> Result<Option<Arc<dyn TableProvider>>> {
    if raw.release.is_none() {
        Ok(None)
    } else {
        Ok(Some(Arc::new(ImportedTableProvider::try_new(raw)?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Schema;
    use datafusion::catalog::{
        MemoryCatalogProvider, MemoryCatalogProviderList, MemorySchemaProvider,
    };
    use datafusion::datasource::empty::EmptyTable;
    use datafusion::prelude::SessionContext;

    unsafe extern "C" fn failing_register_catalog(
        _self_: *const SedonaCCatalogProviderList,
        _name: *const c_char,
        catalog: *mut SedonaCCatalogProvider,
        _out: *mut SedonaCCatalogProvider,
        error: *mut SedonaCError,
    ) -> c_int {
        if !catalog.is_null() {
            drop(std::ptr::replace(
                catalog,
                SedonaCCatalogProvider::default(),
            ));
        }
        crate::extension::write_ffi_error(error, "catalog registration failed");
        libc::EIO
    }

    fn runtime() -> Arc<RuntimeHandle> {
        Arc::new(RuntimeHandle::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        ))
    }

    fn empty_table() -> Arc<dyn TableProvider> {
        Arc::new(EmptyTable::new(Arc::new(Schema::empty())))
    }

    fn round_trip() -> (ImportedCatalogProviderList, Arc<RuntimeHandle>) {
        let schema = Arc::new(MemorySchemaProvider::new());
        schema
            .register_table("table_one".to_owned(), empty_table())
            .unwrap();

        let catalog = Arc::new(MemoryCatalogProvider::new());
        catalog.register_schema("schema_one", schema).unwrap();

        let catalogs = Arc::new(MemoryCatalogProviderList::new());
        catalogs.register_catalog("catalog_one".to_owned(), catalog);

        let context = SessionContext::new();
        let session = Arc::new(context.state());
        let runtime = runtime();
        let raw =
            ExportedCatalogProviderList::new(catalogs, session.clone(), runtime.clone()).into();
        let imported = ImportedCatalogProviderList::try_new(raw, session, runtime.clone()).unwrap();
        (imported, runtime)
    }

    #[test]
    fn round_trips_the_catalog_hierarchy() {
        let (catalogs, runtime) = round_trip();
        assert_eq!(catalogs.catalog_names(), vec!["catalog_one"]);

        let catalog = catalogs.catalog("catalog_one").unwrap();
        assert_eq!(catalog.schema_names(), vec!["schema_one"]);

        let schema = catalog.schema("schema_one").unwrap();
        assert_eq!(schema.table_names(), vec!["table_one"]);
        assert!(schema.table_exist("table_one"));
        assert!(!schema.table_exist("missing"));

        let table = runtime
            .block_on(schema.table("table_one"))
            .unwrap()
            .unwrap();
        assert_eq!(table.schema().fields().len(), 0);
        assert!(runtime.block_on(schema.table("missing")).unwrap().is_none());
    }

    #[test]
    fn round_trips_catalog_mutations_and_returned_providers() {
        let (catalogs, runtime) = round_trip();
        let catalog = catalogs.catalog("catalog_one").unwrap();
        let schema = catalog.schema("schema_one").unwrap();

        assert!(schema
            .register_table("table_two".to_owned(), empty_table())
            .unwrap()
            .is_none());
        assert!(schema.table_names().contains(&"table_two".to_owned()));
        assert!(schema.deregister_table("table_two").unwrap().is_some());

        assert!(catalog
            .register_schema("schema_two", Arc::new(MemorySchemaProvider::new()))
            .unwrap()
            .is_none());
        assert!(catalog.schema("schema_two").is_some());
        assert!(catalog
            .deregister_schema("schema_two", false)
            .unwrap()
            .is_some());

        assert!(catalogs
            .register_catalog(
                "catalog_two".to_owned(),
                Arc::new(MemoryCatalogProvider::new()),
            )
            .is_none());
        assert!(catalogs.catalog("catalog_two").is_some());

        let replaced = catalogs.register_catalog(
            "catalog_two".to_owned(),
            Arc::new(MemoryCatalogProvider::new()),
        );
        assert!(replaced.is_some());

        // Keep the runtime alive until all returned providers have been dropped.
        drop(replaced);
        drop(runtime);
    }

    #[test]
    fn rejects_invalid_raw_providers() {
        let context = SessionContext::new();
        let session = Arc::new(context.state());
        let runtime = runtime();

        assert!(ImportedCatalogProviderList::try_new(
            SedonaCCatalogProviderList::default(),
            session.clone(),
            runtime.clone(),
        )
        .is_err());
        assert!(ImportedCatalogProvider::try_new(
            SedonaCCatalogProvider::default(),
            session.clone(),
            runtime.clone(),
        )
        .is_err());
        assert!(ImportedSchemaProvider::try_new(
            SedonaCSchemaProvider::default(),
            session,
            runtime,
        )
        .is_err());
    }

    #[test]
    fn fallible_catalog_list_methods_preserve_ffi_errors() {
        let context = SessionContext::new();
        let session = Arc::new(context.state());
        let runtime = runtime();
        let mut raw: SedonaCCatalogProviderList = ExportedCatalogProviderList::new(
            Arc::new(MemoryCatalogProviderList::new()),
            session.clone(),
            runtime.clone(),
        )
        .into();
        raw.register_catalog = Some(failing_register_catalog);
        let imported = ImportedCatalogProviderList::try_new(raw, session, runtime).unwrap();

        let error = imported
            .try_register_catalog("catalog".to_owned(), Arc::new(MemoryCatalogProvider::new()))
            .unwrap_err();
        assert!(error.to_string().contains("catalog registration failed"));

        // The DataFusion trait cannot return this error and deliberately keeps
        // its Option-only behavior.
        assert!(imported
            .register_catalog("catalog".to_owned(), Arc::new(MemoryCatalogProvider::new()),)
            .is_none());
    }

    #[test]
    fn registration_callbacks_invalidate_transferred_inputs() {
        let context = SessionContext::new();
        let session = Arc::new(context.state());
        let runtime = runtime();
        let name = CString::new("registered").unwrap();
        let mut error = SedonaCError::default();

        let raw_list: SedonaCCatalogProviderList = ExportedCatalogProviderList::new(
            Arc::new(MemoryCatalogProviderList::new()),
            session.clone(),
            runtime.clone(),
        )
        .into();
        let mut input_catalog: SedonaCCatalogProvider = ExportedCatalogProvider::new(
            Arc::new(MemoryCatalogProvider::new()),
            session.clone(),
            runtime.clone(),
        )
        .into();
        let mut old_catalog = SedonaCCatalogProvider::default();
        let code = unsafe {
            raw_list.register_catalog.unwrap()(
                &raw_list,
                name.as_ptr(),
                &mut input_catalog,
                &mut old_catalog,
                &mut error,
            )
        };
        assert_eq!(code, ERRNO_OK);
        assert!(input_catalog.release.is_none());

        let raw_catalog: SedonaCCatalogProvider = ExportedCatalogProvider::new(
            Arc::new(MemoryCatalogProvider::new()),
            session.clone(),
            runtime.clone(),
        )
        .into();
        let mut input_schema: SedonaCSchemaProvider = ExportedSchemaProvider::new(
            Arc::new(MemorySchemaProvider::new()),
            session.clone(),
            runtime.clone(),
        )
        .into();
        let mut old_schema = SedonaCSchemaProvider::default();
        let code = unsafe {
            raw_catalog.register_schema.unwrap()(
                &raw_catalog,
                name.as_ptr(),
                &mut input_schema,
                &mut old_schema,
                &mut error,
            )
        };
        assert_eq!(code, ERRNO_OK);
        assert!(input_schema.release.is_none());

        let raw_schema: SedonaCSchemaProvider = ExportedSchemaProvider::new(
            Arc::new(MemorySchemaProvider::new()),
            session.clone(),
            runtime.clone(),
        )
        .into();
        let mut input_table: SedonaCTableProvider =
            ExportedTableProvider::new(empty_table(), session, runtime).into();
        let mut old_table = SedonaCTableProvider::default();
        let code = unsafe {
            raw_schema.register_table.unwrap()(
                &raw_schema,
                name.as_ptr(),
                &mut input_table,
                &mut old_table,
                &mut error,
            )
        };
        assert_eq!(code, ERRNO_OK);
        assert!(input_table.release.is_none());
    }
}
