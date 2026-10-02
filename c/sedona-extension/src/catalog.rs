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

//! Import and export the single async catalog interface through the Sedona C ABI.
use crate::{
    execution_plan::{ExportedExecutionPlan, ImportedSedonaCExec},
    extension::{
        SedonaCCatalogProviderList, SedonaCError, SedonaCExecutionPlan, SedonaCTableProvider,
    },
    runtime::RuntimeHandle,
    set_ffi_error,
    table_provider::{ExportedTableProvider, ImportedTableProvider},
    utils::{
        call_get_json_property_impl, cstr_from_ptr_or_empty, parse_json_c_args,
        write_json_property, write_utf8_property_schema, ERRNO_OK,
    },
};
use arrow_array::ffi::FFI_ArrowArray;
use arrow_schema::ffi::FFI_ArrowSchema;
use async_trait::async_trait;
use datafusion_catalog::{Session, TableProvider};
use datafusion_common::{DataFusionError, Result};
use datafusion_physical_plan::ExecutionPlan;
use sedona_catalog::{CatalogObject, CreateObjectOptions, DropObjectOptions, SedonaCatalogList};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    ffi::{c_char, c_int, CString},
    fmt::{Debug, Formatter},
    future::Future,
    ptr::null_mut,
    sync::Arc,
};

#[derive(Default, Serialize, Deserialize)]
struct ListArgs {
    prefix: Vec<String>,
    depth: Option<usize>,
}

#[derive(Default, Serialize, Deserialize)]
struct ObjectArgs<T> {
    identifier: Vec<String>,
    options: T,
}

fn refs(identifier: &[String]) -> Vec<&str> {
    identifier.iter().map(String::as_str).collect()
}

fn json_string(value: &impl Serialize) -> Result<CString> {
    let json = serde_json::to_vec(value).map_err(|e| DataFusionError::External(Box::new(e)))?;
    CString::new(json).map_err(|e| DataFusionError::External(Box::new(e)))
}

/// Exports a catalog and its producer-side execution context.
pub struct ExportedCatalogProviderList {
    inner: Arc<dyn SedonaCatalogList>,
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
    /// Create an export. Returned tables and plans retain their producer context.
    pub fn new(
        inner: Arc<dyn SedonaCatalogList>,
        session: Arc<dyn Session>,
        runtime: Arc<RuntimeHandle>,
    ) -> Self {
        Self {
            inner,
            session,
            runtime,
        }
    }

    // C callbacks are blocking. Use a fresh thread so direct C callers may also
    // invoke them from inside a Tokio runtime. Runtime::block_on drives timers
    // and I/O even when the producer uses a current-thread runtime.
    fn run<T: Send>(&self, future: impl Future<Output = Result<T>> + Send) -> Result<T> {
        std::thread::scope(|scope| scope.spawn(|| self.runtime.block_on(future)).join()).map_err(
            |e| DataFusionError::External(format!("Catalog callback panicked: {e:?}").into()),
        )?
    }

    fn export_plan(&self, plan: Arc<dyn ExecutionPlan>) -> SedonaCExecutionPlan {
        ExportedExecutionPlan::new(plan, self.session.task_ctx(), self.runtime.clone()).into()
    }
}

impl From<ExportedCatalogProviderList> for SedonaCCatalogProviderList {
    fn from(value: ExportedCatalogProviderList) -> Self {
        Self {
            get_property_schema: Some(get_property_schema),
            get_property: Some(get_property),
            table: Some(table),
            create_object: Some(create_object),
            drop_object: Some(drop_object),
            reserved: null_mut(),
            release: Some(release),
            private_data: Box::into_raw(Box::new(value)).cast(),
        }
    }
}

unsafe extern "C" fn get_property_schema(
    _raw: *const SedonaCCatalogProviderList,
    property: *const c_char,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    match cstr_from_ptr_or_empty(property).as_ref() {
        "name" | "list_identifiers" => write_utf8_property_schema(out, err),
        property => {
            set_ffi_error!(err, "Unknown catalog property: {}", property);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn get_property(
    raw: *const SedonaCCatalogProviderList,
    property: *const c_char,
    args: *const c_char,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    let property = cstr_from_ptr_or_empty(property);
    let exported = &*((*raw).private_data as *const ExportedCatalogProviderList);
    match property.as_ref() {
        "name" => write_json_property(&exported.inner.name(), out, err),
        "list_identifiers" => {
            let result = (|| {
                let args: ListArgs = parse_json_c_args(args)?;
                exported.run(
                    exported
                        .inner
                        .list_identifiers(&refs(&args.prefix), args.depth),
                )
            })();
            write_json_result_property(result, out, err)
        }
        property => {
            set_ffi_error!(err, "Unknown catalog property: {}", property);
            libc::EINVAL
        }
    }
}

unsafe fn write_json_result_property<T: Serialize>(
    result: Result<T>,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    match result {
        Ok(value) => write_json_property(&value, out, err),
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn table(
    raw: *const SedonaCCatalogProviderList,
    identifier: *const c_char,
    out: *mut SedonaCTableProvider,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*raw).private_data as *const ExportedCatalogProviderList);
    let result = (|| {
        let identifier: Vec<String> = parse_json_c_args(identifier)?;
        exported.run(exported.inner.table(&refs(&identifier)))
    })();
    match result {
        Ok(table) => {
            std::ptr::write(
                out,
                table
                    .map(|table| {
                        ExportedTableProvider::new(
                            table,
                            exported.session.clone(),
                            exported.runtime.clone(),
                        )
                        .into()
                    })
                    .unwrap_or_default(),
            );
            ERRNO_OK
        }
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn create_object(
    raw: *const SedonaCCatalogProviderList,
    args: *const c_char,
    input: *mut SedonaCExecutionPlan,
    out: *mut SedonaCExecutionPlan,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*raw).private_data as *const ExportedCatalogProviderList);
    let result = (|| {
        let args: ObjectArgs<CreateObjectOptions> = parse_json_c_args(args)?;
        let input = if input.is_null() {
            None
        } else {
            let raw = std::mem::take(&mut *input);
            Some(Arc::new(ImportedSedonaCExec::try_new(raw)?) as Arc<dyn ExecutionPlan>)
        };
        exported.run(exported.inner.create_object(
            exported.session.as_ref(),
            &refs(&args.identifier),
            &args.options,
            input,
        ))
    })();
    write_plan(exported, result, out, err)
}

unsafe extern "C" fn drop_object(
    raw: *const SedonaCCatalogProviderList,
    args: *const c_char,
    out: *mut SedonaCExecutionPlan,
    err: *mut SedonaCError,
) -> c_int {
    let exported = &*((*raw).private_data as *const ExportedCatalogProviderList);
    let result = (|| {
        let args: ObjectArgs<DropObjectOptions> = parse_json_c_args(args)?;
        exported.run(
            exported
                .inner
                .drop_object(&refs(&args.identifier), &args.options),
        )
    })();
    write_plan(exported, result, out, err)
}

unsafe fn write_plan(
    exported: &ExportedCatalogProviderList,
    result: Result<Arc<dyn ExecutionPlan>>,
    out: *mut SedonaCExecutionPlan,
    err: *mut SedonaCError,
) -> c_int {
    match result {
        Ok(plan) => {
            std::ptr::write(out, exported.export_plan(plan));
            ERRNO_OK
        }
        Err(error) => {
            set_ffi_error!(err, "{}", error);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn release(raw: *mut SedonaCCatalogProviderList) {
    let raw = &mut *raw;
    if !raw.private_data.is_null() {
        drop(Box::from_raw(
            raw.private_data as *mut ExportedCatalogProviderList,
        ));
        raw.private_data = null_mut();
    }
    raw.release = None;
}

/// Imports a catalog without retaining the consumer's session. The implementation
/// name is read once at import. Async operations run blocking C calls off the
/// executor and retain the raw object until completion even if cancelled.
pub struct ImportedCatalogProviderList {
    inner: Arc<SedonaCCatalogProviderList>,
    name: String,
    runtime: Arc<RuntimeHandle>,
}

impl Debug for ImportedCatalogProviderList {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedCatalogProviderList").finish()
    }
}

impl ImportedCatalogProviderList {
    /// Validate the required callbacks and cache the implementation name.
    pub fn try_new(inner: SedonaCCatalogProviderList, runtime: Arc<RuntimeHandle>) -> Result<Self> {
        if inner.release.is_none()
            || inner.get_property_schema.is_none()
            || inner.get_property.is_none()
            || inner.table.is_none()
            || inner.create_object.is_none()
            || inner.drop_object.is_none()
        {
            return datafusion_common::exec_err!(
                "SedonaCCatalogProviderList is missing a required callback"
            );
        }
        let name = call_get_json_property_impl(
            "name",
            "SedonaCCatalogProviderList",
            None::<&()>,
            |property, out, err| unsafe {
                inner.get_property_schema.unwrap()(&inner, property, out, err)
            },
            |property, args, out, err| unsafe {
                inner.get_property.unwrap()(&inner, property, args, out, err)
            },
        )?;
        Ok(Self {
            inner: Arc::new(inner),
            name,
            runtime,
        })
    }

    async fn call<T: Send + 'static>(
        &self,
        callback: impl FnOnce(Arc<SedonaCCatalogProviderList>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let inner = self.inner.clone();
        self.runtime
            .spawn_blocking(move || callback(inner))
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?
    }

    async fn property<T: DeserializeOwned + Send + 'static, A: Serialize + Send + 'static>(
        &self,
        property: &'static str,
        args: Option<A>,
    ) -> Result<T> {
        self.call(move |raw| {
            call_get_json_property_impl(
                property,
                "SedonaCCatalogProviderList",
                args.as_ref(),
                |property, out, err| unsafe {
                    raw.get_property_schema.unwrap()(raw.as_ref(), property, out, err)
                },
                |property, args, out, err| unsafe {
                    raw.get_property.unwrap()(raw.as_ref(), property, args, out, err)
                },
            )
        })
        .await
    }
}

#[async_trait]
impl SedonaCatalogList for ImportedCatalogProviderList {
    fn name(&self) -> &str {
        &self.name
    }

    async fn list_identifiers(
        &self,
        prefix: &[&str],
        depth: Option<usize>,
    ) -> Result<Vec<CatalogObject>> {
        let args = ListArgs {
            prefix: prefix.iter().map(|s| s.to_string()).collect(),
            depth,
        };
        self.property("list_identifiers", Some(args)).await
    }

    async fn table(&self, identifier: &[&str]) -> Result<Option<Arc<dyn TableProvider>>> {
        let identifier = json_string(&identifier)?;
        self.call(move |raw| {
            let mut out = SedonaCTableProvider::default();
            let mut err = SedonaCError::default();
            let code = unsafe {
                raw.table.unwrap()(raw.as_ref(), identifier.as_ptr(), &mut out, &mut err)
            };
            check_error(code, &err)?;
            if out.release.is_none() {
                Ok(None)
            } else {
                Ok(Some(
                    Arc::new(ImportedTableProvider::try_new(out)?) as Arc<dyn TableProvider>
                ))
            }
        })
        .await
    }

    async fn create_object(
        &self,
        session: &dyn Session,
        identifier: &[&str],
        options: &CreateObjectOptions,
        input: Option<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let args = json_string(&ObjectArgs {
            identifier: identifier.iter().map(|s| s.to_string()).collect(),
            options,
        })?;
        let mut input = input.map(|plan| {
            ExportedExecutionPlan::new(plan, session.task_ctx(), self.runtime.clone()).into()
        });
        self.call(move |raw| {
            let mut out = SedonaCExecutionPlan::default();
            let mut err = SedonaCError::default();
            let code = unsafe {
                raw.create_object.unwrap()(
                    raw.as_ref(),
                    args.as_ptr(),
                    input.as_mut().map_or(null_mut(), |input| input),
                    &mut out,
                    &mut err,
                )
            };
            check_error(code, &err)?;
            import_plan(out)
        })
        .await
    }

    async fn drop_object(
        &self,
        identifier: &[&str],
        options: &DropObjectOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let args = json_string(&ObjectArgs {
            identifier: identifier.iter().map(|s| s.to_string()).collect(),
            options,
        })?;
        self.call(move |raw| {
            let mut out = SedonaCExecutionPlan::default();
            let mut err = SedonaCError::default();
            let code = unsafe {
                raw.drop_object.unwrap()(raw.as_ref(), args.as_ptr(), &mut out, &mut err)
            };
            check_error(code, &err)?;
            import_plan(out)
        })
        .await
    }
}

fn check_error(code: c_int, err: &SedonaCError) -> Result<()> {
    if code == ERRNO_OK {
        Ok(())
    } else {
        datafusion_common::exec_err!("Catalog callback failed ({code}): {err}")
    }
}

fn import_plan(raw: SedonaCExecutionPlan) -> Result<Arc<dyn ExecutionPlan>> {
    if raw.release.is_none() {
        return datafusion_common::exec_err!("Catalog callback returned no execution plan");
    }
    Ok(Arc::new(ImportedSedonaCExec::try_new(raw)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Schema;
    use datafusion::{datasource::empty::EmptyTable, prelude::SessionContext};
    use datafusion_common::tree_node::TreeNodeRecursion;
    use datafusion_execution::TaskContext;
    use datafusion_physical_plan::{
        empty::EmptyExec, placeholder_row::PlaceholderRowExec, DisplayAs, DisplayFormatType,
        PhysicalExpr, PlanProperties, SendableRecordBatchStream,
    };
    use sedona_catalog::{CatalogObjectType, CreateMode};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };

    type DropAction = Box<dyn FnOnce() -> Result<()> + Send>;

    struct MutationExec {
        inner: Arc<dyn ExecutionPlan>,
        drop_action: Mutex<Option<DropAction>>,
    }

    impl MutationExec {
        fn new(drop_action: impl FnOnce() -> Result<()> + Send + 'static) -> Self {
            Self {
                inner: Arc::new(EmptyExec::new(Arc::new(Schema::empty()))),
                drop_action: Mutex::new(Some(Box::new(drop_action))),
            }
        }
    }

    impl Debug for MutationExec {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MutationExec").finish()
        }
    }

    impl DisplayAs for MutationExec {
        fn fmt_as(
            &self,
            _t: DisplayFormatType,
            f: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            write!(f, "MutationExec")
        }
    }

    impl ExecutionPlan for MutationExec {
        fn apply_expressions(
            &self,
            f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            self.inner.apply_expressions(f)
        }

        fn name(&self) -> &str {
            "MutationExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.inner.properties()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            if !children.is_empty() {
                return datafusion_common::internal_err!("MutationExec does not have children");
            }
            Ok(self)
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            if let Some(drop_action) = self.drop_action.lock().unwrap().take() {
                drop_action()?;
            }
            self.inner.execute(partition, context)
        }
    }

    #[derive(Debug, Default)]
    struct TestCatalog {
        calls: Mutex<Vec<String>>,
        mutations: Arc<AtomicUsize>,
        fail: bool,
    }

    #[async_trait]
    impl SedonaCatalogList for TestCatalog {
        fn name(&self) -> &str {
            "iceberg"
        }

        async fn list_identifiers(
            &self,
            prefix: &[&str],
            depth: Option<usize>,
        ) -> Result<Vec<CatalogObject>> {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            if self.fail {
                return datafusion_common::exec_err!("listing failed");
            }
            self.calls
                .lock()
                .unwrap()
                .push(format!("list:{prefix:?}:{depth:?}"));
            Ok(vec![CatalogObject {
                identifier: vec!["catalog".into(), "schema.with.dot".into(), "table".into()],
                object_type: CatalogObjectType::Table,
            }])
        }
        async fn table(&self, identifier: &[&str]) -> Result<Option<Arc<dyn TableProvider>>> {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            if self.fail {
                return datafusion_common::exec_err!("lookup failed");
            }
            self.calls
                .lock()
                .unwrap()
                .push(format!("table:{identifier:?}"));
            if identifier.last() == Some(&"missing") {
                return Ok(None);
            }
            Ok(Some(Arc::new(EmptyTable::new(Arc::new(Schema::empty())))))
        }
        async fn create_object(
            &self,
            _session: &dyn Session,
            identifier: &[&str],
            options: &CreateObjectOptions,
            input: Option<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            if self.fail {
                return datafusion_common::exec_err!("create failed");
            }
            self.calls.lock().unwrap().push(format!(
                "create:{identifier:?}:{options:?}:{}",
                input.is_some()
            ));
            if let Some(input) = input {
                return Ok(input);
            }
            let mutations = self.mutations.clone();
            Ok(Arc::new(MutationExec::new(move || {
                mutations.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })))
        }
        async fn drop_object(
            &self,
            identifier: &[&str],
            options: &DropObjectOptions,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            if self.fail {
                return datafusion_common::exec_err!("drop failed");
            }
            self.calls
                .lock()
                .unwrap()
                .push(format!("drop:{identifier:?}:{options:?}"));
            let mutations = self.mutations.clone();
            Ok(Arc::new(MutationExec::new(move || {
                mutations.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })))
        }
    }

    fn runtime() -> Arc<RuntimeHandle> {
        Arc::new(RuntimeHandle::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        ))
    }

    fn raw(catalog: Arc<TestCatalog>) -> SedonaCCatalogProviderList {
        ExportedCatalogProviderList::new(
            catalog,
            Arc::new(SessionContext::new().state()),
            runtime(),
        )
        .into()
    }

    #[tokio::test]
    async fn round_trip_async_operations_and_optional_plans() -> Result<()> {
        let catalog = Arc::new(TestCatalog::default());
        let imported = ImportedCatalogProviderList::try_new(raw(catalog.clone()), runtime())?;
        assert_eq!(imported.name(), "iceberg");
        let objects = imported.list_identifiers(&["catalog"], Some(2)).await?;
        assert_eq!(
            objects[0].identifier,
            ["catalog", "schema.with.dot", "table"]
        );
        assert_eq!(objects[0].object_type, CatalogObjectType::Table);
        imported.list_identifiers(&[], None).await?;
        imported.list_identifiers(&["catalog"], Some(0)).await?;
        assert!(imported
            .table(&["catalog", "schema.with.dot", "table"])
            .await?
            .is_some());
        assert!(imported.table(&["catalog", "missing"]).await?.is_none());
        // JSON preserves arbitrary literal path components, including NULs.
        imported.table(&["a.b", "quote\"", "nul\0"]).await?;
        let host = SessionContext::new();
        let state = host.state();
        let create_options = CreateObjectOptions {
            object_type: CatalogObjectType::Schema,
            mode: CreateMode::CreateOrIgnore,
            temporary: true,
            external: false,
            definition: Some("CREATE SCHEMA catalog.new_schema".into()),
        };
        let plan = imported
            .create_object(&state, &["catalog", "new_schema"], &create_options, None)
            .await?;
        assert_eq!(catalog.mutations.load(Ordering::SeqCst), 0);
        datafusion_physical_plan::collect(plan, state.task_ctx()).await?;
        assert_eq!(catalog.mutations.load(Ordering::SeqCst), 1);
        let input = Arc::new(PlaceholderRowExec::new(Arc::new(Schema::empty())));
        let plan = imported
            .create_object(
                &state,
                &["catalog", "table"],
                &CreateObjectOptions::default(),
                Some(input),
            )
            .await?;
        let batches = datafusion_physical_plan::collect(plan, state.task_ctx()).await?;
        assert_eq!(
            batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
            1
        );
        let drop_options = DropObjectOptions {
            object_type: CatalogObjectType::Schema,
            if_exists: true,
            cascade: true,
            purge: true,
        };
        let plan = imported
            .drop_object(&["catalog", "new_schema"], &drop_options)
            .await?;
        assert_eq!(catalog.mutations.load(Ordering::SeqCst), 1);
        datafusion_physical_plan::collect(plan, state.task_ctx()).await?;
        assert_eq!(catalog.mutations.load(Ordering::SeqCst), 2);
        let calls = catalog.calls.lock().unwrap();
        assert_eq!(calls[0], "list:[\"catalog\"]:Some(2)");
        assert!(calls
            .iter()
            .any(|call| call.contains("CreateOrIgnore") && call.contains("temporary: true")));
        assert!(calls.last().unwrap().contains("cascade: true, purge: true"));
        Ok(())
    }

    #[tokio::test]
    async fn ffi_errors_are_preserved_for_every_operation() -> Result<()> {
        let catalog = Arc::new(TestCatalog {
            fail: true,
            ..Default::default()
        });
        let imported = ImportedCatalogProviderList::try_new(raw(catalog), runtime())?;
        assert!(imported
            .list_identifiers(&[], None)
            .await
            .unwrap_err()
            .to_string()
            .contains("listing failed"));
        assert!(imported
            .table(&["table"])
            .await
            .unwrap_err()
            .to_string()
            .contains("lookup failed"));
        assert!(imported
            .create_object(
                &SessionContext::new().state(),
                &["table"],
                &CreateObjectOptions::default(),
                None
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("create failed"));
        assert!(imported
            .drop_object(&["table"], &DropObjectOptions::default())
            .await
            .unwrap_err()
            .to_string()
            .contains("drop failed"));
        Ok(())
    }

    #[test]
    fn callback_transfers_input_ownership_and_rejects_invalid_json() {
        let raw = raw(Arc::new(TestCatalog::default()));
        let host = SessionContext::new();
        let plan = Arc::new(PlaceholderRowExec::new(Arc::new(Schema::empty())));
        let mut input: SedonaCExecutionPlan =
            ExportedExecutionPlan::new(plan, host.task_ctx(), runtime()).into();
        let args = CString::new(r#"{"identifier":["catalog","table"],"options":{}}"#).unwrap();
        let mut out = SedonaCExecutionPlan::default();
        let mut err = SedonaCError::default();
        let code = unsafe {
            raw.create_object.unwrap()(&raw, args.as_ptr(), &mut input, &mut out, &mut err)
        };
        assert_eq!(code, ERRNO_OK, "{err}");
        assert!(input.release.is_none());
        assert!(out.release.is_some());
        let bad = CString::new("not json").unwrap();
        let mut out = SedonaCExecutionPlan::default();
        let code = unsafe { raw.drop_object.unwrap()(&raw, bad.as_ptr(), &mut out, &mut err) };
        assert_ne!(code, ERRNO_OK);
        assert!(out.release.is_none());
    }

    #[test]
    fn catalog_properties_reject_unknown_names() {
        let raw = raw(Arc::new(TestCatalog::default()));
        let mut err = SedonaCError::default();
        let mut schema = FFI_ArrowSchema::empty();
        let code = unsafe {
            raw.get_property_schema.unwrap()(&raw, c"unknown".as_ptr(), &mut schema, &mut err)
        };
        assert_eq!(code, libc::EINVAL);
        assert!(err
            .to_string()
            .contains("Unknown catalog property: unknown"));

        let mut out = FFI_ArrowArray::empty();
        let code = unsafe {
            raw.get_property.unwrap()(
                &raw,
                c"unknown".as_ptr(),
                std::ptr::null(),
                &mut out,
                &mut err,
            )
        };
        assert_eq!(code, libc::EINVAL);
        assert!(err
            .to_string()
            .contains("Unknown catalog property: unknown"));
    }

    #[tokio::test]
    async fn listing_uses_the_foreign_property_schema() -> Result<()> {
        unsafe extern "C" fn failing_schema(
            _raw: *const SedonaCCatalogProviderList,
            property: *const c_char,
            out: *mut FFI_ArrowSchema,
            err: *mut SedonaCError,
        ) -> c_int {
            if cstr_from_ptr_or_empty(property) == "name" {
                return write_utf8_property_schema(out, err);
            }
            set_ffi_error!(
                err,
                "Schema unavailable for {}",
                cstr_from_ptr_or_empty(property)
            );
            libc::EIO
        }

        let mut raw = raw(Arc::new(TestCatalog::default()));
        raw.get_property_schema = Some(failing_schema);
        let imported = ImportedCatalogProviderList::try_new(raw, runtime())?;
        let error = imported.list_identifiers(&[], None).await.unwrap_err();
        assert!(error
            .to_string()
            .contains("Schema unavailable for list_identifiers"));
        Ok(())
    }

    #[test]
    fn failed_name_import_releases_the_catalog() {
        unsafe extern "C" fn failing_property(
            _raw: *const SedonaCCatalogProviderList,
            _property: *const c_char,
            _args: *const c_char,
            _out: *mut FFI_ArrowArray,
            err: *mut SedonaCError,
        ) -> c_int {
            set_ffi_error!(err, "Name unavailable");
            libc::EIO
        }

        let catalog = Arc::new(TestCatalog::default());
        let weak = Arc::downgrade(&catalog);
        let mut raw = raw(catalog);
        raw.get_property = Some(failing_property);
        let error = ImportedCatalogProviderList::try_new(raw, runtime()).unwrap_err();
        assert!(error.to_string().contains("Name unavailable"));
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn invalid_callbacks_and_missing_output_are_rejected() {
        assert!(ImportedCatalogProviderList::try_new(
            SedonaCCatalogProviderList::default(),
            runtime()
        )
        .is_err());
        assert!(import_plan(SedonaCExecutionPlan::default()).is_err());
        let mut missing_create = raw(Arc::new(TestCatalog::default()));
        missing_create.create_object = None;
        assert!(ImportedCatalogProviderList::try_new(missing_create, runtime()).is_err());

        let mut missing_schema = raw(Arc::new(TestCatalog::default()));
        missing_schema.get_property_schema = None;
        assert!(ImportedCatalogProviderList::try_new(missing_schema, runtime()).is_err());
        let mut missing_property = raw(Arc::new(TestCatalog::default()));
        missing_property.get_property = None;
        assert!(ImportedCatalogProviderList::try_new(missing_property, runtime()).is_err());
    }

    #[tokio::test]
    async fn imported_catalog_does_not_retain_consumer_session() -> Result<()> {
        let imported =
            ImportedCatalogProviderList::try_new(raw(Arc::new(TestCatalog::default())), runtime())?;
        let host = SessionContext::new();
        let weak = host.state_weak_ref();
        let plan = imported
            .create_object(
                &host.state(),
                &["catalog"],
                &CreateObjectOptions {
                    object_type: CatalogObjectType::Catalog,
                    ..Default::default()
                },
                None,
            )
            .await?;
        drop(plan);
        drop(host);
        assert!(weak.upgrade().is_none());
        assert!(imported.table(&["table"]).await?.is_some());
        Ok(())
    }
}
