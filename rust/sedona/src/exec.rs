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
use crate::{context::SedonaContext, object_storage::register_object_store_and_config_extensions};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::prelude::DataFrame;
use datafusion::{error::Result, sql::parser::Statement};
use datafusion_common::{exec_err, SchemaReference, TableReference};
use datafusion_expr::{DdlStatement, LogicalPlan, TableType};
use datafusion_physical_plan::ExecutionPlan;
use sedona_catalog::{CatalogObjectType, CreateMode, CreateObjectOptions, DropObjectOptions};
use std::{fmt::Debug, sync::Arc};

/// Resolve tables asynchronously and configure stores required by SQL I/O.
pub(crate) async fn create_plan_from_sql(
    ctx: &SedonaContext,
    statement: Statement,
) -> Result<LogicalPlan> {
    let mut plan = crate::catalog_planner::statement_to_plan(ctx, statement).await?;
    if let LogicalPlan::Ddl(DdlStatement::CreateExternalTable(cmd)) = &plan {
        let state = ctx.ctx.state();
        let defaults = &state.config_options().catalog;
        let name = cmd
            .name
            .clone()
            .resolve(&defaults.default_catalog, &defaults.default_schema);
        if ctx
            .catalog_registry()
            .foreign_catalog(&name.catalog)
            .await?
            .is_none()
        {
            register_object_store_and_config_extensions(ctx, &cmd.location, &cmd.options).await?;
        }
    }
    if let LogicalPlan::Copy(copy_to) = &mut plan {
        register_object_store_and_config_extensions(ctx, &copy_to.output_url, &copy_to.options)
            .await?;
    }
    Ok(plan)
}

/// Route catalog DDL to one async extension. Built-in targets retain DataFusion behavior.
/// No existence checks or mutations happen here: the returned plan owns them.
pub(crate) async fn execute_sedona_catalog_ddl(
    ctx: &SedonaContext,
    plan: &LogicalPlan,
    statement: &Statement,
) -> Result<Option<DataFrame>> {
    let LogicalPlan::Ddl(ddl) = plan else {
        return Ok(None);
    };
    let state = ctx.ctx.state();
    let defaults = &state.config_options().catalog;
    let table_identifier = |name: &TableReference| {
        let resolved = name
            .clone()
            .resolve(&defaults.default_catalog, &defaults.default_schema);
        vec![
            resolved.catalog.to_string(),
            resolved.schema.to_string(),
            resolved.table.to_string(),
        ]
    };
    let mut create = CreateObjectOptions::default();
    let mut drop = None;
    let mut input = None;
    let identifier = match ddl {
        DdlStatement::CreateCatalog(cmd) => {
            create.object_type = CatalogObjectType::Catalog;
            create.mode = create_mode(cmd.if_not_exists, false)?;
            vec![cmd.catalog_name.clone()]
        }
        DdlStatement::CreateCatalogSchema(cmd) => {
            create.object_type = CatalogObjectType::Schema;
            create.mode = create_mode(cmd.if_not_exists, false)?;
            use datafusion_expr::sqlparser::ast::{SchemaName, Statement as SqlStatement};
            // The logical plan's schema_name has lost identifier quoting.
            let sql = match statement {
                Statement::Statement(sql) => Some(sql.as_ref()),
                _ => None,
            };
            let Some(SqlStatement::CreateSchema { schema_name, .. }) = sql else {
                return sedona_common::sedona_internal_err!(
                    "Expected CREATE SCHEMA statement for CreateCatalogSchema plan"
                );
            };
            let name = match schema_name {
                SchemaName::Simple(name) | SchemaName::NamedAuthorization(name, _) => {
                    name.to_string()
                }
                SchemaName::UnnamedAuthorization(name) => name.to_string(),
            };
            match TableReference::parse_str_normalized(
                &name,
                !state.config_options().sql_parser.enable_ident_normalization,
            ) {
                TableReference::Bare { table } => {
                    vec![defaults.default_catalog.clone(), table.to_string()]
                }
                TableReference::Partial { schema, table } => {
                    vec![schema.to_string(), table.to_string()]
                }
                _ => return exec_err!("Expected schema or catalog.schema: {name}"),
            }
        }
        DdlStatement::CreateMemoryTable(cmd) => {
            create.mode = create_mode(cmd.if_not_exists, cmd.or_replace)?;
            create.temporary = cmd.temporary;
            input = Some(cmd.input.as_ref());
            table_identifier(&cmd.name)
        }
        DdlStatement::CreateExternalTable(cmd) => {
            create.mode = create_mode(cmd.if_not_exists, cmd.or_replace)?;
            create.temporary = cmd.temporary;
            create.external = true;
            create.definition = cmd.definition.clone();
            table_identifier(&cmd.name)
        }
        DdlStatement::CreateView(cmd) => {
            create.object_type = CatalogObjectType::View;
            create.definition = cmd.definition.clone();
            create.mode = create_mode(false, cmd.or_replace)?;
            create.temporary = cmd.temporary;
            input = Some(cmd.input.as_ref());
            table_identifier(&cmd.name)
        }
        DdlStatement::CreateIndex(cmd) => {
            create.object_type = CatalogObjectType::Index;
            create.mode = create_mode(cmd.if_not_exists, false)?;
            // Indexes belong to a table. An absent final component requests an
            // implementation-defined index name.
            let mut identifier = table_identifier(&cmd.table);
            if let Some(name) = &cmd.name {
                identifier.push(name.clone());
            }
            identifier
        }
        DdlStatement::DropTable(cmd) => {
            drop = Some(DropObjectOptions {
                object_type: CatalogObjectType::Table,
                if_exists: cmd.if_exists,
                ..Default::default()
            });
            table_identifier(&cmd.name)
        }
        DdlStatement::DropView(cmd) => {
            drop = Some(DropObjectOptions {
                object_type: CatalogObjectType::View,
                if_exists: cmd.if_exists,
                ..Default::default()
            });
            table_identifier(&cmd.name)
        }
        DdlStatement::DropCatalogSchema(cmd) => {
            drop = Some(DropObjectOptions {
                object_type: CatalogObjectType::Schema,
                if_exists: cmd.if_exists,
                cascade: cmd.cascade,
                ..Default::default()
            });
            match &cmd.name {
                SchemaReference::Bare { schema } => {
                    vec![defaults.default_catalog.clone(), schema.to_string()]
                }
                SchemaReference::Full { catalog, schema } => {
                    vec![catalog.to_string(), schema.to_string()]
                }
            }
        }
        DdlStatement::CreateFunction(_) | DdlStatement::DropFunction(_) => return Ok(None),
    };
    // DataFusion's logical DropTable/DropView omit these SQL flags.
    if let (Some(options), Statement::Statement(sql)) = (&mut drop, statement) {
        if let datafusion_expr::sqlparser::ast::Statement::Drop { cascade, purge, .. } =
            sql.as_ref()
        {
            options.cascade = *cascade;
            options.purge = *purge;
        }
    }
    let registry = ctx.catalog_registry();
    let owner = registry.foreign_catalog(&identifier[0]).await?;
    let owner = if matches!(ddl, DdlStatement::CreateCatalog(_)) && owner.is_none() {
        // Existing built-in catalogs remain owned by DataFusion.
        if state.catalog_list().catalog(&identifier[0]).is_some() {
            return Ok(None);
        }
        registry.latest_foreign()
    } else {
        owner
    };
    let Some(owner) = owner else {
        return Ok(None);
    };
    let identifier: Vec<&str> = identifier.iter().map(String::as_str).collect();
    let plan = if let Some(options) = drop {
        owner.drop_object(&identifier, &options).await?
    } else {
        let input = if let Some(input) = input {
            Some(state.create_physical_plan(input).await?)
        } else {
            None
        };
        owner
            .create_object(&state, &identifier, &create, input)
            .await?
    };
    Ok(Some(
        ctx.ctx.read_table(Arc::new(CatalogDdlProvider { plan }))?,
    ))
}

fn create_mode(if_not_exists: bool, or_replace: bool) -> Result<CreateMode> {
    match (if_not_exists, or_replace) {
        (true, true) => exec_err!("'IF NOT EXISTS' cannot coexist with 'REPLACE'"),
        (true, false) => Ok(CreateMode::CreateOrIgnore),
        (false, true) => Ok(CreateMode::Replace),
        (false, false) => Ok(CreateMode::Create),
    }
}

/// Presents a catalog DDL execution plan as a one-shot DataFrame so DDL keeps
/// SedonaDB's lazy `.execute()`/`.collect()` behavior.
struct CatalogDdlProvider {
    plan: Arc<dyn ExecutionPlan>,
}

impl Debug for CatalogDdlProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatalogDdlProvider")
            .field("plan", &self.plan)
            .finish()
    }
}

#[async_trait]
impl TableProvider for CatalogDdlProvider {
    fn schema(&self) -> arrow_schema::SchemaRef {
        self.plan.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[datafusion_expr::Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self.plan.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        common::{plan_datafusion_err, plan_err},
        datasource::{empty::EmptyTable, listing::ListingTableUrl},
        sql::parser::DFParser,
    };
    use datafusion_execution::TaskContext;
    use datafusion_expr::sqlparser::dialect::dialect_from_str;
    use datafusion_physical_plan::{
        empty::EmptyExec, DisplayAs, DisplayFormatType, PlanProperties, SendableRecordBatchStream,
    };
    use sedona_catalog::{CatalogObject, SedonaCatalogList};
    use std::{collections::HashMap, sync::Mutex};
    use url::Url;

    type DropAction = Box<dyn FnOnce() -> Result<()> + Send>;

    struct DropTestExec {
        inner: Arc<dyn ExecutionPlan>,
        drop_action: Mutex<Option<DropAction>>,
    }

    impl DropTestExec {
        fn new(drop_action: impl FnOnce() -> Result<()> + Send + 'static) -> Self {
            Self {
                inner: Arc::new(EmptyExec::new(Arc::new(Schema::empty()))),
                drop_action: Mutex::new(Some(Box::new(drop_action))),
            }
        }
    }

    impl Debug for DropTestExec {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("DropTestExec").finish()
        }
    }

    impl DisplayAs for DropTestExec {
        fn fmt_as(
            &self,
            _t: DisplayFormatType,
            f: &mut std::fmt::Formatter<'_>,
        ) -> std::fmt::Result {
            write!(f, "DropTestExec")
        }
    }

    impl ExecutionPlan for DropTestExec {
        fn name(&self) -> &str {
            "DropTestExec"
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
                return datafusion_common::internal_err!("DropTestExec does not have children");
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
        objects: Arc<Mutex<HashMap<Vec<String>, CatalogObjectType>>>,
        creates: Mutex<Vec<(Vec<String>, CreateObjectOptions, bool)>>,
        drops: Mutex<Vec<(Vec<String>, DropObjectOptions)>>,
        fail: bool,
    }

    fn owned(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[async_trait]
    impl SedonaCatalogList for TestCatalog {
        fn name(&self) -> &str {
            "test"
        }

        async fn list_identifiers(
            &self,
            prefix: &[&str],
            depth: Option<usize>,
        ) -> Result<Vec<CatalogObject>> {
            tokio::task::yield_now().await;
            if self.fail {
                return exec_err!("foreign catalog lookup failed");
            }
            let prefix = owned(prefix);
            Ok(self
                .objects
                .lock()
                .unwrap()
                .iter()
                .filter(|(id, _)| {
                    id.starts_with(&prefix)
                        && depth.is_none_or(|depth| id.len() <= prefix.len() + depth)
                })
                .map(|(id, kind)| CatalogObject {
                    identifier: id.clone(),
                    object_type: *kind,
                })
                .collect())
        }

        async fn table(&self, identifier: &[&str]) -> Result<Option<Arc<dyn TableProvider>>> {
            tokio::task::yield_now().await;
            if matches!(
                self.objects.lock().unwrap().get(&owned(identifier)),
                Some(CatalogObjectType::Table | CatalogObjectType::View)
            ) {
                Ok(Some(Arc::new(EmptyTable::new(Arc::new(Schema::new(
                    vec![Field::new("value", DataType::Int64, true)],
                ))))))
            } else {
                Ok(None)
            }
        }

        async fn create_object(
            &self,
            _session: &dyn Session,
            identifier: &[&str],
            options: &CreateObjectOptions,
            input: Option<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            tokio::task::yield_now().await;
            let identifier = owned(identifier);
            self.creates.lock().unwrap().push((
                identifier.clone(),
                options.clone(),
                input.is_some(),
            ));
            let objects = self.objects.clone();
            let options = options.clone();
            Ok(Arc::new(DropTestExec::new(move || {
                let mut objects = objects.lock().unwrap();
                if objects.contains_key(&identifier) {
                    match options.mode {
                        CreateMode::Create => return exec_err!("already exists"),
                        CreateMode::CreateOrIgnore => return Ok(()),
                        CreateMode::Replace => {}
                    }
                }
                objects.insert(identifier, options.object_type);
                Ok(())
            })))
        }

        async fn drop_object(
            &self,
            identifier: &[&str],
            options: &DropObjectOptions,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            tokio::task::yield_now().await;
            let identifier = owned(identifier);
            self.drops
                .lock()
                .unwrap()
                .push((identifier.clone(), *options));
            let objects = self.objects.clone();
            let options = *options;
            Ok(Arc::new(DropTestExec::new(move || {
                let mut objects = objects.lock().unwrap();
                match objects.get(&identifier) {
                    None if options.if_exists => return Ok(()),
                    None => return exec_err!("no longer exists"),
                    Some(kind) if *kind != options.object_type => {
                        return exec_err!("wrong object kind")
                    }
                    _ => {}
                }
                if options.cascade {
                    objects.retain(|id, _| !id.starts_with(&identifier));
                } else {
                    objects.remove(&identifier);
                }
                Ok(())
            })))
        }
    }

    fn test_context() -> (SedonaContext, Arc<TestCatalog>) {
        let catalog = Arc::new(TestCatalog::default());
        catalog.objects.lock().unwrap().extend([
            (owned(&["foreign"]), CatalogObjectType::Catalog),
            (owned(&["foreign", "public"]), CatalogObjectType::Schema),
            (
                owned(&["foreign", "public", "existing"]),
                CatalogObjectType::Table,
            ),
        ]);
        let ctx = SedonaContext::new();
        ctx.register_catalog_list(catalog.clone());
        (ctx, catalog)
    }

    #[tokio::test]
    async fn foreign_create_routes_all_object_kinds_and_is_deferred() -> Result<()> {
        let (ctx, catalog) = test_context();
        for (sql, kind, identifier, input) in [
            (
                "CREATE DATABASE new_catalog",
                CatalogObjectType::Catalog,
                owned(&["new_catalog"]),
                false,
            ),
            (
                "CREATE SCHEMA foreign.new_schema",
                CatalogObjectType::Schema,
                owned(&["foreign", "new_schema"]),
                false,
            ),
            (
                "CREATE TABLE foreign.public.created AS SELECT 1 AS value",
                CatalogObjectType::Table,
                owned(&["foreign", "public", "created"]),
                true,
            ),
            (
                "CREATE VIEW foreign.public.view_one AS SELECT value FROM foreign.public.existing",
                CatalogObjectType::View,
                owned(&["foreign", "public", "view_one"]),
                true,
            ),
            (
                "CREATE INDEX idx ON foreign.public.existing (value)",
                CatalogObjectType::Index,
                owned(&["foreign", "public", "existing", "idx"]),
                false,
            ),
        ] {
            let df = ctx.sql(sql).await?;
            assert!(
                !catalog.objects.lock().unwrap().contains_key(&identifier),
                "{sql}"
            );
            let received = catalog.creates.lock().unwrap().last().unwrap().clone();
            assert_eq!(received.0, identifier);
            assert_eq!(received.1.object_type, kind);
            assert_eq!(received.2, input);
            df.collect().await?;
            assert_eq!(
                catalog.objects.lock().unwrap().get(&identifier),
                Some(&kind)
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn foreign_create_modes_and_view_definition_are_preserved() -> Result<()> {
        let (ctx, catalog) = test_context();
        for (prefix, mode) in [
            ("CREATE TABLE", CreateMode::Create),
            ("CREATE TABLE IF NOT EXISTS", CreateMode::CreateOrIgnore),
            ("CREATE OR REPLACE TABLE", CreateMode::Replace),
        ] {
            let sql = format!("{prefix} foreign.public.existing (value BIGINT)");
            let df = ctx.sql(&sql).await?;
            let options = catalog.creates.lock().unwrap().last().unwrap().1.clone();
            assert_eq!(options.mode, mode);
            if mode == CreateMode::Create {
                assert!(df.collect().await.is_err());
            } else {
                df.collect().await?;
            }
        }
        ctx.sql("CREATE OR REPLACE TEMPORARY VIEW foreign.public.view_one AS SELECT 1 AS value")
            .await?;
        let options = catalog.creates.lock().unwrap().last().unwrap().1.clone();
        assert_eq!(options.mode, CreateMode::Replace);
        assert!(options.temporary);
        assert!(options.definition.unwrap().contains("SELECT 1 AS value"));
        Ok(())
    }

    #[tokio::test]
    async fn foreign_drops_route_flags_and_recheck_at_execution() -> Result<()> {
        let (ctx, catalog) = test_context();
        let df = ctx
            .sql("DROP TABLE IF EXISTS foreign.public.existing PURGE")
            .await?;
        assert!(catalog
            .objects
            .lock()
            .unwrap()
            .remove(&owned(&["foreign", "public", "existing"]))
            .is_some());
        df.collect().await?;
        assert!(catalog.drops.lock().unwrap().last().unwrap().1.purge);
        let df = ctx.sql("DROP TABLE foreign.public.missing").await?;
        assert!(df
            .collect()
            .await
            .unwrap_err()
            .to_string()
            .contains("no longer exists"));
        catalog.objects.lock().unwrap().insert(
            owned(&["foreign", "public", "view_one"]),
            CatalogObjectType::View,
        );
        let wrong = ctx.sql("DROP TABLE foreign.public.view_one").await?;
        assert!(wrong
            .collect()
            .await
            .unwrap_err()
            .to_string()
            .contains("wrong object kind"));
        ctx.sql("DROP VIEW foreign.public.view_one")
            .await?
            .collect()
            .await?;
        let df = ctx
            .sql("DROP SCHEMA IF EXISTS foreign.public CASCADE")
            .await?;
        assert!(catalog
            .objects
            .lock()
            .unwrap()
            .contains_key(&owned(&["foreign", "public"])));
        df.collect().await?;
        let options = catalog.drops.lock().unwrap().last().unwrap().1;
        assert!(options.cascade && options.if_exists);
        assert_eq!(options.object_type, CatalogObjectType::Schema);
        Ok(())
    }

    #[tokio::test]
    async fn foreign_async_lookup_and_builtin_fallback() -> Result<()> {
        let (ctx, catalog) = test_context();
        ctx.sql("SELECT value FROM foreign.public.existing")
            .await?
            .collect()
            .await?;
        ctx.sql("CREATE TABLE builtin AS SELECT 1 AS value")
            .await?
            .collect()
            .await?;
        ctx.sql("SELECT * FROM builtin JOIN foreign.public.existing USING (value)")
            .await?
            .collect()
            .await?;
        assert!(catalog.creates.lock().unwrap().is_empty());
        // No synchronous adapter is exposed.
        assert!(ctx.ctx.catalog("foreign").is_none());
        let failing = Arc::new(TestCatalog {
            fail: true,
            ..Default::default()
        });
        ctx.register_catalog_list(failing);
        assert!(ctx
            .sql("SELECT * FROM builtin")
            .await
            .unwrap_err()
            .to_string()
            .contains("foreign catalog lookup failed"));
        Ok(())
    }

    #[tokio::test]
    async fn foreign_quoted_identifiers_and_listing() -> Result<()> {
        let (ctx, catalog) = test_context();
        ctx.sql("CREATE SCHEMA foreign.\"schema.with.dot\"")
            .await?
            .collect()
            .await?;
        ctx.sql("CREATE TABLE foreign.\"schema.with.dot\".\"table.with.dot\" AS SELECT 1 AS value")
            .await?
            .collect()
            .await?;
        ctx.sql("SELECT * FROM foreign.\"schema.with.dot\".\"table.with.dot\"")
            .await?
            .collect()
            .await?;
        assert_eq!(
            catalog.list_identifiers(&["foreign"], Some(0)).await?.len(),
            1
        );
        assert_eq!(
            catalog.list_identifiers(&["foreign"], Some(1)).await?.len(),
            3
        );
        assert_eq!(
            catalog
                .list_identifiers(&["foreign", "schema.with.dot", "table.with.dot"], None)
                .await?
                .len(),
            1
        );
        assert!(catalog
            .list_identifiers(&["missing"], None)
            .await?
            .is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn foreign_external_table_preserves_definition_without_opening_source() -> Result<()> {
        let (ctx, catalog) = test_context();
        let df = ctx.sql("CREATE EXTERNAL TABLE foreign.public.external (value BIGINT) STORED AS CSV LOCATION 'does-not-exist.csv'").await?;
        let received = catalog.creates.lock().unwrap().last().unwrap().clone();
        assert!(received.1.external && !received.2);
        assert!(received
            .1
            .definition
            .unwrap()
            .contains("does-not-exist.csv"));
        df.collect().await?;
        Ok(())
    }

    #[tokio::test]
    async fn function_ddl_is_delegated_to_datafusion() -> Result<()> {
        let (ctx, catalog) = test_context();
        for sql in [
            "CREATE FUNCTION fun() RETURNS INT RETURN 1",
            "DROP FUNCTION IF EXISTS fun",
        ] {
            let statement = DFParser::parse_sql(sql)?.pop_front().unwrap();
            let plan = create_plan_from_sql(&ctx, statement.clone()).await?;
            assert!(execute_sedona_catalog_ddl(&ctx, &plan, &statement)
                .await?
                .is_none());
        }
        assert!(catalog.creates.lock().unwrap().is_empty());
        assert!(catalog.drops.lock().unwrap().is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn foreign_ownership_precedence_and_default_namespaces() -> Result<()> {
        let (ctx, older) = test_context();
        let newer = Arc::new(TestCatalog::default());
        newer.objects.lock().unwrap().extend([
            (owned(&["foreign"]), CatalogObjectType::Catalog),
            (owned(&["datafusion"]), CatalogObjectType::Catalog),
            (owned(&["datafusion", "public"]), CatalogObjectType::Schema),
        ]);
        ctx.register_catalog_list(newer.clone());
        // A newer owner shadows the entire older catalog, including missing tables.
        assert!(ctx
            .sql("SELECT * FROM foreign.public.existing")
            .await
            .is_err());
        ctx.sql("CREATE TABLE t AS SELECT 1 AS value")
            .await?
            .collect()
            .await?;
        ctx.sql("CREATE SCHEMA s").await?.collect().await?;
        assert!(newer
            .objects
            .lock()
            .unwrap()
            .contains_key(&owned(&["datafusion", "public", "t"])));
        assert!(newer
            .objects
            .lock()
            .unwrap()
            .contains_key(&owned(&["datafusion", "s"])));
        assert!(older.creates.lock().unwrap().is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn foreign_planner_preserves_session_features() -> Result<()> {
        let (ctx, _) = test_context();
        for sql in [
            "SELECT ST_AsText(ST_Point(1, 2))",
            "SELECT count(*), sum(value), row_number() OVER () FROM foreign.public.existing",
            "WITH RECURSIVE t(n) AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM t WHERE n < 3) SELECT * FROM t",
        ] {
            ctx.sql(sql).await?.collect().await?;
        }
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("output.parquet");
        ctx.sql(&format!(
            "COPY (SELECT * FROM foreign.public.existing) TO '{}' STORED AS PARQUET",
            path.display()
        ))
        .await?
        .collect()
        .await?;
        Ok(())
    }

    async fn create_external_table_test(location: &str, sql: &str) -> Result<()> {
        let ctx = SedonaContext::new();
        let plan = ctx.ctx.state().create_logical_plan(sql).await?;

        if let LogicalPlan::Ddl(DdlStatement::CreateExternalTable(cmd)) = &plan {
            register_object_store_and_config_extensions(&ctx, &cmd.location, &cmd.options).await?;
        } else {
            return plan_err!("LogicalPlan is not a CreateExternalTable");
        }

        // Ensure the URL is supported by the object store
        ctx.ctx
            .runtime_env()
            .object_store(ListingTableUrl::parse(location)?)?;

        Ok(())
    }

    async fn copy_to_table_test(location: &str, sql: &str) -> Result<()> {
        let ctx = SedonaContext::new();
        // AWS CONFIG register.

        let plan = ctx.ctx.state().create_logical_plan(sql).await?;

        if let LogicalPlan::Copy(cmd) = &plan {
            register_object_store_and_config_extensions(&ctx, &cmd.output_url, &cmd.options)
                .await?;
        } else {
            return plan_err!("LogicalPlan is not a CreateExternalTable");
        }

        // Ensure the URL is supported by the object store
        ctx.ctx
            .runtime_env()
            .object_store(ListingTableUrl::parse(location)?)?;

        Ok(())
    }

    #[tokio::test]
    async fn create_object_store_table_http() -> Result<()> {
        // Should be OK
        let location = "http://example.com/file.parquet";
        let sql = format!("CREATE EXTERNAL TABLE test STORED AS PARQUET LOCATION '{location}'");
        create_external_table_test(location, &sql).await?;

        Ok(())
    }
    #[tokio::test]
    async fn copy_to_external_object_store_test() -> Result<()> {
        let locations = vec![
            "s3://bucket/path/file.parquet",
            "oss://bucket/path/file.parquet",
            "cos://bucket/path/file.parquet",
            "gcs://bucket/path/file.parquet",
        ];
        let ctx = SedonaContext::new();
        let task_ctx = ctx.ctx.task_ctx();
        let dialect = &task_ctx.session_config().options().sql_parser.dialect;
        let dialect = dialect_from_str(dialect).ok_or_else(|| {
            plan_datafusion_err!(
                "Unsupported SQL dialect: {dialect}. Available dialects: \
                 Generic, MySQL, PostgreSQL, Hive, SQLite, Snowflake, Redshift, \
                 MsSQL, ClickHouse, BigQuery, Ansi, DuckDB, Databricks."
            )
        })?;
        for location in locations {
            let sql = format!("copy (values (1,2)) to '{location}' STORED AS PARQUET;");
            let statements = DFParser::parse_sql_with_dialect(&sql, dialect.as_ref())?;
            for statement in statements {
                //Should not fail
                let mut plan = create_plan_from_sql(&ctx, statement).await?;
                if let LogicalPlan::Copy(copy_to) = &mut plan {
                    assert_eq!(copy_to.output_url, location);
                    assert_eq!(copy_to.file_type.get_ext(), "parquet".to_string());
                    ctx.ctx
                        .runtime_env()
                        .object_store_registry
                        .get_store(&Url::parse(&copy_to.output_url).unwrap())?;
                } else {
                    return plan_err!("LogicalPlan is not a CopyTo");
                }
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn copy_to_object_store_table_s3() -> Result<()> {
        let access_key_id = "fake_access_key_id";
        let secret_access_key = "fake_secret_access_key";
        let location = "s3://bucket/path/file.parquet";

        // Missing region, use object_store defaults
        let sql = format!("COPY (values (1,2)) TO '{location}' STORED AS PARQUET
            OPTIONS ('aws.access_key_id' '{access_key_id}', 'aws.secret_access_key' '{secret_access_key}')");
        copy_to_table_test(location, &sql).await?;

        Ok(())
    }

    #[tokio::test]
    async fn create_object_store_table_s3() -> Result<()> {
        let access_key_id = "fake_access_key_id";
        let secret_access_key = "fake_secret_access_key";
        let region = "fake_us-east-2";
        let session_token = "fake_session_token";
        let location = "s3://bucket/path/file.parquet";

        // Missing region, use object_store defaults
        let sql = format!("CREATE EXTERNAL TABLE test STORED AS PARQUET
            OPTIONS('aws.access_key_id' '{access_key_id}', 'aws.secret_access_key' '{secret_access_key}') LOCATION '{location}'");
        create_external_table_test(location, &sql).await?;

        // Should be OK
        let sql = format!("CREATE EXTERNAL TABLE test STORED AS PARQUET
            OPTIONS('aws.access_key_id' '{access_key_id}', 'aws.secret_access_key' '{secret_access_key}', 'aws.region' '{region}', 'aws.session_token' '{session_token}') LOCATION '{location}'");
        create_external_table_test(location, &sql).await?;

        Ok(())
    }

    #[tokio::test]
    async fn create_object_store_table_oss() -> Result<()> {
        let access_key_id = "fake_access_key_id";
        let secret_access_key = "fake_secret_access_key";
        let endpoint = "fake_endpoint";
        let location = "oss://bucket/path/file.parquet";

        // Should be OK
        let sql = format!("CREATE EXTERNAL TABLE test STORED AS PARQUET
            OPTIONS('aws.access_key_id' '{access_key_id}', 'aws.secret_access_key' '{secret_access_key}', 'aws.oss.endpoint' '{endpoint}') LOCATION '{location}'");
        create_external_table_test(location, &sql).await?;

        Ok(())
    }

    #[tokio::test]
    async fn create_object_store_table_cos() -> Result<()> {
        let access_key_id = "fake_access_key_id";
        let secret_access_key = "fake_secret_access_key";
        let endpoint = "fake_endpoint";
        let location = "cos://bucket/path/file.parquet";

        // Should be OK
        let sql = format!("CREATE EXTERNAL TABLE test STORED AS PARQUET
            OPTIONS('aws.access_key_id' '{access_key_id}', 'aws.secret_access_key' '{secret_access_key}', 'aws.cos.endpoint' '{endpoint}') LOCATION '{location}'");
        create_external_table_test(location, &sql).await?;

        Ok(())
    }

    #[tokio::test]
    async fn create_object_store_table_gcs() -> Result<()> {
        let service_account_path = "fake_service_account_path";
        let service_account_key =
            "{\"private_key\": \"fake_private_key.pem\",\"client_email\":\"fake_client_email\", \"private_key_id\":\"id\"}";
        let application_credentials_path = "fake_application_credentials_path";
        let location = "gcs://bucket/path/file.parquet";

        // for service_account_path
        let sql = format!(
            "CREATE EXTERNAL TABLE test STORED AS PARQUET
            OPTIONS('gcp.service_account_path' '{service_account_path}') LOCATION '{location}'"
        );
        let err = create_external_table_test(location, &sql)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("os error 2"));

        // for service_account_key
        let sql = format!("CREATE EXTERNAL TABLE test STORED AS PARQUET OPTIONS('gcp.service_account_key' '{service_account_key}') LOCATION '{location}'");
        let err = create_external_table_test(location, &sql)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no items found"), "{err}");

        // for application_credentials_path
        let sql = format!("CREATE EXTERNAL TABLE test STORED AS PARQUET
            OPTIONS('gcp.application_credentials_path' '{application_credentials_path}') LOCATION '{location}'");
        let err = create_external_table_test(location, &sql)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("os error 2"));

        Ok(())
    }

    #[tokio::test]
    async fn create_external_table_local_file() -> Result<()> {
        let location = "path/to/file.parquet";

        // Ensure that local files are also registered
        let sql = format!("CREATE EXTERNAL TABLE test STORED AS PARQUET LOCATION '{location}'");
        create_external_table_test(location, &sql).await.unwrap();

        Ok(())
    }

    #[tokio::test]
    async fn create_external_table_format_option() -> Result<()> {
        let location = "path/to/file.cvs";

        // Test with format options
        let sql =
            format!("CREATE EXTERNAL TABLE test STORED AS CSV LOCATION '{location}' OPTIONS('format.has_header' 'true')");
        create_external_table_test(location, &sql).await.unwrap();

        Ok(())
    }
}
