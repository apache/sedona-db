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
//! Resolve foreign table sources asynchronously before invoking the SQL planner.
use crate::context::SedonaContext;
use arrow_schema::DataType;
use datafusion::{
    datasource::provider_as_source,
    execution::session_state::{SessionState, SessionStateBuilder},
    optimizer::simplify_expressions::ExprSimplifier,
    sql::{parser::Statement, planner::SqlToRel},
};
use datafusion_common::{
    config::ConfigOptions, plan_datafusion_err, DFSchema, ResolvedTableReference, Result,
    TableReference,
};
use datafusion_expr::{
    planner::{ContextProvider, ExprPlanner, RelationPlanner, TypePlanner},
    simplify::SimplifyContext,
    AggregateUDF, Expr, HigherOrderUDF, LogicalPlan, ScalarUDF, TableSource, WindowUDF,
};
use std::{collections::HashMap, sync::Arc};

pub(crate) async fn statement_to_plan(
    ctx: &SedonaContext,
    statement: Statement,
) -> Result<LogicalPlan> {
    let state = ctx.ctx.state();
    if ctx.catalog_registry().latest_foreign().is_none() {
        return state.statement_to_plan(statement).await;
    }
    let mut builder = SessionStateBuilder::from(state.clone());
    let mut provider = CatalogContextProvider {
        state: &state,
        tables: HashMap::new(),
        type_planner: builder.type_planner().clone(),
    };
    for reference in state.resolve_table_references(&statement)? {
        let resolved = reference.resolve(
            &state.config_options().catalog.default_catalog,
            &state.config_options().catalog.default_schema,
        );
        if provider.tables.contains_key(&resolved) {
            continue;
        }
        let table = if let Some(catalog) = ctx
            .catalog_registry()
            .foreign_catalog(&resolved.catalog)
            .await?
        {
            catalog
                .table(&[&resolved.catalog, &resolved.schema, &resolved.table])
                .await?
        } else if let Ok(schema) = state.schema_for_ref(resolved.clone()) {
            schema.table(&resolved.table).await?
        } else {
            None
        };
        if let Some(table) = table {
            provider.tables.insert(resolved, provider_as_source(table));
        }
    }
    SqlToRel::new_with_options(&provider, (&state.config_options().sql_parser).into())
        .statement_to_plan(statement)
}

struct CatalogContextProvider<'a> {
    state: &'a SessionState,
    tables: HashMap<ResolvedTableReference, Arc<dyn TableSource>>,
    type_planner: Option<Arc<dyn TypePlanner>>,
}

impl ContextProvider for CatalogContextProvider<'_> {
    fn get_expr_planners(&self) -> &[Arc<dyn ExprPlanner>] {
        self.state.expr_planners()
    }

    fn get_relation_planners(&self) -> &[Arc<dyn RelationPlanner>] {
        self.state.relation_planners()
    }

    fn get_type_planner(&self) -> Option<Arc<dyn TypePlanner>> {
        if let Some(type_planner) = &self.type_planner {
            Some(Arc::clone(type_planner))
        } else {
            None
        }
    }

    fn get_table_source(
        &self,
        name: TableReference,
    ) -> datafusion_common::Result<Arc<dyn TableSource>> {
        let name = name.resolve(
            &self.state.config_options().catalog.default_catalog,
            &self.state.config_options().catalog.default_schema,
        );
        self.tables
            .get(&name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("table '{name}' not found"))
    }

    fn get_table_function_source(
        &self,
        name: &str,
        args: Vec<Expr>,
    ) -> datafusion_common::Result<Arc<dyn TableSource>> {
        use datafusion_catalog::TableFunctionArgs;

        let tbl_func = self
            .state
            .table_functions()
            .get(name)
            .cloned()
            .ok_or_else(|| plan_datafusion_err!("table function '{name}' not found"))?;
        let simplify_context = SimplifyContext::builder()
            .with_config_options(Arc::clone(self.state.config_options()))
            .with_query_execution_start_time(
                self.state.execution_props().query_execution_start_time,
            )
            .build();
        let simplifier = ExprSimplifier::new(simplify_context);
        let schema = DFSchema::empty();
        let args = args
            .into_iter()
            .map(|arg| {
                simplifier
                    .coerce(arg, &schema)
                    .and_then(|e| simplifier.simplify(e))
            })
            .collect::<datafusion_common::Result<Vec<_>>>()?;
        let provider =
            tbl_func.create_table_provider_with_args(TableFunctionArgs::new(&args, self.state))?;

        Ok(provider_as_source(provider))
    }

    /// Create a new CTE work table for a recursive CTE logical plan
    /// This table will be used in conjunction with a Worktable physical plan
    /// to read and write each iteration of a recursive CTE
    fn create_cte_work_table(
        &self,
        name: &str,
        schema: arrow_schema::SchemaRef,
    ) -> datafusion_common::Result<Arc<dyn TableSource>> {
        let table = Arc::new(datafusion::datasource::cte_worktable::CteWorkTable::new(
            name, schema,
        ));
        Ok(provider_as_source(table))
    }

    fn get_function_meta(&self, name: &str) -> Option<Arc<ScalarUDF>> {
        self.state.scalar_functions().get(name).cloned()
    }

    fn get_higher_order_meta(&self, name: &str) -> Option<Arc<HigherOrderUDF>> {
        self.state.higher_order_functions().get(name).cloned()
    }

    fn get_aggregate_meta(&self, name: &str) -> Option<Arc<AggregateUDF>> {
        self.state.aggregate_functions().get(name).cloned()
    }

    fn get_window_meta(&self, name: &str) -> Option<Arc<WindowUDF>> {
        self.state.window_functions().get(name).cloned()
    }

    fn get_variable_type(&self, variable_names: &[String]) -> Option<DataType> {
        use datafusion_expr::var_provider::{is_system_variables, VarType};

        if variable_names.is_empty() {
            return None;
        }

        let provider_type = if is_system_variables(variable_names) {
            VarType::System
        } else {
            VarType::UserDefined
        };

        self.state
            .execution_props()
            .var_providers
            .as_ref()
            .and_then(|provider| provider.get(&provider_type)?.get_type(variable_names))
    }

    fn options(&self) -> &ConfigOptions {
        self.state.config_options()
    }

    fn udf_names(&self) -> Vec<String> {
        self.state.scalar_functions().keys().cloned().collect()
    }

    fn higher_order_function_names(&self) -> Vec<String> {
        self.state
            .higher_order_functions()
            .keys()
            .cloned()
            .collect()
    }

    fn udaf_names(&self) -> Vec<String> {
        self.state.aggregate_functions().keys().cloned().collect()
    }

    fn udwf_names(&self) -> Vec<String> {
        self.state.window_functions().keys().cloned().collect()
    }

    fn get_file_type(
        &self,
        ext: &str,
    ) -> datafusion_common::Result<Arc<dyn datafusion_common::file_options::file_type::FileType>>
    {
        self.state
            .get_file_format_factory(ext)
            .map(datafusion::datasource::file_format::format_as_file_type)
            .ok_or_else(|| {
                plan_datafusion_err!("There is no registered file format with ext {ext}")
            })
    }
}
