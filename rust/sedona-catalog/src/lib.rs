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

//! Asynchronous catalog operations used by SedonaDB extensions.
//!
//! Identifiers are paths of literal components, never dot-separated SQL names.
//! No catalog or schema objects, or synchronous DataFusion adapters, are needed.

use async_trait::async_trait;
use datafusion_catalog::{Session, TableProvider};
use datafusion_common::Result;
use datafusion_physical_plan::ExecutionPlan;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};

/// Behavior when creating an object that already exists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreateMode {
    /// Fail if the object exists.
    #[default]
    Create,
    /// Leave an existing object unchanged.
    CreateOrIgnore,
    /// Replace the existing object.
    Replace,
}

/// Kind of object in the catalog hierarchy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogObjectType {
    /// A catalog (database).
    Catalog,
    /// A schema (namespace).
    Schema,
    /// A physical table.
    #[default]
    Table,
    /// A non-materialized view.
    View,
    /// An index belonging to a table.
    Index,
}

/// An entry returned by [`SedonaCatalogList::list_identifiers`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogObject {
    /// Full path from the root, including all parent components.
    pub identifier: Vec<String>,
    /// Kind of this object, independent of its path length.
    pub object_type: CatalogObjectType,
}

/// Options for creating any catalog object.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CreateObjectOptions {
    /// Kind of object to create.
    pub object_type: CatalogObjectType,
    /// Conflict behavior, applied when the returned plan executes.
    pub mode: CreateMode,
    /// Whether the object is temporary.
    pub temporary: bool,
    /// Whether this is an external table.
    pub external: bool,
    /// Optional SQL definition for views, external tables, and indexes. Index
    /// definitions preserve the SQL AST, including expressions, method, and
    /// uniqueness. View and external table definitions are supplied by the
    /// planner and may omit DDL clauses; Sedona SQL rejects external table
    /// metadata that cannot be preserved.
    pub definition: Option<String>,
}

/// Options for dropping any catalog object.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DropObjectOptions {
    /// Expected kind. Implementations must not drop an object of another kind.
    pub object_type: CatalogObjectType,
    /// Ignore a missing object at execution time.
    pub if_exists: bool,
    /// Also drop contained/dependent objects.
    pub cascade: bool,
    /// Delete referenced data as well as metadata, where applicable.
    pub purge: bool,
}

/// A catalog extension with asynchronous lookup and deferred mutations.
///
/// SQL uses `[catalog]`, `[catalog, schema]`, and `[catalog, schema, table]`.
/// Implementations may expose deeper hierarchies through listing. All errors,
/// including listing errors, are propagated to the caller.
#[async_trait]
pub trait SedonaCatalogList: Debug + Send + Sync {
    /// Return the stable name of this catalog implementation, such as `iceberg`.
    /// This identifies the implementation for provider selection (for example,
    /// a SQL `USING` clause), independently of the catalog names in its hierarchy.
    /// The name must be available without catalog I/O.
    fn name(&self) -> &str;

    /// List objects beneath an exact, literal prefix, including the prefix itself
    /// if it identifies an object. `depth` limits the number of additional path
    /// components: zero is an exact lookup, one includes immediate children,
    /// and `None` includes all descendants. An empty prefix addresses the root.
    ///
    /// Return full identifiers, without duplicates; ordering is unspecified.
    /// Missing prefixes return an empty list.
    async fn list_identifiers(
        &self,
        prefix: &[&str],
        depth: Option<usize>,
    ) -> Result<Vec<CatalogObject>>;

    /// Look up a table or view by its full identifier; missing objects return None.
    async fn table(&self, identifier: &[&str]) -> Result<Option<Arc<dyn TableProvider>>>;

    /// Build a create plan without changing catalog state. `input` provides the
    /// managed table/view schema and data when applicable; it is absent for
    /// external tables, namespaces, and indexes. `session` belongs to
    /// this call and must not be retained by the catalog. The returned plan applies
    /// conflict behavior and validates the parent namespace when executed.
    async fn create_object(
        &self,
        session: &dyn Session,
        identifier: &[&str],
        options: &CreateObjectOptions,
        input: Option<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>>;

    /// Build a drop plan without changing catalog state. Missing-object and
    /// object-kind checks belong to execution, including `if_exists` handling.
    async fn drop_object(
        &self,
        identifier: &[&str],
        options: &DropObjectOptions,
    ) -> Result<Arc<dyn ExecutionPlan>>;
}
