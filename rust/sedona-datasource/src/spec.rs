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

use std::{collections::HashMap, fmt::Debug, sync::Arc};

use arrow_array::RecordBatchReader;
use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;

use datafusion::{config::TableOptions, datasource::listing::FileRange};
use datafusion_common::{Result, Statistics};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_expr::PhysicalExpr;
use object_store::{ObjectMeta, ObjectStore};

/// Simple file format specification
///
/// In DataFusion, various parts of the file format are split among the
/// FileFormatFactory, the FileFormat, the FileSource, the FileOpener,
/// and a few other traits. This trait is designed to provide a few
/// important features of a natively implemented FileFormat but consolidating
/// the components of implementing the format in the same place. This is
/// intended to provide a less verbose way to implement readers for a wide
/// variety of spatial formats.
#[async_trait]
pub trait ExternalFormatSpec: Debug + Send + Sync {
    /// Infer a schema for a given file
    ///
    /// Given a single file, infer what schema [ExternalFormatSpec::open_reader]
    /// would produce in the absence of any other guidance.
    async fn infer_schema(&self, location: &Object) -> Result<Schema>;

    /// Open a [RecordBatchReader] for a given file
    ///
    /// The implementation must handle the `file_projection`; however,
    /// need not handle the `filters` (but may use them for pruning).
    async fn open_reader(&self, args: &OpenReaderArgs)
        -> Result<Box<dyn RecordBatchReader + Send>>;

    /// A file extension or `""` if this concept does not apply
    fn extension(&self) -> &str {
        ""
    }

    /// Compute a clone of self but with the key/value options specified
    ///
    /// Implementations should error for invalid key/value input that does
    /// not apply to this reader.
    fn with_options(
        &self,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn ExternalFormatSpec>>;

    /// Fill in default options from [TableOptions]
    ///
    /// The TableOptions are a DataFusion concept that provide a means by which
    /// options can be set for various table formats. If the defaults for a built-in
    /// table format are reasonable to fill in or if Extensions have been set,
    /// these can be accessed and used to fill default options. Note that any options
    /// set with [ExternalFormatSpec::with_options] should take precedent.
    fn with_table_options(
        &self,
        _table_options: &TableOptions,
    ) -> Option<Arc<dyn ExternalFormatSpec>> {
        None
    }

    /// Allow repartitioning
    ///
    /// This allows an implementation to opt in to DataFusion's built-in file size
    /// based partitioner, which works well for partitioning files where a simple
    /// file plus byte range is sufficient. The default opts out of this feature
    /// (i.e., every file is passed exactly one to [ExternalFormatSpec::open_reader]
    /// without a `range`).
    fn supports_repartition(&self) -> SupportsRepartition {
        SupportsRepartition::None
    }

    /// Infer [Statistics] for a given file
    async fn infer_stats(&self, _location: &Object, table_schema: &Schema) -> Result<Statistics> {
        Ok(Statistics::new_unknown(table_schema))
    }
}

/// Enumerator for repartitioning support
#[derive(Debug, Clone, Copy)]
pub enum SupportsRepartition {
    /// This implementation does not support repartitioning beyond the file level
    None,
    /// This implementation supports partitioning by arbitrary ranges with a file
    ///
    /// Implementations that return this must check [Object::range] when opening
    /// a file and ensure that each record is read exactly once from one file
    /// with potentially multiple ranges.
    ByRange,
}

/// Arguments to [ExternalFormatSpec::open_reader]
#[derive(Debug, Clone)]
pub struct OpenReaderArgs {
    /// The input file, or partial file if [SupportsRepartition::ByRange] is used
    pub src: Object,

    /// The requested batch size
    ///
    /// DataFusion will usually fill this in to a default of 8192 or a user-specified
    /// default in the session configuration.
    pub batch_size: Option<usize>,

    /// The requested file schema, if specified
    ///
    /// DataFusion will usually fill this in to the schema inferred by
    /// [ExternalFormatSpec::infer_schema].
    pub file_schema: Option<SchemaRef>,

    /// The requested field indices
    ///
    /// Implementations must handle this (e.g., using `RecordBatch::project`
    /// or by implementing partial reads).
    pub file_projection: Option<Vec<usize>>,

    /// Filter expressions
    ///
    /// Expressions that may be used for pruning. Implementations need not
    /// apply these filters.
    pub filters: Vec<Arc<dyn PhysicalExpr>>,
}

/// The information required to specify a file or partial file
///
/// Depending exactly where in DataFusion we are calling in from, we might
/// have various information about the file. In general, implementations should
/// use [ObjectStore] and [ObjectMeta] to access the file to use DataFusion's
/// registered IO for these protocols. When implementing a filename-based reader
/// (e.g., that uses some external API to read files), use [Object::to_url_string].
#[derive(Debug, Clone)]
pub struct Object {
    /// The object store reference
    pub store: Option<Arc<dyn ObjectStore>>,

    /// A URL that may be used to retrieve an [ObjectStore] from a registry
    ///
    /// These URLs typically are populated only with the scheme.
    pub url: Option<ObjectStoreUrl>,

    /// An individual object in an ObjectStore
    pub meta: Option<ObjectMeta>,

    /// If this represents a partial file, the byte range within the file
    ///
    /// This is only set if partitioning other than `None` is provided
    pub range: Option<FileRange>,
}

impl Object {
    /// Convert this object to a URL string, if possible
    ///
    /// Returns `None` if there is not sufficient information in the Object to calculate
    /// this.
    pub fn to_url_string(&self) -> Option<String> {
        match (&self.url, &self.meta) {
            (None, Some(meta)) => {
                // There's no great way to map an object_store to a url prefix if we're not
                // provided the `url`; however, this is what we have access to in the
                // Schema and Statistics resolution phases of the FileFormat.
                // This is a heuristic that should work for https and a local filesystem,
                // which is what we might be able to expect a non-DataFusion system like
                // GDAL to be able to translate.
                let object_store_debug = format!("{:?}", self.store).to_lowercase();
                if object_store_debug.contains("http") {
                    Some(format!("https://{}", meta.location))
                } else if object_store_debug.contains("local") {
                    Some(format!("file:///{}", meta.location))
                } else {
                    None
                }
            }
            (Some(url), None) => Some(url.to_string()),
            (Some(url), Some(meta)) => Some(format!("{url}/{}", meta.location)),
            (None, None) => None,
        }
    }
}
