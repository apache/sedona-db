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

//! TEMPORARY workaround for DataFusion #22662
//! (<https://github.com/apache/datafusion/issues/22662>): async scalar UDFs
//! drop their `return_field` metadata at the physical layer, so the output
//! of `RS_EnsureLoaded` (async) loses its extension type metadata and
//! downstream functions stop recognising it.
//!
//! As a stopgap, optimizer rules wrap each async load in this sync
//! `sd_restore_metadata` UDF — `sd_restore_metadata(rs_ensureloaded(x))`. A
//! sync `ScalarFunctionExpr` *does* preserve `return_field`, so this node
//! re-applies the extension type metadata onto the (otherwise stripped)
//! struct. The data passes through untouched (identity).
//!
//! **Removal:** once #22662 is fixed and downstreamed, delete this whole
//! file, its `mod` declaration, and revert the `DF-22662`-tagged
//! touchpoints in optimizer rules. `grep DF-22662` finds every site.

use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::Result;
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use sedona_common::sedona_internal_err;

/// Registry/UDF name of the metadata restoration helper.
pub const RESTORE_METADATA_NAME: &str = "sd_restore_metadata";

/// Build the sync metadata-restore UDF with the given metadata map. The rule
/// constructs one per rewrite, embedding the metadata that should be stamped
/// onto the output field. `Stable` volatility so CSE can deduplicate identical
/// `sd_restore_metadata(...)` subtrees.
pub fn restore_metadata_udf(metadata: HashMap<String, String>) -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(RestoreMetadata::new(metadata)))
}

/// Sync identity UDF that forces the stored metadata onto its argument's
/// output field. See the module docs (DF-22662).
#[derive(Debug, Clone)]
struct RestoreMetadata {
    signature: Signature,
    /// Metadata to stamp onto the output field.
    metadata: HashMap<String, String>,
}

impl RestoreMetadata {
    fn new(metadata: HashMap<String, String>) -> Self {
        Self {
            signature: Signature::any(1, Volatility::Stable),
            metadata,
        }
    }
}

// Identity by name + metadata so CSE can dedup equivalent instances.
impl PartialEq for RestoreMetadata {
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata
    }
}
impl Eq for RestoreMetadata {}
impl Hash for RestoreMetadata {
    fn hash<H: Hasher>(&self, state: &mut H) {
        RESTORE_METADATA_NAME.hash(state);
        // Hash metadata in sorted order for determinism
        let mut pairs: Vec<_> = self.metadata.iter().collect();
        pairs.sort_by_key(|(k, _)| *k);
        for (k, v) in pairs {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl ScalarUDFImpl for RestoreMetadata {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        RESTORE_METADATA_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // return_field_from_args is authoritative (it carries the extension
        // metadata a bare DataType would drop).
        sedona_internal_err!("sd_restore_metadata::return_type should not be called")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        // Force the stored metadata onto the output field regardless of the
        // input field. At physical planning the input is the async output,
        // whose metadata has already been stripped (DF-22662), so an identity
        // return_field would NOT restore it — we must stamp it unconditionally.
        let (name, data_type, nullable) = args
            .arg_fields
            .first()
            .map(|f| (f.name().clone(), f.data_type().clone(), f.is_nullable()))
            .unwrap_or_else(|| ("output".to_string(), DataType::Null, true));
        Ok(Arc::new(
            Field::new(name, data_type, nullable).with_metadata(self.metadata.clone()),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if args.args.len() != 1 {
            return sedona_internal_err!("sd_restore_metadata expects exactly one argument");
        }

        Ok(args.args[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Field;

    #[test]
    fn restore_metadata_stamps_metadata_onto_field() {
        // Models the physical-planning input: a struct that has lost its
        // extension metadata (the DF-22662 symptom). restore_metadata must
        // stamp it back.
        let mut metadata = HashMap::new();
        metadata.insert(
            "ARROW:extension:name".to_string(),
            "sedona.raster".to_string(),
        );
        metadata.insert("ARROW:extension:metadata".to_string(), "{}".to_string());

        let udf = RestoreMetadata::new(metadata.clone());
        let bare = Arc::new(Field::new(
            "loaded",
            DataType::Struct(Vec::<FieldRef>::new().into()),
            true,
        ));
        let args = ReturnFieldArgs {
            arg_fields: &[bare],
            scalar_arguments: &[None],
        };
        let out = udf.return_field_from_args(args).unwrap();
        assert_eq!(
            out.metadata(),
            &metadata,
            "restore_metadata must re-stamp the extension metadata: {out:?}"
        );
    }

    #[test]
    fn restore_metadata_equality_based_on_metadata() {
        let mut meta1 = HashMap::new();
        meta1.insert("key".to_string(), "value".to_string());

        let mut meta2 = HashMap::new();
        meta2.insert("key".to_string(), "value".to_string());

        let mut meta3 = HashMap::new();
        meta3.insert("key".to_string(), "different".to_string());

        let udf1 = RestoreMetadata::new(meta1);
        let udf2 = RestoreMetadata::new(meta2);
        let udf3 = RestoreMetadata::new(meta3);

        assert_eq!(udf1, udf2, "same metadata should be equal");
        assert_ne!(udf1, udf3, "different metadata should not be equal");
    }
}
