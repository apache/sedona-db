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

//! Prototype: user-defined types via [`crate::datatypes::SedonaType::Extension`].
//!
//! Two separate problems, each solved differently:
//!
//! - **Forward direction** (build a value, match it in a function signature,
//!   display it): handled entirely by [`SedonaExtensionType`] + the
//!   `SedonaType::Extension` variant. No registry needed -- a caller who
//!   already has a concrete `Arc<dyn SedonaExtensionType>` in hand just
//!   wraps it.
//! - **Backward direction** (recover a `SedonaType` from a bare external
//!   `Field`'s Arrow extension metadata -- e.g. reading a Parquet file back):
//!   the field only carries an `extension_name` string, not a live Rust
//!   value, so *something* has to map that name back to the right concrete
//!   type. That's [`register_extension_type`]/[`lookup_extension_type_factory`]
//!   below, consumed by [`crate::datatypes::SedonaType::from_extension_type`].
//!
//! The registry is **global (process-wide), not session-scoped** -- contrast
//! `sedona_raster::raster_loader::RasterLoaderRegistry`, the closest existing
//! precedent for "pluggable, named backend registered by an extension crate."
//! That registry lives on `SedonaContext` because its only caller
//! (`RS_EnsureLoaded`) already has a context in hand. `SedonaType::
//! from_storage_field` has no such luxury: it's called from roughly 40 sites
//! across nearly every crate in the workspace (schema introspection, UDF
//! argument-type resolution, physical planners, spatial-join operand
//! evaluation, spill/serialization, Python FFI schema conversion, tests),
//! many of them pure functions with no context object reachable at all.
//! Threading a registry handle through all of them would be a sprawling,
//! invasive change -- exactly what this mechanism is trying to avoid.
//! `sedona-schema` also sits below any session/context type in the
//! dependency graph, so it couldn't depend on one even if we wanted to.

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, LazyLock, RwLock};

use arrow_schema::DataType;
use datafusion_common::Result;

use crate::extension_type::ExtensionType;

/// Implemented by a user-defined type that wants to flow through
/// [`crate::datatypes::SedonaType::Extension`] -- function signatures,
/// coercion, display, equality -- without adding a variant to the core
/// `SedonaType` enum.
///
/// Every method mirrors an existing `SedonaType` accessor of the same name;
/// see `datatypes.rs`'s `Extension` match arms for exactly how each is used.
pub trait SedonaExtensionType: Debug + Send + Sync + 'static {
    /// Arrow extension name, e.g. `"myorg.custom_type"`. Must be `'static`
    /// (in practice always a string literal) so
    /// `SedonaType::extension_name() -> Option<&'static str>` doesn't need
    /// to change shape for this variant.
    fn extension_name(&self) -> &'static str;

    /// The physical Arrow storage type. Returns `&DataType` (not an owned
    /// value) so `SedonaType::storage_type() -> &DataType` doesn't need to
    /// change shape either -- implementers cache their `DataType` the same
    /// way `RASTER_DATATYPE` does (a `LazyLock`, or a field computed once at
    /// construction).
    fn storage_type(&self) -> &DataType;

    /// Logical name for `DESCRIBE`/schema printing. Defaults to
    /// `extension_name()` if not overridden.
    fn logical_type_name(&self) -> String {
        self.extension_name().to_string()
    }

    /// Arrow `ARROW:extension:metadata` payload, if any. Must round-trip
    /// through whatever this type's registered [`ExtensionTypeFactory`]
    /// expects to parse back out.
    fn extension_metadata(&self) -> Option<String> {
        None
    }

    /// Downcast support, so a kernel that knows the concrete type (e.g. the
    /// Tensor crate's own kernels) can get at fields `SedonaExtensionType`
    /// doesn't expose generically.
    fn as_any(&self) -> &dyn Any;

    /// Backs `PartialEq for dyn SedonaExtensionType` below. Implementers
    /// downcast `other` and delegate to their own `PartialEq`:
    /// ```ignore
    /// fn dyn_eq(&self, other: &dyn SedonaExtensionType) -> bool {
    ///     other.as_any().downcast_ref::<Self>() == Some(self)
    /// }
    /// ```
    fn dyn_eq(&self, other: &dyn SedonaExtensionType) -> bool;
}

impl PartialEq for dyn SedonaExtensionType {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}

/// Reconstructs a concrete `Arc<dyn SedonaExtensionType>` from the
/// `(extension_name, storage_type, extension_metadata)` triple recovered
/// from an external `Field`. Registered once per `extension_name`.
pub type ExtensionTypeFactory =
    dyn Fn(&ExtensionType) -> Result<Arc<dyn SedonaExtensionType>> + Send + Sync;

static EXTENSION_TYPE_REGISTRY: LazyLock<RwLock<HashMap<&'static str, Arc<ExtensionTypeFactory>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Register a factory for `extension_name`. Call once per process -- e.g.
/// from the extension crate's own setup/init path -- before any data tagged
/// with this extension name needs to be read back into a `SedonaType`.
///
/// Last registration for a given name wins (overwrites silently) rather than
/// erroring on re-registration -- re-running a test or reloading an
/// extension in a REPL is a normal thing to do, not a bug.
pub fn register_extension_type<F>(extension_name: &'static str, factory: F)
where
    F: Fn(&ExtensionType) -> Result<Arc<dyn SedonaExtensionType>> + Send + Sync + 'static,
{
    let mut registry = EXTENSION_TYPE_REGISTRY
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.insert(extension_name, Arc::new(factory));
}

/// Look up the registered factory for `extension_name`, if any. `pub(crate)`
/// -- the only caller is `SedonaType::from_extension_type`; external code
/// registers via [`register_extension_type`] but never needs to look up.
pub(crate) fn lookup_extension_type_factory(
    extension_name: &str,
) -> Option<Arc<ExtensionTypeFactory>> {
    let registry = EXTENSION_TYPE_REGISTRY
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.get(extension_name).cloned()
}

/// Names of every currently-registered extension type. Diagnostic use only
/// (e.g. an error message listing what *is* registered when a lookup
/// fails), mirroring `RasterLoaderRegistry::loader_names`.
pub fn registered_extension_type_names() -> Vec<&'static str> {
    let registry = EXTENSION_TYPE_REGISTRY
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    registry.keys().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock as StdLazyLock;

    #[derive(Debug, PartialEq)]
    struct DemoExtensionType {
        label: String,
    }

    static DEMO_STORAGE_TYPE: StdLazyLock<DataType> = StdLazyLock::new(|| {
        DataType::Struct(vec![arrow_schema::Field::new("label", DataType::Utf8, false)].into())
    });

    impl SedonaExtensionType for DemoExtensionType {
        fn extension_name(&self) -> &'static str {
            "sedona.test.demo"
        }

        fn storage_type(&self) -> &DataType {
            &DEMO_STORAGE_TYPE
        }

        fn extension_metadata(&self) -> Option<String> {
            Some(self.label.clone())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn dyn_eq(&self, other: &dyn SedonaExtensionType) -> bool {
            other.as_any().downcast_ref::<Self>() == Some(self)
        }
    }

    #[test]
    fn register_and_lookup_roundtrip() {
        register_extension_type("sedona.test.demo.roundtrip", |ext| {
            Ok(Arc::new(DemoExtensionType {
                label: ext.extension_metadata.clone().unwrap_or_default(),
            }))
        });

        let factory = lookup_extension_type_factory("sedona.test.demo.roundtrip")
            .expect("factory should be registered");
        let ext = ExtensionType::new(
            "sedona.test.demo.roundtrip",
            DEMO_STORAGE_TYPE.clone(),
            Some("hello".to_string()),
        );
        let rebuilt = factory(&ext).unwrap();
        let demo = rebuilt
            .as_any()
            .downcast_ref::<DemoExtensionType>()
            .unwrap();
        assert_eq!(demo.label, "hello");
    }

    #[test]
    fn lookup_missing_returns_none() {
        assert!(lookup_extension_type_factory("sedona.test.demo.does-not-exist").is_none());
    }

    #[test]
    fn re_registration_overwrites_rather_than_errors() {
        register_extension_type("sedona.test.demo.overwrite", |_| {
            Ok(Arc::new(DemoExtensionType {
                label: "first".to_string(),
            }))
        });
        register_extension_type("sedona.test.demo.overwrite", |_| {
            Ok(Arc::new(DemoExtensionType {
                label: "second".to_string(),
            }))
        });
        let factory = lookup_extension_type_factory("sedona.test.demo.overwrite").unwrap();
        let ext = ExtensionType::new(
            "sedona.test.demo.overwrite",
            DEMO_STORAGE_TYPE.clone(),
            None,
        );
        let rebuilt = factory(&ext).unwrap();
        let demo = rebuilt
            .as_any()
            .downcast_ref::<DemoExtensionType>()
            .unwrap();
        assert_eq!(demo.label, "second");
    }

    #[test]
    fn dyn_eq_compares_by_value_not_pointer() {
        let a: Arc<dyn SedonaExtensionType> = Arc::new(DemoExtensionType {
            label: "x".to_string(),
        });
        let b: Arc<dyn SedonaExtensionType> = Arc::new(DemoExtensionType {
            label: "x".to_string(),
        });
        let c: Arc<dyn SedonaExtensionType> = Arc::new(DemoExtensionType {
            label: "y".to_string(),
        });
        assert!(!Arc::ptr_eq(&a, &b), "sanity check: distinct allocations");
        // assert_eq!/assert_ne! internally dereference through the shared
        // reference they hold (`*left_val == *right_val`), which tries to
        // move a non-Copy `Arc<dyn Trait>` out from behind `&Arc<dyn Trait>`.
        // Plain `==`/`!=` don't have this problem (standard auto-ref).
        assert!(a == b, "equal by value despite different allocations");
        assert!(a != c);
    }
}
