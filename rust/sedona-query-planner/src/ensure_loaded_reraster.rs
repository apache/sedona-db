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
//! of `RS_EnsureLoaded` (async) loses its `sedona.raster` extension type and
//! downstream raster functions stop recognising it.
//!
//! As a stopgap, the EnsureLoaded optimizer rule wraps each async load in
//! this sync `sd_reraster` UDF — `sd_reraster(rs_ensureloaded(x))`. A
//! sync `ScalarFunctionExpr` *does* preserve `return_field`, so this node
//! re-applies the raster extension type onto the (otherwise stripped)
//! struct. The data passes through untouched (identity).
//!
//! **Removal:** once #22662 is fixed and downstreamed, delete this whole
//! file, its `mod` declaration, and revert the three `DF-22662`-tagged
//! touchpoints in `ensure_loaded.rs` (so the rule emits `rs_ensureloaded(x)`
//! directly again). `grep DF-22662` finds every site.

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow_schema::{DataType, FieldRef};
use datafusion_common::Result;
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_schema::datatypes::SedonaType;

/// Registry/UDF name of the re-stamp helper.
pub const RERASTER_NAME: &str = "sd_reraster";

/// Build the sync re-stamp UDF. Cheap and stateless; the rule constructs one
/// per rewrite and clones the `Arc` for each wrapped argument. `Stable`
/// volatility so CSE can deduplicate identical `sd_reraster(...)` subtrees.
pub fn reraster_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(RsReRaster::new()))
}

/// Sync identity UDF that forces the `sedona.raster` extension type onto its
/// argument's output field. See the module docs (DF-22662).
#[derive(Debug)]
struct RsReRaster {
    signature: Signature,
}

impl RsReRaster {
    fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Stable),
        }
    }
}

// Identity by name so CSE can dedup `sd_reraster(...)` across exprs.
impl PartialEq for RsReRaster {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for RsReRaster {}
impl Hash for RsReRaster {
    fn hash<H: Hasher>(&self, state: &mut H) {
        RERASTER_NAME.hash(state);
    }
}

impl ScalarUDFImpl for RsReRaster {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        RERASTER_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // return_field_from_args is authoritative (it carries the extension
        // metadata a bare DataType would drop).
        sedona_internal_err!("sd_reraster::return_type should not be called")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        // Force the raster extension type regardless of the input field. At
        // physical planning the input is the async output, whose `sedona.raster`
        // metadata has already been stripped (DF-22662), so an identity
        // return_field would NOT restore it — we must stamp it unconditionally.
        let (name, nullable) = args
            .arg_fields
            .first()
            .map(|f| (f.name().as_str(), f.is_nullable()))
            .unwrap_or(("raster", true));
        Ok(Arc::new(
            SedonaType::Raster.to_storage_field(name, nullable)?,
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // Identity on the data; only the output field's type changes.
        let Some(arg) = args.args.into_iter().next() else {
            return sedona_internal_err!("sd_reraster expects exactly one argument");
        };
        Ok(arg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Field;

    #[test]
    fn reraster_forces_raster_type_on_a_bare_struct_field() {
        // Models the physical-planning input: a struct that has lost its
        // sedona.raster extension (the DF-22662 symptom). reraster must
        // stamp it back.
        let udf = RsReRaster::new();
        let raster_storage = SedonaType::Raster
            .to_storage_field("x", true)
            .unwrap()
            .data_type()
            .clone();
        let bare = Arc::new(Field::new("loaded", raster_storage, true));
        let args = ReturnFieldArgs {
            arg_fields: &[bare],
            scalar_arguments: &[None],
        };
        let out = udf.return_field_from_args(args).unwrap();
        assert_eq!(
            SedonaType::from_storage_field(&out).unwrap(),
            SedonaType::Raster,
            "reraster must re-stamp the raster extension type: {out:?}"
        );
    }
}
