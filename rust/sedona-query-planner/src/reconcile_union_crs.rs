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

//! Reconcile equivalent geometry CRS metadata dropped by Union.
//!
//! DataFusion builds Union schemas with [`intersect_metadata_for_union`], which
//! keeps a metadata key only when every input has the exact same string value.
//! GeoArrow CRS metadata for the same CRS can differ in spelling (for example
//! `"OGC:CRS84"` versus its PROJJSON), so the intersection drops
//! `ARROW:extension:metadata` and the result reads as a geometry without a CRS.
//!
//! This analyzer restores a reconciled CRS when every Union input agrees via
//! [`crs_equals`] (keeping the first representation, consistent with argument
//! CRS matching in sedona-schema). See <https://github.com/apache/sedona-db/issues/1480>.

use std::sync::Arc;

use arrow_schema::Field;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{DFSchema, Result};
use datafusion_expr::{LogicalPlan, Union};
use datafusion_optimizer::AnalyzerRule;
use sedona_schema::crs::{Crs, first_if_all_equal};
use sedona_schema::datatypes::SedonaType;

/// Analyzer rule that restores equivalent CRS metadata on Union geometry fields.
#[derive(Debug, Default)]
pub struct ReconcileUnionCrs;

impl AnalyzerRule for ReconcileUnionCrs {
    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_up(reconcile_union_node).data()
    }

    fn name(&self) -> &str {
        "sedona.reconcile_union_crs"
    }
}

fn reconcile_union_node(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    let LogicalPlan::Union(union) = &plan else {
        return Ok(Transformed::no(plan));
    };

    let Some(new_schema) = reconcile_union_schema(union)? else {
        return Ok(Transformed::no(plan));
    };

    Ok(Transformed::yes(LogicalPlan::Union(Union {
        inputs: union.inputs.clone(),
        schema: Arc::new(new_schema),
    })))
}

/// Returns a new schema when at least one geometry field's CRS metadata was
/// restored; `None` when the Union schema already matches the reconciled types.
fn reconcile_union_schema(union: &Union) -> Result<Option<DFSchema>> {
    let field_count = union.schema.fields().len();
    let mut changed = false;
    let mut qualified_fields = Vec::with_capacity(field_count);

    for i in 0..field_count {
        let (qualifier, field) = union.schema.qualified_field(i);
        let input_fields: Vec<_> = union
            .inputs
            .iter()
            .map(|input| input.schema().field(i).as_ref().clone())
            .collect();

        let reconciled = match reconcile_geometry_field(field.as_ref(), &input_fields)? {
            Some(f) => {
                changed = true;
                Arc::new(f)
            }
            None => Arc::clone(field),
        };
        qualified_fields.push((qualifier.cloned(), reconciled));
    }

    if !changed {
        return Ok(None);
    }

    Ok(Some(DFSchema::new_with_metadata(
        qualified_fields,
        union.schema.metadata().clone(),
    )?))
}

fn reconcile_geometry_field(union_field: &Field, input_fields: &[Field]) -> Result<Option<Field>> {
    let mut input_types = Vec::with_capacity(input_fields.len());
    for field in input_fields {
        match SedonaType::from_storage_field(field)? {
            ty @ (SedonaType::Wkb(_, _)
            | SedonaType::WkbLarge(_, _)
            | SedonaType::WkbView(_, _)) => input_types.push(ty),
            // Non-geometry columns are left alone (DataFusion's intersection is fine).
            _ => return Ok(None),
        }
    }

    let Some(first) = input_types.first() else {
        return Ok(None);
    };

    // All inputs must share the same storage variant and edge type.
    if !input_types.iter().all(|ty| ty.match_signature(first)) {
        return Ok(None);
    }

    let crses: Vec<&Crs> = input_types.iter().map(|ty| ty.crs()).collect();
    let Some(reconciled_crs) = reconcile_equivalent_crs(&crses) else {
        return Ok(None);
    };

    let Some(reconciled_ty) = with_crs(first, reconciled_crs) else {
        return Ok(None);
    };

    let template = reconciled_ty.to_storage_field(union_field.name(), union_field.is_nullable())?;
    // Storage DataType is already unified by Union; only metadata needs restoring.
    if template.metadata() == union_field.metadata() {
        return Ok(None);
    }

    Ok(Some(
        Field::new(
            union_field.name(),
            union_field.data_type().clone(),
            union_field.is_nullable(),
        )
        .with_metadata(template.metadata().clone()),
    ))
}

/// When every CRS is equivalent under `crs_equals`, return the first one.
/// Returns `None` if any pair disagrees (including mixing a missing CRS with a
/// present one), or when every input already lacks a CRS (nothing to restore).
fn reconcile_equivalent_crs(crses: &[&Crs]) -> Option<Crs> {
    // Nothing to restore when every input already lacks a CRS.
    first_if_all_equal(crses.iter().copied()).filter(Option::is_some)
}

fn with_crs(ty: &SedonaType, crs: Crs) -> Option<SedonaType> {
    match ty {
        SedonaType::Wkb(edges, _) => Some(SedonaType::Wkb(*edges, crs)),
        SedonaType::WkbLarge(edges, _) => Some(SedonaType::WkbLarge(*edges, crs)),
        SedonaType::WkbView(edges, _) => Some(SedonaType::WkbView(*edges, crs)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::DFSchema;
    use datafusion_expr::EmptyRelation;
    use sedona_schema::crs::{OGC_CRS84_PROJJSON, deserialize_crs, lnglat};
    use sedona_schema::datatypes::{Edges, SedonaType};

    fn empty_with_field(field: Field) -> Arc<LogicalPlan> {
        let schema = Arc::new(
            DFSchema::from_unqualified_fields(vec![Arc::new(field)].into(), Default::default())
                .unwrap(),
        );
        Arc::new(LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema,
        }))
    }

    fn geom_field(name: &str, crs: Crs) -> Field {
        SedonaType::WkbView(Edges::Planar, crs)
            .to_storage_field(name, true)
            .unwrap()
    }

    #[test]
    fn restores_crs_when_authority_and_projjson_spellings_differ() {
        let auth = geom_field("g", deserialize_crs("OGC:CRS84").unwrap());
        let proj = geom_field("g", deserialize_crs(OGC_CRS84_PROJJSON).unwrap());
        assert_ne!(
            auth.metadata().get("ARROW:extension:metadata"),
            proj.metadata().get("ARROW:extension:metadata"),
            "precondition: metadata strings must differ"
        );

        let union = Union::try_new(vec![empty_with_field(auth), empty_with_field(proj)]).unwrap();
        let before = SedonaType::from_storage_field(union.schema.field(0)).unwrap();
        assert_eq!(
            before.crs(),
            &Crs::None,
            "DataFusion intersection drops differing CRS metadata"
        );

        let rule = ReconcileUnionCrs;
        let out = rule
            .analyze(LogicalPlan::Union(union), &ConfigOptions::default())
            .unwrap();
        let LogicalPlan::Union(out_union) = out else {
            panic!("expected Union");
        };
        let after = SedonaType::from_storage_field(out_union.schema.field(0)).unwrap();
        assert_eq!(after.crs(), &lnglat());
        assert!(matches!(after, SedonaType::WkbView(Edges::Planar, _)));
    }

    #[test]
    fn leaves_matching_metadata_alone() {
        let a = geom_field("g", deserialize_crs("OGC:CRS84").unwrap());
        let b = geom_field("g", deserialize_crs("OGC:CRS84").unwrap());
        let union = Union::try_new(vec![empty_with_field(a), empty_with_field(b)]).unwrap();
        let before_meta = union.schema.field(0).metadata().clone();

        let rule = ReconcileUnionCrs;
        let out = rule
            .analyze(LogicalPlan::Union(union), &ConfigOptions::default())
            .unwrap();
        let LogicalPlan::Union(out_union) = out else {
            panic!("expected Union");
        };
        assert_eq!(out_union.schema.field(0).metadata(), &before_meta);
    }

    #[test]
    fn does_not_invent_crs_when_inputs_disagree() {
        let a = geom_field("g", deserialize_crs("OGC:CRS84").unwrap());
        let b = geom_field("g", deserialize_crs("EPSG:3857").unwrap());
        let union = Union::try_new(vec![empty_with_field(a), empty_with_field(b)]).unwrap();
        let before = SedonaType::from_storage_field(union.schema.field(0)).unwrap();
        assert_eq!(before.crs(), &Crs::None);

        let rule = ReconcileUnionCrs;
        let out = rule
            .analyze(LogicalPlan::Union(union), &ConfigOptions::default())
            .unwrap();
        let LogicalPlan::Union(out_union) = out else {
            panic!("expected Union");
        };
        let after = SedonaType::from_storage_field(out_union.schema.field(0)).unwrap();
        assert_eq!(after.crs(), &Crs::None);
    }

    #[test]
    fn returns_first_when_equivalent() {
        let crses = [
            deserialize_crs(OGC_CRS84_PROJJSON).unwrap(),
            deserialize_crs("OGC:CRS84").unwrap(),
        ];
        let refs: Vec<&Crs> = crses.iter().collect();
        let out = reconcile_equivalent_crs(&refs).unwrap();
        // Keep the first representation (PROJJSON here); do not normalize to
        // authority:code. Values still compare equal under crs_equals.
        assert_eq!(
            out.as_ref().unwrap().to_json(),
            crses[0].as_ref().unwrap().to_json()
        );
        assert_ne!(
            out.as_ref().unwrap().to_json(),
            crses[1].as_ref().unwrap().to_json()
        );
        assert_eq!(out, crses[1]);
    }
}
