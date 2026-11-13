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

//! SQL type planning for Sedona's geometry and geography extension types.

use std::sync::Arc;

use arrow_schema::FieldRef;
use datafusion::sql::sqlparser::ast;
use datafusion_common::{plan_datafusion_err, Result};
use datafusion_expr::planner::TypePlanner;
use sedona_functions::st_setsrid::validate_crs_for_type;
use sedona_geometry::types::Edges;
use sedona_schema::{crs::deserialize_crs, datatypes::SedonaType};

const DEFAULT_CRS: &str = "OGC:CRS84";

/// Plans SQL `GEOMETRY` and `GEOGRAPHY` types as GeoArrow WKB extension types.
///
/// `GEOMETRY` accepts an optional CRS. `GEOGRAPHY` accepts an optional CRS and
/// spherical edge interpolation.
#[derive(Debug, Default)]
pub struct SedonaTypePlanner;

impl TypePlanner for SedonaTypePlanner {
    fn plan_type_field(&self, sql_type: &ast::DataType) -> Result<Option<FieldRef>> {
        let ast::DataType::Custom(name, modifiers) = sql_type else {
            return Ok(None);
        };

        let [name_part] = name.0.as_slice() else {
            return Ok(None);
        };
        let Some(type_name) = name_part.as_ident().map(|ident| ident.value.as_str()) else {
            return Ok(None);
        };

        let (edges, crs) = if type_name.eq_ignore_ascii_case("geometry") {
            require_at_most("GEOMETRY", modifiers, 1)?;
            let crs = modifiers.first().map(String::as_str).unwrap_or(DEFAULT_CRS);
            (Edges::Planar, deserialize_crs(crs)?)
        } else if type_name.eq_ignore_ascii_case("geography") {
            require_at_most("GEOGRAPHY", modifiers, 2)?;
            let crs = modifiers.first().map(String::as_str).unwrap_or(DEFAULT_CRS);
            let edges = parse_edges(modifiers.get(1).map(String::as_str).unwrap_or("spherical"))?;
            (edges, deserialize_crs(crs)?)
        } else {
            return Ok(None);
        };

        let sedona_type = SedonaType::Wkb(edges, crs);
        validate_crs_for_type(sedona_type.crs(), &sedona_type)?;
        let field = sedona_type.to_storage_field("", true)?;
        Ok(Some(Arc::new(field)))
    }
}

fn require_at_most(type_name: &str, modifiers: &[String], maximum: usize) -> Result<()> {
    if modifiers.len() > maximum {
        return Err(plan_datafusion_err!(
            "{type_name} accepts at most {maximum} type modifier(s), got {}",
            modifiers.len()
        ));
    }
    Ok(())
}

fn parse_edges(value: &str) -> Result<Edges> {
    if value.eq_ignore_ascii_case("spherical") {
        Ok(Edges::Spherical)
    } else {
        Err(plan_datafusion_err!(
            "Invalid GEOGRAPHY edge interpolation '{value}'; expected 'spherical'"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::SedonaContext;
    use datafusion::sql::sqlparser::ast::{Ident, ObjectName};
    use sedona_schema::{crs::lnglat, datatypes::SedonaType};

    fn custom_type(name: &str, modifiers: &[&str]) -> ast::DataType {
        ast::DataType::Custom(
            ObjectName::from(Ident::new(name)),
            modifiers.iter().map(|value| (*value).to_string()).collect(),
        )
    }

    fn plan(sql_type: ast::DataType) -> SedonaType {
        let field = SedonaTypePlanner
            .plan_type_field(&sql_type)
            .unwrap()
            .unwrap();
        SedonaType::from_storage_field(&field).unwrap()
    }

    #[test]
    fn plans_geometry() {
        assert_eq!(
            plan(custom_type("geometry", &[])),
            SedonaType::Wkb(Edges::Planar, lnglat())
        );
        assert_eq!(
            plan(custom_type("GEOMETRY", &["EPSG:3857"])),
            SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap())
        );
    }

    #[test]
    fn plans_geography() {
        assert_eq!(
            plan(custom_type("GeOgRaPhY", &[])),
            SedonaType::Wkb(Edges::Spherical, lnglat())
        );
        assert_eq!(
            plan(custom_type("GEOGRAPHY", &["EPSG:4269", "SPHERICAL"])),
            SedonaType::Wkb(Edges::Spherical, deserialize_crs("EPSG:4269").unwrap())
        );
    }

    #[test]
    fn ignores_other_or_qualified_types() {
        assert!(SedonaTypePlanner
            .plan_type_field(&custom_type("other", &[]))
            .unwrap()
            .is_none());

        let qualified = ast::DataType::Custom(
            ObjectName::from(vec![Ident::new("schema"), Ident::new("geometry")]),
            vec![],
        );
        assert!(SedonaTypePlanner
            .plan_type_field(&qualified)
            .unwrap()
            .is_none());
    }

    #[test]
    fn rejects_invalid_modifiers() {
        let error = SedonaTypePlanner
            .plan_type_field(&custom_type("geometry", &["EPSG:3857", "planar"]))
            .unwrap_err();
        assert!(error.to_string().contains("at most 1 type modifier"));

        let error = SedonaTypePlanner
            .plan_type_field(&custom_type("geography", &["OGC:CRS84", "rhumb"]))
            .unwrap_err();
        assert!(error.to_string().contains("expected 'spherical'"));

        let error = SedonaTypePlanner
            .plan_type_field(&custom_type("geography", &["OGC:CRS84", "planar"]))
            .unwrap_err();
        assert!(error.to_string().contains("expected 'spherical'"));

        let error = SedonaTypePlanner
            .plan_type_field(&custom_type("geography", &["EPSG:3857"]))
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("Can't assign non-geographic CRS"));
    }

    #[tokio::test]
    async fn planner_is_registered_with_sedona_context() -> Result<()> {
        let ctx = SedonaContext::new();
        ctx.sql(
            "CREATE TABLE spatial_types (\
                geom GEOMETRY, \
                geog GEOGRAPHY, \
                projected GEOMETRY('EPSG:3857'), \
                nad83_geog GEOGRAPHY('EPSG:4269', spherical)\
            )",
        )
        .await?
        .collect()
        .await?;

        let table = ctx.table("spatial_types").await?;
        let schema = table.schema();
        assert_eq!(
            SedonaType::from_storage_field(schema.field_with_name(None, "geom")?.as_ref())?,
            SedonaType::Wkb(Edges::Planar, lnglat())
        );
        assert_eq!(
            SedonaType::from_storage_field(schema.field_with_name(None, "geog")?.as_ref())?,
            SedonaType::Wkb(Edges::Spherical, lnglat())
        );
        assert_eq!(
            SedonaType::from_storage_field(schema.field_with_name(None, "projected")?.as_ref())?,
            SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857")?)
        );
        assert_eq!(
            SedonaType::from_storage_field(schema.field_with_name(None, "nad83_geog")?.as_ref())?,
            SedonaType::Wkb(Edges::Spherical, deserialize_crs("EPSG:4269")?)
        );

        let error = ctx
            .sql("SELECT CAST(ST_Point(0, 1) AS GEOGRAPHY('EPSG:3857'))")
            .await
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("Can't assign non-geographic CRS"));
        Ok(())
    }
}
