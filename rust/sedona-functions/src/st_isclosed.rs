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

use std::sync::Arc;

use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, internal_err};
use datafusion_expr::{scalar_doc_sections::DOC_SECTION_OTHER, Documentation, Volatility};
use geo_traits::{
    to_geo::{
        ToGeoGeometry, ToGeoLineString, ToGeoMultiLineString, ToGeoMultiPoint, ToGeoMultiPolygon,
        ToGeoPolygon,
    },
    GeometryTrait, LineStringTrait, PointTrait,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::error::SedonaGeometryError;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

use crate::executor::WkbExecutor;
pub fn st_isclosed_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_isclosed",
        vec![Arc::new(STIsClosed {})],
        Volatility::Immutable,
        Some(st_is_closed_doc()),
    )
}

fn st_is_closed_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return true if the geometry is closed",
        "ST_IsClosed (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_IsClosed(ST_GeomFromWKT('POLYGON CLOSED'))")
    .build()
}

#[derive(Debug)]
struct STIsClosed {}

impl SedonaScalarKernel for STIsClosed {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Boolean),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[datafusion_expr::ColumnarValue],
    ) -> Result<datafusion_expr::ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_value(invoke_scalar(&item)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(item: &Wkb) -> Result<bool> {
    is_geometry_closed(item).map_err(|e| {
        datafusion_common::error::DataFusionError::Execution(format!(
            "Failed to check if geometry is empty: {e}"
        ))
    })
}

fn is_geometry_closed(item: &Wkb) -> Result<bool> {
    match item.as_type() {
        geo_traits::GeometryType::Point(point) => Ok(point.coord().is_none()),
        geo_traits::GeometryType::LineString(linestring) => {
            Ok(linestring.to_line_string().is_closed())
        }
        geo_traits::GeometryType::Polygon(polygon) => {
            Ok(polygon.to_polygon().exterior().is_closed())
        }
        geo_traits::GeometryType::MultiLineString(multilinestring) => {
            Ok(multilinestring.to_multi_line_string().is_closed())
        }
        _ => internal_err!("Invalid geometry type"),
    }
}
