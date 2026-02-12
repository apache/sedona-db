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

use arrow_array::builder::BinaryBuilder;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{
    error::SedonaGeometryError,
    transform::{transform, CrsTransform},
    wkb_factory::WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::WkbExecutor;

/// ST_Force2D() scalar UDF
pub fn st_force2d_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_force2d",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STForce2D {
                is_geography: false,
            }),
            Arc::new(STForce2D { is_geography: true }),
        ]),
        Volatility::Immutable,
        Some(st_force2d_doc()),
    )
}

fn st_force2d_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Update coordinates of geom by a fixed offset",
        "ST_Force2D (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_Force2D(ST_GeomFromWKT('POINT Z (1 2 3)'))")
    .build()
}

#[derive(Debug)]
struct STForce2D {
    is_geography: bool,
}

impl SedonaScalarKernel for STForce2D {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = if self.is_geography {
            ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY)
        } else {
            ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY)
        };

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let trans = Force2DTransform {};
        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    transform(wkb, &trans, &mut builder)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    builder.append_value([]);
                }
                _ => {
                    builder.append_null();
                }
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct Force2DTransform {}

impl CrsTransform for Force2DTransform {
    fn output_dim(&self) -> Option<geo_traits::Dimensions> {
        Some(geo_traits::Dimensions::Xy)
    }

    fn transform_coord(
        &self,
        _coord: &mut (f64, f64),
    ) -> std::result::Result<(), SedonaGeometryError> {
        Ok(())
    }
}
