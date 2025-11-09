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

use crate::executor::WkbExecutor;
use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::GeometryTrait;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::{
    bounds::{geo_traits_bounds_m, geo_traits_bounds_xy, geo_traits_bounds_z},
    interval::{Interval, IntervalTrait},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

pub fn st_xmin_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_xmin",
        vec![Arc::new(STXyzmMinMax {
            dim: "x",
            is_max: false,
        })],
        Volatility::Immutable,
        Some(st_xyzm_minmax_doc("x", false)),
    )
}

pub fn st_xmax_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_xmax",
        vec![Arc::new(STXyzmMinMax {
            dim: "x",
            is_max: true,
        })],
        Volatility::Immutable,
        Some(st_xyzm_minmax_doc("x", true)),
    )
}

pub fn st_ymin_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_ymin",
        vec![Arc::new(STXyzmMinMax {
            dim: "y",
            is_max: false,
        })],
        Volatility::Immutable,
        Some(st_xyzm_minmax_doc("y", false)),
    )
}

pub fn st_ymax_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_ymax",
        vec![Arc::new(STXyzmMinMax {
            dim: "y",
            is_max: true,
        })],
        Volatility::Immutable,
        Some(st_xyzm_minmax_doc("y", true)),
    )
}

pub fn st_zmin_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_zmin",
        vec![Arc::new(STXyzmMinMax {
            dim: "z",
            is_max: false,
        })],
        Volatility::Immutable,
        Some(st_xyzm_minmax_doc("z", false)),
    )
}

pub fn st_zmax_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_zmax",
        vec![Arc::new(STXyzmMinMax {
            dim: "z",
            is_max: true,
        })],
        Volatility::Immutable,
        Some(st_xyzm_minmax_doc("z", true)),
    )
}

pub fn st_mmin_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_mmin",
        vec![Arc::new(STXyzmMinMax {
            dim: "m",
            is_max: false,
        })],
        Volatility::Immutable,
        Some(st_xyzm_minmax_doc("m", false)),
    )
}

pub fn st_mmax_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_mmax",
        vec![Arc::new(STXyzmMinMax {
            dim: "m",
            is_max: true,
        })],
        Volatility::Immutable,
        Some(st_xyzm_minmax_doc("m", true)),
    )
}

fn st_xyzm_minmax_doc(dim: &str, is_max: bool) -> Documentation {
    let min_or_max = if is_max { "Max" } else { "Min" };
    let func_name = format!("ST_{}{}", dim.to_uppercase(), min_or_max);
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!(
            "Return the {} of the {} dimension of the geometry",
            min_or_max.to_lowercase(),
            dim.to_uppercase()
        ),
        format!("{func_name} (A: Geometry)"),
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example(format!(
        "SELECT {func_name}(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'))",
    ))
    .build()
}

#[derive(Debug)]
struct STXyzmMinMax {
    dim: &'static str,
    is_max: bool,
}

impl SedonaScalarKernel for STXyzmMinMax {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_option(invoke_scalar(&item, self.dim, self.is_max)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    item: impl GeometryTrait<T = f64>,
    dim: &'static str,
    is_max: bool,
) -> Result<Option<f64>> {
    let interval: Interval = match dim {
        "x" => {
            let xy_bounds = geo_traits_bounds_xy(item)
                .map_err(|e| DataFusionError::Internal(format!("Error updating bounds: {e}")))?;
            Interval::try_from(*xy_bounds.x()).map_err(|e| {
                DataFusionError::Internal(format!("Error converting to interval: {e}"))
            })?
        }
        "y" => {
            let xy_bounds = geo_traits_bounds_xy(item)
                .map_err(|e| DataFusionError::Internal(format!("Error updating bounds: {e}")))?;
            *xy_bounds.y()
        }
        "z" => {
            let z_bounds = geo_traits_bounds_z(item)
                .map_err(|e| DataFusionError::Internal(format!("Error updating bounds: {e}")))?;
            z_bounds
        }
        "m" => {
            let m_bounds = geo_traits_bounds_m(item)
                .map_err(|e| DataFusionError::Internal(format!("Error updating bounds: {e}")))?;
            m_bounds
        }
        _ => sedona_internal_err!("unexpected dim index")?,
    };

    if interval.is_empty() {
        return Ok(None);
    }
    Ok(Some(if is_max { interval.hi() } else { interval.lo() }))
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_xmin_udf().into();
        assert_eq!(udf.name(), "st_xmin");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_xmax_udf().into();
        assert_eq!(udf.name(), "st_xmax");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_ymin_udf().into();
        assert_eq!(udf.name(), "st_ymin");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_ymax_udf().into();
        assert_eq!(udf.name(), "st_ymax");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_zmin_udf().into();
        assert_eq!(udf.name(), "st_zmin");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_zmax_udf().into();
        assert_eq!(udf.name(), "st_zmax");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_mmin_udf().into();
        assert_eq!(udf.name(), "st_mmin");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_mmax_udf().into();
        assert_eq!(udf.name(), "st_mmax");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let xmin_tester = ScalarUdfTester::new(st_xmin_udf().into(), vec![sedona_type.clone()]);
        let ymin_tester = ScalarUdfTester::new(st_ymin_udf().into(), vec![sedona_type.clone()]);
        let xmax_tester = ScalarUdfTester::new(st_xmax_udf().into(), vec![sedona_type.clone()]);
        let ymax_tester = ScalarUdfTester::new(st_ymax_udf().into(), vec![sedona_type.clone()]);

        let zmin_tester = ScalarUdfTester::new(st_zmin_udf().into(), vec![sedona_type.clone()]);
        let zmax_tester = ScalarUdfTester::new(st_zmax_udf().into(), vec![sedona_type.clone()]);
        let mmin_tester = ScalarUdfTester::new(st_mmin_udf().into(), vec![sedona_type.clone()]);
        let mmax_tester = ScalarUdfTester::new(st_mmax_udf().into(), vec![sedona_type.clone()]);

        xmin_tester.assert_return_type(DataType::Float64);
        ymin_tester.assert_return_type(DataType::Float64);
        xmax_tester.assert_return_type(DataType::Float64);
        ymax_tester.assert_return_type(DataType::Float64);

        zmin_tester.assert_return_type(DataType::Float64);
        zmax_tester.assert_return_type(DataType::Float64);
        mmin_tester.assert_return_type(DataType::Float64);
        mmax_tester.assert_return_type(DataType::Float64);

        let input_wkt = "POLYGON ((-1 0, 0 -2, 3 1, 0 4))";
        xmin_tester.assert_scalar_result_equals(xmin_tester.invoke_scalar(input_wkt).unwrap(), -1);
        ymin_tester.assert_scalar_result_equals(ymin_tester.invoke_scalar(input_wkt).unwrap(), -2);
        xmax_tester.assert_scalar_result_equals(xmax_tester.invoke_scalar(input_wkt).unwrap(), 3);
        ymax_tester.assert_scalar_result_equals(ymax_tester.invoke_scalar(input_wkt).unwrap(), 4);

        zmin_tester.assert_scalar_result_equals(
            zmin_tester.invoke_scalar(input_wkt).unwrap(),
            ScalarValue::Null,
        );
        zmax_tester.assert_scalar_result_equals(
            zmax_tester.invoke_scalar(input_wkt).unwrap(),
            ScalarValue::Null,
        );
        mmin_tester.assert_scalar_result_equals(
            mmin_tester.invoke_scalar(input_wkt).unwrap(),
            ScalarValue::Null,
        );
        mmax_tester.assert_scalar_result_equals(
            mmax_tester.invoke_scalar(input_wkt).unwrap(),
            ScalarValue::Null,
        );

        // Test example with zm and coordinates
        let input_wkt = "LINESTRING ZM (1 2 3 4, 5 6 7 8)";
        xmin_tester.assert_scalar_result_equals(xmin_tester.invoke_scalar(input_wkt).unwrap(), 1);
        xmax_tester.assert_scalar_result_equals(xmax_tester.invoke_scalar(input_wkt).unwrap(), 5);
        ymin_tester.assert_scalar_result_equals(ymin_tester.invoke_scalar(input_wkt).unwrap(), 2);
        ymax_tester.assert_scalar_result_equals(ymax_tester.invoke_scalar(input_wkt).unwrap(), 6);

        zmin_tester.assert_scalar_result_equals(zmin_tester.invoke_scalar(input_wkt).unwrap(), 3);
        zmax_tester.assert_scalar_result_equals(zmax_tester.invoke_scalar(input_wkt).unwrap(), 7);
        mmin_tester.assert_scalar_result_equals(mmin_tester.invoke_scalar(input_wkt).unwrap(), 4);
        mmax_tester.assert_scalar_result_equals(mmax_tester.invoke_scalar(input_wkt).unwrap(), 8);

        // Test array input
        let input_wkt = vec![None, Some("POINT EMPTY"), Some("GEOMETRYCOLLECTION EMPTY")];

        let expected: ArrayRef = arrow_array!(Float64, [None, None, None]);
        assert_array_equal(
            &xmin_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
        assert_array_equal(
            &ymin_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
        assert_array_equal(
            &xmax_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
        assert_array_equal(
            &ymax_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );

        assert_array_equal(
            &zmin_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
        assert_array_equal(
            &zmax_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
        assert_array_equal(
            &mmin_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
        assert_array_equal(
            &mmax_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
    }
}
