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
use arrow_array::builder::{BinaryBuilder, Int64Builder, ListBuilder, StructBuilder};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{CoordTrait, GeometryTrait, PointTrait};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::{
    error::SedonaGeometryError,
    wkb_factory::{write_wkb_coord_trait, write_wkb_point_header, WKB_MIN_PROBABLE_BYTES},
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{io::Write, sync::Arc};

use crate::executor::WkbExecutor;

/// ST_Dump() scalar UDF
///
/// Native implementation to get all the points of a geometry as MULTIPOINT
pub fn st_dump_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_dump",
        vec![Arc::new(STDump)],
        Volatility::Immutable,
        Some(st_dump_doc()),
    )
}

fn st_dump_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Extracts the components of a geometry.",
        "ST_Dump (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_Dump(ST_GeomFromWKT('MULTIPOINT (0 1, 2 3, 4 5)'))")
    .build()
}

#[derive(Debug)]
struct STDump;

impl SedonaScalarKernel for STDump {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], geometry_dump_type());
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);

        let num_iter = executor.num_iterations();
        let path_builder =
            ListBuilder::with_capacity(Int64Builder::with_capacity(num_iter), num_iter);
        let geom_builder =
            BinaryBuilder::with_capacity(num_iter, WKB_MIN_PROBABLE_BYTES * num_iter);
        let struct_builder = StructBuilder::new(
            geometry_dump_fields(),
            vec![Box::new(path_builder), Box::new(geom_builder)],
        );
        let mut builder =
            ListBuilder::with_capacity(struct_builder, WKB_MIN_PROBABLE_BYTES * num_iter);

        executor.execute_wkb_void(|maybe_wkb| {
            if let Some(wkb) = maybe_wkb {
                let struct_builder = builder.values();

                // Test: This should add { path: [1], geom: POINT } for a POINT geometry
                match wkb.as_type() {
                    geo_traits::GeometryType::Point(point) => match point.coord() {
                        Some(coord) => {
                            // TODO: struct_builder cannot borrow more than once. But this is too lengthy to be inlined here.

                            // Write path
                            {
                                let path_array_builder = struct_builder
                                    .field_builder::<ListBuilder<Int64Builder>>(0)
                                    .unwrap();
                                let path_builder = path_array_builder.values();
                                path_builder.append_value(1);
                                path_array_builder.append(true);
                            }

                            // Write geom
                            {
                                let geom_builder =
                                    struct_builder.field_builder::<BinaryBuilder>(1).unwrap();

                                write_wkb_point_from_coord(geom_builder, coord).unwrap();
                                geom_builder.append_value([]);
                            }

                            struct_builder.append(true);
                        }
                        None => struct_builder.append_null(),
                    },
                    _ => todo!(),
                }

                struct_builder.finish();
                builder.append(true);
            } else {
                builder.append_null();
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn geometry_dump_fields() -> Fields {
    let path = Field::new(
        "path",
        DataType::List(Field::new("item", DataType::Int64, true).into()),
        true,
    );
    let geom = WKB_GEOMETRY.to_storage_field("geom", true).unwrap();
    vec![path, geom].into()
}

fn geometry_dump_type() -> SedonaType {
    let fields = geometry_dump_fields();
    let struct_type = DataType::Struct(fields);

    SedonaType::Arrow(DataType::List(Field::new("item", struct_type, true).into()))
}

fn write_wkb_point_from_coord(
    buf: &mut impl Write,
    coord: impl CoordTrait<T = f64>,
) -> Result<(), SedonaGeometryError> {
    write_wkb_point_header(buf, coord.dim())?;
    write_wkb_coord_trait(buf, &coord)
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_dump_udf: ScalarUDF = st_dump_udf().into();
        assert_eq!(st_dump_udf.name(), "st_dump");
        assert!(st_dump_udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_dump_udf().into(), vec![sedona_type.clone()]);

        let input = create_array(&[Some("POINT (1 2)")], &sedona_type);

        // let expected = create_array(&[Some("POINT (1 2)")], &WKB_GEOMETRY);

        let result = tester.invoke_array(input.clone()).unwrap();
        // assert_array_equal(&result, &expected);
    }
}
