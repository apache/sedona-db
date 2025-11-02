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
use arrow_array::{
    builder::{BinaryBuilder, Int64Builder, ListBuilder, StructBuilder},
    ListArray,
};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::error::{DataFusionError, Result};
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

struct STDumpBuilder {
    builder: ListBuilder<StructBuilder>,
}

impl STDumpBuilder {
    fn new(num_iter: usize) -> Self {
        let path_builder =
            ListBuilder::with_capacity(Int64Builder::with_capacity(num_iter), num_iter);
        let geom_builder =
            BinaryBuilder::with_capacity(num_iter, WKB_MIN_PROBABLE_BYTES * num_iter);
        let struct_builder = StructBuilder::new(
            geometry_dump_fields(),
            vec![Box::new(path_builder), Box::new(geom_builder)],
        );
        let builder = ListBuilder::with_capacity(struct_builder, WKB_MIN_PROBABLE_BYTES * num_iter);

        Self { builder }
    }

    fn append(&mut self, is_valid: bool) {
        self.builder.append(is_valid);
    }

    fn append_null(&mut self) {
        self.builder.append_null();
    }

    fn struct_builder<'a>(&'a mut self) -> STDumpStructBuilder<'a> {
        STDumpStructBuilder {
            struct_builder: self.builder.values(),
        }
    }

    fn finish(&mut self) -> ListArray {
        self.builder.finish()
    }
}

struct STDumpStructBuilder<'a> {
    struct_builder: &'a mut StructBuilder,
}

impl<'a> STDumpStructBuilder<'a> {
    fn append(
        &mut self,
        path: &[i64],
        coord: impl CoordTrait<T = f64>, // TODO: acceept Geometry here
    ) -> std::result::Result<(), DataFusionError> {
        let path_builder = self
            .struct_builder
            .field_builder::<ListBuilder<Int64Builder>>(0)
            .expect("path field exists");
        let values_builder = path_builder.values();
        for value in path {
            values_builder.append_value(*value);
        }
        path_builder.append(true);

        let geom_builder = self
            .struct_builder
            .field_builder::<BinaryBuilder>(1)
            .expect("geom field exists");
        write_wkb_point_from_coord(geom_builder, coord)
            .map_err(|err| DataFusionError::External(Box::new(err)))?;
        geom_builder.append_value([]);

        self.struct_builder.append(true);

        Ok(())
    }

    fn append_null(&mut self) {
        self.struct_builder.append_null();
    }
}

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

        let mut builder = STDumpBuilder::new(executor.num_iterations());

        executor.execute_wkb_void(|maybe_wkb| {
            if let Some(wkb) = maybe_wkb {
                let mut struct_builder = builder.struct_builder();
                {
                    match wkb.as_type() {
                        geo_traits::GeometryType::Point(point) => {
                            if let Some(coord) = point.coord() {
                                struct_builder.append(&[1], coord)?;
                            } else {
                                struct_builder.append_null();
                            }
                        }
                        _ => todo!(),
                    }
                }

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
    use arrow_array::{Array, ArrayRef, Int64Array, ListArray, StructArray};
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
        let result = tester.invoke_array(input).unwrap();
        let expected: &[(&[i64], Option<&str>)] = &[(&[1], Some("POINT (1 2)"))];
        assert_dump_row(&result, 0, expected);

        let null_input = create_array(&[None], &sedona_type);
        let result = tester.invoke_array(null_input).unwrap();
        assert_dump_row_null(&result, 0);
    }

    fn assert_dump_row(result: &ArrayRef, row: usize, expected: &[(&[i64], Option<&str>)]) {
        let list_array = result
            .as_ref()
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("result should be a ListArray");
        assert!(
            !list_array.is_null(row),
            "row {row} should not be null in dump result"
        );
        let dumped = list_array.value(row);
        let dumped = dumped
            .as_ref()
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("list elements should be StructArray");
        assert_eq!(dumped.len(), expected.len());

        let path_array = dumped
            .column(0)
            .as_ref()
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("path should be a ListArray");
        assert_eq!(path_array.len(), expected.len());
        for (i, (expected_path, _)) in expected.iter().enumerate() {
            let path_array_value = path_array.value(i);
            let path_values = path_array_value
                .as_ref()
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("path values should be Int64Array");
            assert_eq!(
                path_values.len(),
                expected_path.len(),
                "unexpected path length at index {i}"
            );
            for (j, expected_value) in expected_path.iter().enumerate() {
                assert_eq!(
                    path_values.value(j),
                    *expected_value,
                    "unexpected path value at index {i}:{j}"
                );
            }
        }

        let expected_geom_values: Vec<Option<&str>> =
            expected.iter().map(|(_, geom)| *geom).collect();
        let expected_geom_array = create_array(&expected_geom_values, &WKB_GEOMETRY);
        assert_array_equal(dumped.column(1), &expected_geom_array);
    }

    fn assert_dump_row_null(result: &ArrayRef, row: usize) {
        let list_array = result
            .as_ref()
            .as_any()
            .downcast_ref::<ListArray>()
            .expect("result should be a ListArray");
        assert!(list_array.is_null(row), "row {row} should be null");
    }
}
