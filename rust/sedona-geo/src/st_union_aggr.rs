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
use std::{collections::HashMap, sync::Arc, vec};

use arrow_array::{ArrayRef, StructArray};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{
    error::{DataFusionError, Result},
    ScalarValue,
};
use datafusion_expr::{Accumulator, ColumnarValue};
use geo::BooleanOps;
use geo_traits::to_geo::ToGeoGeometry;
use sedona_expr::aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef};
use sedona_functions::executor::WkbExecutor;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::writer::write_geometry;
use wkb::Endianness;
use wkb::{reader::Wkb, writer::WriteOptions};

/// ST_Union_Aggr() implementation
pub fn st_union_aggr_impl() -> SedonaAccumulatorRef {
    Arc::new(STUnionAggr {})
}

#[derive(Debug)]
struct STUnionAggr {}

impl SedonaAccumulator for STUnionAggr {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // Return Struct{"geoarrow.wkb": Binary} for compatibility with Spark/Comet
        let mut field = Field::new("geoarrow.wkb", DataType::Binary, true);
        field.set_metadata(HashMap::from([
            (
                "ARROW:extension:name".to_string(),
                "geoarrow.wkb".to_string(),
            ),
            ("ARROW:extension:metadata".to_string(), "{}".to_string()),
        ]));
        let r_type = SedonaType::Arrow(DataType::Struct(vec![field].into()));
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], r_type);
        matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(UnionAccumulator::new(args[0].clone())))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![Arc::new(
            WKB_GEOMETRY.to_storage_field("union", true)?,
        )])
    }
}

#[derive(Debug)]
struct UnionAccumulator {
    input_type: SedonaType,
    current_union: Option<geo::Geometry>,
}

impl UnionAccumulator {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            current_union: None,
        }
    }

    fn update_union(&mut self, geom: &Wkb) -> Result<()> {
        let geo_geom = geom.to_geometry();

        if self.current_union.is_none() {
            self.init_union(&geo_geom);
            return Ok(());
        }

        // Clone the current geometry to avoid borrowing issues
        let current_geo_geom = self.current_union.take().unwrap();

        match (&current_geo_geom, geo_geom) {
            (geo::Geometry::MultiPolygon(multi_poly1), geo_geom2) => {
                // Convert the second geometry to MultiPolygon if needed
                let multi_poly2 = match geo_geom2 {
                    geo::Geometry::Polygon(poly) => geo::MultiPolygon(vec![poly]),
                    geo::Geometry::MultiPolygon(multi) => multi.clone(),
                    _ => {
                        return Err(DataFusionError::Internal(
                            "Unsupported geometry type for union operation".to_string(),
                        ));
                    }
                };

                self.current_union =
                    Some(geo::Geometry::MultiPolygon(multi_poly1.union(&multi_poly2)));
                Ok(())
            }
            _ => unreachable!("Geometry of the internal state is always a MultiPolygon"),
        }
    }

    fn init_union(&mut self, geom: &geo::Geometry) {
        match geom {
            geo::Geometry::Polygon(poly) => {
                let multi_poly = geo::MultiPolygon(vec![poly.clone()]);
                self.current_union = Some(geo::Geometry::MultiPolygon(multi_poly));
            }
            geo::Geometry::MultiPolygon(multi) => {
                self.current_union = Some(geo::Geometry::MultiPolygon(multi.clone()));
            }
            _ => {
                self.current_union = None;
            }
        }
    }

    fn geometry_to_wkb(&self, geom: &geo::Geometry) -> Option<Vec<u8>> {
        let mut wkb_bytes = Vec::new();
        match write_geometry(
            &mut wkb_bytes,
            geom,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        ) {
            Ok(_) => Some(wkb_bytes),
            Err(_) => None,
        }
    }

    fn make_wkb_result(&self) -> Result<Option<Vec<u8>>> {
        // Convert the stored geometry to WKB bytes only when needed
        Ok(self
            .current_union
            .as_ref()
            .and_then(|geom| self.geometry_to_wkb(geom)))
    }

    fn execute_update(&mut self, executor: WkbExecutor) -> Result<()> {
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                self.update_union(&item)?;
            }
            Ok(())
        })?;
        Ok(())
    }
}

impl Accumulator for UnionAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Err(DataFusionError::Internal(
                "No input arrays provided to accumulator in update_batch".to_string(),
            ));
        }
        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &args);
        self.execute_update(executor)?;
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let wkb = self.make_wkb_result()?;
        // Wrap Binary in Struct{"geoarrow.wkb": Binary} for compatibility with Spark/Comet
        let binary_scalar = ScalarValue::Binary(wkb);

        let mut field = Field::new("geoarrow.wkb", DataType::Binary, true);
        field.set_metadata(HashMap::from([
            (
                "ARROW:extension:name".to_string(),
                "geoarrow.wkb".to_string(),
            ),
            ("ARROW:extension:metadata".to_string(), "{}".to_string()),
        ]));

        let binary_array = binary_scalar.to_array()?;
        let struct_array = StructArray::from(vec![(Arc::new(field), binary_array)]);

        Ok(ScalarValue::Struct(Arc::new(struct_array)))
    }

    fn size(&self) -> usize {
        let mut size = size_of_val(self);

        // Add size of the geometry data if it exists
        if let Some(geo::Geometry::MultiPolygon(mp)) = &self.current_union {
            for poly in &mp.0 {
                // Count exterior ring points
                size += size_of::<geo::Coord>() * poly.exterior().0.len();

                // Count interior ring points
                for ring in poly.interiors() {
                    size += size_of::<geo::Coord>() * ring.0.len();
                }
            }
        }

        size
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let wkb = self.make_wkb_result()?;
        Ok(vec![ScalarValue::Binary(wkb)])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        // Check input length (expecting 1 state field)
        if states.is_empty() {
            return Err(DataFusionError::Internal(
                "No input arrays provided to accumulator in merge_batch".to_string(),
            ));
        }
        let array = &states[0];
        let args = [ColumnarValue::Array(array.clone())];
        let arg_types = [WKB_GEOMETRY.clone()];
        let executor = WkbExecutor::new(&arg_types, &args);
        self.execute_update(executor)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;
    use sedona_functions::st_union_aggr::st_union_aggr_udf;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{
        compare::assert_scalar_equal_wkb_geometry_topologically, testers::AggregateUdfTester,
    };

    #[rstest]
    fn polygon_polygon_cases(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let mut udaf = st_union_aggr_udf();
        udaf.add_kernel(st_union_aggr_impl());

        let tester = AggregateUdfTester::new(udaf.into(), vec![sedona_type.clone()]);

        // Expected return type: Struct{"geoarrow.wkb": Binary} for Spark/Comet compatibility
        let mut expected_field = Field::new("geoarrow.wkb", DataType::Binary, true);
        expected_field.set_metadata(HashMap::from([
            ("ARROW:extension:name".to_string(), "geoarrow.wkb".to_string()),
            ("ARROW:extension:metadata".to_string(), "{}".to_string()),
        ]));
        let expected_return_type = SedonaType::Arrow(DataType::Struct(vec![expected_field].into()));
        assert_eq!(tester.return_type().unwrap(), expected_return_type);

        // Basic polygon union
        let batches = vec![
            vec![Some("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")],
            vec![Some("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))")],
        ];

        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("MULTIPOLYGON(((0 0, 2 0, 2 1, 3 1, 3 3, 1 3, 1 2, 0 2, 0 0)))"),
        );

        // Empty input
        assert_scalar_equal_wkb_geometry_topologically(&tester.aggregate(&vec![]).unwrap(), None);

        // Single polygon input
        assert_scalar_equal_wkb_geometry_topologically(
            &tester
                .aggregate_wkt(vec![vec![Some("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")]])
                .unwrap(),
            Some("MULTIPOLYGON(((0 0, 2 0, 2 2, 0 2, 0 0)))"),
        );

        // Non-intersecting polygons should still produce a union
        let non_intersecting = vec![
            vec![Some("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")],
            vec![Some("POLYGON((2 2, 3 2, 3 3, 2 3, 2 2))")],
        ];
        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(non_intersecting).unwrap(),
            Some("MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)),((2 2, 3 2, 3 3, 2 3, 2 2)))"),
        );

        // Input with nulls
        let nulls_input = vec![
            vec![Some("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))"), None],
            vec![Some("POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))"), None],
        ];
        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(nulls_input).unwrap(),
            Some("MULTIPOLYGON(((0 0, 2 0, 2 1, 3 1, 3 3, 1 3, 1 2, 0 2, 0 0)))"),
        );

        // Fully contained polygon should merge into one
        let contained = vec![
            vec![Some("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))")],
            vec![Some("POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))")],
        ];
        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(contained).unwrap(),
            Some("MULTIPOLYGON(((0 0, 3 0, 3 3, 0 3, 0 0)))"),
        );
    }

    #[rstest]
    fn polygon_multipolygon_cases(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let mut udaf = st_union_aggr_udf();
        udaf.add_kernel(st_union_aggr_impl());

        let tester = AggregateUdfTester::new(udaf.into(), vec![sedona_type.clone()]);

        // Polygon unioning with MultiPolygon
        let poly_and_multi = vec![
            vec![Some("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))")],
            vec![Some(
                "MULTIPOLYGON(((1 1, 2 1, 2 2, 1 2, 1 1)), ((4 4, 5 4, 5 5, 4 5, 4 4)))",
            )],
        ];
        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(poly_and_multi).unwrap(),
            Some("MULTIPOLYGON(((0 0, 3 0, 3 3, 0 3, 0 0)),((4 4, 5 4, 5 5, 4 5, 4 4)))"),
        );

        // Polygon with non-overlapping MultiPolygon (should return union of all)
        let poly_and_nonoverlap_multi = vec![
            vec![Some("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")],
            vec![Some(
                "MULTIPOLYGON(((2 2, 3 2, 3 3, 2 3, 2 2)), ((4 4, 5 4, 5 5, 4 5, 4 4)))",
            )],
        ];
        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(poly_and_nonoverlap_multi).unwrap(),
            Some("MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)),((2 2, 3 2, 3 3, 2 3, 2 2)),((4 4, 5 4, 5 5, 4 5, 4 4)))"),
        );

        // MultiPolygon with MultiPolygon (should return union of all)
        let multi_and_multi = vec![
            vec![Some(
                "MULTIPOLYGON(((0 0, 3 0, 3 3, 0 3, 0 0)), ((10 10, 12 10, 12 12, 10 12, 10 10)))",
            )],
            vec![Some(
                "MULTIPOLYGON(((1 1, 2 1, 2 2, 1 2, 1 1)), ((11 11, 13 11, 13 13, 11 13, 11 11)))",
            )],
        ];
        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(multi_and_multi).unwrap(),
            Some("MULTIPOLYGON(((0 0, 3 0, 3 3, 0 3, 0 0)),((10 10, 12 10, 12 11, 13 11, 13 13, 11 13, 11 12, 10 12, 10 10)))"),
        );
    }

    #[rstest]
    fn multipolygon_multipolygon_cases(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let mut udaf = st_union_aggr_udf();
        udaf.add_kernel(st_union_aggr_impl());

        let tester = AggregateUdfTester::new(udaf.into(), vec![sedona_type.clone()]);

        // Test case 1: Two MultiPolygons with intersecting polygons
        let multi_multi_case1 = vec![
            vec![Some(
                "MULTIPOLYGON(((0 0, 3 0, 3 3, 0 3, 0 0)), ((5 5, 8 5, 8 8, 5 8, 5 5)))",
            )],
            vec![Some(
                "MULTIPOLYGON(((2 2, 5 2, 5 5, 2 5, 2 2)), ((7 7, 10 7, 10 10, 7 10, 7 7)))",
            )],
        ];
        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(multi_multi_case1).unwrap(),
            Some("MULTIPOLYGON(((0 0, 3 0, 3 2, 5 2, 5 5, 2 5, 2 3, 0 3, 0 0)),((5 5, 8 5, 8 7, 10 7, 10 10, 7 10, 7 8, 5 8, 5 5)))"),
        );

        // Test case 2: MultiPolygons with non-intersecting polygons
        let multi_multi_case2 = vec![
            vec![Some(
                "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((5 5, 6 5, 6 6, 5 6, 5 5)))",
            )],
            vec![Some(
                "MULTIPOLYGON(((2 2, 3 2, 3 3, 2 3, 2 2)), ((7 7, 8 7, 8 8, 7 8, 7 7)))",
            )],
        ];
        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(multi_multi_case2).unwrap(),
            Some("MULTIPOLYGON(((0 0,1 0,1 1,0 1,0 0)),((2 2,3 2,3 3,2 3,2 2)),((5 5,6 5,6 6,5 6,5 5)),((7 7,8 7,8 8,7 8,7 7)))"),
        );

        // Test case 3: Three MultiPolygons with some overlap
        let multi_multi_case3 = vec![
            vec![Some("MULTIPOLYGON(((0 0, 4 0, 4 4, 0 4, 0 0)), ((10 10, 14 10, 14 14, 10 14, 10 10)))")],
            vec![Some("MULTIPOLYGON(((3 3, 7 3, 7 7, 3 7, 3 3)), ((13 13, 17 13, 17 17, 13 17, 13 13)))")],
            vec![Some("MULTIPOLYGON(((6 6, 10 6, 10 10, 6 10, 6 6)), ((16 16, 20 16, 20 20, 16 20, 16 16)))")],
        ];
        assert_scalar_equal_wkb_geometry_topologically(
            &tester.aggregate_wkt(multi_multi_case3).unwrap(),
            Some("MULTIPOLYGON(((0 0, 4 0, 4 3, 7 3, 7 6, 10 6, 10 10, 6 10, 6 7, 3 7, 3 4, 0 4, 0 0)),((10 10, 14 10, 14 13, 17 13, 17 16, 20 16, 20 20, 16 20, 16 17, 13 17, 13 14, 10 14, 10 10)))"),
        );
    }
}
