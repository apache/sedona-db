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
use std::{sync::Arc, vec};

use crate::executor::WkbExecutor;
use arrow_array::ArrayRef;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{
    cast::{as_binary_array, as_int64_array, as_uint32_array},
    error::{DataFusionError, Result},
    exec_err, ScalarValue,
};
use datafusion_expr::{Accumulator, ColumnarValue, Volatility};
use geo_traits::Dimensions;
use sedona_common::sedona_internal_err;
use sedona_expr::{
    aggregate_udf::{SedonaAccumulator, SedonaAggregateUDF},
    item_crs::ItemCrsSedonaAccumulator,
};
use sedona_geometry::{
    types::{GeometryTypeAndDimensions, GeometryTypeAndDimensionsSet, GeometryTypeId},
    wkb_factory::{
        write_wkb_geometrycollection_header, write_wkb_multilinestring_header,
        write_wkb_multipoint_header, write_wkb_multipolygon_header,
    },
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

/// ST_Collect_Agg() aggregate UDF implementation
///
/// An implementation of envelope (bounding shape) calculation.
pub fn st_collect_agg_udf() -> SedonaAggregateUDF {
    SedonaAggregateUDF::new(
        "st_collect_agg",
        ItemCrsSedonaAccumulator::wrap_impl(vec![
            Arc::new(STCollectAggr {
                is_geography: false,
            }),
            Arc::new(STCollectAggr { is_geography: true }),
        ]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STCollectAggr {
    is_geography: bool,
}

impl SedonaAccumulator for STCollectAggr {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = match self.is_geography {
            true => ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY),
            false => ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
        };
        matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(CollectionAccumulator::try_new(
            args[0].clone(),
            output_type.clone(),
        )?))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Arc::new(Field::new("types_and_dims", DataType::UInt32, false)),
            Arc::new(Field::new("count", DataType::Int64, false)),
            Arc::new(WKB_GEOMETRY.to_storage_field("item", true)?),
        ])
    }
}

#[derive(Debug)]
struct CollectionAccumulator {
    input_type: SedonaType,
    types_and_dims: GeometryTypeAndDimensionsSet,
    count: i64,
    item: Option<Vec<u8>>,
}

const WKB_HEADER_SIZE: usize = 1 + 4 + 4;

impl CollectionAccumulator {
    pub fn try_new(input_type: SedonaType, _output_type: SedonaType) -> Result<Self> {
        // Write a dummy header with the correct number of bytes. We'll rewrite this later
        // when we know what type/dimension of geometrycollection we have based on the
        // items encountered.
        let mut item = Vec::new();
        write_wkb_geometrycollection_header(&mut item, Dimensions::Xy, 0)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(Self {
            input_type,
            types_and_dims: GeometryTypeAndDimensionsSet::new(),
            count: 0,
            item: Some(item),
        })
    }

    // Create a WKB result based on the current state of the accumulator.
    fn make_wkb_result(&mut self) -> Result<Option<Vec<u8>>> {
        if self.count == 0 {
            return Ok(None);
        }

        // Generate the correct header: collections of points become multipoint, ensure
        // dimensions are preserved if possible.
        let mut new_header = Vec::new();
        let count_usize = self.count.try_into().unwrap();

        let unique_dimensions = self.types_and_dims.dimensions();
        if unique_dimensions.len() != 1 {
            return exec_err!("Can't ST_Collect_Agg() mixed dimension geometries");
        }

        let dimensions = unique_dimensions[0];
        let unique_geometry_types = self.types_and_dims.geometry_types();
        if unique_geometry_types.len() == 1 {
            match unique_geometry_types[0] {
                GeometryTypeId::Point => {
                    write_wkb_multipoint_header(&mut new_header, dimensions, count_usize)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
                GeometryTypeId::LineString => {
                    write_wkb_multilinestring_header(&mut new_header, dimensions, count_usize)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
                GeometryTypeId::Polygon => {
                    write_wkb_multipolygon_header(&mut new_header, dimensions, count_usize)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
                _ => {
                    write_wkb_geometrycollection_header(&mut new_header, dimensions, count_usize)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }
            }
        } else {
            write_wkb_geometrycollection_header(&mut new_header, dimensions, count_usize)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
        }

        // Update the header bytes of the output and return it
        if let Some(mut out) = self.item.take() {
            out[0..WKB_HEADER_SIZE].copy_from_slice(&new_header);
            Ok(Some(out))
        } else {
            sedona_internal_err!("Unexpected internal state in ST_Collect_Agg()")
        }
    }
}

impl Accumulator for CollectionAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let item_ref = if let Some(item_ref) = self.item.as_mut() {
            item_ref
        } else {
            return sedona_internal_err!("Unexpected internal state in ST_Collect_Agg()");
        };

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &args);
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                let types_and_dims = GeometryTypeAndDimensions::try_from_geom(&item)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                // insert() (not insert_or_ignore()) so geometries with
                // unknown dimensions fail loudly instead of being silently
                // dropped from the dimension check in make_wkb_result().
                self.types_and_dims
                    .insert(&types_and_dims)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.count += 1;
                item_ref.extend_from_slice(item.buf());
            }
            Ok(())
        })?;
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let wkb = self.make_wkb_result()?;
        Ok(ScalarValue::Binary(wkb))
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            ScalarValue::UInt32(Some(self.types_and_dims.bits())),
            ScalarValue::Int64(Some(self.count)),
            ScalarValue::Binary(self.item.take()),
        ])
    }

    fn size(&self) -> usize {
        let item_capacity = self.item.as_ref().map(|e| e.capacity()).unwrap_or(0);
        size_of::<CollectionAccumulator>() + item_capacity
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 3 {
            return sedona_internal_err!(
                "Unexpected number of state fields for st_collect() (expected 3, got {})",
                states.len()
            );
        }

        let item_ref = if let Some(item_ref) = self.item.as_mut() {
            item_ref
        } else {
            return sedona_internal_err!("Unexpected internal state in ST_Collect_Agg()");
        };

        let mut bits_iter = as_uint32_array(&states[0])?.into_iter();
        let mut count_iter = as_int64_array(&states[1])?.into_iter();
        let mut item_iter = as_binary_array(&states[2])?.into_iter();

        for _ in 0..bits_iter.len() {
            match (bits_iter.next(), count_iter.next(), item_iter.next()) {
                (Some(Some(bits)), Some(Some(count)), Some(Some(item))) => {
                    self.types_and_dims
                        .merge(&GeometryTypeAndDimensionsSet::from_bits(bits));
                    self.count += count;
                    item_ref.extend_from_slice(&item[WKB_HEADER_SIZE..item.len()]);
                }
                _ => {
                    return sedona_internal_err!(
                        "unexpected nulls in st_collect() serialized state"
                    )
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use datafusion_expr::AggregateUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{
        compare::{assert_scalar_equal, assert_scalar_equal_wkb_geometry},
        create::{create_array, create_array_item_crs, create_scalar, create_scalar_item_crs},
        testers::AggregateUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = st_collect_agg_udf().into();
        assert_eq!(udf.name(), "st_collect_agg");
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Finite point input with nulls
        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("MULTIPOINT (0 1, 2 3, 4 5, 6 7)"),
        );

        // Finite linestring input with nulls
        let batches = vec![
            vec![Some("LINESTRING (0 1, 2 3)"), None],
            vec![Some("LINESTRING (4 5, 6 7)"), None],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("MULTILINESTRING ((0 1, 2 3), (4 5, 6 7))"),
        );

        // Finite polygon input with nulls
        let batches = vec![
            vec![Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"), None],
            vec![Some("POLYGON ((10 10, 11 10, 10 11, 10 10))"), None],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((10 10, 11 10, 10 11, 10 10)))"),
        );

        // Mixed input
        let batches = vec![
            vec![Some("POINT (0 1)"), None],
            vec![Some("LINESTRING (4 5, 6 7)"), None],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("GEOMETRYCOLLECTION (POINT (0 1), LINESTRING (4 5, 6 7))"),
        );

        // Empty input
        assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(vec![]).unwrap(), None);

        // Error for mixed dimensions
        let batches = vec![
            vec![Some("POINT (0 1)"), None],
            vec![Some("POINT Z (0 1 2)"), None],
        ];
        let err = tester.aggregate_wkt(batches).unwrap_err();
        assert_eq!(
            err.message(),
            "Can't ST_Collect_Agg() mixed dimension geometries"
        );
    }

    #[rstest]
    fn udf_geog(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOGRAPHY);
    }

    #[rstest]
    fn udf_invoke_item_crs(
        #[values(WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batch0 = create_array(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            &sedona_type,
        );
        let batch1 = create_array(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            &sedona_type,
        );

        let batches = vec![batch0, batch1];
        let expected = create_scalar(Some("MULTIPOINT (0 1, 2 3, 4 5, 6 7)"), &sedona_type);

        assert_scalar_equal(&tester.aggregate(&batches).unwrap(), &expected);
    }

    #[rstest]
    fn udf_invoke_item_crs_idential_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batch0 = create_array_item_crs(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            [Some("EPSG:4326"), None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );
        let batch1 = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [Some("EPSG:4326"), None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );

        let expected = create_scalar_item_crs(
            Some("MULTIPOINT (0 1, 2 3, 4 5, 6 7)"),
            Some("EPSG:4326"),
            &WKB_GEOMETRY,
        );

        assert_scalar_equal(
            &tester
                .aggregate(&vec![batch0.clone(), batch1.clone()])
                .unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn udf_invoke_item_crs_multiple_compatible_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batch0 = create_array_item_crs(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            [Some("OGC:CRS84"), None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );
        let batch1 = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [Some("EPSG:4326"), None, Some("OGC:CRS84")],
            &WKB_GEOMETRY,
        );

        let expected = create_scalar_item_crs(
            Some("MULTIPOINT (0 1, 2 3, 4 5, 6 7)"),
            Some("OGC:CRS84"),
            &WKB_GEOMETRY,
        );

        assert_scalar_equal(
            &tester
                .aggregate(&vec![batch0.clone(), batch1.clone()])
                .unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn udf_invoke_item_crs_incompatible_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let tester =
            AggregateUdfTester::new(st_collect_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batch0 = create_array_item_crs(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            [Some("OGC:CRS84"), None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );

        // We should error if we see incompatible CRSes between batches
        let batch1_other_crs = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [Some("EPSG:3857"), None, Some("EPSG:3857")],
            &WKB_GEOMETRY,
        );
        let err = tester
            .aggregate(&vec![batch0.clone(), batch1_other_crs.clone()])
            .unwrap_err();
        assert_eq!(
            err.message(),
            "CRS values not equal: ogc:crs84 vs epsg:3857"
        );

        // We should error if we see incompatible CRSes between batches (None
        // should be considered an incompatible CRS)
        let batch1_other_crs = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [None, None, None],
            &WKB_GEOMETRY,
        );
        let err = tester
            .aggregate(&vec![batch0.clone(), batch1_other_crs.clone()])
            .unwrap_err();
        assert_eq!(err.message(), "CRS values not equal: ogc:crs84 vs None");

        // Or if we see incompatible CRSes in a single batch
        let batch0_incompatible_crses = create_array_item_crs(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            [Some("OGC:CRS84"), None, Some("EPSG:3857")],
            &WKB_GEOMETRY,
        );

        let err = tester
            .aggregate(&vec![batch0_incompatible_crses.clone()])
            .unwrap_err();
        assert_eq!(
            err.message(),
            "CRS values not equal: ogc:crs84 vs epsg:3857"
        );
    }

    /// A serialized state round-trips: emitting state and merging it back into
    /// a fresh accumulator reconstructs the same type/dimension set and count.
    #[test]
    fn state_round_trips() {
        let mut acc = CollectionAccumulator::try_new(WKB_GEOMETRY, WKB_GEOMETRY).unwrap();
        for td in [
            GeometryTypeAndDimensions::new(GeometryTypeId::Point, Dimensions::Xy),
            GeometryTypeAndDimensions::new(GeometryTypeId::LineString, Dimensions::Xyz),
        ] {
            acc.types_and_dims.insert(&td).unwrap();
        }
        acc.count = 3;
        let expected_set = acc.types_and_dims.clone();

        let state = acc.state().unwrap();
        let arrays: Vec<ArrayRef> = state.iter().map(|s| s.to_array().unwrap()).collect();

        let mut merged = CollectionAccumulator::try_new(WKB_GEOMETRY, WKB_GEOMETRY).unwrap();
        merged.merge_batch(&arrays).unwrap();

        assert_eq!(merged.types_and_dims, expected_set);
        assert_eq!(merged.count, 3);
    }

    /// Regression test for memory accounting: the accumulator must report an
    /// exact size with no unaccounted per-group heap. The HashSets this
    /// replaces allocated ~100 heap bytes per non-empty group that size()
    /// reported as zero, hiding gigabytes of aggregate state from the memory
    /// pool on large aggregations (no spill, unaccounted anon growth).
    #[test]
    fn accumulator_size_is_exact() {
        let mut acc = CollectionAccumulator::try_new(WKB_GEOMETRY, WKB_GEOMETRY).unwrap();
        let empty_size = acc.size();

        acc.types_and_dims
            .insert(&GeometryTypeAndDimensions::new(
                GeometryTypeId::Point,
                Dimensions::Xy,
            ))
            .unwrap();

        // Inserting into the inline bitset allocates nothing.
        assert_eq!(acc.size(), empty_size);
        assert_eq!(
            acc.size(),
            size_of::<CollectionAccumulator>() + acc.item.as_ref().unwrap().capacity()
        );
    }
}
