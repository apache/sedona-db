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

use datafusion_common::{Result, ScalarValue};
use parquet::{
    basic::LogicalType,
    geospatial::accumulator::{
        init_geo_stats_accumulator_factory, GeoStatsAccumulator, GeoStatsAccumulatorFactory,
        ParquetGeoStatsAccumulator, VoidGeoStatsAccumulator,
    },
    schema::types::ColumnDescPtr,
};
use sedona_common::sedona_internal_err;
use sedona_expr::spatial_filter::LiteralBounder;
use sedona_schema::datatypes::SedonaType;

pub struct SedonaGeoStatsAccumulatorFactory;

impl SedonaGeoStatsAccumulatorFactory {
    pub fn try_init() -> Result<()> {
        init_geo_stats_accumulator_factory(Arc::new(Self))?;
        Ok(())
    }
}

impl GeoStatsAccumulatorFactory for SedonaGeoStatsAccumulatorFactory {
    fn new_accumulator(&self, descr: &ColumnDescPtr) -> Box<dyn GeoStatsAccumulator> {
        if let Some(LogicalType::Geometry { .. }) = descr.logical_type_ref() {
            return Box::new(ParquetGeoStatsAccumulator::default());
        }

        #[cfg(feature = "s2geography")]
        if let Some(LogicalType::Geography {
            crs: _,
            algorithm: None,
        }) = descr.logical_type_ref()
        {
            return Box::new(GeographyGeoStatsAccumulator::default());
        }

        #[cfg(feature = "s2geography")]
        if let Some(LogicalType::Geography {
            crs: _,
            algorithm: Some(parquet::basic::EdgeInterpolationAlgorithm::SPHERICAL),
        }) = descr.logical_type_ref()
        {
            return Box::new(GeographyGeoStatsAccumulator::default());
        }

        Box::new(VoidGeoStatsAccumulator::default())
    }
}

#[derive(Debug, Default)]
pub struct GeographyLiteralBounder;

#[cfg(not(feature = "s2geography"))]
impl LiteralBounder for GeographyLiteralBounder {
    fn can_bound(&self, _sedona_type: &SedonaType) -> bool {
        false
    }

    fn bound_scalar(
        &self,
        _value: &ScalarValue,
        _sedona_type: &SedonaType,
        _distance: Option<f64>,
    ) -> Result<sedona_geometry::bounding_box::BoundingBox> {
        sedona_internal_err!("dummy bound_scalar() should not be called")
    }
}

#[cfg(feature = "s2geography")]
impl LiteralBounder for GeographyLiteralBounder {
    fn can_bound(&self, sedona_type: &SedonaType) -> bool {
        matches!(
            sedona_type,
            SedonaType::Wkb(sedona_schema::datatypes::Edges::Spherical, _)
                | SedonaType::WkbView(sedona_schema::datatypes::Edges::Spherical, _)
        )
    }

    fn bound_scalar(
        &self,
        value: &ScalarValue,
        sedona_type: &SedonaType,
        distance: Option<f64>,
    ) -> Result<sedona_geometry::bounding_box::BoundingBox> {
        match &sedona_type {
            SedonaType::Wkb(sedona_schema::datatypes::Edges::Spherical, _)
            | SedonaType::WkbView(sedona_schema::datatypes::Edges::Spherical, _) => match value {
                ScalarValue::Binary(maybe_vec) | ScalarValue::BinaryView(maybe_vec) => {
                    if let Some(vec) = maybe_vec {
                        let mut bounder = sedona_s2geography::rect_bounder::RectBounder::new();
                        let mut factory = sedona_s2geography::geography::GeographyFactory::new();
                        let geog = factory.from_wkb(vec).map_err(|e| {
                            datafusion_common::exec_datafusion_err!(
                                "Error parsing geography literal: {e}"
                            )
                        })?;
                        bounder.bound(&geog).map_err(|e| {
                            datafusion_common::exec_datafusion_err!(
                                "Error bounding geography literal: {e}"
                            )
                        })?;

                        if let Some(distance) = distance {
                            bounder.expand_by_distance(distance);
                        }

                        if let Some((xmin, ymin, xmax, ymax)) = bounder.finish().map_err(|e| {
                            datafusion_common::exec_datafusion_err!(
                                "Error finishing geography bounds: {e}"
                            )
                        })? {
                            return Ok(sedona_geometry::bounding_box::BoundingBox::xy(
                                (xmin, xmax),
                                (ymin, ymax),
                            ));
                        }
                    }

                    // Null scalar or empty geography returns an empty bounding box
                    return Ok(sedona_geometry::bounding_box::BoundingBox::empty());
                }
                _ => {}
            },
            _ => {}
        }

        sedona_internal_err!("Unexpected scalar type in filter expression ({sedona_type:?})")
    }
}

#[cfg(feature = "s2geography")]
#[derive(Debug)]
struct GeographyGeoStatsAccumulator {
    invalid: bool,
    geog_bounder: sedona_s2geography::rect_bounder::RectBounder,
    bounder: parquet_geospatial::bounding::GeometryBounder,
    geography_factory: sedona_s2geography::geography::GeographyFactory,
}

#[cfg(feature = "s2geography")]
impl Default for GeographyGeoStatsAccumulator {
    fn default() -> Self {
        Self {
            invalid: false,
            geog_bounder: Default::default(),
            bounder: parquet_geospatial::bounding::GeometryBounder::empty(),
            geography_factory: Default::default(),
        }
    }
}

#[cfg(feature = "s2geography")]
impl GeographyGeoStatsAccumulator {
    fn clear(&mut self) {
        self.invalid = false;
        self.bounder = parquet_geospatial::bounding::GeometryBounder::empty();
        self.geog_bounder.clear();
    }
}

#[cfg(feature = "s2geography")]
impl GeoStatsAccumulator for GeographyGeoStatsAccumulator {
    fn is_valid(&self) -> bool {
        !self.invalid
    }

    fn update_wkb(&mut self, wkb: &[u8]) {
        if self.invalid {
            return;
        }

        // Update the geometry bounder. We use this for geometry types and ZM bounds.
        if self.bounder.update_wkb(wkb).is_err() {
            self.invalid = true;
            return;
        }

        let Ok(geog) = self.geography_factory.from_wkb(wkb) else {
            self.invalid = true;
            return;
        };

        if self.geog_bounder.bound(&geog).is_err() {
            self.invalid = true;
        }
    }

    fn finish(&mut self) -> Option<Box<parquet::geospatial::statistics::GeospatialStatistics>> {
        use parquet_geospatial::interval::IntervalTrait;

        use parquet::geospatial::bounding_box::BoundingBox;

        if self.invalid {
            self.clear();
            return None;
        }

        let bounder_geometry_types = self.bounder.geometry_types();
        let geometry_types = if bounder_geometry_types.is_empty() {
            None
        } else {
            Some(bounder_geometry_types)
        };

        // If we have completely empty bounds, we can still communicate the visited geometry types
        let Ok(Some(geog_box)) = self.geog_bounder.finish() else {
            self.clear();
            return Some(Box::new(
                parquet::geospatial::statistics::GeospatialStatistics::new(None, geometry_types),
            ));
        };

        // s2geography's rect bounder outputs xmin, ymin, xmax, ymax; BoundingBox::new()
        // accepts xmin, xmax, ymin, ymax.
        let mut bbox = BoundingBox::new(geog_box.0, geog_box.2, geog_box.1, geog_box.3);

        // Use the Z, M, and geometry type bounds from the Parquet implementation
        if !self.bounder.z().is_empty() {
            bbox = bbox.with_zrange(self.bounder.z().lo(), self.bounder.z().hi());
        }

        if !self.bounder.m().is_empty() {
            bbox = bbox.with_mrange(self.bounder.m().lo(), self.bounder.m().hi());
        }

        // Reset
        self.clear();

        Some(Box::new(
            parquet::geospatial::statistics::GeospatialStatistics::new(Some(bbox), geometry_types),
        ))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use parquet::geospatial::bounding_box::BoundingBox;
    use parquet_geospatial::testing::{wkb_point_xy, wkb_point_xyzm};
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY};
    use sedona_testing::create::create_scalar;

    #[test]
    fn test_factory() {}

    #[cfg(feature = "s2geography")]
    #[test]
    fn test_literal_bounder() {
        use sedona_geometry::interval::IntervalTrait;

        for sedona_type in [WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY] {
            let bounder = GeographyLiteralBounder;
            assert!(bounder.can_bound(&sedona_type));

            let scalar = create_scalar(Some("MULTIPOINT ((-179 42), (179 43))"), &sedona_type);

            // Check approximate raw bounds
            let raw_bounds = bounder.bound_scalar(&scalar, &sedona_type, None).unwrap();
            assert!(raw_bounds.x().is_wraparound());
            assert!(raw_bounds.x().lo() > 178.99999 && raw_bounds.x().lo() <= 179.0);
            assert!(raw_bounds.x().hi() >= -179.0 && raw_bounds.x().hi() < -178.99999);
            assert!(raw_bounds.y().lo() > 41.0 && raw_bounds.y().lo() <= 42.0);
            assert!(raw_bounds.y().hi() >= 43.0 && raw_bounds.y().hi() < 44.0);

            // Check that expanded bounds are bigger (100 km ~ 0.9 degrees)
            let expanded_bounds = bounder
                .bound_scalar(&scalar, &sedona_type, Some(100000.0))
                .unwrap();
            assert!(expanded_bounds.x().is_wraparound());
            assert!(raw_bounds.x().lo() - expanded_bounds.x().lo() > 0.5);
            assert!(expanded_bounds.x().hi() - raw_bounds.x().hi() > 0.5);
            assert!(raw_bounds.y().lo() - expanded_bounds.y().lo() > 0.5);
            assert!(expanded_bounds.y().hi() - raw_bounds.y().hi() > 0.5);

            let null_scalar = create_scalar(None, &sedona_type);
            let null_bounds = bounder
                .bound_scalar(&null_scalar, &sedona_type, None)
                .unwrap();
            assert!(null_bounds.is_empty());
        }
    }

    #[cfg(not(feature = "s2geography"))]
    #[test]
    fn test_literal_bounder() {
        for sedona_type in [WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY] {
            let scalar = create_scalar(Some("MULTIPOINT ((-179 42), (179 43))"), &sedona_type);
            let bounder = GeographyLiteralBounder;
            assert!(!bounder.can_bound(&sedona_type));
            bounder
                .bound_scalar(&scalar, &sedona_type, None)
                .unwrap_err();
        }
    }

    #[cfg(feature = "s2geography")]
    fn assert_bbox_contains(actual: &BoundingBox, expected: &BoundingBox, epsilon: f64) {
        // actual.xmin should be <= expected.xmin (can be slightly smaller)
        assert!(
            expected.get_xmin() - actual.get_xmin() >= 0.0
                && expected.get_xmin() - actual.get_xmin() < epsilon,
            "x_min mismatch: actual {} should be <= expected {} (within {})",
            actual.get_xmin(),
            expected.get_xmin(),
            epsilon
        );
        // actual.xmax should be >= expected.xmax (can be slightly larger)
        assert!(
            actual.get_xmax() - expected.get_xmax() >= 0.0
                && actual.get_xmax() - expected.get_xmax() < epsilon,
            "x_max mismatch: actual {} should be >= expected {} (within {})",
            actual.get_xmax(),
            expected.get_xmax(),
            epsilon
        );
        // actual.ymin should be <= expected.ymin (can be slightly smaller)
        assert!(
            expected.get_ymin() - actual.get_ymin() >= 0.0
                && expected.get_ymin() - actual.get_ymin() < epsilon,
            "y_min mismatch: actual {} should be <= expected {} (within {})",
            actual.get_ymin(),
            expected.get_ymin(),
            epsilon
        );
        // actual.ymax should be >= expected.ymax (can be slightly larger)
        assert!(
            actual.get_ymax() - expected.get_ymax() >= 0.0
                && actual.get_ymax() - expected.get_ymax() < epsilon,
            "y_max mismatch: actual {} should be >= expected {} (within {})",
            actual.get_ymax(),
            expected.get_ymax(),
            epsilon
        );

        assert_eq!(actual.get_zmin(), expected.get_zmin(), "z_min mismatch");
        assert_eq!(actual.get_zmax(), expected.get_zmax(), "z_max mismatch");
        assert_eq!(actual.get_mmin(), expected.get_mmin(), "m_min mismatch");
        assert_eq!(actual.get_mmax(), expected.get_mmax(), "m_max mismatch");
    }

    #[cfg(feature = "s2geography")]
    #[test]
    fn test_geography_accumulator() {
        // The geography bounder produces slightly expanded bounds compared to the
        // geometry bounder due to spherical interpolation along geodesics.
        const EPSILON: f64 = 1e-10;
        let mut accumulator = GeographyGeoStatsAccumulator::default();

        // A fresh instance should be able to bound input
        assert!(accumulator.is_valid());
        accumulator.update_wkb(&wkb_point_xy(1.0, 2.0));
        accumulator.update_wkb(&wkb_point_xy(11.0, 12.0));
        let stats = accumulator.finish().unwrap();
        assert_eq!(stats.geospatial_types().unwrap(), &vec![1]);
        assert_bbox_contains(
            stats.bounding_box().unwrap(),
            &BoundingBox::new(1.0, 11.0, 2.0, 12.0),
            EPSILON,
        );

        // finish() should have reset the bounder such that the first values
        // aren't when computing the next bound of statistics.
        assert!(accumulator.is_valid());
        accumulator.update_wkb(&wkb_point_xy(21.0, 22.0));
        accumulator.update_wkb(&wkb_point_xy(31.0, 32.0));
        let stats = accumulator.finish().unwrap();
        assert_eq!(stats.geospatial_types().unwrap(), &vec![1]);
        assert_bbox_contains(
            stats.bounding_box().unwrap(),
            &BoundingBox::new(21.0, 31.0, 22.0, 32.0),
            EPSILON,
        );

        // When an accumulator encounters invalid input, it reports is_valid() false
        // and does not compute subsequent statistics
        assert!(accumulator.is_valid());
        accumulator.update_wkb(&wkb_point_xy(41.0, 42.0));
        accumulator.update_wkb("these bytes are not WKB".as_bytes());
        assert!(!accumulator.is_valid());
        assert!(accumulator.finish().is_none());

        // Subsequent rounds of accumulation should work as expected
        assert!(accumulator.is_valid());
        accumulator.update_wkb(&wkb_point_xy(41.0, 42.0));
        accumulator.update_wkb(&wkb_point_xy(51.0, 52.0));
        let stats = accumulator.finish().unwrap();
        assert_eq!(stats.geospatial_types().unwrap(), &vec![1]);
        assert_bbox_contains(
            stats.bounding_box().unwrap(),
            &BoundingBox::new(41.0, 51.0, 42.0, 52.0),
            EPSILON,
        );

        // Antimeridian-crossing input should result in a wraparound box
        assert!(accumulator.is_valid());
        accumulator.update_wkb(&wkb_point_xy(-179.0, 42.0));
        accumulator.update_wkb(&wkb_point_xy(179.0, 52.0));
        let stats = accumulator.finish().unwrap();
        assert!(
            stats.bounding_box().unwrap().get_xmin() > stats.bounding_box().unwrap().get_xmax()
        );

        // When there was no input at all (occurs in the all null case), both geometry
        // types and bounding box will be None. This is because Parquet Thrift statistics
        // have no mechanism to communicate "empty". (The all null situation may be determined
        // from the null count in this case).
        assert!(accumulator.is_valid());
        let stats = accumulator.finish().unwrap();
        assert!(stats.geospatial_types().is_none());
        assert!(stats.bounding_box().is_none());

        // When there was 100% "empty" input (i.e., non-null geometries without
        // coordinates), there should be statistics with geometry types but no
        // bounding box.
        assert!(accumulator.is_valid());
        accumulator.update_wkb(&wkb_point_xy(f64::NAN, f64::NAN));
        let stats = accumulator.finish().unwrap();
        assert_eq!(stats.geospatial_types().unwrap(), &vec![1]);
        assert!(stats.bounding_box().is_none());

        // If Z and/or M are present, they should be reported in the bounding box
        assert!(accumulator.is_valid());
        accumulator.update_wkb(&wkb_point_xyzm(1.0, 2.0, 3.0, 4.0));
        accumulator.update_wkb(&wkb_point_xyzm(5.0, 6.0, 7.0, 8.0));
        let stats = accumulator.finish().unwrap();
        assert_eq!(stats.geospatial_types().unwrap(), &vec![3001]);
        assert_bbox_contains(
            stats.bounding_box().unwrap(),
            &BoundingBox::new(1.0, 5.0, 2.0, 6.0)
                .with_zrange(3.0, 7.0)
                .with_mrange(4.0, 8.0),
            EPSILON,
        );
    }
}
