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

use datafusion_common::Result;
use parquet::{
    basic::LogicalType,
    geospatial::accumulator::{
        init_geo_stats_accumulator_factory, GeoStatsAccumulator, GeoStatsAccumulatorFactory,
        ParquetGeoStatsAccumulator, VoidGeoStatsAccumulator,
    },
    schema::types::ColumnDescPtr,
};
use sedona_common::SedonaRuntime;
use sedona_geometry::{
    bounds::{WkbBounder2D, WkbBounder2DFactory},
    types::Edges,
};

/// Factory for Parquet [GeoStatsAccumulator]s that can handle Geography
pub struct SedonaGeoStatsAccumulatorFactory {
    bounder_factory: WkbBounder2DFactory,
}

impl SedonaGeoStatsAccumulatorFactory {
    /// Initialize the global geo-statistics accumulator factory from the given runtime.
    ///
    /// # First-Session-Wins Semantics
    ///
    /// This function snapshots the runtime's `bounder_factory` into arrow-rs's process-wide
    /// `OnceLock`. Subsequent calls with different runtimes will fail with a `ParquetError`
    /// (typically swallowed by the caller). This means:
    ///
    /// - The first `SedonaContext` created in a process determines the bounders used for
    ///   writing Parquet geography statistics for the lifetime of that process.
    /// - Bounders registered via `SedonaRuntime::with_bounder` *after* context creation,
    ///   or in a second context with a different runtime, will **not** affect Parquet writes.
    /// - The read path (`GeoParquetFormat`) re-reads live options per plan, so pruning uses
    ///   fresh bounders while writes use the snapshotted ones.
    ///
    /// This asymmetry is forced by arrow-rs's global API and cannot be worked around without
    /// upstream changes.
    pub fn try_init(runtime: &SedonaRuntime) -> Result<()> {
        let bounder_factory = runtime.bounder_factory().clone();
        init_geo_stats_accumulator_factory(Arc::new(Self { bounder_factory }))?;
        Ok(())
    }
}

impl GeoStatsAccumulatorFactory for SedonaGeoStatsAccumulatorFactory {
    fn new_accumulator(&self, descr: &ColumnDescPtr) -> Box<dyn GeoStatsAccumulator> {
        if let Some(LogicalType::Geometry { .. }) = descr.logical_type_ref() {
            return Box::new(ParquetGeoStatsAccumulator::default());
        }

        // Handle Geography with either no algorithm specified (defaults to spherical)
        // or explicit SPHERICAL algorithm
        if let Some(LogicalType::Geography {
            crs: _,
            algorithm: None | Some(parquet::basic::EdgeInterpolationAlgorithm::SPHERICAL),
        }) = descr.logical_type_ref()
        {
            if let Some(bounder) = self.bounder_factory.bounder_for_edge_type(Edges::Spherical) {
                return Box::new(GeographyGeoStatsAccumulator::new(bounder));
            }
        }

        Box::new(VoidGeoStatsAccumulator::default())
    }
}

#[derive(Debug)]
struct GeographyGeoStatsAccumulator {
    invalid: bool,
    geog_bounder: Box<dyn WkbBounder2D>,
    bounder: parquet_geospatial::bounding::GeometryBounder,
}

impl GeographyGeoStatsAccumulator {
    fn new(geog_bounder: Box<dyn WkbBounder2D>) -> Self {
        Self {
            invalid: false,
            geog_bounder,
            bounder: parquet_geospatial::bounding::GeometryBounder::empty(),
        }
    }

    fn clear(&mut self) {
        self.invalid = false;
        self.bounder = parquet_geospatial::bounding::GeometryBounder::empty();
        self.geog_bounder.clear();
    }
}

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

        if self.geog_bounder.update_wkb_bytes(wkb).is_err() {
            self.invalid = true;
        }
    }

    fn finish(&mut self) -> Option<Box<parquet::geospatial::statistics::GeospatialStatistics>> {
        use parquet::geospatial::bounding_box::BoundingBox;
        use parquet_geospatial::interval::IntervalTrait as ParquetIntervalTrait;
        use sedona_geometry::interval::IntervalTrait;

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
        let (x, y) = self.geog_bounder.finish();

        if x.is_empty() || y.is_empty() {
            self.clear();
            return Some(Box::new(
                parquet::geospatial::statistics::GeospatialStatistics::new(None, geometry_types),
            ));
        };

        // The WkbBounder2D's rect bounder outputs (x, y) intervals; BoundingBox::new()
        // accepts xmin, xmax, ymin, ymax.
        let mut bbox = BoundingBox::new(x.lo(), x.hi(), y.lo(), y.hi());

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

    #[cfg(feature = "s2geography_tests")]
    use parquet::geospatial::bounding_box::BoundingBox;

    #[cfg(feature = "s2geography_tests")]
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

    #[cfg(feature = "s2geography_tests")]
    #[test]
    fn test_geography_accumulator() {
        use parquet_geospatial::testing::{wkb_point_xy, wkb_point_xyzm};
        use sedona_s2geography::rect_bounder::WkbGeographyBounder;

        // The geography bounder produces slightly expanded bounds compared to the
        // geometry bounder due to spherical interpolation along geodesics.
        const EPSILON: f64 = 1e-10;
        let mut accumulator =
            GeographyGeoStatsAccumulator::new(Box::new(WkbGeographyBounder::default()));

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
