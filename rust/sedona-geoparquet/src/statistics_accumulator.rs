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

use datafusion_common::{exec_datafusion_err, Result, ScalarValue};
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
                            exec_datafusion_err!("Error parsing geography literal: {e}")
                        })?;
                        bounder.bound(&geog).map_err(|e| {
                            exec_datafusion_err!("Error bounding geography literal: {e}")
                        })?;

                        if let Some(distance) = distance {
                            bounder.expand_by_distance(distance);
                        }

                        if let Some((xmin, ymin, xmax, ymax)) = bounder.finish().map_err(|e| {
                            exec_datafusion_err!("Error finishing geography bounds: {e}")
                        })? {
                            return Ok(sedona_geometry::bounding_box::BoundingBox::xy(
                                (xmin, xmax),
                                (ymin, ymax),
                            ));
                        } else {
                            return Ok(sedona_geometry::bounding_box::BoundingBox::empty());
                        }
                    }
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
