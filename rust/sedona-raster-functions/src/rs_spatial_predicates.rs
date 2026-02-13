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

//! RS_Intersects, RS_Contains, RS_Within functions
//!
//! These functions test spatial relationships between rasters and geometries.
//! CRS transformation rules:
//! - If the raster or geometry does not have a defined SRID, it is assumed to be in WGS84
//! - If both sides are in the same CRS, perform the relationship test directly
//! - Otherwise, both sides will be transformed to WGS84 before the relationship test

use std::sync::Arc;

use crate::crs_utils::crs_transform_wkb;
use crate::crs_utils::default_crs;
use crate::crs_utils::resolve_crs;
use crate::executor::RasterExecutor;
use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::DataFusionError;
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::write_wkb_polygon;
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use sedona_tg::tg;

/// RS_Intersects() scalar UDF documentation
///
/// Returns true if raster A intersects geometry B.
pub fn rs_intersects_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_intersects",
        vec![
            Arc::new(RsSpatialPredicate::<tg::Intersects>::raster_geom()),
            Arc::new(RsSpatialPredicate::<tg::Intersects>::geom_raster()),
            Arc::new(RsSpatialPredicate::<tg::Intersects>::raster_raster()),
        ],
        Volatility::Immutable,
    )
}

/// RS_Contains() scalar UDF
///
/// Returns true if raster A contains geometry B.
pub fn rs_contains_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_contains",
        vec![
            Arc::new(RsSpatialPredicate::<tg::Contains>::raster_geom()),
            Arc::new(RsSpatialPredicate::<tg::Contains>::geom_raster()),
            Arc::new(RsSpatialPredicate::<tg::Contains>::raster_raster()),
        ],
        Volatility::Immutable,
    )
}

/// RS_Within() scalar UDF
///
/// Returns true if raster A is within geometry B.
pub fn rs_within_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_within",
        vec![
            Arc::new(RsSpatialPredicate::<tg::Within>::raster_geom()),
            Arc::new(RsSpatialPredicate::<tg::Within>::geom_raster()),
            Arc::new(RsSpatialPredicate::<tg::Within>::raster_raster()),
        ],
        Volatility::Immutable,
    )
}

/// Argument order for the spatial predicate
#[derive(Debug, Clone, Copy)]
enum ArgOrder {
    /// First arg is raster, second is geometry
    RasterGeom,
    /// First arg is geometry, second is raster
    GeomRaster,
    /// Both args are rasters
    RasterRaster,
}

#[derive(Debug)]
struct RsSpatialPredicate<Op: tg::BinaryPredicate> {
    arg_order: ArgOrder,
    _op: std::marker::PhantomData<Op>,
}

impl<Op: tg::BinaryPredicate> RsSpatialPredicate<Op> {
    fn raster_geom() -> Self {
        Self {
            arg_order: ArgOrder::RasterGeom,
            _op: std::marker::PhantomData,
        }
    }

    fn geom_raster() -> Self {
        Self {
            arg_order: ArgOrder::GeomRaster,
            _op: std::marker::PhantomData,
        }
    }

    fn raster_raster() -> Self {
        Self {
            arg_order: ArgOrder::RasterRaster,
            _op: std::marker::PhantomData,
        }
    }
}

impl<Op: tg::BinaryPredicate + Send + Sync> SedonaScalarKernel for RsSpatialPredicate<Op> {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = match self.arg_order {
            ArgOrder::RasterGeom => ArgMatcher::new(
                vec![ArgMatcher::is_raster(), ArgMatcher::is_geometry()],
                SedonaType::Arrow(DataType::Boolean),
            ),
            ArgOrder::GeomRaster => ArgMatcher::new(
                vec![ArgMatcher::is_geometry(), ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Boolean),
            ),
            ArgOrder::RasterRaster => ArgMatcher::new(
                vec![ArgMatcher::is_raster(), ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Boolean),
            ),
        };

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        match self.arg_order {
            ArgOrder::RasterGeom => self.invoke_raster_geom(arg_types, args),
            ArgOrder::GeomRaster => self.invoke_geom_raster(arg_types, args),
            ArgOrder::RasterRaster => self.invoke_raster_raster(arg_types, args),
        }
    }
}

impl<Op: tg::BinaryPredicate + Send + Sync> RsSpatialPredicate<Op> {
    /// Invoke RS_<Predicate>(raster, geometry)
    fn invoke_raster_geom(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Ensure executor always sees (raster, geom)
        let exec_arg_types = vec![arg_types[0].clone(), arg_types[1].clone()];
        let exec_args = vec![args[0].clone(), args[1].clone()];
        let executor = RasterExecutor::new(&exec_arg_types, &exec_args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
        let mut raster_wkb = Vec::with_capacity(CONVEXHULL_WKB_SIZE);

        executor.execute_raster_wkb_crs_void(|raster_opt, maybe_wkb, maybe_geom_crs| {
            match (raster_opt, maybe_wkb) {
                (Some(raster), Some(geom_wkb)) => {
                    raster_wkb.clear();
                    write_convexhull_wkb(raster, &mut raster_wkb)?;

                    let result = evaluate_predicate_with_crs::<Op>(
                        &raster_wkb,
                        raster.crs(),
                        geom_wkb,
                        maybe_geom_crs,
                        false,
                    )?;
                    builder.append_value(result);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }

    /// Invoke RS_<Predicate>(geometry, raster)
    fn invoke_geom_raster(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Reorder so executor always sees (raster, geom)
        let exec_arg_types = vec![arg_types[1].clone(), arg_types[0].clone()];
        let exec_args = vec![args[1].clone(), args[0].clone()];
        let executor = RasterExecutor::new(&exec_arg_types, &exec_args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
        let mut raster_wkb = Vec::with_capacity(CONVEXHULL_WKB_SIZE);

        executor.execute_raster_wkb_crs_void(|raster_opt, maybe_wkb, maybe_geom_crs| {
            match (raster_opt, maybe_wkb) {
                (Some(raster), Some(geom_wkb)) => {
                    raster_wkb.clear();
                    write_convexhull_wkb(raster, &mut raster_wkb)?;

                    // Note: order is geometry, raster for the predicate
                    let result = evaluate_predicate_with_crs::<Op>(
                        geom_wkb,
                        maybe_geom_crs,
                        &raster_wkb,
                        raster.crs(),
                        true,
                    )?;
                    builder.append_value(result);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }

    /// Invoke RS_<Predicate>(raster1, raster2)
    fn invoke_raster_raster(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Ensure executor always sees (raster, raster)
        let exec_arg_types = vec![arg_types[0].clone(), arg_types[1].clone()];
        let exec_args = vec![args[0].clone(), args[1].clone()];
        let executor = RasterExecutor::new(&exec_arg_types, &exec_args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
        let mut wkb0 = Vec::with_capacity(CONVEXHULL_WKB_SIZE);
        let mut wkb1 = Vec::with_capacity(CONVEXHULL_WKB_SIZE);

        executor.execute_raster_raster_void(|_i, r0_opt, r1_opt| {
            match (r0_opt, r1_opt) {
                (Some(r0), Some(r1)) => {
                    wkb0.clear();
                    wkb1.clear();
                    write_convexhull_wkb(r0, &mut wkb0)?;
                    write_convexhull_wkb(r1, &mut wkb1)?;

                    let result =
                        evaluate_predicate_with_crs::<Op>(&wkb0, r0.crs(), &wkb1, r1.crs(), false)?;
                    builder.append_value(result);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Evaluate a spatial predicate with CRS handling
///
/// Rules:
/// - If no CRS defined, assume WGS84
/// - If both same CRS, compare directly
/// - Otherwise, try transforming one side to the other's CRS for comparison.
///   If that fails, transform both to WGS84 and compare.
fn evaluate_predicate_with_crs<Op: tg::BinaryPredicate>(
    wkb_a: &[u8],
    crs_a: Option<&str>,
    wkb_b: &[u8],
    crs_b: Option<&str>,
    from_a_to_b: bool,
) -> Result<bool> {
    let crs_a = resolve_crs(crs_a)?;
    let crs_b = resolve_crs(crs_b)?;

    if crs_a.crs_equals(crs_b.as_ref()) {
        return evaluate_predicate::<Op>(wkb_a, wkb_b);
    }

    if from_a_to_b {
        // Transform A to B's CRS for comparison
        if let Ok(wkb_a) = crs_transform_wkb(wkb_a, crs_a.as_ref(), crs_b.as_ref()) {
            return evaluate_predicate::<Op>(&wkb_a, wkb_b);
        }
    } else {
        // Transform B to A's CRS for comparison
        if let Ok(wkb_b) = crs_transform_wkb(wkb_b, crs_b.as_ref(), crs_a.as_ref()) {
            return evaluate_predicate::<Op>(wkb_a, &wkb_b);
        }
    }

    // If CRS transformation fails, fall back to transforming both to default CRS (WGS84) for comparison
    let default_crs = default_crs();
    let wkb_a = crs_transform_wkb(wkb_a, crs_a.as_ref(), default_crs)?;
    let wkb_b = crs_transform_wkb(wkb_b, crs_b.as_ref(), default_crs)?;
    evaluate_predicate::<Op>(&wkb_a, &wkb_b)
}

/// Evaluate a spatial predicate between two WKB geometries
fn evaluate_predicate<Op: tg::BinaryPredicate>(wkb_a: &[u8], wkb_b: &[u8]) -> Result<bool> {
    let geom_a = tg::Geom::parse_wkb(wkb_a, tg::IndexType::Default)
        .map_err(|e| DataFusionError::Execution(format!("Failed to parse WKB A: {e}")))?;
    let geom_b = tg::Geom::parse_wkb(wkb_b, tg::IndexType::Default)
        .map_err(|e| DataFusionError::Execution(format!("Failed to parse WKB B: {e}")))?;

    Ok(Op::evaluate(&geom_a, &geom_b))
}

/// Exact WKB byte size for a 2D polygon with 1 ring of 5 points (the raster convex hull).
///
/// Layout: polygon header (1 byte order + 4 type + 4 ring count)
///       + ring header (4 point count)
///       + 5 points × 2 coordinates × 8 bytes
///       = 9 + 4 + 80 = 93
const CONVEXHULL_WKB_SIZE: usize = 93;

/// Create WKB for a convex hull polygon for the raster
fn write_convexhull_wkb(raster: &dyn RasterRef, out: &mut impl std::io::Write) -> Result<()> {
    let width = raster.metadata().width() as i64;
    let height = raster.metadata().height() as i64;

    let (ulx, uly) = to_world_coordinate(raster, 0, 0);
    let (urx, ury) = to_world_coordinate(raster, width, 0);
    let (lrx, lry) = to_world_coordinate(raster, width, height);
    let (llx, lly) = to_world_coordinate(raster, 0, height);

    write_wkb_polygon(
        out,
        [(ulx, uly), (urx, ury), (lrx, lry), (llx, lly), (ulx, uly)].into_iter(),
    )
    .map_err(|e| DataFusionError::External(e.into()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::crs_utils::crs_transform_coord;

    use super::*;
    use arrow_array::{create_array, ArrayRef};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::{BandMetadata, RasterMetadata};
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::crs::OGC_CRS84_PROJJSON;
    use sedona_schema::datatypes::Edges;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_schema::raster::{BandDataType, StorageType};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array as create_geom_array;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn rs_intersects_udf_docs() {
        let udf: ScalarUDF = rs_intersects_udf().into();
        assert_eq!(udf.name(), "rs_intersects");
    }

    #[test]
    fn rs_contains_udf_docs() {
        let udf: ScalarUDF = rs_contains_udf().into();
        assert_eq!(udf.name(), "rs_contains");
    }

    #[test]
    fn rs_within_udf_docs() {
        let udf: ScalarUDF = rs_within_udf().into();
        assert_eq!(udf.name(), "rs_within");
    }

    #[rstest]
    fn rs_intersects_raster_geom() {
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Test rasters:
        // Raster 1: corners at approximately (2.0, 3.0), (2.2, 3.08), (2.29, 2.48), (2.09, 2.4)
        // Raster 2: corners at approximately (3.0, 4.0), (3.6, 4.24), (3.84, 2.64), (3.24, 2.4)

        // Points that should intersect with raster 1 (approximately)
        // Point inside raster 1
        let geoms = create_geom_array(
            &[
                None,
                Some("POINT (2.15 2.75)"), // Inside raster 1
                Some("POINT (0.0 0.0)"),   // Outside all rasters
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(false)]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_intersects_raster_geom_crs_mismatch() {
        let udf = rs_intersects_udf();
        let geom_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, geom_type.clone()]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();
        let (x, y) = crs_transform_coord((2.15, 2.75), "OGC:CRS84", "EPSG:3857").unwrap();
        let point_3857 = format!("POINT ({} {})", x, y);
        let wkt_values: [Option<&str>; 3] = [None, Some(point_3857.as_str()), Some("POINT (0 0)")];

        let geoms = create_geom_array(&wkt_values, &geom_type);

        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(false)]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[test]
    fn rs_intersects_raster_geom_projjson_crs() {
        // Use an authority code at the geometry type-level so we don't need PROJ for this test.
        // The raster side exercises PROJJSON CRS deserialization.
        let geom_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:4326").unwrap());

        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, geom_type.clone()]);

        // 1x1 raster whose convex hull covers (0,0) to (1,1)
        let mut builder = RasterBuilder::new(1);
        let raster_metadata = RasterMetadata {
            width: 1,
            height: 1,
            upperleft_x: 0.0,
            upperleft_y: 1.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
        builder
            .start_raster(&raster_metadata, Some(OGC_CRS84_PROJJSON))
            .unwrap();
        builder
            .start_band(BandMetadata {
                datatype: BandDataType::UInt8,
                nodata_value: None,
                storage_type: StorageType::InDb,
                outdb_url: None,
                outdb_band_id: None,
            })
            .unwrap();
        builder.band_data_writer().append_value([0u8]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let rasters = builder.finish().unwrap();

        let geoms = create_geom_array(&[Some("POINT (0.5 0.5)")], &geom_type);
        let expected: ArrayRef = create_array!(Boolean, [Some(true)]);
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_contains_raster_geom() {
        let udf = rs_contains_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Point inside raster 1 should be contained
        let geoms = create_geom_array(
            &[
                None,
                Some("POINT (2.15 2.75)"), // Inside raster 1
                Some("POINT (0.0 0.0)"),   // Outside all rasters
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(false)]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_within_raster_geom() {
        let udf = rs_within_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Test rasters:
        // Raster 1: corners at approximately (2.0, 3.0), (2.2, 3.08), (2.29, 2.48), (2.09, 2.4)

        // Large polygon that contains raster 1
        let geoms = create_geom_array(
            &[
                None,
                Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"), // Contains raster 1
                Some("POLYGON ((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0))"), // Does not contain raster 2
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(false)]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_intersects_geom_raster() {
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY, RASTER]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Test with geometry as first argument
        let geoms = create_geom_array(
            &[
                None,
                Some("POINT (2.15 2.75)"), // Inside raster 1
                Some("POINT (0.0 0.0)"),   // Outside all rasters
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(false)]);

        let result = tester
            .invoke_arrays(vec![geoms, Arc::new(rasters)])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_intersects_raster_raster() {
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, RASTER]);

        let rasters1 = generate_test_rasters(3, Some(0)).unwrap();
        let rasters2 = generate_test_rasters(3, Some(0)).unwrap();

        // Same rasters should intersect with themselves
        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(true)]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters1), Arc::new(rasters2)])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_intersects_null_handling() {
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Test with null geometry
        let geoms = create_geom_array(&[None::<&str>, None::<&str>, None::<&str>], &WKB_GEOMETRY);

        let expected: ArrayRef = create_array!(Boolean, [None, None, None]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();

        assert_array_equal(&result, &expected);
    }
}
