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
//! Each function supports three overloads: (raster, geometry), (geometry, raster),
//! and (raster, raster). Rasters are compared via their convex hulls.
//!
//! CRS transformation rules:
//! - If neither side has a CRS, the comparison is performed directly without CRS transformation
//! - If one side has a CRS but the other does not, an error is returned
//! - If both sides are in the same CRS, perform the relationship test directly
//! - For raster/geometry pairs, the geometry is transformed into the raster's CRS
//! - For raster/raster pairs, the second raster is transformed into the first's CRS
//! - If the preferred transformation fails, both sides are transformed to WGS84 as a fallback

use std::sync::Arc;

use crate::crs_utils::crs_transform_wkb;
use crate::crs_utils::resolve_crs;
use crate::executor::RasterExecutor;
use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::exec_datafusion_err;
use datafusion_common::exec_err;
use datafusion_common::DataFusionError;
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::transform::CrsEngine;
use sedona_geometry::wkb_factory::write_wkb_polygon;
use sedona_proj::transform::with_global_proj_engine;
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_raster::traits::RasterRef;
use sedona_schema::crs::{lnglat, CrsRef};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use sedona_tg::tg;

/// RS_Intersects() scalar UDF
///
/// Returns true if the extents of the two arguments intersect. Supports
/// (raster, geometry), (geometry, raster), and (raster, raster) overloads.
/// Rasters are compared via their convex hulls.
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
/// Returns true if the first argument's extent completely contains the second.
/// Supports (raster, geometry), (geometry, raster), and (raster, raster) overloads.
/// Rasters are compared via their convex hulls.
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
/// Returns true if the first argument's extent is completely within the second.
/// Supports (raster, geometry), (geometry, raster), and (raster, raster) overloads.
/// Rasters are compared via their convex hulls.
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

        with_global_proj_engine(|engine| {
            executor.execute_raster_wkb_crs_void(|raster_opt, maybe_wkb, geom_crs| {
                match (raster_opt, maybe_wkb) {
                    (Some(raster), Some(geom_wkb)) => {
                        raster_wkb.clear();
                        write_convexhull_wkb(raster, &mut raster_wkb)?;

                        let raster_crs = resolve_crs(raster.crs())?;
                        let result = evaluate_predicate_with_crs::<Op>(
                            &raster_wkb,
                            raster_crs.as_deref(),
                            geom_wkb,
                            geom_crs,
                            false,
                            engine,
                        )?;
                        builder.append_value(result);
                    }
                    _ => builder.append_null(),
                }
                Ok(())
            })
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

        with_global_proj_engine(|engine| {
            executor.execute_raster_wkb_crs_void(|raster_opt, maybe_wkb, geom_crs| {
                match (raster_opt, maybe_wkb) {
                    (Some(raster), Some(geom_wkb)) => {
                        raster_wkb.clear();
                        write_convexhull_wkb(raster, &mut raster_wkb)?;

                        let raster_crs = resolve_crs(raster.crs())?;
                        // Note: order is geometry, raster for the predicate
                        let result = evaluate_predicate_with_crs::<Op>(
                            geom_wkb,
                            geom_crs,
                            &raster_wkb,
                            raster_crs.as_deref(),
                            true,
                            engine,
                        )?;
                        builder.append_value(result);
                    }
                    _ => builder.append_null(),
                }
                Ok(())
            })
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

        with_global_proj_engine(|engine| {
            executor.execute_raster_raster_void(|_i, r0_opt, r1_opt| {
                match (r0_opt, r1_opt) {
                    (Some(r0), Some(r1)) => {
                        wkb0.clear();
                        wkb1.clear();
                        write_convexhull_wkb(r0, &mut wkb0)?;
                        write_convexhull_wkb(r1, &mut wkb1)?;

                        let crs0 = resolve_crs(r0.crs())?;
                        let crs1 = resolve_crs(r1.crs())?;
                        let result = evaluate_predicate_with_crs::<Op>(
                            &wkb0,
                            crs0.as_deref(),
                            &wkb1,
                            crs1.as_deref(),
                            false,
                            engine,
                        )?;
                        builder.append_value(result);
                    }
                    _ => builder.append_null(),
                }
                Ok(())
            })
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Evaluate a spatial predicate with CRS handling
///
/// Rules:
/// - If neither side has a CRS, compare directly without transformation
/// - If one side has a CRS but the other does not, return an error
/// - If both same CRS, compare directly
/// - Otherwise, try transforming one side to the other's CRS for comparison.
///   If that fails, transform both to WGS84 and compare.
fn evaluate_predicate_with_crs<Op: tg::BinaryPredicate>(
    wkb_a: &[u8],
    crs_a: CrsRef<'_>,
    wkb_b: &[u8],
    crs_b: CrsRef<'_>,
    from_a_to_b: bool,
    engine: &dyn CrsEngine,
) -> Result<bool> {
    // If either side has no CRS, compare directly without transformation.
    let (crs_a, crs_b) = match (crs_a, crs_b) {
        (Some(a), Some(b)) => (a, b),
        (None, None) => return evaluate_predicate::<Op>(wkb_a, wkb_b),
        (Some(_), None) => {
            return exec_err!(
                "Cannot evaluate spatial predicate: \
                left geometry has CRS but right geometry does not"
            )
        }
        (None, Some(_)) => {
            return exec_err!(
                "Cannot evaluate spatial predicate: \
                right geometry has CRS but left geometry does not"
            )
        }
    };

    // If both CRSes are equal, compare directly.
    if crs_a.crs_equals(crs_b) {
        return evaluate_predicate::<Op>(wkb_a, wkb_b);
    }

    // Try preferred transformation direction.
    if from_a_to_b {
        if let Ok(wkb_a) = crs_transform_wkb(wkb_a, crs_a, crs_b, engine) {
            return evaluate_predicate::<Op>(&wkb_a, wkb_b);
        }
    } else if let Ok(wkb_b) = crs_transform_wkb(wkb_b, crs_b, crs_a, engine) {
        return evaluate_predicate::<Op>(wkb_a, &wkb_b);
    }

    // Fallback: transform both sides to WGS84 for comparison.
    let lnglat_crs = lnglat().expect("lnglat() should always return Some");
    let wkb_a = crs_transform_wkb(wkb_a, crs_a, lnglat_crs.as_ref(), engine)?;
    let wkb_b = crs_transform_wkb(wkb_b, crs_b, lnglat_crs.as_ref(), engine)?;
    evaluate_predicate::<Op>(&wkb_a, &wkb_b)
}

/// Evaluate a spatial predicate between two WKB geometries
fn evaluate_predicate<Op: tg::BinaryPredicate>(wkb_a: &[u8], wkb_b: &[u8]) -> Result<bool> {
    let geom_a = tg::Geom::parse_wkb(wkb_a, tg::IndexType::Default)
        .map_err(|e| exec_datafusion_err!("Failed to parse WKB A: {e}"))?;
    let geom_b = tg::Geom::parse_wkb(wkb_b, tg::IndexType::Default)
        .map_err(|e| exec_datafusion_err!("Failed to parse WKB B: {e}"))?;

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

    /// Transform a coordinate from one CRS to another, using the provided CRS engine.
    fn crs_transform_coord(
        coord: (f64, f64),
        from_crs: &str,
        to_crs: &str,
        engine: &dyn CrsEngine,
    ) -> Result<(f64, f64)> {
        let trans = engine
            .get_transform_crs_to_crs(from_crs, to_crs, None, "")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let mut coord = coord;
        trans
            .transform_coord(&mut coord)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(coord)
    }

    /// Build a 1×1 raster whose convex hull covers (0,0) to (1,1).
    ///
    /// If `crs` is `None`, the raster has no CRS.
    fn build_unit_raster(crs: Option<&str>) -> arrow_array::StructArray {
        let mut builder = RasterBuilder::new(1);
        let metadata = RasterMetadata {
            width: 1,
            height: 1,
            upperleft_x: 0.0,
            upperleft_y: 1.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
        builder.start_raster(&metadata, crs).unwrap();
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
        builder.finish().unwrap()
    }

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
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, geom_type.clone()]);

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
            &geom_type,
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
        let (x, y) = with_global_proj_engine(|engine| {
            crs_transform_coord((2.15, 2.75), "OGC:CRS84", "EPSG:3857", engine)
        })
        .unwrap();
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

        let rasters = build_unit_raster(Some(OGC_CRS84_PROJJSON));

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
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, geom_type.clone()]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Point inside raster 1 should be contained
        let geoms = create_geom_array(
            &[
                None,
                Some("POINT (2.15 2.75)"), // Inside raster 1
                Some("POINT (0.0 0.0)"),   // Outside all rasters
            ],
            &geom_type,
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
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, geom_type.clone()]);

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
            &geom_type,
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
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf.into(), vec![geom_type.clone(), RASTER]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Test with geometry as first argument
        let geoms = create_geom_array(
            &[
                None,
                Some("POINT (2.15 2.75)"), // Inside raster 1
                Some("POINT (0.0 0.0)"),   // Outside all rasters
            ],
            &geom_type,
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

    /// When neither the raster nor the geometry has a CRS, the comparison
    /// should succeed (both sides are in an unknown/assumed-same CRS).
    #[test]
    fn rs_intersects_both_no_crs_succeeds() {
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);

        // Raster covers (0,0)–(1,1) with no CRS
        let rasters = build_unit_raster(None);
        let geoms = create_geom_array(&[Some("POINT (0.5 0.5)")], &WKB_GEOMETRY);

        let expected: ArrayRef = create_array!(Boolean, [Some(true)]);
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();
        assert_array_equal(&result, &expected);
    }

    /// When one side has a CRS but the other does not, an error must be returned.
    ///
    /// Covers three overloads:
    /// - (raster with CRS, geometry without CRS) — raster_geom
    /// - (geometry with CRS, raster without CRS) — geom_raster
    /// - (raster with CRS, raster without CRS) — raster_raster
    #[test]
    fn rs_intersects_crs_mismatch_one_missing_errors() {
        let rasters_with_crs = build_unit_raster(Some("OGC:CRS84"));
        let rasters_no_crs = build_unit_raster(None);

        // raster (has CRS) + geometry (no CRS)
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![RASTER, WKB_GEOMETRY]);
        let geoms = create_geom_array(&[Some("POINT (0.5 0.5)")], &WKB_GEOMETRY);
        let err = tester
            .invoke_arrays(vec![Arc::new(rasters_with_crs.clone()), geoms])
            .unwrap_err();
        assert!(
            err.message().contains("has CRS but"),
            "unexpected error: {err}"
        );

        // geometry (has CRS) + raster (no CRS)
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf.clone().into(), vec![geom_type.clone(), RASTER]);
        let geoms = create_geom_array(&[Some("POINT (0.5 0.5)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![geoms, Arc::new(rasters_no_crs.clone())])
            .unwrap_err();
        assert!(
            err.message().contains("has CRS but"),
            "unexpected error: {err}"
        );

        // raster (has CRS) + raster (no CRS)
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, RASTER]);
        let err = tester
            .invoke_arrays(vec![
                Arc::new(rasters_with_crs.clone()),
                Arc::new(rasters_no_crs.clone()),
            ])
            .unwrap_err();
        assert!(
            err.message().contains("has CRS but"),
            "unexpected error: {err}"
        );
    }
}
