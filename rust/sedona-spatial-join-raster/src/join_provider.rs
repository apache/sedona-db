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

use arrow_array::{builder::BinaryBuilder, Array, ArrayRef, StructArray};
use datafusion_common::{exec_datafusion_err, JoinType, Result};
use datafusion_expr::ColumnarValue;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err, SpatialJoinOptions};
use sedona_expr::statistics::GeoStatistics;
use sedona_geometry::{transform::CrsEngine, wkb_factory::write_wkb_polygon};
use sedona_proj::transform::with_global_proj_engine;
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::crs_utils::resolve_crs;
use sedona_raster_functions::footprint::{raster_footprint_corners, write_convexhull_wkb};
use sedona_schema::{
    crs::{lnglat, CoordinateReferenceSystem, Crs},
    datatypes::SedonaType,
    datatypes::WKB_GEOMETRY,
};
use sedona_spatial_join::{
    index::{spatial_index_builder::SpatialJoinBuildMetrics, SpatialIndexBuilder},
    join_provider::{DefaultSpatialJoinProvider, SpatialJoinProvider},
    operand_evaluator::{EvaluatedGeometryArray, EvaluatedGeometryArrayFactory},
    utils::bounds::Bounds2D,
    SpatialPredicate,
};

/// [`SpatialJoinProvider`] for raster/geometry spatial joins.
///
/// The R-tree index builder and the memory estimate delegate to the default
/// provider; only the operand evaluator is raster-aware. The factory it produces
/// reprojects raster footprints into `target_crs` (the geometry operand's CRS).
#[derive(Debug)]
pub(crate) struct RasterJoinProvider {
    default: DefaultSpatialJoinProvider,
    target_crs: Crs,
}

impl RasterJoinProvider {
    pub(crate) fn new(target_crs: Crs) -> Self {
        Self {
            default: DefaultSpatialJoinProvider,
            target_crs,
        }
    }
}

impl SpatialJoinProvider for RasterJoinProvider {
    fn try_new_spatial_index_builder(
        &self,
        schema: arrow_schema::SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Box<dyn SpatialIndexBuilder>> {
        // Footprints are ordinary planar WKB polygons, so the default R-tree
        // builder and WKB refiner apply unchanged.
        self.default.try_new_spatial_index_builder(
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
        )
    }

    fn estimate_extra_memory_usage(
        &self,
        geo_stats: &GeoStatistics,
        spatial_predicate: &SpatialPredicate,
        options: &SpatialJoinOptions,
    ) -> usize {
        self.default
            .estimate_extra_memory_usage(geo_stats, spatial_predicate, options)
    }

    fn evaluated_array_factory(&self) -> Arc<dyn EvaluatedGeometryArrayFactory> {
        Arc::new(RasterGeometryArrayFactory {
            target_crs: self.target_crs.clone(),
        })
    }
}

/// Evaluates the operands of a raster/geometry spatial predicate.
///
/// The same factory sees both operands of the join. A raster operand is turned
/// into its footprint — the convex hull of the raster's four corners
/// (see [`RasterGeometryArrayFactory::evaluate_raster`]) — and a geometry operand
/// is evaluated with the default planar behavior, since the geometry is already
/// in the target CRS.
///
/// Cross-CRS raster footprints are indexed and refined as the convex hull of the
/// raster's four reprojected corners; projection curvature along the edges is not
/// modeled, so for large-extent rasters reprojected between very different CRSs
/// the hull can slightly under-cover the true footprint (rare missed matches at
/// the extreme edges). Same-CRS joins are exact.
#[derive(Debug)]
struct RasterGeometryArrayFactory {
    /// The geometry operand's CRS: the common CRS this join compares in. Raster
    /// footprints are reprojected into it. WGS84 when the geometry operand carries
    /// no CRS — a missing CRS is assumed WGS84 (matching the `RS_*` kernel and
    /// Sedona Spark), so the CRS-less geometry is compared as-is in a WGS84 frame.
    target_crs: Crs,
}

impl EvaluatedGeometryArrayFactory for RasterGeometryArrayFactory {
    fn try_new_evaluated_array(
        &self,
        geometry_array: ArrayRef,
        sedona_type: &SedonaType,
        distance_columnar_value: Option<&ColumnarValue>,
    ) -> Result<EvaluatedGeometryArray> {
        // Relation predicates (Intersects/Contains/Within) carry no distance;
        // this factory is only wired up for those by the raster planner.
        if distance_columnar_value.is_some() {
            return sedona_internal_err!(
                "Raster spatial joins do not support a distance predicate"
            );
        }

        match sedona_type {
            SedonaType::Raster => self.evaluate_raster(geometry_array),
            // The geometry operand is already in the target CRS, so its default
            // planar evaluation (Cartesian WKB bounds + WKB) is exactly right.
            _ => EvaluatedGeometryArray::try_new(geometry_array, sedona_type),
        }
    }
}

impl RasterGeometryArrayFactory {
    /// Evaluate a raster struct array into footprint polygons plus bounding
    /// rectangles, reprojecting each footprint into the target CRS.
    fn evaluate_raster(&self, raster_array: ArrayRef) -> Result<EvaluatedGeometryArray> {
        let struct_array = raster_array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                sedona_internal_datafusion_err!("Expected StructArray for raster operand")
            })?;
        let rasters = RasterStructArray::try_new(struct_array)
            .map_err(|e| exec_datafusion_err!("Failed to read raster array: {e}"))?;

        let num_rows = rasters.len();
        let mut builder = BinaryBuilder::with_capacity(num_rows, num_rows * 96);
        let mut rects = Vec::with_capacity(num_rows);

        // A missing CRS on either side is assumed to be WGS84 (matching the `RS_*`
        // kernel and Sedona Spark); resolve it once rather than per row.
        let lnglat_crs = lnglat();
        let wgs84 = lnglat_crs.as_deref().expect("lnglat() is always Some");

        with_global_proj_engine(|engine| {
            for i in 0..num_rows {
                // A null raster produces a null footprint that never matches.
                if rasters.is_null(i) {
                    builder.append_null();
                    rects.push(Bounds2D::empty());
                    continue;
                }

                let raster = rasters
                    .get(i)
                    .map_err(|e| exec_datafusion_err!("Failed to read raster row {i}: {e}"))?;
                let rect = self.append_footprint(&raster, wgs84, engine, &mut builder)?;
                rects.push(rect);
            }
            Ok(())
        })?;

        let footprint_array: ArrayRef = Arc::new(builder.finish());
        EvaluatedGeometryArray::try_new_with_rects(footprint_array, rects, &WKB_GEOMETRY)
    }

    /// Append one raster's footprint WKB to `builder` and return its bounding
    /// rectangle, reconciling the raster's CRS against the target CRS. `wgs84` is
    /// the WGS84 CRS a missing side is assumed to be.
    ///
    /// CRS rules mirror the `RS_*` kernel: a missing CRS on either side is assumed
    /// to be WGS84 (matching Sedona Spark), after which an equal CRS compares
    /// directly and a genuine CRS difference reprojects the footprint's four
    /// corners into the target CRS. The footprint is always the convex hull of the
    /// four corners — projection curvature along the edges is not modeled (see the
    /// type-level note on [`RasterGeometryArrayFactory`]).
    fn append_footprint(
        &self,
        raster: &dyn RasterRef,
        wgs84: &(dyn CoordinateReferenceSystem + Send + Sync),
        engine: &dyn CrsEngine,
        builder: &mut BinaryBuilder,
    ) -> Result<Bounds2D> {
        let raster_crs = resolve_crs(raster.crs())?;
        let corners = raster_footprint_corners(raster);

        // Assume WGS84 for whichever side lacks a CRS, then compare in the target
        // frame. Both-absent and same-CRS collapse to a direct comparison.
        let raster_crs = raster_crs.as_deref().unwrap_or(wgs84);
        let target_crs = self.target_crs.as_deref().unwrap_or(wgs84);

        if raster_crs.crs_equals(target_crs) {
            // Same (possibly assumed-WGS84) CRS: compare directly. The footprint is
            // byte-identical to the one the `RS_*` kernel builds.
            write_convexhull_wkb(raster, builder)?;
            builder.append_value([]);
            Ok(bounds_from_coords(&corners))
        } else {
            // Genuine CRS difference: reproject the four corners into the target CRS
            // and emit their convex hull.
            append_reprojected_footprint(corners, raster_crs, target_crs, engine, builder)
        }
    }
}

/// Reproject a raster's four `corners` from `from_crs` to `to_crs`, append the
/// convex hull of the reprojected corners to `builder`, and return its bounding
/// rectangle.
fn append_reprojected_footprint(
    corners: [(f64, f64); 4],
    from_crs: &(dyn CoordinateReferenceSystem + Send + Sync),
    to_crs: &(dyn CoordinateReferenceSystem + Send + Sync),
    engine: &dyn CrsEngine,
    builder: &mut BinaryBuilder,
) -> Result<Bounds2D> {
    let transform = engine
        .get_transform_crs_to_crs(&from_crs.to_crs_string(), &to_crs.to_crs_string(), None, "")
        .map_err(|e| exec_datafusion_err!("CRS transform error: {e}"))?;

    let mut hull = corners;
    for corner in hull.iter_mut() {
        transform
            .transform_coord(corner)
            .map_err(|e| exec_datafusion_err!("Transform error: {e}"))?;
    }

    // Closed ring: the four reprojected corners plus the first repeated.
    write_wkb_polygon(
        builder,
        [hull[0], hull[1], hull[2], hull[3], hull[0]].into_iter(),
    )
    .map_err(|e| exec_datafusion_err!("Failed to write footprint WKB: {e}"))?;
    builder.append_value([]);
    Ok(bounds_from_coords(&hull))
}

/// Bounding rectangle of a set of coordinates. [`Bounds2D::new`] enlarges the
/// f32 bounds outward so the rectangle conservatively contains every input
/// coordinate.
fn bounds_from_coords(coords: &[(f64, f64)]) -> Bounds2D {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &(x, y) in coords {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    Bounds2D::new((min_x, max_x), (min_y, max_y))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bounds_from_coords_covers_corners() {
        let corners = [(2.0, 3.08), (2.29, 2.4), (2.09, 3.0), (2.2, 2.48)];
        let bounds = bounds_from_coords(&corners);
        let ((min_x, max_x), (min_y, max_y)) = bounds.into_inner();

        // f32 bounds must conservatively contain the f64 extent.
        assert!((min_x as f64) <= 2.0);
        assert!((max_x as f64) >= 2.29);
        assert!((min_y as f64) <= 2.4);
        assert!((max_y as f64) >= 3.08);
    }

    // --- Evaluator tests over real rasters -------------------------------

    use arrow_array::BinaryArray;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::{BandMetadata, RasterMetadata};
    use sedona_schema::crs::{deserialize_crs, lnglat};
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::{BandDataType, StorageType};
    use sedona_testing::rasters::generate_test_rasters;

    /// A 1x1 raster over world coords (0,0)-(1,1), with `crs` (or none).
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

    /// Footprint WKB the factory produces for row 0 of a single-row raster,
    /// comparing footprints against `target_crs`.
    fn footprint_bytes(target_crs: Crs, raster: arrow_array::StructArray) -> Vec<u8> {
        let factory = RasterGeometryArrayFactory { target_crs };
        let evaluated = factory.evaluate_raster(Arc::new(raster)).unwrap();
        evaluated
            .geometry_array()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .value(0)
            .to_vec()
    }

    /// The native (non-reprojected) convex-hull footprint the `RS_*` kernel builds
    /// for row 0 of `raster`.
    fn native_hull(raster: &arrow_array::StructArray) -> Vec<u8> {
        let arr = RasterStructArray::try_new(raster).unwrap();
        let mut wkb = Vec::new();
        write_convexhull_wkb(&arr.get(0).unwrap(), &mut wkb).unwrap();
        wkb
    }

    /// Same-CRS: the footprint is byte-identical to the `RS_*` kernel's convex
    /// hull, its MBR covers the hand-computed corners of raster 1, and a null
    /// raster yields a null footprint with an empty rect.
    #[test]
    fn same_crs_footprint_matches_kernel_and_bounds() {
        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Expected footprint bytes for raster 1, straight from the shared kernel helper.
        let mut expected_wkb = Vec::new();
        {
            let arr = RasterStructArray::try_new(&rasters).unwrap();
            let raster1 = arr.get(1).unwrap();
            write_convexhull_wkb(&raster1, &mut expected_wkb).unwrap();
        }

        let factory = RasterGeometryArrayFactory {
            target_crs: lnglat(),
        };
        let evaluated = factory.evaluate_raster(Arc::new(rasters)).unwrap();
        let footprints = evaluated
            .geometry_array()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();

        // Null raster -> null footprint, empty rect.
        assert!(footprints.is_null(0));
        assert!(evaluated.rect(0).is_empty());

        // Same-CRS footprint is byte-identical to the kernel's convex hull.
        assert_eq!(footprints.value(1), expected_wkb.as_slice());

        // Raster 1's footprint corners (from GDAL): (2.0, 3.0), (2.2, 3.08),
        // (2.29, 2.48), (2.09, 2.4). The MBR must conservatively cover them.
        let ((min_x, max_x), (min_y, max_y)) = evaluated.rect(1).clone().into_inner();
        assert!((min_x as f64) <= 2.0);
        assert!((max_x as f64) >= 2.29);
        assert!((min_y as f64) <= 2.4);
        assert!((max_y as f64) >= 3.08);
    }

    /// A raster with a CRS joined against a CRS-less geometry (target CRS `None`)
    /// no longer errors: the missing target CRS is assumed to be WGS84 (matching
    /// the `RS_*` kernel and Sedona Spark). The WGS84 raster then compares
    /// directly, giving the kernel's native hull and matching an explicit WGS84
    /// target.
    #[test]
    fn raster_with_crs_and_crsless_target_assumes_wgs84() {
        let expected = native_hull(&build_unit_raster(Some("OGC:CRS84")));

        let assumed = footprint_bytes(None, build_unit_raster(Some("OGC:CRS84")));
        let tagged = footprint_bytes(lnglat(), build_unit_raster(Some("OGC:CRS84")));

        assert_eq!(
            assumed, expected,
            "assumed-WGS84 target compares directly (native hull)"
        );
        assert_eq!(
            assumed, tagged,
            "assumed WGS84 must match an explicit WGS84 target"
        );
    }

    /// A CRS-less raster joined against a geometry that has a CRS also assumes
    /// WGS84 for the raster. Against a WGS84 target it compares directly; against a
    /// non-WGS84 target it reprojects WGS84 -> target, matching a raster explicitly
    /// tagged WGS84.
    #[test]
    fn crsless_raster_assumes_wgs84() {
        let native = native_hull(&build_unit_raster(None));

        // WGS84 target: the assumed-WGS84 raster compares directly (native hull).
        let direct = footprint_bytes(lnglat(), build_unit_raster(None));
        assert_eq!(
            direct, native,
            "assumed-WGS84 raster vs WGS84 target compares directly"
        );

        // Non-WGS84 target: reproject WGS84 -> target, matching an explicit tag.
        let epsg_3857 = deserialize_crs("EPSG:3857").unwrap();
        let assumed = footprint_bytes(epsg_3857.clone(), build_unit_raster(None));
        let tagged = footprint_bytes(epsg_3857, build_unit_raster(Some("OGC:CRS84")));
        assert_eq!(
            assumed, tagged,
            "assumed WGS84 must match an explicit WGS84 tag under reprojection"
        );
        assert_ne!(
            assumed, native,
            "reprojection to EPSG:3857 must change the footprint"
        );
    }

    /// A CRS-less raster with a CRS-less geometry compares directly (no error,
    /// no reprojection): the footprint is the native convex hull.
    #[test]
    fn crsless_both_sides_compares_directly() {
        let rasters = build_unit_raster(None);
        let mut expected_wkb = Vec::new();
        {
            let arr = RasterStructArray::try_new(&rasters).unwrap();
            let raster0 = arr.get(0).unwrap();
            write_convexhull_wkb(&raster0, &mut expected_wkb).unwrap();
        }

        let factory = RasterGeometryArrayFactory { target_crs: None };
        let evaluated = factory
            .try_new_evaluated_array(Arc::new(rasters), &RASTER, None)
            .unwrap();
        let footprints = evaluated
            .geometry_array()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(footprints.value(0), expected_wkb.as_slice());
    }
}
