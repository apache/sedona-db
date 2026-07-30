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
use datafusion_common::{exec_datafusion_err, exec_err, JoinType, Result};
use datafusion_expr::ColumnarValue;
use sedona_common::{
    sedona_internal_datafusion_err, sedona_internal_err, SpatialJoinOptions, SpatialLibrary,
};
use sedona_expr::statistics::GeoStatistics;
use sedona_geometry::{
    interval::{Interval, IntervalTrait},
    transform::CrsEngine,
};
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::crs_utils::resolve_crs;
use sedona_raster_functions::footprint::{
    densify_footprint_ring, raster_footprint_corners, write_footprint_ring_wkb, write_footprint_wkb,
};
use sedona_schema::{
    crs::{CoordinateReferenceSystem, Crs},
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
/// reprojects raster footprints into `target_crs` (the geometry operand's CRS)
/// using the session's [`CrsEngine`], captured at plan time.
#[derive(Debug)]
pub(crate) struct RasterJoinProvider {
    default: DefaultSpatialJoinProvider,
    target_crs: Crs,
    engine: Arc<dyn CrsEngine + Send + Sync>,
}

impl RasterJoinProvider {
    pub(crate) fn new(target_crs: Crs, engine: Arc<dyn CrsEngine + Send + Sync>) -> Self {
        Self {
            default: DefaultSpatialJoinProvider,
            target_crs,
            engine,
        }
    }
}

/// Pin the join's refiner to `tg`, the engine the `RS_*` predicate kernel uses,
/// so the accelerated join and the kernel resolve boundary/touching cases
/// identically regardless of the session's `spatial_join.spatial_library`.
fn pin_refiner_to_tg(mut options: SpatialJoinOptions) -> SpatialJoinOptions {
    options.spatial_library = SpatialLibrary::Tg;
    options
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
            pin_refiner_to_tg(options),
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
        // Match the refiner the join actually uses (pinned to `tg`) so the memory
        // estimate reflects it.
        self.default.estimate_extra_memory_usage(
            geo_stats,
            spatial_predicate,
            &pin_refiner_to_tg(options.clone()),
        )
    }

    fn evaluated_array_factory(&self) -> Arc<dyn EvaluatedGeometryArrayFactory> {
        Arc::new(RasterGeometryArrayFactory {
            target_crs: self.target_crs.clone(),
            engine: self.engine.clone(),
        })
    }
}

/// Evaluates the operands of a raster/geometry spatial predicate.
///
/// The same factory sees both operands of the join. A raster operand is turned
/// into its footprint — the polygon through the raster's four corners
/// (see [`RasterGeometryArrayFactory::evaluate_raster`]) — and a geometry operand
/// is evaluated with the default planar behavior, since the geometry is already
/// in the target CRS.
///
/// A cross-CRS raster footprint densifies each of its four edges
/// ([`FOOTPRINT_POINTS_PER_EDGE`] interior points) in the raster's own CRS, where
/// the edges are exact straight lines, then reprojects every densified point into
/// the target CRS. The indexed and refined footprint therefore follows the
/// curved image of each edge rather than chording straight across it. Same-CRS
/// joins keep the exact four-corner footprint (no reprojection, no densification).
#[derive(Debug)]
struct RasterGeometryArrayFactory {
    /// The geometry operand's CRS: the common CRS this join compares in. Raster
    /// footprints are reprojected into it. `None` when the geometry operand
    /// carries no CRS (the raster operand must then also be CRS-less).
    target_crs: Crs,
    /// The session's CRS engine, used to reproject raster footprints into
    /// `target_crs`.
    engine: Arc<dyn CrsEngine + Send + Sync>,
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

        // Reused across rows so densifying each cross-CRS footprint (below)
        // allocates its point ring once, not per raster.
        let mut ring = Vec::new();

        let engine: &dyn CrsEngine = self.engine.as_ref();
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
            let rect = self.append_footprint(&raster, engine, &mut ring, &mut builder)?;
            rects.push(rect);
        }

        let footprint_array: ArrayRef = Arc::new(builder.finish());
        EvaluatedGeometryArray::try_new_with_rects(footprint_array, rects, &WKB_GEOMETRY)
    }

    /// Append one raster's footprint WKB to `builder` and return its bounding
    /// rectangle, reconciling the raster's CRS against the target CRS.
    ///
    /// CRS rules mirror the `RS_*` kernel: equal (or both-absent) CRS compares
    /// directly; a genuine CRS difference densifies and reprojects the footprint
    /// into the target CRS; a one-sided CRS is an error. Same-CRS footprints are
    /// the exact four-corner polygon; cross-CRS footprints densify each edge
    /// before reprojection so they follow its curve (see the type-level note on
    /// [`RasterGeometryArrayFactory`]). `ring` is scratch space reused across rows
    /// for the densified reprojected ring.
    fn append_footprint(
        &self,
        raster: &dyn RasterRef,
        engine: &dyn CrsEngine,
        ring: &mut Vec<(f64, f64)>,
        builder: &mut BinaryBuilder,
    ) -> Result<Bounds2D> {
        let raster_crs = resolve_crs(raster.crs())?;
        // A raster with no spatial (y, x) pair has no footprint — treat it as a
        // null footprint (no matches), mirroring the un-reprojectable-footprint path.
        let Ok(corners) = raster_footprint_corners(raster) else {
            builder.append_null();
            return Ok(Bounds2D::empty());
        };

        match (raster_crs.as_deref(), self.target_crs.as_deref()) {
            // No CRS on either side, or identical CRS: compare directly. The
            // footprint is byte-identical to the one the `RS_*` kernel builds.
            (None, None) => {
                write_footprint_wkb(corners, builder)?;
                builder.append_value([]);
                Ok(bounds_from_coords(&corners))
            }
            (Some(raster_crs), Some(target_crs)) if raster_crs.crs_equals(target_crs) => {
                write_footprint_wkb(corners, builder)?;
                builder.append_value([]);
                Ok(bounds_from_coords(&corners))
            }
            // Genuine CRS difference: densify each edge in the raster's own CRS
            // and reproject every point into the target CRS.
            (Some(raster_crs), Some(target_crs)) => {
                append_reprojected_footprint(corners, raster_crs, target_crs, engine, ring, builder)
            }
            (Some(_), None) => {
                exec_err!(
                    "Cannot evaluate spatial predicate: raster has a CRS but the geometry does not"
                )
            }
            (None, Some(_)) => {
                exec_err!(
                    "Cannot evaluate spatial predicate: geometry has a CRS but the raster does not"
                )
            }
        }
    }
}

/// Densify a raster's four `corners` in `from_crs`, reproject every densified
/// point to `to_crs`, append the resulting footprint polygon to `builder`, and
/// return its bounding rectangle.
///
/// The footprint is densified while it is still in the raster's own CRS, where
/// each edge is an exact straight line, and every densified point is then
/// reprojected — so the reprojected ring follows the curved image of each edge
/// instead of chording straight across it. `ring` is caller-managed scratch
/// space for the densified ring.
///
/// A transform that cannot be built, a transform that fails on any densified
/// point, or a reprojected point that is non-finite yields a *null* footprint for
/// this row (`append_null()` + [`Bounds2D::empty`], mirroring the null-raster
/// path) so the row contributes no matches — rather than writing a garbage
/// footprint/MBR or aborting the whole join. Recovering a failed reprojection via
/// a WGS84 fallback (as the `RS_*` kernel does for the comparison as a whole) is a
/// separate open question and is not attempted here.
fn append_reprojected_footprint(
    corners: [(f64, f64); 4],
    from_crs: &(dyn CoordinateReferenceSystem + Send + Sync),
    to_crs: &(dyn CoordinateReferenceSystem + Send + Sync),
    engine: &dyn CrsEngine,
    ring: &mut Vec<(f64, f64)>,
    builder: &mut BinaryBuilder,
) -> Result<Bounds2D> {
    let Ok(transform) = engine.get_transform_crs_to_crs(
        &from_crs.to_crs_string(),
        &to_crs.to_crs_string(),
        None,
        "",
    ) else {
        builder.append_null();
        return Ok(Bounds2D::empty());
    };

    // Densify the four-corner ring in the native CRS (exact straight edges),
    // then reproject every point so the footprint follows the reprojected curve.
    densify_footprint_ring(corners, ring);
    for point in ring.iter_mut() {
        if transform.transform_coord(point).is_err() {
            builder.append_null();
            return Ok(Bounds2D::empty());
        }
    }

    // A single non-finite reprojected point (e.g. a point outside the target
    // projection's valid domain) would poison the footprint WKB and its MBR, so
    // emit a null footprint for this row instead of writing garbage.
    if ring.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
        builder.append_null();
        return Ok(Bounds2D::empty());
    }

    write_footprint_ring_wkb(ring, builder)?;
    builder.append_value([]);
    Ok(bounds_from_coords(ring))
}

/// Bounding rectangle of a set of coordinates, accumulated as f64 [`Interval`]s
/// (the same primitive the geometry operand's bounds use). [`Bounds2D::new`]
/// enlarges the f32 bounds outward so the rectangle conservatively contains
/// every input coordinate.
fn bounds_from_coords(coords: &[(f64, f64)]) -> Bounds2D {
    let mut x = Interval::empty();
    let mut y = Interval::empty();
    for &(cx, cy) in coords {
        x.update_value(cx);
        y.update_value(cy);
    }
    Bounds2D::new(x, y)
}

#[cfg(test)]
mod test {
    use super::*;
    use sedona_proj::transform::LazyProjEngine;

    /// The session's default CRS engine. The same-CRS / no-CRS / one-sided-CRS
    /// paths exercised by these tests never invoke it (reprojection is tested
    /// separately with a mock engine below), so any engine works here.
    fn test_engine() -> Arc<dyn CrsEngine + Send + Sync> {
        Arc::new(LazyProjEngine)
    }

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

    use std::rc::Rc;

    use arrow_array::BinaryArray;
    use sedona_geometry::bounding_box::BoundingBox;
    use sedona_geometry::error::SedonaGeometryError;
    use sedona_geometry::transform::CrsTransform;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::{BandMetadata, RasterMetadata};
    use sedona_raster_functions::footprint::write_convexhull_wkb;
    use sedona_schema::crs::lnglat;
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
            engine: test_engine(),
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

    /// A raster with a CRS joined against a CRS-less geometry (target CRS
    /// `None`) is a one-sided CRS: it must error rather than silently compare.
    #[test]
    fn raster_with_crs_but_crsless_target_errors() {
        let rasters = build_unit_raster(Some("OGC:CRS84"));
        let factory = RasterGeometryArrayFactory {
            target_crs: None,
            engine: test_engine(),
        };
        let err = factory.evaluate_raster(Arc::new(rasters)).err().unwrap();
        assert!(err.message().contains("has a CRS but"), "unexpected: {err}");
    }

    /// A CRS-less raster joined against a geometry that has a CRS is also a
    /// one-sided CRS and must error.
    #[test]
    fn crsless_raster_with_crs_target_errors() {
        let rasters = build_unit_raster(None);
        let factory = RasterGeometryArrayFactory {
            target_crs: lnglat(),
            engine: test_engine(),
        };
        let err = factory.evaluate_raster(Arc::new(rasters)).err().unwrap();
        assert!(err.message().contains("has a CRS but"), "unexpected: {err}");
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

        let factory = RasterGeometryArrayFactory {
            target_crs: None,
            engine: test_engine(),
        };
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

    /// How a [`MockTransform`] treats each corner it is asked to reproject.
    #[derive(Debug, Clone, Copy)]
    enum CornerFate {
        /// Leave the coordinate unchanged (a well-behaved transform).
        Identity,
        /// Reproject the corner to a non-finite coordinate.
        NonFinite,
        /// Fail while transforming the corner.
        Error,
    }

    #[derive(Debug)]
    struct MockTransform(CornerFate);

    impl CrsTransform for MockTransform {
        fn transform_coord(&self, coord: &mut (f64, f64)) -> Result<(), SedonaGeometryError> {
            match self.0 {
                CornerFate::Identity => Ok(()),
                CornerFate::NonFinite => {
                    coord.1 = f64::INFINITY;
                    Ok(())
                }
                CornerFate::Error => Err(SedonaGeometryError::Invalid("boom".to_string())),
            }
        }
    }

    /// A [`CrsEngine`] that hands back a [`MockTransform`], or fails to build a
    /// transform at all when `fate` is `None`.
    #[derive(Debug)]
    struct MockEngine {
        fate: Option<CornerFate>,
    }

    impl CrsEngine for MockEngine {
        fn get_transform_crs_to_crs(
            &self,
            _from: &str,
            _to: &str,
            _area_of_interest: Option<BoundingBox>,
            _options: &str,
        ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
            match self.fate {
                Some(fate) => Ok(Rc::new(MockTransform(fate))),
                None => Err(SedonaGeometryError::Invalid("no transform".to_string())),
            }
        }

        fn get_transform_pipeline(
            &self,
            _pipeline: &str,
            _options: &str,
        ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
            Err(SedonaGeometryError::Unknown)
        }

        fn to_projjson(&self, _crs_string: &str) -> Result<String, SedonaGeometryError> {
            Err(SedonaGeometryError::Unknown)
        }
    }

    /// A footprint reprojection that fails — the transform can't be built, a
    /// densified point's transform errors, or a reprojected point is non-finite —
    /// yields a null footprint with an empty MBR for that row (mirroring the
    /// null-raster path) rather than writing a garbage footprint/MBR or aborting
    /// the whole join. A transform that succeeds with finite points still produces
    /// a real, non-null footprint.
    #[test]
    fn failed_reprojection_yields_null_footprint() {
        let crs = lnglat();
        let crs = crs.as_deref().unwrap();
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

        for fate in [None, Some(CornerFate::Error), Some(CornerFate::NonFinite)] {
            let engine = MockEngine { fate };
            let mut ring = Vec::new();
            let mut builder = BinaryBuilder::new();
            let bounds =
                append_reprojected_footprint(corners, crs, crs, &engine, &mut ring, &mut builder)
                    .unwrap();
            let footprints = builder.finish();
            assert!(
                footprints.is_null(0),
                "a failed reprojection ({fate:?}) must produce a null footprint"
            );
            assert!(
                bounds.is_empty(),
                "a null footprint ({fate:?}) must have an empty MBR so it never matches"
            );
        }

        // A finite transform still produces a real footprint with a non-empty MBR.
        let engine = MockEngine {
            fate: Some(CornerFate::Identity),
        };
        let mut ring = Vec::new();
        let mut builder = BinaryBuilder::new();
        let bounds =
            append_reprojected_footprint(corners, crs, crs, &engine, &mut ring, &mut builder)
                .unwrap();
        let footprints = builder.finish();
        assert!(
            !footprints.is_null(0),
            "a finite reprojection must produce a real footprint"
        );
        assert!(!bounds.is_empty());
    }
}
