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
use std::borrow::Cow;
use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::mem::transmute;
use std::num::NonZeroUsize;
use std::rc::Rc;

use lru::LruCache;
use std::fmt::Debug;

use crate::bounding_box::BoundingBox;
use crate::error::SedonaGeometryError;
use crate::interval::IntervalTrait;
use crate::wkb_factory::{
    write_wkb_coord, write_wkb_empty_point, write_wkb_geometrycollection_header,
    write_wkb_linestring_header, write_wkb_multilinestring_header, write_wkb_multipoint_header,
    write_wkb_multipolygon_header, write_wkb_point_header, write_wkb_polygon_header,
    write_wkb_polygon_ring_header,
};
use geo_traits::{
    CoordTrait, Dimensions, GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};

/// Represents a coordinate reference system (CRS) transformation engine.
pub trait CrsEngine: Debug {
    fn get_transform_crs_to_crs(
        &self,
        from: &str,
        to: &str,
        area_of_interest: Option<BoundingBox>,
        options: &str,
    ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError>;
    fn get_transform_pipeline(
        &self,
        pipeline: &str,
        options: &str,
    ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError>;
}

/// Trait for transforming coordinates in a geometry from one CRS to another.
pub trait CrsTransform: std::fmt::Debug {
    fn transform_coord(
        &self,
        coord: &mut (f64, f64),
        input_dims: Dimensions,
    ) -> Result<(), SedonaGeometryError>;

    // CrsTransform can optionally handle 3D coordinates. If this method is not implemented,
    // the Z coordinate is simply ignored.
    fn transform_coord_3d(
        &self,
        coord: &mut (f64, f64, f64),
        input_dims: Dimensions,
    ) -> Result<(), SedonaGeometryError> {
        let mut coord_2d = (coord.0, coord.1);
        self.transform_coord(&mut coord_2d, input_dims)?;
        coord.0 = coord_2d.0;
        coord.1 = coord_2d.1;
        Ok(())
    }

    // The dimension of the output. If `None`, the input dimension is preserved.
    fn output_dim(&self) -> Option<Dimensions> {
        None
    }
}

/// A boxed trait object for dynamic dispatch of CRS transformations.
impl CrsTransform for Box<dyn CrsTransform> {
    fn transform_coord(
        &self,
        coord: &mut (f64, f64),
        input_dims: Dimensions,
    ) -> Result<(), SedonaGeometryError> {
        self.as_ref().transform_coord(coord, input_dims)
    }

    fn transform_coord_3d(
        &self,
        coord: &mut (f64, f64, f64),
        input_dims: Dimensions,
    ) -> Result<(), SedonaGeometryError> {
        self.as_ref().transform_coord_3d(coord, input_dims)
    }
}

/// A caching wrapper around any CRS transformation engine.
///
/// This provides automatic caching of coordinate transformation objects to improve performance
/// when the same transformations are used repeatedly. Uses LRU (Least Recently Used) eviction
/// policy when the cache reaches its capacity.
///
/// # Example
///
/// ```rust,ignore
/// use sedona_geometry::transform::{CachingCrsEngine, CrsEngine};
///
/// let engine = SomeCrsEngine::new();
/// let cached_engine = CachingCrsEngine::new(engine);
///
/// // Subsequent calls with the same parameters will use cached transforms
/// let transform1 = cached_engine.get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", None, "")?;
/// let transform2 = cached_engine.get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", None, "")?;
/// // transform2 is retrieved from cache
/// ```
#[derive(Debug)]
pub struct CachingCrsEngine<T: CrsEngine> {
    engine: T,
    crs_to_crs_cache: RefCell<LruCache<CrsToCrsCacheKey<'static>, Rc<dyn CrsTransform>>>,
    pipeline_cache: RefCell<LruCache<PipelineCacheKey<'static>, Rc<dyn CrsTransform>>>,
}

/// Cache key for CRS to CRS transforms
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct CrsToCrsCacheKey<'a> {
    from: Cow<'a, str>,
    to: Cow<'a, str>,
    area_of_interest: Option<SerializableBoundingBox>,
    options: Cow<'a, str>,
}

/// Cache key for pipeline transforms
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PipelineCacheKey<'a> {
    pipeline: Cow<'a, str>,
    options: Cow<'a, str>,
}

/// A serializable version of BoundingBox that implements Hash
#[derive(Clone, Debug, PartialEq, Eq)]
struct SerializableBoundingBox {
    x_lo: u64,
    x_hi: u64,
    y_lo: u64,
    y_hi: u64,
}

impl Hash for SerializableBoundingBox {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x_lo.hash(state);
        self.x_hi.hash(state);
        self.y_lo.hash(state);
        self.y_hi.hash(state);
    }
}

impl From<BoundingBox> for SerializableBoundingBox {
    fn from(bbox: BoundingBox) -> Self {
        Self {
            x_lo: bbox.x().lo().to_bits(),
            x_hi: bbox.x().hi().to_bits(),
            y_lo: bbox.y().lo().to_bits(),
            y_hi: bbox.y().hi().to_bits(),
        }
    }
}

/// Default cache size for transform objects
const DEFAULT_TRANSFORM_CACHE_SIZE: usize = 100;

impl<T: CrsEngine> CachingCrsEngine<T> {
    /// Creates a new caching engine wrapper with the default cache size.
    pub fn new(engine: T) -> Self {
        Self::with_cache_size(engine, DEFAULT_TRANSFORM_CACHE_SIZE)
    }

    /// Creates a new caching engine wrapper with a specified cache size.
    ///
    /// # Arguments
    ///
    /// * `engine` - The underlying CRS engine to wrap
    /// * `cache_size` - Maximum number of transforms to cache (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `cache_size` is 0.
    pub fn with_cache_size(engine: T, cache_size: usize) -> Self {
        let cache_size = NonZeroUsize::new(cache_size).unwrap();
        Self {
            engine,
            crs_to_crs_cache: RefCell::new(LruCache::new(cache_size)),
            pipeline_cache: RefCell::new(LruCache::new(cache_size)),
        }
    }
}

impl<T: CrsEngine> CrsEngine for CachingCrsEngine<T> {
    fn get_transform_crs_to_crs(
        &self,
        from: &str,
        to: &str,
        area_of_interest: Option<BoundingBox>,
        options: &str,
    ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
        let serializable_aoi = area_of_interest.as_ref().map(|bbox| bbox.clone().into());
        unsafe {
            // Safety: we know that the string references in cache key will only be ephemeral and won't be
            // stored inside `crs_to_crs_cache` or referenced by the CrsTransform object retrieved from the
            // cache.
            // We prefer transmute over messing around with the type system to stick to safe code. Here is
            // a more complicated but safe version:
            // https://idubrov.name/rust/2018/06/01/tricking-the-hashmap.html
            let from_static: &'static str = transmute(from);
            let to_static: &'static str = transmute(to);
            let options_static: &'static str = transmute(options);
            let cache_key = CrsToCrsCacheKey {
                from: Cow::Borrowed(from_static),
                to: Cow::Borrowed(to_static),
                area_of_interest: serializable_aoi.clone(),
                options: Cow::Borrowed(options_static),
            };
            // Check cache first
            if let Some(cached) = self.crs_to_crs_cache.borrow_mut().get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        // Not in cache, create via underlying engine
        let transform =
            self.engine
                .get_transform_crs_to_crs(from, to, area_of_interest, options)?;

        // Cache and return
        let static_cache_key = CrsToCrsCacheKey {
            from: from.to_string().into(),
            to: to.to_string().into(),
            area_of_interest: serializable_aoi,
            options: options.to_string().into(),
        };
        self.crs_to_crs_cache
            .borrow_mut()
            .put(static_cache_key, transform.clone());
        Ok(transform)
    }

    fn get_transform_pipeline(
        &self,
        pipeline: &str,
        options: &str,
    ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
        unsafe {
            // Safety: we know that the string references in cache key will only be ephemeral and won't be
            // stored inside `pipeline_cache` or referenced by the CrsTransform object retrieved from the
            // cache.
            // We prefer transmute over messing around with the type system to stick to safe code. Here is
            // a more complicated but safe version:
            // https://idubrov.name/rust/2018/06/01/tricking-the-hashmap.html
            let pipeline_static: &'static str = transmute(pipeline);
            let options_static: &'static str = transmute(options);
            let cache_key = PipelineCacheKey {
                pipeline: Cow::Borrowed(pipeline_static),
                options: Cow::Borrowed(options_static),
            };
            // Check cache first
            if let Some(cached) = self.pipeline_cache.borrow_mut().get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        // Not in cache, create via underlying engine
        let transform = self.engine.get_transform_pipeline(pipeline, options)?;

        // Cache and return
        let static_cache_key = PipelineCacheKey {
            pipeline: pipeline.to_string().into(),
            options: options.to_string().into(),
        };
        self.pipeline_cache
            .borrow_mut()
            .put(static_cache_key, transform.clone());
        Ok(transform)
    }
}

/// Transforms a geometry from one CRS to another using the provided transformation.
pub fn transform(
    geom: impl GeometryTrait<T = f64>,
    trans: &dyn CrsTransform,
    out: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    // If the CrsTransform specifies the dimension, use it.
    // Otherwise, the input dimension is preserved.
    let output_dims = trans.output_dim().unwrap_or_else(|| geom.dim());
    match geom.as_type() {
        GeometryType::Point(pt) => {
            if pt.coord().is_some() {
                write_wkb_point_header(out, output_dims)?;
                transform_and_write_coords(out, trans, pt.coord().into_iter(), output_dims)?;
            } else {
                write_wkb_empty_point(out, output_dims)?;
            }
        }
        GeometryType::LineString(ls) => {
            write_wkb_linestring_header(out, output_dims, ls.coords().count())?;
            transform_and_write_coords(out, trans, ls.coords(), output_dims)?;
        }
        GeometryType::Polygon(pl) => {
            let num_rings = pl.interiors().count() + pl.exterior().is_some() as usize;
            write_wkb_polygon_header(out, output_dims, num_rings)?;

            if let Some(exterior) = pl.exterior() {
                transform_and_write_ring(out, trans, exterior, output_dims)?;
            }

            for interior in pl.interiors() {
                transform_and_write_ring(out, trans, interior, output_dims)?;
            }
        }
        GeometryType::MultiPoint(multi_pt) => {
            write_wkb_multipoint_header(out, output_dims, multi_pt.points().count())?;
            for pt in multi_pt.points() {
                transform(pt, trans, out)?;
            }
        }
        GeometryType::MultiLineString(multi_ls) => {
            write_wkb_multilinestring_header(out, output_dims, multi_ls.line_strings().count())?;
            for ls in multi_ls.line_strings() {
                transform(ls, trans, out)?;
            }
        }
        GeometryType::MultiPolygon(multi_pl) => {
            write_wkb_multipolygon_header(out, output_dims, multi_pl.polygons().count())?;
            for pl in multi_pl.polygons() {
                transform(pl, trans, out)?;
            }
        }
        GeometryType::GeometryCollection(collection) => {
            write_wkb_geometrycollection_header(out, output_dims, collection.geometries().count())?;
            for geom in collection.geometries() {
                transform(geom, trans, out)?;
            }
        }
        _ => {
            return Err(SedonaGeometryError::Invalid(
                "GeometryType not supported for transform".to_string(),
            ))
        }
    }

    Ok(())
}

fn transform_and_write_ring<'a, L>(
    buf: &mut impl Write,
    trans: &dyn CrsTransform,
    ring: L,
    output_dims: Dimensions,
) -> Result<(), SedonaGeometryError>
where
    L: LineStringTrait<T = f64> + 'a,
{
    let num_points = ring.coords().count();
    write_wkb_polygon_ring_header(buf, num_points)?;
    transform_and_write_coords(buf, trans, ring.coords(), output_dims)?;
    Ok(())
}

fn transform_and_write_coords<'a, C, I>(
    buf: &mut impl Write,
    trans: &dyn CrsTransform,
    coords: I,
    output_dims: Dimensions,
) -> Result<(), SedonaGeometryError>
where
    C: CoordTrait<T = f64> + 'a,
    I: Iterator<Item = C>,
{
    for coord in coords {
        let input_dims = coord.dim();
        match output_dims {
            Dimensions::Xy => {
                let mut xy: (f64, f64) = (coord.x(), coord.y());
                trans.transform_coord(&mut xy, input_dims)?;
                write_wkb_coord(buf, (xy.0, xy.1))?;
            }
            Dimensions::Xyz => {
                let mut xyz = fill_or_extract_coord(&coord, input_dims);
                trans.transform_coord_3d(&mut xyz, input_dims)?;
                write_wkb_coord(buf, (xyz.0, xyz.1, xyz.2))?;
            }
            Dimensions::Xym => {
                let mut xy: (f64, f64) = (coord.x(), coord.y());
                trans.transform_coord(&mut xy, input_dims)?;
                write_wkb_coord(buf, (xy.0, xy.1, coord.nth_or_panic(2)))?;
            }
            Dimensions::Xyzm => {
                let mut xyz = fill_or_extract_coord(&coord, input_dims);
                trans.transform_coord_3d(&mut xyz, input_dims)?;
                write_wkb_coord(buf, (xyz.0, xyz.1, xyz.2, coord.nth_or_panic(3)))?;
            }
            _ => {
                return Err(SedonaGeometryError::Invalid(
                    "Unsupported dimensions for coordinate transformation".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn fill_or_extract_coord<C>(coord: &C, input_dims: Dimensions) -> (f64, f64, f64)
where
    C: CoordTrait<T = f64>,
{
    match input_dims {
        // If the input doesn't have Z coordinate, fill with 0.
        Dimensions::Xy | Dimensions::Xym | Dimensions::Unknown(_) => (coord.x(), coord.y(), 0.0),
        Dimensions::Xyz | Dimensions::Xyzm => (coord.x(), coord.y(), coord.nth_or_panic(2)),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::str::FromStr;
    use wkb::reader::read_wkb;
    use wkt::Wkt;

    #[derive(Debug)]
    struct MockTransform {}
    impl CrsTransform for MockTransform {
        fn transform_coord(
            &self,
            coord: &mut (f64, f64),
            _input_dims: Dimensions,
        ) -> Result<(), SedonaGeometryError> {
            coord.0 += 10.0;
            coord.1 += 20.0;
            Ok(())
        }
    }

    #[derive(Debug)]
    struct Mock3DTransform {}
    impl CrsTransform for Mock3DTransform {
        // This transforms 2D and 3D differently for testing purposes
        fn transform_coord(
            &self,
            coord: &mut (f64, f64),
            _input_dims: Dimensions,
        ) -> Result<(), SedonaGeometryError> {
            coord.0 += 100.0;
            coord.1 += 200.0;
            Ok(())
        }

        fn transform_coord_3d(
            &self,
            coord: &mut (f64, f64, f64),
            _input_dims: Dimensions,
        ) -> Result<(), SedonaGeometryError> {
            coord.0 += 10.0;
            coord.1 += 20.0;
            coord.2 += 30.0;
            Ok(())
        }
    }

    fn test_transform_inner(
        geom: impl GeometryTrait<T = f64>,
        expected: &str,
        mock_transform: impl CrsTransform,
    ) {
        let mut wkb_bytes = Vec::new();

        transform(geom, &mock_transform, &mut wkb_bytes).unwrap();
        let wkb_reader = read_wkb(&wkb_bytes).unwrap();
        let mut wkt = String::new();
        wkt::to_wkt::write_geometry(&mut wkt, &wkb_reader).unwrap();
        assert_eq!(wkt, expected);
    }

    fn test_transform(geom: impl GeometryTrait<T = f64>, expected: &str) {
        test_transform_inner(geom, expected, MockTransform {})
    }

    fn test_transform_3d(geom: impl GeometryTrait<T = f64>, expected: &str) {
        test_transform_inner(geom, expected, Mock3DTransform {})
    }

    #[test]
    fn test_transform_point() {
        let point = geo_types::Point::new(1.0, 2.0);
        test_transform(point, "POINT(11 22)");

        let nan_point = geo_types::Point::new(f64::NAN, f64::NAN);
        test_transform(nan_point, "POINT EMPTY");
    }

    #[test]
    fn test_transform_linestring() {
        let linestring_xy = geo_types::LineString::from(vec![(1.0, 2.0), (3.0, 4.0)]);
        test_transform(linestring_xy, "LINESTRING(11 22,13 24)");

        let empty_linestring = geo_types::LineString::new(vec![]);
        test_transform(empty_linestring, "LINESTRING EMPTY");
    }

    #[test]
    fn test_transform_polygon() {
        let polygon = geo_types::Polygon::new(
            geo_types::LineString::from(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]),
            vec![],
        );
        test_transform(polygon, "POLYGON((11 22,13 24,15 26,17 28,11 22))");

        let polygon_multi_rings = geo_types::Polygon::new(
            geo_types::LineString::from(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]),
            vec![geo_types::LineString::from(vec![
                (9.0, 10.0),
                (11.0, 12.0),
                (13.0, 14.0),
                (15.0, 16.0),
            ])],
        );
        test_transform(
            polygon_multi_rings,
            "POLYGON((11 22,13 24,15 26,17 28,11 22),(19 30,21 32,23 34,25 36,19 30))",
        );

        let empty_polygon = geo_types::Polygon::new(geo_types::LineString::new(vec![]), vec![]);
        test_transform(empty_polygon, "POLYGON EMPTY");
    }

    #[test]
    fn test_transform_multipoint() {
        let multipoint = geo_types::MultiPoint::from(vec![
            geo_types::Point::new(1.0, 2.0),
            geo_types::Point::new(3.0, 4.0),
        ]);
        test_transform(multipoint, "MULTIPOINT((11 22),(13 24))");

        let empty_multipoint = geo_types::MultiPoint::new(vec![]);
        test_transform(empty_multipoint, "MULTIPOINT EMPTY");
    }

    #[test]
    fn test_transform_multilinestring() {
        let multilinestring = geo_types::MultiLineString(vec![
            geo_types::LineString::from(vec![(1.0, 2.0), (3.0, 4.0)]),
            geo_types::LineString::from(vec![(5.0, 6.0), (7.0, 8.0)]),
        ]);
        test_transform(
            multilinestring,
            "MULTILINESTRING((11 22,13 24),(15 26,17 28))",
        );

        let empty_multilinestring = geo_types::MultiLineString::new(vec![]);
        test_transform(empty_multilinestring, "MULTILINESTRING EMPTY");
    }

    #[test]
    fn test_transform_multipolygon() {
        let multipolygon = geo_types::MultiPolygon(vec![
            geo_types::Polygon::new(
                geo_types::LineString::from(vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]),
                vec![],
            ),
            geo_types::Polygon::new(
                geo_types::LineString::from(vec![
                    (9.0, 10.0),
                    (11.0, 12.0),
                    (13.0, 14.0),
                    (15.0, 16.0),
                ]),
                vec![],
            ),
        ]);
        test_transform(
            multipolygon,
            "MULTIPOLYGON(((11 22,13 24,15 26,17 28,11 22)),((19 30,21 32,23 34,25 36,19 30)))",
        );

        let empty_multipolygon = geo_types::MultiPolygon::new(vec![]);
        test_transform(empty_multipolygon, "MULTIPOLYGON EMPTY");
    }

    #[test]
    fn test_transform_geometrycollection() {
        let geometry_collection = geo_types::GeometryCollection::from(vec![
            geo_types::Geometry::Point(geo_types::Point::new(1.0, 2.0)),
            geo_types::Geometry::LineString(geo_types::LineString::from(vec![
                (3.0, 4.0),
                (5.0, 6.0),
            ])),
        ]);
        test_transform(
            geometry_collection,
            "GEOMETRYCOLLECTION(POINT(11 22),LINESTRING(13 24,15 26))",
        );

        let empty_collection = geo_types::GeometryCollection::new_from(vec![]);
        test_transform(empty_collection, "GEOMETRYCOLLECTION EMPTY");
    }

    #[test]
    fn test_transform_dimensions() {
        let ls_xy_wkt = "LINESTRING(1.0 2.0, 3.0 4.0)";
        let ls_xy: Wkt = Wkt::from_str(ls_xy_wkt).unwrap();
        test_transform(ls_xy, "LINESTRING(11 22,13 24)");

        let ls_xyz_wkt = "LINESTRING Z(1.0 2.0 3.0, 4.0 5.0 6.0)";
        let ls_xyz: Wkt = Wkt::from_str(ls_xyz_wkt).unwrap();
        test_transform(ls_xyz, "LINESTRING Z(11 22 3,14 25 6)");

        let ls_xym_wkt = "LINESTRING M(1.0 2.0 3.0, 4.0 5.0 6.0)";
        let ls_xym: Wkt = Wkt::from_str(ls_xym_wkt).unwrap();
        test_transform(ls_xym, "LINESTRING M(11 22 3,14 25 6)");

        let ls_xyzm_wkt = "LINESTRING ZM(1.0 2.0 3.0 4.0, 5.0 6.0 7.0 8.0)";
        let ls_xyzm: Wkt = Wkt::from_str(ls_xyzm_wkt).unwrap();
        test_transform(ls_xyzm, "LINESTRING ZM(11 22 3 4,15 26 7 8)");
    }

    #[test]
    fn test_transform_point_3d() {
        let point = wkt::Wkt::from_str("POINT Z(1 2 3)").unwrap();
        test_transform_3d(point, "POINT Z(11 22 33)");

        let nan_point = wkt::Wkt::from_str("POINT Z EMPTY").unwrap();
        test_transform_3d(nan_point, "POINT Z EMPTY");
    }

    #[test]
    fn test_transform_dimensions_3d() {
        let ls_xy_wkt = "LINESTRING(1.0 2.0, 3.0 4.0)";
        let ls_xy: Wkt = Wkt::from_str(ls_xy_wkt).unwrap();
        test_transform_3d(ls_xy, "LINESTRING(101 202,103 204)");

        let ls_xyz_wkt = "LINESTRING Z(1.0 2.0 3.0, 4.0 5.0 6.0)";
        let ls_xyz: Wkt = Wkt::from_str(ls_xyz_wkt).unwrap();
        test_transform_3d(ls_xyz, "LINESTRING Z(11 22 33,14 25 36)");

        let ls_xym_wkt = "LINESTRING M(1.0 2.0 3.0, 4.0 5.0 6.0)";
        let ls_xym: Wkt = Wkt::from_str(ls_xym_wkt).unwrap();
        test_transform_3d(ls_xym, "LINESTRING M(101 202 3,104 205 6)");

        let ls_xyzm_wkt = "LINESTRING ZM(1.0 2.0 3.0 4.0, 5.0 6.0 7.0 8.0)";
        let ls_xyzm: Wkt = Wkt::from_str(ls_xyzm_wkt).unwrap();
        test_transform_3d(ls_xyzm, "LINESTRING ZM(11 22 33 4,15 26 37 8)");
    }

    /// Mock CRS engine for testing caching behavior
    #[derive(Debug)]
    struct MockCrsEngine {
        crs_to_crs_call_count: RefCell<usize>,
        pipeline_call_count: RefCell<usize>,
    }

    impl MockCrsEngine {
        fn new() -> Self {
            Self {
                crs_to_crs_call_count: RefCell::new(0),
                pipeline_call_count: RefCell::new(0),
            }
        }

        fn crs_to_crs_calls(&self) -> usize {
            *self.crs_to_crs_call_count.borrow()
        }

        fn pipeline_calls(&self) -> usize {
            *self.pipeline_call_count.borrow()
        }
    }

    impl CrsEngine for MockCrsEngine {
        fn get_transform_crs_to_crs(
            &self,
            _from: &str,
            _to: &str,
            _area_of_interest: Option<BoundingBox>,
            _options: &str,
        ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
            *self.crs_to_crs_call_count.borrow_mut() += 1;
            Ok(Rc::new(MockTransform {}))
        }

        fn get_transform_pipeline(
            &self,
            _pipeline: &str,
            _options: &str,
        ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
            *self.pipeline_call_count.borrow_mut() += 1;
            Ok(Rc::new(MockTransform {}))
        }
    }

    #[test]
    fn test_caching_crs_engine_crs_to_crs_basic_caching() {
        let mock_engine = MockCrsEngine::new();
        let caching_engine = CachingCrsEngine::new(mock_engine);

        // First call should create a new transform
        let transform1 = caching_engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", None, "")
            .unwrap();
        assert_eq!(caching_engine.engine.crs_to_crs_calls(), 1);

        // Second call with same parameters should use cache
        let transform2 = caching_engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", None, "")
            .unwrap();
        assert_eq!(caching_engine.engine.crs_to_crs_calls(), 1); // Still 1

        // Should be the same object
        assert!(Rc::ptr_eq(&transform1, &transform2));
    }

    #[test]
    fn test_caching_crs_engine_crs_to_crs_with_aoi() {
        let mock_engine = MockCrsEngine::new();
        let caching_engine = CachingCrsEngine::new(mock_engine);

        // First call should create a new transform
        let aoi = BoundingBox::xy((1.0, 2.0), (3.0, 4.0));
        let transform1 = caching_engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", Some(aoi.clone()), "")
            .unwrap();

        // Second call with same parameters should use cache
        let transform2 = caching_engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", Some(aoi), "")
            .unwrap();
        assert_eq!(caching_engine.engine.crs_to_crs_calls(), 1); // Still 1

        // Should be the same object
        assert!(Rc::ptr_eq(&transform1, &transform2));

        // Third call with a different aoi should not use cache
        let aoi2 = BoundingBox::xy((1.0, 2.0), (3.0, 40.0));
        let transform3 = caching_engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", Some(aoi2), "")
            .unwrap();
        assert_eq!(caching_engine.engine.crs_to_crs_calls(), 2); // Now 2

        // Should be a different object
        assert!(!Rc::ptr_eq(&transform1, &transform3));
    }

    #[test]
    fn test_caching_crs_engine_crs_to_crs_different_params() {
        let mock_engine = MockCrsEngine::new();
        let caching_engine = CachingCrsEngine::new(mock_engine);

        // Different from CRS
        let _transform1 = caching_engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", None, "")
            .unwrap();
        let _transform2 = caching_engine
            .get_transform_crs_to_crs("EPSG:2154", "EPSG:3857", None, "")
            .unwrap();
        assert_eq!(caching_engine.engine.crs_to_crs_calls(), 2);

        // Different to CRS
        let _transform3 = caching_engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:2154", None, "")
            .unwrap();
        assert_eq!(caching_engine.engine.crs_to_crs_calls(), 3);

        // Different options
        let _transform4 = caching_engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", None, "+proj=utm")
            .unwrap();
        assert_eq!(caching_engine.engine.crs_to_crs_calls(), 4);

        // With area of interest
        let aoi = BoundingBox::xy((1.0, 2.0), (3.0, 4.0));
        let _transform5 = caching_engine
            .get_transform_crs_to_crs("EPSG:4326", "EPSG:3857", Some(aoi), "")
            .unwrap();
        assert_eq!(caching_engine.engine.crs_to_crs_calls(), 5);
    }

    #[test]
    fn test_caching_crs_engine_pipeline_basic_caching() {
        let mock_engine = MockCrsEngine::new();
        let caching_engine = CachingCrsEngine::new(mock_engine);

        // First call should create a new transform
        let transform1 = caching_engine
            .get_transform_pipeline("+proj=utm +zone=33 +datum=WGS84", "")
            .unwrap();
        assert_eq!(caching_engine.engine.pipeline_calls(), 1);

        // Second call with same parameters should use cache
        let transform2 = caching_engine
            .get_transform_pipeline("+proj=utm +zone=33 +datum=WGS84", "")
            .unwrap();
        assert_eq!(caching_engine.engine.pipeline_calls(), 1); // Still 1

        // Should be the same object
        assert!(Rc::ptr_eq(&transform1, &transform2));
    }

    #[test]
    fn test_caching_crs_engine_pipeline_different_params() {
        let mock_engine = MockCrsEngine::new();
        let caching_engine = CachingCrsEngine::new(mock_engine);

        // Different pipeline
        let _transform1 = caching_engine
            .get_transform_pipeline("+proj=utm +zone=33 +datum=WGS84", "")
            .unwrap();
        let _transform2 = caching_engine
            .get_transform_pipeline("+proj=utm +zone=34 +datum=WGS84", "")
            .unwrap();
        assert_eq!(caching_engine.engine.pipeline_calls(), 2);

        // Different options
        let _transform3 = caching_engine
            .get_transform_pipeline("+proj=utm +zone=33 +datum=WGS84", "+over")
            .unwrap();
        assert_eq!(caching_engine.engine.pipeline_calls(), 3);
    }
}
