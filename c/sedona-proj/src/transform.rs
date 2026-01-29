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
use crate::error::SedonaProjError;
use crate::proj::{Proj, ProjContext};
use sedona_geometry::bounding_box::BoundingBox;
use sedona_geometry::error::SedonaGeometryError;
use sedona_geometry::interval::IntervalTrait;
use sedona_geometry::transform::{CrsEngine, CrsTransform};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

/// Builder for a [ProjCrsEngine]
///
/// API for specifying various engine parameters. More parameters may
/// be added.
#[derive(Default)]
pub struct ProjCrsEngineBuilder {
    shared_library: Option<PathBuf>,
    database_path: Option<PathBuf>,
    search_paths: Option<Vec<PathBuf>>,
    log_level: Option<u32>,
}

impl ProjCrsEngineBuilder {
    /// Set the library source from whence PROJ should be loaded
    ///
    /// If unset, proj_sys will be attempted. This is required to be set
    /// for applications or bindings that build sedona-proj without the
    /// proj-sys feature.
    ///
    /// Note that path will be passed directly to dlopen()/LoadLibrary(),
    /// which takes into account the working directory. As a security
    /// measure, applications may wish to check that path is an absolute
    /// path and/or forbid the use of this mechanism. This should not
    /// be specified from untrusted input.
    pub fn with_shared_library(self, path: PathBuf) -> Self {
        Self {
            shared_library: Some(path),
            ..self
        }
    }

    /// Set the path to the PROJ database
    ///
    /// Set the path to proj.db (including the filename, which is normally
    /// proj.db). This file is specific to the PROJ version and it is usually
    /// required when using a shared PROJ at a non-system location.
    pub fn with_database_path(self, database_path: PathBuf) -> Self {
        Self {
            database_path: Some(database_path),
            ..self
        }
    }

    /// Set a vector of paths to search for PROJ data files
    ///
    /// PROJ data files include grids and other types of resources. A small set of these
    /// are normally installed alongside a PROJ build; however, a much larger set can be
    /// installed separately or downloaded dynamically using PROJ's network capability.
    /// These files are typically (but not always) tied to the PROJ version and it is
    /// usually a good idea to specify this explicitly when setting a specific shared
    /// PROJ at a non-system location.
    pub fn with_search_paths(self, search_paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths: Some(search_paths),
            ..self
        }
    }

    /// Set the PROJ log level
    ///
    /// Set the verbosity of PROJ logging. The default is no logging,
    /// however errors will still be propagated through the error
    /// handling.
    ///
    /// Log level constants are defined in proj_sys:
    /// - PJ_LOG_LEVEL_PJ_LOG_NONE (0): No logging
    /// - PJ_LOG_LEVEL_PJ_LOG_ERROR (1): Error messages
    /// - PJ_LOG_LEVEL_PJ_LOG_DEBUG (2): Debug messages
    /// - PJ_LOG_LEVEL_PJ_LOG_TRACE (3): Trace
    /// - PJ_LOG_LEVEL_PJ_LOG_TELL (4): Tell
    pub fn with_log_level(self, log_level: u32) -> Self {
        Self {
            log_level: Some(log_level),
            ..self
        }
    }

    /// Build a [ProjCrsEngine] with the specified options
    pub fn build(&self) -> Result<ProjCrsEngine, SedonaProjError> {
        let mut ctx = if let Some(shared_library) = self.shared_library.clone() {
            ProjContext::try_from_shared_library(shared_library)?
        } else {
            ProjContext::try_from_proj_sys()?
        };

        if let Some(database_path) = &self.database_path {
            ctx.set_database_path(database_path.to_string_lossy().as_ref())?;
        }

        if let Some(search_paths) = &self.search_paths {
            let string_vec = search_paths
                .iter()
                .map(|path| path.to_string_lossy().to_string())
                .collect::<Vec<_>>();
            ctx.set_search_paths(&string_vec)?;
        }

        if let Some(log_level) = &self.log_level {
            ctx.set_log_level(*log_level)?;
        } else {
            // Default log level to none
            ctx.set_log_level(0)?;
        }

        Ok(ProjCrsEngine { ctx: Rc::new(ctx) })
    }
}

/// A [CrsEngine] implemented using PROJ
///
/// Use the [ProjCrsEngineBuilder] to create this object.
#[derive(Debug)]
pub struct ProjCrsEngine {
    ctx: Rc<ProjContext>,
}

impl CrsEngine for ProjCrsEngine {
    fn get_transform_crs_to_crs(
        &self,
        from: &str,
        to: &str,
        area_of_interest: Option<BoundingBox>,
        options: &str,
    ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
        if !options.is_empty() {
            return Err(SedonaGeometryError::Invalid(
                "Options for area of use not supported yet".to_string(),
            ));
        }

        let area = area_of_interest.map(|bbox| {
            (
                bbox.x().lo(), // west
                bbox.y().lo(), // south
                bbox.x().hi(), // east
                bbox.y().hi(), // north
            )
        });

        let source_crs = Proj::try_new(self.ctx.clone(), from).map_err(|e| {
            SedonaGeometryError::Invalid(format!("Failed to create CRS from source '{from}': {e}"))
        })?;
        let target_crs = Proj::try_new(self.ctx.clone(), to).map_err(|e| {
            SedonaGeometryError::Invalid(format!(
                "Failed to create CRS from destination '{to}': {e}"
            ))
        })?;
        let proj = Proj::try_crs_to_crs(self.ctx.clone(), &source_crs, &target_crs, area).map_err(
            |e| {
                SedonaGeometryError::Invalid(format!(
                    "Failed to create transform from '{from}' to '{to}': {e}"
                ))
            },
        )?;

        Ok(Rc::new(ProjTransform::new(proj)))
    }

    fn get_transform_pipeline(
        &self,
        pipeline: &str,
        options: &str,
    ) -> Result<Rc<dyn CrsTransform>, SedonaGeometryError> {
        if !options.is_empty() {
            return Err(SedonaGeometryError::Invalid(
                "Options for transform not supported yet".to_string(),
            ));
        }

        let transform = Proj::try_new(self.ctx.clone(), pipeline).map_err(|e| {
            SedonaGeometryError::Invalid(format!(
                "Failed to create pipeline transform from source '{pipeline}': {e}"
            ))
        })?;

        Ok(Rc::new(ProjTransform::new(transform)))
    }
}

/// A [CrsTransform] implemented using the [proj] crate
#[derive(Debug)]
pub struct ProjTransform {
    proj: RefCell<Proj>,
}

impl ProjTransform {
    fn new(proj: Proj) -> Self {
        Self {
            proj: RefCell::new(proj),
        }
    }
}

impl CrsTransform for ProjTransform {
    fn transform_coord(&self, coord: &mut (f64, f64)) -> Result<(), SedonaGeometryError> {
        let res = self.proj.borrow_mut().transform_xy(*coord).map_err(|e| {
            SedonaGeometryError::Invalid(format!(
                "PROJ coordinate transformation failed with error: {e}"
            ))
        })?;
        coord.0 = res.0;
        coord.1 = res.1;
        Ok(())
    }

    fn transform_coord_3d(&self, coord: &mut (f64, f64, f64)) -> Result<(), SedonaGeometryError> {
        let res = self.proj.borrow_mut().transform_xyz(*coord).map_err(|e| {
            SedonaGeometryError::Invalid(format!(
                "PROJ coordinate transformation failed with error: {e}"
            ))
        })?;
        coord.0 = res.0;
        coord.1 = res.1;
        coord.2 = res.2;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    use geo_traits::{CoordTrait, GeometryTrait, GeometryType, PointTrait};
    use geo_types::Point;
    use sedona_geometry::transform::transform;
    use wkb::reader::read_wkb;

    #[test]
    fn proj_crs_to_crs() {
        let engine = ProjCrsEngineBuilder::default().build().unwrap();
        let trans = engine
            .get_transform_crs_to_crs("EPSG:2230", "EPSG:26946", None, "")
            .unwrap();
        test_2230_to_26946(trans.as_ref());

        // use a different transformation where the AOI is valid
        let aoi = BoundingBox::xy((37, 38), (-122, -120));
        let trans_with_aoi = engine
            .get_transform_crs_to_crs("EPSG:2230", "EPSG:4326", Some(aoi), "")
            .unwrap();
        test_4269_to_4326(trans_with_aoi.as_ref());

        let trans_error = engine
            .get_transform_crs_to_crs("foo", "bar", None, "")
            .unwrap_err();
        assert_eq!(
            trans_error.to_string(),
            "Failed to create CRS from source 'foo': Invalid PROJ string syntax"
        );
    }

    #[test]
    fn proj_pipeline() {
        let engine = ProjCrsEngineBuilder::default().build().unwrap();
        let trans = engine
            .get_transform_pipeline(
                "+proj=pipeline +step +inv +proj=lcc +lat_1=33.88333333333333 +lat_2=32.78333333333333 \
                 +lat_0=32.16666666666666 +lon_0=-116.25 +x_0=2000000.0001016 +y_0=500000.0001016001 \
                 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=us-ft +no_defs \
                 +step +proj=lcc +lat_1=33.88333333333333 +lat_2=32.78333333333333 \
                 +lat_0=32.16666666666666 +lon_0=-116.25 +x_0=2000000 +y_0=500000 \
                 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs",
                "",
            )
            .unwrap();
        test_2230_to_26946(trans.as_ref());

        let trans_error = engine.get_transform_pipeline("foo", "");
        assert!(
            trans_error.is_err(),
            "Expected an error for invalid pipeline"
        );
    }

    #[test]
    fn transform_coord() {
        let engine = ProjCrsEngineBuilder::default().build().unwrap();
        let trans = engine
            .get_transform_crs_to_crs("EPSG:2230", "EPSG:26946", None, "")
            .unwrap();

        let mut coord = (4_760_096.4, 3_744_293.5);
        trans.as_ref().transform_coord(&mut coord).unwrap();
        assert_relative_eq!(coord.x(), 1_450_880.284_378, epsilon = 1e-6);
        assert_relative_eq!(coord.y(), 1_141_262.941_224, epsilon = 1e-6);

        coord = (f64::NAN, f64::NAN);
        trans.as_ref().transform_coord(&mut coord).unwrap();
        assert!(
            coord.x().is_nan() && coord.y().is_nan(),
            "Expected NaN coordinates"
        );
    }

    fn test_2230_to_26946(trans: &dyn CrsTransform) {
        let point = Point::new(4_760_096.421_921, 3_744_293.729_449);
        let mut wkb_bytes = Vec::new();
        transform(point, trans, &mut wkb_bytes).unwrap();

        let wkb_reader = read_wkb(&wkb_bytes).unwrap();
        assert!(
            matches!(wkb_reader.as_type(), GeometryType::Point(_)),
            "Expected a Point geometry type"
        );
        if let GeometryType::Point(point) = wkb_reader.as_type() {
            let x = point.coord().unwrap().x();
            let y = point.coord().unwrap().y();
            // expected values are from proj's example page
            assert_relative_eq!(x, 1_450_880.291_060_5, epsilon = 1e-6);
            assert_relative_eq!(y, 1_141_263.011_160_45, epsilon = 1e-6);
        }
    }
    fn test_4269_to_4326(trans: &dyn CrsTransform) {
        let point = Point::new(4760096.0, 5044293.0);
        let mut wkb_bytes = Vec::new();
        transform(point, trans, &mut wkb_bytes).unwrap();

        let wkb_reader = read_wkb(&wkb_bytes).unwrap();
        assert!(
            matches!(wkb_reader.as_type(), GeometryType::Point(_)),
            "Expected a Point geometry type"
        );
        if let GeometryType::Point(point) = wkb_reader.as_type() {
            let x = point.coord().unwrap().x();
            let y = point.coord().unwrap().y();
            assert_relative_eq!(x, -122.748_786_5, epsilon = 1e-6);
            assert_relative_eq!(y, 41.335_173_4, epsilon = 1e-6);
        }
    }
}
