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
use geo_traits::{
    CoordTrait, Dimensions, GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};

use crate::{
    bounding_box::BoundingBox,
    error::SedonaGeometryError,
    interval::{Interval, IntervalTrait},
};

/// Calculate the Cartesian XY bounds of a well-known binary geometry blob
///
/// Note that this bounder ignores Z or M coordinates that may or may not be present
/// for applications where only the XY bounding box is needed.
pub fn wkb_bounds_xy(wkb_value: &[u8]) -> Result<BoundingBox, SedonaGeometryError> {
    let wkb =
        wkb::reader::read_wkb(wkb_value).map_err(|e| SedonaGeometryError::External(Box::new(e)))?;
    geo_traits_bounds_xy(wkb)
}

/// Calculate the Cartesian XY bounds of a geometry
///
/// Note that this bounder ignores Z or M coordinates that may or may not be present
/// for applications where only the XY bounding box is needed.
pub fn geo_traits_bounds_xy(
    geom: impl GeometryTrait<T = f64>,
) -> Result<BoundingBox, SedonaGeometryError> {
    let mut x = Interval::empty();
    let mut y = Interval::empty();
    geo_traits_update_xy_bounds(geom, &mut x, &mut y)?;
    Ok(BoundingBox::xy(x, y))
}

/// Calculate the Z value interval of a geometry
pub fn geo_traits_bounds_z(
    geom: impl GeometryTrait<T = f64>,
) -> Result<Interval, SedonaGeometryError> {
    let mut z = Interval::empty();
    geo_traits_update_dimension_bounds(geom, &mut z, "z")?;
    Ok(z)
}

/// Calculate the M value interval of a geometry
pub fn geo_traits_bounds_m(
    geom: impl GeometryTrait<T = f64>,
) -> Result<Interval, SedonaGeometryError> {
    let mut m = Interval::empty();
    geo_traits_update_dimension_bounds(geom, &mut m, "m")?;
    Ok(m)
}

/// Update a pair of intervals for x and y bounds
///
/// Useful for updating bounds in-place when accumulating
/// bounds for statistics or function implementations.
pub fn geo_traits_update_xy_bounds(
    geom: impl GeometryTrait<T = f64>,
    x: &mut Interval,
    y: &mut Interval,
) -> Result<(), SedonaGeometryError> {
    match geom.as_type() {
        GeometryType::Point(pt) => {
            if let Some(coord) = PointTrait::coord(pt) {
                x.update_value(coord.x());
                y.update_value(coord.y());
            }
        }
        GeometryType::LineString(ls) => {
            for coord in ls.coords() {
                x.update_value(coord.x());
                y.update_value(coord.y());
            }
        }
        GeometryType::Polygon(pl) => {
            if let Some(exterior) = pl.exterior() {
                for coord in exterior.coords() {
                    x.update_value(coord.x());
                    y.update_value(coord.y());
                }
            }

            for interior in pl.interiors() {
                for coord in interior.coords() {
                    x.update_value(coord.x());
                    y.update_value(coord.y());
                }
            }
        }
        GeometryType::MultiPoint(multi_pt) => {
            for pt in multi_pt.points() {
                geo_traits_update_xy_bounds(pt, x, y)?;
            }
        }
        GeometryType::MultiLineString(multi_ls) => {
            for ls in multi_ls.line_strings() {
                geo_traits_update_xy_bounds(ls, x, y)?;
            }
        }
        GeometryType::MultiPolygon(multi_pl) => {
            for pl in multi_pl.polygons() {
                geo_traits_update_xy_bounds(pl, x, y)?;
            }
        }
        GeometryType::GeometryCollection(collection) => {
            for geom in collection.geometries() {
                geo_traits_update_xy_bounds(geom, x, y)?;
            }
        }
        _ => {
            return Err(SedonaGeometryError::Invalid(
                "GeometryType not supported for XY bounds".to_string(),
            ))
        }
    }

    Ok(())
}

/// Update a single interval for bounds of a particular dimension
///
/// Useful for updating bounds when only a single dimension is required.
/// target must be either "x", "y", "z", or "m"
pub fn geo_traits_update_dimension_bounds(
    geom: impl GeometryTrait<T = f64>,
    interval: &mut Interval,
    target: &str,
) -> Result<(), SedonaGeometryError> {
    let n = if let Some(n) = dimension_index(geom.dim(), target) {
        n
    } else {
        return Ok(());
    };

    match geom.as_type() {
        GeometryType::Point(pt) => {
            if let Some(coord) = PointTrait::coord(pt) {
                interval.update_value(unsafe { coord.nth_unchecked(n) });
            }
        }
        GeometryType::LineString(ls) => {
            for coord in ls.coords() {
                interval.update_value(unsafe { coord.nth_unchecked(n) });
            }
        }
        GeometryType::Polygon(pl) => {
            if let Some(exterior) = pl.exterior() {
                for coord in exterior.coords() {
                    interval.update_value(unsafe { coord.nth_unchecked(n) });
                }
            }

            for interior in pl.interiors() {
                for coord in interior.coords() {
                    interval.update_value(unsafe { coord.nth_unchecked(n) });
                }
            }
        }
        GeometryType::MultiPoint(multi_pt) => {
            for pt in multi_pt.points() {
                geo_traits_update_dimension_bounds(pt, interval, target)?;
            }
        }
        GeometryType::MultiLineString(multi_ls) => {
            for ls in multi_ls.line_strings() {
                geo_traits_update_dimension_bounds(ls, interval, target)?;
            }
        }
        GeometryType::MultiPolygon(multi_pl) => {
            for pl in multi_pl.polygons() {
                geo_traits_update_dimension_bounds(pl, interval, target)?;
            }
        }
        GeometryType::GeometryCollection(collection) => {
            for geom in collection.geometries() {
                geo_traits_update_dimension_bounds(geom, interval, target)?;
            }
        }
        _ => {
            return Err(SedonaGeometryError::Invalid(
                "GeometryType not supported for dimension bounds".to_string(),
            ))
        }
    }

    Ok(())
}

fn dimension_index(dim: Dimensions, target: &str) -> Option<usize> {
    match target {
        "x" => return Some(0),
        "y" => return Some(1),
        _ => {}
    }

    match (dim, target) {
        (geo_traits::Dimensions::Xyz, "z") => Some(2),
        (geo_traits::Dimensions::Xym, "m") => Some(2),
        (geo_traits::Dimensions::Xyzm, "z") => Some(2),
        (geo_traits::Dimensions::Xyzm, "m") => Some(3),
        (_, _) => None,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;
    use std::{iter::zip, str::FromStr};
    use wkb::{writer::WriteOptions, Endianness};
    use wkt::Wkt;

    pub fn wkt_bounds_xy(wkt_value: &str) -> Result<BoundingBox, SedonaGeometryError> {
        let wkt: Wkt =
            Wkt::from_str(wkt_value).map_err(|e| SedonaGeometryError::Invalid(e.to_string()))?;
        geo_traits_bounds_xy(wkt)
    }

    pub fn wkt_bounds_xyzm(wkt_value: &str) -> Result<BoundingBox, SedonaGeometryError> {
        let wkt: Wkt =
            Wkt::from_str(wkt_value).map_err(|e| SedonaGeometryError::Invalid(e.to_string()))?;
        let mut x = Interval::empty();
        let mut y = Interval::empty();
        let mut z = Interval::empty();
        let mut m = Interval::empty();
        for (interval, target) in zip([&mut x, &mut y, &mut z, &mut m], ["x", "y", "z", "m"]) {
            geo_traits_update_dimension_bounds(&wkt, interval, target)?
        }

        Ok(BoundingBox::xyzm(x, y, Some(z), Some(m)))
    }

    #[test]
    fn test_wkt_bounds_xy() {
        assert_eq!(
            wkt_bounds_xy("POINT EMPTY").unwrap(),
            BoundingBox::xy(Interval::empty(), Interval::empty())
        );
        assert_eq!(
            wkt_bounds_xy("POINT (0 1)").unwrap(),
            BoundingBox::xy((0, 0), (1, 1))
        );
        assert_eq!(
            wkt_bounds_xy("LINESTRING (0 1, 2 3)").unwrap(),
            BoundingBox::xy((0, 2), (1, 3))
        );
        assert_eq!(
            wkt_bounds_xy("POLYGON ((0 1, 0 2, 1 1, 0 1))").unwrap(),
            BoundingBox::xy((0, 1), (1, 2))
        );
        // Not a well-behaved polygon (interior rings outside the exterior rings) but
        // we need to test that the interior rings are considered in the bounding
        assert_eq!(
            wkt_bounds_xy("POLYGON ((0 1, 0 2, 1 1, 0 1), (10 11, 11 11, 10 12, 10 11))").unwrap(),
            BoundingBox::xy((0, 11), (1, 12))
        );

        assert_eq!(
            wkt_bounds_xy("MULTIPOINT (0 1, 2 3)").unwrap(),
            BoundingBox::xy((0, 2), (1, 3))
        );

        assert_eq!(
            wkt_bounds_xy("MULTILINESTRING ((0 1, 2 3))").unwrap(),
            BoundingBox::xy((0, 2), (1, 3))
        );
        assert_eq!(
            wkt_bounds_xy("MULTIPOLYGON (((0 1, 0 2, 1 1, 0 1)))").unwrap(),
            BoundingBox::xy((0, 1), (1, 2))
        );

        assert_eq!(
            wkt_bounds_xy("GEOMETRYCOLLECTION (POINT (0 1), POINT (2 3))").unwrap(),
            BoundingBox::xy((0, 2), (1, 3))
        );
    }

    #[test]
    fn test_wkt_bounds_xy_with_zm_input() {
        // Ensure Z/M/ZM values don't cause an error and are ignored
        assert_eq!(
            wkt_bounds_xy("LINESTRING Z (0 1 2, 3 4 5)").unwrap(),
            BoundingBox::xy((0, 3), (1, 4))
        );

        assert_eq!(
            wkt_bounds_xy("LINESTRING M (0 1 2, 3 4 5)").unwrap(),
            BoundingBox::xy((0, 3), (1, 4))
        );

        assert_eq!(
            wkt_bounds_xy("LINESTRING ZM (0 1 2 3, 4 5 6 7)").unwrap(),
            BoundingBox::xy((0, 4), (1, 5))
        );
    }

    #[rstest]
    fn test_wkt_bounds_xyzm_empty(
        #[values(
            "POINT EMPTY",
            "LINESTRING EMPTY",
            "POLYGON EMPTY",
            "MULTIPOINT EMPTY",
            "MULTILINESTRING EMPTY",
            "MULTIPOLYGON EMPTY",
            "GEOMETRYCOLLECTION EMPTY"
        )]
        wkt: &str,
    ) {
        assert_eq!(
            wkt_bounds_xyzm(wkt).unwrap(),
            BoundingBox::xyzm(
                Interval::empty(),
                Interval::empty(),
                Some(Interval::empty()),
                Some(Interval::empty())
            )
        );
    }

    #[test]
    fn test_wkt_bounds_xyzm_point() {
        assert_eq!(
            wkt_bounds_xyzm("POINT (0 1)").unwrap(),
            BoundingBox::xyzm(
                (0, 0),
                (1, 1),
                Some(Interval::empty()),
                Some(Interval::empty())
            )
        );
        assert_eq!(
            wkt_bounds_xyzm("POINT Z (0 1 2)").unwrap(),
            BoundingBox::xyzm((0, 0), (1, 1), Some((2, 2).into()), Some(Interval::empty()))
        );
        assert_eq!(
            wkt_bounds_xyzm("POINT M (0 1 3)").unwrap(),
            BoundingBox::xyzm((0, 0), (1, 1), Some(Interval::empty()), Some((3, 3).into()))
        );
        assert_eq!(
            wkt_bounds_xyzm("POINT ZM (0 1 2 3)").unwrap(),
            BoundingBox::xyzm((0, 0), (1, 1), Some((2, 2).into()), Some((3, 3).into()))
        );
    }

    #[test]
    fn test_wkt_bounds_xyzm_linestring() {
        assert_eq!(
            wkt_bounds_xyzm("LINESTRING (0 1, 4 5)").unwrap(),
            BoundingBox::xyzm(
                (0, 4),
                (1, 5),
                Some(Interval::empty()),
                Some(Interval::empty())
            )
        );
        assert_eq!(
            wkt_bounds_xyzm("LINESTRING Z (0 1 2, 4 5 6)").unwrap(),
            BoundingBox::xyzm((0, 4), (1, 5), Some((2, 6).into()), Some(Interval::empty()))
        );
        assert_eq!(
            wkt_bounds_xyzm("LINESTRING M (0 1 3, 4 5 7)").unwrap(),
            BoundingBox::xyzm((0, 4), (1, 5), Some(Interval::empty()), Some((3, 7).into()))
        );
        assert_eq!(
            wkt_bounds_xyzm("LINESTRING ZM (0 1 2 3, 4 5 6 7)").unwrap(),
            BoundingBox::xyzm((0, 4), (1, 5), Some((2, 6).into()), Some((3, 7).into()))
        );
    }

    #[test]
    fn test_wkt_bounds_xyzm_polygon() {
        assert_eq!(
            wkt_bounds_xyzm("POLYGON ((0 1, 0 2, 1 1, 0 1), (10 11, 11 11, 10 12, 10 11))")
                .unwrap(),
            BoundingBox::xyzm(
                (0, 11),
                (1, 12),
                Some(Interval::empty()),
                Some(Interval::empty())
            )
        );

        assert_eq!(
            wkt_bounds_xyzm(
                "
                POLYGON ZM (
                    (0 1 20 21, 0 2 20 21, 1 1 20 21, 0 1 20 21),
                    (10 11 20 21, 11 11 20 21, 10 12 20 21, 10 11 20 21)
                )
            "
            )
            .unwrap(),
            BoundingBox::xyzm(
                (0, 11),
                (1, 12),
                Some((20, 20).into()),
                Some((21, 21).into())
            )
        );
    }

    #[test]
    fn test_wkt_bounds_xyzm_collections() {
        assert_eq!(
            wkt_bounds_xyzm("MULTIPOINT ZM (0 1 2 3, 4 5 6 7)").unwrap(),
            BoundingBox::xyzm((0, 4), (1, 5), Some((2, 6).into()), Some((3, 7).into()))
        );
        assert_eq!(
            wkt_bounds_xyzm("MULTILINESTRING ZM ((0 1 2 3, 4 5 6 7))").unwrap(),
            BoundingBox::xyzm((0, 4), (1, 5), Some((2, 6).into()), Some((3, 7).into()))
        );
        assert_eq!(
            wkt_bounds_xyzm("MULTIPOLYGON ZM (((0 1 3 4, 0 2 3 4, 1 1 3 4, 0 1 3 4)))").unwrap(),
            BoundingBox::xyzm((0, 1), (1, 2), Some((3, 3).into()), Some((4, 4).into()))
        );

        assert_eq!(
            wkt_bounds_xyzm("GEOMETRYCOLLECTION ZM (POINT ZM (0 1 2 3), POINT ZM (4 5 6 7))")
                .unwrap(),
            BoundingBox::xyzm((0, 4), (1, 5), Some((2, 6).into()), Some((3, 7).into()))
        );
    }

    #[test]
    fn test_wkb_bounds_xy() {
        let wkt: Wkt = Wkt::from_str("POINT (0 1)").unwrap();
        let mut out = Vec::new();
        wkb::writer::write_geometry(
            &mut out,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        assert_eq!(
            wkb_bounds_xy(&out).unwrap(),
            BoundingBox::xy((0, 0), (1, 1))
        );
    }
}
