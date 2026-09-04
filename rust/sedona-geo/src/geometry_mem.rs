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

//! Heap-size measurement for geo geometries, used by aggregate accumulators
//! to report their memory use to the memory pool.

use std::mem::size_of;

/// The heap bytes owned by a [geo::Geometry].
///
/// The enum itself lives inline in its container; this counts only owned
/// buffers. Capacity is used where the backing Vec is reachable; interior
/// rings expose only a slice, so their container overhead is counted by
/// length, which matches the allocated size for geometries produced by
/// construction or boolean ops.
pub(crate) fn geometry_heap_size(geom: &geo::Geometry) -> usize {
    use geo::Geometry::*;
    match geom {
        Point(_) | Line(_) | Rect(_) | Triangle(_) => 0,
        LineString(ls) => line_string_heap_size(ls),
        Polygon(p) => polygon_heap_size(p),
        MultiPoint(mp) => mp.0.capacity() * size_of::<geo::Point>(),
        MultiLineString(mls) => {
            mls.0.capacity() * size_of::<geo::LineString>()
                + mls.0.iter().map(line_string_heap_size).sum::<usize>()
        }
        MultiPolygon(mp) => {
            mp.0.capacity() * size_of::<geo::Polygon>()
                + mp.0.iter().map(polygon_heap_size).sum::<usize>()
        }
        GeometryCollection(gc) => {
            gc.0.capacity() * size_of::<geo::Geometry>()
                + gc.0.iter().map(geometry_heap_size).sum::<usize>()
        }
    }
}

fn line_string_heap_size(ls: &geo::LineString) -> usize {
    ls.0.capacity() * size_of::<geo::Coord>()
}

fn polygon_heap_size(p: &geo::Polygon) -> usize {
    line_string_heap_size(p.exterior())
        + std::mem::size_of_val(p.interiors())
        + p.interiors()
            .iter()
            .map(line_string_heap_size)
            .sum::<usize>()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn counts_rings_and_container_overhead() {
        let exterior = geo::LineString::from(vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 0.0)]);
        let hole = geo::LineString::from(vec![(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 1.0)]);
        let coords_bytes = (exterior.0.len() + hole.0.len()) * size_of::<geo::Coord>();
        let poly = geo::Polygon::new(exterior, vec![hole]);
        let mp = geo::Geometry::MultiPolygon(geo::MultiPolygon(vec![poly]));

        let measured = geometry_heap_size(&mp);
        assert!(
            measured >= coords_bytes + size_of::<geo::Polygon>() + size_of::<geo::LineString>()
        );

        assert_eq!(
            geometry_heap_size(&geo::Geometry::Point(geo::Point::new(0.0, 0.0))),
            0
        );

        let gc = geo::Geometry::GeometryCollection(geo::GeometryCollection(vec![mp]));
        assert!(geometry_heap_size(&gc) >= measured + size_of::<geo::Geometry>());
    }
}
