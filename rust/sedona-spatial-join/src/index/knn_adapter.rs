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

use once_cell::sync::OnceCell;

use geo::{Centroid, Distance, Euclidean, Haversine};
use geo_types::{Geometry, Point};
use sedona_expr::statistics::GeoStatistics;
use sedona_geo::to_geo::item_to_geometry;

use crate::evaluated_batch::EvaluatedBatch;

/// Shared KNN components that can be reused across queries
pub(crate) struct KnnComponents {
    /// Pre-allocated vector for geometry cache - lock-free access
    /// Indexed by rtree data index for O(1) access
    geometry_cache: Vec<OnceCell<Geometry<f64>>>,
    /// Estimated memory usage for decoded geometries
    estimated_memory_usage: usize,
}

/// Heap bytes of the eagerly-allocated `geometry_cache` backbone: one `OnceCell`
/// slot per indexed geometry, allocated up front in [`KnnComponents::new`]. Each
/// slot reserves inline space for the decoded `Geometry`, so this term is exact
/// and independent of how many geometries are later decoded on demand. The
/// nested coordinate buffers of decoded geometries are allocated lazily and are
/// not counted here.
fn cache_backbone_bytes(cache_size: usize) -> usize {
    cache_size * std::mem::size_of::<OnceCell<Geometry<f64>>>()
}

impl KnnComponents {
    pub fn new(
        cache_size: usize,
        indexed_batches: &[EvaluatedBatch],
    ) -> datafusion_common::Result<Self> {
        // Pre-allocate OnceCell vector
        let geometry_cache = (0..cache_size).map(|_| OnceCell::new()).collect();
        let mut total_wkb_size = 0;
        for batch in indexed_batches {
            for wkb in batch.geom_array.wkbs().iter().flatten() {
                total_wkb_size += wkb.buf().len();
            }
        }

        Ok(Self {
            geometry_cache,
            estimated_memory_usage: total_wkb_size + cache_backbone_bytes(cache_size),
        })
    }

    /// Estimate the maximum memory usage for decoded geometries based on statistics
    pub fn estimate_max_memory_usage(build_stats: &GeoStatistics) -> usize {
        let geom_count = build_stats.total_geometries().unwrap_or(0) as usize;
        build_stats.total_size_bytes().unwrap_or(0) as usize + cache_backbone_bytes(geom_count)
    }

    pub fn estimated_memory_usage(&self) -> usize {
        self.estimated_memory_usage
    }
}

/// Geometry accessor for SedonaDB KNN queries.
/// This accessor provides on-demand WKB decoding and geometry caching for efficient
/// KNN queries with support for both Euclidean and Haversine distance metrics.
pub(crate) struct SedonaKnnAdapter<'a> {
    indexed_batches: &'a [EvaluatedBatch],
    data_id_to_batch_pos: &'a [(i32, i32)],
    // Reference to KNN components for cache and memory tracking
    knn_components: &'a KnnComponents,
}

impl<'a> SedonaKnnAdapter<'a> {
    /// Create a new adapter
    pub fn new(
        indexed_batches: &'a [EvaluatedBatch],
        data_id_to_batch_pos: &'a [(i32, i32)],
        knn_components: &'a KnnComponents,
    ) -> Self {
        Self {
            indexed_batches,
            data_id_to_batch_pos,
            knn_components,
        }
    }

    /// Get geometry for the given item index with lock-free caching
    pub fn get_geometry(&self, item_index: usize) -> Option<&Geometry<f64>> {
        let geometry_cache = &self.knn_components.geometry_cache;

        // Bounds check
        if item_index >= geometry_cache.len() || item_index >= self.data_id_to_batch_pos.len() {
            return None;
        }

        // Try to get from cache first
        if let Some(geom) = geometry_cache[item_index].get() {
            return Some(geom);
        }

        // Cache miss - decode from WKB
        let (batch_idx, row_idx) = self.data_id_to_batch_pos[item_index];
        let indexed_batch = &self.indexed_batches[batch_idx as usize];

        if let Some(wkb) = indexed_batch.geom_array.wkb(row_idx as usize)
            && let Ok(geom) = item_to_geometry(wkb)
        {
            // Try to store in cache - if another thread got there first, we just use theirs
            let _ = geometry_cache[item_index].set(geom);
            // Return reference to the cached geometry
            return geometry_cache[item_index].get();
        }

        // Failed to decode - don't cache invalid results
        None
    }

    pub fn distance(
        &self,
        probe: &Geometry<f64>,
        item_index: usize,
        use_spheroid: bool,
    ) -> Option<f64> {
        let item = self.get_geometry(item_index)?;
        if use_spheroid {
            let probe_centroid = probe.centroid()?;
            let item_centroid = item.centroid()?;
            Some(Haversine.distance(probe_centroid, item_centroid))
        } else {
            Some(Euclidean.distance(probe, item))
        }
    }
}

/// Euclidean distance from a point to an axis-aligned box. This is the exact distance and
/// much cheaper than wrapping the box as a [`Geometry::Rect`] and using the generic
/// geometry distance.
pub(crate) fn euclidean_point_to_box_distance(p: Point<f64>, bbox: [f64; 4]) -> f64 {
    let [min_x, min_y, max_x, max_y] = bbox;
    let dx = axis_dist(p.x(), min_x, max_x);
    let dy = axis_dist(p.y(), min_y, max_y);
    (dx * dx + dy * dy).sqrt()
}

fn axis_dist(v: f64, min: f64, max: f64) -> f64 {
    if v < min {
        min - v
    } else if v > max {
        v - max
    } else {
        0.0
    }
}

/// Lower bound of the Haversine distance from a longitude/latitude point to any point inside
/// a longitude/latitude box, in the same units as [`Haversine`].
///
/// If the point's longitude (modulo 360) is within the box's longitude range, the closest
/// point is on the same meridian and the distance is the latitude gap. Otherwise the closest
/// point lies on one of the box's two meridian edges. On a meridian edge, the distance is
/// minimized at the latitude closest to the great-circle foot of the perpendicular from the
/// point, or at one of the edge's endpoints. This handles boxes adjacent to the antimeridian,
/// boxes touching a pole and longitudes outside [-180, 180]. Invalid latitudes or non-finite
/// coordinates give a bound of zero.
pub(crate) fn haversine_point_to_box_lower_bound(p: Point<f64>, bbox: [f64; 4]) -> f64 {
    let [min_lon, min_lat, max_lon, max_lat] = bbox;
    let (lon, lat) = (p.x(), p.y());

    // Longitudes are not required to be normalized, but latitudes outside [-90, 90] or
    // non-finite coordinates have no meaningful spherical bound, so don't prune those
    let lat_range = -90.0..=90.0;
    if !(lat_range.contains(&lat) && lat_range.contains(&min_lat) && lat_range.contains(&max_lat))
        || !(lon.is_finite() && min_lon.is_finite() && max_lon.is_finite())
    {
        return 0.0;
    }

    // Haversine distance is periodic in longitude, so test containment modulo 360
    if (lon - min_lon).rem_euclid(360.0) <= max_lon - min_lon {
        let closest_lat = lat.clamp(min_lat, max_lat);
        return Haversine.distance(p, Point::new(lon, closest_lat));
    }

    let (sin_lat, cos_lat) = lat.to_radians().sin_cos();
    [min_lon, max_lon]
        .into_iter()
        .map(|edge_lon| {
            // Latitude on the edge's meridian closest to the point. When it is within
            // [-90, 90], the distance along the meridian is unimodal around it, so the clamped
            // latitude is the closest point on the edge. Otherwise (the meridian is more than
            // 90 degrees away), the closest point is one of the edge's endpoints.
            let d_lon = (edge_lon - lon).to_radians();
            let foot_lat = sin_lat.atan2(cos_lat * d_lon.cos()).to_degrees();
            let dist_at = |edge_lat: f64| Haversine.distance(p, Point::new(edge_lon, edge_lat));
            if (-90.0..=90.0).contains(&foot_lat) {
                dist_at(foot_lat.clamp(min_lat, max_lat))
            } else {
                dist_at(min_lat).min(dist_at(max_lat))
            }
        })
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimated_memory_usage_includes_cache_backbone() {
        let slot = std::mem::size_of::<OnceCell<Geometry<f64>>>();

        // Empty cache: no slots and no decoded geometries.
        let empty = KnnComponents::new(0, &[]).unwrap();
        assert_eq!(empty.estimated_memory_usage(), 0);

        // With N slots and no build batches, the estimate is exactly the eager
        // backbone (one OnceCell slot per indexed geometry), with no WKB bytes.
        let n = 1000;
        let components = KnnComponents::new(n, &[]).unwrap();
        assert_eq!(components.estimated_memory_usage(), n * slot);
    }

    #[test]
    fn estimate_max_memory_usage_includes_cache_backbone() {
        let slot = std::mem::size_of::<OnceCell<Geometry<f64>>>();
        let stats = GeoStatistics::empty()
            .with_total_geometries(1000)
            .with_total_size_bytes(4096);
        assert_eq!(
            KnnComponents::estimate_max_memory_usage(&stats),
            4096 + 1000 * slot
        );
    }

    /// Minimum distance from `p` to a dense grid of points covering `bbox`, including its
    /// boundary, along with the distance between neighbouring grid points
    fn sampled_min_distance(
        p: Point<f64>,
        bbox: [f64; 4],
        distance: impl Fn(Point<f64>, Point<f64>) -> f64,
    ) -> (f64, f64) {
        let [min_x, min_y, max_x, max_y] = bbox;
        let n = 40;
        let (step_x, step_y) = ((max_x - min_x) / n as f64, (max_y - min_y) / n as f64);
        let mut min_dist = f64::INFINITY;
        for i in 0..=n {
            for j in 0..=n {
                let q = Point::new(min_x + step_x * i as f64, min_y + step_y * j as f64);
                min_dist = min_dist.min(distance(p, q));
            }
        }
        let corner = Point::new(min_x, min_y);
        let grid_step = distance(corner, Point::new(min_x + step_x, min_y + step_y)).max(distance(
            Point::new(min_x, max_y),
            Point::new(min_x + step_x, max_y - step_y),
        ));
        (min_dist, grid_step)
    }

    fn random_box(rng: &mut fastrand::Rng) -> [f64; 4] {
        let (lon0, lon1) = (rng.f64() * 360.0 - 180.0, rng.f64() * 360.0 - 180.0);
        let (lat0, lat1) = (rng.f64() * 180.0 - 90.0, rng.f64() * 180.0 - 90.0);
        [
            lon0.min(lon1),
            lat0.min(lat1),
            lon0.max(lon1),
            lat0.max(lat1),
        ]
    }

    #[test]
    fn euclidean_point_to_box_distance_matches_geo() {
        let mut rng = fastrand::Rng::with_seed(7);
        for _ in 0..1000 {
            let bbox = random_box(&mut rng);
            let p = Point::new(rng.f64() * 400.0 - 200.0, rng.f64() * 200.0 - 100.0);
            let rect = Geometry::Rect(geo::Rect::new(
                geo::coord! { x: bbox[0], y: bbox[1] },
                geo::coord! { x: bbox[2], y: bbox[3] },
            ));
            let expected = Euclidean.distance(&Geometry::Point(p), &rect);
            let actual = euclidean_point_to_box_distance(p, bbox);
            assert!((actual - expected).abs() <= 1e-9, "{p:?} {bbox:?}");
        }
    }

    #[test]
    fn haversine_point_to_box_lower_bound_is_tight_lower_bound() {
        let mut rng = fastrand::Rng::with_seed(11);
        let mut boxes: Vec<[f64; 4]> = (0..60).map(|_| random_box(&mut rng)).collect();
        // Small boxes, boxes touching the antimeridian and boxes containing a pole
        for _ in 0..30 {
            let (lon, lat) = (rng.f64() * 350.0 - 180.0, rng.f64() * 170.0 - 90.0);
            boxes.push([lon, lat, lon + rng.f64() * 10.0, lat + rng.f64() * 10.0]);
        }
        boxes.push([-180.0, -10.0, -179.0, 10.0]);
        boxes.push([179.0, -10.0, 180.0, 10.0]);
        boxes.push([-30.0, 80.0, 30.0, 90.0]);
        boxes.push([100.0, -90.0, 170.0, -75.0]);
        boxes.push([-180.0, -90.0, 180.0, 90.0]);
        // Longitudes outside [-180, 180]
        boxes.push([370.0, 10.0, 380.0, 20.0]);
        boxes.push([-725.0, -5.0, -700.0, 5.0]);

        let mut points: Vec<Point<f64>> = (0..8)
            .map(|_| Point::new(rng.f64() * 360.0 - 180.0, rng.f64() * 180.0 - 90.0))
            .collect();
        points.extend([
            Point::new(179.9, 0.0),
            Point::new(-179.9, 0.0),
            Point::new(0.0, 89.9),
            Point::new(-150.0, -89.9),
            Point::new(0.0, 0.0),
            Point::new(1095.0, 0.0),
            Point::new(-345.0, 15.0),
        ]);

        for bbox in &boxes {
            for &p in &points {
                let bound = haversine_point_to_box_lower_bound(p, *bbox);
                let (sampled, grid_step) =
                    sampled_min_distance(p, *bbox, |a, b| Haversine.distance(a, b));
                assert!(
                    bound <= sampled + 1e-6,
                    "bound {bound} > sampled {sampled} for {p:?} {bbox:?}"
                );
                assert!(
                    bound >= sampled - grid_step,
                    "bound {bound} not tight vs sampled {sampled} for {p:?} {bbox:?}"
                );
            }
        }
    }

    #[test]
    fn haversine_point_to_box_lower_bound_wraps_antimeridian() {
        // Planar clamping would put this box ~359 degrees away
        let p = Point::new(179.5, 0.0);
        let bound = haversine_point_to_box_lower_bound(p, [-180.0, -1.0, -179.0, 1.0]);
        let expected = Haversine.distance(p, Point::new(180.0, 0.0));
        assert!((bound - expected).abs() < 1e-6, "{bound} vs {expected}");
    }

    #[test]
    fn haversine_point_to_box_lower_bound_unnormalized_longitude() {
        // 1095 degrees is 15 degrees, which is inside the box
        let bound =
            haversine_point_to_box_lower_bound(Point::new(1095.0, 0.0), [10.0, -1.0, 20.0, 1.0]);
        assert_eq!(bound, 0.0);
        let bound =
            haversine_point_to_box_lower_bound(Point::new(15.0, 0.0), [370.0, -1.0, 380.0, 1.0]);
        assert_eq!(bound, 0.0);
    }

    #[test]
    fn haversine_point_to_box_lower_bound_invalid_coordinates() {
        let bbox = [10.0, -1.0, 20.0, 1.0];
        assert_eq!(
            haversine_point_to_box_lower_bound(Point::new(100.0, 95.0), bbox),
            0.0
        );
        assert_eq!(
            haversine_point_to_box_lower_bound(Point::new(f64::NAN, 0.0), bbox),
            0.0
        );
        assert_eq!(
            haversine_point_to_box_lower_bound(Point::new(100.0, 0.0), [10.0, -1.0, 20.0, 91.0]),
            0.0
        );
    }
}
