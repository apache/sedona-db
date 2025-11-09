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
//! Euclidean distance helper utilities (generic)
//!
//! Ported (and contains copied code) from `geo::algorithm::line_measures::metric_spaces::euclidean::utils`:
//! <https://github.com/georust/geo/blob/5d667f844716a3d0a17aa60bc0a58528cb5808c3/geo/src/algorithm/line_measures/metric_spaces/euclidean/utils.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
use crate::algorithm::Intersects;
use crate::coordinate_position::{coord_pos_relative_to_ring, CoordPos};
use crate::geometry::*;
use crate::{CoordFloat, GeoFloat, GeoNum};
use geo_traits::{CoordTrait, LineStringTrait};
use num_traits::{Bounded, Float};
use sedona_geo_traits_ext::{
    LineStringTraitExt, LineTraitExt, PointTraitExt, PolygonTraitExt, TriangleTraitExt,
};

// ┌────────────────────────────────────────────────────────────┐
// │ Helper functions for generic distance calculations         │
// └────────────────────────────────────────────────────────────┘

pub fn nearest_neighbour_distance<F: GeoFloat>(geom1: &LineString<F>, geom2: &LineString<F>) -> F {
    let mut min_distance: F = Bounded::max_value();

    // Primary computation: line-to-line distances
    for line1 in geom1.lines() {
        for line2 in geom2.lines() {
            let line_distance = distance_line_to_line_generic(&line1, &line2);
            min_distance = min_distance.min(line_distance);

            // Early exit if we found an intersection
            if line_distance == F::zero() {
                return F::zero();
            }
        }
    }

    // Check points of geom2 against lines of geom1
    for point in geom2.points() {
        if let Some(coord) = point.coord_ext() {
            for line1 in geom1.lines() {
                let dist = distance_coord_to_line_generic(&coord, &line1);
                min_distance = min_distance.min(dist);
            }
        }
    }

    // Check points of geom1 against lines of geom2
    for point in geom1.points() {
        if let Some(coord) = point.coord_ext() {
            for line2 in geom2.lines() {
                let dist = distance_coord_to_line_generic(&coord, &line2);
                min_distance = min_distance.min(dist);
            }
        }
    }

    min_distance
}

pub fn ring_contains_coord<T: GeoNum>(ring: &LineString<T>, c: Coord<T>) -> bool {
    match coord_pos_relative_to_ring(c, ring) {
        CoordPos::Inside => true,
        CoordPos::OnBoundary | CoordPos::Outside => false,
    }
}

/// Generic point-to-point Euclidean distance calculation.
///
/// # Algorithm Equivalence to Concrete Implementation
///
/// This function is algorithmically identical to the concrete `Distance<F, Coord<F>, Coord<F>>` implementation.
///
/// **Equivalence Details:**
/// - **Same mathematical formula**: Both use Euclidean distance `sqrt(Δx² + Δy²)` via `hypot()`
/// - **Same calculation steps**: Extract coordinates, compute deltas, apply `hypot()`
/// - **Same edge case handling**: Both return 0 for invalid/empty points
/// - **Same numerical precision**: Both use identical `hypot()` implementation
///
/// The only difference is the abstraction layer - this generic version works with any
/// type implementing `PointTraitExt`, while concrete works with `Coord<F>` directly.
pub fn distance_point_to_point_generic<F, P1, P2>(p1: &P1, p2: &P2) -> F
where
    F: CoordFloat,
    P1: PointTraitExt<T = F>,
    P2: PointTraitExt<T = F>,
{
    if let (Some(c1), Some(c2)) = (p1.coord(), p2.coord()) {
        let delta_x = c1.x() - c2.x();
        let delta_y = c1.y() - c2.y();
        delta_x.hypot(delta_y)
    } else {
        F::zero()
    }
}

/// Generic coordinate-to-line-segment distance calculation.
///
/// # Algorithm Equivalence to Concrete Implementation
///
/// This function is algorithmically identical to the concrete `line_segment_distance` function
/// in `geo-types/src/private_utils.rs`.
///
/// **Equivalence Details:**
/// - **Same parametric approach**: Both compute parameter `r` to find the closest point on the line
/// - **Same boundary handling**: Both check if `r <= 0` (closest to start) or `r >= 1` (closest to end)
/// - **Same degenerate case**: Both handle zero-length lines by computing direct point distance
/// - **Same perpendicular distance formula**: Both use cross product formula `s.abs() * dx.hypot(dy)` for interior points
/// - **Same numerical precision**: Both use identical calculations and `hypot()` calls
///
/// The concrete implementation uses `line_euclidean_length()` helper for endpoint distances,
/// while this uses inline `delta.hypot()` - both compute the same Euclidean distance.
pub fn distance_coord_to_line_generic<F, C, L>(coord: &C, line: &L) -> F
where
    F: CoordFloat,
    C: CoordTrait<T = F>,
    L: LineTraitExt<T = F>,
{
    let point_x = coord.x();
    let point_y = coord.y();
    let start = line.start_coord();
    let end = line.end_coord();

    // Handle degenerate case: line segment is a point
    if start.x == end.x && start.y == end.y {
        let delta_x = point_x - start.x;
        let delta_y = point_y - start.y;
        return delta_x.hypot(delta_y);
    }

    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let d_squared = dx * dx + dy * dy;
    let r = ((point_x - start.x) * dx + (point_y - start.y) * dy) / d_squared;

    if r <= F::zero() {
        // Closest point is the start point
        let delta_x = point_x - start.x;
        let delta_y = point_y - start.y;
        return delta_x.hypot(delta_y);
    }
    if r >= F::one() {
        // Closest point is the end point
        let delta_x = point_x - end.x;
        let delta_y = point_y - end.y;
        return delta_x.hypot(delta_y);
    }

    // Closest point is on the line segment - use perpendicular distance
    let s = ((start.y - point_y) * dx - (start.x - point_x) * dy) / d_squared;
    s.abs() * dx.hypot(dy)
}

/// Generic point-to-linestring distance calculation.
///
/// # Algorithm Equivalence to Concrete Implementation
///
/// This function is algorithmically identical to the concrete `point_line_string_euclidean_distance` function
/// in `geo-types/src/private_utils.rs`.
///
/// **Equivalence Details:**
/// - **Same containment check optimization**: Both check if point intersects/is contained in the linestring first
/// - **Same early exit**: Both return 0 immediately if point is on the linestring
/// - **Same iteration approach**: Both iterate through all line segments to find minimum distance
/// - **Same distance calculation**: Both use point-to-line-segment distance for each segment
/// - **Same empty handling**: Both return 0 for empty linestrings
///
/// The concrete implementation uses `line_string_contains_point()` while this uses `intersects()` trait method,
/// but both perform the same containment check. The iteration pattern and minimum distance logic are identical.
pub fn distance_point_to_linestring_generic<F, P, LS>(point: &P, linestring: &LS) -> F
where
    F: GeoFloat,
    P: PointTraitExt<T = F>,
    LS: LineStringTraitExt<T = F>,
{
    if let Some(coord) = point.coord() {
        // Early exit optimization: if point is on the linestring, distance is 0
        // Check if the point is contained in the linestring using intersects
        if linestring.intersects(point) {
            return F::zero();
        }

        let mut lines = linestring.lines();
        if let Some(first_line) = lines.next() {
            let mut min_distance = distance_coord_to_line_generic(&coord, &first_line);
            for line in lines {
                min_distance = min_distance.min(distance_coord_to_line_generic(&coord, &line));
            }
            min_distance
        } else {
            F::zero()
        }
    } else {
        F::zero()
    }
}

/// Point to Polygon distance
///
/// # Algorithm Equivalence
///
/// This generic implementation is algorithmically identical to the concrete
/// `Distance<F, &Point<F>, &Polygon<F>>` implementation:
///
/// 1. **Intersection Check**: First checks if the point intersects the polygon
///    using the same `Intersects` trait, returning zero for any intersection
///    (boundary or interior).
///
/// 2. **Ring Distance Calculation**: If no intersection, computes minimum distance
///    by iterating through all polygon rings (exterior and all interior holes).
///
/// 3. **Minimum Selection**: Uses the same fold pattern to find the minimum
///    distance across all rings, starting with F::max_value().
///
/// The only difference is the generic trait-based interface for accessing
/// polygon components, while the core distance logic remains identical.
pub fn distance_point_to_polygon_generic<F, P, Poly>(point: &P, polygon: &Poly) -> F
where
    F: GeoFloat,
    P: PointTraitExt<T = F>,
    Poly: PolygonTraitExt<T = F>,
{
    // Check if the polygon is empty
    if polygon.exterior_ext().is_none() {
        return F::zero();
    }

    // If the point intersects the polygon (is inside or on boundary), distance is 0
    if polygon.intersects(point) {
        return F::zero();
    }

    // Point is outside the polygon, calculate minimum distance to edges
    if let (Some(coord), Some(exterior)) = (point.coord_ext(), polygon.exterior_ext()) {
        let mut min_dist: F = Float::max_value();

        // Calculate minimum distance to exterior ring - single loop
        for line in exterior.lines() {
            let dist = distance_coord_to_line_generic(&coord, &line);
            min_dist = min_dist.min(dist);
        }

        // Only check interior rings if they exist
        if polygon.interiors_ext().next().is_some() {
            for interior in polygon.interiors_ext() {
                for line in interior.lines() {
                    let dist = distance_coord_to_line_generic(&coord, &line);
                    min_dist = min_dist.min(dist);
                }
            }
        }

        min_dist
    } else {
        F::zero()
    }
}

/// Line to Line distance
///
/// # Algorithm Equivalence
///
/// This generic implementation is algorithmically identical to the concrete
/// `Distance<F, &Line<F>, &Line<F>>` implementation:
///
/// 1. **Intersection Check**: First uses the `Intersects` trait to check if the
///    lines intersect, returning zero if they do.
///
/// 2. **Four-Point Distance**: If no intersection, computes the minimum distance
///    by checking all four possible point-to-line segment distances:
///
/// 3. **Minimum Selection**: Uses the same chained min() operations to find
///    the shortest distance among all four calculations.
///
/// The generic trait interface provides the same coordinate access while
/// maintaining identical distance computation logic.
pub fn distance_line_to_line_generic<F, L1, L2>(line1: &L1, line2: &L2) -> F
where
    F: GeoFloat,
    L1: LineTraitExt<T = F>,
    L2: LineTraitExt<T = F>,
{
    let start1 = line1.start_coord();
    let end1 = line1.end_coord();
    let start2 = line2.start_coord();
    let end2 = line2.end_coord();

    // Check if lines intersect using generic intersects
    if line1.intersects(line2) {
        return F::zero();
    }

    // Find minimum distance between all endpoint combinations
    let dist1 = distance_coord_to_line_generic(&start1, line2);
    let dist2 = distance_coord_to_line_generic(&end1, line2);
    let dist3 = distance_coord_to_line_generic(&start2, line1);
    let dist4 = distance_coord_to_line_generic(&end2, line1);

    dist1.min(dist2).min(dist3).min(dist4)
}

/// Line to LineString distance
///
/// # Algorithm Equivalence
///
/// This generic implementation is algorithmically identical to the concrete
/// `Distance<F, &Line<F>, &LineString<F>>` implementation:
///
/// 1. **Segment Iteration**: Maps over each line segment in the LineString
///    using the same `lines()` iterator approach.
///
/// 2. **Line-to-Line Distance**: For each segment, calls the same line-to-line
///    distance function that handles intersection detection and four-point
///    distance calculations.
///
/// 3. **Minimum Selection**: Uses the same fold pattern with F::max_value()
///    as the starting accumulator and min() reduction to find the shortest
///    distance across all segments.
///
/// The generic trait interface provides equivalent LineString iteration while
/// maintaining identical distance computation logic.
pub fn distance_line_to_linestring_generic<F, L, LS>(line: &L, linestring: &LS) -> F
where
    F: GeoFloat,
    L: LineTraitExt<T = F>,
    LS: LineStringTraitExt<T = F>,
{
    linestring
        .lines()
        .map(|ls_line| distance_line_to_line_generic(line, &ls_line))
        .fold(Float::max_value(), |acc, dist| acc.min(dist))
}

/// Line to Polygon distance
///
/// # Algorithm Equivalence
///
/// This generic implementation is algorithmically identical to the concrete
/// `Distance<F, &Line<F>, &Polygon<F>>` implementation:
///
/// 1. **Line-to-LineString Conversion**: Converts the line segment into a
///    two-point LineString containing the start and end coordinates.
///
/// 2. **Delegation to LineString-Polygon**: Uses the same delegation pattern
///    as the concrete implementation by calling the LineString-to-Polygon
///    distance function.
///
/// 3. **Identical Logic Path**: This ensures the same containment checks,
///    intersection detection, and ring distance calculations are applied
///    as in the concrete implementation.
///
/// The conversion approach maintains algorithmic equivalence while leveraging
/// the more comprehensive LineString-to-Polygon distance logic.
pub fn distance_line_to_polygon_generic<F, L, Poly>(line: &L, polygon: &Poly) -> F
where
    F: GeoFloat,
    L: LineTraitExt<T = F>,
    Poly: PolygonTraitExt<T = F>,
{
    // Convert line to linestring and use existing linestring-to-polygon function
    let line_coords = vec![line.start_coord(), line.end_coord()];
    let line_as_ls = LineString::from(line_coords);
    distance_linestring_to_polygon_generic(&line_as_ls, polygon)
}

/// LineString to LineString distance
///
/// # Algorithm Equivalence
///
/// This generic implementation is algorithmically identical to the concrete
/// `Distance<F, &LineString<F>, &LineString<F>>` implementation:
///
/// 1. **Cartesian Product**: Uses flat_map to create all possible combinations
///    of line segments between the two LineStrings, matching the nested
///    iteration pattern of the concrete implementation.
///
/// 2. **Line-to-Line Distance**: For each segment pair, applies the same
///    line-to-line distance function with intersection detection and
///    four-point distance calculations.
///
/// 3. **Minimum Selection**: Uses the same fold pattern with F::max_value()
///    as the starting accumulator and min() reduction to find the shortest
///    distance across all segment combinations.
///
/// The generic trait interface provides equivalent segment iteration while
/// maintaining identical pairwise distance computation logic.
pub fn distance_linestring_to_linestring_generic<F, LS1, LS2>(ls1: &LS1, ls2: &LS2) -> F
where
    F: GeoFloat,
    LS1: LineStringTraitExt<T = F>,
    LS2: LineStringTraitExt<T = F>,
{
    ls1.lines()
        .flat_map(|line1| {
            ls2.lines()
                .map(move |line2| distance_line_to_line_generic(&line1, &line2))
        })
        .fold(Float::max_value(), |acc, dist| acc.min(dist))
}

/// LineString to Polygon distance
///
/// # Algorithm Equivalence
///
/// This generic implementation is algorithmically identical to the concrete
/// `Distance<F, &LineString<F>, &Polygon<F>>` implementation:
///
/// 1. **Intersection Check**: First uses the `Intersects` trait to check if
///    the LineString intersects the polygon, returning zero if they do.
///
/// 2. **Containment-Based Logic**: Implements the same containment logic as
///    the concrete implementation:
///    - If polygon has holes AND first point of LineString is inside exterior
///      ring, only check distance to interior rings (holes)
///    - Otherwise, check distance to exterior ring only
///
/// 3. **Ray Casting Algorithm**: Uses identical ray casting point-in-polygon
///    test to determine if the first LineString point is inside the exterior.
///
/// 4. **Direct Nested Loop Approach**: Unlike simpler functions that use
///    `nearest_neighbour_distance`, this function implements the distance
///    calculation directly with nested loops over LineString and polygon
///    ring segments. This matches the concrete implementation's approach
///    which requires the specialized containment logic for polygons with holes.
///
/// 5. **Early Exit**: Includes the same zero-distance early exit optimization
///    when any line segments intersect during the nested iteration.
///
/// Note:
/// The direct nested loop approach (rather than delegating to helper functions)
/// is necessary to maintain the exact containment-based ring selection logic
/// that the concrete implementation uses for polygons with holes.
/// We have seen sufficient performance improvements in benchmarks by avoiding
/// the overhead of additional function calls and iterator abstractions.
///
pub fn distance_linestring_to_polygon_generic<F, LS, Poly>(linestring: &LS, polygon: &Poly) -> F
where
    F: GeoFloat,
    LS: LineStringTraitExt<T = F>,
    Poly: PolygonTraitExt<T = F>,
{
    // Early intersect check
    if polygon.intersects(linestring) {
        return F::zero();
    }

    if let Some(exterior) = polygon.exterior_ext() {
        // Check containment logic: if polygon has holes AND first point of LineString is inside exterior ring,
        // then only consider distance to holes (interior rings). Otherwise, consider distance to exterior.
        let has_holes = polygon.interiors_ext().next().is_some();

        let first_point_inside = if has_holes {
            // Check if first point of LineString is inside the exterior ring
            if let Some(first_coord) = linestring.coords().next() {
                // Simple point-in-polygon test using ray casting
                let point_x = first_coord.x();
                let point_y = first_coord.y();
                let mut inside = false;
                let ring_coords: Vec<_> = exterior.coords().collect();
                let n = ring_coords.len();

                if n > 0 {
                    let mut j = n - 1;
                    for i in 0..n {
                        let xi = ring_coords[i].x();
                        let yi = ring_coords[i].y();
                        let xj = ring_coords[j].x();
                        let yj = ring_coords[j].y();

                        if ((yi > point_y) != (yj > point_y))
                            && (point_x < (xj - xi) * (point_y - yi) / (yj - yi) + xi)
                        {
                            inside = !inside;
                        }
                        j = i;
                    }
                }
                inside
            } else {
                false // Empty LineString
            }
        } else {
            false
        };

        if has_holes && first_point_inside {
            // LineString is inside polygon: only check distance to interior rings (holes)
            let mut min_dist: F = Float::max_value();
            for interior in polygon.interiors_ext() {
                for line1 in linestring.lines() {
                    for line2 in interior.lines() {
                        let line_dist = distance_line_to_line_generic(&line1, &line2);
                        min_dist = min_dist.min(line_dist);

                        if line_dist == F::zero() {
                            return F::zero();
                        }
                    }
                }
            }
            min_dist
        } else {
            // LineString is outside polygon or polygon has no holes: check distance to exterior ring only
            let mut min_dist: F = Float::max_value();
            for line1 in linestring.lines() {
                for line2 in exterior.lines() {
                    let line_dist = distance_line_to_line_generic(&line1, &line2);
                    min_dist = min_dist.min(line_dist);

                    if line_dist == F::zero() {
                        return F::zero();
                    }
                }
            }
            min_dist
        }
    } else {
        F::zero()
    }
}

/// Polygon to Polygon distance
///
/// # Algorithm Equivalence
///
/// This generic implementation is algorithmically identical to the concrete
/// `Distance<F, &Polygon<F>, &Polygon<F>>` implementation:
///
/// 1. **Intersection Check**: First uses the `Intersects` trait to check if
///    the polygons intersect, returning zero if they do.
///
/// 2. **Fast Path Optimization**: If neither polygon has holes, directly
///    delegates to LineString-to-LineString distance between exterior rings.
///
/// 3. **Symmetric Containment Logic**: Implements the same bidirectional
///    containment checks as the concrete implementation:
///    - If polygon1 has holes AND polygon2's first point is inside polygon1's
///      exterior, check distance from polygon2's exterior to polygon1's holes
///    - If polygon2 has holes AND polygon1's first point is inside polygon2's
///      exterior, check distance from polygon1's exterior to polygon2's holes
///
/// 4. **Mixed Approach**: Uses `nearest_neighbour_distance` for the contained
///    polygon cases (for efficiency when checking against multiple holes),
///    but delegates to `distance_linestring_to_linestring_generic` for the
///    default exterior-to-exterior case.
///
/// 5. **Point-in-Polygon Test**: Uses the same `ring_contains_coord` helper
///    function for containment detection as the concrete implementation.
///
/// The mixed approach (using both helper functions and direct delegation)
/// matches the concrete implementation's optimization strategy for different
/// geometric configurations.
pub fn distance_polygon_to_polygon_generic<F, P1, P2>(polygon1: &P1, polygon2: &P2) -> F
where
    F: GeoFloat,
    P1: PolygonTraitExt<T = F>,
    P2: PolygonTraitExt<T = F>,
{
    // Check if polygons intersect using generic intersects
    if polygon1.intersects(polygon2) {
        return F::zero();
    }

    if let (Some(ext1), Some(ext2)) = (polygon1.exterior_ext(), polygon2.exterior_ext()) {
        let has_interiors1 = polygon1.interiors_ext().next().is_some();
        let has_interiors2 = polygon2.interiors_ext().next().is_some();

        // Fast path: if no interiors in either polygon, skip containment logic entirely
        if !has_interiors1 && !has_interiors2 {
            return distance_linestring_to_linestring_generic(&ext1, &ext2);
        }

        // Symmetric containment logic matching concrete implementation exactly
        // Check if polygon_b is contained within polygon_a (has holes)
        if has_interiors1 {
            if let Some(first_coord_b) = ext2.coords_ext().next() {
                let ext1_ls = LineString::from(
                    ext1.coords_ext()
                        .map(|c| (c.x(), c.y()))
                        .collect::<Vec<_>>(),
                );

                let coord_b = Coord::from((first_coord_b.x(), first_coord_b.y()));
                if ring_contains_coord(&ext1_ls, coord_b) {
                    // polygon_b is inside polygon_a: check distance to polygon_a's holes
                    let ext2_concrete = LineString::from(
                        ext2.coords_ext()
                            .map(|c| (c.x(), c.y()))
                            .collect::<Vec<_>>(),
                    );

                    let mut mindist: F = Float::max_value();
                    for ring in polygon1.interiors_ext() {
                        let ring_concrete = LineString::from(
                            ring.coords_ext()
                                .map(|c| (c.x(), c.y()))
                                .collect::<Vec<_>>(),
                        );
                        mindist =
                            mindist.min(nearest_neighbour_distance(&ext2_concrete, &ring_concrete));
                    }
                    return mindist;
                }
            }
        }

        // Check if polygon_a is contained within polygon_b (has holes)
        if has_interiors2 {
            if let Some(first_coord_a) = ext1.coords_ext().next() {
                let ext2_ls = LineString::from(
                    ext2.coords_ext()
                        .map(|c| (c.x(), c.y()))
                        .collect::<Vec<_>>(),
                );

                let coord_a = Coord::from((first_coord_a.x(), first_coord_a.y()));
                if ring_contains_coord(&ext2_ls, coord_a) {
                    // polygon_a is inside polygon_b: check distance to polygon_b's holes
                    let ext1_concrete = LineString::from(
                        ext1.coords_ext()
                            .map(|c| (c.x(), c.y()))
                            .collect::<Vec<_>>(),
                    );

                    let mut mindist: F = Float::max_value();
                    for ring in polygon2.interiors_ext() {
                        let ring_concrete = LineString::from(
                            ring.coords_ext()
                                .map(|c| (c.x(), c.y()))
                                .collect::<Vec<_>>(),
                        );
                        mindist =
                            mindist.min(nearest_neighbour_distance(&ext1_concrete, &ring_concrete));
                    }
                    return mindist;
                }
            }
        }

        // Default case - distance between exterior rings
        distance_linestring_to_linestring_generic(&ext1, &ext2)
    } else {
        F::zero()
    }
}

/// Triangle to Point distance
pub fn distance_triangle_to_point_generic<F, T, P>(triangle: &T, point: &P) -> F
where
    F: GeoFloat,
    T: TriangleTraitExt<T = F>,
    P: PointTraitExt<T = F>,
{
    // Convert triangle to polygon and use existing point-to-polygon function
    let tri_poly = triangle.to_polygon();
    distance_point_to_polygon_generic(point, &tri_poly)
}

// ┌────────────────────────────────────────────────────────────┐
// │ Symmetric Distance Function Generator Macro                │
// └────────────────────────────────────────────────────────────┘

/// Macro to generate symmetric distance functions
/// For distance operations that are symmetric (distance(a, b) == distance(b, a)),
/// this macro generates the reverse function that calls the primary implementation
macro_rules! symmetric_distance_generic_impl {
    ($func_name_ab:ident, $func_name_ba:ident, $trait_a:ident, $trait_b:ident) => {
        #[allow(dead_code)]
        pub fn $func_name_ba<F, A, B>(b: &B, a: &A) -> F
        where
            F: GeoFloat,
            A: $trait_a<T = F>,
            B: $trait_b<T = F>,
        {
            $func_name_ab(a, b)
        }
    };
}

// Generate symmetric distance functions
symmetric_distance_generic_impl!(
    distance_point_to_linestring_generic,
    distance_linestring_to_point_generic,
    PointTraitExt,
    LineStringTraitExt
);

symmetric_distance_generic_impl!(
    distance_point_to_polygon_generic,
    distance_polygon_to_point_generic,
    PointTraitExt,
    PolygonTraitExt
);

symmetric_distance_generic_impl!(
    distance_linestring_to_polygon_generic,
    distance_polygon_to_linestring_generic,
    LineStringTraitExt,
    PolygonTraitExt
);

symmetric_distance_generic_impl!(
    distance_line_to_linestring_generic,
    distance_linestring_to_line_generic,
    LineTraitExt,
    LineStringTraitExt
);

symmetric_distance_generic_impl!(
    distance_line_to_polygon_generic,
    distance_polygon_to_line_generic,
    LineTraitExt,
    PolygonTraitExt
);

symmetric_distance_generic_impl!(
    distance_triangle_to_point_generic,
    distance_point_to_triangle_generic,
    TriangleTraitExt,
    PointTraitExt
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{coord, Line, LineString, Point, Polygon, Triangle};
    use approx::assert_relative_eq;
    use geo::algorithm::line_measures::{Distance, Euclidean};

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for point_distance_generic function                  │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_point_distance_generic_basic() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);

        let distance = distance_point_to_point_generic(&p1, &p2);
        assert_relative_eq!(distance, 5.0); // 3-4-5 triangle

        // Test symmetry
        let distance_reverse = distance_point_to_point_generic(&p2, &p1);
        assert_relative_eq!(distance, distance_reverse);
    }

    #[test]
    fn test_point_distance_generic_same_point() {
        let p = Point::new(2.5, -1.5);
        let distance = distance_point_to_point_generic(&p, &p);
        assert_relative_eq!(distance, 0.0);
    }

    #[test]
    fn test_point_distance_generic_negative_coordinates() {
        let p1 = Point::new(-2.0, -3.0);
        let p2 = Point::new(1.0, 1.0);

        let distance = distance_point_to_point_generic(&p1, &p2);
        assert_relative_eq!(distance, 5.0); // sqrt((1-(-2))^2 + (1-(-3))^2) = sqrt(9+16) = 5
    }

    #[test]
    fn test_point_distance_generic_empty_points() {
        // Test with empty points (no coordinates)
        let empty_point: Point<f64> = Point::new(f64::NAN, f64::NAN);
        let regular_point = Point::new(1.0, 1.0);

        // When either point has no valid coordinates, distance should be 0
        let distance = distance_point_to_point_generic(&empty_point, &regular_point);
        assert!(distance.is_nan() || distance == 0.0); // Implementation dependent
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for line_segment_distance_generic function           │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_line_segment_distance_generic_point_on_line() {
        let coord = coord! { x: 2.0, y: 0.0 };
        let line = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 4.0, y: 0.0 });

        let distance = distance_coord_to_line_generic(&coord, &line);
        assert_relative_eq!(distance, 0.0);
    }

    #[test]
    fn test_line_segment_distance_generic_perpendicular() {
        let coord = coord! { x: 2.0, y: 3.0 };
        let line = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 4.0, y: 0.0 });

        let distance = distance_coord_to_line_generic(&coord, &line);
        assert_relative_eq!(distance, 3.0);
    }

    #[test]
    fn test_line_segment_distance_generic_beyond_endpoint() {
        let coord = coord! { x: 6.0, y: 0.0 };
        let line = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 4.0, y: 0.0 });

        let distance = distance_coord_to_line_generic(&coord, &line);
        assert_relative_eq!(distance, 2.0); // Distance to closest endpoint (4,0)
    }

    #[test]
    fn test_line_segment_distance_generic_before_startpoint() {
        let coord = coord! { x: -2.0, y: 0.0 };
        let line = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 4.0, y: 0.0 });

        let distance = distance_coord_to_line_generic(&coord, &line);
        assert_relative_eq!(distance, 2.0); // Distance to start point (0,0)
    }

    #[test]
    fn test_line_segment_distance_generic_zero_length_line() {
        let coord = coord! { x: 2.0, y: 3.0 };
        let line = Line::new(coord! { x: 1.0, y: 1.0 }, coord! { x: 1.0, y: 1.0 });

        let distance = distance_coord_to_line_generic(&coord, &line);
        let expected = ((2.0 - 1.0).powi(2) + (3.0 - 1.0).powi(2)).sqrt();
        assert_relative_eq!(distance, expected);
    }

    #[test]
    fn test_line_segment_distance_generic_diagonal_line() {
        let coord = coord! { x: 0.0, y: 2.0 };
        let line = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 2.0, y: 2.0 });

        let distance = distance_coord_to_line_generic(&coord, &line);
        // Point (0,2) to line from (0,0) to (2,2) - should be sqrt(2)
        assert_relative_eq!(distance, std::f64::consts::SQRT_2, epsilon = 1e-10);
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for nearest_neighbour_distance function              │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_nearest_neighbour_distance_basic() {
        let ls1 = LineString::from(vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]);
        let ls2 = LineString::from(vec![(3.0, 0.0), (5.0, 0.0), (5.0, 2.0)]);

        let distance = nearest_neighbour_distance(&ls1, &ls2);
        assert_relative_eq!(distance, 1.0); // Distance between (2,0)-(2,2) and (3,0)-(5,0)
    }

    #[test]
    fn test_nearest_neighbour_distance_intersecting() {
        let ls1 = LineString::from(vec![(0.0, 0.0), (4.0, 0.0)]);
        let ls2 = LineString::from(vec![(2.0, -1.0), (2.0, 1.0)]);

        let distance = nearest_neighbour_distance(&ls1, &ls2);
        // The linestrings intersect at (2,0), so distance should be 0.0
        assert_relative_eq!(distance, 0.0);
    }

    #[test]
    fn test_nearest_neighbour_distance_parallel_lines() {
        let ls1 = LineString::from(vec![(0.0, 0.0), (4.0, 0.0)]);
        let ls2 = LineString::from(vec![(0.0, 2.0), (4.0, 2.0)]);

        let distance = nearest_neighbour_distance(&ls1, &ls2);
        assert_relative_eq!(distance, 2.0); // Perpendicular distance between parallel lines
    }

    #[test]
    fn test_nearest_neighbour_distance_single_segment_each() {
        let ls1 = LineString::from(vec![(0.0, 0.0), (1.0, 0.0)]);
        let ls2 = LineString::from(vec![(2.0, 1.0), (3.0, 1.0)]);

        let distance = nearest_neighbour_distance(&ls1, &ls2);
        let expected = ((2.0 - 1.0).powi(2) + (1.0 - 0.0).powi(2)).sqrt();
        assert_relative_eq!(distance, expected);
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for ring_contains_coord function                     │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_ring_contains_coord_inside() {
        let ring = LineString::from(vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]);
        let coord = coord! { x: 2.0, y: 2.0 };

        assert!(ring_contains_coord(&ring, coord));
    }

    #[test]
    fn test_ring_contains_coord_outside() {
        let ring = LineString::from(vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]);
        let coord = coord! { x: 5.0, y: 2.0 };

        assert!(!ring_contains_coord(&ring, coord));
    }

    #[test]
    fn test_ring_contains_coord_on_boundary() {
        let ring = LineString::from(vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]);
        let coord = coord! { x: 2.0, y: 0.0 };

        assert!(!ring_contains_coord(&ring, coord)); // On boundary = false
    }

    #[test]
    fn test_ring_contains_coord_triangle() {
        let ring = LineString::from(vec![(0.0, 0.0), (3.0, 0.0), (1.5, 2.0), (0.0, 0.0)]);
        let inside_coord = coord! { x: 1.5, y: 0.5 };
        let outside_coord = coord! { x: 3.0, y: 3.0 };

        assert!(ring_contains_coord(&ring, inside_coord));
        assert!(!ring_contains_coord(&ring, outside_coord));
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for distance_point_to_linestring_generic             │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_distance_point_to_linestring_generic_basic() {
        let point = Point::new(1.0, 2.0);
        let linestring = LineString::from(vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]);

        let distance = distance_point_to_linestring_generic(&point, &linestring);
        assert_relative_eq!(distance, 1.0); // Distance to closest segment
    }

    #[test]
    fn test_distance_point_to_linestring_generic_empty() {
        let point = Point::new(1.0, 2.0);
        let linestring = LineString::<f64>::new(vec![]);

        let distance = distance_point_to_linestring_generic(&point, &linestring);
        assert_relative_eq!(distance, 0.0);
    }

    #[test]
    fn test_distance_point_to_linestring_generic_single_point() {
        let point = Point::new(1.0, 2.0);
        let linestring = LineString::from(vec![(0.0, 0.0)]);

        let distance = distance_point_to_linestring_generic(&point, &linestring);
        assert_relative_eq!(distance, 0.0); // Single point linestring
    }

    #[test]
    fn test_distance_point_to_linestring_generic_on_linestring() {
        let point = Point::new(1.0, 0.0);
        let linestring = LineString::from(vec![(0.0, 0.0), (2.0, 0.0)]);

        let distance = distance_point_to_linestring_generic(&point, &linestring);
        assert_relative_eq!(distance, 0.0);
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for distance_point_to_polygon_generic                │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_distance_point_to_polygon_generic_outside() {
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(exterior, vec![]);
        let point = Point::new(6.0, 2.0);

        let distance = distance_point_to_polygon_generic(&point, &polygon);
        assert_relative_eq!(distance, 2.0); // Distance to right edge
    }

    #[test]
    fn test_distance_point_to_polygon_generic_inside() {
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(exterior, vec![]);
        let point = Point::new(2.0, 2.0);

        let distance = distance_point_to_polygon_generic(&point, &polygon);
        assert_relative_eq!(distance, 0.0); // Inside polygon
    }

    #[test]
    fn test_distance_point_to_polygon_generic_on_boundary() {
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(exterior, vec![]);
        let point = Point::new(2.0, 0.0);

        let distance = distance_point_to_polygon_generic(&point, &polygon);
        assert_relative_eq!(distance, 0.0); // On boundary
    }

    #[test]
    fn test_distance_point_to_polygon_generic_with_hole() {
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (6.0, 0.0),
            (6.0, 6.0),
            (0.0, 6.0),
            (0.0, 0.0),
        ]);
        let interior = LineString::from(vec![
            (2.0, 2.0),
            (4.0, 2.0),
            (4.0, 4.0),
            (2.0, 4.0),
            (2.0, 2.0),
        ]);
        let polygon = Polygon::new(exterior, vec![interior]);
        let point = Point::new(3.0, 3.0); // Inside the hole

        let distance = distance_point_to_polygon_generic(&point, &polygon);
        assert_relative_eq!(distance, 1.0); // Distance to closest hole edge
    }

    #[test]
    fn test_distance_point_to_polygon_generic_empty() {
        let empty_polygon = Polygon::new(LineString::<f64>::new(vec![]), vec![]);
        let point = Point::new(1.0, 1.0);

        let distance = distance_point_to_polygon_generic(&point, &empty_polygon);
        assert_relative_eq!(distance, 0.0);
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for distance_line_to_line_generic                    │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_distance_line_to_line_generic_parallel() {
        let line1 = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 2.0, y: 0.0 });
        let line2 = Line::new(coord! { x: 0.0, y: 3.0 }, coord! { x: 2.0, y: 3.0 });

        let distance = distance_line_to_line_generic(&line1, &line2);
        assert_relative_eq!(distance, 3.0);
    }

    #[test]
    fn test_distance_line_to_line_generic_intersecting() {
        let line1 = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 2.0, y: 0.0 });
        let line2 = Line::new(coord! { x: 1.0, y: -1.0 }, coord! { x: 1.0, y: 1.0 });

        let distance = distance_line_to_line_generic(&line1, &line2);
        assert_relative_eq!(distance, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_distance_line_to_line_generic_skew() {
        let line1 = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 1.0, y: 0.0 });
        let line2 = Line::new(coord! { x: 2.0, y: 1.0 }, coord! { x: 3.0, y: 1.0 });

        let distance = distance_line_to_line_generic(&line1, &line2);
        let expected = ((2.0 - 1.0).powi(2) + (1.0 - 0.0).powi(2)).sqrt();
        assert_relative_eq!(distance, expected);
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for distance_linestring_to_polygon_generic           │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_distance_linestring_to_polygon_generic_outside() {
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(exterior, vec![]);
        let linestring = LineString::from(vec![(3.0, 0.0), (4.0, 1.0)]);

        let distance = distance_linestring_to_polygon_generic(&linestring, &polygon);
        assert_relative_eq!(distance, 1.0); // Distance to right edge of polygon
    }

    #[test]
    fn test_distance_linestring_to_polygon_generic_intersecting() {
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(exterior, vec![]);
        let linestring = LineString::from(vec![(-1.0, 1.0), (3.0, 1.0)]);

        let distance = distance_linestring_to_polygon_generic(&linestring, &polygon);
        // The linestring intersects the polygon, so distance should be 0.0
        assert_relative_eq!(distance, 0.0);
    }

    #[test]
    fn test_distance_linestring_to_polygon_generic_empty_polygon() {
        let empty_polygon = Polygon::new(LineString::<f64>::new(vec![]), vec![]);
        let linestring = LineString::from(vec![(0.0, 0.0), (1.0, 1.0)]);

        let distance = distance_linestring_to_polygon_generic(&linestring, &empty_polygon);
        assert_relative_eq!(distance, 0.0);
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for distance_polygon_to_polygon_generic              │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_distance_polygon_to_polygon_generic_separate() {
        let exterior1 = LineString::from(vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.0, 0.0),
        ]);
        let polygon1 = Polygon::new(exterior1, vec![]);

        let exterior2 = LineString::from(vec![
            (4.0, 0.0),
            (6.0, 0.0),
            (6.0, 2.0),
            (4.0, 2.0),
            (4.0, 0.0),
        ]);
        let polygon2 = Polygon::new(exterior2, vec![]);

        let distance = distance_polygon_to_polygon_generic(&polygon1, &polygon2);
        assert_relative_eq!(distance, 2.0); // Distance between closest edges
    }

    #[test]
    fn test_distance_polygon_to_polygon_generic_intersecting() {
        let exterior1 = LineString::from(vec![
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 3.0),
            (0.0, 3.0),
            (0.0, 0.0),
        ]);
        let polygon1 = Polygon::new(exterior1, vec![]);

        let exterior2 = LineString::from(vec![
            (1.0, 1.0),
            (4.0, 1.0),
            (4.0, 4.0),
            (1.0, 4.0),
            (1.0, 1.0),
        ]);
        let polygon2 = Polygon::new(exterior2, vec![]);

        let distance = distance_polygon_to_polygon_generic(&polygon1, &polygon2);
        assert_relative_eq!(distance, 0.0, epsilon = 1e-10); // Polygons intersect
    }

    #[test]
    fn test_distance_polygon_to_polygon_generic_one_in_others_hole() {
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let interior = LineString::from(vec![
            (2.0, 2.0),
            (8.0, 2.0),
            (8.0, 8.0),
            (2.0, 8.0),
            (2.0, 2.0),
        ]);
        let polygon_with_hole = Polygon::new(exterior, vec![interior]);

        let small_exterior = LineString::from(vec![
            (4.0, 4.0),
            (6.0, 4.0),
            (6.0, 6.0),
            (4.0, 6.0),
            (4.0, 4.0),
        ]);
        let small_polygon = Polygon::new(small_exterior, vec![]);

        let distance = distance_polygon_to_polygon_generic(&polygon_with_hole, &small_polygon);
        assert_relative_eq!(distance, 2.0); // Distance to hole boundary
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for symmetric distance functions                     │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_symmetric_distance_point_linestring() {
        let point = Point::new(1.0, 2.0);
        let linestring = LineString::from(vec![(0.0, 0.0), (2.0, 0.0)]);

        let dist1 = distance_point_to_linestring_generic(&point, &linestring);
        let dist2 = distance_linestring_to_point_generic(&linestring, &point);

        assert_relative_eq!(dist1, dist2);
        assert_relative_eq!(dist1, 2.0);
    }

    #[test]
    fn test_symmetric_distance_point_polygon() {
        let point = Point::new(5.0, 2.0);
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(exterior, vec![]);

        let dist1 = distance_point_to_polygon_generic(&point, &polygon);
        let dist2 = distance_polygon_to_point_generic(&polygon, &point);

        assert_relative_eq!(dist1, dist2);
        assert_relative_eq!(dist1, 1.0);
    }

    #[test]
    fn test_symmetric_distance_linestring_polygon() {
        let linestring = LineString::from(vec![(5.0, 1.0), (6.0, 2.0)]);
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(exterior, vec![]);

        let dist1 = distance_linestring_to_polygon_generic(&linestring, &polygon);
        let dist2 = distance_polygon_to_linestring_generic(&polygon, &linestring);

        assert_relative_eq!(dist1, dist2);
        assert_relative_eq!(dist1, 1.0);
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for line-to-linestring and line-to-polygon functions │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_distance_line_to_linestring_generic() {
        let line = Line::new(coord! { x: 0.0, y: 3.0 }, coord! { x: 2.0, y: 3.0 });
        let linestring = LineString::from(vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]);

        let distance = distance_line_to_linestring_generic(&line, &linestring);
        assert_relative_eq!(distance, 1.0); // Distance to closest segment
    }

    #[test]
    fn test_distance_line_to_polygon_generic() {
        let line = Line::new(coord! { x: 5.0, y: 1.0 }, coord! { x: 6.0, y: 2.0 });
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(exterior, vec![]);

        let distance = distance_line_to_polygon_generic(&line, &polygon);
        assert_relative_eq!(distance, 1.0);
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Tests for distance_triangle_to_point_generic               │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_distance_triangle_to_point_generic() {
        let triangle = Triangle::new(
            coord! { x: 0.0, y: 0.0 },
            coord! { x: 3.0, y: 0.0 },
            coord! { x: 1.5, y: 3.0 },
        );
        let point = Point::new(1.5, 1.0); // Inside triangle

        let distance = distance_triangle_to_point_generic(&triangle, &point);
        assert_relative_eq!(distance, 0.0);
    }

    #[test]
    fn test_distance_triangle_to_point_generic_outside() {
        let triangle = Triangle::new(
            coord! { x: 0.0, y: 0.0 },
            coord! { x: 3.0, y: 0.0 },
            coord! { x: 1.5, y: 3.0 },
        );
        let point = Point::new(5.0, 0.0); // Outside triangle

        let distance = distance_triangle_to_point_generic(&triangle, &point);
        assert_relative_eq!(distance, 2.0); // Distance to right vertex
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Edge case tests                                            │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_empty_geometries_edge_cases() {
        // Empty LineString
        let empty_ls = LineString::<f64>::new(vec![]);
        let point = Point::new(1.0, 1.0);

        let dist = distance_point_to_linestring_generic(&point, &empty_ls);
        assert_relative_eq!(dist, 0.0);

        // Empty Polygon
        let empty_poly = Polygon::new(LineString::<f64>::new(vec![]), vec![]);
        let dist2 = distance_point_to_polygon_generic(&point, &empty_poly);
        assert_relative_eq!(dist2, 0.0);
    }

    #[test]
    fn test_degenerate_geometries() {
        // Single point LineString
        let single_point_ls = LineString::from(vec![(1.0, 1.0)]);
        let point = Point::new(2.0, 2.0);

        let dist = distance_point_to_linestring_generic(&point, &single_point_ls);
        assert_relative_eq!(dist, 0.0); // Should handle gracefully

        // Two identical points in LineString
        let two_same_points_ls = LineString::from(vec![(1.0, 1.0), (1.0, 1.0)]);
        let dist2 = distance_point_to_linestring_generic(&point, &two_same_points_ls);
        assert_relative_eq!(dist2, std::f64::consts::SQRT_2); // Distance to the point
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Performance comparison tests (basic)                       │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_generic_vs_concrete_point_distance() {
        let p1 = Point::new(-72.1235, 42.3521);
        let p2 = Point::new(72.1260, 70.612);

        // Test generic implementation
        let generic_dist = distance_point_to_point_generic(&p1, &p2);

        // Test concrete implementation via Euclidean trait
        let concrete_dist = Euclidean.distance(&p1, &p2);

        // Both should give the same result
        assert_relative_eq!(generic_dist, concrete_dist, epsilon = 1e-10);
        assert_relative_eq!(generic_dist, 146.99163308930207);
    }

    #[test]
    fn test_cross_validation_with_existing_tests() {
        // Test cases from existing distance.rs tests to ensure compatibility
        let o1 = Point::new(8.0, 0.0);
        let p1 = Point::new(7.2, 2.0);
        let p2 = Point::new(6.0, 1.0);

        // Create line from p1 to p2
        let line_seg = Line::new(
            coord! { x: p1.x(), y: p1.y() },
            coord! { x: p2.x(), y: p2.y() },
        );

        if let Some(o1_coord) = o1.coord_ext() {
            let generic_dist = distance_coord_to_line_generic(&o1_coord, &line_seg);

            // This should match the expected value from the original test
            assert_relative_eq!(generic_dist, 2.0485900789263356, epsilon = 1e-10);
        }
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Property-based tests with random inputs                    │
    // └────────────────────────────────────────────────────────────┘

    fn generate_random_point(seed: u64) -> Point<f64> {
        // Simple LCG for deterministic "random" numbers
        let mut rng = seed;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x = ((rng >> 16) as i16) as f64 * 0.001;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y = ((rng >> 16) as i16) as f64 * 0.001;
        Point::new(x, y)
    }

    fn generate_random_line(seed: u64) -> Line<f64> {
        let mut rng = seed;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x1 = ((rng >> 16) as i16) as f64 * 0.001;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y1 = ((rng >> 16) as i16) as f64 * 0.001;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let x2 = ((rng >> 16) as i16) as f64 * 0.001;
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let y2 = ((rng >> 16) as i16) as f64 * 0.001;
        Line::new(coord! { x: x1, y: y1 }, coord! { x: x2, y: y2 })
    }

    fn generate_random_linestring(seed: u64, num_points: usize) -> LineString<f64> {
        let mut rng = seed;
        let mut points = Vec::new();
        for _ in 0..num_points {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let x = ((rng >> 16) as i16) as f64 * 0.001;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let y = ((rng >> 16) as i16) as f64 * 0.001;
            points.push((x, y));
        }
        LineString::from(points)
    }

    fn generate_random_polygon(seed: u64, num_exterior_points: usize) -> Polygon<f64> {
        let mut rng = seed;
        let mut points = Vec::new();

        // Generate points around a circle to ensure a valid polygon
        let center_x = 0.0;
        let center_y = 0.0;
        let radius = 10.0;

        for i in 0..num_exterior_points {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / num_exterior_points as f64;
            // Add some random noise
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((rng >> 16) as i16) as f64 * 0.0001;
            let x = center_x + (radius + noise) * angle.cos();
            let y = center_y + (radius + noise) * angle.sin();
            points.push((x, y));
        }

        // Close the polygon
        if !points.is_empty() {
            points.push(points[0]);
        }

        Polygon::new(LineString::from(points), vec![])
    }

    #[test]
    fn test_random_point_to_point_distance() {
        // Test point-to-point distance with random inputs
        for i in 0..100 {
            let seed1 = 12345 + i * 17;
            let seed2 = 54321 + i * 23;

            let p1 = generate_random_point(seed1);
            let p2 = generate_random_point(seed2);

            let concrete_dist = Euclidean.distance(&p1, &p2);
            let generic_dist = distance_point_to_point_generic(&p1, &p2);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn test_random_point_to_linestring_distance() {
        // Test point-to-linestring distance with random inputs
        for i in 0..100 {
            let seed1 = 11111 + i * 31;
            let seed2 = 22222 + i * 37;

            let point = generate_random_point(seed1);
            let linestring = generate_random_linestring(seed2, 3 + (i % 5) as usize); // 3-7 points

            let concrete_dist = Euclidean.distance(&point, &linestring);
            let generic_dist = distance_point_to_linestring_generic(&point, &linestring);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn test_random_point_to_polygon_distance() {
        // Test point-to-polygon distance with random inputs
        for i in 0..100 {
            let seed1 = 33333 + i * 41;
            let seed2 = 44444 + i * 43;

            let point = generate_random_point(seed1);
            let polygon = generate_random_polygon(seed2, 4 + (i % 4) as usize); // 4-7 sides

            let concrete_dist = Euclidean.distance(&point, &polygon);
            let generic_dist = distance_point_to_polygon_generic(&point, &polygon);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }
    }

    #[test]
    fn test_random_line_to_line_distance() {
        // Test line-to-line distance with random inputs
        for i in 0..100 {
            let seed1 = 55555 + i * 47;
            let seed2 = 66666 + i * 53;

            let line1 = generate_random_line(seed1);
            let line2 = generate_random_line(seed2);

            let concrete_dist = Euclidean.distance(&line1, &line2);
            let generic_dist = distance_line_to_line_generic(&line1, &line2);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn test_random_linestring_to_linestring_distance() {
        // Test linestring-to-linestring distance with random inputs
        for i in 0..100 {
            let seed1 = 77777 + i * 59;
            let seed2 = 88888 + i * 61;

            let ls1 = generate_random_linestring(seed1, 3 + (i % 3) as usize); // 3-5 points
            let ls2 = generate_random_linestring(seed2, 3 + ((i + 1) % 3) as usize); // 3-5 points

            let concrete_dist = Euclidean.distance(&ls1, &ls2);
            // Use our actual generic implementation via nearest_neighbour_distance
            let generic_dist = nearest_neighbour_distance(&ls1, &ls2);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }
    }

    #[test]
    fn test_random_polygon_to_polygon_distance() {
        // Test polygon-to-polygon distance with random inputs
        for i in 0..100 {
            let seed1 = 99999 + i * 67;
            let seed2 = 10101 + i * 71;

            let poly1 = generate_random_polygon(seed1, 4 + (i % 3) as usize); // 4-6 sides
            let poly2 = generate_random_polygon(seed2, 4 + ((i + 1) % 3) as usize); // 4-6 sides

            let concrete_dist = Euclidean.distance(&poly1, &poly2);
            let generic_dist = distance_polygon_to_polygon_generic(&poly1, &poly2);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-8,
                max_relative = 1e-8
            );
        }
    }

    #[test]
    fn test_random_line_to_polygon_distance() {
        // Test line-to-polygon distance with random inputs
        for i in 0..100 {
            let seed1 = 12121 + i * 73;
            let seed2 = 13131 + i * 79;

            let line = generate_random_line(seed1);
            let polygon = generate_random_polygon(seed2, 4 + (i % 3) as usize); // 4-6 sides

            let concrete_dist = Euclidean.distance(&line, &polygon);
            let generic_dist = distance_line_to_polygon_generic(&line, &polygon);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }
    }

    #[test]
    fn test_random_linestring_to_polygon_distance() {
        // Test linestring-to-polygon distance with random inputs
        for i in 0..100 {
            let seed1 = 14141 + i * 83;
            let seed2 = 15151 + i * 89;

            let linestring = generate_random_linestring(seed1, 3 + (i % 3) as usize); // 3-5 points
            let polygon = generate_random_polygon(seed2, 4 + (i % 3) as usize); // 4-6 sides

            let concrete_dist = Euclidean.distance(&linestring, &polygon);
            let generic_dist = distance_linestring_to_polygon_generic(&linestring, &polygon);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-8,
                max_relative = 1e-8
            );
        }
    }

    #[test]
    fn test_random_symmetry_properties() {
        // Test symmetry properties with random inputs
        for i in 0..100 {
            let seed1 = 16161 + i * 97;
            let seed2 = 17171 + i * 101;

            let point = generate_random_point(seed1);
            let linestring = generate_random_linestring(seed2, 4);

            // Test point-linestring symmetry
            let dist1 = distance_point_to_linestring_generic(&point, &linestring);
            let dist2 = distance_linestring_to_point_generic(&linestring, &point);
            assert_relative_eq!(dist1, dist2, epsilon = 1e-12);

            // Test with polygon
            if i % 2 == 0 {
                let polygon = generate_random_polygon(seed1 + seed2, 5);
                let dist3 = distance_point_to_polygon_generic(&point, &polygon);
                let dist4 = distance_polygon_to_point_generic(&polygon, &point);
                assert_relative_eq!(dist3, dist4, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_random_edge_cases_and_boundaries() {
        // Test edge cases with specific patterns
        for i in 0..100 {
            // Same point distance should be zero
            let point = generate_random_point(12345 + i);
            let same_point_dist = distance_point_to_point_generic(&point, &point);
            assert_relative_eq!(same_point_dist, 0.0);

            // Zero-length line segment
            let coord = coord! { x: point.x(), y: point.y() };
            let zero_line = Line::new(coord, coord);
            let dist_to_zero_line = distance_coord_to_line_generic(&coord, &zero_line);
            assert_relative_eq!(dist_to_zero_line, 0.0);

            // Point on line segment should have zero distance
            let seed = 54321 + i * 13;
            let line = generate_random_line(seed);
            let start_coord = line.start_coord();
            let dist_to_start = distance_coord_to_line_generic(&start_coord, &line);
            assert_relative_eq!(dist_to_start, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_random_large_coordinates() {
        // Test with large coordinate values to check numerical stability
        for i in 0..100 {
            let mut rng: u64 = 98765 + i * 107;

            // Generate large coordinates
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let scale = 1e6 + (rng % 1000000) as f64;

            let p1 = Point::new(scale, scale * 0.5);
            let p2 = Point::new(scale * 1.1, scale * 0.7);

            let concrete_dist = Euclidean.distance(&p1, &p2);
            let generic_dist = distance_point_to_point_generic(&p1, &p2);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-10,
                max_relative = 1e-10
            );
        }
    }

    #[test]
    fn test_random_small_coordinates() {
        // Test with very small coordinate values to check numerical precision
        for i in 0..100 {
            let mut rng: u64 = 13579 + i * 109;

            // Generate small coordinates
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let scale = 1e-6 * (1.0 + (rng % 100) as f64 * 0.01);

            let p1 = Point::new(scale, scale * 0.5);
            let p2 = Point::new(scale * 1.1, scale * 0.7);

            let concrete_dist = Euclidean.distance(&p1, &p2);
            let generic_dist = distance_point_to_point_generic(&p1, &p2);

            assert_relative_eq!(
                concrete_dist,
                generic_dist,
                epsilon = 1e-15,
                max_relative = 1e-12
            );
        }
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Geometric Edge Cases Tests                                 │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_collinear_linestring_geometries() {
        // Test linestrings where all points are collinear
        let collinear_ls1 = LineString::from(vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]);
        let collinear_ls2 = LineString::from(vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]);

        let concrete_dist = Euclidean.distance(&collinear_ls1, &collinear_ls2);
        let generic_dist = nearest_neighbour_distance(&collinear_ls1, &collinear_ls2);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-10);
        // Distance should be sqrt(2)/2 (perpendicular distance between parallel lines)
        assert_relative_eq!(
            concrete_dist,
            std::f64::consts::SQRT_2 / 2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_degenerate_triangle_as_line() {
        // Triangle where all three points are collinear (degenerate triangle)
        let degenerate_triangle = Triangle::new(
            coord! { x: 0.0, y: 0.0 },
            coord! { x: 1.0, y: 1.0 },
            coord! { x: 2.0, y: 2.0 },
        );
        let point = Point::new(0.0, 1.0);

        let concrete_dist = Euclidean.distance(&degenerate_triangle, &point);
        let generic_dist = distance_triangle_to_point_generic(&degenerate_triangle, &point);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-10);
        // Distance should be sqrt(2)/2 (distance from point to line y=x)
        assert_relative_eq!(
            concrete_dist,
            std::f64::consts::SQRT_2 / 2.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_self_intersecting_polygon() {
        // Create a bowtie/figure-8 shaped self-intersecting polygon
        let self_intersecting = LineString::from(vec![
            (0.0, 0.0),
            (2.0, 2.0),
            (2.0, 0.0),
            (0.0, 2.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(self_intersecting, vec![]);
        let point = Point::new(3.0, 1.0); // Outside the polygon

        let concrete_dist = Euclidean.distance(&point, &polygon);
        let generic_dist = distance_point_to_polygon_generic(&point, &polygon);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-10);
        assert_relative_eq!(concrete_dist, 1.0, epsilon = 1e-10); // Distance to closest edge
    }

    #[test]
    fn test_nearly_touching_geometries() {
        // Test geometries separated by very small distances
        let epsilon_dist = 1e-12;

        let line1 = Line::new(coord! { x: 0.0, y: 0.0 }, coord! { x: 1.0, y: 0.0 });
        let line2 = Line::new(
            coord! { x: 0.0, y: epsilon_dist },
            coord! { x: 1.0, y: epsilon_dist },
        );

        let concrete_dist = Euclidean.distance(&line1, &line2);
        let generic_dist = distance_line_to_line_generic(&line1, &line2);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-15);
        assert_relative_eq!(concrete_dist, epsilon_dist, epsilon = 1e-15);
    }

    #[test]
    fn test_very_close_but_separate_polygons() {
        // Two polygons separated by extremely small distance
        let tiny_gap = 1e-14;

        let poly1_exterior = LineString::from(vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]);
        let poly1 = Polygon::new(poly1_exterior, vec![]);

        let poly2_exterior = LineString::from(vec![
            (1.0 + tiny_gap, 0.0),
            (2.0 + tiny_gap, 0.0),
            (2.0 + tiny_gap, 1.0),
            (1.0 + tiny_gap, 1.0),
            (1.0 + tiny_gap, 0.0),
        ]);
        let poly2 = Polygon::new(poly2_exterior, vec![]);

        let concrete_dist = Euclidean.distance(&poly1, &poly2);
        let generic_dist = distance_polygon_to_polygon_generic(&poly1, &poly2);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-15);
        assert_relative_eq!(concrete_dist, tiny_gap, epsilon = 1e-16);
    }

    #[test]
    fn test_overlapping_but_not_intersecting_linestrings() {
        // LineStrings that overlap in projection but are at different heights
        let ls1 = LineString::from(vec![(0.0, 0.0), (2.0, 0.0)]);
        let ls2 = LineString::from(vec![(1.0, 1e-13), (3.0, 1e-13)]);

        let concrete_dist = Euclidean.distance(&ls1, &ls2);
        let generic_dist = nearest_neighbour_distance(&ls1, &ls2);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-15);
        assert_relative_eq!(concrete_dist, 1e-13, epsilon = 1e-16);
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Numerical Precision Tests                                  │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_very_close_but_non_zero_distances() {
        // Test extremely small but non-zero distances to check floating-point precision
        let test_cases = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10];

        for &tiny_dist in &test_cases {
            let p1 = Point::new(0.0, 0.0);
            let p2 = Point::new(tiny_dist, 0.0);

            let concrete_dist = Euclidean.distance(&p1, &p2);
            let generic_dist = distance_point_to_point_generic(&p1, &p2);

            assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-16);
            assert_relative_eq!(concrete_dist, tiny_dist, epsilon = 1e-16);
            assert!(
                concrete_dist > 0.0,
                "Distance should be positive for tiny_dist = {tiny_dist}"
            );
        }
    }

    #[test]
    fn test_numerical_precision_near_floating_point_limits() {
        // Test with coordinates that produce distances near floating-point precision limits
        let base = 1.0;
        let tiny_offset = f64::EPSILON * 10.0; // Slightly above machine epsilon

        let p1 = Point::new(base, base);
        let p2 = Point::new(base + tiny_offset, base);

        let concrete_dist = Euclidean.distance(&p1, &p2);
        let generic_dist = distance_point_to_point_generic(&p1, &p2);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-15);
        assert!(concrete_dist > 0.0);
        assert!(concrete_dist < 1e-14); // Should be very small but measurable
    }

    #[test]
    fn test_precision_with_large_coordinate_differences() {
        // Test with one geometry having small coordinates and another having large coordinates
        let small_point = Point::new(1e-10, 1e-10);
        let large_polygon = Polygon::new(
            LineString::from(vec![
                (1e8, 1e8),
                (1e8 + 1.0, 1e8),
                (1e8 + 1.0, 1e8 + 1.0),
                (1e8, 1e8 + 1.0),
                (1e8, 1e8),
            ]),
            vec![],
        );

        let concrete_dist = Euclidean.distance(&small_point, &large_polygon);
        let generic_dist = distance_point_to_polygon_generic(&small_point, &large_polygon);

        assert_relative_eq!(concrete_dist, generic_dist, max_relative = 1e-10);
        assert!(concrete_dist > 1e7); // Should be very large distance
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Robustness Tests                                           │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_nan_coordinate_handling() {
        // Test behavior with NaN coordinates
        let nan_point = Point::new(f64::NAN, 0.0);
        let normal_point = Point::new(1.0, 1.0);

        let distance = distance_point_to_point_generic(&nan_point, &normal_point);

        // Distance involving NaN should be NaN
        assert!(
            distance.is_nan(),
            "Distance with NaN coordinate should be NaN"
        );
    }

    #[test]
    fn test_infinity_coordinate_handling() {
        // Test behavior with infinite coordinates
        let inf_point = Point::new(f64::INFINITY, 0.0);
        let normal_point = Point::new(1.0, 1.0);

        let distance = distance_point_to_point_generic(&inf_point, &normal_point);

        // Distance involving infinity should be infinity
        assert!(
            distance.is_infinite(),
            "Distance with infinite coordinate should be infinite"
        );
    }

    #[test]
    fn test_negative_infinity_coordinate_handling() {
        // Test behavior with negative infinite coordinates
        let neg_inf_point = Point::new(f64::NEG_INFINITY, 0.0);
        let normal_point = Point::new(1.0, 1.0);

        let distance = distance_point_to_point_generic(&neg_inf_point, &normal_point);

        // Distance involving negative infinity should be infinity
        assert!(
            distance.is_infinite(),
            "Distance with negative infinite coordinate should be infinite"
        );
    }

    #[test]
    fn test_mixed_special_values() {
        // Test combinations of NaN and infinity
        let nan_point = Point::new(f64::NAN, f64::INFINITY);
        let inf_point = Point::new(f64::INFINITY, f64::NEG_INFINITY);

        let distance = distance_point_to_point_generic(&nan_point, &inf_point);

        // Any operation involving NaN should result in NaN or Infinity depending on the math
        // Since we're using hypot which can handle NaN differently, let's test that it's either NaN or infinite
        assert!(
            distance.is_nan() || distance.is_infinite(),
            "Distance involving NaN and Infinity should be NaN or Infinite, got: {distance}"
        );
    }

    #[test]
    fn test_subnormal_number_handling() {
        // Test with subnormal (denormalized) numbers
        let subnormal = f64::MIN_POSITIVE / 2.0; // This creates a subnormal number
        assert!(subnormal > 0.0 && subnormal < f64::MIN_POSITIVE);

        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(subnormal, 0.0);

        let concrete_dist = Euclidean.distance(&p1, &p2);
        let generic_dist = distance_point_to_point_generic(&p1, &p2);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-16);
        assert_relative_eq!(concrete_dist, subnormal, epsilon = 1e-16);
        assert!(concrete_dist > 0.0);
    }

    #[test]
    fn test_zero_vs_negative_zero() {
        // Test behavior with positive zero vs negative zero
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(-0.0, -0.0); // Negative zero

        let distance = distance_point_to_point_generic(&p1, &p2);

        // Distance between +0 and -0 should be exactly 0
        assert_eq!(
            distance, 0.0,
            "Distance between +0 and -0 should be exactly 0"
        );
    }

    // ┌────────────────────────────────────────────────────────────┐
    // │ Algorithmic Correctness Validation Tests                   │
    // └────────────────────────────────────────────────────────────┘

    #[test]
    fn test_linestring_inside_polygon_with_holes_correctness() {
        // This test exposes the algorithmic difference between generic and concrete implementations

        // Create a polygon with a hole
        let outer = LineString::from(vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let hole = LineString::from(vec![
            (3.0, 3.0),
            (7.0, 3.0),
            (7.0, 7.0),
            (3.0, 7.0),
            (3.0, 3.0),
        ]);
        let polygon = Polygon::new(outer, vec![hole]);

        // LineString that is INSIDE the polygon but OUTSIDE the hole
        let linestring_inside = LineString::from(vec![(1.0, 1.0), (2.0, 2.0)]);

        let concrete_dist = Euclidean.distance(&linestring_inside, &polygon);
        let generic_dist = distance_linestring_to_polygon_generic(&linestring_inside, &polygon);

        // The results should be identical
        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_linestring_outside_polygon_with_holes_correctness() {
        // Test case where LineString is completely outside the polygon

        let outer = LineString::from(vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let hole = LineString::from(vec![
            (3.0, 3.0),
            (7.0, 3.0),
            (7.0, 7.0),
            (3.0, 7.0),
            (3.0, 3.0),
        ]);
        let polygon = Polygon::new(outer, vec![hole]);

        // LineString that is OUTSIDE the polygon entirely
        let linestring_outside = LineString::from(vec![(12.0, 12.0), (13.0, 13.0)]);

        let concrete_dist = Euclidean.distance(&linestring_outside, &polygon);
        let generic_dist = distance_linestring_to_polygon_generic(&linestring_outside, &polygon);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_linestring_crossing_polygon_boundary_correctness() {
        // Test case where LineString crosses the polygon boundary

        let outer = LineString::from(vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let polygon = Polygon::new(outer, vec![]);

        // LineString that crosses the polygon boundary (should intersect)
        let linestring_crossing = LineString::from(vec![(-1.0, 5.0), (11.0, 5.0)]);

        let concrete_dist = Euclidean.distance(&linestring_crossing, &polygon);
        let generic_dist = distance_linestring_to_polygon_generic(&linestring_crossing, &polygon);

        // Both should be 0.0 since they intersect
        assert_eq!(
            concrete_dist, 0.0,
            "Concrete should return 0 for intersecting geometries"
        );
        assert_eq!(
            generic_dist, 0.0,
            "Generic should return 0 for intersecting geometries"
        );

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-10);
    }

    #[test]
    fn test_containment_logic_specific() {
        // This test specifically checks the containment logic for polygons with holes
        use geo_types::{LineString, Polygon};

        // Create a larger polygon with a hole
        let exterior = LineString::from(vec![
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 20.0),
            (0.0, 20.0),
            (0.0, 0.0),
        ]);
        let hole = LineString::from(vec![
            (8.0, 8.0),
            (12.0, 8.0),
            (12.0, 12.0),
            (8.0, 12.0),
            (8.0, 8.0),
        ]);
        let polygon = Polygon::new(exterior, vec![hole]);

        // LineString that is INSIDE the polygon but OUTSIDE the hole,
        // Create a small LineString very close to itself to avoid intersection
        let inside_linestring = LineString::from(vec![(5.0, 5.0), (5.1, 5.1)]);

        let concrete_distance = Euclidean.distance(&inside_linestring, &polygon);
        let generic_distance = distance_linestring_to_polygon_generic(&inside_linestring, &polygon);

        // Check if LineString actually intersects with the polygon
        use crate::algorithm::Intersects;
        let _does_intersect = inside_linestring.intersects(&polygon);

        assert_relative_eq!(concrete_distance, generic_distance, epsilon = 1e-10);
    }

    #[test]
    fn test_polygon_to_polygon_symmetric_containment_correctness() {
        // Test that both A contains B and B contains A cases work correctly
        use geo_types::{LineString, Polygon};

        // Case 1: Large polygon with hole contains small polygon
        let large_exterior = LineString::from(vec![
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 20.0),
            (0.0, 20.0),
            (0.0, 0.0),
        ]);
        let large_hole = LineString::from(vec![
            (8.0, 8.0),
            (12.0, 8.0),
            (12.0, 12.0),
            (8.0, 12.0),
            (8.0, 8.0),
        ]);
        let large_polygon = Polygon::new(large_exterior, vec![large_hole]);

        // Small polygon inside the large polygon (but outside the hole)
        let small_exterior = LineString::from(vec![
            (2.0, 2.0),
            (6.0, 2.0),
            (6.0, 6.0),
            (2.0, 6.0),
            (2.0, 2.0),
        ]);
        let small_polygon = Polygon::new(small_exterior, vec![]);

        // Test A contains B: large polygon with hole contains small polygon
        let concrete_dist_ab = Euclidean.distance(&small_polygon, &large_polygon);
        let generic_dist_ab = distance_polygon_to_polygon_generic(&small_polygon, &large_polygon);

        // Test B contains A: small polygon contains large polygon (should be distance between exteriors)
        let concrete_dist_ba = Euclidean.distance(&large_polygon, &small_polygon);
        let generic_dist_ba = distance_polygon_to_polygon_generic(&large_polygon, &small_polygon);

        // Both directions should match between concrete and generic
        assert_relative_eq!(concrete_dist_ab, generic_dist_ab, epsilon = 1e-10);
        assert_relative_eq!(concrete_dist_ba, generic_dist_ba, epsilon = 1e-10);

        // The distances should be the same due to symmetry
        assert_relative_eq!(concrete_dist_ab, concrete_dist_ba, epsilon = 1e-10);
        assert_relative_eq!(generic_dist_ab, generic_dist_ba, epsilon = 1e-10);
    }

    #[test]
    fn test_polygon_to_polygon_both_have_holes_correctness() {
        // Test case where both polygons have holes
        use geo_types::{LineString, Polygon};

        // Polygon A with hole
        let exterior_a = LineString::from(vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]);
        let hole_a = LineString::from(vec![
            (3.0, 3.0),
            (7.0, 3.0),
            (7.0, 7.0),
            (3.0, 7.0),
            (3.0, 3.0),
        ]);
        let polygon_a = Polygon::new(exterior_a, vec![hole_a]);

        // Polygon B with hole (separate from A)
        let exterior_b = LineString::from(vec![
            (15.0, 0.0),
            (25.0, 0.0),
            (25.0, 10.0),
            (15.0, 10.0),
            (15.0, 0.0),
        ]);
        let hole_b = LineString::from(vec![
            (18.0, 3.0),
            (22.0, 3.0),
            (22.0, 7.0),
            (18.0, 7.0),
            (18.0, 3.0),
        ]);
        let polygon_b = Polygon::new(exterior_b, vec![hole_b]);

        // Neither polygon contains the other, so should calculate distance between exteriors
        let concrete_dist = Euclidean.distance(&polygon_a, &polygon_b);
        let generic_dist = distance_polygon_to_polygon_generic(&polygon_a, &polygon_b);

        assert_relative_eq!(concrete_dist, generic_dist, epsilon = 1e-10);

        // Test symmetry
        let concrete_dist_reverse = Euclidean.distance(&polygon_b, &polygon_a);
        let generic_dist_reverse = distance_polygon_to_polygon_generic(&polygon_b, &polygon_a);

        assert_relative_eq!(concrete_dist_reverse, generic_dist_reverse, epsilon = 1e-10);
        assert_relative_eq!(concrete_dist, concrete_dist_reverse, epsilon = 1e-10);
    }

    #[test]
    fn test_point_to_linestring_containment_optimization() {
        // Test that the containment check optimization works correctly
        use geo_types::{LineString, Point};

        // Create a LineString
        let linestring = LineString::from(vec![(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (10.0, 5.0)]);

        // Point ON the LineString (should return 0 due to containment check)
        let point_on_line = Point::new(2.5, 0.0); // On first segment
        let concrete_dist_on = Euclidean.distance(&point_on_line, &linestring);
        let generic_dist_on = distance_point_to_linestring_generic(&point_on_line, &linestring);

        // Both should be exactly 0 due to containment
        assert_eq!(concrete_dist_on, 0.0);
        assert_eq!(generic_dist_on, 0.0);
        assert_relative_eq!(concrete_dist_on, generic_dist_on, epsilon = 1e-10);

        // Point ON a vertex (should return 0)
        let point_on_vertex = Point::new(5.0, 0.0);
        let concrete_dist_vertex = Euclidean.distance(&point_on_vertex, &linestring);
        let generic_dist_vertex =
            distance_point_to_linestring_generic(&point_on_vertex, &linestring);

        assert_eq!(concrete_dist_vertex, 0.0);
        assert_eq!(generic_dist_vertex, 0.0);
        assert_relative_eq!(concrete_dist_vertex, generic_dist_vertex, epsilon = 1e-10);

        // Point NOT on the LineString (should calculate actual distance)
        let point_off_line = Point::new(2.5, 3.0);
        let concrete_dist_off = Euclidean.distance(&point_off_line, &linestring);
        let generic_dist_off = distance_point_to_linestring_generic(&point_off_line, &linestring);

        // Should be greater than 0 and both implementations should match
        assert!(concrete_dist_off > 0.0);
        assert!(generic_dist_off > 0.0);
        assert_relative_eq!(concrete_dist_off, generic_dist_off, epsilon = 1e-10);
    }

    #[test]
    fn test_line_segment_distance_algorithm_equivalence() {
        // Test that the updated generic algorithm produces identical results to concrete
        use geo_types::{coord, Line, Point};

        // Test cases covering different scenarios
        let test_cases = vec![
            // Point, Line start, Line end
            (
                coord! { x: 0.0, y: 0.0 },
                coord! { x: 1.0, y: 0.0 },
                coord! { x: 3.0, y: 0.0 },
            ), // Before start
            (
                coord! { x: 2.0, y: 1.0 },
                coord! { x: 1.0, y: 0.0 },
                coord! { x: 3.0, y: 0.0 },
            ), // Perpendicular
            (
                coord! { x: 4.0, y: 0.0 },
                coord! { x: 1.0, y: 0.0 },
                coord! { x: 3.0, y: 0.0 },
            ), // Beyond end
            (
                coord! { x: 2.0, y: 0.0 },
                coord! { x: 1.0, y: 0.0 },
                coord! { x: 3.0, y: 0.0 },
            ), // On line
            (
                coord! { x: 1.0, y: 0.0 },
                coord! { x: 1.0, y: 0.0 },
                coord! { x: 3.0, y: 0.0 },
            ), // On start point
            (
                coord! { x: 3.0, y: 0.0 },
                coord! { x: 1.0, y: 0.0 },
                coord! { x: 3.0, y: 0.0 },
            ), // On end point
            (
                coord! { x: 0.0, y: 0.0 },
                coord! { x: 1.0, y: 1.0 },
                coord! { x: 1.0, y: 1.0 },
            ), // Degenerate line
            (
                coord! { x: 2.5, y: 3.0 },
                coord! { x: 0.0, y: 0.0 },
                coord! { x: 5.0, y: 5.0 },
            ), // Diagonal line
        ];

        for (point_coord, start_coord, end_coord) in test_cases {
            let point = Point::from(point_coord);
            let line = Line::new(start_coord, end_coord);

            // Test concrete implementation
            let concrete_distance = Euclidean.distance(&point, &line);

            // Test generic implementation
            let generic_distance = distance_coord_to_line_generic(&point_coord, &line);

            // They should be identical now
            assert_relative_eq!(concrete_distance, generic_distance, epsilon = 1e-15);
        }
    }
}
