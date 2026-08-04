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

//! Linear interpolation helpers shared by geometry densification paths
//! (`ST_Segmentize`, reprojected raster footprints).

/// Linearly interpolate between `a` and `b` at parameter `t`.
///
/// `lerp(a, b, 0.0) == a` and `lerp(a, b, 1.0) == b`. This is the single-scalar
/// primitive behind every densification path so the interpolation math is not
/// re-derived per caller (per-dimension `ST_Segmentize` interpolation, per-axis
/// footprint densification).
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Append `n` evenly spaced interior points of the segment from `start` to `end`
/// to `out` — the points at parameters `t = 1/(n+1), 2/(n+1), …, n/(n+1)`.
///
/// Neither endpoint is emitted; the caller owns `start` and `end`. Densifying a
/// polygon edge this way makes the polygon follow the curved image of that
/// straight edge under a nonlinear reprojection instead of chording across it.
/// `out` is appended to (not cleared) so a ring can be assembled edge by edge
/// into a caller-managed scratch buffer.
pub fn densify_segment(start: (f64, f64), end: (f64, f64), n: usize, out: &mut Vec<(f64, f64)>) {
    for i in 1..=n {
        let t = i as f64 / (n + 1) as f64;
        out.push((lerp(start.0, end.0, t), lerp(start.1, end.1, t)));
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn lerp_hits_endpoints_and_midpoint() {
        assert_eq!(lerp(2.0, 6.0, 0.0), 2.0);
        assert_eq!(lerp(2.0, 6.0, 1.0), 6.0);
        assert_eq!(lerp(2.0, 6.0, 0.5), 4.0);
        assert_eq!(lerp(2.0, 6.0, 0.25), 3.0);
    }

    #[test]
    fn densify_segment_emits_evenly_spaced_interior_points() {
        let mut out = Vec::new();
        densify_segment((0.0, 0.0), (4.0, 8.0), 3, &mut out);
        // Interior points at t = 1/4, 2/4, 3/4 (endpoints excluded).
        assert_eq!(out, vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]);
    }

    #[test]
    fn densify_segment_zero_points_is_noop() {
        let mut out = vec![(9.0, 9.0)];
        densify_segment((0.0, 0.0), (1.0, 1.0), 0, &mut out);
        assert_eq!(out, vec![(9.0, 9.0)]);
    }
}
