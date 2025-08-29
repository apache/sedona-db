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
use serde::{Deserialize, Serialize};

use crate::interval::{Interval, IntervalTrait, WraparoundInterval};

/// Bounding Box implementation with wraparound support
///
/// Conceptually, this BoundingBox is a [WraparoundInterval] (x), an
/// [Interval] (y), and optional [Interval]s for z and m. This BoundingBox
/// intentionally separates the case where no information was provided
/// (i.e., there is no information about the presence or absence of values
/// in a given dimension) and [Interval::empty] (i.e., we are absolutely
/// and positively sure there are zero values present for a given dimension).
/// If in doubt, it is safer to use `None` for z and m because their missingness
/// is propagated when merging and calculating a potential intersection.
///
/// This structure implements Serialize and Deserialize to support passing
/// it between query engine components where there is not yet a mechanism
/// to do so.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    x: WraparoundInterval,
    y: Interval,
    z: Option<Interval>,
    m: Option<Interval>,
}

impl BoundingBox {
    /// Create a BoundingBox with unspecified z and m intervals
    pub fn xy(x: impl Into<WraparoundInterval>, y: impl Into<Interval>) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z: None,
            m: None,
        }
    }

    /// Create a BoundingBox from intervals by dimension
    pub fn xyzm(
        x: impl Into<WraparoundInterval>,
        y: impl Into<Interval>,
        z: Option<Interval>,
        m: Option<Interval>,
    ) -> Self {
        Self {
            x: x.into(),
            y: y.into(),
            z,
            m,
        }
    }

    /// The x interval
    pub fn x(&self) -> &WraparoundInterval {
        &self.x
    }

    /// The y interval
    pub fn y(&self) -> &Interval {
        &self.y
    }

    /// The z interval if any information is known (or `None` if the presence, absence, or
    /// content of the Z dimension is not known)
    pub fn z(&self) -> &Option<Interval> {
        &self.z
    }

    /// The M interval if any information is known (or `None` if the presence, absence, or
    /// content of the M dimension is not known)
    pub fn m(&self) -> &Option<Interval> {
        &self.m
    }

    /// Calculate intersection with another BoundingBox
    ///
    /// Returns true if this bounding box may intersect other or false otherwise. This
    /// method will consider Z and M dimension if and only if those dimensions are present
    /// in both bounding boxes.
    pub fn intersects(&self, other: &Self) -> bool {
        let intersects_xy =
            self.x.intersects_interval(&other.x) && self.y.intersects_interval(&other.y);
        let may_intersect_z = match (self.z, other.z) {
            (Some(z), Some(other_z)) => z.intersects_interval(&other_z),
            _ => true,
        };
        let may_intersect_m = match (self.m, other.m) {
            (Some(m), Some(other_m)) => m.intersects_interval(&other_m),
            _ => true,
        };

        intersects_xy && may_intersect_z && may_intersect_m
    }

    /// Update this BoundingBox to include the bounds of another
    ///
    /// This method will propagate missingness of Z or M dimensions from the two boxes
    /// (e.g., Z will be `None` if Z if `self.z().is_none()` OR `other.z().is_none()`).
    /// Note that this method is intended for accumulating bounds at the file level and
    /// is not performant for accumulating bounds for individual geometries. For this case, use
    /// a set of [Interval]s, (perhaps merging them into [WraparoundInterval]s at the
    /// geometry or array level if working with longitudes and latitudes and the performance
    /// overhead is acceptable).
    pub fn update_box(&mut self, other: &Self) {
        self.x = self.x.merge_interval(&other.x);
        self.y = self.y.merge_interval(&other.y);
        self.z = match (self.z, other.z) {
            (Some(z), Some(other_z)) => Some(z.merge_interval(&other_z)),
            _ => None,
        };
        self.m = match (self.m, other.m) {
            (Some(m), Some(other_m)) => Some(m.merge_interval(&other_m)),
            _ => None,
        };
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bounding_box_intersects() {
        let xyzm = BoundingBox::xyzm(
            (10, 20),
            (30, 40),
            Some((50, 60).into()),
            Some((70, 80).into()),
        );
        assert_eq!(xyzm.x(), &WraparoundInterval::new(10.0, 20.0));
        assert_eq!(xyzm.y(), &Interval::new(30.0, 40.0));
        assert_eq!(xyzm.z(), &Some(Interval::new(50.0, 60.0)));
        assert_eq!(xyzm.m(), &Some(Interval::new(70.0, 80.0)));

        // Should intersect a box without z or m information
        assert!(xyzm.intersects(&BoundingBox::xy((14, 16), (34, 36))));

        // Should intersect without z information but with intersecting m
        assert!(xyzm.intersects(&BoundingBox::xyzm(
            (14, 16),
            (34, 36),
            None,
            Some((74, 76).into())
        )));

        // Should intersect without z information but with intersecting m
        assert!(xyzm.intersects(&BoundingBox::xyzm(
            (14, 16),
            (34, 36),
            Some((54, 56).into()),
            None,
        )));

        // Should *not* intersect if x or y is disjoint
        assert!(!xyzm.intersects(&BoundingBox::xy((4, 6), (34, 36))));
        assert!(!xyzm.intersects(&BoundingBox::xy((14, 16), (24, 26))));

        // Should *not* intersect if z is provided but is disjoint
        assert!(!xyzm.intersects(&BoundingBox::xyzm(
            (14, 16),
            (34, 36),
            Some((44, 46).into()),
            None
        )));

        // Should *not* intersect if m is provided but is disjoint
        assert!(!xyzm.intersects(&BoundingBox::xyzm(
            (14, 16),
            (34, 36),
            None,
            Some((64, 66).into())
        )));
    }

    #[test]
    fn bounding_box_update() {
        let xyzm = BoundingBox::xyzm(
            (10, 20),
            (30, 40),
            Some((50, 60).into()),
            Some((70, 80).into()),
        );

        let empty = BoundingBox::xyzm(
            Interval::empty(),
            Interval::empty(),
            Some(Interval::empty()),
            Some(Interval::empty()),
        );

        let mut bounding_box = empty.clone();

        // Update with empty should still be empty
        bounding_box.update_box(&bounding_box.clone());
        assert_eq!(bounding_box, bounding_box);

        // Update empty with finite should be finite
        bounding_box.update_box(&xyzm);
        assert_eq!(bounding_box, xyzm);

        // Update finite with empty should be unchanged
        bounding_box.update_box(&empty);
        assert_eq!(bounding_box, xyzm);

        // Update with a box that has unspecified z should also have unspecified z
        bounding_box.update_box(&BoundingBox::xyzm(
            Interval::empty(),
            Interval::empty(),
            None,
            Some(Interval::empty()),
        ));
        assert_eq!(bounding_box.x(), xyzm.x());
        assert_eq!(bounding_box.y(), xyzm.y());
        assert!(bounding_box.z().is_none());
        assert_eq!(bounding_box.m(), xyzm.m());

        // Update with a box that has unspecified m should also have unspecified m
        bounding_box.update_box(&BoundingBox::xyzm(
            Interval::empty(),
            Interval::empty(),
            Some(Interval::empty()),
            None,
        ));
        assert_eq!(bounding_box.x(), xyzm.x());
        assert_eq!(bounding_box.y(), xyzm.y());
        assert!(bounding_box.z().is_none());
        assert!(bounding_box.m().is_none());
    }

    fn check_serialize_deserialize_roundtrip(bounding_box: BoundingBox) {
        let json_bytes = serde_json::to_vec(&bounding_box).unwrap();
        let bounding_box_roundtrip: BoundingBox = serde_json::from_slice(&json_bytes).unwrap();
        assert_eq!(bounding_box, bounding_box_roundtrip)
    }

    #[test]
    fn serialize_deserialize() {
        // All finite
        check_serialize_deserialize_roundtrip(BoundingBox::xyzm(
            (10, 20),
            (30, 40),
            Some((50, 60).into()),
            Some((70, 80).into()),
        ));

        // Missing m
        check_serialize_deserialize_roundtrip(BoundingBox::xyzm(
            (10, 20),
            (30, 40),
            Some((50, 60).into()),
            None,
        ));

        // Missing z
        check_serialize_deserialize_roundtrip(BoundingBox::xyzm(
            (10, 20),
            (30, 40),
            None,
            Some((70, 80).into()),
        ));

        // Missing z and m
        check_serialize_deserialize_roundtrip(BoundingBox::xy((10, 20), (30, 40)));

        // Empty x
        check_serialize_deserialize_roundtrip(BoundingBox::xy((10, 20), Interval::empty()));

        // Empty y
        check_serialize_deserialize_roundtrip(BoundingBox::xy(Interval::empty(), (30, 40)));

        // Arbitrary precision floating point values should roundtrip
        check_serialize_deserialize_roundtrip(BoundingBox::xy(
            (10.0 / 17.0, 20.0 / 13.0),
            (30.0 / 11.0, 40.0 / 7.0),
        ));

        // NaN values should survive roundtrip (although we can't use == to test because NAN != NAN)
        let bbox_nan = BoundingBox::xy((f64::NAN, f64::NAN), (f64::NAN, f64::NAN));
        let json_bytes = serde_json::to_vec(&bbox_nan).unwrap();
        let bbox_nan2: BoundingBox = serde_json::from_slice(&json_bytes).unwrap();
        assert!(bbox_nan2.x().lo().is_nan());
        assert!(bbox_nan2.x().hi().is_nan());
        assert!(bbox_nan2.y().lo().is_nan());
        assert!(bbox_nan2.y().hi().is_nan());
    }
}
