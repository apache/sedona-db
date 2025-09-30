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
use super::{CoordNum, Kernel, Orientation};
use crate::Coord;

use num_traits::{Float, NumCast};

/// Robust kernel that uses [fast robust
/// predicates](//www.cs.cmu.edu/~quake/robust.html) to
/// provide robust floating point predicates. Should only be
/// used with types that can _always_ be casted to `f64`
/// _without loss in precision_.
#[derive(Default, Debug)]
pub struct RobustKernel;

impl<T> Kernel<T> for RobustKernel
where
    T: CoordNum + Float,
{
    fn orient2d(p: Coord<T>, q: Coord<T>, r: Coord<T>) -> Orientation {
        use robust::{orient2d, Coord};

        let orientation = orient2d(
            Coord {
                x: <f64 as NumCast>::from(p.x).unwrap(),
                y: <f64 as NumCast>::from(p.y).unwrap(),
            },
            Coord {
                x: <f64 as NumCast>::from(q.x).unwrap(),
                y: <f64 as NumCast>::from(q.y).unwrap(),
            },
            Coord {
                x: <f64 as NumCast>::from(r.x).unwrap(),
                y: <f64 as NumCast>::from(r.y).unwrap(),
            },
        );

        if orientation < 0. {
            Orientation::Clockwise
        } else if orientation > 0. {
            Orientation::CounterClockwise
        } else {
            Orientation::Collinear
        }
    }
}
