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

//! Shared helpers for partitioner benchmarks.

use rand::{rngs::StdRng, Rng, SeedableRng};
use sedona_geometry::{bounding_box::BoundingBox, interval::IntervalTrait};

pub const GRID_DIM: usize = 4; // 4x4 grid => 16 partitions like typical workloads
pub const QUERY_BATCH_SIZE: usize = 1_024;
pub const RNG_SEED: u64 = 0x5ED0_4A7E;

pub fn default_extent() -> BoundingBox {
    BoundingBox::xy((0.0, 10_000.0), (0.0, 10_000.0))
}

pub fn grid_partitions(extent: &BoundingBox, cells_per_axis: usize) -> Vec<BoundingBox> {
    let min_x = extent.x().lo();
    let max_x = extent.x().hi();
    let min_y = extent.y().lo();
    let max_y = extent.y().hi();
    let cell_width = (max_x - min_x) / cells_per_axis as f64;
    let cell_height = (max_y - min_y) / cells_per_axis as f64;

    let mut partitions = Vec::with_capacity(cells_per_axis * cells_per_axis);
    for ix in 0..cells_per_axis {
        for iy in 0..cells_per_axis {
            let x0 = min_x + ix as f64 * cell_width;
            let y0 = min_y + iy as f64 * cell_height;
            let x1 = if ix + 1 == cells_per_axis {
                max_x
            } else {
                x0 + cell_width
            };
            let y1 = if iy + 1 == cells_per_axis {
                max_y
            } else {
                y0 + cell_height
            };
            partitions.push(BoundingBox::xy((x0, x1), (y0, y1)));
        }
    }

    partitions
}

pub fn sample_queries(extent: &BoundingBox, batch_size: usize) -> Vec<BoundingBox> {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let characteristic_span = extent_span(extent) / 8.0;
    (0..batch_size)
        .map(|_| random_bbox(extent, &mut rng, characteristic_span))
        .collect()
}

fn random_bbox(extent: &BoundingBox, rng: &mut StdRng, max_span: f64) -> BoundingBox {
    let (min_x, max_x) = (extent.x().lo(), extent.x().hi());
    let (min_y, max_y) = (extent.y().lo(), extent.y().hi());

    let span_x = rng.gen_range(0.01..max_span).min(max_x - min_x);
    let span_y = rng.gen_range(0.01..max_span).min(max_y - min_y);

    let start_x = rng.gen_range(min_x..=max_x - span_x);
    let start_y = rng.gen_range(min_y..=max_y - span_y);

    BoundingBox::xy((start_x, start_x + span_x), (start_y, start_y + span_y))
}

fn extent_span(extent: &BoundingBox) -> f64 {
    let width = extent.x().hi() - extent.x().lo();
    let height = extent.y().hi() - extent.y().lo();
    width.min(height)
}
