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

#![allow(unused)]
use datafusion_common::{DataFusionError, Result};
use fastrand::Rng;
use sedona_geometry::bounding_box::BoundingBox;

/// Multi-stage bounding box sampling algorithm for spatial join partitioning.
///
/// # Purpose
///
/// This sampler is designed to collect a representative sample of bounding boxes from a large
/// spatial dataset in a single pass, without knowing the total count beforehand. The samples
/// are used to build high-quality spatial partitioning grids for out-of-core spatial joins,
/// where datasets may be too large to fit entirely in memory.
///
/// # Goals
///
/// 1. **Uniform Sampling**: Collect samples that faithfully represent the spatial distribution
///    of the entire dataset, ensuring good partitioning quality.
///
/// 2. **Memory Bounded**: Never exceed a maximum sample count to avoid running out of memory,
///    even for very large datasets.
///
/// 3. **Minimum Sample Guarantee**: Collect at least a minimum number of samples to ensure
///    partitioning quality, even for small datasets.
///
/// 4. **Single Pass**: Collect samples in one pass without requiring prior knowledge of the
///    dataset size, which is critical for avoiding expensive double-scans in out-of-core scenarios.
///
/// 5. **Target Sampling Rate**: Maintain samples at a target sampling rate before hitting the maximum
///    sample count, balancing between sample quality and memory usage.
///
/// # Algorithm Stages
///
/// The algorithm uses a combination of reservoir sampling and Bernoulli sampling across 4 stages:
///
/// 1. **Stage 1 - Filling the small reservoir** (k < min_samples):
///    Simply collect all bounding boxes to ensure we have enough samples for small datasets.
///
/// 2. **Stage 2 - Small reservoir sampling** (min_samples ≤ k < min_samples / target_sampling_rate):
///    Use reservoir sampling to maintain exactly min_samples samples while keeping the sampling
///    rate above the target.
///
/// 3. **Stage 3 - Bernoulli sampling** (k ≥ min_samples / target_sampling_rate && |samples| < max_samples):
///    The reservoir can no longer guarantee the target sampling rate. Use Bernoulli sampling
///    at the target rate to grow the sample set.
///
/// 4. **Stage 4 - Large reservoir sampling** (|samples| = max_samples):
///    We've hit the memory limit. Use reservoir sampling to maintain exactly max_samples,
///    preventing memory overflow.
///
/// This multi-stage approach ensures uniform sampling across all stages while satisfying all
/// the goals above.
#[derive(Debug)]
pub(crate) struct BoundingBoxSampler {
    /// Minimum number of samples to collect
    min_samples: usize,

    /// Maximum number of samples to collect
    max_samples: usize,

    /// Target sampling rate (minimum sampling rate before hitting max_samples)
    target_sampling_rate: f64,

    /// The threshold count for switching from stage 2 to stage 3
    /// (min_samples / target_sampling_rate)
    reservoir_sampling_max_count: usize,

    /// The collected samples
    samples: Vec<BoundingBox>,

    /// The number of bounding boxes seen so far
    population_count: usize,

    /// Random number generator (fast, non-cryptographic)
    rng: Rng,
}

/// Samples collected by [`BoundingBoxSampler`].
#[derive(Debug, PartialEq, Clone, Default)]
pub(crate) struct BoundingBoxSamples {
    /// The collected samples
    samples: Vec<BoundingBox>,

    /// The size of population where the samples were collected from
    population_count: usize,
}

impl BoundingBoxSampler {
    /// Create a new [`BoundingBoxSampler`]
    ///
    /// # Arguments
    /// * `min_samples` - Minimum number of samples to collect (corresponds to minNumSamples in Java)
    /// * `max_samples` - Maximum number of samples to collect (corresponds to maxNumSamples in Java)
    /// * `target_sampling_rate` - Target sampling rate (corresponds to minSamplingRate in Java)
    /// * `seed` - Seed for the random number generator to ensure reproducible sampling
    ///
    /// # Errors
    /// Returns an error if:
    /// - `min_samples` is 0
    /// - `max_samples` is less than `min_samples`
    /// - `target_sampling_rate` is not in the range (0, 1]
    pub fn try_new(
        min_samples: usize,
        max_samples: usize,
        target_sampling_rate: f64,
        seed: u64,
    ) -> Result<Self> {
        if min_samples == 0 {
            return Err(DataFusionError::Plan(
                "min_samples must be positive".to_string(),
            ));
        }
        if max_samples < min_samples {
            return Err(DataFusionError::Plan(
                "max_samples must be >= min_samples".to_string(),
            ));
        }
        if target_sampling_rate <= 0.0 || target_sampling_rate > 1.0 {
            return Err(DataFusionError::Plan(
                "target_sampling_rate must be in (0, 1]".to_string(),
            ));
        }

        let reservoir_sampling_max_count = (min_samples as f64 / target_sampling_rate) as usize;

        Ok(Self {
            min_samples,
            max_samples,
            target_sampling_rate,
            reservoir_sampling_max_count,
            samples: Vec::with_capacity(min_samples),
            population_count: 0,
            rng: Rng::with_seed(seed),
        })
    }

    /// Add a bounding box and update the samples using the multi-stage sampling algorithm
    pub fn add_bbox(&mut self, bbox: &BoundingBox) {
        self.population_count += 1;

        if self.samples.len() < self.min_samples {
            // Stage 1: Filling the small reservoir
            self.samples.push(bbox.clone());
        } else if self.population_count <= self.reservoir_sampling_max_count {
            // Stage 2: Small reservoir sampling
            // Use reservoir sampling to keep min_samples samples
            let index = self.rng.usize(..self.population_count);
            if index < self.min_samples {
                self.samples[index] = bbox.clone();
            }
        } else if self.samples.len() < self.max_samples {
            // Stage 3: Bernoulli sampling
            // The reservoir cannot guarantee the minimum sampling rate
            if self.rng.f64() < self.target_sampling_rate {
                self.samples.push(bbox.clone());
            }
        } else {
            // Stage 4: Large reservoir sampling
            // We've reached the maximum number of samples
            let index = self.rng.usize(..self.population_count);
            if index < self.max_samples {
                self.samples[index] = bbox.clone();
            }
        }
    }

    /// Estimate the maximum amount memory used by this sampler
    pub fn estimate_maximum_memory_usage(&self) -> usize {
        self.max_samples * size_of::<BoundingBox>()
    }

    /// Get the number of samples collected so far
    #[cfg(test)]
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Consume the sampler and return the collected samples
    pub fn into_samples(self) -> BoundingBoxSamples {
        BoundingBoxSamples {
            samples: self.samples,
            population_count: self.population_count,
        }
    }
}

impl BoundingBoxSamples {
    /// Create an empty bounding box samples
    pub fn empty() -> Self {
        Self {
            samples: Vec::new(),
            population_count: 0,
        }
    }

    /// Samples collected from the population
    #[cfg(test)]
    pub fn samples(&self) -> &Vec<BoundingBox> {
        &self.samples
    }

    /// Move samples out and consume this value
    pub fn take_samples(self) -> Vec<BoundingBox> {
        self.samples
    }

    /// Size of the population
    pub fn population_count(&self) -> usize {
        self.population_count
    }

    /// Actual sampling rate
    pub fn sampling_rate(&self) -> f64 {
        if self.population_count() == 0 {
            0.0
        } else {
            self.samples.len() as f64 / self.population_count() as f64
        }
    }

    /// Combine 2 samples into one. The combined samples remain uniformly sampled from the
    /// combined population.
    pub fn combine(self, other: BoundingBoxSamples, rng: &mut Rng) -> BoundingBoxSamples {
        if self.population_count() == 0 {
            return other;
        }
        if other.population_count() == 0 {
            return self;
        }

        let self_sampling_rate = self.sampling_rate();
        let other_sampling_rate = other.sampling_rate();
        if self_sampling_rate > other_sampling_rate {
            return other.combine(self, rng);
        }
        if other_sampling_rate <= 0.0 {
            return BoundingBoxSamples {
                samples: Vec::new(),
                population_count: self.population_count() + other.population_count(),
            };
        }

        // self has smaller sampling rate. We need to subsample other to
        // make both sides having the same sampling rate
        let subsampling_rate = self_sampling_rate / other_sampling_rate;
        let mut samples = self.samples;
        for bbox in other.samples {
            if rng.f64() < subsampling_rate {
                samples.push(bbox);
            }
        }
        BoundingBoxSamples {
            samples,
            population_count: self.population_count + other.population_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use sedona_geometry::interval::IntervalTrait;
    use std::mem::size_of;

    #[test]
    fn test_empty() {
        let sampler = BoundingBoxSampler::try_new(10, 100, 0.1, 42).unwrap();
        assert_eq!(
            sampler.estimate_maximum_memory_usage(),
            100 * size_of::<BoundingBox>()
        );
        let samples = sampler.into_samples();
        assert_eq!(samples.population_count(), 0);
        assert_eq!(samples.samples().len(), 0);
        assert_eq!(samples.sampling_rate(), 0.0);
    }

    #[test]
    fn test_all_stages_of_sampling() {
        let mut sampler = BoundingBoxSampler::try_new(100, 200, 0.01, 42).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        let mut count = 0;

        // Stage 1
        for _ in 0..100 {
            let bbox = BoundingBox::xy(
                (rng.gen::<f64>(), rng.gen::<f64>()),
                (rng.gen::<f64>(), rng.gen::<f64>()),
            );
            sampler.add_bbox(&bbox);
            count += 1;
        }
        assert_eq!(sampler.num_samples(), 100);

        // Stage 2
        for _ in 0..9900 {
            let bbox = BoundingBox::xy(
                (rng.gen::<f64>(), rng.gen::<f64>()),
                (rng.gen::<f64>(), rng.gen::<f64>()),
            );
            sampler.add_bbox(&bbox);
            count += 1;
        }
        assert_eq!(sampler.num_samples(), 100);

        // Stage 3
        for _ in 0..5000 {
            let bbox = BoundingBox::xy(
                (rng.gen::<f64>(), rng.gen::<f64>()),
                (rng.gen::<f64>(), rng.gen::<f64>()),
            );
            sampler.add_bbox(&bbox);
            count += 1;
        }
        // More lenient tolerance due to randomness in Bernoulli sampling
        assert!((sampler.num_samples() as i32 - 150).abs() < 20);

        for _ in 0..5000 {
            let bbox = BoundingBox::xy(
                (rng.gen::<f64>(), rng.gen::<f64>()),
                (rng.gen::<f64>(), rng.gen::<f64>()),
            );
            sampler.add_bbox(&bbox);
            count += 1;
        }
        assert!((sampler.num_samples() as i32 - 200).abs() < 20);

        // Stage 4
        for _ in 0..20000 {
            let bbox = BoundingBox::xy(
                (-rng.gen::<f64>(), -rng.gen::<f64>()),
                (rng.gen::<f64>(), rng.gen::<f64>()),
            );
            sampler.add_bbox(&bbox);
            count += 1;
        }

        let samples = sampler.into_samples();
        assert_eq!(samples.samples().len(), 200);
        assert_eq!(samples.population_count(), count);

        // Should sample evenly
        let sample_vec = samples.samples();
        let left_plane_count = sample_vec.iter().filter(|bbox| bbox.x().lo() < 0.0).count();
        let right_plane_count = sample_vec.iter().filter(|bbox| bbox.x().lo() > 0.0).count();
        assert!(left_plane_count > 50 && left_plane_count < 150);
        assert!(right_plane_count > 50 && right_plane_count < 150);
    }

    #[test]
    fn test_uniform_sampling_across_space() {
        // Test that sampling is uniform across spatial regions
        let mut sampler = BoundingBoxSampler::try_new(100, 200, 0.01, 42).unwrap();
        let mut rng = StdRng::seed_from_u64(123);

        // Add 10000 points uniformly distributed
        for _ in 0..10000 {
            let x = rng.gen::<f64>() * 100.0;
            let y = rng.gen::<f64>() * 100.0;
            let bbox = BoundingBox::xy((x, x), (y, y));
            sampler.add_bbox(&bbox);
        }

        let samples = sampler.into_samples();
        assert_eq!(samples.population_count(), 10000);
        assert_eq!(samples.samples().len(), 100);

        // Check distribution in 4 quadrants
        let sample_vec = samples.samples();
        let q1 = sample_vec
            .iter()
            .filter(|b| b.x().lo() < 50.0 && b.y().lo() < 50.0)
            .count();
        let q2 = sample_vec
            .iter()
            .filter(|b| b.x().lo() >= 50.0 && b.y().lo() < 50.0)
            .count();
        let q3 = sample_vec
            .iter()
            .filter(|b| b.x().lo() < 50.0 && b.y().lo() >= 50.0)
            .count();
        let q4 = sample_vec
            .iter()
            .filter(|b| b.x().lo() >= 50.0 && b.y().lo() >= 50.0)
            .count();

        // Each quadrant should have roughly 25 samples (with some variance)
        assert!(q1 > 10 && q1 < 40);
        assert!(q2 > 10 && q2 < 40);
        assert!(q3 > 10 && q3 < 40);
        assert!(q4 > 10 && q4 < 40);
    }

    #[test]
    fn test_deterministic_with_seeded_rng() {
        // Two samplers with the same seed must produce identical samples
        let seed = 12345;
        let mut ref_sampler = BoundingBoxSampler::try_new(10, 100, 0.1, seed).unwrap();
        let bboxes: Vec<BoundingBox> = (0..1000)
            .map(|i| BoundingBox::xy((i as f64, (i + 1) as f64), (i as f64, (i + 1) as f64)))
            .collect();
        for bbox in &bboxes {
            ref_sampler.add_bbox(bbox);
        }

        let reference = ref_sampler.into_samples();
        let reference_samples = reference.samples().clone();

        for _ in 0..5 {
            let mut sampler = BoundingBoxSampler::try_new(10, 100, 0.1, seed).unwrap();
            for bbox in &bboxes {
                sampler.add_bbox(bbox);
            }
            let samples = sampler.into_samples();
            let candidate = samples.samples().clone();
            assert_eq!(reference_samples, candidate);
        }
    }

    #[test]
    fn test_large_dataset() {
        // Test with a larger dataset to ensure all stages work correctly
        let mut sampler = BoundingBoxSampler::try_new(100, 200, 0.01, 42).unwrap();
        let mut rng = StdRng::seed_from_u64(999);

        for _ in 0..50000 {
            let bbox = BoundingBox::xy(
                (rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 100.0),
                (rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 100.0),
            );
            sampler.add_bbox(&bbox);
        }

        let memory_usage = sampler.estimate_maximum_memory_usage();
        let samples = sampler.into_samples();
        assert_eq!(samples.population_count(), 50000);
        // Should hit max_samples in stage 4
        assert_eq!(samples.samples().len(), 200);
        assert_eq!(memory_usage, 200 * size_of::<BoundingBox>());
    }

    #[test]
    fn test_combine_with_zero_population() {
        let mut rng = fastrand::Rng::with_seed(7);
        let empty = BoundingBoxSamples::empty();
        let non_empty = BoundingBoxSamples {
            samples: vec![
                BoundingBox::xy((1, 2), (3, 4)),
                BoundingBox::xy((3, 5), (7, 9)),
            ],
            population_count: 100,
        };

        let samples0 = empty.clone();
        let samples1 = non_empty.clone();
        let combined = samples0.combine(samples1, &mut rng);
        assert_eq!(combined, non_empty);

        let samples0 = non_empty.clone();
        let samples1 = empty.clone();
        let combined = samples0.combine(samples1, &mut rng);
        assert_eq!(combined, non_empty);

        let samples0 = empty.clone();
        let samples1 = empty.clone();
        let combined = samples0.combine(samples1, &mut rng);
        assert_eq!(combined, empty);
    }

    #[test]
    fn test_combine_with_zero_sampling_rate() {
        let mut rng = fastrand::Rng::with_seed(7);
        let empty = BoundingBoxSamples {
            samples: Vec::new(),
            population_count: 100,
        };
        let samples = vec![BoundingBox::xy((1, 2), (3, 4)); 20];
        let non_empty = BoundingBoxSamples {
            samples,
            population_count: 100,
        };

        let samples0 = empty.clone();
        let samples1 = non_empty.clone();
        let combined = samples0.combine(samples1, &mut rng);
        assert!(combined.samples().is_empty());
        assert_eq!(combined.population_count(), 200);

        let samples0 = non_empty.clone();
        let samples1 = empty.clone();
        let combined = samples0.combine(samples1, &mut rng);
        assert!(combined.samples().is_empty());
        assert_eq!(combined.population_count(), 200);

        let samples0 = empty.clone();
        let samples1 = empty.clone();
        let combined = samples0.combine(samples1, &mut rng);
        assert!(combined.samples().is_empty());
        assert_eq!(combined.population_count(), 200);
    }

    #[test]
    fn test_combine_samples() {
        let mut rng = fastrand::Rng::with_seed(7);
        let mut left_samples = Vec::new();
        let mut right_samples = Vec::new();
        for k in 0..100 {
            left_samples.push(BoundingBox::xy((-10, -5), (k, k + 1)));
            right_samples.push(BoundingBox::xy((5, 10), (k, k + 1)));
        }
        let left = BoundingBoxSamples {
            samples: left_samples,
            population_count: 1000,
        };
        let right = BoundingBoxSamples {
            samples: right_samples,
            population_count: 2000,
        };

        // The combined samples should have approximately 150 samples.
        // 50 from the left side, 100 from the right side.
        let combined = left.combine(right, &mut rng);
        assert_eq!(combined.population_count(), 3000);

        let mut left_count: i32 = 0;
        let mut right_count: i32 = 0;
        for bbox in combined.samples() {
            if bbox.x().lo() < 0.0 {
                left_count += 1;
            } else {
                right_count += 1;
            }
        }
        assert!((left_count - 50).abs() < 10);
        assert!((right_count - 100).abs() < 10);
    }
}
