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

//! Decides when a long-running loop of the spatial join should yield to the scheduler.
//!
//! A query is cancelled by dropping its stream, which only takes effect when the stream
//! returns `Poll::Pending`. The join's probe loops can run for a long time without any
//! output, for example when a slow predicate rarely matches, so they yield after a fixed
//! amount of work. The amount is measured in time rather than in rows, because one row
//! can take anything from a fraction of a microsecond to milliseconds: yielding after a
//! fixed number of rows either costs time on cheap rows or waits too long on slow ones.
//! Reading the clock is cheap but not free, so the timer reads it at every step while
//! steps are slow and only every few steps, up to [`MAX_STEPS_PER_CLOCK_READ`], while
//! they are cheap.

use std::time::{Duration, Instant};

use sedona_common::option::SpatialJoinOptions;

/// How long the join works before it yields, unless `debug.yield_interval_ms` is set.
pub(crate) const DEFAULT_YIELD_INTERVAL: Duration = Duration::from_millis(10);

/// Largest number of steps between two reads of the clock.
pub(crate) const MAX_STEPS_PER_CLOCK_READ: u32 = 16;

/// The yield interval set by `options`.
pub(crate) fn yield_interval(options: &SpatialJoinOptions) -> Duration {
    options
        .debug
        .yield_interval_ms
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_YIELD_INTERVAL)
}

/// Tells a loop when it has worked for the yield interval since it started or last
/// yielded.
#[derive(Debug)]
pub(crate) struct YieldTimer {
    interval: Duration,
    /// Steps that took less than this between two reads of the clock are cheap
    cheap: Duration,
    /// Steps since the clock was last read
    steps: u32,
    /// Steps between two reads of the clock, from 1 to [`MAX_STEPS_PER_CLOCK_READ`]
    steps_per_read: u32,
    last_read: Instant,
    last_yield: Instant,
}

impl YieldTimer {
    pub(crate) fn new(interval: Duration) -> Self {
        Self::starting_at(interval, Instant::now())
    }

    /// A timer whose first interval started at `start`.
    pub(crate) fn starting_at(interval: Duration, start: Instant) -> Self {
        Self {
            interval,
            cheap: interval / MAX_STEPS_PER_CLOCK_READ,
            steps: 0,
            steps_per_read: 1,
            last_read: start,
            last_yield: start,
        }
    }

    /// Records one step of work. Returns `true` when the caller should yield now, and
    /// then starts a new interval.
    #[inline]
    pub(crate) fn step(&mut self) -> bool {
        self.step_with(Instant::now)
    }

    /// [`Self::step`] with the clock passed in, so that tests can control time.
    #[inline]
    fn step_with(&mut self, clock: impl FnOnce() -> Instant) -> bool {
        self.steps += 1;
        if self.steps < self.steps_per_read {
            return false;
        }
        self.read_clock(clock())
    }

    fn read_clock(&mut self, now: Instant) -> bool {
        self.steps = 0;
        // Read the clock half as often after steps that took well under the interval,
        // and at every step again after steps that did not.
        if now.duration_since(self.last_read) < self.cheap {
            self.steps_per_read = (self.steps_per_read * 2).min(MAX_STEPS_PER_CLOCK_READ);
        } else {
            self.steps_per_read = 1;
        }
        self.last_read = now;
        self.due_at(now)
    }

    /// Reads the clock now, however many steps were recorded. Returns `true` when the
    /// interval has passed since the last yield, and then starts a new interval.
    pub(crate) fn due(&mut self) -> bool {
        self.due_at(Instant::now())
    }

    fn due_at(&mut self, now: Instant) -> bool {
        if now.duration_since(self.last_yield) < self.interval {
            return false;
        }
        self.last_yield = now;
        true
    }
}

#[cfg(test)]
mod tests {
    use sedona_common::option::SpatialJoinDebugOptions;

    use super::*;

    /// Runs `steps` steps that each take `step_time`, on a fake clock. Returns the steps
    /// after which the timer said to yield, and how often it read the clock.
    fn run(interval: Duration, step_time: Duration, steps: u32) -> (Vec<u32>, u32) {
        let start = Instant::now();
        let mut timer = YieldTimer::starting_at(interval, start);
        let mut reads = 0;
        let mut yielding_steps = Vec::new();
        for i in 1..=steps {
            let now = start + step_time * i;
            if timer.step_with(|| {
                reads += 1;
                now
            }) {
                yielding_steps.push(i);
            }
        }
        (yielding_steps, reads)
    }

    #[test]
    fn zero_interval_yields_at_every_step() {
        let (yielding_steps, _) = run(Duration::ZERO, Duration::from_micros(1), 8);
        assert_eq!(yielding_steps, (1..=8).collect::<Vec<_>>());
    }

    #[test]
    fn slow_steps_yield_as_soon_as_the_interval_has_passed() {
        // Steps of 5 ms with a 10 ms interval, like a few complex polygons.
        let (yielding_steps, _) = run(Duration::from_millis(10), Duration::from_millis(5), 8);
        assert_eq!(yielding_steps, vec![2, 4, 6, 8]);
    }

    #[test]
    fn cheap_steps_read_the_clock_rarely() {
        let (yielding_steps, reads) =
            run(Duration::from_millis(10), Duration::from_micros(1), 1600);
        assert!(yielding_steps.is_empty());
        // One read per 16 steps, plus a few while the timer finds out the steps are cheap.
        assert!(reads <= 1600 / 16 + 5, "{reads} reads");
    }

    #[test]
    fn long_interval_does_not_yield() {
        let mut timer = YieldTimer::new(Duration::from_secs(3600));
        assert!((0..10_000).all(|_| !timer.step()));
    }

    #[test]
    fn interval_comes_from_the_debug_option() {
        assert_eq!(
            yield_interval(&SpatialJoinOptions::default()),
            DEFAULT_YIELD_INTERVAL
        );
        let options = SpatialJoinOptions {
            debug: SpatialJoinDebugOptions {
                yield_interval_ms: Some(0),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(yield_interval(&options), Duration::ZERO);
    }
}
