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

use std::{collections::HashMap, num::NonZeroUsize, path::PathBuf, sync::Arc};

use datafusion::{
    error::{DataFusionError, Result},
    execution::{
        disk_manager::{DiskManagerBuilder, DiskManagerMode},
        memory_pool::{GreedyMemoryPool, MemoryPool, TrackConsumersPool},
        runtime_env::{RuntimeEnv, RuntimeEnvBuilder},
    },
};

use crate::{
    context::SedonaContext,
    memory_pool::{SedonaFairSpillPool, DEFAULT_UNSPILLABLE_RESERVE_RATIO},
    pool_type::PoolType,
    size_parser,
};

/// Builder for constructing a [`SedonaContext`] with configurable runtime
/// environment settings.
///
/// This builder centralizes the construction of memory pools, disk managers,
/// and runtime environments so that the same logic can be reused across the
/// CLI, Python bindings, ADBC driver, and any future entry points.
///
/// # Examples
///
/// ```rust,no_run
/// # async fn example() -> datafusion::error::Result<()> {
/// use sedona::context_builder::SedonaContextBuilder;
/// use sedona::pool_type::PoolType;
///
/// let ctx = SedonaContextBuilder::new()
///     .with_memory_limit(4 * 1024 * 1024 * 1024)
///     .with_pool_type(PoolType::Fair)
///     .with_temp_dir("/tmp/sedona-spill".to_string())
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
/// String-based configuration (useful for ADBC connection options, etc.):
///
/// ```rust,no_run
/// # async fn example() -> datafusion::error::Result<()> {
/// use std::collections::HashMap;
/// use sedona::context_builder::SedonaContextBuilder;
///
/// let mut opts = HashMap::new();
/// opts.insert("memory_limit".to_string(), "4gb".to_string());
/// opts.insert("memory_pool_type".to_string(), "fair".to_string());
///
/// let ctx = SedonaContextBuilder::from_options(&opts)?.build().await?;
/// # Ok(())
/// # }
/// ```
pub struct SedonaContextBuilder {
    memory_limit: Option<usize>,
    temp_dir: Option<String>,
    pool_type: PoolType,
    unspillable_reserve_ratio: f64,
}

impl Default for SedonaContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SedonaContextBuilder {
    /// Create a new builder with default settings.
    ///
    /// Defaults:
    /// - `memory_limit`: `None` (no limit, uses DataFusion's default unbounded pool)
    /// - `pool_type`: `PoolType::Greedy`
    /// - `unspillable_reserve_ratio`: `0.2`
    /// - `temp_dir`: `None` (uses DataFusion's default temp directory)
    pub fn new() -> Self {
        Self {
            memory_limit: None,
            temp_dir: None,
            pool_type: PoolType::Greedy,
            unspillable_reserve_ratio: DEFAULT_UNSPILLABLE_RESERVE_RATIO,
        }
    }

    /// Create a builder from string-based key-value options.
    ///
    /// Recognized keys:
    /// - `"memory_limit"`: Memory limit as a human-readable size string
    ///   (e.g., `"4gb"`, `"512m"`, `"1.5g"`) or plain bytes (e.g.,
    ///   `"4294967296"`). See [`size_parser::parse_size_string`] for
    ///   supported suffixes.
    /// - `"temp_dir"`: Path for temporary/spill files
    /// - `"memory_pool_type"`: `"greedy"` or `"fair"`
    /// - `"unspillable_reserve_ratio"`: Float between 0.0 and 1.0
    ///
    /// Unrecognized keys are ignored.
    pub fn from_options(options: &HashMap<String, String>) -> Result<Self> {
        let mut builder = Self::new();

        if let Some(memory_limit) = options.get("memory_limit") {
            let limit = size_parser::parse_size_string(memory_limit)?;
            builder = builder.with_memory_limit(limit);
        }

        if let Some(temp_dir) = options.get("temp_dir") {
            builder = builder.with_temp_dir(temp_dir.clone());
        }

        if let Some(pool_type) = options.get("memory_pool_type") {
            let pt: PoolType = pool_type
                .parse()
                .map_err(|e: String| DataFusionError::Configuration(e))?;
            builder = builder.with_pool_type(pt);
        }

        if let Some(ratio) = options.get("unspillable_reserve_ratio") {
            let r: f64 = ratio.parse().map_err(|_| {
                DataFusionError::Configuration(format!(
                    "Invalid unspillable_reserve_ratio value '{ratio}': expected a float"
                ))
            })?;
            builder = builder.with_unspillable_reserve_ratio(r)?;
        }

        Ok(builder)
    }

    /// Set the memory limit in bytes.
    ///
    /// When set, a memory pool is created to enforce this limit. Without a
    /// memory limit, DataFusion's default unbounded memory pool is used.
    pub fn with_memory_limit(mut self, memory_limit: usize) -> Self {
        self.memory_limit = Some(memory_limit);
        self
    }

    /// Set the directory for temporary/spill files.
    pub fn with_temp_dir(mut self, temp_dir: String) -> Self {
        self.temp_dir = Some(temp_dir);
        self
    }

    /// Set the memory pool type.
    ///
    /// - `PoolType::Greedy`: A simple pool that grants reservations on a
    ///   first-come-first-served basis. This is the default.
    /// - `PoolType::Fair`: A pool that fairly distributes memory among
    ///   spillable consumers and reserves a fraction of memory for
    ///   unspillable consumers (configured via
    ///   [`with_unspillable_reserve_ratio`](Self::with_unspillable_reserve_ratio)).
    ///
    /// Only takes effect when `memory_limit` is set.
    pub fn with_pool_type(mut self, pool_type: PoolType) -> Self {
        self.pool_type = pool_type;
        self
    }

    /// Set the fraction of memory reserved for unspillable consumers.
    ///
    /// Must be between 0.0 and 1.0 (inclusive). Only applies when
    /// `pool_type` is `PoolType::Fair` and `memory_limit` is set.
    ///
    /// Returns an error if the value is out of range.
    pub fn with_unspillable_reserve_ratio(mut self, ratio: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&ratio) {
            return Err(DataFusionError::Configuration(format!(
                "unspillable_reserve_ratio must be between 0.0 and 1.0, got {ratio}"
            )));
        }
        self.unspillable_reserve_ratio = ratio;
        Ok(self)
    }

    /// Build a [`RuntimeEnv`] from the current configuration.
    ///
    /// This constructs the memory pool and disk manager based on the
    /// builder settings and returns the resulting runtime environment.
    pub fn build_runtime_env(&self) -> Result<Arc<RuntimeEnv>> {
        let mut rt_builder = RuntimeEnvBuilder::new();

        if let Some(memory_limit) = self.memory_limit {
            let track_capacity = NonZeroUsize::new(10).expect("track capacity must be non-zero");
            let pool: Arc<dyn MemoryPool> = match self.pool_type {
                PoolType::Fair => Arc::new(TrackConsumersPool::new(
                    SedonaFairSpillPool::new(memory_limit, self.unspillable_reserve_ratio),
                    track_capacity,
                )),
                PoolType::Greedy => Arc::new(TrackConsumersPool::new(
                    GreedyMemoryPool::new(memory_limit),
                    track_capacity,
                )),
            };
            rt_builder = rt_builder.with_memory_pool(pool);
        }

        if let Some(ref temp_dir) = self.temp_dir {
            let dm_builder = DiskManagerBuilder::default()
                .with_mode(DiskManagerMode::Directories(vec![PathBuf::from(temp_dir)]));
            rt_builder = rt_builder.with_disk_manager_builder(dm_builder);
        }

        rt_builder.build_arc()
    }

    /// Build a [`SedonaContext`] from the current configuration.
    ///
    /// This constructs the runtime environment and then creates a fully
    /// configured interactive `SedonaContext`.
    pub async fn build(self) -> Result<SedonaContext> {
        let runtime_env = self.build_runtime_env()?;
        SedonaContext::new_local_interactive_with_runtime_env(runtime_env).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_builder() {
        let builder = SedonaContextBuilder::new();
        assert!(builder.memory_limit.is_none());
        assert!(builder.temp_dir.is_none());
        assert_eq!(builder.pool_type, PoolType::Greedy);
        assert!(
            (builder.unspillable_reserve_ratio - DEFAULT_UNSPILLABLE_RESERVE_RATIO).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_builder_with_methods() {
        let builder = SedonaContextBuilder::new()
            .with_memory_limit(1024)
            .with_temp_dir("/tmp/test".to_string())
            .with_pool_type(PoolType::Fair)
            .with_unspillable_reserve_ratio(0.3)
            .unwrap();
        assert_eq!(builder.memory_limit, Some(1024));
        assert_eq!(builder.temp_dir, Some("/tmp/test".to_string()));
        assert_eq!(builder.pool_type, PoolType::Fair);
        assert!((builder.unspillable_reserve_ratio - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_invalid_unspillable_reserve_ratio() {
        let result = SedonaContextBuilder::new().with_unspillable_reserve_ratio(-0.1);
        assert!(result.is_err());

        let result = SedonaContextBuilder::new().with_unspillable_reserve_ratio(1.1);
        assert!(result.is_err());

        // Edge cases: 0.0 and 1.0 should be valid
        let result = SedonaContextBuilder::new().with_unspillable_reserve_ratio(0.0);
        assert!(result.is_ok());

        let result = SedonaContextBuilder::new().with_unspillable_reserve_ratio(1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_options() {
        let mut opts = HashMap::new();
        opts.insert("memory_limit".to_string(), "4096".to_string());
        opts.insert("temp_dir".to_string(), "/tmp/spill".to_string());
        opts.insert("memory_pool_type".to_string(), "fair".to_string());
        opts.insert("unspillable_reserve_ratio".to_string(), "0.3".to_string());

        let builder = SedonaContextBuilder::from_options(&opts).unwrap();
        assert_eq!(builder.memory_limit, Some(4096));
        assert_eq!(builder.temp_dir, Some("/tmp/spill".to_string()));
        assert_eq!(builder.pool_type, PoolType::Fair);
        assert!((builder.unspillable_reserve_ratio - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_options_empty() {
        let opts = HashMap::new();
        let builder = SedonaContextBuilder::from_options(&opts).unwrap();
        assert!(builder.memory_limit.is_none());
        assert!(builder.temp_dir.is_none());
        assert_eq!(builder.pool_type, PoolType::Greedy);
    }

    #[test]
    fn test_from_options_invalid_memory_limit() {
        let mut opts = HashMap::new();
        opts.insert("memory_limit".to_string(), "not_a_number".to_string());
        assert!(SedonaContextBuilder::from_options(&opts).is_err());
    }

    #[test]
    fn test_from_options_human_readable_memory_limit() {
        let mut opts = HashMap::new();
        opts.insert("memory_limit".to_string(), "4gb".to_string());
        let builder = SedonaContextBuilder::from_options(&opts).unwrap();
        assert_eq!(builder.memory_limit, Some(4 * 1024 * 1024 * 1024));

        let mut opts = HashMap::new();
        opts.insert("memory_limit".to_string(), "512m".to_string());
        let builder = SedonaContextBuilder::from_options(&opts).unwrap();
        assert_eq!(builder.memory_limit, Some(512 * 1024 * 1024));

        let mut opts = HashMap::new();
        opts.insert("memory_limit".to_string(), "1.5g".to_string());
        let builder = SedonaContextBuilder::from_options(&opts).unwrap();
        assert_eq!(
            builder.memory_limit,
            Some((1.5 * 1024.0 * 1024.0 * 1024.0) as usize)
        );
    }

    #[test]
    fn test_from_options_invalid_pool_type() {
        let mut opts = HashMap::new();
        opts.insert("memory_pool_type".to_string(), "invalid".to_string());
        assert!(SedonaContextBuilder::from_options(&opts).is_err());
    }

    #[test]
    fn test_from_options_invalid_ratio() {
        let mut opts = HashMap::new();
        opts.insert("unspillable_reserve_ratio".to_string(), "2.0".to_string());
        assert!(SedonaContextBuilder::from_options(&opts).is_err());
    }

    #[test]
    fn test_from_options_unrecognized_keys_ignored() {
        let mut opts = HashMap::new();
        opts.insert("unknown_key".to_string(), "value".to_string());
        let builder = SedonaContextBuilder::from_options(&opts).unwrap();
        assert!(builder.memory_limit.is_none());
    }

    #[test]
    fn test_build_runtime_env_no_memory_limit() {
        let builder = SedonaContextBuilder::new();
        let result = builder.build_runtime_env();
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_runtime_env_with_greedy_pool() {
        let builder = SedonaContextBuilder::new()
            .with_memory_limit(1024 * 1024)
            .with_pool_type(PoolType::Greedy);
        let result = builder.build_runtime_env();
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_runtime_env_with_fair_pool() {
        let builder = SedonaContextBuilder::new()
            .with_memory_limit(1024 * 1024)
            .with_pool_type(PoolType::Fair)
            .with_unspillable_reserve_ratio(0.2)
            .unwrap();
        let result = builder.build_runtime_env();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_build_context_default() {
        let ctx = SedonaContextBuilder::new().build().await;
        assert!(ctx.is_ok());
    }

    #[tokio::test]
    async fn test_build_context_with_memory_limit() {
        let ctx = SedonaContextBuilder::new()
            .with_memory_limit(100 * 1024 * 1024)
            .with_pool_type(PoolType::Fair)
            .with_unspillable_reserve_ratio(0.2)
            .unwrap()
            .build()
            .await;
        assert!(ctx.is_ok());
    }
}
