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
use std::fmt::Display;

use datafusion::config::{ConfigEntry, ConfigExtension, ConfigField, ExtensionOptions, Visit};
use datafusion::prelude::SessionConfig;
use datafusion_common::config_namespace;
use datafusion_common::Result;
use regex::Regex;

/// Default minimum number of analyzed geometries for speculative execution mode to select an
/// optimal execution mode.
pub const DEFAULT_SPECULATIVE_THRESHOLD: usize = 1000;

/// Default minimum number of points per geometry to use prepared geometries for the build side.
pub const DEFAULT_MIN_POINTS_FOR_BUILD_PREPARATION: usize = 50;

/// Helper function to register the spatial join optimizer with a session config
pub fn add_sedona_option_extension(config: SessionConfig) -> SessionConfig {
    config.with_option_extension(SedonaOptions::default())
}

config_namespace! {
    /// Configuration options for Sedona.
    pub struct SedonaOptions {
        /// Options for spatial join
        pub spatial_join: SpatialJoinOptions, default = SpatialJoinOptions::default()
    }
}

config_namespace! {
    /// Configuration options for spatial join.
    ///
    /// This struct controls various aspects of how spatial joins are performed,
    /// including prepared geometry usage and spatial library used for evaluating
    /// spatial predicates.
    pub struct SpatialJoinOptions {
        /// Enable optimized spatial join
        pub enable: bool, default = true

        /// Spatial library to use for spatial join
        pub spatial_library: SpatialLibrary, default = SpatialLibrary::Tg

        /// Options for configuring the GEOS spatial library
        pub geos: GeosOptions, default = GeosOptions::default()

        /// Options for configuring the TG spatial library
        pub tg: TgOptions, default = TgOptions::default()

        /// The execution mode determining how prepared geometries are used
        pub execution_mode: ExecutionMode, default = ExecutionMode::Speculative(DEFAULT_SPECULATIVE_THRESHOLD)

        /// Collect build side partitions concurrently (using spawned tasks).
        /// Set to false for contexts where spawning new tasks is not supported.
        pub concurrent_build_side_collection: bool, default = true

        /// Include tie-breakers in KNN join results when there are tied distances
        pub knn_include_tie_breakers: bool, default = false
    }
}

config_namespace! {
    /// Configuration options for the GEOS spatial library
    pub struct GeosOptions {
        /// The minimum number of points per geometry to use prepared geometries for the build side.
        pub min_points_for_build_preparation: usize, default = DEFAULT_MIN_POINTS_FOR_BUILD_PREPARATION
    }
}

config_namespace! {
    /// Configuration options for the TG spatial library
    pub struct TgOptions {
        /// The index type to use for the TG spatial library
        pub index_type: TgIndexType, default = TgIndexType::YStripes
    }
}

impl ConfigExtension for SedonaOptions {
    const PREFIX: &'static str = "sedona";
}

impl ExtensionOptions for SedonaOptions {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> Result<()> {
        <Self as ConfigField>::set(self, key, value)
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        struct Visitor(Vec<ConfigEntry>);

        impl Visit for Visitor {
            fn some<V: Display>(&mut self, key: &str, value: V, description: &'static str) {
                self.0.push(ConfigEntry {
                    key: key.to_string(),
                    value: Some(value.to_string()),
                    description,
                })
            }

            fn none(&mut self, key: &str, description: &'static str) {
                self.0.push(ConfigEntry {
                    key: key.to_string(),
                    value: None,
                    description,
                })
            }
        }

        let mut v = Visitor(vec![]);
        self.visit(&mut v, Self::PREFIX, "");
        v.0
    }
}

/// Execution mode for spatial join operations, controlling prepared geometry usage.
///
/// Prepared geometries are pre-processed spatial objects that can significantly
/// improve performance for spatial predicate evaluation when the same geometry
/// is used multiple times in comparisons.
///
/// The choice of execution mode depends on the specific characteristics of your
/// spatial join workload, as well as the spatial relation predicate between the
/// two tables. Some of the spatial relation computations cannot be accelerated by
/// prepared geometries at all (for example, ST_Touches, ST_Crosses, ST_DWithin).
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum ExecutionMode {
    /// Don't use prepared geometries for spatial predicate evaluation.
    PrepareNone,

    /// Create prepared geometries for the build side (left/smaller table).
    PrepareBuild,

    /// Create prepared geometries for the probe side (right/larger table).
    PrepareProbe,

    /// Automatically choose the best execution mode based on the characteristics of
    /// first few geometries on the probe side.
    Speculative(usize),
}

impl ExecutionMode {
    /// Convert the execution mode to a usize value.
    ///
    /// This is used to show the execution mode in the metrics. We use a gauge value
    /// to represent the execution mode.
    pub fn to_usize(&self) -> usize {
        match self {
            ExecutionMode::PrepareNone => 0,
            ExecutionMode::PrepareBuild => 1,
            ExecutionMode::PrepareProbe => 2,
            ExecutionMode::Speculative(_) => 3,
        }
    }
}

impl ConfigField for ExecutionMode {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        let value = match self {
            ExecutionMode::PrepareNone => "prepare_none".into(),
            ExecutionMode::PrepareBuild => "prepare_build".into(),
            ExecutionMode::PrepareProbe => "prepare_probe".into(),
            ExecutionMode::Speculative(n) => format!("auto[{n}]"),
        };
        v.some(key, value, description);
    }

    fn set(&mut self, _key: &str, value: &str) -> Result<()> {
        let value = value.to_lowercase();
        let mode = match value.as_str() {
            "prepare_none" => ExecutionMode::PrepareNone,
            "prepare_build" => ExecutionMode::PrepareBuild,
            "prepare_probe" => ExecutionMode::PrepareProbe,
            _ => {
                // Match "auto" or "auto[number]" pattern
                let auto_regex = Regex::new(r"^auto(?:\[(\d+)\])?$").unwrap();

                if let Some(captures) = auto_regex.captures(&value) {
                    // If there's a captured group (the number), use it; otherwise default to 100
                    let n = if let Some(number_match) = captures.get(1) {
                        match number_match.as_str().parse::<usize>() {
                            Ok(n) => {
                                if n == 0 {
                                    return Err(datafusion_common::DataFusionError::Configuration(
                                        "Invalid number in auto mode: 0 is not allowed".to_string(),
                                    ));
                                }
                                n
                            }
                            Err(_) => {
                                return Err(datafusion_common::DataFusionError::Configuration(
                                    format!(
                                        "Invalid number in auto mode: {}",
                                        number_match.as_str()
                                    ),
                                ));
                            }
                        }
                    } else {
                        DEFAULT_SPECULATIVE_THRESHOLD // Default for plain "auto"
                    };
                    ExecutionMode::Speculative(n)
                } else {
                    return Err(datafusion_common::DataFusionError::Configuration(
                        format!("Unknown execution mode: {value}. Expected formats: prepare_none, prepare_build, prepare_probe, auto, auto[number]")
                    ));
                }
            }
        };
        *self = mode;
        Ok(())
    }
}

/// The spatial library to use for evaluating spatial predicates
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SpatialLibrary {
    /// Use georust/geo library (<https://github.com/georust/geo>)
    Geo,

    /// Use GEOS library via georust/geos (<https://github.com/georust/geos>)
    Geos,

    /// Use tiny geometry library (<https://github.com/tidwall/tg>)
    Tg,
}

impl ConfigField for SpatialLibrary {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        let value = match self {
            SpatialLibrary::Geo => "geo",
            SpatialLibrary::Geos => "geos",
            SpatialLibrary::Tg => "tg",
        };
        v.some(key, value, description);
    }

    fn set(&mut self, _key: &str, value: &str) -> Result<()> {
        let value = value.to_lowercase();
        let library = match value.as_str() {
            "geo" => SpatialLibrary::Geo,
            "geos" => SpatialLibrary::Geos,
            "tg" => SpatialLibrary::Tg,
            _ => {
                return Err(datafusion_common::DataFusionError::Configuration(format!(
                    "Unknown spatial library: {value}. Expected: geo, geos, tg"
                )));
            }
        };
        *self = library;
        Ok(())
    }
}

/// The index type to use for the TG spatial library
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TgIndexType {
    /// Natural index
    Natural,

    /// Y-stripes index
    YStripes,
}

impl ConfigField for TgIndexType {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        let value = match self {
            TgIndexType::Natural => "natural",
            TgIndexType::YStripes => "ystripes",
        };
        v.some(key, value, description);
    }

    fn set(&mut self, _key: &str, value: &str) -> Result<()> {
        let value = value.to_lowercase();
        let index_type = match value.as_str() {
            "natural" => TgIndexType::Natural,
            "ystripes" => TgIndexType::YStripes,
            _ => {
                return Err(datafusion_common::DataFusionError::Configuration(format!(
                    "Unknown TG index type: {value}. Expected: natural, ystripes"
                )));
            }
        };
        *self = index_type;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::config::ConfigField;

    #[test]
    fn test_execution_mode_parsing_basic_modes() {
        let mut mode = ExecutionMode::PrepareNone;

        // Test basic modes
        assert!(mode.set("", "prepare_none").is_ok());
        assert_eq!(mode, ExecutionMode::PrepareNone);

        assert!(mode.set("", "prepare_build").is_ok());
        assert_eq!(mode, ExecutionMode::PrepareBuild);

        assert!(mode.set("", "prepare_probe").is_ok());
        assert_eq!(mode, ExecutionMode::PrepareProbe);
    }

    #[test]
    fn test_execution_mode_parsing_auto_modes() {
        let mut mode = ExecutionMode::PrepareNone;

        // Test auto mode with default value
        assert!(mode.set("", "auto").is_ok());
        assert_eq!(mode, ExecutionMode::Speculative(1000));

        // Test auto mode with specific values
        assert!(mode.set("", "auto[10]").is_ok());
        assert_eq!(mode, ExecutionMode::Speculative(10));

        assert!(mode.set("", "auto[500]").is_ok());
        assert_eq!(mode, ExecutionMode::Speculative(500));

        assert!(mode.set("", "auto[1]").is_ok());
        assert_eq!(mode, ExecutionMode::Speculative(1));
    }

    #[test]
    fn test_execution_mode_parsing_case_insensitive() {
        let mut mode = ExecutionMode::PrepareNone;

        // Test case insensitivity
        assert!(mode.set("", "PREPARE_NONE").is_ok());
        assert_eq!(mode, ExecutionMode::PrepareNone);

        assert!(mode.set("", "PREPARE_BUILD").is_ok());
        assert_eq!(mode, ExecutionMode::PrepareBuild);

        assert!(mode.set("", "PREPARE_PROBE").is_ok());
        assert_eq!(mode, ExecutionMode::PrepareProbe);

        assert!(mode.set("", "AUTO").is_ok());
        assert_eq!(mode, ExecutionMode::Speculative(1000));

        assert!(mode.set("", "Auto[50]").is_ok());
        assert_eq!(mode, ExecutionMode::Speculative(50));
    }

    #[test]
    fn test_execution_mode_parsing_invalid_formats() {
        let mut mode = ExecutionMode::PrepareNone;

        // Test invalid modes
        assert!(mode.set("", "invalid").is_err());
        assert!(mode.set("", "").is_err());
        assert!(mode.set("", "auto[0]").is_err());
        assert!(mode.set("", "auto[]").is_err());
        assert!(mode.set("", "auto[abc]").is_err());
        assert!(mode.set("", "auto[10").is_err());
        assert!(mode.set("", "auto10]").is_err());
        assert!(mode.set("", "auto[10][20]").is_err());
        assert!(mode.set("", "auto 10").is_err());
        assert!(mode.set("", "auto:10").is_err());
        assert!(mode.set("", "auto(10)").is_err());
    }

    #[test]
    fn test_tg_index_type_parsing() {
        let mut index_type = TgIndexType::YStripes;

        assert!(index_type.set("", "natural").is_ok());
        assert_eq!(index_type, TgIndexType::Natural);

        assert!(index_type.set("", "Natural").is_ok());
        assert_eq!(index_type, TgIndexType::Natural);

        assert!(index_type.set("", "ystripes").is_ok());
        assert_eq!(index_type, TgIndexType::YStripes);

        assert!(index_type.set("", "YStripes").is_ok());
        assert_eq!(index_type, TgIndexType::YStripes);
    }

    #[test]
    fn test_tg_index_type_parsing_invalid_formats() {
        let mut index_type = TgIndexType::YStripes;

        assert!(index_type.set("", "unindexed").is_err());
        assert!(index_type.set("", "invalid").is_err());
        assert!(index_type.set("", "").is_err());
    }
}
