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

use std::{fmt::Display, str::FromStr};

use datafusion_common::{
    config::{ConfigExtension, ConfigField, Visit},
    config_err,
    error::DataFusionError,
    extensions_options,
};

/// Geometry representation
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum LasPointEncoding {
    // #[default]
    Plain,
    #[default]
    Wkb,
    Nativ,
}

impl Display for LasPointEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LasPointEncoding::Plain => f.write_str("plain"),
            LasPointEncoding::Wkb => f.write_str("wkb"),
            LasPointEncoding::Nativ => f.write_str("nativ"),
        }
    }
}

impl FromStr for LasPointEncoding {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "plain" => Ok(Self::Plain),
            "wkb" => Ok(Self::Wkb),
            "native" => Ok(Self::Nativ),
            s => Err(format!("Unable to parse from `{s}`")),
        }
    }
}

impl ConfigField for LasPointEncoding {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, _description: &'static str) {
        v.some(
            &format!("{key}.point_encoding"),
            self,
            "Specify point encoding",
        );
    }

    fn set(&mut self, key: &str, value: &str) -> Result<(), DataFusionError> {
        // let (key, rem) = key.split_once('.').unwrap_or((key, ""));
        match key {
            "point_encoding" => {
                *self = value.parse().map_err(DataFusionError::Configuration)?;
            }
            _ => return config_err!("Config value \"{}\" not found on LasExtraBytes", key),
        }
        Ok(())
    }
}

/// LAS extra bytes handling
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum LasExtraBytes {
    /// Resolve to typed and named attributes
    Typed,
    /// Keep as binary blob
    Blob,
    /// Drop/ignore extrabytes
    #[default]
    Ignore,
}

impl Display for LasExtraBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LasExtraBytes::Typed => f.write_str("typed"),
            LasExtraBytes::Blob => f.write_str("blob"),
            LasExtraBytes::Ignore => f.write_str("ignore"),
        }
    }
}

impl FromStr for LasExtraBytes {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "typed" => Ok(Self::Typed),
            "blob" => Ok(Self::Blob),
            "ignore" => Ok(Self::Ignore),
            s => Err(format!("Unable to parse from `{s}`")),
        }
    }
}

impl ConfigField for LasExtraBytes {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, _description: &'static str) {
        v.some(
            &format!("{key}.extra_bytes"),
            self,
            "Specify extra bytes handling",
        );
    }

    fn set(&mut self, key: &str, value: &str) -> Result<(), DataFusionError> {
        // let (key, rem) = key.split_once('.').unwrap_or((key, ""));
        match key {
            "extra_bytes" => {
                *self = value.parse().map_err(DataFusionError::Configuration)?;
            }
            _ => return config_err!("Config value \"{}\" not found on LasExtraBytes", key),
        }
        Ok(())
    }
}

// Define a new configuration struct using the `extensions_options` macro
extensions_options! {
    /// The LAZ config options
    pub struct LazConfig {
        pub point_encoding: LasPointEncoding, default = LasPointEncoding::Plain
        pub extra_bytes: LasExtraBytes, default = LasExtraBytes::Ignore
    }

}

impl ConfigExtension for LazConfig {
    const PREFIX: &'static str = "laz";
}
