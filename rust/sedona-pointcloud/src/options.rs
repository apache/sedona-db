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
    error::DataFusionError,
    extensions_options,
};

use crate::laz::options::{LasExtraBytes, LasOptions};

/// Geometry representation
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum GeometryEncoding {
    /// Use plain coordinates as three fields `x`, `y`, `z` with datatype Float64 encoding.
    #[default]
    Plain,
    /// Resolves the coordinates to a fields `geometry` with WKB encoding.
    Wkb,
    /// Resolves the coordinates to a fields `geometry` with separated GeoArrow encoding.
    Native,
}

impl Display for GeometryEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryEncoding::Plain => f.write_str("plain"),
            GeometryEncoding::Wkb => f.write_str("wkb"),
            GeometryEncoding::Native => f.write_str("native"),
        }
    }
}

impl FromStr for GeometryEncoding {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "plain" => Ok(Self::Plain),
            "wkb" => Ok(Self::Wkb),
            "native" => Ok(Self::Native),
            s => Err(format!("Unable to parse from `{s}`")),
        }
    }
}

impl ConfigField for GeometryEncoding {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, _description: &'static str) {
        v.some(
            &format!("{key}.geometry_encoding"),
            self,
            "Specify point geometry encoding",
        );
    }

    fn set(&mut self, _key: &str, value: &str) -> Result<(), DataFusionError> {
        *self = value.parse().map_err(DataFusionError::Configuration)?;
        Ok(())
    }
}

extensions_options! {
    /// Pointcloud configuration options
    pub struct PointcloudOptions {
        pub geometry_encoding: GeometryEncoding, default = GeometryEncoding::default()
        pub las: LasOptions, default = LasOptions::default()
    }

}

impl ConfigExtension for PointcloudOptions {
    const PREFIX: &'static str = "pointcloud";
}

impl PointcloudOptions {
    pub fn with_geometry_encoding(mut self, geometry_encoding: GeometryEncoding) -> Self {
        self.geometry_encoding = geometry_encoding;
        self
    }

    pub fn with_las_extra_bytes(mut self, extra_bytes: LasExtraBytes) -> Self {
        self.las.extra_bytes = extra_bytes;
        self
    }
}
