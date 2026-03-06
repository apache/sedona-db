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

    fn set(&mut self, _key: &str, value: &str) -> Result<(), DataFusionError> {
        *self = value.parse().map_err(DataFusionError::Configuration)?;
        Ok(())
    }
}

extensions_options! {
    /// LAS/LAZ configuration options
    ///
    /// * `geometry encoding`: plain (x, y, z), wkb or native (geoarrow)
    /// * `collect statistics`: extract las/laz chunk statistics (requires a full scan on registration)
    /// * `parallel statistics extraction`: extract statistics in parallel
    /// * `persist statistics`: store statistics in a sidecar file for future reuse (requires write access)
    /// * `round robin partitioning`: read chunks in parallel with round robin instead of byte range (default)
    /// * `extra bytes`: las extra byte attributes handling, ignore, keep as binary blob, or typed
    pub struct LasOptions {
        pub geometry_encoding: GeometryEncoding, default = GeometryEncoding::default()
        pub extra_bytes: LasExtraBytes, default = LasExtraBytes::default()
        pub collect_statistics: bool, default = false
        pub parallel_statistics_extraction: bool, default = false
        pub persist_statistics: bool, default = false
        pub round_robin_partitioning: bool, default = false
    }

}

impl ConfigExtension for LasOptions {
    const PREFIX: &'static str = "las";
}

impl LasOptions {
    pub fn with_geometry_encoding(mut self, geometry_encoding: GeometryEncoding) -> Self {
        self.geometry_encoding = geometry_encoding;
        self
    }

    pub fn with_las_extra_bytes(mut self, extra_bytes: LasExtraBytes) -> Self {
        self.extra_bytes = extra_bytes;
        self
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use datafusion::{
        execution::SessionStateBuilder,
        prelude::{SessionConfig, SessionContext},
    };

    use crate::las::{
        format::{Extension, LasFormatFactory},
        options::LasOptions,
    };

    fn setup_context() -> SessionContext {
        let config = SessionConfig::new().with_option_extension(LasOptions::default());
        let mut state = SessionStateBuilder::new().with_config(config).build();

        let file_format = Arc::new(LasFormatFactory::new(Extension::Las));
        state.register_file_format(file_format, true).unwrap();

        let file_format = Arc::new(LasFormatFactory::new(Extension::Laz));
        state.register_file_format(file_format, true).unwrap();

        SessionContext::new_with_state(state).enable_url_table()
    }

    #[tokio::test]
    async fn projection() {
        let ctx = setup_context();

        // default options
        let df = ctx
            .sql("SELECT x, y, z FROM 'tests/data/extra.las'")
            .await
            .unwrap();

        assert_eq!(df.schema().fields().len(), 3);

        let df = ctx
            .sql("SELECT x, y, z FROM 'tests/data/extra.laz'")
            .await
            .unwrap();

        assert_eq!(df.schema().fields().len(), 3);

        // overwrite options
        ctx.sql("SET las.geometry_encoding = 'wkb'").await.unwrap();
        ctx.sql("SET las.extra_bytes = 'blob'").await.unwrap();

        let df = ctx
            .sql("SELECT geometry, extra_bytes FROM 'tests/data/extra.las'")
            .await
            .unwrap();

        assert_eq!(df.schema().fields().len(), 2);

        let df = ctx
            .sql("SELECT geometry, extra_bytes FROM 'tests/data/extra.laz'")
            .await
            .unwrap();

        assert_eq!(df.schema().fields().len(), 2);
    }
}
