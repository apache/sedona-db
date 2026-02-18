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
    config::{ConfigField, Visit},
    config_namespace,
    error::DataFusionError,
};

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

config_namespace! {
    /// The LAS config options
    pub struct LasOptions {
        pub extra_bytes: LasExtraBytes, default = LasExtraBytes::default()
    }

}

impl LasOptions {
    pub fn with_extra_bytes(mut self, extra_bytes: LasExtraBytes) -> Self {
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

    use crate::{
        las::format::{Extension, LasFormatFactory},
        options::PointcloudOptions,
    };

    fn setup_context() -> SessionContext {
        let config = SessionConfig::new().with_option_extension(PointcloudOptions::default());
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
        ctx.sql("SET pointcloud.geometry_encoding = 'wkb'")
            .await
            .unwrap();
        ctx.sql("SET pointcloud.las.extra_bytes = 'blob'")
            .await
            .unwrap();

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
