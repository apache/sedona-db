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

use sedona_common::CrsProvider;

use crate::transform::with_global_proj_engine;

#[derive(Debug, Default)]
pub struct ProjCrsProvider {}

impl CrsProvider for ProjCrsProvider {
    fn to_projjson(&self, crs_string: &str) -> datafusion_common::Result<String> {
        with_global_proj_engine(|e| e.engine().to_projjson(crs_string))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn proj_crs_provider() {
        let provider = ProjCrsProvider{};
        let projjson = provider.to_projjson("EPSG:3857").unwrap();
        assert!(
            projjson.starts_with("{"),
            "Unexpected PROJJSON output: {projjson}"
        );
    }
}
