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

/// Macro to create Sedona Internal Error that avoids the misleading error message from
/// DataFusionError::Internal.
#[macro_export]
macro_rules! sedona_internal_err {
    ($($args:expr),*) => {{
        let msg = std::format!(
            "SedonaDB internal error: {}{}.\nThis issue was likely caused by a bug in SedonaDB's code. \
            Please help us to resolve this by filing a bug report in our issue tracker: \
            https://github.com/apache/sedona-db/issues",
            std::format!($($args),*),
            datafusion_common::DataFusionError::get_back_trace(),
        );
        // We avoid using Internal to avoid the message suggesting it's internal to DataFusion
        Err(datafusion_common::DataFusionError::External(msg.into()))
    }};
}

#[cfg(test)]
mod tests {
    use datafusion_common::DataFusionError;

    #[test]
    fn test_sedona_internal_err() {
        let result: Result<(), DataFusionError> = sedona_internal_err!("Test error: {}", "details");
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_string = err.to_string();
        assert!(err_string.contains("SedonaDB internal error: Test error: details"));

        // Ensure the message doesn't contain the 'DataFusion'-specific messages
        assert!(!err_string.contains("DataFusion's code"));
        assert!(!err_string.contains("https://github.com/apache/datafusion/issues"));
    }
}
