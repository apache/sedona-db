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

use datafusion_common::error::Result;
use datafusion_common::exec_err;

/// Validate that a 1-based band index is within `[1, num_bands]`.
pub fn validate_band_index(band_index: i32, num_bands: usize) -> Result<()> {
    if band_index < 1 || band_index as usize > num_bands {
        return exec_err!(
            "Provided band index {} is not in the range [1, {}]",
            band_index,
            num_bands
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_band_index_valid() {
        assert!(validate_band_index(1, 3).is_ok());
        assert!(validate_band_index(2, 3).is_ok());
        assert!(validate_band_index(3, 3).is_ok());
    }

    #[test]
    fn test_validate_band_index_zero() {
        assert!(validate_band_index(0, 3).is_err());
    }

    #[test]
    fn test_validate_band_index_negative() {
        assert!(validate_band_index(-1, 3).is_err());
    }

    #[test]
    fn test_validate_band_index_out_of_range() {
        assert!(validate_band_index(4, 3).is_err());
    }
}
