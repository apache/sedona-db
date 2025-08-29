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
use std::{io, num};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SedonaGeometryError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    IO(io::Error),
    #[error("{0}")]
    External(Box<dyn std::error::Error + Send + Sync>),
    #[error("Unknown geometry error")]
    Unknown,
}

impl From<io::Error> for SedonaGeometryError {
    fn from(value: io::Error) -> Self {
        SedonaGeometryError::IO(value)
    }
}

impl From<num::TryFromIntError> for SedonaGeometryError {
    fn from(value: num::TryFromIntError) -> Self {
        SedonaGeometryError::External(Box::new(value))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn errors() {
        let invalid = SedonaGeometryError::Invalid("foofy".to_string());
        assert_eq!(invalid.to_string(), "foofy");

        let some_err = Box::new(std::io::Error::new(std::io::ErrorKind::Deadlock, "foofy"));
        let external = SedonaGeometryError::External(some_err);
        assert_eq!(external.to_string(), "foofy");

        let unknown = SedonaGeometryError::Unknown;
        assert_eq!(unknown.to_string(), "Unknown geometry error");
    }
}
