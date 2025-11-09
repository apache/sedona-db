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
use arrow_schema::ArrowError;
use datafusion_common::DataFusionError;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RSedonaError {
    #[error("{0}")]
    DF(Box<DataFusionError>),
    #[error("{0}")]
    TokioIO(tokio::io::Error),
    #[error("{0}")]
    TokioJoin(tokio::task::JoinError),
    #[error("{0}")]
    Internal(String),
    #[error("Interrupted")]
    Interrupted,
}

impl From<DataFusionError> for RSedonaError {
    fn from(other: DataFusionError) -> Self {
        RSedonaError::DF(Box::new(other))
    }
}

impl From<tokio::io::Error> for RSedonaError {
    fn from(other: tokio::io::Error) -> Self {
        RSedonaError::TokioIO(other)
    }
}

impl From<tokio::task::JoinError> for RSedonaError {
    fn from(other: tokio::task::JoinError) -> Self {
        RSedonaError::TokioJoin(other)
    }
}

impl From<ArrowError> for RSedonaError {
    fn from(other: ArrowError) -> Self {
        RSedonaError::DF(Box::new(DataFusionError::ArrowError(Box::new(other), None)))
    }
}
