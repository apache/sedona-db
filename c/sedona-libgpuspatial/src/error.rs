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
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuSpatialError {
    Arrow(ArrowError),
    Init(String),
    PushBuild(String),
    FinishBuild(String),
    PushStream(String),
}

impl From<ArrowError> for GpuSpatialError {
    fn from(value: ArrowError) -> Self {
        GpuSpatialError::Arrow(value)
    }
}

impl fmt::Display for GpuSpatialError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GpuSpatialError::Arrow(error) => {
                write!(f, "{error}")
            }
            GpuSpatialError::Init(errmsg) => {
                write!(f, "Initialization failed: {}", errmsg)
            }
            GpuSpatialError::PushBuild(errmsg) => {
                write!(f, "Push build failed: {}", errmsg)
            }
            GpuSpatialError::FinishBuild(errmsg) => {
                write!(f, "Finish building failed: {}", errmsg)
            }
            GpuSpatialError::PushStream(errmsg) => {
                write!(f, "Push stream failed: {}", errmsg)
            }
        }
    }
}
