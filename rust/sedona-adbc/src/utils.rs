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
use adbc_core::{
    error::{Error, Result, Status},
    options::OptionValue,
};
use datafusion::error::DataFusionError;

pub(crate) fn from_datafusion_error(value: DataFusionError) -> Error {
    let status = match &value {
        DataFusionError::IoError(_) => Status::IO,
        DataFusionError::NotImplemented(_) => Status::NotImplemented,
        _ => Status::Internal,
    };

    Error::with_message_and_status(value.message(), status)
}

#[macro_export]
macro_rules! err_not_implemented {
    () => {
        Err(Error::with_message_and_status(
            "not implemented",
            Status::NotImplemented,
        ))
    };
    ($msg:expr) => {
        Err(Error::with_message_and_status($msg, Status::NotImplemented))
    };
}

#[macro_export]
macro_rules! err_unrecognized_option {
    ($key:expr) => {
        Err(Error::with_message_and_status(
            format!("Unrecognized option: {:?}", $key),
            Status::NotFound,
        ))
    };
}

pub(crate) trait OptionValueExt {
    fn as_bool(&self) -> Result<bool>;
}

impl OptionValueExt for OptionValue {
    fn as_bool(&self) -> Result<bool> {
        match self {
            OptionValue::Int(value) => Ok(*value != 0),
            OptionValue::String(value) => Ok(value == "true"),
            _ => Err(Error::with_message_and_status(
                "Expected boolean option".to_string(),
                Status::InvalidArguments,
            )),
        }
    }
}
