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
    options::{OptionConnection, OptionDatabase, OptionValue},
    Database, Optionable,
};

use crate::{connection::SedonaConnection, err_unrecognized_option};

pub struct SedonaDatabase {}

impl Optionable for SedonaDatabase {
    type Option = OptionDatabase;

    fn set_option(&mut self, key: Self::Option, _value: OptionValue) -> Result<()> {
        err_unrecognized_option!(key)
    }

    fn get_option_string(&self, key: Self::Option) -> Result<String> {
        err_unrecognized_option!(key)
    }

    fn get_option_bytes(&self, key: Self::Option) -> Result<Vec<u8>> {
        err_unrecognized_option!(key)
    }

    fn get_option_int(&self, key: Self::Option) -> Result<i64> {
        err_unrecognized_option!(key)
    }

    fn get_option_double(&self, key: Self::Option) -> Result<f64> {
        err_unrecognized_option!(key)
    }
}

impl Database for SedonaDatabase {
    type ConnectionType = SedonaConnection;

    fn new_connection(&self) -> Result<SedonaConnection> {
        self.new_connection_with_opts([])
    }

    fn new_connection_with_opts(
        &self,
        opts: impl IntoIterator<Item = (OptionConnection, OptionValue)>,
    ) -> Result<SedonaConnection> {
        SedonaConnection::try_new(opts)
    }
}
