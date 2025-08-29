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
    error::Result,
    options::{OptionDatabase, OptionValue},
    Driver, Optionable,
};

use crate::database::SedonaDatabase;

#[derive(Default)]
pub struct SedonaDriver {}

impl Driver for SedonaDriver {
    type DatabaseType = SedonaDatabase;

    fn new_database(&mut self) -> Result<Self::DatabaseType> {
        Ok(Self::DatabaseType {})
    }

    fn new_database_with_opts(
        &mut self,
        opts: impl IntoIterator<Item = (OptionDatabase, OptionValue)>,
    ) -> Result<Self::DatabaseType> {
        let mut database = Self::DatabaseType {};
        for (key, value) in opts {
            database.set_option(key, value)?;
        }
        Ok(database)
    }
}
