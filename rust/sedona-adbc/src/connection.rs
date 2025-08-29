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
// Because a number of methods only return Err() for not implemented,
// the compiler doesn't know how to guess which impl RecordBatchReader
// will be returned. When we implement the methods, we can remove this.
#![allow(refining_impl_trait)]

use adbc_core::{
    options::{InfoCode, ObjectDepth},
    Connection,
};
use sedona::{context::SedonaContext, reader::SedonaStreamReader};
use std::sync::Arc;
use tokio::runtime::Runtime;

use adbc_core::{
    error::{Error, Result, Status},
    options::{OptionConnection, OptionValue},
    Optionable,
};

use crate::{
    err_not_implemented, err_unrecognized_option, statement::SedonaStatement,
    utils::from_datafusion_error, utils::OptionValueExt,
};

pub struct SedonaConnection {
    runtime: Arc<Runtime>,
    ctx: Arc<SedonaContext>,
    autocommit_on: bool,
}

impl SedonaConnection {
    pub(crate) fn try_new(
        opts: impl IntoIterator<Item = (OptionConnection, OptionValue)>,
    ) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                Error::with_message_and_status(
                    format!("Failed to build multithreaded runtime: {e}"),
                    Status::Internal,
                )
            })?;

        let ctx = runtime.block_on(async {
            SedonaContext::new_local_interactive()
                .await
                .map_err(from_datafusion_error)
        })?;

        let mut connection = Self {
            runtime: Arc::new(runtime),
            ctx: Arc::new(ctx),
            autocommit_on: true,
        };

        for (key, value) in opts {
            connection.set_option(key, value)?;
        }

        Ok(connection)
    }
}

impl Optionable for SedonaConnection {
    type Option = OptionConnection;

    fn set_option(&mut self, key: Self::Option, value: OptionValue) -> Result<()> {
        match &key {
            OptionConnection::AutoCommit => {
                self.autocommit_on = value.as_bool()?;
                Ok(())
            }
            _ => err_unrecognized_option!(key),
        }
    }

    fn get_option_string(&self, key: Self::Option) -> Result<String> {
        match &key {
            OptionConnection::AutoCommit => Ok(if self.autocommit_on {
                "true".to_string()
            } else {
                "false".to_string()
            }),
            _ => err_unrecognized_option!(key),
        }
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

impl Connection for SedonaConnection {
    type StatementType = SedonaStatement;

    fn new_statement(&mut self) -> Result<SedonaStatement> {
        Ok(SedonaStatement::new(self.runtime.clone(), self.ctx.clone()))
    }

    fn cancel(&mut self) -> Result<()> {
        err_not_implemented!()
    }

    fn get_info(
        &self,
        _codes: Option<std::collections::HashSet<InfoCode>>,
    ) -> Result<SedonaStreamReader> {
        err_not_implemented!()
    }

    fn get_objects(
        &self,
        _depth: ObjectDepth,
        _catalog: Option<&str>,
        _db_schema: Option<&str>,
        _table_name: Option<&str>,
        _table_type: Option<Vec<&str>>,
        _column_name: Option<&str>,
    ) -> Result<SedonaStreamReader> {
        err_not_implemented!()
    }

    fn get_table_schema(
        &self,
        _catalog: Option<&str>,
        _db_schema: Option<&str>,
        _table_name: &str,
    ) -> Result<arrow_schema::Schema> {
        err_not_implemented!()
    }

    fn get_table_types(&self) -> Result<SedonaStreamReader> {
        err_not_implemented!()
    }

    fn get_statistic_names(&self) -> Result<SedonaStreamReader> {
        err_not_implemented!()
    }

    fn get_statistics(
        &self,
        _catalog: Option<&str>,
        _db_schema: Option<&str>,
        _table_name: Option<&str>,
        _approximate: bool,
    ) -> Result<SedonaStreamReader> {
        err_not_implemented!()
    }

    fn commit(&mut self) -> Result<()> {
        err_not_implemented!()
    }

    fn rollback(&mut self) -> Result<()> {
        err_not_implemented!()
    }

    fn read_partition(&self, _partition: impl AsRef<[u8]>) -> Result<SedonaStreamReader> {
        err_not_implemented!()
    }
}

#[cfg(test)]
mod test {

    use adbc_core::{Database, Driver};

    use crate::driver::SedonaDriver;

    use super::*;

    #[test]
    fn autocommit() {
        let mut connection = SedonaDriver::default()
            .new_database()
            .unwrap()
            .new_connection()
            .unwrap();

        // Turn autocommit on
        connection
            .set_option(
                OptionConnection::AutoCommit,
                OptionValue::String("true".to_string()),
            )
            .unwrap();
        assert_eq!(
            connection
                .get_option_string(OptionConnection::AutoCommit)
                .unwrap(),
            "true"
        );

        // Turn autocommit off
        connection
            .set_option(
                OptionConnection::AutoCommit,
                OptionValue::String("false".to_string()),
            )
            .unwrap();
        assert_eq!(
            connection
                .get_option_string(OptionConnection::AutoCommit)
                .unwrap(),
            "false"
        );

        // Try to set autocommit with an in appropriate value
        let err = connection
            .set_option(OptionConnection::AutoCommit, OptionValue::Bytes(vec![]))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "InvalidArguments: Expected boolean option (sqlstate: [0, 0, 0, 0, 0], vendor_code: 0)"
        );
    }
}
