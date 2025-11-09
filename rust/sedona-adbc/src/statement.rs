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
use adbc_core::PartitionedResult;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::Schema;
use sedona::{context::SedonaContext, reader::SedonaStreamReader};
use std::sync::Arc;
use tokio::runtime::Runtime;

use adbc_core::{
    error::{Error, Result, Status},
    options::{OptionStatement, OptionValue},
    Optionable, Statement,
};

use crate::{err_not_implemented, err_unrecognized_option, utils::from_datafusion_error};

pub struct SedonaStatement {
    runtime: Arc<Runtime>,
    ctx: Arc<SedonaContext>,
    sql_query: Option<String>,
}

impl SedonaStatement {
    pub(crate) fn new(runtime: Arc<Runtime>, ctx: Arc<SedonaContext>) -> SedonaStatement {
        Self {
            runtime,
            ctx,
            sql_query: None,
        }
    }
}

impl Optionable for SedonaStatement {
    type Option = OptionStatement;

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

impl Statement for SedonaStatement {
    fn set_sql_query(&mut self, query: impl AsRef<str>) -> Result<()> {
        self.sql_query = Some(query.as_ref().to_string());
        Ok(())
    }

    fn prepare(&mut self) -> Result<()> {
        Ok(())
    }

    fn execute_schema(&mut self) -> Result<Schema> {
        if let Some(query) = self.sql_query.clone() {
            self.runtime.block_on(async {
                let df = self.ctx.sql(&query).await.map_err(from_datafusion_error)?;
                Ok(df.schema().as_arrow().clone())
            })
        } else {
            Err(Error::with_message_and_status(
                "query not set yet",
                Status::InvalidState,
            ))
        }
    }

    fn execute(&mut self) -> Result<impl RecordBatchReader + Send> {
        if let Some(query) = self.sql_query.clone() {
            self.runtime.block_on(async {
                let df = self.ctx.sql(&query).await.map_err(from_datafusion_error)?;
                let stream = df.execute_stream().await.map_err(from_datafusion_error)?;
                Ok(SedonaStreamReader::new(self.runtime.clone(), stream))
            })
        } else {
            Err(Error::with_message_and_status(
                "query not set yet",
                Status::InvalidState,
            ))
        }
    }

    fn execute_update(&mut self) -> Result<Option<i64>> {
        err_not_implemented!()
    }

    fn execute_partitions(&mut self) -> Result<PartitionedResult> {
        err_not_implemented!()
    }

    fn bind(&mut self, _batch: RecordBatch) -> Result<()> {
        err_not_implemented!()
    }

    fn bind_stream(&mut self, _reader: Box<dyn RecordBatchReader + Send>) -> Result<()> {
        err_not_implemented!()
    }

    fn get_parameter_schema(&self) -> Result<Schema> {
        err_not_implemented!()
    }

    fn set_substrait_plan(&mut self, _plan: impl AsRef<[u8]>) -> Result<()> {
        err_not_implemented!()
    }

    fn cancel(&mut self) -> Result<()> {
        err_not_implemented!()
    }
}

#[cfg(test)]
mod test {
    use std::ops::Deref;

    use adbc_core::{Connection, Database, Driver, Statement};
    use arrow_array::RecordBatch;
    use arrow_schema::{Field, Schema};
    use datafusion::assert_batches_eq;

    use crate::driver::SedonaDriver;

    #[test]
    fn statement() {
        let mut statement = SedonaDriver::default()
            .new_database()
            .unwrap()
            .new_connection()
            .unwrap()
            .new_statement()
            .unwrap();

        // Can't execute_schema() or execute() before setting a query
        let maybe_err = statement.execute_schema();
        assert_eq!(maybe_err.err().unwrap().message, "query not set yet");
        let maybe_err = statement.execute();
        assert_eq!(maybe_err.err().unwrap().message, "query not set yet");

        statement
            .set_sql_query("SELECT ST_AsText(ST_Point(1, 2)) AS geom")
            .unwrap();

        statement.prepare().unwrap();

        assert_eq!(
            statement.execute_schema().unwrap(),
            Schema::new(vec![Field::new("geom", arrow_schema::DataType::Utf8, true)])
        );

        let batches: Result<Vec<RecordBatch>, _> = statement.execute().unwrap().collect();

        assert_batches_eq!(
            [
                "+------------+",
                "| geom       |",
                "+------------+",
                "| POINT(1 2) |",
                "+------------+",
            ],
            batches.unwrap().deref()
        );
    }
}
