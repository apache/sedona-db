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
use arrow_schema::SchemaRef;
use futures::Stream;
use futures::TryStreamExt;
use sedona_expr::projection::unwrap_batch;
use sedona_expr::projection::unwrap_expressions;
use sedona_expr::projection::wrap_expressions;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use arrow_array::RecordBatch;
use datafusion::error::Result;
use datafusion::execution::RecordBatchStream;
use datafusion::prelude::DataFrame;
use datafusion::{common::DFSchema, execution::SendableRecordBatchStream};
use sedona_schema::projection::unwrap_schema_maybe_deprecated;

/// Possibly project a DataFrame such that the output expresses extension types as data types
///
/// This is a "lazy" version of wrap_arrow_batch() that appends a projection to a DataFrame.
pub fn wrap_df(df: DataFrame) -> Result<DataFrame> {
    if let Some(exprs) = wrap_expressions(df.schema())? {
        df.select(exprs)
    } else {
        Ok(df)
    }
}

/// Possibly project a DataFrame such that the output expresses extension types as data types
///
/// This is a "lazy" version of unwrap_arrow_batch() that appends a projection to a DataFrame.
pub fn unwrap_df(df: DataFrame) -> Result<(DFSchema, DataFrame)> {
    if let Some((schema, exprs)) = unwrap_expressions(df.schema())? {
        Ok((schema, df.select(exprs)?))
    } else {
        Ok((df.schema().clone(), df))
    }
}

/// Possibly project a SendableRecordBatchStream such that the output expresses extension
/// types as data types
pub fn unwrap_stream(stream: SendableRecordBatchStream) -> SendableRecordBatchStream {
    let wrapper = UnwrapRecordBatchStream { parent: stream };
    Box::pin(wrapper)
}

struct UnwrapRecordBatchStream {
    parent: SendableRecordBatchStream,
}

impl RecordBatchStream for UnwrapRecordBatchStream {
    fn schema(&self) -> SchemaRef {
        Arc::new(unwrap_schema_maybe_deprecated(&self.parent.schema()))
    }
}

impl Stream for UnwrapRecordBatchStream {
    type Item = Result<RecordBatch>;
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        match self.parent.try_poll_next_unpin(cx) {
            Poll::Ready(maybe_parent) => {
                Poll::Ready(maybe_parent.map(|parent| parent.map(unwrap_batch)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, record_batch, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use sedona_expr::projection::wrap_batch;
    use sedona_schema::extension_type::ExtensionType;

    use super::*;

    /// An ExtensionType for tests
    pub fn geoarrow_wkt() -> ExtensionType {
        ExtensionType::new("geoarrow.wkt", DataType::Utf8, None)
    }

    #[tokio::test]
    async fn df_wrap_unwrap() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("col1", DataType::Utf8, true),
            geoarrow_wkt().to_field("col2", true),
        ]);
        let col1 = create_array!(Utf8, ["POINT (0 1)", "POINT (2 3)"]);
        let col2 = col1.clone();

        let batch_no_extensions = record_batch!(("col1", Utf8, ["POINT (0 1)", "POINT (2 3)"]))?;
        let batch = RecordBatch::try_new(schema.clone().into(), vec![col1, col2])?;

        let ctx = SessionContext::new();

        // A batch with no extensions should be unchanged by wrap_df()
        let df_no_extensions = wrap_df(ctx.read_batch(batch_no_extensions.clone())?)?;
        let results_no_extensions = df_no_extensions.clone().collect().await?;
        assert_eq!(results_no_extensions.len(), 1);
        assert_eq!(results_no_extensions[0], batch_no_extensions);

        // A batch with no extensions should be unchanged by unwrap_df()
        let (schema_roundtrip_no_extensions, roundtrip_no_extensions) =
            unwrap_df(df_no_extensions.clone())?;
        assert_eq!(&schema_roundtrip_no_extensions, df_no_extensions.schema());
        assert_eq!(
            roundtrip_no_extensions.collect().await?[0],
            batch_no_extensions
        );

        // A batch with extensions should have extension fields wrapped as structs by df_wrap()
        let df = wrap_df(ctx.read_batch(batch.clone())?)?;
        let results = df.clone().collect().await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], wrap_batch(batch.clone()));

        // unwrap_df() will result in a batch with no extensions in the results
        // (but with the extension information communicated in the returned schema)
        let batch_without_extensions = record_batch!(
            ("col1", Utf8, ["POINT (0 1)", "POINT (2 3)"]),
            ("col2", Utf8, ["POINT (0 1)", "POINT (2 3)"])
        )?;
        let (schema_roundtrip, roundtrip) = unwrap_df(df)?;
        assert_eq!(schema_roundtrip.as_arrow(), &schema);

        assert_eq!(roundtrip.collect().await?[0], batch_without_extensions);

        Ok(())
    }
}
