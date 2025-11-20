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

use datafusion::{common::Result, prelude::*};
use sedona::context::{SedonaContext, SedonaDataFrame};

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = SedonaContext::new_local_interactive().await?;
    let url = "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities_geo.parquet";
    let df = ctx.read_parquet(url, Default::default()).await?;
    let output = df
        .sort_by(vec![col("name")])?
        .show_sedona(&ctx, Some(5), Default::default())
        .await?;
    println!("{output}");
    Ok(())
}
