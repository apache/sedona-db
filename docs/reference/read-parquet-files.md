
<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Reading Parquet Files

To read a Parquet file, you must use the dedicated `sd.read_parquet()` method. You cannot query a file path directly within the `sd.sql()` `FROM` clause.

The `sd.sql()` function is designed to query tables that have already been registered in the session. When you pass a path like `'s3://...'` to `FROM`, the SQL engine searches for a registered table with that literal name and fails when it's not found, producing a `table not found` error.

## Usage

The correct process is a two-step approach:

1. **Load** the Parquet file into a DataFrame using `sd.read_parquet()`.
1. **Register** the DataFrame as a temporary view using `.createOrReplaceTempView()`.
1. **Query** the view using `sd.sql()`.

```python linenums="1" title="Read a parquet file with SedonaDB"

import sedona.db
sd = sedona.db.connect()

df = sd.read_parquet(
    's3://wherobots-benchmark-prod/SpatialBench_sf=1_format=parquet/'
    'building/building.parquet'
)

# Load the Parquet file, which creates a Pandas DataFrame
df = sd.read_parquet('s3://wherobots-benchmark-prod/SpatialBench_sf=1_format=parquet/building/building.parquet')

# Convert the Pandas DataFrame to a Spark DataFrame AND
#    register it as a temporary view in a single line.
spark.createDataFrame(df).createOrReplaceTempView('zone')

# Now, query the view using SQL
sd.sql("SELECT * FROM zone LIMIT 10").show()
```

### Common Errors

Directly using a file path within `sd.sql()` is a common mistake that will result in an error.

**Incorrect Code:**

```python
# This will fail because the SQL engine looks for a table named 's3://...'
sd.sql("SELECT * FROM 's3://wherobots-benchmark-prod/SpatialBench_sf=1_format=parquet/building/building.parquet'")
```

**Resulting Error:**

```bash
sedonadb._lib.SedonaError: Error during planning: table '...s3://...' not found
```
