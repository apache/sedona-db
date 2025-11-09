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

# Working with SQL in SedonaDB

This page details several nuances of using SQL in SedonaDB.

## Creating Arrays of Spatial Types in SQL

When constructing an array of spatial objects (like `ST_POINT`) in SedonaDB, you must use bracket notation `[...]` instead of the standard `ARRAY()` function.

### The Incorrect Method: `ARRAY()`

Attempting to use the `ARRAY()` function to create an array of spatial types is not supported and will result in a planning error. SedonaDB will not recognize `ARRAY` as a valid function for this operation.

```python title="Example (Fails)"
>>> sd.sql("SELECT ARRAY(ST_POINT(1,2), ST_POINT(3,4))")
...
Error during planning: Invalid function 'array'
```

### The Correct Method: Brackets

To correctly build an array, enclose your comma-separated spatial objects in **square brackets `[]`**. This syntax
successfully creates a list containing the spatial data structures.

```python title="Example (Works)"
>>> sd.sql("SELECT [ST_POINT(1,2), ST_POINT(3,4)]").show()
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│            make_array(st_point(Int64(1),Int64(2)),st_point(Int64(3),Int64(4)))           │
│                                           list                                           │
╞══════════════════════════════════════════════════════════════════════════════════════════╡
│ [0101000000000000000000f03f0000000000000040, 010100000000000000000008400000000000001040] │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

This approach correctly instructs SedonaDB to construct an array containing the two `ST_POINT` objects.

## Temporary Views Not Supported in SQL

SedonaDB does not support the `CREATE TEMP VIEW` or `CREATE TEMPORARY VIEW` SQL commands. Executing these statements will result in an error.

Attempting to create a temporary view directly with `sd.sql()` will fail, as shown below.

```python title="Unsupported Example"
>>> sd.sql("CREATE TEMP VIEW b AS SELECT * FROM '/path/to/building.parquet'")
Traceback (most recent call last):
  ...
sedonadb._lib.SedonaError: Temporary views not supported
```

### Recommended Alternative

The correct way to create a view is to load your data and use `to_view()`.

This approach provides the same functionality and is the standard practice in Spark-based environments.

```python title="Working Example"
# Step 1: Load your data into a DataFrame first
>>> building_df = sd.read_parquet("/path/to/building.parquet")

# Step 2: Register the DataFrame as a temporary view
>>> building_df.to_view("b")

# Step 3: You can now successfully query the view using SQL
>>> sd.sql("SELECT * FROM b LIMIT 5").show()
```
