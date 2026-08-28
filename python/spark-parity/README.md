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

# SedonaDB vs Sedona Spark parity tests

Sedona Spark is the compatibility target for SedonaDB's SQL surface. These tests
broadcast one shared SQL string to both engines and compare the results
strictly, so they answer the parity question directly rather than pinning either
engine against a fixed expected value. SedonaDB's own correctness is covered by
the rasterio-oracle tests in `python/sedonadb/tests`.

## Why this is a separate suite

Running these needs pyspark, a JVM, and (on a cold Ivy cache) a Maven download of
the Sedona jars — tens of seconds of startup that most contributors have no
reason to pay. Rather than ship that as skip logic inside the main suite, where
it silently degrades to "passed" for anyone without a Spark toolchain, the tests
live here and are run deliberately.

Nothing else collects this directory: every other pytest invocation in the repo
is scoped to its own project directory, so `python/sedonadb`'s suite and this one
never see each other. There is deliberately no opt-in environment variable — each
test constructs both engines outright, so a missing pyspark, JVM, or jar is a
failure with a real traceback, not a skip.

This suite is not wired into CI.

## Running

```bash
pip install -e "python/sedonadb[test]"      # the engine under test
pip install "pyspark>=4.0" apache-sedona    # the compatibility target
cd python/spark-parity
pytest -v
```

Spark 4.0 is the floor: results leave the JVM through `DataFrame.toArrow`, which
does not exist before then. `SedonaSpark` checks this at session setup and fails
with an explicit message rather than dying later on a missing attribute.

Useful knobs, both read by `sedonadb.testing_spark`:

- `SEDONADB_SEDONA_SPARK_PACKAGES` — full Maven coordinates, to test against a
  Sedona release other than the pinned one.
- `SEDONADB_SPARK_IVY_DIR` — where Ivy resolves those jars (defaults to
  `~/.ivy2`). Pin it to keep one jar cache across runs.

## Conventions

Where the two engines are known to diverge and we intend to close the gap, mark
the case `xfail(reason=...)` rather than deleting or loosening it. The suite then
doubles as a catalog of what to fix, and flips to `xpass` the day the fix lands.
