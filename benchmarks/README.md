<!-- Licensed to the Apache Software Foundation (ASF) under one
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
under the License. -->

# Running Benchmarks

## pytest-benchmark

These benchmarks provide a convenient way to compare the results of running queries on sedona-db to other engines like DuckDB and postgis.

### Setup

Install pytest-benchmark:
```bash
pip install pytest-benchmark
```

Please also remember to install sedonadb in release mode and not debug mode (avoid using the `-e` mentioned in the development docs). Currently we also need to include the test dependencies.

```bash
pip install "python/sedonadb[test]"
```

### Running benchmarks

The below commands assume your working directory is in `benchmarks`.

```bash
cd benchmarks/
```

Please also make sure you have PostGIS running. Instructions for starting PostGIS using the provided docker image can be found in the [contributors-guide](../docs/contributors-guide.md)

To run a benchmark, simply run the corresponding test function. For example, to run the benchmarks for st_buffer, you can run

```bash
pytest test_functions.py::TestBenchFunctions::test_st_buffer
```

Note: It is recommended to run a single (pytest) benchmark function at a time instead of the whole suite because these benchmarks take a long time. This is because they run multiple iterations by default. For example, it often takes 2-3 minutes to run a single benchmark for a basic function.

Most of the time, you'll also want to group by `param:table` or `func` (function) by using the `--benchmark-group-by=param:table` flag. pytest-benchmark will highlight the "best" value in green (e.g fastest for median, lowest for stddev) and "worse" value in red for each column per each group.

```bash
pytest --benchmark-group-by=param:table test_functions.py::TestBenchFunctions::test_st_buffer
```

You can also reduce the number of columns that display by using the `--benchmark-columns` flag.

```bash
pytest --benchmark-group-by=param:table --benchmark-columns=median,mean,stddev test_functions.py::TestBenchFunctions::test_st_buffer
```

Example output of the last command:

```
----------------------------- benchmark 'table=collections_complex': 3 tests -----------------------------
Name (time in ms)                                  Median                Mean             StdDev
----------------------------------------------------------------------------------------------------------
test_st_buffer[collections_complex-SedonaDB]      87.0095 (1.0)       87.7874 (1.0)       3.7269 (1.0)
test_st_buffer[collections_complex-DuckDB]       440.4810 (5.06)     444.6948 (5.07)     12.1143 (3.25)
test_st_buffer[collections_complex-PostGIS]      864.5841 (9.94)     883.3661 (10.06)    50.4996 (13.55)
----------------------------------------------------------------------------------------------------------

---------------------------- benchmark 'table=collections_simple': 3 tests -----------------------------
Name (time in ms)                                 Median                Mean            StdDev
--------------------------------------------------------------------------------------------------------
test_st_buffer[collections_simple-SedonaDB]      85.8510 (1.0)       86.5050 (1.0)      3.8481 (1.0)
test_st_buffer[collections_simple-DuckDB]       442.6664 (5.16)     444.5187 (5.14)     5.6186 (1.46)
test_st_buffer[collections_simple-PostGIS]      855.3329 (9.96)     854.7194 (9.88)     7.6190 (1.98)
--------------------------------------------------------------------------------------------------------
```

For more details and command line options, refer to the official [pytest-benchmark documentation](https://pytest-benchmark.readthedocs.io/en/latest/usage.html)

### Adding New Benchmarks

There are two types of engines, each type serving a different purpose:

- `SedonaDBSingleThread`, `DuckDBSingleThread`, `PostGISSingleThread`:
  Micro / UDF benchmarks that measure the per-function cost (e.g. ST_Area, ST_Contains). These should run engines in a comparable, single-thread style configuration (where possible) to make function-level performance differences clearer.
- `SedonaDB`, `DuckDB`, `PostGIS`:
  Macro / complex query benchmarks (e.g. KNN joins) that represent perceived end-user performance. Engines run with their default / natural configuration (multi-threading, internal parallelism, etc.).

Please choose the appropriate engines when adding a new benchmark. All existing benchmarks have been annotated accordingly.

Example (UDF micro benchmark in single-thread mode):
```python
import pytest
from sedonadb.testing import SedonaDBSingleThread, DuckDBSingleThread, PostGISSingleThread

@pytest.mark.parametrize("eng", [SedonaDBSingleThread, PostGISSingleThread, DuckDBSingleThread])
def test_st_area(benchmark, eng):
    ...
```

Example (Query / macro benchmark in default mode):
```python
import pytest
from sedonadb.testing import SedonaDB, DuckDB, PostGIS

@pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
def test_knn_performance(benchmark, eng):
    ...
```
