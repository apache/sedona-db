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

### Setup

Install pytest-benchmark:
```bash
pip install pytest-benchmark
```

### Running benchmarks

The below commands assume your working directory is in `benchmarks`.

```bash
cd benchmarks/
```

To run a benchmark, simply run the corresponding test function. For example, to run the benchmarks for st_buffer, you can run

```bash
pytest test_functions.py::TestBenchFunctions::test_st_buffer
```

Most of the time, you'll also want to group by `param:table` or `func` (function) by using the `--benchmark-group-by=param:table` flag.

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
