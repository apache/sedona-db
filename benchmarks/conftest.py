# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "bench_udf: mark a benchmark as a UDF (micro) benchmark"
    )
    config.addinivalue_line(
        "markers", "bench_query: mark a benchmark as a complex query (macro) benchmark"
    )


@pytest.fixture(autouse=True)
def _apply_bench_mode(request):
    """Apply per-benchmark execution mode engine configuration.

    Logic:
    - If test/function/class has marker bench_udf -> single-thread settings
    - If marker bench_query -> default settings (do nothing)
    - Otherwise do nothing.

    This is intentionally lightweight and only targets engines used in benchmarks
    (SedonaDB + DuckDB). PostGIS currently left unchanged.
    """
    force_single_thread = False
    if request.node.get_closest_marker("bench_udf"):
        force_single_thread = True

    inst = getattr(request, "instance", None)
    if force_single_thread and inst is not None:
        # SedonaDB configuration
        if hasattr(inst, "sedonadb") and getattr(inst, "sedonadb") is not None:
            inst.sedonadb.force_single_thread()

        # DuckDB configuration
        if hasattr(inst, "duckdb") and getattr(inst, "duckdb") is not None:
            inst.duckdb.force_single_thread()

        # PostGIS configuration
        if hasattr(inst, "postgis") and getattr(inst, "postgis") is not None:
            inst.postgis.force_single_thread()

    # No teardown necessary (settings are idempotent per test)
