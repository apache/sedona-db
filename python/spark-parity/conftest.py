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
"""Engine fixtures for the Spark parity suite.

This suite is only ever run deliberately (see README.md), so it assumes both
engines are available: a missing pyspark, JVM, or Sedona jar is a failure, not a
skip. That is the whole reason it lives outside ``python/sedonadb`` — the tests
there must stay runnable by a contributor with no Spark toolchain, which forced
the old opt-in-and-skip dance that this replaces.

Both engines are session-scoped. ``SedonaSpark`` bootstraps one JVM per process
regardless, and ``SedonaDB`` is cheap, so sharing them keeps the fixtures
symmetric and the run fast. Views are per-test and overwrite by name, so tests
do not leak catalog state into each other.
"""

import pytest

from sedonadb.testing import SedonaDB
from sedonadb.testing_spark import SedonaSpark


@pytest.fixture(scope="session")
def sedona() -> SedonaDB:
    """The engine under test."""
    return SedonaDB()


@pytest.fixture(scope="session")
def spark() -> SedonaSpark:
    """The compatibility target. Bootstrapping this costs tens of seconds on the
    first use in a process (JVM startup, and a Maven fetch of the Sedona jars if
    they are not already in the Ivy cache)."""
    return SedonaSpark()


@pytest.fixture(scope="session")
def engines(sedona, spark) -> tuple:
    """Both engines, in comparison order: ``(sedonadb, spark)``."""
    return sedona, spark
