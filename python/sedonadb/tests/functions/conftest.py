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

"""Shared fixtures for the RS_* parity test modules.

Parity comparisons are asymmetric (see `sedonadb.raster_testing`): the
`subject` fixture is always SedonaDB — the engine under test — and the
`comparator` fixture parametrizes each test over the engines it is checked
against. Comparator↔comparator agreement is never asserted.
"""

import pytest

from sedonadb.raster_testing import Rasterio, SedonaDB, SedonaSpark

COMPARATORS = [Rasterio, SedonaSpark]


@pytest.fixture()
def subject(con):
    """The engine under test, on the test session's connection."""
    return SedonaDB(con)


@pytest.fixture(params=COMPARATORS, ids=lambda engine: engine.name())
def comparator(request):
    """One comparator engine per parametrization; skips when its backend is
    unavailable (rasterio not installed) or opt-in (Sedona Spark)."""
    return request.param.create_or_skip()
