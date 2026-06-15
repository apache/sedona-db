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
import sedonadb


@pytest.fixture
def raster_con():
    """A connection with a single `RS_Example()` raster registered as `rasters`.

    Uses the in-DB `RS_Example()` raster (64x32, three UInt8 bands) so the
    sedonadb package's RS_ function tests carry no zarr dependency — zarr
    reader behaviors are tested in the sedonadb-zarr package. A dedicated
    connection (not the shared module-level one) keeps the view from leaking
    into other tests.
    """
    con = sedonadb.connect()
    con.sql("SELECT RS_Example() AS raster").to_view("rasters")
    return con
