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
import sedonadb_zarr


@pytest.fixture
def zarr_group(tmp_path):
    """Build a tiny 2x2 UInt8 Zarr v3 group with two chunks.

    The array [[10, 11], [20, 21]] is chunked (1, 2), so it reads as two
    raster rows of 1x2 pixels each (one row per chunk).
    """
    # The fixture uses the zarr-python 3.x API (create_array,
    # dimension_names); zarr 2.x (the newest available on Python < 3.11)
    # can't write these fixtures.
    zarr = pytest.importorskip("zarr", minversion="3.0")
    np = pytest.importorskip("numpy")
    root = zarr.open_group(str(tmp_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=(2, 2),
        chunks=(1, 2),
        dtype="uint8",
        dimension_names=["y", "x"],
    )
    arr[:] = np.array([[10, 11], [20, 21]], dtype=np.uint8)
    return tmp_path


@pytest.fixture
def raster_con(zarr_group):
    """A connection with the `zarr_group` rasters registered as a `rasters` view.

    Uses its own connection (not the shared module-level one) so the view
    doesn't leak into other tests.
    """
    sd = sedonadb.connect()
    sd.register(sedonadb_zarr.ZarrExtension())

    df = sd.read(f"file://{zarr_group}", format="zarr")
    df.to_view("rasters")

    return sd
