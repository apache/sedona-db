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

"""Tests for the `sedonadb-zarr` plugin.

Three flavours of test together establish the plugin contract:

1. **Smoke**: after `register(con)`, `sd_read_zarr` works as a SQL UDTF.
2. **Negative**: before `register(con)`, `sd_read_zarr` is *not* a known
   SQL function. Proves opt-in: importing `sedonadb_zarr` alone doesn't
   register anything.
3. **Idempotence**: calling `register(con)` twice doesn't error or
   double-register pathologically.
"""

import numpy as np
import pytest

import sedonadb
import sedonadb_zarr


@pytest.fixture
def zarr_group(tmp_path):
    """Build a tiny 2x2 UInt8 Zarr v3 group with two chunks."""
    zarr = pytest.importorskip("zarr")
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


def test_smoke_register_enables_sql_udtf(zarr_group):
    con = sedonadb.connect()
    sedonadb_zarr.register(con)
    df = con.sql(f"SELECT count(*) FROM sd_read_zarr('file://{zarr_group}')")
    arrow_tab = df.to_arrow_table()
    assert arrow_tab.num_rows == 1
    assert arrow_tab.column(0)[0].as_py() == 2


def test_sql_udtf_is_not_registered_before_register_is_called(zarr_group):
    # Importing `sedonadb_zarr` (already done at module top) must NOT
    # register anything globally — registration is per-context and
    # explicit. A fresh connection without `register(con)` should fail
    # the SQL with a planner-level "unknown function" error.
    con = sedonadb.connect()
    with pytest.raises(Exception, match=r"sd_read_zarr|function|table function"):
        con.sql(f"SELECT * FROM sd_read_zarr('file://{zarr_group}')")


def test_register_is_idempotent(zarr_group):
    con = sedonadb.connect()
    sedonadb_zarr.register(con)
    # Second registration should not error (the underlying datafusion
    # `register_udtf` overwrites by name, which is fine).
    sedonadb_zarr.register(con)
    df = con.sql(f"SELECT count(*) FROM sd_read_zarr('file://{zarr_group}')")
    assert df.to_arrow_table().num_rows == 1


def test_arrays_option_threads_through_sql(zarr_group):
    con = sedonadb.connect()
    sedonadb_zarr.register(con)
    df = con.sql(
        f"SELECT count(*) FROM sd_read_zarr("
        f"'file://{zarr_group}', '{{\"arrays\":[\"temperature\"]}}')"
    )
    assert df.to_arrow_table().column(0)[0].as_py() == 2


def test_format_spec_constructs_and_threads_options():
    # End-to-end `con.read_format(ZarrFormatSpec(), uri)` is gated on
    # directory-format support in `ExternalFormatSpec`'s ListingTableUrl
    # path — Zarr groups are directories, not single files, so the
    # listing returns zero objects. This test pins down the Python
    # surface shape so the plumbing doesn't regress while the
    # directory-listing gap is being closed.
    spec = sedonadb_zarr.ZarrFormatSpec()
    assert spec.extension == ".zarr"
    spec2 = spec.with_options({"arrays": ["temperature"]})
    assert spec2 is not spec
    assert spec2._options.get("arrays") == ["temperature"]
