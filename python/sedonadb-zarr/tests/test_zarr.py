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


def test_format_spec_via_read_format(zarr_group):
    # The second user-facing surface: `con.read_format(spec, uri)`,
    # which uses ExternalFormatSpec.open_reader -> PyZarrChunkReader's
    # __arrow_c_stream__ to plumb data through.
    con = sedonadb.connect()
    df = con.read_format(
        sedonadb_zarr.ZarrFormatSpec(), f"file://{zarr_group}"
    )
    arrow_tab = df.to_arrow_table()
    assert arrow_tab.num_rows == 2
    assert arrow_tab.column_names == ["raster"]

    # Inspect the raster cell as a Python dict — every row should carry
    # transform + bands + the OutDb anchor URI for each band.
    raster = arrow_tab["raster"][0].as_py()
    assert isinstance(raster, dict), f"raster row is {type(raster).__name__}"
    for field in ("transform", "bands"):
        assert field in raster, f"raster row missing {field!r}: {sorted(raster)}"
    # Bands list shape: one entry per array in the group (here, one).
    assert isinstance(raster["bands"], list) and len(raster["bands"]) >= 1
    band = raster["bands"][0]
    # `data` is empty (OutDb scan); `outdb_uri` points at this chunk.
    assert band.get("data") in (None, b"", bytes()), (
        f"OutDb band should have empty data; got {band.get('data')!r}"
    )
    anchor = band.get("outdb_uri")
    assert anchor and "#array=temperature" in anchor, f"unexpected anchor: {anchor!r}"


def test_format_spec_with_arrays_option(zarr_group):
    con = sedonadb.connect()
    spec = sedonadb_zarr.ZarrFormatSpec().with_options({"arrays": ["temperature"]})
    df = con.read_format(spec, f"file://{zarr_group}")
    assert df.to_arrow_table().num_rows == 2


def test_format_spec_load_eager_errors(zarr_group):
    con = sedonadb.connect()
    spec = sedonadb_zarr.ZarrFormatSpec().with_options({"load_eager": True})
    with pytest.raises(Exception, match=r"load_eager"):
        con.read_format(spec, f"file://{zarr_group}").to_arrow_table()
