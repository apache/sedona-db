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


import numpy as np
import pytest


def test_random_geometry(con):
    df = con.funcs.table.sd_random_geometry("Point", 5, seed=99873)

    # Ensure we produce the correct number of rows
    assert df.count() == 5

    # Ensure the output is reproducible
    assert df.to_arrow_table() == df.to_arrow_table()


def test_read_zarr(con, tmp_path):
    # `zarr` is an optional fixture dep — skip if not installed. The
    # loader itself is exercised by the Rust-side integration tests;
    # this test verifies the Python wrapper threads arguments through
    # and materialises a raster row into Python.
    zarr = pytest.importorskip("zarr")

    # Build a 2x2 UInt8 array with two chunks, dim_names=["y","x"], inside
    # a group at the temp path. Matches the minimal shape sd_read_zarr
    # expects (2-D array with [y, x] suffix).
    root = zarr.open_group(str(tmp_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=(2, 2),
        chunks=(1, 2),
        dtype="uint8",
        dimension_names=["y", "x"],
    )
    arr[:] = np.array([[10, 11], [20, 21]], dtype=np.uint8)

    # Default read emits OutDb-style rows — `data` is empty,
    # `outdb_uri` carries a chunk anchor. Pixel-byte resolution is
    # deferred to the future RS_EnsureLoaded resolver. Materialise
    # through Arrow so we can inspect the Raster struct.
    df = con.funcs.table.sd_read_zarr(f"file://{tmp_path}")
    arrow_tab = df.to_arrow_table()
    assert arrow_tab.num_rows == 2
    assert arrow_tab.column_names == ["raster"]
    raster = arrow_tab["raster"][0].as_py()
    assert isinstance(raster, dict)
    # The Raster struct exposes at least these top-level fields; their
    # exact contents are covered by the Rust tests.
    for field in ("transform", "bands"):
        assert field in raster, f"raster row missing {field!r}: {sorted(raster)}"

    # Options thread through: rows_per_batch slices the output.
    df = con.funcs.table.sd_read_zarr(f"file://{tmp_path}", rows_per_batch=1)
    assert df.count() == 2

    # load_eager=True is not yet supported — errors with a clear
    # pointer at the future RS_EnsureLoaded resolver.
    with pytest.raises(Exception, match="load_eager"):
        con.funcs.table.sd_read_zarr(f"file://{tmp_path}", load_eager=True).count()
