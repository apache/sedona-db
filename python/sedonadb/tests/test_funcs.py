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


def test_random_geometry(con):
    df = con.funcs.table.sd_random_geometry("Point", 5, seed=99873)

    # Ensure we produce the correct number of rows
    assert df.count() == 5

    # Ensure the output is reproducible
    assert df.to_arrow_table() == df.to_arrow_table()


def test_read_zarr(con, tmp_path):
    # Skip cleanly if the optional `zarr` Python lib isn't installed —
    # the binding is exercised by the Rust-side integration tests; this
    # test only verifies the Python wrapper threads arguments through.
    import pytest

    zarr = pytest.importorskip("zarr")
    np = pytest.importorskip("numpy")

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

    # Default-mode (InDb) read — every chunk materialized into the
    # Arrow `data` column.
    df = con.funcs.table.sd_read_zarr(f"file://{tmp_path}")
    assert df.count() == 2

    # Options thread through: rows_per_batch slices the output.
    df = con.funcs.table.sd_read_zarr(f"file://{tmp_path}", rows_per_batch=1)
    assert df.count() == 2
