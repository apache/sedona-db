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

Plugin surface: `ZarrFormatSpec(ExternalFormatSpec)` paired with
`con.read_format(spec, uri)`. The SQL UDTF form (`sd_read_zarr`) is
deferred to a follow-up PR.
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


def test_format_spec_via_read_format(zarr_group):
    con = sedonadb.connect()
    df = con.read_format(sedonadb_zarr.ZarrFormatSpec(), f"file://{zarr_group}")
    arrow_tab = df.to_arrow_table()
    assert arrow_tab.num_rows == 2
    assert arrow_tab.column_names == ["raster"]

    raster = arrow_tab["raster"][0].as_py()
    assert isinstance(raster, dict), f"raster row is {type(raster).__name__}"
    for field in ("transform", "bands"):
        assert field in raster, f"raster row missing {field!r}: {sorted(raster)}"
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


def test_format_spec_class_invariants():
    spec = sedonadb_zarr.ZarrFormatSpec()
    assert spec.extension == ".zarr"
    spec2 = spec.with_options({"arrays": ["temperature"]})
    assert spec2 is not spec
    assert spec2._options.get("arrays") == ["temperature"]


# Each numpy dtype below maps to a different `BandDataType` arm in
# `rust/sedona-raster-zarr/src/dtype.rs::zarr_to_band_data_type`.
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_dtype_mapping_roundtrips(tmp_path, numpy_dtype):
    zarr = pytest.importorskip("zarr")
    root = zarr.open_group(str(tmp_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=(2, 2),
        chunks=(1, 2),
        dtype=numpy_dtype,
        dimension_names=["y", "x"],
    )
    arr[:] = np.ones((2, 2), dtype=numpy_dtype)

    con = sedonadb.connect()
    df = con.read_format(sedonadb_zarr.ZarrFormatSpec(), f"file://{tmp_path}")
    assert df.to_arrow_table().num_rows == 2
