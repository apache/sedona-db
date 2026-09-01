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

from pathlib import Path
import threading

import geopandas
import geopandas.testing
import pyarrow as pa
import pytest

import sedonadb
from sedonadb.datasource import ExternalFormatSpec


class ReaderLifecycleState:
    def __init__(self):
        self.lock = threading.Lock()
        self._condition = threading.Condition(self.lock)
        self.active = 0
        self.max_active = 0
        self.opened = 0
        self.closed = 0

    def enter(self):
        with self._condition:
            self.active += 1
            self.opened += 1
            self.max_active = max(self.max_active, self.active)
            self._condition.notify_all()

    def wait_for_peer(self):
        with self._condition:
            if self.active == 1:
                self._condition.wait_for(lambda: self.active > 1, timeout=0.25)

    def exit(self):
        with self._condition:
            self.active -= 1
            self.closed += 1
            self._condition.notify_all()


class TrackingArrowReader:
    def __init__(self, state, src):
        self._state = state
        self._closed = False
        self._state.enter()
        self._state.wait_for_peer()
        value = int(Path(src.to_url()).stem)
        self._reader = pa.RecordBatchReader.from_batches(
            pa.schema({"value": pa.int64()}),
            [pa.record_batch({"value": [value]})],
        )

    def __del__(self):
        if not self._closed:
            self._closed = True
            self._state.exit()

    def __arrow_c_stream__(self, requested_schema=None):
        if requested_schema is None:
            return self._reader.__arrow_c_stream__()
        return self._reader.__arrow_c_stream__(requested_schema)


class TrackingFormatSpec(ExternalFormatSpec):
    def __init__(self, state):
        self._state = state

    @property
    def extension(self):
        return "tracking"

    @property
    def supports_concurrent_file_reads(self):
        return False

    def infer_schema(self, src):
        return pa.schema({"value": pa.int64()})

    def open_reader(self, args):
        return TrackingArrowReader(self._state, args.src)


def test_read_guess_format(con):
    read = con.read

    # Empty list raises ValueError
    with pytest.raises(
        ValueError, match="Can't guess table paths from empty path list"
    ):
        read._guess_format([])

    # No extension raises ValueError
    with pytest.raises(ValueError, match="no item has an extension"):
        read._guess_format(["/path/to/file"])

    # Multiple different extensions raises ValueError
    with pytest.raises(ValueError, match="multiple extensions"):
        read._guess_format(["/path/to/file.parquet", "/path/to/file.fgb"])

    # Single format guesses correctly
    assert read._guess_format(["/path/to/file.parquet"]) == "parquet"
    assert read._guess_format(["/path/to/file.fgb"]) == "fgb"
    assert read._guess_format(["/path/to/file.gpkg"]) == "gpkg"

    # Multiple files with same format works
    assert read._guess_format(["/a.parquet", "/b.parquet"]) == "parquet"

    # URLs with query strings are handled correctly
    assert (
        read._guess_format(["https://example.com/file.parquet?token=abc"]) == "parquet"
    )

    # URLs with fragments are handled correctly
    assert read._guess_format(["https://example.com/file.fgb#section"]) == "fgb"


def test_read_pyogrio_guessed(con, tmp_path):
    # Create a test GeoDataFrame
    gdf = geopandas.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=geopandas.GeoSeries.from_wkt(
            ["POINT (0 1)", "POINT (1 2)", "POINT (2 3)"], crs="EPSG:4326"
        ),
    )

    # Write to FlatGeoBuf
    fgb_path = tmp_path / "test.fgb"
    gdf.to_file(fgb_path)

    # Read using con.read() which should guess the format
    df = con.read(fgb_path).select("id", geometry="wkb_geometry").sort("id")
    geopandas.testing.assert_geodataframe_equal(df.to_pandas(), gdf)


def test_read_parquet_guessed(con, geoarrow_data):
    parquet_path = geoarrow_data / "quadrangles/files/quadrangles_100k_geo.parquet"

    # Read using con.read() which should guess the format
    df = con.read(parquet_path).sort("quadrangle_id")

    geopandas.testing.assert_geodataframe_equal(
        df.to_pandas(), con.read_parquet(parquet_path).sort("quadrangle_id").to_pandas()
    )


class TestFormatSpec(ExternalFormatSpec):
    @property
    def extension(self):
        return "foofy"

    def with_options(self, options):
        raise ValueError("test format spec!")


def test_format_register():
    sd = sedonadb.connect()
    sd.register(TestFormatSpec())

    with pytest.raises(ValueError, match="test format spec!"):
        sd.read("test.foofy", options={"k": "v"})


def test_external_format_serializes_reader_lifecycles(con, tmp_path):
    paths = []
    for value in range(16):
        path = tmp_path / f"{value}.tracking"
        path.write_text("tracking")
        paths.append(path)

    state = ReaderLifecycleState()
    spec = TrackingFormatSpec(state)
    con.sql("SET datafusion.execution.target_partitions TO 8").execute()

    table = con.read(paths, format=spec).to_arrow_table()

    assert table.num_rows == 16
    assert sorted(table.column("value").to_pylist()) == list(range(16))
    assert state.max_active == 1
    assert state.active == 0
    assert state.opened == state.closed
