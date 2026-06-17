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

import geopandas
import geopandas.testing
import pytest

import sedonadb
from sedonadb.datasource import ExternalFormatSpec


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
