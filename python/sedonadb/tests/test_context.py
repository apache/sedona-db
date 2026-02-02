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
import json
from pathlib import Path
from typing import Any, Mapping

import geoarrow.pyarrow as ga  # noqa: F401
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import sedonadb
import shapely


def _parse_geo_metadata(geoparquet_path: Path) -> Mapping[str, Any]:
    """Return the GeoParquet "geo" metadata map, asserting it exists."""
    metadata = pq.read_metadata(geoparquet_path).metadata
    assert metadata is not None

    geo = metadata.get(b"geo")
    assert geo is not None

    return json.loads(geo.decode("utf-8"))


def _geom_column_metadata(
    geoparquet_path: Path, column_name: str = "geom"
) -> Mapping[str, Any]:
    geo_metadata = _parse_geo_metadata(geoparquet_path)
    columns = geo_metadata.get("columns")
    assert isinstance(columns, dict)
    assert column_name in columns
    return columns[column_name]


def test_options():
    sd = sedonadb.connect()
    assert "DataFrame object at" in repr(sd.sql("SELECT 1 as one"))

    sd.options.interactive = True
    assert "DataFrame object at" not in repr(sd.sql("SELECT 1 as one"))


def test_read_parquet(con, geoarrow_data):
    # Check one file
    tab = con.read_parquet(
        geoarrow_data / "quadrangles/files/quadrangles_100k_geo.parquet"
    ).to_arrow_table()
    assert tab["geometry"].type.extension_name == "geoarrow.wkb"

    # Check many files
    tab = con.read_parquet(
        geoarrow_data.glob("example/files/*_geo.parquet")
    ).to_arrow_table()
    assert tab["geometry"].type.extension_name == "geoarrow.wkb"
    assert len(tab) == 244


def test_read_parquet_local_glob(con, geoarrow_data):
    # The above test uses .glob() method, this test uses the raw string
    tab = con.read_parquet(
        geoarrow_data / "example/files/*_geo.parquet"
    ).to_arrow_table()
    assert tab["geometry"].type.extension_name == "geoarrow.wkb"
    assert len(tab) == 244

    tab = con.read_parquet(
        geoarrow_data / "example/files/example_polygon-*geo.parquet"
    ).to_arrow_table()
    assert len(tab) == 12


def test_read_parquet_error(con):
    with pytest.raises(sedonadb._lib.SedonaError, match="No table paths were provided"):
        con.read_parquet([])


def test_dynamic_object_stores():
    # We need a fresh connection here to make sure other autoregistration
    # doesn't affect the ability of read_parquet() to use an object store
    con = sedonadb.connect()
    url = "https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/example/files/example_geometry_geo.parquet"

    schema = pa.schema(con.read_parquet(url))
    assert schema.field("geometry").type.extension_name == "geoarrow.wkb"


def test_read_parquet_options_parameter(con, geoarrow_data):
    """Test the options parameter functionality for read_parquet()"""
    test_file = geoarrow_data / "quadrangles/files/quadrangles_100k_geo.parquet"

    # Test 1: Backward compatibility - no options parameter
    tab1 = con.read_parquet(test_file).to_arrow_table()
    assert tab1["geometry"].type.extension_name == "geoarrow.wkb"

    # Test 2: options=None (explicit None)
    tab2 = con.read_parquet(test_file, options=None).to_arrow_table()
    assert tab2["geometry"].type.extension_name == "geoarrow.wkb"
    assert len(tab2) == len(tab1)  # Should be identical

    # Test 3: Empty options dictionary
    tab3 = con.read_parquet(test_file, options={}).to_arrow_table()
    assert tab3["geometry"].type.extension_name == "geoarrow.wkb"
    assert len(tab3) == len(tab1)  # Should be identical

    # Test 4: Options with string values
    tab4 = con.read_parquet(
        test_file, options={"test.option": "value"}
    ).to_arrow_table()
    assert tab4["geometry"].type.extension_name == "geoarrow.wkb"
    assert len(tab4) == len(
        tab1
    )  # Should be identical (option ignored but not errored)


# Basic test for `geometry_columns` option for `read_parquet(..)`
def test_read_parquet_geometry_columns_roundtrip(con, tmp_path):
    # Write a regular Parquet table with a Binary WKB column.
    geom = shapely.from_wkt("POINT (0 1)").wkb
    table = pa.table({"id": [1], "geom": [geom]})
    src = tmp_path / "plain.parquet"
    pq.write_table(table, src)

    # GeoParquet metadata should not be present.
    metadata = pq.read_metadata(src).metadata
    assert metadata is not None
    assert b"geo" not in metadata

    # Test 1: when adding a new geometry column, `encoding` must be provided.
    geometry_columns = json.dumps({"geom": {"crs": "EPSG:4326"}})
    with pytest.raises(
        sedonadb._lib.SedonaError,
        match="Geometry column 'geom' missing required key 'encoding'",
    ):
        con.read_parquet(src, geometry_columns=geometry_columns)

    # Test 2: mark 'geom' as geometry and round-trip to GeoParquet.
    geometry_columns = json.dumps({"geom": {"encoding": "WKB"}})
    df = con.read_parquet(src, geometry_columns=geometry_columns)
    out_geo1 = tmp_path / "geo1.parquet"
    df.to_parquet(out_geo1)

    geom_meta = _geom_column_metadata(out_geo1)
    assert geom_meta["encoding"] == "WKB"

    # Test 3: `geom` already geometry; add a missing CRS.
    geometry_columns = json.dumps({"geom": {"crs": "EPSG:3857"}})
    df = con.read_parquet(out_geo1, geometry_columns=geometry_columns)
    out_geo2 = tmp_path / "geo2.parquet"
    df.to_parquet(out_geo2)

    geom_meta = _geom_column_metadata(out_geo2)
    assert geom_meta["encoding"] == "WKB"
    assert geom_meta["crs"] == "EPSG:3857"

    # Test 4: conflicting CRS override should error.
    geometry_columns = json.dumps({"geom": {"crs": "EPSG:4326"}})
    with pytest.raises(
        sedonadb._lib.SedonaError,
        match="Geometry column 'geom' override conflicts with existing 'crs' value",
    ):
        con.read_parquet(out_geo2, geometry_columns=geometry_columns)

    # Test 5: adding `geometry_types` is allowed.
    geometry_columns = json.dumps({"geom": {"geometry_types": ["Point"]}})
    df = con.read_parquet(out_geo2, geometry_columns=geometry_columns)
    tab = df.to_arrow_table()
    assert tab["geom"].type.extension_name == "geoarrow.wkb"

    # Test 6: specify multiple options on plain Parquet input.
    geometry_columns = json.dumps(
        {
            "geom": {
                "encoding": "WKB",
                "crs": "EPSG:3857",
                "edges": "spherical",
                "geometry_types": ["Point"],
            }
        }
    )
    df = con.read_parquet(src, geometry_columns=geometry_columns)
    out_geo_multi = tmp_path / "geo_multi.parquet"
    df.to_parquet(out_geo_multi)
    geom_meta = _geom_column_metadata(out_geo_multi)
    assert geom_meta["encoding"] == "WKB"
    assert geom_meta["crs"] == "EPSG:3857"
    assert geom_meta["edges"] == "spherical"


def test_read_parquet_geometry_columns_multiple_columns(con, tmp_path):
    # Write a regular Parquet table with two Binary WKB columns.
    geom1 = shapely.from_wkt("POINT (0 1)").wkb
    geom2 = shapely.from_wkt("POINT (1 2)").wkb
    table = pa.table({"id": [1], "geom1": [geom1], "geom2": [geom2]})
    src = tmp_path / "plain_multi.parquet"
    pq.write_table(table, src)

    # Mark geom1 as geometry and write GeoParquet.
    geometry_columns = json.dumps({"geom1": {"encoding": "WKB"}})
    df = con.read_parquet(src, geometry_columns=geometry_columns)
    out_geo1 = tmp_path / "geo_multi1.parquet"
    df.to_parquet(out_geo1)

    geo_metadata = _parse_geo_metadata(out_geo1)
    assert "geom1" in geo_metadata["columns"]
    assert "geom2" not in geo_metadata["columns"]

    # Mark geom2 as geometry and add a missing CRS to geom1 in one call.
    geometry_columns = json.dumps(
        {
            "geom1": {"crs": "EPSG:3857"},
            "geom2": {"encoding": "WKB"},
        }
    )
    df = con.read_parquet(out_geo1, geometry_columns=geometry_columns)
    out_geo2 = tmp_path / "geo_multi2.parquet"
    df.to_parquet(out_geo2)

    geom1_meta = _geom_column_metadata(out_geo2, "geom1")
    geom2_meta = _geom_column_metadata(out_geo2, "geom2")
    assert geom1_meta["encoding"] == "WKB"
    assert geom1_meta["crs"] == "EPSG:3857"
    assert geom2_meta["encoding"] == "WKB"


def test_read_geoparquet_s3_anonymous_access():
    """Test reading from a public S3 bucket geoparquet file with anonymous access"""
    con = sedonadb.connect()
    s3_url = "s3://wherobots-examples/data/onboarding_1/nyc_buildings.parquet"

    # Use aws.skip_signature with region for anonymous access
    tab = con.read_parquet(
        s3_url, options={"aws.skip_signature": True, "aws.region": "us-west-2"}
    ).to_arrow_table()
    assert len(tab) > 0
    assert "geom" in tab.schema.names  # This dataset uses 'geom' instead of 'geometry'


def test_read_parquet_invalid_aws_option():
    """Test that invalid AWS options are caught and provide helpful error messages"""
    con = sedonadb.connect()
    url = "s3://wherobots-examples/data/onboarding_1/nyc_buildings.parquet"

    # Test with a misspelled AWS option
    with pytest.raises(
        sedonadb._lib.SedonaError,
        match=r"Unknown AWS option 'aws\.skip_sig'\..*aws\.skip_signature",
    ):
        con.read_parquet(url, options={"aws.skip_sig": "true"})

    # Test with completely unknown AWS option
    with pytest.raises(
        sedonadb._lib.SedonaError,
        match="Unknown AWS option.*aws.unknown_option.*Valid options are",
    ):
        con.read_parquet(url, options={"aws.unknown_option": "value"})


def test_read_parquet_invalid_azure_option():
    con = sedonadb.connect()
    url = "az://container/path/file.parquet"

    with pytest.raises(
        sedonadb._lib.SedonaError,
        match=r"Unknown Azure option 'azure\.acc_name'\..*azure\.account_name",
    ):
        con.read_parquet(url, options={"azure.acc_name": "test"})

    with pytest.raises(
        sedonadb._lib.SedonaError,
        match="Unknown Azure option.*azure.unknown_option.*Valid options are",
    ):
        con.read_parquet(url, options={"azure.unknown_option": "value"})
