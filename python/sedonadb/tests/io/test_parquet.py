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
import tempfile
import shapely
import geopandas
import geopandas.testing
from pyarrow import parquet
from pathlib import Path
from sedonadb.testing import geom_or_null, SedonaDB, DuckDB, skip_if_not_exists


@pytest.mark.parametrize("name", ["water-junc", "water-point"])
def test_read_whole_geoparquet(geoarrow_data, name):
    # Checks a read of some non-trivial files and ensures we match a GeoPandas read
    eng = SedonaDB()
    path = geoarrow_data / "ns-water" / "files" / f"ns-water_{name}_geo.parquet"
    skip_if_not_exists(path)

    gdf = geopandas.read_parquet(path).sort_values(by="OBJECTID").reset_index(drop=True)

    eng.create_view_parquet("tab", path)
    result = eng.execute_and_collect("""SELECT * FROM tab ORDER BY "OBJECTID";""")
    eng.assert_result(result, gdf)


@pytest.mark.parametrize("name", ["geoparquet-1.0.0", "geoparquet-1.1.0", "plain"])
def test_read_sedona_testing(sedona_testing, name):
    # Checks a read of trivial files (some GeoParquet and some not) against a DuckDB read
    duckdb = DuckDB.create_or_skip()
    sedonadb = SedonaDB()
    path = sedona_testing / "data" / "parquet" / f"{name}.parquet"
    skip_if_not_exists(path)

    duckdb.create_view_parquet("tab", path)
    result_duckdb = duckdb.execute_and_collect("SELECT * FROM tab")
    df_duckdb = duckdb.result_to_pandas(result_duckdb)

    # DuckDB never returns CRSes
    kwargs = {}
    if isinstance(df_duckdb, geopandas.GeoDataFrame):
        kwargs["check_crs"] = False

    sedonadb.create_view_parquet("tab", path)
    sedonadb.assert_query_result("SELECT * FROM tab", df_duckdb, **kwargs)


@pytest.mark.parametrize("name", ["water-junc", "water-point"])
@pytest.mark.parametrize(
    "predicate",
    [
        "intersects",
        "coveredby",
        "within",
        "equals",
        "disjoint",
    ],
)
def test_read_geoparquet_prune_points(geoarrow_data, name, predicate):
    # Note that this doesn't check that pruning actually occurred, just that
    # for a query where we should be pruning automatically that we don't omit results.
    eng = SedonaDB()
    path = geoarrow_data / "ns-water" / "files" / f"ns-water_{name}_geo.parquet"
    skip_if_not_exists(path)

    gdf = geopandas.read_parquet(path)

    # Roughly a diamond around Gaspereau Lake, Nova Scotia, in UTM zone 20
    wkt_filter = """
        POLYGON ((
            371000 4978000, 376000 4972000, 381000 4978000,
            376000 4983000, 371000 4978000
        ))
    """

    # Use a wkt_filter that will lead to non-empty results
    if predicate == "equals":
        wkt_filter = gdf.geometry.iloc[0].wkt

    poly_filter = shapely.from_wkt(wkt_filter)

    if predicate == "equals":
        # Geopandas does not have an equals predicate, so we use the == operator
        mask = gdf.geometry == poly_filter
    elif predicate == "coveredby":
        mask = gdf.geometry.covered_by(poly_filter)
    else:
        mask = getattr(gdf.geometry, predicate)(poly_filter)

    gdf = gdf[mask].sort_values(by="OBJECTID").reset_index(drop=True)
    gdf = gdf[["OBJECTID", "geometry"]]

    # Make sure this isn't a bogus test
    assert len(gdf) > 0

    with tempfile.TemporaryDirectory() as td:
        # Write using GeoPandas, which implements GeoParquet 1.1 bbox covering
        # Write tiny row groups so that many bounding boxes have to be checked
        tmp_parquet = Path(td) / f"{name}.parquet"
        geopandas.read_parquet(path).to_parquet(
            tmp_parquet,
            schema_version="1.1.0",
            write_covering_bbox=True,
            row_group_size=1024,
        )

        eng.create_view_parquet("tab", tmp_parquet)
        result = eng.execute_and_collect(
            f"""
            SELECT "OBJECTID", geometry FROM tab
            WHERE ST_{predicate}(geometry, ST_SetCRS({geom_or_null(wkt_filter)}, '{gdf.crs.to_json()}'))
            ORDER BY "OBJECTID";
        """
        )
        eng.assert_result(result, gdf)

        # Write a dataset with one file per row group to check file pruning correctness
        ds_dir = Path(td) / "ds"
        ds_dir.mkdir()
        ds_paths = []

        with parquet.ParquetFile(tmp_parquet) as f:
            for i in range(f.metadata.num_row_groups):
                tab = f.read_row_group(i, ["OBJECTID", "geometry"])
                df = geopandas.GeoDataFrame.from_arrow(tab)
                ds_path = ds_dir / f"file{i}.parquet"
                df.to_parquet(ds_path)
                ds_paths.append(ds_path)

        # Check a query against the same dataset without the bbox column but with file-level
        # geoparquet metadata bounding boxes
        eng.create_view_parquet("tab_dataset", ds_paths)
        result = eng.execute_and_collect(
            f"""
            SELECT * FROM tab_dataset
            WHERE ST_{predicate}(geometry, ST_SetCRS({geom_or_null(wkt_filter)}, '{gdf.crs.to_json()}'))
            ORDER BY "OBJECTID";
        """
        )
        eng.assert_result(result, gdf)


@pytest.mark.parametrize(
    "predicate",
    [
        "contains",
        "covers",
        "touches",
    ],
)
def test_read_geoparquet_prune_polygons(sedona_testing, predicate):
    # Note that this doesn't check that pruning actually occurred, just that
    # for a query where we should be pruning automatically that we don't omit results.
    eng = SedonaDB()
    path = sedona_testing / "data" / "parquet" / "geoparquet-1.0.0.parquet"
    skip_if_not_exists(path)

    # A point inside of a polygon for contains / covers
    wkt_filter = "POINT (33.60 -5.54)"

    if predicate == "touches":
        # A point on the boundary of a polygon
        wkt_filter = "POINT (33.90371119710453 -0.9500000000000001)"

    poly_filter = shapely.from_wkt(wkt_filter)

    gdf = geopandas.read_parquet(path)
    if predicate == "equals":
        # Geopandas does not have an equals predicate, so we use the == operator
        mask = gdf.geometry == poly_filter
    elif predicate == "coveredby":
        mask = gdf.geometry.covered_by(poly_filter)
    else:
        mask = getattr(gdf.geometry, predicate)(poly_filter)

    gdf = gdf[mask].sort_values(by="pop_est").reset_index(drop=True)
    gdf = gdf[["pop_est", "geometry"]]

    # Make sure this isn't a bogus test
    assert len(gdf) > 0

    with tempfile.TemporaryDirectory() as td:
        # Write using GeoPandas, which implements GeoParquet 1.1 bbox covering
        # Write tiny row groups so that many bounding boxes have to be checked
        tmp_parquet = Path(td) / "geoparquet-1.0.0.parquet"
        geopandas.read_parquet(path).to_parquet(
            tmp_parquet,
            schema_version="1.1.0",
            write_covering_bbox=True,
            row_group_size=1024,
        )

        eng.create_view_parquet("tab", tmp_parquet)
        result = eng.execute_and_collect(
            f"""
            SELECT "pop_est", geometry FROM tab
            WHERE ST_{predicate}(geometry, ST_SetCRS({geom_or_null(wkt_filter)}, '{gdf.crs.to_json()}'))
            ORDER BY "pop_est";
        """
        )
        eng.assert_result(result, gdf)

        # Write a dataset with one file per row group to check file pruning correctness
        ds_dir = Path(td) / "ds"
        ds_dir.mkdir()
        ds_paths = []

        with parquet.ParquetFile(tmp_parquet) as f:
            for i in range(f.metadata.num_row_groups):
                tab = f.read_row_group(i, ["pop_est", "geometry"])
                df = geopandas.GeoDataFrame.from_arrow(tab)
                ds_path = ds_dir / f"file{i}.parquet"
                df.to_parquet(ds_path)
                ds_paths.append(ds_path)

        # Check a query against the same dataset without the bbox column but with file-level
        # geoparquet metadata bounding boxes
        eng.create_view_parquet("tab_dataset", ds_paths)
        result = eng.execute_and_collect(
            f"""
            SELECT * FROM tab_dataset
            WHERE ST_{predicate}(geometry, ST_SetCRS({geom_or_null(wkt_filter)}, '{gdf.crs.to_json()}'))
            ORDER BY "pop_est";
        """
        )
        eng.assert_result(result, gdf)


@pytest.mark.parametrize("name", ["water-junc", "water-point"])
def test_write_geoparquet_geometry(con, geoarrow_data, name):
    # Checks a read and write of some non-trivial files and ensures we match GeoPandas
    path = geoarrow_data / "ns-water" / "files" / f"ns-water_{name}_geo.parquet"
    skip_if_not_exists(path)

    gdf = geopandas.read_parquet(path).sort_values(by="OBJECTID").reset_index(drop=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_parquet = Path(td) / "tmp.parquet"
        con.create_data_frame(gdf).to_parquet(tmp_parquet)

        gdf_roundtrip = geopandas.read_parquet(tmp_parquet)
        geopandas.testing.assert_geodataframe_equal(gdf_roundtrip, gdf)


def test_write_geoparquet_geography(con, geoarrow_data):
    # Checks a read and write of geography (rounctrip, since nobody else can read/write)
    path = (
        geoarrow_data
        / "natural-earth"
        / "files"
        / "natural-earth_countries-geography_geo.parquet"
    )
    skip_if_not_exists(path)

    table = con.read_parquet(path).to_arrow_table()

    with tempfile.TemporaryDirectory() as td:
        tmp_parquet = Path(td) / "tmp.parquet"
        con.create_data_frame(table).to_parquet(tmp_parquet)

        table_roundtrip = con.read_parquet(tmp_parquet).to_arrow_table()
        assert table_roundtrip == table
