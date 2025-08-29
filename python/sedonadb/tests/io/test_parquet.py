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
from pyarrow import parquet
from pathlib import Path
from sedonadb.testing import geom_or_null, SedonaDB, DuckDB


@pytest.mark.parametrize("name", ["water-junc", "water-point"])
def test_read_whole_geoparquet(geoarrow_data, name):
    # Checks a read of some non-trivial files and ensures we match a GeoPandas read
    eng = SedonaDB()
    path = geoarrow_data / "ns-water" / "files" / f"ns-water_{name}_geo.parquet"
    gdf = geopandas.read_parquet(path).sort_values(by="OBJECTID").reset_index(drop=True)

    eng.create_view_parquet("tab", path)
    result = eng.execute_and_collect("""SELECT * FROM tab ORDER BY "OBJECTID";""")
    eng.assert_result(result, gdf)


@pytest.mark.parametrize(
    "name", ["geoparquet-1.0.0", "geoparquet-1.1.0", "overature-bbox", "plain"]
)
def test_read_sedona_testing(sedona_testing, name):
    # Checks a read of trivial files (some GeoParquet and some not) against a DuckDB read
    duckdb = DuckDB.create_or_skip()
    sedonadb = SedonaDB()
    path = sedona_testing / "data" / "parquet" / f"{name}.parquet"
    if not path.exists():
        pytest.skip("submodules/sedona-testing not present or not initialized")

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
def test_read_geoparquet_pruned(geoarrow_data, name):
    # Note that this doesn't check that pruning actually occurred, just that
    # for a query where we should be pruning automatically that we don't omit results.
    eng = SedonaDB()
    path = geoarrow_data / "ns-water" / "files" / f"ns-water_{name}_geo.parquet"
    if not path.exists():
        pytest.skip(
            "submodules/geoarrow-data not present or submodules/download-assets.py not run"
        )

    # Roughly a diamond around Gaspereau Lake, Nova Scotia, in UTM zone 20
    wkt_filter = """
        POLYGON ((
            371000 4978000, 376000 4972000, 381000 4978000,
            376000 4983000, 371000 4978000
        ))
    """
    poly_filter = shapely.from_wkt(wkt_filter)

    gdf = geopandas.read_parquet(path)
    gdf = (
        gdf[gdf.geometry.intersects(poly_filter)]
        .sort_values(by="OBJECTID")
        .reset_index(drop=True)
    )
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
            WHERE ST_Intersects(geometry, ST_SetSRID({geom_or_null(wkt_filter)}, '{gdf.crs.to_json()}'))
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
            WHERE ST_Intersects(geometry, ST_SetSRID({geom_or_null(wkt_filter)}, '{gdf.crs.to_json()}'))
            ORDER BY "OBJECTID";
        """
        )
        eng.assert_result(result, gdf)
