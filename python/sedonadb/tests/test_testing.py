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
import tempfile
from pathlib import Path

import geoarrow.pyarrow as ga
import geopandas
import pandas as pd
import pyarrow as pa
import pyproj
import pytest
from sedonadb.testing import DuckDB, PostGIS, SedonaDB


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
def test_assert_result_nonspatial(eng):
    q = "SELECT 'foofy' as f"
    with eng.create_or_skip() as eng:
        eng.assert_query_result(q, "foofy")
        eng.assert_query_result(q, ("foofy",))
        eng.assert_query_result(q, [("foofy",)])
        eng.assert_query_result(q, pd.DataFrame({"f": ["foofy"]}))

        # DataFusion aggressively generates non-nullable outputs
        if eng.name() == "sedonadb":
            eng.assert_query_result(
                q,
                pa.table(
                    [["foofy"]], schema=pa.schema([pa.field("f", pa.string(), False)])
                ),
            )
        else:
            eng.assert_query_result(q, pa.table({"f": ["foofy"]}))

        with pytest.raises(AssertionError):
            eng.assert_query_result(q, "not foofy")

        # Asserting against a string for a result with count != 1 should fail
        with pytest.raises(AssertionError):
            eng.assert_query_result("SELECT 'foofy' as f WHERE false", "not foofy")

        with pytest.raises(AssertionError):
            eng.assert_query_result(q, ("not foofy",))

        # Asserting against a tuple for a result with count != 1 should fail
        with pytest.raises(AssertionError):
            eng.assert_query_result("SELECT 'foofy' as f WHERE false", ("not foofy",))

        with pytest.raises(AssertionError):
            eng.assert_query_result(q, [("not foofy",)])

        with pytest.raises(AssertionError):
            eng.assert_query_result(q, pd.DataFrame({"f": ["not foofy"]}))

        with pytest.raises(AssertionError):
            eng.assert_query_result(q, pa.table({"f": ["not foofy"]}))

        # Because the result is not a GeoDataFrame
        with pytest.raises(AssertionError):
            eng.assert_query_result(q, geopandas.GeoDataFrame({"f": ["foofy"]}))

        with pytest.raises(
            TypeError, match="Can't assert result equality against dict"
        ):
            eng.assert_query_result(q, {})


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
def test_assert_result_spatial(eng):
    q = "SELECT ST_GeomFromText('POINT (0 1)') as geom"
    with eng.create_or_skip() as eng:
        # Check tuples/single-element target (with optional WKT precision)
        eng.assert_query_result(q, "POINT (0 1)")
        eng.assert_query_result(
            "SELECT ST_GeomFromText('POINT (0 1.111111)') as geom",
            "POINT (0 1.1)",
            wkt_precision=1,
        )
        eng.assert_query_result(
            q,
            geopandas.GeoDataFrame(
                {"geom": geopandas.GeoSeries.from_wkt(["POINT (0 1)"])}
            ).set_geometry("geom"),
        )

        # SedonaDB aggressively returns non-nullable literals
        eng.assert_query_result(
            q,
            pa.table(
                [ga.as_wkb(["POINT (0 1)"])],
                schema=pa.schema(
                    [pa.field("geom", ga.wkb(), nullable=not isinstance(eng, SedonaDB))]
                ),
            ),
        )

        with pytest.raises(AssertionError):
            eng.assert_query_result(q, "POINT (0 2)")

        with pytest.raises(AssertionError):
            eng.assert_query_result(
                q,
                geopandas.GeoDataFrame(
                    {"geom": geopandas.GeoSeries.from_wkt(["POINT (0 2)"])}
                ).set_geometry("geom"),
            )

        with pytest.raises(AssertionError):
            eng.assert_query_result(q, pa.table({"geom": ga.as_wkb(["POINT (0 2)"])}))

        with pytest.raises(AssertionError):
            eng.assert_query_result(
                q,
                pa.table({"not_geom": [1]}),
            )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
def test_table_arrow_no_crs(eng):
    with eng.create_or_skip() as eng:
        tab_no_crs = pa.table(
            {"idx": [1, 2], "geom": ga.as_wkb(["POINT (0 1)", "POINT (2 3)"])}
        )
        assert eng.create_table_arrow("tab_no_crs", tab_no_crs) is eng

        eng.assert_query_result("SELECT * FROM tab_no_crs ORDER BY idx", tab_no_crs)

        with pytest.raises(AssertionError):
            eng.assert_query_result(
                "SELECT * FROM tab_no_crs ORDER BY idx DESC", tab_no_crs
            )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_table_arrow_crs(eng):
    with eng.create_or_skip() as eng:
        tab_no_crs = pa.table(
            {"idx": [1, 2], "geom": ga.as_wkb(["POINT (0 1)", "POINT (2 3)"])}
        )
        tab_with_crs = tab_no_crs.set_column(
            1, "geom", ga.with_crs(tab_no_crs["geom"], ga.OGC_CRS84)
        )
        tab_with_other_crs = tab_no_crs.set_column(
            1, "geom", ga.with_crs(tab_no_crs["geom"], pyproj.CRS("EPSG:3857"))
        )
        df_no_crs = geopandas.GeoDataFrame.from_arrow(tab_no_crs)
        df_with_crs = geopandas.GeoDataFrame.from_arrow(tab_with_crs)
        df_with_other_crs = geopandas.GeoDataFrame.from_arrow(tab_with_other_crs)

        eng.create_table_arrow("tab_with_crs", tab_with_crs)
        eng.create_table_arrow("tab_no_crs", tab_no_crs)
        eng.create_table_arrow("tab_with_other_crs", tab_with_other_crs)

        # Check against Table
        eng.assert_query_result("SELECT * FROM tab_with_crs ORDER BY idx", tab_with_crs)
        eng.assert_query_result(
            "SELECT * FROM tab_with_other_crs ORDER BY idx", tab_with_other_crs
        )

        # Check against GeoDataFrame
        eng.assert_query_result("SELECT * FROM tab_with_crs ORDER BY idx", df_with_crs)
        eng.assert_query_result(
            "SELECT * FROM tab_with_other_crs ORDER BY idx", df_with_other_crs
        )

        # Check that Table comparison fails on CRS mismatch
        with pytest.raises(AssertionError):
            eng.assert_query_result(
                "SELECT * FROM tab_with_crs ORDER BY idx", tab_no_crs
            )
        with pytest.raises(AssertionError):
            eng.assert_query_result(
                "SELECT * FROM tab_no_crs ORDER BY idx", tab_with_crs
            )
        with pytest.raises(AssertionError):
            eng.assert_query_result(
                "SELECT * FROM tab_with_crs ORDER BY idx", tab_with_other_crs
            )

        # Check that GeoDataFrame comparison fails on CRS mismatch
        with pytest.raises(AssertionError):
            eng.assert_query_result(
                "SELECT * FROM tab_with_crs ORDER BY idx", df_no_crs
            )
        with pytest.raises(AssertionError):
            eng.assert_query_result(
                "SELECT * FROM tab_no_crs ORDER BY idx", df_with_crs
            )
        with pytest.raises(AssertionError):
            eng.assert_query_result(
                "SELECT * FROM tab_with_crs ORDER BY idx", df_with_other_crs
            )

        # ...but we can pass an argument to disable the CRS check
        eng.assert_query_result(
            "SELECT * FROM tab_with_crs ORDER BY idx",
            df_with_other_crs,
            check_crs=False,
        )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
def test_table_arrow_geog(eng):
    with eng.create_or_skip() as eng:
        tab_geometry = pa.table(
            {"idx": [1, 2], "geom": ga.as_wkb(["POINT (0 1)", "POINT (2 3)"])}
        )
        tab_geog = tab_geometry.set_column(
            1, "geom", ga.with_edge_type(tab_geometry["geom"], ga.EdgeType.SPHERICAL)
        )

        eng.create_table_arrow("tab_geometry", tab_geometry)
        eng.create_table_arrow("tab_geog", tab_geog)

        eng.assert_query_result("SELECT * FROM tab_geometry ORDER BY idx", tab_geometry)
        eng.assert_query_result("SELECT * FROM tab_geog ORDER BY idx", tab_geog)

        with pytest.raises(AssertionError):
            eng.assert_query_result("SELECT * FROM tab_geometry ORDER BY idx", tab_geog)
        with pytest.raises(AssertionError):
            eng.assert_query_result("SELECT * FROM tab_geog ORDER BY idx", tab_geometry)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS, DuckDB])
def test_table_parquet(eng):
    with eng.create_or_skip() as eng:
        df = geopandas.GeoDataFrame(
            {
                "idx": [1, 2, 3],
                "geometry": geopandas.GeoSeries.from_wkt(
                    ["POINT (0 1)", "POINT (2 3)", "POINT (4 5)"], crs="EPSG:3857"
                ),
            }
        ).set_geometry("geometry")

        # DuckDB doesn't support CRSes
        if eng.name() == "duckdb":
            check_crs = False
        else:
            check_crs = True

        with tempfile.TemporaryDirectory() as td:
            parquet_file = Path(td) / "df.parquet"
            df.to_parquet(parquet_file)

            # PostGIS doesn't support views
            if eng.name() != "postgis":
                eng.create_view_parquet("test_df", str(parquet_file))
                eng.assert_query_result(
                    "SELECT * FROM test_df ORDER BY idx", df, check_crs=check_crs
                )

            eng.create_table_parquet("test_df", str(parquet_file))
            eng.assert_query_result(
                "SELECT * FROM test_df ORDER BY idx", df, check_crs=check_crs
            )
