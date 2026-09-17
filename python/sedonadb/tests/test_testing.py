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
import struct
import tempfile
from pathlib import Path

import geoarrow.pyarrow as ga
import geopandas
import pandas as pd
import pyarrow as pa
import pyproj
import pytest
import shapely
from sedonadb.testing import DuckDB, PostGIS, SedonaDB, _row_level_crs


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


def _ewkb_point(x, y, srid):
    """A little-endian EWKB point carrying `srid` (0 for no SRID).

    Written with shapely, like the ST_AsEWKB tests in
    `tests/functions/test_wkb.py`, rather than packing the bytes by hand.
    """

    point = shapely.Point(x, y)
    if srid:
        point = shapely.set_srid(point, srid)
    return shapely.to_wkb(
        point, byte_order=1, flavor="extended", include_srid=bool(srid)
    )


def test_row_level_crs_is_distinct_from_column_crs():
    """Only a *row-level* CRS is surfaced per row — a per-geometry SRID in the
    WKB bytes (how Sedona Spark encodes it) or SedonaDB's item-level
    ``struct<item, crs>``. A column-level (type) CRS is deliberately excluded,
    so geometry that carries its CRS at the wrong level renders differently
    instead of being normalized into agreement."""
    if shapely.geos_version < (3, 12, 0):
        pytest.skip("GEOS version 3.12+ required for EWKB tests")

    def epsg(col):
        return [c.to_epsg() if c else None for c in _row_level_crs(col)]

    # 1. Per-geometry SRID in the WKB bytes, differing per row (+ a bare row).
    ewkb = ga.wkb().wrap_array(
        pa.array(
            [
                _ewkb_point(1.0, 2.0, 4326),
                _ewkb_point(3.0, 4.0, 3857),
                _ewkb_point(5.0, 6.0, 0),
            ],
            pa.binary(),
        )
    )
    assert epsg(ewkb) == [4326, 3857, None]

    # 2. SedonaDB's item-level struct<item, crs>, per row ('0' means no CRS).
    item = ga.as_wkb(["POINT (0 1)", "POINT (2 3)"])
    struct = pa.StructArray.from_arrays(
        [item, pa.array(["EPSG:4326", "0"])], names=["item", "crs"]
    )
    assert epsg(struct) == [4326, None]

    # 3. A column-level geoarrow CRS is NOT a row-level CRS: this column has
    # no per-geometry CRS, so it renders as bare WKT rather than (crs, wkt).
    column = ga.with_crs(
        ga.as_wkb(["POINT (0 1)", "POINT (2 3)"]), pyproj.CRS("EPSG:3857").to_json()
    )
    assert _row_level_crs(column) is None

    # ...as does geometry with no CRS anywhere, and a non-geometry column.
    assert _row_level_crs(ga.as_wkb(["POINT (0 1)"])) is None
    assert _row_level_crs(pa.chunked_array([pa.array([1, 2])])) is None


def test_row_level_crs_tolerates_geos_invalid_wkb():
    """Reading the CRS must not blow up on geometry GEOS rejects.

    Some geometry this harness compares happily as WKT is invalid to GEOS —
    e.g. the unclosed LinearRing ST_TessellateGeog can produce. A plain
    shapely.from_wkb raises there and failed the whole comparison, hence
    on_invalid="fix", which still yields the SRID.
    """
    # A polygon whose ring is not closed: GEOS refuses to construct it.
    unclosed_ring = struct.pack("<BIII", 1, 3, 1, 3) + b"".join(
        struct.pack("<dd", x, y) for x, y in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    )
    with pytest.raises(shapely.errors.GEOSException):
        shapely.from_wkb(unclosed_ring)

    col = ga.wkb().wrap_array(pa.array([unclosed_ring], pa.binary()))
    # No SRID in these bytes — and crucially, no exception.
    assert _row_level_crs(col) is None
