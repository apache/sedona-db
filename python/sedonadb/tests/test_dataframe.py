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
import geoarrow.types as gat
import geopandas.testing
import pandas as pd
import pyarrow as pa
import pytest
import sedonadb
from sedonadb.testing import skip_if_not_exists


def test_dataframe_from_dataframe(con):
    # DataFrame from DataFrame on the same context with no schema
    # should be just a Python reference
    df = con.sql("SELECT ST_Point(0, 1) as geom")
    assert con.create_data_frame(df) is df

    # On a separate context the table should still be collected the same
    # but should be a separate Python reference. This also has the effect
    # of testing the __datafusion_table_provider__ interface.
    new_con = sedonadb.connect()
    new_df = new_con.create_data_frame(df)
    assert new_df is not df
    pd.testing.assert_frame_equal(df.to_pandas(), new_df.to_pandas())


def test_dataframe_from_table(con):
    tab = pa.table(
        {
            "not_geom": [1, 2, 3],
            "geom": ga.array(["POINT (0 1)", "POINT (2 3)", "POINT (4 5)"]),
        }
    )

    df = con.create_data_frame(tab)

    # Ensure that we can collect once
    assert df.to_arrow_table() == tab

    # ...and ensure we can collect again
    assert df.to_arrow_table() == tab


def test_dataframe_from_pandas(con):
    pd_df = pd.DataFrame({"col1": [1, 2, 3]})

    df = con.create_data_frame(pd_df)

    # Ensure that we can collect once
    pd.testing.assert_frame_equal(df.to_pandas(), pd_df)

    # ...and ensure we can collect again
    pd.testing.assert_frame_equal(df.to_pandas(), pd_df)


def test_dataframe_from_geopandas(con):
    gpd_df = geopandas.GeoDataFrame(
        {"geometry": geopandas.GeoSeries.from_wkt(["POINT (0 1)"], crs="OGC:CRS84")}
    )

    df = con.create_data_frame(gpd_df)

    # Ensure that we can collect once
    geopandas.testing.assert_geodataframe_equal(df.to_pandas(), gpd_df)

    # ...and ensure we can collect again
    geopandas.testing.assert_geodataframe_equal(df.to_pandas(), gpd_df)


def test_dataframe_from_polars(con):
    pl = pytest.importorskip("polars")

    pl_df = pl.DataFrame({"col1": [1, 2, 3]})

    df = con.create_data_frame(pl_df)

    # Ensure that we can collect once
    pd.testing.assert_frame_equal(df.to_pandas(), pd.DataFrame({"col1": [1, 2, 3]}))

    # ...and ensure we can collect again
    pd.testing.assert_frame_equal(df.to_pandas(), pd.DataFrame({"col1": [1, 2, 3]}))


def test_dataframe_from_array_stream(con):
    tab = pa.table(
        {
            "not_geom": [1, 2, 3],
            "geom": ga.array(["POINT (0 1)", "POINT (2 3)", "POINT (4 5)"]),
        }
    )

    # Ensure we're working from a stateful reader that can only be consumed
    # exactly once
    one_way_stream = pa.RecordBatchReader.from_stream(tab)
    df = con.create_data_frame(one_way_stream)
    assert pa.schema(df) == tab.schema

    # Ensure that we can collect once
    assert df.to_arrow_table() == tab

    # Ensure that we exhausted the reader
    assert list(one_way_stream) == []

    with pytest.raises(
        sedonadb._lib.SedonaError,
        match="Can't scan RecordBatchReader provider more than once.",
    ):
        # Ensure we can't collect again
        df.to_arrow_table()


def test_schema(con):
    df = con.sql("SELECT 1 as one, ST_GeomFromText('POINT (0 1)') as geom")

    # Non-geometry field accessor
    assert df.schema.field(0).name == "one"
    assert df.schema.field("one").name == "one"
    assert repr(df.schema.field(0).type) == "SedonaType int64<Int64>"
    assert df.schema.field(0).type.edge_type is None
    assert df.schema.field(0).type.crs is None

    # Geometry field accessor
    assert df.schema.field(1).name == "geom"
    assert df.schema.field("geom").name == "geom"
    assert repr(df.schema.field(1).type) == "SedonaType geometry<Wkb>"
    assert df.schema.field(1).type.edge_type == gat.EdgeType.PLANAR
    assert df.schema.field(1).type.crs is None

    # Arrow export
    assert pa.schema(df.schema) == pa.schema(
        [pa.field("one", pa.int64(), False), pa.field("geom", ga.wkb())]
    )
    assert pa.field(df.schema.field(0)) == pa.field("one", pa.int64(), False)
    assert pa.field(df.schema.field(0).type) == pa.field("", pa.int64(), True)

    with pytest.raises(IndexError):
        df.schema.field(100)

    with pytest.raises(KeyError):
        df.schema.field("foofy")

    with pytest.raises(TypeError):
        df.schema.field({})


def test_columns(con):
    df = con.sql("SELECT 1 as one, ST_GeomFromWKT('POINT (0 1)') as geom")
    assert len(df.columns) == 2

    pdf = df.to_pandas()
    assert set(df.columns) == set(pdf.columns)


def test_schema_non_null_crs(con):
    tab = pa.table({"geom": ga.with_crs(ga.as_wkb(["POINT (0 1)"]), gat.OGC_CRS84)})
    df = con.create_data_frame(tab)
    assert df.schema.field("geom").type.crs == gat.OGC_CRS84


def test_to_memtable(con):
    df = con.sql("SELECT 1 as one")
    pd.testing.assert_frame_equal(df.to_memtable().to_pandas(), df.to_pandas())


def test_to_view(con):
    try:
        df = con.sql("SELECT 1 as one")
        df.to_view("foofy")
        pd.testing.assert_frame_equal(
            con.sql("SELECT * FROM foofy").to_pandas(), df.to_pandas()
        )

        new_df = con.sql("SELECT 2 as two")
        with pytest.raises(
            sedonadb._lib.SedonaError, match="The table foofy already exist"
        ):
            new_df.to_view("foofy")

        new_df.to_view("foofy", overwrite=True)
        pd.testing.assert_frame_equal(
            con.sql("SELECT * FROM foofy").to_pandas(), new_df.to_pandas()
        )

    finally:
        con.drop_view("foofy")


def test_head_limit(con):
    df = con.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")

    pd.testing.assert_frame_equal(
        df.head(1).to_pandas(), pd.DataFrame({"val": ["one"]})
    )

    pd.testing.assert_frame_equal(
        df.limit(1).to_pandas(), pd.DataFrame({"val": ["one"]})
    )

    pd.testing.assert_frame_equal(
        df.limit(1, offset=2).to_pandas(), pd.DataFrame({"val": ["three"]})
    )

    pd.testing.assert_frame_equal(
        df.limit(None, offset=1).to_pandas(), pd.DataFrame({"val": ["two", "three"]})
    )


def test_execute(con):
    df = con.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
    assert df.execute() == 3

    df = con.sql("CREATE OR REPLACE VIEW temp_view AS SELECT 1 as one")
    assert df.execute() == 0
    assert con.view("temp_view").count() == 1
    con.drop_view("temp_view")


def test_count(con):
    df = con.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
    assert df.count() == 3


def test_dataframe_to_arrow(con):
    df = con.sql("SELECT 1 as one, ST_GeomFromWKT('POINT (0 1)') as geom")
    expected_schema = pa.schema(
        [pa.field("one", pa.int64(), nullable=False), pa.field("geom", ga.wkb())]
    )

    assert pa.schema(df) == expected_schema
    assert (
        df.to_arrow_table().columns
        == pa.table(
            {"one": [1], "geom": ga.as_wkb(["POINT (0 1)"])}, schema=expected_schema
        ).columns
    )

    # Make sure we can request a schema if the schema is identical
    assert (
        df.to_arrow_table(schema=expected_schema).columns == df.to_arrow_table().columns
    )

    # ...but not otherwise (yet)
    with pytest.raises(
        sedonadb._lib.SedonaError,
        match="Requested schema != DataFrame schema not yet supported",
    ):
        df.to_arrow_table(schema=pa.schema({}))


def test_dataframe_to_arrow_empty_batches(con, geoarrow_data):
    # It's difficult to trigger this with a simpler example
    # https://github.com/apache/sedona-db/issues/156
    path_water_junc = (
        geoarrow_data / "ns-water" / "files" / "ns-water_water-junc_geo.parquet"
    )
    path_water_point = (
        geoarrow_data / "ns-water" / "files" / "ns-water_water-point_geo.parquet"
    )
    skip_if_not_exists(path_water_junc)
    skip_if_not_exists(path_water_point)

    con.read_parquet(path_water_junc).to_view("junc", overwrite=True)
    con.read_parquet(path_water_point).to_view("point", overwrite=True)
    con.sql("""SELECT geometry FROM junc WHERE "OBJECTID" = 1814""").to_view(
        "junc_filter", overwrite=True
    )

    joined = con.sql("""
        SELECT "OBJECTID", "FEAT_CODE", point.geometry
        FROM point
        JOIN junc_filter ON ST_DWithin(junc_filter.geometry, point.geometry, 10000)
    """)

    reader = pa.RecordBatchReader.from_stream(joined)
    batch_rows = [len(batch) for batch in reader]
    assert batch_rows == [24]


def test_dataframe_to_pandas(con):
    # Check with a geometry column
    df_with_geo = con.sql("SELECT 1 as one, ST_GeomFromWKT('POINT (0 1)') as geom")
    geopandas.testing.assert_geodataframe_equal(
        df_with_geo.to_pandas(),
        geopandas.GeoDataFrame(
            {"one": [1], "geom": geopandas.GeoSeries.from_wkt(["POINT (0 1)"])}
        ).set_geometry("geom"),
    )

    # Check with more than one geometry column
    df_with_multi_geo = con.sql(
        "SELECT ST_GeomFromWKT('POINT (0 1)') as geom1, ST_GeomFromWKT('POINT (2 3)') as geom2"
    )
    geodf_with_multi_geo = geopandas.GeoDataFrame(
        {
            "geom1": geopandas.GeoSeries.from_wkt(["POINT (0 1)"]),
            "geom2": geopandas.GeoSeries.from_wkt(["POINT (2 3)"]),
        }
    )

    geopandas.testing.assert_geodataframe_equal(
        df_with_multi_geo.to_pandas(geometry="geom1"),
        geodf_with_multi_geo.set_geometry("geom1"),
    )

    geopandas.testing.assert_geodataframe_equal(
        df_with_multi_geo.to_pandas(geometry="geom2"),
        geodf_with_multi_geo.set_geometry("geom2"),
    )

    # Check without geometry column
    df_without_geo = con.sql("SELECT 1 as one")
    pd.testing.assert_frame_equal(
        df_without_geo.to_pandas(), pd.DataFrame({"one": [1]})
    )


def test_dataframe_to_parquet(con):
    df = con.sql(
        "SELECT * FROM (VALUES ('one', 1), ('two', 2), ('three', 3)) AS t(a, b)"
    )

    with tempfile.TemporaryDirectory() as td:
        # Defaults with a path that ends with .parquet (single file)
        tmp_parquet_file = Path(td) / "tmp.parquet"
        df.to_parquet(tmp_parquet_file)

        assert tmp_parquet_file.exists()
        assert tmp_parquet_file.is_file()
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_parquet_file),
            pd.DataFrame({"a": ["one", "two", "three"], "b": [1, 2, 3]}),
        )

        # Defaults with a path that doesn't end in .parquet (directory)
        tmp_parquet_dir = Path(td) / "tmp"
        df.to_parquet(tmp_parquet_dir)

        assert tmp_parquet_dir.exists()
        assert tmp_parquet_dir.is_dir()
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_parquet_dir),
            pd.DataFrame({"a": ["one", "two", "three"], "b": [1, 2, 3]}),
        )

        # With partition_by
        tmp_parquet_dir = Path(td) / "tmp_partitioned"
        df.to_parquet(tmp_parquet_dir, partition_by=["a"])
        assert tmp_parquet_dir.exists()
        assert tmp_parquet_dir.is_dir()
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_parquet_dir).sort_values("b").reset_index(drop=True),
            pd.DataFrame(
                {"b": [1, 2, 3], "a": pd.Categorical(["one", "two", "three"])}
            ),
        )

        # With order_by
        tmp_parquet = Path(td) / "tmp_ordered.parquet"
        df.to_parquet(tmp_parquet, sort_by=["a"])
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_parquet),
            pd.DataFrame({"a": ["one", "three", "two"], "b": [1, 3, 2]}),
        )


def test_record_batch_reader_projection(con):
    def batches():
        for _ in range(3):
            yield pa.record_batch({"a": ["a", "b", "c"], "b": [1, 2, 3]})

    reader = pa.RecordBatchReader.from_batches(next(batches()).schema, batches())
    df = con.create_data_frame(reader)
    df.to_view("temp_rbr_proj", overwrite=True)
    try:
        # Query the view with projection (only select column b)
        proj_df = con.sql("SELECT b FROM temp_rbr_proj")
        tbl = proj_df.to_arrow_table()
        assert tbl.column_names == ["b"]
        assert tbl.to_pydict()["b"] == [1, 2, 3] * 3
    finally:
        con.drop_view("temp_rbr_proj")


def test_show(con, capsys):
    con.sql("SELECT 1 as one").show()
    expected = """
┌───────┐
│  one  │
│ int64 │
╞═══════╡
│     1 │
└───────┘
    """.strip()
    assert capsys.readouterr().out.strip() == expected

    con.sql("SELECT 1 as one").show(ascii=True)
    expected = """
+-------+
|  one  |
| int64 |
+-------+
|     1 |
+-------+
    """.strip()
    assert capsys.readouterr().out.strip() == expected

    # Make sure width parameter can be specified
    con.sql("SELECT 123456789 as col1, 2 as a_very_long_column_name").show(width=10)
    expected = """
┌───────────┬───┐
│    col1   ┆ … │
│   int64   ┆   │
╞═══════════╪═══╡
│ 123456789 ┆ … │
└───────────┴───┘
    """.strip()
    assert capsys.readouterr().out.strip() == expected


def test_show_explained(con, capsys):
    con.sql("EXPLAIN SELECT 1 as one").show()
    expected = """
┌───────────────┬─────────────────────────────────┐
│   plan_type   ┆               plan              │
│      utf8     ┆               utf8              │
╞═══════════════╪═════════════════════════════════╡
│ logical_plan  ┆ Projection: Int64(1) AS one     │
│               ┆   EmptyRelation: rows=1         │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ physical_plan ┆ ProjectionExec: expr=[1 as one] │
│               ┆   PlaceholderRowExec            │
│               ┆                                 │
└───────────────┴─────────────────────────────────┘
    """.strip()
    assert capsys.readouterr().out.strip() == expected


def test_explain(con, capsys):
    con.sql("SELECT 1 as one").explain().show()
    expected = """
┌───────────────┬─────────────────────────────────┐
│   plan_type   ┆               plan              │
│      utf8     ┆               utf8              │
╞═══════════════╪═════════════════════════════════╡
│ logical_plan  ┆ Projection: Int64(1) AS one     │
│               ┆   EmptyRelation: rows=1         │
├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ physical_plan ┆ ProjectionExec: expr=[1 as one] │
│               ┆   PlaceholderRowExec            │
│               ┆                                 │
└───────────────┴─────────────────────────────────┘
    """.strip()
    assert capsys.readouterr().out.strip() == expected

    con.sql("SELECT 1 as one").explain(format="tree").show()
    expected = """
┌───────────────┬───────────────────────────────┐
│   plan_type   ┆              plan             │
│      utf8     ┆              utf8             │
╞═══════════════╪═══════════════════════════════╡
│ physical_plan ┆ ┌───────────────────────────┐ │
│               ┆ │       ProjectionExec      │ │
│               ┆ │    --------------------   │ │
│               ┆ │           one: 1          │ │
│               ┆ └─────────────┬─────────────┘ │
│               ┆ ┌─────────────┴─────────────┐ │
│               ┆ │     PlaceholderRowExec    │ │
│               ┆ └───────────────────────────┘ │
│               ┆                               │
└───────────────┴───────────────────────────────┘
    """.strip()
    assert capsys.readouterr().out.strip() == expected

    query_plan = con.sql("SELECT 1 as one").explain(type="analyze").to_pandas()
    assert query_plan.iloc[0, 0] == "Plan with Metrics"

    query_plan = con.sql("SELECT 1 as one").explain(type="extended").to_pandas()
    assert query_plan.iloc[0, 0] == "initial_logical_plan"
    assert len(query_plan) > 10


def test_repr(con):
    assert repr(con.sql("SELECT 1 as one")).startswith(
        "<sedonadb.dataframe.DataFrame object"
    )

    try:
        con.options.interactive = True
        repr_interactive = repr(con.sql("SELECT 1 as one"))
        expected = """
┌───────┐
│  one  │
│ int64 │
╞═══════╡
│     1 │
└───────┘
    """.strip()
        assert repr_interactive == expected
    finally:
        con.options.interactive = False
