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

"""Integration tests for RS_Tile.

RS_Tile returns one list item per tile, so these tests assert the tile
count and each tile's (x, y) grid position through the full execution path
(materialized via `to_arrow_table`). Pixel-level tiling correctness and band
selection are covered exhaustively by the Rust unit tests against a numpy
reference; here we only verify that the positional overloads flow through the
expression API and SQL.

Results are inspected via Arrow (reading only the integer x/y struct fields)
rather than `Scalar.as_py()`, which would try to materialize the nested tile
raster.

RS_Example() is a 64x32, 3-band raster, so a 32x16 tile yields a 2x2 grid
(4 tiles) at grid positions (0,0), (1,0), (0,1), (1,1).
"""

import pyarrow as pa

EXPECTED_POSITIONS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def _example_raster_df(con):
    """A one-row data frame with a real raster column.

    RS_Example() is round-tripped into table data so RS_Tile runs over a
    column (its array path) rather than constant-folding a literal.
    """
    table = con.sql("SELECT RS_Example() AS rast").to_arrow_table()
    return con.create_data_frame(table)


def _tile_positions(list_scalar_values) -> list:
    """Read the (x, y) grid positions from a flattened tile struct array."""
    xs = list_scalar_values.field("x").to_pylist()
    ys = list_scalar_values.field("y").to_pylist()
    return list(zip(xs, ys))


def _tile_column(df, tiles):
    return df.select(tiles=tiles).to_arrow_table()["tiles"].combine_chunks()


def test_rs_tile_count_and_positions(con):
    # RS_Tile(raster, width, height): the no-band overload tiles every
    # band into the 2x2 grid.
    df = _example_raster_df(con)
    tiles = df.rast.funcs.rs_tile(32, 16)
    column = _tile_column(df, tiles)

    # One list per input row; the single row holds the 2x2 = 4-tile grid.
    assert column.value_lengths().to_pylist() == [4]
    assert _tile_positions(column.values) == EXPECTED_POSITIONS


def test_rs_tile_pad_with_nodata_overload(con):
    # RS_Tile(raster, width, height, padWithNoData, noDataVal): padding
    # 40x20 tiles over the 64x32 raster still yields a 2x2 grid (the edge tiles
    # are padded rather than shrunk).
    df = _example_raster_df(con)
    tiles = df.rast.funcs.rs_tile(40, 20, True, 0.0)
    column = _tile_column(df, tiles)
    assert column.value_lengths().to_pylist() == [4]
    assert _tile_positions(column.values) == EXPECTED_POSITIONS


def test_rs_tile_over_multiple_rows(con):
    # Two raster rows: each explodes independently into its own list of tiles.
    table = con.sql("SELECT RS_Example() AS rast").to_arrow_table()
    df = con.create_data_frame(pa.concat_tables([table, table]))
    tiles = df.rast.funcs.rs_tile(32, 16)
    column = _tile_column(df, tiles)
    assert column.value_lengths().to_pylist() == [4, 4]


def test_rs_tile_band_indices_array_sql_unnest(con):
    # SQL parser-path smoke for the bandIndices Array[Int] overload: UNNEST
    # expands the list so the result has one row per tile, and the tile struct
    # carries (x, y).
    tile_struct = (
        con.sql(
            "SELECT UNNEST(RS_Tile(RS_Example(), make_array(1, 3), 32, 16)) AS tile"
        )
        .to_arrow_table()["tile"]
        .combine_chunks()
    )

    assert len(tile_struct) == 4
    positions = list(
        zip(tile_struct.field("x").to_pylist(), tile_struct.field("y").to_pylist())
    )
    assert sorted(positions) == sorted(EXPECTED_POSITIONS)
