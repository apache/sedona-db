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

"""Integration tests for RS_TileExplode.

RS_TileExplode returns one list item per tile, so these tests assert the tile
count and each tile's (x, y) grid position through the full execution path
(materialized via `to_arrow_table`). Pixel-level tiling correctness is covered
exhaustively by the Rust unit tests against a numpy reference; here we only
verify that the list-of-tiles output flows through SQL and the expression API.

Results are inspected via Arrow (reading only the integer x/y struct fields)
rather than `Scalar.as_py()`, which would try to materialize the nested tile
raster.

RS_Example() is a 64x32 raster, so a 32x16 tile yields a 2x2 grid (4 tiles) at
grid positions (0,0), (1,0), (0,1), (1,1).
"""

import pyarrow as pa

EXPECTED_POSITIONS = [(0, 0), (1, 0), (0, 1), (1, 1)]


def _example_raster_df(con):
    """A one-row data frame with a real raster column.

    RS_Example() is round-tripped into table data so RS_TileExplode runs over a
    column (its array path) rather than constant-folding a literal.
    """
    table = con.sql("SELECT RS_Example() AS rast").to_arrow_table()
    return con.create_data_frame(table)


def _tile_positions(list_scalar_values) -> list:
    """Read the (x, y) grid positions from a flattened tile struct array."""
    xs = list_scalar_values.field("x").to_pylist()
    ys = list_scalar_values.field("y").to_pylist()
    return list(zip(xs, ys))


def test_rs_tileexplode_count_and_positions(con):
    df = _example_raster_df(con)
    tiles = df.rast.funcs.rs_tileexplode(32, 16)
    column = df.select(tiles=tiles).to_arrow_table()["tiles"].combine_chunks()

    # One list per input row; the single row holds the 2x2 = 4-tile grid.
    assert column.value_lengths().to_pylist() == [4]
    assert _tile_positions(column.values) == EXPECTED_POSITIONS


def test_rs_tileexplode_options_argument(con):
    # The JSON options argument is accepted and parsed: selecting a single band
    # and padding still yields the 2x2 grid.
    df = _example_raster_df(con)
    tiles = df.rast.funcs.rs_tileexplode(
        32, 16, '{"bands": [1], "pad_with_nodata": true, "nodata": 0}'
    )
    column = df.select(tiles=tiles).to_arrow_table()["tiles"].combine_chunks()
    assert column.value_lengths().to_pylist() == [4]
    assert _tile_positions(column.values) == EXPECTED_POSITIONS


def test_rs_tileexplode_over_multiple_rows(con):
    # Two raster rows: each explodes independently into its own list of tiles.
    table = con.sql("SELECT RS_Example() AS rast").to_arrow_table()
    df = con.create_data_frame(pa.concat_tables([table, table]))
    tiles = df.rast.funcs.rs_tileexplode(32, 16)
    column = df.select(tiles=tiles).to_arrow_table()["tiles"].combine_chunks()
    assert column.value_lengths().to_pylist() == [4, 4]


def test_rs_tileexplode_sql_unnest_row_count(con):
    # SQL parser-path smoke: UNNEST expands the list so the result has one row
    # per tile (row count == tile count), and the tile struct carries (x, y).
    tile_struct = (
        con.sql("SELECT UNNEST(RS_TileExplode(RS_Example(), 32, 16)) AS tile")
        .to_arrow_table()["tile"]
        .combine_chunks()
    )

    assert len(tile_struct) == 4
    positions = list(
        zip(tile_struct.field("x").to_pylist(), tile_struct.field("y").to_pylist())
    )
    assert sorted(positions) == sorted(EXPECTED_POSITIONS)
