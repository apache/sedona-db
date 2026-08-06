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

"""Parity tests for the RS_TileExplode generator surfaces.

RS_TileExplode emits one row per tile with top-level `(x, y, tile)` columns —
the row-multiplying form of `RS_Tile`, which returns every tile of a raster in
one list cell. Both surfaces are exercised here:

- the `DataFrame.tile_explode(...)` method (the primary API), and
- the SQL `RS_TileExplode(...)` generator (one parser-path smoke),

each proven byte-identical to `UNNEST(RS_Tile(...))` on the same input — same
tile count, same row-major order, and exact tile pixels/transform/nodata. Because
RS_Tile shares the tiling core with RS_TileExplode (and RS_Tile is itself pinned
against a rasterio window reference in `test_rs_tile`), that parity is the
authoritative correctness pin; one direct rasterio cross-check is kept too.

RS_Example() is a 64x32, 3-band raster, so a 40x20 tiling yields a 2x2 grid (4
tiles) with partial right/bottom edge tiles at grid positions
(0,0), (1,0), (0,1), (1,1). Results are materialized via `to_arrow_table()` so
the full execution path runs.
"""

import pyarrow as pa
import pytest
from sedonadb.raster_testing import (
    DecodedRaster,
    assert_decoded_equal,
    decode_raster,
    random_raster_data,
    write_geotiff,
)


def _raster_df(con, *, extra_columns=""):
    """A one-row DataFrame with a real raster column `rast` (plus any
    `extra_columns` SQL prefix, e.g. `"7 AS id, "`).

    RS_Example() is round-tripped into table data so RS_TileExplode runs over a
    raster *column* (which the generator requires) rather than a constant.
    """
    table = con.sql(f"SELECT {extra_columns}RS_Example() AS rast").to_arrow_table()
    return con.create_data_frame(table)


def _explode_rows(table):
    """`(x, y, DecodedRaster)` per row of a tile-explode result table whose last
    three columns are the top-level `x`, `y`, `tile`."""
    xs = table["x"].combine_chunks()
    ys = table["y"].combine_chunks()
    tiles = table["tile"].combine_chunks()
    return [
        (xs[i].as_py(), ys[i].as_py(), decode_raster(tiles[i]))
        for i in range(table.num_rows)
    ]


def _unnest_reference(con, tile_args_sql):
    """`(x, y, DecodedRaster)` per tile of `UNNEST(RS_Tile(RS_Example(), ...))`,
    the shared-core oracle. `tile_args_sql` is the argument list after the raster
    (e.g. `"40, 20"`)."""
    struct = (
        con.sql(f"SELECT UNNEST(RS_Tile(RS_Example(), {tile_args_sql})) AS tile")
        .to_arrow_table()["tile"]
        .combine_chunks()
    )
    xs, ys, tiles = struct.field("x"), struct.field("y"), struct.field("tile")
    return [
        (xs[i].as_py(), ys[i].as_py(), decode_raster(tiles[i]))
        for i in range(len(struct))
    ]


def _assert_rows_equal(got, expected):
    """Same tile count, same row-major `(x, y)` order, exact tile pixels."""
    assert [(x, y) for x, y, _ in got] == [(x, y) for x, y, _ in expected]
    for (x, y, got_tile), (_, _, expected_tile) in zip(got, expected):
        assert_decoded_equal(got_tile, expected_tile, context=(x, y))


@pytest.mark.parametrize(
    ("kwargs", "tile_args_sql"),
    [
        ({}, "40, 20"),
        ({"pad_with_no_data": True, "no_data_value": 0.0}, "40, 20, true, 0.0"),
    ],
    ids=["partial-edges", "padded-edges"],
)
def test_df_tile_explode_matches_unnest_rs_tile(con, kwargs, tile_args_sql):
    # df.tile_explode(...) must reproduce UNNEST(RS_Tile(...)) exactly: the same
    # 4 tiles, in the same row-major order, with byte-identical pixels — for both
    # the ragged-edge and padded-edge overloads.
    df = _raster_df(con)
    got = _explode_rows(df.tile_explode("rast", 40, 20, **kwargs).to_arrow_table())
    expected = _unnest_reference(con, tile_args_sql)
    assert len(got) == 4
    _assert_rows_equal(got, expected)


def test_df_tile_explode_band_subset_scalar_and_list(con):
    # The scalar-band arg (band_indices=1) and the single-element list
    # (band_indices=[1]) select band 1 identically, matching
    # UNNEST(RS_Tile(rast, make_array(1), ...)). This also pins that a scalar band
    # is normalized to a one-element list (the SQL scalar-band overload).
    df = _raster_df(con)
    scalar = _explode_rows(
        df.tile_explode("rast", 40, 20, band_indices=1).to_arrow_table()
    )
    listed = _explode_rows(
        df.tile_explode("rast", 40, 20, band_indices=[1]).to_arrow_table()
    )
    expected = _unnest_reference(con, "make_array(1), 40, 20")
    assert len(scalar) == 4
    _assert_rows_equal(scalar, expected)
    _assert_rows_equal(listed, expected)


def test_df_tile_explode_null_raster_yields_zero_rows(con):
    # A null raster row contributes no tiles. Build a single-row table whose
    # raster is NULL (reusing the raster field's type + metadata, made nullable so
    # the null is allowed), then explode it: zero output rows.
    rast_field = (
        con.sql("SELECT RS_Example() AS rast")
        .to_arrow_table()
        .schema.field("rast")
        .with_nullable(True)
    )
    null_table = pa.Table.from_arrays(
        [pa.array([None], type=rast_field.type)], schema=pa.schema([rast_field])
    )
    df = con.create_data_frame(null_table)
    out = df.tile_explode("rast", 40, 20).to_arrow_table()
    assert out.num_rows == 0


def test_df_tile_explode_replicates_sibling_columns(con):
    # A sibling `id` column is carried through and replicated across every tile
    # row (the raster argument itself is consumed, not re-emitted), and the
    # appended (x, y, tile) still match the RS_Tile oracle.
    df = _raster_df(con, extra_columns="7 AS id, ")
    table = df.tile_explode("rast", 40, 20).to_arrow_table()

    assert table.column_names == ["id", "x", "y", "tile"]
    assert table["id"].to_pylist() == [7, 7, 7, 7]
    _assert_rows_equal(_explode_rows(table), _unnest_reference(con, "40, 20"))


def test_sql_rs_tileexplode_matches_unnest_rs_tile(con):
    # SQL parser-path smoke: the RS_TileExplode generator (raster read from a
    # derived-table column) lifts to top-level (x, y, tile) and matches
    # UNNEST(RS_Tile(...)) row for row.
    table = con.sql(
        "SELECT RS_TileExplode(rast, 40, 20) FROM (SELECT RS_Example() AS rast)"
    ).to_arrow_table()
    got = _explode_rows(table)
    assert len(got) == 4
    _assert_rows_equal(got, _unnest_reference(con, "40, 20"))


# --- Independent rasterio window oracle (partial edges) ---
#
# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up pixels;
# a 7x6 raster has extent x in [100, 114], y in [482, 500]. A 4x4 tiling makes
# both dimensions ragged.
_PARITY_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
_PARITY_HEIGHT, _PARITY_WIDTH = 6, 7


def _rasterio_tile_explode(path, tile_width, tile_height):
    """Rasterio window reference: one `(x, y, DecodedRaster)` per tile, row-major."""
    import rasterio
    from rasterio.windows import Window

    out = []
    with rasterio.open(str(path)) as src:
        for tile_y, row_off in enumerate(range(0, src.height, tile_height)):
            for tile_x, col_off in enumerate(range(0, src.width, tile_width)):
                window = Window(
                    col_off,
                    row_off,
                    min(tile_width, src.width - col_off),
                    min(tile_height, src.height - row_off),
                )
                out.append(
                    (
                        tile_x,
                        tile_y,
                        DecodedRaster(
                            src.read(window=window),
                            tuple(src.window_transform(window).to_gdal()),
                            list(src.nodatavals),
                        ),
                    )
                )
    return out


def test_df_tile_explode_matches_rasterio(con, tmp_path):
    # An independent oracle: both sides read the same independently-written
    # GeoTIFF (SedonaDB via RS_FromPath), and df.tile_explode must reproduce the
    # rasterio window read of every ragged tile exactly.
    pytest.importorskip("rasterio")
    tiff = tmp_path / "tiles.tif"
    write_geotiff(
        tiff,
        random_raster_data(
            "uint8", bands=3, height=_PARITY_HEIGHT, width=_PARITY_WIDTH
        ),
        gdal_transform=_PARITY_TRANSFORM,
        nodata=200.0,
    )

    path_df = con.create_data_frame(
        pa.table({"path": pa.array([str(tiff)], pa.utf8())})
    )
    # RS_FromPath yields an OutDb raster; tiling reads pixels, so materialize it
    # in-database with RS_EnsureLoaded first (the current OutDb workaround).
    raster_df = path_df.select(
        rast=path_df["path"].funcs.rs_frompath().funcs.rs_ensureloaded()
    )
    got = _explode_rows(raster_df.tile_explode("rast", 4, 4).to_arrow_table())
    got = sorted(got, key=lambda t: (t[1], t[0]))

    expected = _rasterio_tile_explode(tiff, 4, 4)
    assert [(x, y) for x, y, _ in got] == [(x, y) for x, y, _ in expected]
    for (x, y, got_tile), (_, _, expected_tile) in zip(got, expected):
        assert_decoded_equal(got_tile, expected_tile, context=(x, y))
