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
(materialized via `to_arrow_table`), and — against a rasterio window
reference — the tile pixels, transform, and band nodata.

The count/position tests inspect Arrow (reading only the integer x/y struct
fields) rather than `Scalar.as_py()`, which would try to materialize the nested
tile raster; the parity tests decode each tile through the raster engines.

RS_Example() is a 64x32, 3-band raster, so a 32x16 tile yields a 2x2 grid
(4 tiles) at grid positions (0,0), (1,0), (0,1), (1,1).
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


# --- Cross-engine pixel parity against a rasterio window reference ---
#
# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up pixels;
# with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
_PARITY_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
_PARITY_HEIGHT, _PARITY_WIDTH = 6, 7


def _sedonadb_tile_explode(con, path, tile_width, tile_height):
    """RS_Tile (all bands, pad_with_nodata off) unnested to one
    `(x, y, DecodedRaster)` tuple per tile, sorted by `(y, x)`. Arguments travel
    as table columns so the kernel runs its real array path."""
    table = pa.table(
        {
            "path": pa.array([str(path)], pa.utf8()),
            "w": pa.array([int(tile_width)], pa.int32()),
            "h": pa.array([int(tile_height)], pa.int32()),
        }
    )
    df = con.create_data_frame(table)
    raster = df.path.funcs.rs_frompath()
    struct = (
        df.select(tile=raster.funcs.rs_tile(df.w, df.h))
        .unnest("tile")
        .to_arrow_table()["tile"]
        .combine_chunks()
    )
    xs, ys, tiles = struct.field("x"), struct.field("y"), struct.field("tile")
    out = [
        (xs[i].as_py(), ys[i].as_py(), decode_raster(tiles[i]))
        for i in range(len(struct))
    ]
    return sorted(out, key=lambda t: (t[1], t[0]))


def _rasterio_tile_explode(path, tile_width, tile_height):
    """Rasterio window reference: one `(x, y, DecodedRaster)` per tile, iterated
    row-major (the same `(y, x)` order the SedonaDB helper sorts into)."""
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


@pytest.mark.parametrize(
    ("tile_width", "tile_height"),
    [(4, 4), (2, 3), (_PARITY_WIDTH, _PARITY_HEIGHT)],
    ids=["ragged-edges", "exact-grid", "single-tile"],
)
def test_rs_tile_matches_rasterio(con, tmp_path, tile_width, tile_height):
    # With pad_with_nodata off, every tile must reproduce the source pixels
    # verbatim with a window-shifted transform, keep all bands and the band
    # nodata, and edge tiles keep their partial size. The 4x4 case makes both
    # dimensions ragged on the 7x6 fixture; 7x6 is the identity single tile.
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
    got = _sedonadb_tile_explode(con, tiff, tile_width, tile_height)
    expected = _rasterio_tile_explode(tiff, tile_width, tile_height)
    assert [(x, y) for x, y, _ in got] == [(x, y) for x, y, _ in expected]
    for (x, y, got_tile), (_, _, expected_tile) in zip(got, expected):
        assert_decoded_equal(got_tile, expected_tile, context=(x, y))


def test_rs_tile_nodata_without_pad_errors(con, tmp_path):
    # A noDataVal supplied with pad_with_nodata = false raises in SedonaDB
    # (Sedona Spark silently ignores it — the documented divergence). Asserting
    # the raise pins SedonaDB's stricter "error on ambiguous" contract; the
    # raster travels as a table column so the kernel runs its real array path.
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
    df = con.create_data_frame(pa.table({"path": pa.array([str(tiff)], pa.utf8())}))
    tiles = df.path.funcs.rs_frompath().funcs.rs_tile(4, 4, False, 0.0)
    with pytest.raises(Exception, match="only meaningful with pad_with_nodata"):
        df.select(tiles=tiles).to_arrow_table()
