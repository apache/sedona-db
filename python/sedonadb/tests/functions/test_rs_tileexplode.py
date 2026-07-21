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

"""RS_TileExplode parity across its Spark overloads.

Every tile must reproduce the source pixels verbatim with a window-shifted
transform. Which of the positional shapes runs is chosen by the band selection:
every band (`RS_TileExplode(raster, w, h, ...)`), a single band
(`(raster, bandIndex, w, h, ...)`), or a subset in order
(`(raster, bandIndices, w, h, ...)`). With `pad_with_no_data` the last partial
row/column of tiles is grown to the full tile size and the extra pixels take the
fill (`no_data_val`, else the band nodata, else the dtype minimum), which the
tile also records as its nodata; otherwise the smaller edge tile is emitted with
the source band nodata. The 4x4 case makes both dimensions ragged on the 7x6
fixture."""

import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    SedonaDB,
    assert_decoded_equal,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

pytestmark = pytest.mark.skipif(
    not SedonaDB.implements("tile_explode"),
    reason="RS_TileExplode is not implemented in SedonaDB (the parity subject)",
)

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7


def _write(tmp_path, name, *, bands=3, nodata=200.0):
    tiff = tmp_path / f"tiles_{name}.tif"
    write_geotiff(
        tiff,
        random_raster_data("uint8", bands=bands, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
        nodata=nodata,
    )
    return tiff


def _assert_tiles_equal(got, expected):
    assert [(x, y) for x, y, _ in got] == [(x, y) for x, y, _ in expected]
    for (x, y, got_tile), (_, _, expected_tile) in zip(got, expected):
        assert_decoded_equal(got_tile, expected_tile, context=(x, y))


@pytest.mark.parametrize(
    ("tile_width", "tile_height"),
    [(4, 4), (2, 3), (WIDTH, HEIGHT)],
    ids=["ragged-edges", "exact-grid", "single-tile"],
)
def test_rs_tileexplode_matches_comparators(
    subject, comparator, tmp_path, tile_width, tile_height
):
    """Every band tiled, no padding: edge tiles keep their partial size and the
    source band nodata; the transform shifts to each tile's upper-left corner."""
    tiff = _write(tmp_path, "allbands")
    got = subject.tile_explode(tiff, tile_width, tile_height)
    expected = comparator.tile_explode(tiff, tile_width, tile_height)
    _assert_tiles_equal(got, expected)


def test_rs_tileexplode_band_index_matches_comparators(subject, comparator, tmp_path):
    """The single-band overload keeps only that band in each tile."""
    tiff = _write(tmp_path, "bandindex")
    got = subject.tile_explode(tiff, 4, 4, band_index=2)
    expected = comparator.tile_explode(tiff, 4, 4, band_index=2)
    _assert_tiles_equal(got, expected)
    assert all(tile.pixels.shape[0] == 1 for _, _, tile in got)


def test_rs_tileexplode_band_indices_matches_comparators(subject, comparator, tmp_path):
    """The array overload keeps the named bands in the given order (here band 3
    before band 1)."""
    tiff = _write(tmp_path, "bandindices")
    got = subject.tile_explode(tiff, 4, 4, band_indices=[3, 1])
    expected = comparator.tile_explode(tiff, 4, 4, band_indices=[3, 1])
    _assert_tiles_equal(got, expected)
    assert all(tile.pixels.shape[0] == 2 for _, _, tile in got)


def test_rs_tileexplode_pad_explicit_nodata_matches_comparators(
    subject, comparator, tmp_path
):
    """With padding on and an explicit fill, every tile is the full tile size,
    the padded edge pixels take the fill, and each tile records the fill as its
    nodata."""
    tiff = _write(tmp_path, "pad_explicit")
    got = subject.tile_explode(tiff, 4, 4, pad_with_no_data=True, no_data_val=0.0)
    expected = comparator.tile_explode(
        tiff, 4, 4, pad_with_no_data=True, no_data_val=0.0
    )
    _assert_tiles_equal(got, expected)
    assert all(tile.pixels.shape[1:] == (4, 4) for _, _, tile in got)
    assert all(nd == 0 for _, _, tile in got for nd in tile.nodata)


def test_rs_tileexplode_pad_default_nodata_matches_comparators(
    subject, comparator, tmp_path
):
    """With padding on and no explicit fill, the fill defaults to the band's own
    nodata (200 here); every tile records it, and the padded pixels take it."""
    tiff = _write(tmp_path, "pad_default")
    got = subject.tile_explode(tiff, 4, 4, pad_with_no_data=True)
    expected = comparator.tile_explode(tiff, 4, 4, pad_with_no_data=True)
    _assert_tiles_equal(got, expected)
    assert all(nd == 200 for _, _, tile in got for nd in tile.nodata)


def test_rs_tileexplode_nodata_requires_pad_with_nodata(con, tmp_path):
    """A `no_data_val` given without `pad_with_no_data` raises in SedonaDB.

    Sedona Spark silently ignores `no_data_val` when padding is off (the
    documented divergence); asserting the raise pins SedonaDB's stricter contract
    — an option that would never be applied is an error, not a no-op. This is a
    subject-error case, so a plain `pytest.raises` on the subject is the right
    shape; it does not go through the comparator/deviation ledger. The raster
    travels as a table column so the kernel runs its real array path."""
    tiff = _write(tmp_path, "pad_error")
    df = con.create_data_frame(pa.table({"path": pa.array([str(tiff)], pa.utf8())}))
    # (raster, width, height, padWithNoData=false, noDataVal=0) — a fill with
    # padding off.
    tiles = df.path.funcs.rs_frompath().funcs.rs_tileexplode(4, 4, False, 0.0)
    with pytest.raises(Exception, match="only meaningful with pad_with_nodata"):
        df.select(tiles=tiles).to_arrow_table()


def test_rs_tileexplode_sql_smoke(con, tmp_path):
    """One SQL-text invocation keeps the parser path covered: UNNEST expands the
    tile list to one row per tile."""
    tiff = _write(tmp_path, "smoke")
    tab = con.sql(
        "SELECT COUNT(*) AS n FROM "
        "(SELECT UNNEST(RS_TileExplode(RS_FromPath($1), 4, 4)) AS t)",
        params=(str(tiff),),
    ).to_arrow_table()
    # A 7x6 raster in 4x4 tiles is a 2x2 grid.
    assert tab["n"][0].as_py() == 4
