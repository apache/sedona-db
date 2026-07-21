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

"""RS_TileExplode parity against rasterio window reads.

Every tile must reproduce the source pixels verbatim with a window-shifted
transform, keep all bands and the band nodata, and edge tiles keep their
partial size (no nodata padding). The 4x4 case makes both dimensions ragged
on the 7x6 fixture; 7x6 is the identity single tile."""

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


@pytest.mark.parametrize(
    ("tile_width", "tile_height"),
    [(4, 4), (2, 3), (WIDTH, HEIGHT)],
    ids=["ragged-edges", "exact-grid", "single-tile"],
)
def test_rs_tileexplode_matches_comparators(
    subject, comparator, tmp_path, tile_width, tile_height
):
    tiff = tmp_path / "tiles.tif"
    write_geotiff(
        tiff,
        random_raster_data("uint8", bands=3, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
        nodata=200.0,
    )

    got = subject.tile_explode(tiff, tile_width, tile_height)
    expected = comparator.tile_explode(tiff, tile_width, tile_height)
    assert [(x, y) for x, y, _ in got] == [(x, y) for x, y, _ in expected]
    for (x, y, got_tile), (_, _, expected_tile) in zip(got, expected):
        assert_decoded_equal(got_tile, expected_tile, context=(x, y))


def test_nodata_requires_pad_with_nodata(con, tmp_path):
    """A `nodata` fill given without `pad_with_nodata` raises in SedonaDB.

    Sedona Spark silently ignores `nodata` when padding is off (the documented
    divergence); asserting the raise pins SedonaDB's stricter contract — an
    option that would never be applied is an error, not a no-op.

    This is a subject-error case (the parity subject itself raises), so a plain
    `pytest.raises` on the subject is the right shape; it does not go through the
    comparator/deviation ledger. A ledger-integrated "subject_error" Deviation
    kind that also captured the Spark silent-ignore behavior declaratively would
    be a possible future enhancement.

    The options argument is a JSON object matching the RS_TileExplode function
    surface (`df.rast.funcs.rs_tileexplode(w, h, options)`); the raster travels
    as a table column so the kernel runs its real array path.
    """
    tiff = tmp_path / "tiles.tif"
    write_geotiff(
        tiff,
        random_raster_data("uint8", bands=3, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
        nodata=200.0,
    )
    df = con.create_data_frame(pa.table({"path": pa.array([str(tiff)], pa.utf8())}))
    tiles = df.path.funcs.rs_frompath().funcs.rs_tileexplode(4, 4, '{"nodata": 0}')
    with pytest.raises(Exception, match="only meaningful with pad_with_nodata"):
        df.select(tiles=tiles).to_arrow_table()
