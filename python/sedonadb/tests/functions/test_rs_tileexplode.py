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

import pytest

from sedonadb.raster_testing import (
    Rasterio,
    SedonaSpark,
    assert_decoded_equal,
    create_dialect_engine,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

# SedonaDB does not implement RS_TileExplode.
DIALECTS = [SedonaSpark]

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7


@pytest.fixture(params=DIALECTS, ids=lambda engine: engine.name())
def dialect(request):
    return create_dialect_engine(request.param)


@pytest.fixture()
def reference():
    return Rasterio.create_or_skip()


@pytest.mark.parametrize(
    ("tile_width", "tile_height"),
    [(4, 4), (2, 3), (WIDTH, HEIGHT)],
    ids=["ragged-edges", "exact-grid", "single-tile"],
)
def test_rs_tileexplode_matches_reference(
    dialect, reference, tmp_path, tile_width, tile_height
):
    tiff = tmp_path / "tiles.tif"
    write_geotiff(
        tiff,
        random_raster_data("uint8", bands=3, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
        nodata=200.0,
    )

    got = dialect.tile_explode(tiff, tile_width, tile_height)
    expected = reference.tile_explode(tiff, tile_width, tile_height)
    assert [(x, y) for x, y, _ in got] == [(x, y) for x, y, _ in expected]
    for (x, y, got_tile), (_, _, expected_tile) in zip(got, expected):
        assert_decoded_equal(got_tile, expected_tile, context=(x, y))
