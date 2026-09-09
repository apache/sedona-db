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
"""SedonaDB vs Sedona Spark parity for RS_Tile.

The engines agree on how many tiles a grid cuts into, but the element
shape diverges structurally: SedonaDB returns an array of
`{x, y, tile}` structs (the tile indices travel with each tile) where
Sedona Spark returns an array of bare rasters — so extracting a tile
needs different SQL per dialect (`tiles[i]['tile']` vs `tiles[i]`), and
per-tile content parity cannot be expressed as one shared query today.
The xfail catalogs the shape divergence; the counts are anchored.
"""

import pytest

from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark


def _engines(name, tmp_path):
    sedona, spark = SedonaDB(), SedonaSpark()
    for eng in (sedona, spark):
        eng.create_random_raster_view(name, tmp_path / f"{name}.tif")
    return sedona, spark


@pytest.mark.parametrize(
    "tile_args,count",
    [
        # 7x6 cut into 4x3 tiles: 2x2.
        pytest.param("4, 3", 4, id="4x3"),
        # 2x2 tiles: ceil(7/2) x ceil(6/2) = 4 x 3.
        pytest.param("2, 2", 12, id="2x2"),
        # A tile the size of the raster: 1.
        pytest.param("7, 6", 1, id="whole"),
    ],
)
def test_rs_tile_count(tile_args, count, tmp_path):
    """Both engines cut the standard grid into the same number of tiles,
    padding partial tiles at the edges."""
    sedona, spark = _engines("tile_src", tmp_path)
    sql = f"SELECT CARDINALITY(RS_Tile(rast, {tile_args})) FROM tile_src"
    compare(sql, sedona, spark, expected=count)


@pytest.mark.xfail(
    reason="the element shapes diverge: SedonaDB returns an array of "
    "{x, y, tile} structs (RS_NumBands(tiles[1]) finds no kernel; "
    "tiles[1]['tile'] is the raster) where Sedona Spark returns an array of "
    "bare rasters (tiles[1] is the raster; ['tile'] cannot be extracted)"
)
def test_rs_tile_element_shape(tmp_path):
    """A tile extracted by array index is a raster on both engines."""
    sedona, spark = _engines("tile_el_src", tmp_path)
    sql = (
        "SELECT RS_NumBands(tiles[1]) FROM "
        "(SELECT RS_Tile(rast, 4, 3) AS tiles FROM tile_el_src) q"
    )
    compare(sql, sedona, spark)
