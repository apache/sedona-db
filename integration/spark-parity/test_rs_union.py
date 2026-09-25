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

"""SedonaDB vs Sedona Spark parity for RS_Union.

Each case compares the whole output raster, decoded on both engines, and
anchors it to the inputs' bands stacked in argument order under the first
raster's grid. The fixtures carry no nodata value: Sedona Spark's raster
transport is a GeoTIFF, which holds one nodata value per file, so it cannot
carry bands with different nodata values. One known divergence is an xfail:
Sedona Spark casts every band to the first raster's pixel type where SedonaDB
keeps each band's own.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import DecodedRaster, write_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

BBOX = (100, 482, 114, 500)


def _views(tmp_path, rasters):
    """Register each `(name, DecodedRaster)` as a view on both engines."""
    sedona, spark = SedonaDB(), SedonaSpark()
    for name, raster in rasters:
        path = tmp_path / f"{name}.tif"
        write_geotiff(
            path, raster.pixels, gdal_transform=raster.gdal_transform, nodata=None
        )
        for eng in (sedona, spark):
            eng.create_raster_view(name, path)
    return sedona, spark


def _raster(dtype="uint8", bands=2, plant=0, bbox=BBOX, width=7):
    """A random raster, made distinct from its siblings by a planted value."""
    raster = DecodedRaster.random(
        dtype, bands=bands, width=width, bbox=bbox, plants={(2, 3): plant}
    )
    raster.nodata = [None] * bands
    return raster


def _stacked(*rasters):
    return DecodedRaster(
        np.concatenate([r.pixels for r in rasters]),
        gdal_transform=rasters[0].gdal_transform,
        nodata=[None] * sum(len(r.pixels) for r in rasters),
    )


def test_rs_union(tmp_path):
    a, b = _raster(plant=1), _raster(plant=2)
    sedona, spark = _views(tmp_path, [("un_a", a), ("un_b", b)])
    sql = "SELECT RS_Union(un_a.rast, un_b.rast) FROM un_a, un_b"
    compare(sql, sedona, spark, expected=_stacked(a, b))


def test_rs_union_three_rasters(tmp_path):
    a, b, c = _raster(plant=1), _raster(bands=1, plant=2), _raster(plant=3)
    sedona, spark = _views(tmp_path, [("un_a", a), ("un_b", b), ("un_c", c)])
    sql = "SELECT RS_Union(un_a.rast, un_b.rast, un_c.rast) FROM un_a, un_b, un_c"
    compare(sql, sedona, spark, expected=_stacked(a, b, c))


def test_rs_union_takes_the_first_georeference(tmp_path):
    """The second raster's grid is elsewhere; the result keeps the first's."""
    a, b = _raster(plant=1), _raster(plant=2, bbox=(0, 0, 7, 6))
    sedona, spark = _views(tmp_path, [("un_a", a), ("un_b", b)])
    sql = "SELECT RS_Union(un_a.rast, un_b.rast) FROM un_a, un_b"
    compare(sql, sedona, spark, expected=_stacked(a, b))


@pytest.mark.parametrize(
    "width,height", [(5, 4), (5, 6), (7, 4)], ids=["both", "width", "height"]
)
def test_rs_union_shape_mismatch(width, height, tmp_path):
    """Both engines refuse rasters whose width or height differ. Error types
    differ, so parity here is parity on refusal."""
    a = _raster(plant=1)
    b = DecodedRaster.random(
        bands=1, width=width, height=height, bbox=(0, 0, width, height)
    )
    sedona, spark = _views(tmp_path, [("un_a", a), ("un_b", b)])
    sql = "SELECT RS_Union(un_a.rast, un_b.rast) FROM un_a, un_b"
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)


@pytest.mark.xfail(
    reason="Sedona Spark 1.9.1 casts every band to the first raster's pixel "
    "type; SedonaDB keeps each band's own"
)
def test_rs_union_mixed_pixel_types(tmp_path):
    a, b = _raster(plant=1), _raster("float64", bands=1, plant=2)
    sedona, spark = _views(tmp_path, [("un_a", a), ("un_b", b)])
    sql = "SELECT RS_BandPixelType(RS_Union(un_a.rast, un_b.rast), 3) FROM un_a, un_b"
    compare(sql, sedona, spark, expected="REAL_64BITS")
