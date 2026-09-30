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
"""SedonaDB vs Sedona Spark parity for RS_Stack_Aggr.

Every case is an xfail for now: the Sedona Spark 1.9.1 release the suite pins
calls this aggregate RS_Union_Aggr, and the rename to RS_Stack_Aggr
(apache/sedona#3427) is not released yet. The cases start passing once the pin
moves to a release that has RS_Stack_Aggr; drop the module-level xfail then.
(The refusal cases already xpass, because Sedona Spark refuses the name it does
not know.)

Each raster is registered as its own view and the rows are gathered with
UNION ALL, each with its index, listed out of index order so the index and not
the row order decides the band order. Each case compares the whole output
raster, decoded on both engines, and anchors it to the inputs' bands stacked in
index order under the lowest-index raster's grid. As in test_rs_stack.py, the
fixtures carry no nodata value, and Sedona Spark ignoring every raster's
georeference but the first is an xfail of its own.
"""

import numpy as np
import pytest

from sedonadb.raster_testing import DecodedRaster, write_geotiff
from sedonadb.testing import SedonaDB, compare
from sedonadb.testing_spark import SedonaSpark

pytestmark = pytest.mark.xfail(
    reason="Sedona Spark 1.9.1 has no RS_Stack_Aggr; it calls the aggregate "
    "RS_Union_Aggr until the rename in apache/sedona#3427 is released"
)

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


def _raster(dtype="uint8", bands=1, plant=0, bbox=BBOX, width=7, height=6):
    """A random raster, made distinct from its siblings by a planted value."""
    raster = DecodedRaster.random(
        dtype,
        bands=bands,
        width=width,
        height=height,
        bbox=bbox,
        plants={(2, 3): plant},
    )
    raster.nodata = [None] * bands
    return raster


def _stacked(*rasters):
    return DecodedRaster(
        np.concatenate([r.pixels for r in rasters]),
        gdal_transform=rasters[0].gdal_transform,
        nodata=[None] * sum(len(r.pixels) for r in rasters),
    )


def _sql(rows, select="RS_Stack_Aggr(rast, i)"):
    """`select` over `(view, index)` rows gathered with UNION ALL"""
    union = " UNION ALL ".join(f"SELECT rast, {i} AS i FROM {view}" for view, i in rows)
    return f"SELECT {select} FROM ({union}) AS t"


def test_rs_stack_aggr(tmp_path):
    a, b, c = _raster(plant=1), _raster(bands=2, plant=2), _raster(plant=3)
    sedona, spark = _views(tmp_path, [("sa_a", a), ("sa_b", b), ("sa_c", c)])
    sql = _sql([("sa_c", 3), ("sa_a", 1), ("sa_b", 2)])
    compare(sql, sedona, spark, expected=_stacked(a, b, c))


def test_rs_stack_aggr_index_step(tmp_path):
    """Evenly spaced indexes need not step by one."""
    a, b = _raster(plant=1), _raster(plant=2)
    sedona, spark = _views(tmp_path, [("sa_a", a), ("sa_b", b)])
    sql = _sql([("sa_b", 20), ("sa_a", 10)])
    compare(sql, sedona, spark, expected=_stacked(a, b))


@pytest.mark.parametrize("indexes", [(1, 1), (1, 2, 4)], ids=["repeated", "uneven"])
def test_rs_stack_aggr_bad_indexes(indexes, tmp_path):
    """Both engines refuse a repeated index and unevenly spaced indexes. Error
    types differ, so parity here is parity on refusal."""
    rasters = [(f"sa_{n}", _raster(plant=n)) for n in range(len(indexes))]
    sedona, spark = _views(tmp_path, rasters)
    sql = _sql([(view, i) for (view, _), i in zip(rasters, indexes)])
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)


@pytest.mark.xfail(
    reason="SedonaDB rejects a raster on another grid; Sedona Spark keeps the "
    "first raster's grid and ignores the others' georeference"
)
def test_rs_stack_aggr_another_grid(tmp_path):
    """The second raster has the same shape but is georeferenced elsewhere."""
    a, b = _raster(plant=1), _raster(plant=2, bbox=(0, 0, 7, 6))
    sedona, spark = _views(tmp_path, [("sa_a", a), ("sa_b", b)])
    sql = _sql([("sa_a", 1), ("sa_b", 2)])
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)


def test_rs_stack_aggr_shape_mismatch(tmp_path):
    """Both engines refuse rasters whose width or height differ."""
    a, b = _raster(plant=1), _raster(plant=2, width=5, height=4, bbox=(0, 0, 5, 4))
    sedona, spark = _views(tmp_path, [("sa_a", a), ("sa_b", b)])
    sql = _sql([("sa_a", 1), ("sa_b", 2)])
    for eng in (sedona, spark):
        with pytest.raises(Exception):
            eng.decode_raster_result(sql)
