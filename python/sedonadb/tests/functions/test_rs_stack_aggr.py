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
"""RS_Stack_Aggr against a numpy reference.

The inputs are GeoTIFFs read through RS_FromPath, which stay out-of-database:
RS_Stack_Aggr carries their bands over by reference, so each test loads the
result with RS_EnsureLoaded before decoding it and comparing with the input
pixels, stacked in index order.
"""

import numpy as np
import pyarrow as pa
import pytest
import sedonadb

from sedonadb.raster_testing import (
    DecodedRaster,
    assert_decoded_equal,
    decode_raster,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")

BBOX = (100.0, 482.0, 114.0, 500.0)


def _tiff(tmp_path, name, *, bbox=BBOX, crs=None, seed=42):
    data = random_raster_data("uint8", bands=1, height=6, width=7, seed=seed)
    path = tmp_path / f"{name}.tif"
    write_geotiff(path, data, bbox=bbox, crs=crs)
    return path, data


def _stack_aggr(con, rows, group_by=False):
    """RS_Stack_Aggr over (group, path, index) rows; returns {group: raster}"""
    groups, paths, indexes = zip(*rows) if rows else ([], [], [])
    con.create_data_frame(
        pa.table(
            {
                "g": pa.array(groups, pa.string()),
                "p": pa.array([p and str(p) for p in paths], pa.string()),
                "i": pa.array(indexes, pa.int64()),
            }
        )
    ).to_view("stack_aggr_src", overwrite=True)
    stack = "RS_EnsureLoaded(RS_Stack_Aggr(RS_FromPath(p), i))"
    if group_by:
        sql = f"SELECT g, {stack} AS r FROM stack_aggr_src GROUP BY g ORDER BY g"
    else:
        sql = f"SELECT 'all' AS g, {stack} AS r FROM stack_aggr_src"
    table = con.sql(sql).to_arrow_table()
    return dict(zip(table.column("g").to_pylist(), table.column("r")))


def test_rs_stack_aggr_stacks_each_group_in_index_order(tmp_path):
    """Rows arrive out of index order. Four partitions split the aggregate into
    partial and final stages, so each group's state is serialized and merged
    across a repartition before the rasters are stacked."""
    con = sedonadb.connect()
    con.sql("SET datafusion.execution.target_partitions = 4").execute()
    r1, d1 = _tiff(tmp_path, "r1", seed=1)
    r2, d2 = _tiff(tmp_path, "r2", seed=2)
    r3, d3 = _tiff(tmp_path, "r3", seed=3)
    got = _stack_aggr(
        con,
        [
            ("a", r3, 3),
            ("a", r1, 1),
            ("b", r2, 20),
            ("a", r2, 2),
            ("b", r1, 10),
        ],
        group_by=True,
    )
    assert_decoded_equal(
        decode_raster(got["a"]),
        DecodedRaster(np.concatenate([d1, d2, d3]), bbox=BBOX, nodata=[None] * 3),
    )
    assert_decoded_equal(
        decode_raster(got["b"]),
        DecodedRaster(np.concatenate([d1, d2]), bbox=BBOX, nodata=[None] * 2),
    )


def test_rs_stack_aggr_skips_null_rows(con, tmp_path):
    """A row with a NULL raster or a NULL index is left out, and a group with
    no other rows gives NULL."""
    r1, d1 = _tiff(tmp_path, "r1", seed=1)
    r2, _ = _tiff(tmp_path, "r2", seed=2)
    got = _stack_aggr(
        con,
        [
            ("a", r1, 1),
            ("a", None, 2),
            ("a", r2, None),
            ("b", r2, None),
        ],
        group_by=True,
    )
    assert_decoded_equal(
        decode_raster(got["a"]), DecodedRaster(d1, bbox=BBOX, nodata=[None])
    )
    assert not got["b"].is_valid
    assert not _stack_aggr(con, [])["all"].is_valid


def test_rs_stack_aggr_rejects_bad_indexes(con, tmp_path):
    r1, _ = _tiff(tmp_path, "r1", seed=1)
    r2, _ = _tiff(tmp_path, "r2", seed=2)
    r3, _ = _tiff(tmp_path, "r3", seed=3)
    with pytest.raises(Exception, match="index 1 is given to more than one raster"):
        _stack_aggr(con, [("a", r1, 1), ("a", r2, 1)])
    with pytest.raises(Exception, match="indexes must be evenly spaced"):
        _stack_aggr(con, [("a", r1, 1), ("a", r2, 2), ("a", r3, 4)])


def test_rs_stack_aggr_rejects_another_grid(con, tmp_path):
    """A raster of the same shape but georeferenced elsewhere, or in another
    CRS, is not on the lowest-index raster's grid."""
    r1, _ = _tiff(tmp_path, "r1", seed=1)
    moved, _ = _tiff(tmp_path, "moved", bbox=(0.0, 0.0, 7.0, 6.0), seed=2)
    with pytest.raises(Exception, match="the raster with index 2 has geotransform"):
        _stack_aggr(con, [("a", r1, 1), ("a", moved, 2)])
    mercator, _ = _tiff(tmp_path, "mercator", crs="EPSG:3857", seed=3)
    with pytest.raises(Exception, match="the raster with index 2 has a different CRS"):
        _stack_aggr(con, [("a", r1, 1), ("a", mercator, 2)])
