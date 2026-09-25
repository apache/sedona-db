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

"""RS_Union against a numpy reference.

The inputs are GeoTIFFs read through RS_FromPath, which stay out-of-database:
RS_Union carries their bands over by reference, so the test loads the result
with RS_EnsureLoaded before decoding it and comparing with the input pixels,
stacked in argument order.
"""

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    DecodedRaster,
    assert_decoded_equal,
    decode_raster,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")

BBOX = (100.0, 482.0, 114.0, 500.0)


def _tiff(tmp_path, name, dtype, bands, *, bbox=BBOX, nodata=None, seed=42):
    data = random_raster_data(dtype, bands=bands, height=6, width=7, seed=seed)
    path = tmp_path / f"{name}.tif"
    write_geotiff(path, data, bbox=bbox, nodata=nodata)
    return path, data


def _union(con, paths, select="RS_EnsureLoaded({union})"):
    """RS_Union over the rasters at `paths`, as table columns so the kernel runs
    its real array path (literals constant-fold)."""
    names = [f"p{i}" for i in range(len(paths))]
    con.create_data_frame(
        pa.table({name: [str(path)] for name, path in zip(names, paths)})
    ).to_view("union_src", overwrite=True)
    union = "RS_Union(" + ", ".join(f"RS_FromPath({n})" for n in names) + ")"
    sql = f"SELECT {select.format(union=union)} FROM union_src"
    return con.sql(sql).to_arrow_table().column(0)[0]


def test_rs_union(con, tmp_path):
    a, a_data = _tiff(tmp_path, "a", "uint8", 2, nodata=7)
    b, b_data = _tiff(tmp_path, "b", "uint8", 1, seed=1)
    c, c_data = _tiff(tmp_path, "c", "uint8", 2, seed=2)
    got = decode_raster(_union(con, [a, b, c]))
    expected = DecodedRaster(
        np.concatenate([a_data, b_data, c_data]),
        bbox=BBOX,
        nodata=[7, 7, None, None, None],
    )
    assert_decoded_equal(got, expected)


def test_rs_union_keeps_each_bands_pixel_type(con, tmp_path):
    a, _ = _tiff(tmp_path, "a", "uint8", 1)
    b, _ = _tiff(tmp_path, "b", "float64", 1)
    c, _ = _tiff(tmp_path, "c", "int16", 1)
    union = "RS_Union(RS_FromPath(p0), RS_FromPath(p1), RS_FromPath(p2))"
    con.create_data_frame(
        pa.table({"p0": [str(a)], "p1": [str(b)], "p2": [str(c)]})
    ).to_view("union_types", overwrite=True)
    types = (
        con.sql(
            f"SELECT RS_BandPixelType({union}, 1), RS_BandPixelType({union}, 2), "
            f"RS_BandPixelType({union}, 3) FROM union_types"
        )
        .to_arrow_table()
        .to_pylist()[0]
    )
    assert list(types.values()) == ["UNSIGNED_8BITS", "REAL_64BITS", "SIGNED_16BITS"]


def test_rs_union_takes_the_first_georeference(con, tmp_path):
    a, a_data = _tiff(tmp_path, "a", "uint8", 1)
    b, b_data = _tiff(tmp_path, "b", "uint8", 1, bbox=(0.0, 0.0, 7.0, 6.0), seed=1)
    got = decode_raster(_union(con, [a, b]))
    expected = DecodedRaster(
        np.concatenate([a_data, b_data]), bbox=BBOX, nodata=[None, None]
    )
    assert_decoded_equal(got, expected)


def test_rs_union_stays_out_of_database(con, tmp_path):
    """Without RS_EnsureLoaded the joined bands still point at their files."""
    a, _ = _tiff(tmp_path, "a", "uint8", 1)
    b, _ = _tiff(tmp_path, "b", "uint8", 1, seed=1)
    select = "RS_BandPath({union}, 1), RS_BandPath({union}, 2)"
    con.create_data_frame(pa.table({"p0": [str(a)], "p1": [str(b)]})).to_view(
        "union_src", overwrite=True
    )
    union = "RS_Union(RS_FromPath(p0), RS_FromPath(p1))"
    paths = (
        con.sql(f"SELECT {select.format(union=union)} FROM union_src")
        .to_arrow_table()
        .to_pylist()[0]
    )
    assert [p.endswith(n) for p, n in zip(paths.values(), ["a.tif", "b.tif"])] == [
        True,
        True,
    ]


def test_rs_union_shape_mismatch(con, tmp_path):
    a, _ = _tiff(tmp_path, "a", "uint8", 1)
    data = random_raster_data("uint8", bands=1, height=6, width=5)
    b = tmp_path / "b.tif"
    write_geotiff(b, data, bbox=(100.0, 482.0, 110.0, 500.0))
    with pytest.raises(Exception, match="raster 2 is 5 x 6"):
        _union(con, [a, b])
