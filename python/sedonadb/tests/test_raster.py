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

import pytest

from sedonadb.raster import Raster, RasterArray, RasterScalar, RasterType


def test_type_class_resolution(con):
    t = con.sql("SELECT RS_Example() as raster")
    tab = t.to_arrow_table()

    assert isinstance(tab.schema.field(0).type, RasterType)
    assert isinstance(tab["raster"].type, RasterType)
    assert isinstance(tab["raster"].chunk(0), RasterArray)
    assert isinstance(tab["raster"][0], RasterScalar)
    assert isinstance(tab["raster"][0].as_py(), Raster)


def test_raster_accessors(con):
    t = con.sql("SELECT RS_Example() as raster")
    tab = t.to_arrow_table()
    r: Raster = tab["raster"][0].as_py()

    assert r.crs.to_json_dict()["id"] == {"authority": "OGC", "code": "CRS84"}
    assert r.width == 64
    assert r.height == 32
    assert len(r.transform) == 6
    assert len(r.bands) == 3
    assert repr(r) == "<Raster 64x32, 3 band(s)>"

    b = r.bands[0]
    assert b.name is None
    assert b.shape == (32, 64)
    assert b.source_shape == (32, 64)
    assert b.outdb_uri is None
    assert b.data_type == "uint8"
    assert repr(b) == "<Band uint8 32x64>"

    arr = b.to_numpy()
    assert arr.shape == b.shape
    assert arr[0, 0] == 127

    for i, b in enumerate(r.bands):
        assert b.data[1, 1] == i + 1


def test_raster_to_lit(con):
    t = con.sql("SELECT RS_Example() as raster")
    tab = t.to_arrow_table()
    r = tab["raster"][0].as_py()

    t2 = con.sql(
        "SELECT RS_Width($1) AS w, RS_Height($1) AS h", params=(r,)
    ).to_pandas()
    assert t2.iloc[0, 0] == r.width
    assert t2.iloc[0, 1] == r.height


def test_raster_lazy():
    # Basic lazy raster creation
    r = Raster.lazy(
        uri="s3://bucket/path/to/data.zarr",
        shape=(512, 1024),
        dtype="float32",
    )

    assert r.width == 1024
    assert r.height == 512
    assert len(r.bands) == 1

    b = r.bands[0]
    assert b.source_shape == (512, 1024)
    assert b.data_type == "float32"
    assert b.outdb_uri == "s3://bucket/path/to/data.zarr"

    # Lazy raster should have empty data buffer
    assert len(b.source_data) == 0

    # Accessing data should raise an error for lazy rasters
    with pytest.raises(ValueError, match="external data"):
        _ = b.data


def test_raster_lazy_with_crs():
    r = Raster.lazy(
        uri="s3://bucket/path/to/data.zarr",
        shape=(256, 256),
        dtype="uint8",
        format="zarr",
        crs="EPSG:4326",
    )

    assert r.width == 256
    assert r.height == 256
    assert r.crs.to_json_dict()["id"] == {"authority": "EPSG", "code": 4326}


def test_raster_lazy_invalid_shape():
    with pytest.raises(ValueError, match="exactly two dimensions"):
        Raster.lazy(uri="s3://bucket/data.zarr", shape=(10,), dtype="UInt8")

    with pytest.raises(ValueError, match="exactly two dimensions"):
        Raster.lazy(uri="s3://bucket/data.zarr", shape=(10, 20, 30), dtype="UInt8")
