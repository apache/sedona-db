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

"""RS_BandIsNoData against a rasterio reference.

Each fixture is written to a GeoTIFF and read back through RS_FromPath, so the
kernel runs on the band bytes the planner loads for it. The reference reads
the same file with rasterio and checks every pixel against the band's nodata.
"""

import math

import numpy as np
import pyarrow as pa
import pytest

from sedonadb.raster_testing import write_geotiff

rasterio = pytest.importorskip("rasterio")

BBOX = (100.0, 482.0, 114.0, 500.0)


def _rasterio_band_is_nodata(path, band):
    with rasterio.open(path) as src:
        nodata = src.nodatavals[band - 1]
        pixels = src.read(band)
    if nodata is None:
        return False
    if math.isnan(nodata):
        return bool(np.isnan(pixels).all())
    return bool((pixels == nodata).all())


def _sedonadb_band_is_nodata(con, path, band=None):
    """Arguments travel as table columns so the kernel runs its real array
    path (literals constant-fold)."""
    con.create_data_frame(
        pa.table({"path": [str(path)], "band": pa.array([band], pa.int32())})
    ).to_view("bandisnodata_src", overwrite=True)
    band_arg = "" if band is None else ", band"
    sql = f"SELECT RS_BandIsNoData(RS_FromPath(path){band_arg}) FROM bandisnodata_src"
    return con.sql(sql).to_arrow_table().column(0)[0].as_py()


@pytest.fixture()
def two_band_tiff(tmp_path):
    """Band 1 is entirely nodata; band 2 holds one data pixel in its last
    position, so deciding it takes a scan of the whole band."""
    data = np.zeros((2, 6, 7), dtype="uint8")
    data[1, -1, -1] = 1
    path = tmp_path / "bandisnodata.tif"
    write_geotiff(path, data, bbox=BBOX, nodata=0)
    return path


@pytest.mark.parametrize("band", [1, 2])
def test_rs_bandisnodata(con, two_band_tiff, band):
    expected = _rasterio_band_is_nodata(two_band_tiff, band)
    assert expected is (band == 1)
    assert _sedonadb_band_is_nodata(con, two_band_tiff, band) is expected


def test_rs_bandisnodata_defaults_to_band_one(con, two_band_tiff):
    assert _sedonadb_band_is_nodata(con, two_band_tiff) is True


def test_rs_bandisnodata_without_nodata_value(con, tmp_path):
    """Zeros are data when the band declares no nodata value."""
    path = tmp_path / "no_nodata.tif"
    write_geotiff(path, np.zeros((1, 6, 7), dtype="uint8"), bbox=BBOX)
    assert _rasterio_band_is_nodata(path, 1) is False
    assert _sedonadb_band_is_nodata(con, path, 1) is False


def test_rs_bandisnodata_nan_nodata(con, tmp_path):
    """NaN pixels match a NaN nodata value."""
    path = tmp_path / "nan_nodata.tif"
    write_geotiff(
        path, np.full((1, 6, 7), np.nan, dtype="float32"), bbox=BBOX, nodata=np.nan
    )
    assert _rasterio_band_is_nodata(path, 1) is True
    assert _sedonadb_band_is_nodata(con, path, 1) is True


def test_rs_bandisnodata_band_out_of_range(con, two_band_tiff):
    with pytest.raises(Exception, match="RS_BandIsNoData"):
        _sedonadb_band_is_nodata(con, two_band_tiff, 3)
