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

"""RS_AsGeoTiff / binary-constructor round-trip parity.

The comparator for an encode round-trip is the source content itself: after
load -> encode -> decode, pixels, geotransform, nodata, and band type must
be byte-identical for every lossless codec — and the decoded container must
actually carry the codec that was requested, so an encoder that silently
ignores its compression argument fails. These fixtures carry a real CRS
(the same EPSG:3857 on every side, so nothing reprojects): geotools — Sedona
Spark's GeoTIFF writer — cannot encode a CRS-less raster, and no geometry is
involved that a CRS could reinterpret."""

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
    not SedonaDB.implements("as_geotiff"),
    reason="RS_AsGeoTiff is not implemented in SedonaDB (the parity subject)",
)

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7
DTYPES = ["uint8", "uint16", "int16", "int32", "float32", "float64"]
# (compression, quality): lossless codecs only — content must be preserved
# exactly regardless of the quality fraction.
COMPRESSIONS = [(None, None), ("Deflate", 0.75), ("LZW", 0.75), ("PackBits", 0.75)]


def _fixture(tmp_path, dtype):
    tiff = tmp_path / f"asgeotiff_{dtype}.tif"
    write_geotiff(
        tiff,
        random_raster_data(dtype, bands=2, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
        nodata=100.0,
        crs="EPSG:3857",
    )
    return tiff


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("compression", "quality"), COMPRESSIONS, ids=lambda v: str(v))
def test_rs_asgeotiff_roundtrips_content(
    subject, comparator, tmp_path, dtype, compression, quality
):
    tiff = _fixture(tmp_path, dtype)
    got = subject.as_geotiff(tiff, compression=compression, quality=quality)
    expected = comparator.as_geotiff(tiff)
    assert_decoded_equal(got, expected, context=(dtype, compression))
    # The decoded container must carry the requested codec (None = the
    # engine's default, uncompressed) — content survival alone can't tell a
    # working codec path from an ignored argument.
    requested = (compression or "none").lower()
    assert (got.compression or "none").lower() == requested, got.compression


@pytest.mark.skipif(
    not SedonaDB.implements("from_binary"),
    reason="A binary raster constructor is not implemented in SedonaDB "
    "(the parity subject)",
)
def test_rs_from_binary_roundtrips_content(subject, comparator, tmp_path):
    """The binary constructor must decode arbitrary GeoTIFF bytes to the same
    content rasterio reads from them."""
    tiff = _fixture(tmp_path, "uint8")
    data = tiff.read_bytes()
    assert_decoded_equal(subject.from_binary(data), comparator.from_binary(data))
