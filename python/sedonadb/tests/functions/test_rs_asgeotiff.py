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

"""RS_AsGeoTiff / binary-constructor round-trip parity against rasterio.

The comparator for an encode round-trip is the source content itself: after
load -> encode -> decode, pixels, geotransform, nodata, and band type must
be byte-identical for every lossless codec — and the decoded container must
actually carry the codec that was requested, so an encoder that silently
ignores its compression argument fails. The fixtures carry a real CRS (the
same EPSG:3857 on every side, so nothing reprojects) so the round-trip is
also exercised with georeferencing present; no geometry is involved that a
CRS could reinterpret."""

import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    assert_decoded_equal,
    decode_geotiff,
    decode_geotiff_bytes,
    decode_raster,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7
DTYPES = ["uint8", "uint16", "int16", "int32", "float32", "float64"]
# (compression, quality): lossless codecs only — content must be preserved
# exactly regardless of the quality fraction.
COMPRESSIONS = [(None, None), ("Deflate", 0.75), ("LZW", 0.75), ("PackBits", 0.75)]


def _sedonadb_as_geotiff(con, path, *, compression=None, quality=None):
    """Load the GeoTIFF at `path`, encode it back to GeoTIFF with RS_AsGeoTiff,
    and decode the returned bytes with rasterio. The path travels as a table
    column so the kernel runs its real array path (literals constant-fold).
    `compression` None selects the one-argument (default, uncompressed) form;
    otherwise the `(raster, compression, quality)` overload is used."""
    df = con.create_data_frame(pa.table({"path": pa.array([str(path)], pa.utf8())}))
    raster = df.path.funcs.rs_frompath()
    if compression is None:
        encoded = raster.funcs.rs_asgeotiff()
    else:
        encoded = raster.funcs.rs_asgeotiff(compression, float(quality))
    tiff_bytes = df.select(t=encoded).to_arrow_table()["t"][0].as_py()
    return decode_geotiff_bytes(bytes(tiff_bytes))


def _rasterio_as_geotiff(path):
    """Reference for the encode round-trip: the source content itself, decoded
    from `path`. Lossless codecs must preserve pixels, transform, and nodata
    bit for bit, so compression options don't change the expectation."""
    return decode_geotiff(path)


def _sedonadb_from_binary(con, data):
    """Decode GeoTIFF bytes into an in-database raster with RS_FromGDALRaster
    and decode the resulting raster scalar. The bytes travel as a table column
    so the kernel runs its real array path (literals constant-fold)."""
    df = con.create_data_frame(pa.table({"content": pa.array([data], pa.binary())}))
    result = df.select(r=df.content.funcs.rs_fromgdalraster()).to_arrow_table()["r"]
    return decode_raster(result[0])


def _rasterio_from_binary(data):
    """Reference for the binary constructor: decode the same bytes with rasterio."""
    return decode_geotiff_bytes(data)


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
def test_rs_asgeotiff_roundtrips_content(con, tmp_path, dtype, compression, quality):
    tiff = _fixture(tmp_path, dtype)
    got = _sedonadb_as_geotiff(con, tiff, compression=compression, quality=quality)
    expected = _rasterio_as_geotiff(tiff)
    assert_decoded_equal(got, expected, context=(dtype, compression))
    # The decoded container must carry the requested codec (None = the
    # engine's default, uncompressed) — content survival alone can't tell a
    # working codec path from an ignored argument.
    requested = (compression or "none").lower()
    assert (got.compression or "none").lower() == requested, got.compression


def test_rs_from_binary_roundtrips_content(con, tmp_path):
    """The binary constructor must decode arbitrary GeoTIFF bytes to the same
    content rasterio reads from them."""
    tiff = _fixture(tmp_path, "uint8")
    data = tiff.read_bytes()
    assert_decoded_equal(_sedonadb_from_binary(con, data), _rasterio_from_binary(data))
