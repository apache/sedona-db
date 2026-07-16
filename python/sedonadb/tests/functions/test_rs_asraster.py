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

"""RS_AsRaster parity across raster engines.

The reference is `rasterio.features.rasterize` on the same grid. Two
cross-dialect deviations shape what gets compared:

- Fill policy: pixels outside the geometry are initialized with the nodata
  value by SedonaDB but are always 0 in Sedona Spark (which records nodata
  as band metadata only). Parity rows use nodata 0 — where the policies
  coincide — and `test_rs_asraster_nodata_fill_policy` asserts each
  dialect's own fill.
- Centroid rule on diagonal edges: GDAL (SedonaDB, rasterio) burns every
  pixel whose center is inside the geometry, while Sedona Spark's
  geotools/JAI rasterizer drops some center-inside pixels along diagonal
  edges. Diagonal geometry therefore only runs with all_touched=True, where
  the engines agree; the centroid rule is compared on axis-aligned geometry.

Geometries stay inside the reference raster's extent; behavior for
overhanging geometry envelopes is not compared here.
"""

import pytest

from sedonadb.raster_testing import (
    Rasterio,
    SedonaDB,
    SedonaSpark,
    assert_decoded_equal,
    create_dialect_engine,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")
pytest.importorskip("shapely")

DIALECTS = [SedonaDB, SedonaSpark]

# The band types both dialects can express (Sedona Spark has no int8/64-bit
# integer band types).
DTYPES = ["uint8", "uint16", "int16", "int32", "float32", "float64"]

# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up
# pixels; with a 7x6 raster the extent is x in [100, 114], y in [482, 500].
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
HEIGHT, WIDTH = 6, 7
GEOM_RECT = (
    "POLYGON ((102.6 495.8, 109.3 495.8, 109.3 485.9, 102.6 485.9, 102.6 495.8))"
)
# Diagonal edges make all_touched change the selection.
GEOM_TRIANGLE = "POLYGON ((101.3 498.6, 112.4 496.9, 104.2 483.7, 101.3 498.6))"


@pytest.fixture(params=DIALECTS, ids=lambda engine: engine.name())
def dialect(request, con):
    return create_dialect_engine(request.param, con)


@pytest.fixture()
def reference():
    return Rasterio.create_or_skip()


@pytest.fixture()
def tiff(tmp_path):
    path = tmp_path / "asraster_reference.tif"
    write_geotiff(
        path,
        random_raster_data("uint8", bands=1, height=HEIGHT, width=WIDTH),
        gdal_transform=GDAL_TRANSFORM,
    )
    return path


@pytest.mark.parametrize("dtype", DTYPES)
def test_rs_asraster_dtypes_match_reference(dialect, reference, tiff, dtype):
    """Burn value 7 into the geometry's grid-snapped envelope for every band
    type both dialects support."""
    kwargs = dict(burn_value=7.0, nodata=0.0, use_geometry_extent=True)
    got = dialect.as_raster(GEOM_RECT, tiff, dtype, **kwargs)
    expected = reference.as_raster(GEOM_RECT, tiff, dtype, fill=0.0, **kwargs)
    assert_decoded_equal(got, expected, context=dtype)


@pytest.mark.parametrize(
    ("wkt", "all_touched", "use_geometry_extent"),
    [
        (GEOM_RECT, False, True),
        (GEOM_RECT, False, False),
        (GEOM_RECT, True, True),
        (GEOM_TRIANGLE, True, True),
        (GEOM_TRIANGLE, True, False),
    ],
    ids=[
        "rect-centroid-cropped",
        "rect-centroid-full",
        "rect-touched-cropped",
        "triangle-touched-cropped",
        "triangle-touched-full",
    ],
)
def test_rs_asraster_options_match_reference(
    dialect, reference, tiff, wkt, all_touched, use_geometry_extent
):
    """all_touched toggles the selection rule and use_geometry_extent toggles
    between the snapped geometry envelope and the full reference grid. The
    diagonal-edged triangle runs only with all_touched=True (see the module
    docstring for the centroid-rule deviation)."""
    kwargs = dict(
        all_touched=all_touched,
        burn_value=7.0,
        nodata=0.0,
        use_geometry_extent=use_geometry_extent,
    )
    got = dialect.as_raster(wkt, tiff, "uint8", **kwargs)
    expected = reference.as_raster(wkt, tiff, "uint8", fill=0.0, **kwargs)
    assert_decoded_equal(got, expected, context=(wkt, all_touched, use_geometry_extent))


def test_rs_asraster_nodata_fill_policy(dialect, reference, tiff):
    """A nonzero nodata separates the dialects' fill policies: SedonaDB
    initializes outside pixels with the nodata value, Sedona Spark leaves
    them 0 and only records the nodata on the band. Both record nodata 9."""
    fill = 9.0 if isinstance(dialect, SedonaDB) else 0.0
    kwargs = dict(burn_value=7.0, nodata=9.0, use_geometry_extent=False)
    got = dialect.as_raster(GEOM_RECT, tiff, "uint8", **kwargs)
    expected = reference.as_raster(GEOM_RECT, tiff, "uint8", fill=fill, **kwargs)
    assert_decoded_equal(got, expected)


def test_rs_asraster_without_nodata(dialect, reference, tiff):
    """No nodata argument: both dialects burn into zeros and leave the output
    band without a nodata value."""
    got = dialect.as_raster(GEOM_RECT, tiff, "uint8", burn_value=7.0)
    expected = reference.as_raster(GEOM_RECT, tiff, "uint8", burn_value=7.0, fill=0.0)
    assert_decoded_equal(got, expected)
    assert got.nodata == [None]
