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

"""RS_Clip cross-checked against a rasterio reference implementation.

Every parity test defines the raster once (numpy array + GDAL geotransform,
written to a CRS-less GeoTIFF) and clips it through both raster engines from
`sedonadb.raster_testing`, comparing pixels, geotransform, and nodata
exactly. Option permutations run as rows of one query (via
`SedonaDB.clip_rows`) so the kernel executes its real array path rather than
constant-folding literals. Cases rasterio cannot express (empty-mask
lenient/strict semantics, out-of-range nodata) are asserted directly against
RS_Clip's documented behavior.
"""

import numpy as np
import pyarrow as pa
import pytest
import shapely

from sedonadb.expr import lit
from sedonadb.raster_testing import (
    Rasterio,
    SedonaDB,
    decode_raster,
    dtype_min,
    random_raster_data,
    write_geotiff,
)


# GDAL-order geotransform: origin (100, 500), 2-wide by 3-tall north-up pixels.
# With a 7x6 raster the extent is x in [100, 114], y in [482, 500]; pixel
# boundaries sit at even x and at y = 482 + 3k, so geometry coordinates below
# deliberately avoid those values (no floor/ceil ambiguity in either engine).
GDAL_TRANSFORM = (100.0, 2.0, 0.0, 500.0, 0.0, -3.0)
BANDS, HEIGHT, WIDTH = 3, 6, 7

GEOM_RECT = (
    "POLYGON ((102.6 495.8, 109.3 495.8, 109.3 485.9, 102.6 485.9, 102.6 495.8))"
)
GEOM_TRIANGLE = "POLYGON ((101.3 498.6, 112.4 496.9, 104.2 483.7, 101.3 498.6))"
GEOM_HOLE = (
    "POLYGON ((102.6 495.8, 109.3 495.8, 109.3 485.9, 102.6 485.9, 102.6 495.8), "
    "(104.7 492.2, 107.1 492.2, 107.1 489.4, 104.7 489.4, 104.7 492.2))"
)
# Extends past the west and north edges of the raster: the crop window is the
# envelope intersected with the raster extent.
GEOM_OVERHANG = "POLYGON ((95 505, 106.3 505, 106.3 490.7, 95 490.7, 95 505))"
# A strip crossing the x = 104 pixel boundary but containing no pixel center
# (centers sit at odd x): selects nothing unless all_touched.
GEOM_SLIVER = "POLYGON ((103.6 499, 104.4 499, 104.4 483, 103.6 483, 103.6 499))"
GEOM_DISJOINT = "POLYGON ((900 900, 910 900, 910 890, 900 890, 900 900))"

# Explicit no_data_value argument and the band nodata baked into the fixture
# GeoTIFF, per dtype. Values are chosen representable in the dtype and distinct
# from each other so a mixup shows up in the comparison.
EXPLICIT_NODATA = {
    "uint8": 250.0,
    "uint16": 65000.0,
    "int16": -9999.0,
    "int32": -99999.0,
    "float32": -9999.5,
    "float64": -12345.5,
}
BAND_NODATA = {
    "uint8": 200.0,
    "uint16": 60000.0,
    "int16": -8888.0,
    "int32": -88888.0,
    "float32": -8888.5,
    "float64": -23456.5,
}


@pytest.fixture()
def sedona(con):
    return SedonaDB(con)


@pytest.fixture()
def reference():
    return Rasterio.create_or_skip()


def _test_data(dtype, band_nodata=None):
    """Fixture pixels with plants tied to this module's geometries: the dtype
    minimum at (3, 2), whose pixel center (105, 489.5) is inside GEOM_TRIANGLE
    at either all_touched setting — so an extreme value must survive a clip
    verbatim. When ``band_nodata`` is given it is planted at (2, 3), also
    inside GEOM_TRIANGLE: a source pixel *valued* at the band nodata stays
    verbatim under a different explicit no_data_value (RS_Clip does not remap
    it the way rasterio.mask.mask would)."""
    plants = {(3, 2): dtype_min(dtype)}
    if band_nodata is not None:
        plants[(2, 3)] = band_nodata
    return random_raster_data(
        dtype, bands=BANDS, height=HEIGHT, width=WIDTH, plants=plants
    )


def _assert_clip_matches_reference(got, reference, tiff, row, fill):
    """Compare one RS_Clip result against the rasterio reference clip.

    ``fill`` is the resolved nodata (explicit argument or band nodata or dtype
    minimum); it is always passed to the reference explicitly because
    rasterio's own fallback for a nodata-less band is 0 while RS_Clip's is the
    dtype minimum.
    """
    expected = reference.clip(
        tiff,
        row["wkt"],
        band=row["band"],
        all_touched=row["all_touched"],
        nodata=fill,
        crop=row["crop"],
    )

    np.testing.assert_array_equal(got.pixels, expected.pixels, err_msg=f"row: {row}")
    assert got.gdal_transform == pytest.approx(
        expected.gdal_transform, rel=1e-12, abs=1e-12
    )
    assert got.nodata == expected.nodata, f"row: {row}"


@pytest.mark.parametrize("dtype", list(EXPLICIT_NODATA))
def test_rs_clip_matches_rasterio_permutations(sedona, reference, tmp_path, dtype):
    """Cross-product of the optional arguments, one row per combination: band
    (0 = all bands, 1..3), all_touched, explicit vs band nodata, crop — over
    two geometries chosen so every axis discriminates. GEOM_TRIANGLE's
    diagonal edges make all_touched matter but its envelope spans the whole
    raster (crop is a no-op there); GEOM_HOLE's interior window has nonzero
    offsets, so its crop=True rows shift the transform and shrink the shape
    while crop=False rows must fill outside an interior window."""
    tiff = tmp_path / f"clip_{dtype}.tif"
    write_geotiff(
        tiff,
        _test_data(dtype, band_nodata=BAND_NODATA[dtype]),
        gdal_transform=GDAL_TRANSFORM,
        nodata=BAND_NODATA[dtype],
    )

    rows = [
        {
            "wkt": wkt,
            "band": band,
            "all_touched": at,
            "nodata": nd,
            "crop": crop,
        }
        for wkt in (GEOM_TRIANGLE, GEOM_HOLE)
        for band in (0, 1, 2, 3)
        for at in (False, True)
        for nd in (None, EXPLICIT_NODATA[dtype])
        for crop in (True, False)
    ]
    results = sedona.clip_rows(tiff, rows)

    for row, got in zip(rows, results):
        assert got is not None, f"row: {row}"
        fill = row["nodata"] if row["nodata"] is not None else BAND_NODATA[dtype]
        _assert_clip_matches_reference(got, reference, tiff, row, fill)


@pytest.mark.parametrize("all_touched", [False, True])
@pytest.mark.parametrize(
    "wkt",
    [GEOM_RECT, GEOM_TRIANGLE, GEOM_HOLE, GEOM_OVERHANG],
    ids=["rect", "triangle", "hole", "overhang"],
)
def test_rs_clip_matches_rasterio_geometries(
    sedona, reference, tmp_path, wkt, all_touched
):
    """Geometry shapes: axis-aligned, diagonal edges, interior ring, and an
    envelope overhanging the raster edge (crop window = envelope ∩ extent)."""
    tiff = tmp_path / "clip_geom.tif"
    write_geotiff(
        tiff,
        _test_data("uint8"),
        gdal_transform=GDAL_TRANSFORM,
        nodata=BAND_NODATA["uint8"],
    )

    row = {
        "wkt": wkt,
        "band": 0,
        "all_touched": all_touched,
        "nodata": None,
        "crop": True,
    }
    got = sedona.clip(tiff, wkt, all_touched=all_touched)
    assert got is not None
    _assert_clip_matches_reference(got, reference, tiff, row, BAND_NODATA["uint8"])


# A sheared grid (one skew term) and a fully rotated one (~30 degrees, both
# skew terms, 2x3 pixels): the cases where envelope-corner mapping through
# the inverse affine and the full-affine transform shift differ from
# scale-only shortcuts.
SKEWED_TRANSFORM = (100.0, 2.0, 0.5, 500.0, 0.25, -3.0)
ROTATED_TRANSFORM = (100.0, 1.7320508, 1.5, 500.0, 1.0, -2.5980762)


@pytest.mark.parametrize("crop", [True, False])
@pytest.mark.parametrize(
    "gdal_transform",
    [SKEWED_TRANSFORM, ROTATED_TRANSFORM],
    ids=["skewed", "rotated"],
)
def test_rs_clip_skewed_rasters_match_rasterio(
    sedona, reference, tmp_path, gdal_transform, crop
):
    """Clipping a skewed or rotated raster: the crop window is the pixel-space
    bounding box of the geometry envelope mapped through the inverted affine,
    and the output geotransform shifts the origin by the full affine (both
    skew terms). The geometry is defined in pixel space and mapped through
    the transform under test so it overlaps the grid whatever the rotation."""
    from rasterio.transform import Affine

    tiff = tmp_path / "clip_skewed.tif"
    write_geotiff(
        tiff,
        _test_data("uint8"),
        gdal_transform=gdal_transform,
        nodata=BAND_NODATA["uint8"],
    )
    affine = Affine.from_gdal(*gdal_transform)
    geom = shapely.Polygon(
        [affine * corner for corner in [(1.2, 0.8), (5.7, 1.3), (2.4, 4.9)]]
    )

    row = {
        "wkt": geom.wkt,
        "band": 0,
        "all_touched": False,
        "nodata": None,
        "crop": crop,
    }
    got = sedona.clip(tiff, geom.wkt, crop=crop)
    assert got is not None
    _assert_clip_matches_reference(got, reference, tiff, row, BAND_NODATA["uint8"])


@pytest.mark.parametrize("dtype", list(EXPLICIT_NODATA) + ["int8", "int64", "uint64"])
def test_rs_clip_default_nodata_sentinel_matches_rasterio(
    sedona, reference, tmp_path, dtype
):
    """No explicit nodata and no band nodata: masked pixels get the dtype
    minimum (exact even for 64-bit integers — no f64 round-trip), and the
    output band records it. GEOM_TRIANGLE leaves most of its crop window
    outside the geometry, so the sentinel is actually written to pixels, not
    just metadata. rasterio is handed the same sentinel because its own
    nodata-less fallback is 0."""
    tiff = tmp_path / f"clip_sentinel_{dtype}.tif"
    write_geotiff(tiff, _test_data(dtype), gdal_transform=GDAL_TRANSFORM)

    row = {
        "wkt": GEOM_TRIANGLE,
        "band": 1,
        "all_touched": False,
        "nodata": None,
        "crop": True,
    }
    got = sedona.clip(tiff, GEOM_TRIANGLE, band=1)
    assert got is not None
    _assert_clip_matches_reference(got, reference, tiff, row, dtype_min(dtype))


def test_rs_clip_signature_defaults(con, tmp_path):
    """Each shorter signature behaves as the full 7-arg form with the defaults
    filled in: all_touched = false, no_data_value = the band's own, crop = true,
    lenient = true."""
    pytest.importorskip("sedonadb_expr")
    tiff = tmp_path / "clip_sigs.tif"
    write_geotiff(
        tiff,
        _test_data("uint8"),
        gdal_transform=GDAL_TRANSFORM,
        nodata=BAND_NODATA["uint8"],
    )
    df = con.create_data_frame(
        pa.table({"path": pa.array([str(tiff)]), "wkt": pa.array([GEOM_TRIANGLE])})
    )

    def clip(**kwargs):
        expr = df.path.funcs.rs_frompath().rst.clip(
            1, con.funcs.st_geomfromtext(df.wkt), **kwargs
        )
        return decode_raster(df.select(r=expr).to_arrow_table()["r"][0])

    full = clip(all_touched=False, no_data_value=lit(None), crop=True, lenient=True)
    for kwargs in [
        {},
        {"all_touched": False},
        {"all_touched": False, "no_data_value": lit(None)},
    ]:
        result = clip(**kwargs)
        np.testing.assert_array_equal(
            result.pixels, full.pixels, err_msg=f"kwargs: {kwargs!r}"
        )
        assert result.gdal_transform == full.gdal_transform, f"kwargs: {kwargs!r}"
        assert result.nodata == full.nodata, f"kwargs: {kwargs!r}"

    # One SQL-text invocation retained so the SQL parser path stays covered
    # (everything else in this module routes through the expression API).
    sql_tab = con.sql(
        "SELECT RS_Clip(RS_FromPath($1), 1, ST_GeomFromText($2)) AS r",
        params=(str(tiff), GEOM_TRIANGLE),
    ).to_arrow_table()
    sql_result = decode_raster(sql_tab["r"][0])
    np.testing.assert_array_equal(sql_result.pixels, full.pixels)
    assert sql_result.nodata == full.nodata


def test_rs_clip_empty_mask_is_null_when_lenient(sedona, reference, tmp_path):
    """rasterio has no equivalent of lenient: a geometry selecting no pixels
    (sliver between pixel centers, or fully disjoint) yields NULL by default,
    and all_touched rescues the sliver case."""
    tiff = tmp_path / "clip_empty.tif"
    write_geotiff(
        tiff,
        _test_data("uint8"),
        gdal_transform=GDAL_TRANSFORM,
        nodata=BAND_NODATA["uint8"],
    )

    rows = [
        {
            "wkt": GEOM_SLIVER,
            "band": 1,
            "all_touched": False,
            "nodata": None,
            "crop": True,
        },
        {
            "wkt": GEOM_DISJOINT,
            "band": 1,
            "all_touched": False,
            "nodata": None,
            "crop": True,
        },
        {
            "wkt": GEOM_SLIVER,
            "band": 1,
            "all_touched": True,
            "nodata": None,
            "crop": True,
        },
    ]
    results = sedona.clip_rows(tiff, rows)

    assert results[0] is None
    assert results[1] is None
    assert results[2] is not None
    _assert_clip_matches_reference(
        results[2], reference, tiff, rows[2], BAND_NODATA["uint8"]
    )


def test_rs_clip_strict_and_argument_errors(con, tmp_path):
    """Error pathways: strict (lenient = false) empty-mask messages depend on
    all_touched; a non-representable nodata and an out-of-range band error
    regardless of leniency."""
    pytest.importorskip("sedonadb_expr")
    tiff = tmp_path / "clip_errors.tif"
    write_geotiff(
        tiff,
        _test_data("uint8"),
        gdal_transform=GDAL_TRANSFORM,
        nodata=BAND_NODATA["uint8"],
    )

    def clip_strict(geom_wkt, *, all_touched=False, nodata=None, band=1):
        df = con.create_data_frame(
            pa.table({"path": pa.array([str(tiff)]), "wkt": pa.array([geom_wkt])})
        )
        expr = df.path.funcs.rs_frompath().rst.clip(
            band,
            con.funcs.st_geomfromtext(df.wkt),
            all_touched=all_touched,
            no_data_value=lit(None) if nodata is None else nodata,
            crop=True,
            lenient=False,
        )
        return df.select(r=expr).to_arrow_table()

    with pytest.raises(Exception, match="do not intersect"):
        clip_strict(GEOM_DISJOINT, all_touched=True)
    with pytest.raises(Exception, match="selects no pixels"):
        clip_strict(GEOM_SLIVER)
    with pytest.raises(Exception, match="not a valid UInt8 value"):
        clip_strict(GEOM_RECT, nodata=-5.0)
    with pytest.raises(Exception, match="out of range"):
        clip_strict(GEOM_RECT, band=4)
