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

"""RS_Value / RS_Values / RS_BandNoDataValue parity.

Every test defines the raster once (numpy array + bbox, written to a
CRS-less GeoTIFF) and samples it through SedonaDB and a rasterio
reference. The fixture bakes a band nodata into the GeoTIFF *and* plants a
pixel valued at that nodata, so one sweep pins all three None rules together:
nodata-valued pixels sample as None, out-of-extent points sample as None,
everything else samples verbatim. Points just west/north of the origin
discriminate flooring from int-truncation in the world-to-pixel math (they
differ only for negative fractional indices).
"""

import pyarrow as pa
import pytest

from sedonadb.raster_testing import (
    _is_nodata,
    random_raster_data,
    write_geotiff,
)

pytest.importorskip("rasterio")

# The extent (minx, miny, maxx, maxy): a 7x6 raster over it has north-up
# pixels 2 wide by 3 tall with the origin at (100, 500).
BBOX = (100.0, 482.0, 114.0, 500.0)
BANDS, HEIGHT, WIDTH = 2, 6, 7

# The pixel planted with the band's own nodata value; sampling it is None.
NODATA_PLANT = (2, 3)  # (row, col)

# Chosen representable in each dtype so a sampled value compares exactly.
BAND_NODATA = {"uint8": 200.0, "int32": -99999.0, "float64": -12345.5}


def pixel_center(row, col):
    """World coordinates of the center of the 0-based pixel (row, col)."""
    return (100.0 + (col + 0.5) * 2.0, 500.0 - (row + 0.5) * 3.0)


SAMPLE_POINTS = [
    pixel_center(0, 0),  # dtype maximum plant
    pixel_center(HEIGHT - 1, WIDTH - 1),  # dtype minimum plant
    pixel_center(*NODATA_PLANT),  # valued at the band nodata
    (103.7, 490.1),  # off-center interior point
    (100.4, 482.3),  # inside the bottom-left pixel, near the corner
    (99.9, 490.0),  # just west of the extent: floor -> col -1, truncate -> 0
    (105.0, 500.2),  # just north of the extent: floor -> row -1, truncate -> 0
    (999.0, 999.0),  # far outside the extent
]


def _write_fixture(tmp_path, dtype, *, nodata):
    tiff = tmp_path / f"value_{dtype}.tif"
    plants = {NODATA_PLANT: nodata} if nodata is not None else None
    write_geotiff(
        tiff,
        random_raster_data(
            dtype, bands=BANDS, height=HEIGHT, width=WIDTH, plants=plants
        ),
        bbox=BBOX,
        nodata=nodata,
    )
    return tiff


def _sedonadb_value(con, path, x, y, *, band=1):
    """RS_Value samples the band at a single point (None for out-of-extent or
    nodata-valued pixels). Arguments travel as table columns so the kernel
    runs its real array path."""
    df = con.create_data_frame(
        pa.table(
            {
                "path": pa.array([str(path)], pa.utf8()),
                "x": pa.array([float(x)], pa.float64()),
                "y": pa.array([float(y)], pa.float64()),
                "band": pa.array([int(band)], pa.int32()),
            }
        )
    )
    result = df.select(
        v=df.path.funcs.rs_frompath().funcs.rs_value(
            con.funcs.st_point(df.x, df.y), df.band
        )
    ).to_arrow_table()["v"]
    return result[0].as_py()


def _sedonadb_values(con, path, points, *, band=1):
    """RS_Values samples every sub-point of a MULTIPOINT in one call and
    returns a List<Double> in input order (None for out-of-extent or
    nodata-valued pixels). Arguments travel as table columns so the kernel
    runs its real array path."""
    wkt = (
        "MULTIPOINT ("
        + ", ".join(f"{float(x)!r} {float(y)!r}" for x, y in points)
        + ")"
    )
    df = con.create_data_frame(
        pa.table(
            {
                "path": pa.array([str(path)], pa.utf8()),
                "wkt": pa.array([wkt], pa.utf8()),
                "band": pa.array([int(band)], pa.int32()),
            }
        )
    )
    result = df.select(
        v=df.path.funcs.rs_frompath().funcs.rs_values(
            con.funcs.st_geomfromtext(df.wkt), df.band
        )
    ).to_arrow_table()["v"]
    return result[0].as_py()


def _sedonadb_band_nodata(con, path, *, band=1):
    """RS_BandNoDataValue reads back the band's nodata (None when unset).
    Arguments travel as table columns so the kernel runs its real array
    path."""
    df = con.create_data_frame(
        pa.table(
            {
                "path": pa.array([str(path)], pa.utf8()),
                "band": pa.array([int(band)], pa.int32()),
            }
        )
    )
    result = df.select(
        v=df.path.funcs.rs_frompath().funcs.rs_bandnodatavalue(df.band)
    ).to_arrow_table()["v"]
    return result[0].as_py()


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("rs_value_scan", id="table"),
        pytest.param(
            "(SELECT RS_EnsureLoaded(r) AS r FROM rs_value_scan)", id="user_loaded"
        ),
    ],
)
def test_two_reads_sharing_a_raster_column_over_a_scan(con, tmp_path, source):
    """Two RS_Value calls on one raster column of a table, and a read over a
    subquery that already called RS_EnsureLoaded, used to fail with
    'async functions should not be called directly'. The planner wraps each
    call's raster in an async RS_EnsureLoaded; once DataFusion had turned that
    wrap into a plain column (common-subexpression elimination hoists the
    shared loader into a projection below; the user's subquery is such a
    projection already) the next optimizer pass wrapped the column again,
    nesting async calls the physical planner cannot hoist. Both reads are
    cross-checked against rasterio."""
    path = _write_fixture(tmp_path, "int32", nodata=BAND_NODATA["int32"])
    con.create_data_frame(
        con.sql("SELECT RS_FromPath($1) AS r", params=(str(path),)).to_arrow_table()
    ).to_view("rs_value_scan", overwrite=True)
    (x1, y1), (x2, y2) = pixel_center(0, 1), pixel_center(3, 4)

    got = con.sql(
        f"SELECT RS_Value(r, ST_Point({x1}, {y1}), 1) AS a, "
        f"RS_Value(r, ST_Point({x2}, {y2}), 1) AS b FROM {source}"
    ).to_arrow_table()

    assert got["a"][0].as_py() == _rasterio_value(path, x1, y1)
    assert got["b"][0].as_py() == _rasterio_value(path, x2, y2)


def _rasterio_value(path, x, y, *, band=1):
    return _rasterio_values(path, [(x, y)], band=band)[0]


def _rasterio_values(path, points, *, band=1):
    import rasterio

    with rasterio.open(str(path)) as src:
        data = src.read(band)
        nodata = src.nodatavals[band - 1]
        out = []
        for x, y in points:
            # index() floors through the inverse transform, so a pixel
            # owns its upper-left edges — the same ownership rule the
            # dialects use. Fixture points avoid pixel boundaries anyway.
            row, col = src.index(x, y)
            if not (0 <= row < src.height and 0 <= col < src.width):
                out.append(None)
                continue
            sampled = data[row, col]
            out.append(None if _is_nodata(sampled, nodata) else float(sampled))
    return out


def _rasterio_band_nodata(path, *, band=1):
    import rasterio

    with rasterio.open(str(path)) as src:
        return src.nodatavals[band - 1]


@pytest.mark.parametrize("dtype", list(BAND_NODATA))
def test_rs_value_matches_comparators(con, tmp_path, dtype):
    """Point sampling over both bands: the dtype extremes planted in opposite
    corners must survive verbatim, the nodata-valued pixel and every
    out-of-extent point must be None, and off-center points must floor to
    the same owning pixel in both engines."""
    tiff = _write_fixture(tmp_path, dtype, nodata=BAND_NODATA[dtype])

    for band in (1, 2):
        for x, y in SAMPLE_POINTS:
            got = _sedonadb_value(con, tiff, x, y, band=band)
            expected = _rasterio_value(tiff, x, y, band=band)
            assert got == expected, f"band {band}, point ({x}, {y})"


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
def test_rs_values_matches_comparators(con, tmp_path, dtype):
    """Multi-point sampling: every pixel center plus the boundary and
    out-of-extent points in one call, results in input order."""
    tiff = _write_fixture(tmp_path, dtype, nodata=BAND_NODATA[dtype])

    points = [
        pixel_center(row, col) for row in range(HEIGHT) for col in range(WIDTH)
    ] + SAMPLE_POINTS
    for band in (1, 2):
        got = _sedonadb_values(con, tiff, points, band=band)
        expected = _rasterio_values(tiff, points, band=band)
        assert got == expected, f"band {band}"


@pytest.mark.parametrize("dtype", ["uint8", "float64"])
def test_rs_band_nodata_matches_comparators(con, tmp_path, dtype):
    """The band nodata reads back exactly on every band, and a band without
    one reads back as None."""
    with_nodata = _write_fixture(tmp_path, dtype, nodata=BAND_NODATA[dtype])
    without_nodata = tmp_path / f"no_nodata_{dtype}.tif"
    write_geotiff(
        without_nodata,
        random_raster_data(dtype, bands=BANDS, height=HEIGHT, width=WIDTH),
        bbox=BBOX,
    )

    for band in (1, 2):
        assert (
            _sedonadb_band_nodata(con, with_nodata, band=band)
            == _rasterio_band_nodata(with_nodata, band=band)
            == BAND_NODATA[dtype]
        )
        assert _sedonadb_band_nodata(con, without_nodata, band=band) is None
        assert _rasterio_band_nodata(without_nodata, band=band) is None


# A nodata a GeoTIFF can declare but the band dtype cannot hold exactly, and
# the real pixel value a saturating cast into the dtype would turn it into.
UNREPRESENTABLE_NODATA = [
    pytest.param("uint8", -9999.0, 0, id="below_range"),
    pytest.param("uint8", 256.0, 255, id="above_range"),
    pytest.param("int16", float("nan"), 0, id="nan"),
    pytest.param("int32", 0.5, 0, id="fraction"),
]


@pytest.mark.parametrize("loader", ["RS_FromPath(path)", "RS_FromGDALRaster(content)"])
@pytest.mark.parametrize(("dtype", "nodata", "saturated"), UNREPRESENTABLE_NODATA)
def test_unrepresentable_file_nodata_is_dropped(
    con, tmp_path, loader, dtype, nodata, saturated
):
    """A file nodata the band dtype cannot hold exactly matches no pixel, so
    the band reads as having none: RS_BandNoDataValue is NULL and every pixel
    samples verbatim, including the planted one a saturating cast would have
    matched (-9999 on uint8 used to turn every 0 pixel into nodata). Covers
    both the out-db (RS_FromPath) and in-db (RS_FromGDALRaster) loaders."""
    data = random_raster_data(
        dtype, bands=1, height=HEIGHT, width=WIDTH, plants={NODATA_PLANT: saturated}
    )
    tiff = tmp_path / "unrepresentable.tif"
    write_geotiff(tiff, data, bbox=BBOX, nodata=nodata)
    points = [pixel_center(row, col) for row in range(HEIGHT) for col in range(WIDTH)]
    wkt = "MULTIPOINT (" + ", ".join(f"{x!r} {y!r}" for x, y in points) + ")"
    con.create_data_frame(
        pa.table({"path": [str(tiff)], "content": [tiff.read_bytes()], "wkt": [wkt]})
    ).to_view("unrepresentable_src", overwrite=True)

    got = con.sql(
        "SELECT RS_BandNoDataValue(r, 1) AS nodata, "
        "RS_Values(r, ST_GeomFromText(wkt), 1) AS pixels "
        f"FROM (SELECT {loader} AS r, wkt FROM unrepresentable_src)"
    ).to_arrow_table()

    expected = [float(v) for v in data[0].ravel()]
    assert _rasterio_values(tiff, points) == expected
    assert got.to_pylist() == [{"nodata": None, "pixels": expected}]
