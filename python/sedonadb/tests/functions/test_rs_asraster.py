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

"""RS_AsRaster parity against a rasterio reference.

The rasterio reference is `rasterio.features.rasterize` on the same grid,
filling outside the geometry with SedonaDB's policy (the output grid is
initialized with the nodata value, 0 when none is given). Geometries stay
inside the reference raster's extent; behavior for overhanging geometry
envelopes is not compared here.
"""

import random

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
shapely = pytest.importorskip("shapely")

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


def _sedonadb_as_raster(
    con,
    geometry_wkt,
    path,
    pixel_type,
    *,
    all_touched=False,
    burn_value=1.0,
    nodata=None,
    use_geometry_extent=True,
):
    """RS_AsRaster over table columns (arguments travel as columns so the
    kernel runs its real array path rather than constant-folding a literal)."""
    df = con.create_data_frame(
        pa.table(
            {
                "path": pa.array([str(path)], type=pa.utf8()),
                "wkt": pa.array([geometry_wkt], type=pa.utf8()),
                "pixel_type": pa.array([pixel_type], type=pa.utf8()),
                "all_touched": pa.array([all_touched], type=pa.bool_()),
                "burn": pa.array([float(burn_value)], type=pa.float64()),
                "nodata": pa.array(
                    [None if nodata is None else float(nodata)], type=pa.float64()
                ),
                "extent": pa.array([use_geometry_extent], type=pa.bool_()),
            }
        )
    )
    result = df.select(
        r=con.funcs.rs_asraster(
            con.funcs.st_geomfromtext(df.wkt),
            df.path.funcs.rs_frompath(),
            df.pixel_type,
            df.all_touched,
            df.burn,
            df.nodata,
            df.extent,
        )
    ).to_arrow_table()["r"]
    return decode_raster(result[0])


def _rasterio_as_raster(
    geometry_wkt,
    path,
    pixel_type,
    *,
    all_touched=False,
    burn_value=1.0,
    nodata=None,
    use_geometry_extent=True,
):
    """Rasterize with `rasterio.features.rasterize`.

    Pixels outside the geometry are filled with SedonaDB's policy: the output
    grid is initialized with the nodata value (0 when none is given).
    """
    import rasterio
    import rasterio.features

    geom = shapely.from_wkt(geometry_wkt)
    with rasterio.open(str(path)) as src:
        if use_geometry_extent:
            window = rasterio.features.geometry_window(src, [geom])
            transform = src.window_transform(window)
            shape = (int(window.height), int(window.width))
        else:
            transform = src.transform
            shape = (src.height, src.width)

    fill = 0.0 if nodata is None else nodata
    for name, value in [("fill", fill), ("burn_value", burn_value)]:
        if np.asarray(value, dtype=pixel_type) != np.asarray(value, dtype="float64"):
            raise ValueError(
                f"{name} {value} is not exactly representable as {pixel_type}"
            )
    pixels = rasterio.features.rasterize(
        [(geom, burn_value)],
        out_shape=shape,
        transform=transform,
        fill=fill,
        all_touched=all_touched,
        dtype=pixel_type,
    )
    return DecodedRaster(pixels[np.newaxis], tuple(transform.to_gdal()), [nodata])


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
def test_rs_asraster_dtypes_match_comparators(con, tiff, dtype):
    """Burn value 7 into the geometry's grid-snapped envelope for every band
    type both dialects support."""
    kwargs = dict(burn_value=7.0, nodata=0.0, use_geometry_extent=True)
    got = _sedonadb_as_raster(con, GEOM_RECT, tiff, dtype, **kwargs)
    expected = _rasterio_as_raster(GEOM_RECT, tiff, dtype, **kwargs)
    assert_decoded_equal(got, expected, context=dtype)


@pytest.mark.parametrize(
    ("wkt", "all_touched", "use_geometry_extent", "nodata"),
    [
        (GEOM_RECT, False, True, 0.0),
        (GEOM_RECT, False, False, 0.0),
        (GEOM_RECT, True, True, 0.0),
        # The nodata-9 rows need pixels outside the geometry in the output
        # (that's where the fill policies diverge): the full reference grid,
        # and the triangle's cropped envelope, have them; the rect's cropped
        # envelope is fully covered and does not.
        (GEOM_RECT, False, False, 9.0),
        (GEOM_TRIANGLE, True, True, 9.0),
        (GEOM_TRIANGLE, False, True, 0.0),
        (GEOM_TRIANGLE, True, True, 0.0),
        (GEOM_TRIANGLE, True, False, 0.0),
    ],
    ids=[
        "rect-centroid-cropped",
        "rect-centroid-full",
        "rect-touched-cropped",
        "rect-centroid-full-nodata9",
        "triangle-touched-cropped-nodata9",
        "triangle-centroid-cropped",
        "triangle-touched-cropped",
        "triangle-touched-full",
    ],
)
def test_rs_asraster_options_match_comparators(
    con, tiff, wkt, all_touched, use_geometry_extent, nodata
):
    """all_touched toggles the selection rule, use_geometry_extent toggles
    between the snapped geometry envelope and the full reference grid, and a
    nonzero nodata exercises the nodata-fill policy."""
    kwargs = dict(
        all_touched=all_touched,
        burn_value=7.0,
        nodata=nodata,
        use_geometry_extent=use_geometry_extent,
    )
    got = _sedonadb_as_raster(con, wkt, tiff, "uint8", **kwargs)
    expected = _rasterio_as_raster(wkt, tiff, "uint8", **kwargs)
    assert_decoded_equal(got, expected, context=(wkt, all_touched, use_geometry_extent))


def test_rs_asraster_without_nodata(con, tiff):
    """No nodata argument: burn into zeros and leave the output band without a
    nodata value."""
    got = _sedonadb_as_raster(con, GEOM_RECT, tiff, "uint8", burn_value=7.0)
    expected = _rasterio_as_raster(GEOM_RECT, tiff, "uint8", burn_value=7.0)
    assert_decoded_equal(got, expected)
    assert got.nodata == [None]


def _fuzz_cases(count=40, seed=31113):
    """Seeded random polygons over anisotropic north-up and south-up grids.

    The fixed seed makes the corpus deterministic, so a failure reproduces
    from the case id alone. Pixel width and height are drawn independently
    to exercise the non-square aspect ratios where rasterizer arithmetic
    errors hide (square unit grids make pixel-space and world-space slopes
    coincide, see apache/sedona#3111).
    """
    rng = random.Random(seed)
    cases = []
    while len(cases) < count:
        width, height = rng.randint(4, 12), rng.randint(4, 12)
        scale_x = round(rng.uniform(0.3, 5.0), 3)
        scale_y = round(rng.uniform(0.3, 5.0), 3) * rng.choice([-1, 1])
        upper_left_x = round(rng.uniform(-1000, 1000), 2)
        upper_left_y = round(rng.uniform(-1000, 1000), 2)
        xs = sorted([upper_left_x, upper_left_x + width * scale_x])
        ys = sorted([upper_left_y, upper_left_y + height * scale_y])
        margin_x = (xs[1] - xs[0]) * 0.05
        margin_y = (ys[1] - ys[0]) * 0.05

        def random_point():
            return (
                round(rng.uniform(xs[0] + margin_x, xs[1] - margin_x), 3),
                round(rng.uniform(ys[0] + margin_y, ys[1] - margin_y), 3),
            )

        num_points = rng.choice([3, 3, 4, 5])
        for _ in range(50):
            candidate = shapely.Polygon(
                [random_point() for _ in range(num_points)]
            ).buffer(0)
            grid_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
            if candidate.geom_type == "Polygon" and candidate.area > grid_area * 0.02:
                cases.append(
                    (
                        len(cases),
                        width,
                        height,
                        (upper_left_x, scale_x, 0.0, upper_left_y, 0.0, scale_y),
                        candidate.wkt,
                    )
                )
                break
    return cases


def test_rs_asraster_fuzz_matches_comparators(con, tmp_path):
    """Centroid-rule burns over the seeded random corpus must match on every
    grid. Only the centroid rule is fuzzed: allTouched boundary selection
    differs between rasterizers by design and is pinned by the deterministic
    cases above."""
    for case_id, width, height, gdal_transform, wkt in _fuzz_cases():
        path = tmp_path / f"fuzz_{case_id}.tif"
        write_geotiff(
            path,
            random_raster_data("uint8", bands=1, height=height, width=width),
            gdal_transform=gdal_transform,
        )
        kwargs = dict(burn_value=1.0, nodata=0.0, use_geometry_extent=False)
        got = _sedonadb_as_raster(con, wkt, path, "uint8", **kwargs)
        expected = _rasterio_as_raster(wkt, path, "uint8", **kwargs)
        assert_decoded_equal(
            got, expected, context=f"case {case_id}: {gdal_transform} {wkt}"
        )
