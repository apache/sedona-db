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

"""Fuzz ``RS_EnsureLoaded`` across loaders with thousands of tiny rasters.

Builds a few thousand tiny local rasters with per-pixel values that are a
deterministic function of ``(source, band, y, x)``:

* multiband Zarr groups, chunked so every chunk position is its own OutDb
  raster row, read with a seeded subset and permutation of the group's arrays
  so band order varies from group to group;
* multiband GeoTIFFs reached through ``RS_FromPath`` (the GDAL loader);
* InDb rasters built from numpy, which ``RS_EnsureLoaded`` must pass through.

The rows are mixed into one table in a seeded random order, some duplicated,
some nulled, and then loaded through SQL with small byte-bounded batches and
several partitions, alongside the ``needs_pixels`` kernels that share the load.
Every band of every output row is checked against the formula, and every
OutDb band's provenance URI against the source it came from. This exercises,
end to end and across loaders, the bookkeeping of bundled loads (grouping by
loader, request-order results, scatter back to ``(raster, band)``, Zarr
store/array/chunk dedupe), byte-bounded batch slicing, and any cross-call
caching. Seeded, so a failure reproduces.
"""

import re
import sys

import numpy as np
import pyarrow as pa
import pytest
import sedonadb
import sedonadb_zarr
from sedonadb.raster import Raster

rasterio = pytest.importorskip("rasterio")
zarr = pytest.importorskip("zarr", minversion="3.0")

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="zarr v3 requires Python 3.11+"
)

# Two seeds: different subsets, orders, duplicates, nulls and sample pixels.
SEEDS = [20260922, 7]

# Sizes are chosen so one seed runs in a few seconds while still pushing
# several thousand band loads through the loaders.
N_GEOTIFF = 2000
# 84 groups of 2-4 arrays is at most 252 distinct arrays, under the Zarr
# loader's 256-entry array-handle LRU, so the handle assertions below can
# expect every array to be opened exactly once.
N_ZARR_GROUPS = 84
N_INDB = 200

DTYPES = [
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
]

# Source ids keep the three kinds apart in the value formula.
GEOTIFF_BASE = 1000
ZARR_BASE = 2000
INDB_BASE = 3000

ZARR_CHUNK = (2, 3)
ZARR_ANCHOR = re.compile(r"#array=a(\d+)&chunk=(\d+),(\d+)$")


def expected_pixels(source, band, dtype, y0, x0, height, width):
    """The pixels of `(source, band)` over rows `y0..` and columns `x0..`.

    Small integers in `1..=251`, exactly representable in every dtype in
    `DTYPES`, and never zero: a Zarr array's default fill value is zero and
    becomes the band's nodata, which `RS_Value` reports as NULL.
    """
    ys, xs = np.mgrid[y0 : y0 + height, x0 : x0 + width]
    return ((source * 7 + band * 13 + ys * 3 + xs) % 251 + 1).astype(dtype)


def geotiff_layout(i):
    """(dtype, band count, height, width) of GeoTIFF `i`."""
    return DTYPES[i % len(DTYPES)], 1 + i % 3, 2 + i % 2, 3 + i % 3


def zarr_layout(g):
    """(dtype, array count, height, width) of Zarr group `g`.

    Heights and widths are multiples of the chunk shape so every chunk is
    full; the chunk grid is what turns one group into several rows.
    """
    return (
        DTYPES[(g * 5) % len(DTYPES)],
        2 + g % 3,
        ZARR_CHUNK[0] * (2 + g % 2),
        ZARR_CHUNK[1] * 2,
    )


def indb_layout(k):
    """(dtype, height, width) of InDb raster `k` (always one band)."""
    return DTYPES[(k * 3) % len(DTYPES)], 2 + k % 3, 2 + k % 2


def write_geotiffs(root):
    """Write `N_GEOTIFF` tiny multiband GeoTIFFs; returns their paths."""
    from rasterio import Affine

    paths = []
    for i in range(N_GEOTIFF):
        dtype, nbands, height, width = geotiff_layout(i)
        path = root / f"tif_{i}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=nbands,
            dtype=dtype,
            # A distinct, non-identity placement per file; the pixel values
            # are what the test checks, not the georeferencing.
            transform=Affine(1.0, 0.0, 10.0 + i, 0.0, -1.0, 5.0),
        ) as dst:
            for b in range(nbands):
                dst.write(
                    expected_pixels(GEOTIFF_BASE + i, b, dtype, 0, 0, height, width),
                    b + 1,
                )
        paths.append(path)
    return paths


def write_zarr_groups(root):
    """Write `N_ZARR_GROUPS` groups of 2-D arrays `a0..aN`; returns their paths."""
    paths = []
    for g in range(N_ZARR_GROUPS):
        dtype, narrays, height, width = zarr_layout(g)
        path = root / f"group_{g}.zarr"
        group = zarr.open_group(str(path), mode="w")
        for a in range(narrays):
            arr = group.create_array(
                f"a{a}",
                shape=(height, width),
                chunks=ZARR_CHUNK,
                dtype=dtype,
                dimension_names=["y", "x"],
            )
            arr[:] = expected_pixels(ZARR_BASE + g, a, dtype, 0, 0, height, width)
        paths.append(path)
    return paths


def parse_key(key):
    kind, rest = key.split(":", 1)
    return kind, [int(v) for v in rest.split(",")]


def band_layout(key):
    """For a row key: (list of (source, band) per output band, dtype, height, width)."""
    kind, ids = parse_key(key)
    if kind == "tif":
        (i,) = ids
        dtype, nbands, height, width = geotiff_layout(i)
        return (
            [(GEOTIFF_BASE + i, b) for b in range(nbands)],
            dtype,
            height,
            width,
            0,
            0,
        )
    if kind == "zarr":
        g, ci, cj, *arrays = ids
        dtype, _, _, _ = zarr_layout(g)
        bands = [(ZARR_BASE + g, a) for a in arrays]
        return (
            bands,
            dtype,
            ZARR_CHUNK[0],
            ZARR_CHUNK[1],
            ci * ZARR_CHUNK[0],
            cj * ZARR_CHUNK[1],
        )
    if kind == "indb":
        (k,) = ids
        dtype, height, width = indb_layout(k)
        return [(INDB_BASE + k, 0)], dtype, height, width, 0, 0
    raise AssertionError(key)


def expected_band(key, b):
    bands, dtype, height, width, y0, x0 = band_layout(key)
    source, band = bands[b]
    return expected_pixels(source, band, dtype, y0, x0, height, width)


def random_pixel(rng, key):
    _, _, height, width, _, _ = band_layout(key)
    return int(rng.integers(0, width)), int(rng.integers(0, height))


def build_rows(tmp_path, rng):
    """One arrow table of Zarr chunk rows and InDb rows, plus the GeoTIFF path table.

    Both carry the same fuzz columns: `key` identifies the source, `ord` is
    the shuffled position, `nrep` how many copies the query emits, `nullify`
    whether the raster is null, and `(px, py)` a pixel to sample.
    """
    sd = sedonadb.connect()
    ext = sedonadb_zarr.ZarrExtension()
    sd.register(ext)

    keys = []
    raster_chunks = []
    for g, path in enumerate(write_zarr_groups(tmp_path)):
        _, narrays, _, _ = zarr_layout(g)
        # A seeded subset of the group's arrays, named in a seeded order, so
        # the band count differs from group to group. The reader emits bands
        # in the group's own order whatever the option order, so the band
        # order is taken from the anchors it wrote rather than assumed.
        chosen = {
            int(a)
            for a in rng.permutation(narrays)[: int(rng.integers(1, narrays + 1))]
        }
        names = [f"a{a}" for a in chosen]
        tab = sd.read(
            f"file://{path}", format=sedonadb_zarr.Zarr({"arrays": names})
        ).to_arrow_table()
        for i in range(tab.num_rows):
            raster = Raster(tab["raster"], i)
            anchors = [ZARR_ANCHOR.search(band.outdb_uri) for band in raster.bands]
            assert all(anchors), [band.outdb_uri for band in raster.bands]
            order = [int(m.group(1)) for m in anchors]
            assert set(order) == chosen and len(order) == len(chosen), (order, chosen)
            ci, cj = int(anchors[0].group(2)), int(anchors[0].group(3))
            keys.append("zarr:" + ",".join(str(v) for v in [g, ci, cj, *order]))
        raster_chunks.append(tab["raster"].combine_chunks())

    for k in range(N_INDB):
        dtype, height, width = indb_layout(k)
        pixels = expected_pixels(INDB_BASE + k, 0, dtype, 0, 0, height, width)
        raster = Raster.from_numpy(pixels, bbox=[0, 0, width, height])
        raster_chunks.append(pa.array(raster))
        keys.append(f"indb:{k}")

    raster_column = pa.concat_arrays(raster_chunks)
    assert raster_column.type == raster_chunks[0].type
    rows = fuzz_columns(rng, keys)
    rows["raster"] = raster_column
    # Null rasters are nulled here rather than with CASE in SQL, which would
    # strip the raster extension type.
    rows["raster"] = with_nulls(raster_column, np.asarray(rows["nullify"]))

    tif_paths = write_geotiffs(tmp_path)
    tif_keys = [f"tif:{i}" for i in range(len(tif_paths))]
    tifs = fuzz_columns(rng, tif_keys)
    tif_nullify = np.asarray(tifs["nullify"])
    tifs["path"] = pa.array(
        [None if tif_nullify[i] else str(p) for i, p in enumerate(tif_paths)],
        pa.utf8(),
    )
    return sd, ext, pa.table(rows), pa.table(tifs)


def with_nulls(column, nullify):
    """`column` with the rows flagged in `nullify` replaced by nulls."""
    null = pa.ExtensionArray.from_storage(
        column.type, pa.nulls(1, column.type.storage_type)
    )
    return pa.concat_arrays(
        [null if nullify[i] else column.slice(i, 1) for i in range(len(column))]
    )


def fuzz_columns(rng, keys):
    n = len(keys)
    pixels = [random_pixel(rng, key) for key in keys]
    return {
        "key": pa.array(keys, pa.utf8()),
        "ord": pa.array(rng.random(n)),
        "nrep": pa.array(
            rng.choice([1, 1, 1, 1, 1, 1, 1, 1, 2, 3], size=n).astype("int32")
        ),
        "nullify": pa.array(rng.random(n) < 0.05),
        "px": pa.array([px for px, _ in pixels], pa.int32()),
        "py": pa.array([py for _, py in pixels], pa.int32()),
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_fuzz_ensure_loaded_mixed_zarr_geotiff_indb(tmp_path, seed):
    rng = np.random.default_rng(seed)
    sd, ext, rows, tifs = build_rows(tmp_path, rng)
    # Reading the groups goes through the chunk reader, not the loader.
    assert ext.loader.handle_stats() == {
        "store_hits": 0,
        "store_misses": 0,
        "array_hits": 0,
        "array_misses": 0,
    }
    sd.create_data_frame(rows).to_view("rows")
    sd.create_data_frame(tifs).to_view("tifs")

    # Small byte-bounded batches and several partitions, so the load path
    # slices, bundles, and scatters constantly rather than once.
    for stmt in [
        "SET datafusion.execution.target_partitions = 4",
        "SET datafusion.execution.batch_size = 97",
        "SET sedona.raster.max_batch_bytes = 2048",
    ]:
        sd.sql(stmt).execute()

    result = sd.sql(
        """
        WITH mixed AS (
            SELECT key, ord, nrep, nullify, px, py, raster FROM rows
            UNION ALL
            SELECT key, ord, nrep, nullify, px, py, RS_FromPath(path) AS raster
            FROM tifs
        ),
        reps AS (SELECT 1 AS rep UNION ALL SELECT 2 UNION ALL SELECT 3),
        fuzzed AS (
            SELECT mixed.*, reps.rep
            FROM mixed CROSS JOIN reps
            WHERE reps.rep <= mixed.nrep
        )
        SELECT
            key, rep, nullify, px, py,
            RS_EnsureLoaded(raster) AS loaded,
            RS_NumBands(raster) AS nbands,
            -- px/py are 0-based array offsets; RS_Value's grid form is 1-based
            RS_Value(raster, px + 1, py + 1, 1) AS first_value,
            RS_Value(raster, px + 1, py + 1, RS_NumBands(raster)) AS last_value
        FROM fuzzed
        ORDER BY ord, rep
        """
    ).to_arrow_table()

    expected_rows = sum(
        int(v) for v in list(rows["nrep"].to_pylist()) + list(tifs["nrep"].to_pylist())
    )
    assert result.num_rows == expected_rows

    seen_keys = set()
    for i in range(result.num_rows):
        key = result["key"][i].as_py()
        seen_keys.add(key)
        bands, dtype, height, width, _, _ = band_layout(key)
        px, py = result["px"][i].as_py(), result["py"][i].as_py()

        if result["nullify"][i].as_py():
            assert not result["loaded"][i].is_valid, key
            assert result["first_value"][i].as_py() is None, key
            assert result["last_value"][i].as_py() is None, key
            continue

        loaded = Raster(result["loaded"], i)
        assert result["nbands"][i].as_py() == len(bands) == len(loaded.bands), key
        for b, band in enumerate(loaded.bands):
            expected = expected_band(key, b)
            actual = band.to_numpy()
            assert actual.dtype == expected.dtype, (key, b)
            assert actual.shape == (height, width), (key, b)
            np.testing.assert_array_equal(actual, expected, err_msg=f"{key} band {b}")
            # Provenance survives the load: OutDb bands keep the URI they
            # were loaded from; InDb bands never had one.
            kind, ids = parse_key(key)
            if kind == "zarr":
                m = ZARR_ANCHOR.search(band.outdb_uri)
                assert m and int(m.group(1)) == bands[b][1], (
                    key,
                    b,
                    band.outdb_uri,
                )
                assert (int(m.group(2)), int(m.group(3))) == (ids[1], ids[2]), key
            elif kind == "tif":
                # The GDAL loader's anchor names the file and the 1-based band.
                assert band.outdb_uri.endswith(f"tif_{ids[0]}.tif#band={b + 1}"), (
                    key,
                    b,
                    band.outdb_uri,
                )
            else:
                assert band.outdb_uri is None, (key, band.outdb_uri)

        first = expected_band(key, 0)[py, px]
        last = expected_band(key, len(bands) - 1)[py, px]
        assert result["first_value"][i].as_py() == pytest.approx(float(first)), key
        assert result["last_value"][i].as_py() == pytest.approx(float(last)), key

    # Every source appeared at least once.
    assert len(seen_keys) == rows.num_rows + tifs.num_rows

    # Handle reuse inside the Zarr loader: every distinct array that was
    # loaded is opened exactly once, every open after the first reuses the
    # single `file://` store client (the client is looked up once per array
    # open), and chunks of one array are spread over many slices, so later
    # slices reuse the opened array.
    loaded_arrays = set()
    for key, nullify in zip(rows["key"].to_pylist(), rows["nullify"].to_pylist()):
        kind, ids = parse_key(key)
        if kind == "zarr" and not nullify:
            loaded_arrays.update((ids[0], a) for a in ids[3:])
    stats = ext.loader.handle_stats()
    assert stats["array_misses"] == len(loaded_arrays), stats
    assert stats["store_misses"] == 1, stats
    assert stats["store_hits"] == len(loaded_arrays) - 1, stats
    assert stats["array_hits"] > 0, stats
