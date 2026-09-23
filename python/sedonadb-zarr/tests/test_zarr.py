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

import sys

import numpy as np
import pytest
import sedonadb
import sedonadb_zarr
import zarr
from sedonadb.raster import Raster

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="zarr v3 requires Python 3.11+"
)


@pytest.fixture
def zarr_group(tmp_path):
    """Build a tiny 2x2 UInt8 Zarr v3 group with two chunks."""
    root = zarr.open_group(str(tmp_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=(2, 2),
        chunks=(1, 2),
        dtype="uint8",
        dimension_names=["y", "x"],
    )
    arr[:] = np.array([[10, 11], [20, 21]], dtype=np.uint8)
    return tmp_path


def test_format_spec_via_read(zarr_group):
    con = sedonadb.connect()
    df = con.read(f"file://{zarr_group}", format=sedonadb_zarr.Zarr())
    arrow_tab = df.to_arrow_table()
    assert arrow_tab.num_rows == 2
    assert arrow_tab.column_names == ["raster"]

    raster = arrow_tab["raster"][0].as_py()
    assert isinstance(raster, Raster), f"raster row is {type(raster).__name__}"
    assert raster.transform is not None
    assert len(raster.bands) >= 1
    band = raster.bands[0]
    # `source_data` is empty (OutDb scan); `outdb_uri` points at this chunk.
    assert len(band.source_data) == 0, (
        f"OutDb band should have empty data; got {len(band.source_data)} bytes"
    )
    assert band.outdb_uri is not None and "#array=temperature" in band.outdb_uri, (
        f"unexpected anchor: {band.outdb_uri!r}"
    )


def test_zarr_url_as_table(tmp_path):
    """`SELECT * FROM '<.zarr url>'` reads a Zarr group with no explicit format.

    This is the motivating case for the URL-as-table resolver: a bare
    `FROM '<url>'` keys off the `.zarr` extension and routes the directory
    through the single-object table path, yielding the same rasters as the
    explicit `read(url, format="zarr")`. DataFusion's default resolver instead
    lists the directory's inner chunks and tries to parse one as the wrong
    format, failing with `Json error: Not valid JSON: EOF ...`.
    """
    zarr = pytest.importorskip("zarr", minversion="3.0")
    np = pytest.importorskip("numpy")
    pytest.importorskip("sedonadb_zarr")

    # The resolver keys off the path extension, so the group must live at a
    # `.zarr`-suffixed directory (pytest's `tmp_path` itself has no suffix).
    # Same 2x2 UInt8 / (1, 2)-chunk layout as the `zarr_group` fixture, so it
    # reads as two OutDb raster rows (one per chunk).
    zarr_path = tmp_path / "temperature.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=(2, 2),
        chunks=(1, 2),
        dtype="uint8",
        dimension_names=["y", "x"],
    )
    arr[:] = np.array([[10, 11], [20, 21]], dtype=np.uint8)

    con = sedonadb.connect()
    con.register(sedonadb_zarr.ZarrExtension())

    url = zarr_path.as_uri()

    # Ground truth: the working explicit-format read of the same `.zarr` group.
    expected = con.read(url, format="zarr").to_arrow_table()

    def assert_matches_expected(table):
        # A row count of 2 already distinguishes the group resolving as a single
        # table from a listing over its chunks (which would have a different
        # count), so matching the explicit read's row count is enough here.
        assert table.num_rows == expected.num_rows == 2
        assert table.column_names == ["raster"]

    # The `file://` URL form is the primary SQL-text feature under test.
    assert_matches_expected(con.sql(f"SELECT * FROM '{url}'").to_arrow_table())

    # A bare filesystem path (no `file://` scheme) resolves the same way.
    assert_matches_expected(con.sql(f"SELECT * FROM '{zarr_path}'").to_arrow_table())


# A north-up affine in spatial:transform order [a, b, c, d, e, f]: origin
# (10, 20), 1x-1 pixels. A single-chunk 2x2 raster then spans x in [10, 12],
# y in [18, 20]. Encoding the *same* georeferencing under different attribute
# spellings must produce the same RS_Envelope — that is the permutation matrix
# below. A reader that misreads the affine as GDAL order yields a degenerate
# envelope, so RS_Envelope is a tight guard on the transform handling.
_NORTH_UP_AFFINE = [1.0, 0.0, 10.0, 0.0, -1.0, 20.0]
_NORTH_UP_BOUNDS = (10.0, 18.0, 12.0, 20.0)


def _zarr_with_attrs(tmp_path, group_attrs, *, dims=("y", "x")):
    """Write a single-chunk 2x2 Zarr v3 group carrying `group_attrs`."""
    root = zarr.open_group(str(tmp_path), mode="w")
    for key, value in group_attrs.items():
        root.attrs[key] = value
    arr = root.create_array(
        "temperature",
        shape=(2, 2),
        chunks=(2, 2),  # single chunk -> one raster row over the full extent
        dtype="uint8",
        dimension_names=list(dims),
    )
    arr[:] = np.zeros((2, 2), dtype=np.uint8)
    return tmp_path


def _envelope_bounds(con, path):
    """Read the single-chunk zarr and return RS_Envelope bounds of row 0."""
    shapely = pytest.importorskip("shapely")
    df = con.read(f"file://{path}", format=sedonadb_zarr.Zarr())
    raster = df.to_arrow_table()["raster"][0].as_py()
    wkt = (
        con.sql("SELECT ST_AsText(RS_Envelope($1)) AS wkt", params=(raster,))
        .to_arrow_table()["wkt"][0]
        .as_py()
    )
    return shapely.from_wkt(wkt).bounds


@pytest.mark.parametrize(
    "group_attrs, dims",
    [
        # Canonical current convention.
        pytest.param(
            {
                "proj:code": "EPSG:4326",
                "spatial:dimensions": ["y", "x"],
                "spatial:transform": _NORTH_UP_AFFINE,
            },
            ("y", "x"),
            id="proj_code+spatial_dimensions",
        ),
        # Legacy aliases: proj:epsg (int) + spatial:dims.
        pytest.param(
            {
                "proj:epsg": 4326,
                "spatial:dims": ["y", "x"],
                "spatial:transform": _NORTH_UP_AFFINE,
            },
            ("y", "x"),
            id="legacy_proj_epsg+spatial_dims",
        ),
        # Mixed old/new spelling.
        pytest.param(
            {
                "proj:epsg": 4326,
                "spatial:dimensions": ["y", "x"],
                "spatial:transform": _NORTH_UP_AFFINE,
            },
            ("y", "x"),
            id="mixed_proj_epsg+spatial_dimensions",
        ),
        # No spatial:dimensions -> inferred from the recognized (y, x) pair.
        pytest.param(
            {"proj:code": "EPSG:4326", "spatial:transform": _NORTH_UP_AFFINE},
            ("y", "x"),
            id="inferred_dims_y_x",
        ),
        # latitude/longitude is also a recognized spatial pair.
        pytest.param(
            {"proj:code": "EPSG:4326", "spatial:transform": _NORTH_UP_AFFINE},
            ("latitude", "longitude"),
            id="inferred_dims_lat_lon",
        ),
    ],
)
def test_rs_envelope_across_attr_permutations(tmp_path, group_attrs, dims):
    """The same georeferencing under different attribute spellings yields the
    same world-coordinate envelope."""
    con = sedonadb.connect()
    bounds = _envelope_bounds(con, _zarr_with_attrs(tmp_path, group_attrs, dims=dims))
    assert bounds == pytest.approx(_NORTH_UP_BOUNDS)


def test_rs_envelope_honors_skew(tmp_path):
    """A non-zero skew term (`b` in `[a, b, c, d, e, f]`) must land in the
    right transform slot — proves the full affine->GDAL reorder, not just the
    origin. affine [1, 0.5, 10, 0, -1, 20]: wx = col + 0.5*row + 10, wy =
    20 - row; corners -> (10,20),(12,20),(13,18),(11,18); AABB x[10,13]."""
    attrs = {
        "proj:code": "EPSG:4326",
        "spatial:dimensions": ["y", "x"],
        "spatial:transform": [1.0, 0.5, 10.0, 0.0, -1.0, 20.0],
    }
    con = sedonadb.connect()
    bounds = _envelope_bounds(con, _zarr_with_attrs(tmp_path, attrs))
    assert bounds == pytest.approx((10.0, 18.0, 13.0, 20.0))


def test_format_spec_with_arrays_option(zarr_group):
    con = sedonadb.connect()
    spec = sedonadb_zarr.Zarr().with_options({"arrays": ["temperature"]})

    # Check via constructor options
    df = con.read(f"file://{zarr_group}", format=spec)
    assert df.to_arrow_table().num_rows == 2

    # Check via read(..., options={...})
    df = con.read(
        f"file://{zarr_group}",
        options={"arrays": ["temperature"]},
        format=sedonadb_zarr.Zarr(),
    )
    assert df.to_arrow_table().num_rows == 2


def test_arrays_option_order_is_band_order(tmp_path):
    """With `arrays`, band i is arrays[i]; without it, bands are sorted by path."""
    root = zarr.open_group(str(tmp_path), mode="w")
    for name, value in [("a0", 10), ("a1", 20), ("b2", 30)]:
        arr = root.create_array(
            name, shape=(2, 2), chunks=(2, 2), dtype="uint8", dimension_names=["y", "x"]
        )
        arr[:] = value

    con = sedonadb.connect()
    con.register(sedonadb_zarr.ZarrExtension())

    def band_arrays(names):
        fmt = sedonadb_zarr.Zarr({"arrays": names}) if names else sedonadb_zarr.Zarr()
        tab = con.read(f"file://{tmp_path}", format=fmt).to_arrow_table()
        raster = Raster(tab["raster"], 0)
        return [b.outdb_uri.split("#array=")[1].split("&")[0] for b in raster.bands]

    assert band_arrays(["a1", "a0"]) == ["a1", "a0"]
    assert band_arrays(["b2", "a0", "a1"]) == ["b2", "a0", "a1"]
    assert band_arrays(["b2", "a1"]) == ["b2", "a1"]
    assert band_arrays(None) == ["a0", "a1", "b2"]

    # The loaded pixels follow the same order.
    fmt = sedonadb_zarr.Zarr({"arrays": ["b2", "a0"]})
    df = con.read(f"file://{tmp_path}", format=fmt)
    tab = df.select(r=df.raster.funcs.rs_ensureloaded()).to_arrow_table()
    bands = Raster(tab["r"], 0).bands
    assert [int(b.to_numpy()[0, 0]) for b in bands] == [30, 10]


def test_format_spec_class_invariants():
    spec = sedonadb_zarr.Zarr()
    assert spec.extension == "zarr"
    spec2 = spec.with_options({"arrays": ["temperature"]})
    assert spec2 is not spec


def test_zarr_loader_supports_format():
    """Test format support checking."""
    loader = sedonadb_zarr.ZarrRasterLoader()
    assert loader.supports_format("zarr") is True
    assert loader.supports_format(None) is False
    assert loader.supports_format("gdal") is False
    assert "ZarrRasterLoader" in repr(loader)


def test_zarr_loader_io_concurrency_is_configurable():
    assert sedonadb_zarr.ZarrRasterLoader().io_concurrency() > 0
    assert sedonadb_zarr.ZarrRasterLoader(io_concurrency=3).io_concurrency() == 3

    sd = sedonadb.connect()
    ext = sedonadb_zarr.ZarrExtension(io_concurrency=5)
    sd.register(ext)
    assert ext.loader.io_concurrency() == 5


def test_zarr_loader_handle_stats_start_at_zero():
    loader = sedonadb_zarr.ZarrRasterLoader()
    assert loader.handle_stats() == {
        "store_hits": 0,
        "store_misses": 0,
        "array_hits": 0,
        "array_misses": 0,
    }


def test_zarr_extension_reports_handle_reuse_across_calls(zarr_group):
    sd = sedonadb.connect()
    ext = sedonadb_zarr.ZarrExtension()
    assert ext.loader is None
    sd.register(ext)
    assert isinstance(ext.loader, sedonadb_zarr.ZarrRasterLoader)

    t = sd.read(f"file://{zarr_group}", format="zarr")

    def load_all():
        tab = t.select(raster=t.raster.funcs.rs_ensureloaded()).to_arrow_table()
        assert tab.num_rows == 2

    load_all()
    first = ext.loader.handle_stats()
    # The one array in the group is opened once and its client built once.
    assert first["array_misses"] == 1
    assert first["store_misses"] == 1

    load_all()
    second = ext.loader.handle_stats()
    # The second query reuses the opened array and builds no new client.
    assert second["array_hits"] > first["array_hits"]
    assert second["array_misses"] == first["array_misses"]
    assert second["store_misses"] == 1


@pytest.mark.parametrize(
    "numpy_dtype",
    [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_rs_ensure_loaded_with_zarr(tmp_path, numpy_dtype):
    # Tune these for coverage vs speed tradeoff, but ensure a reasonable
    # number of tiles to test a larger degree of concurrency
    width, height = 512, 512
    chunk_width, chunk_height = 16, 16

    # Create a Zarr array with random data
    rng = np.random.default_rng(seed=836)
    if numpy_dtype == "bool":
        numpy_arr = rng.integers(0, 2, (height, width), dtype=np.uint8).astype(
            numpy_dtype
        )
    elif np.issubdtype(np.dtype(numpy_dtype), np.integer):
        numpy_arr = rng.integers(0, 100, (height, width), dtype=numpy_dtype)
    else:
        numpy_arr = rng.random((height, width)).astype(numpy_dtype)

    root = zarr.open_group(str(tmp_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=(height, width),
        chunks=(chunk_height, chunk_width),
        dtype=numpy_dtype,
        dimension_names=["y", "x"],
    )
    arr[:] = numpy_arr

    # Create a fresh connection (new context)
    sd = sedonadb.connect()
    sd.register(sedonadb_zarr.ZarrExtension())

    # Read the Zarr group as loaded rasters
    t = sd.read(f"file://{tmp_path}", format="zarr")
    loaded_tab = t.select(raster=t.raster.funcs.rs_ensureloaded()).to_arrow_table()

    # Verify we get the expected number of chunk rows
    expected_chunks_y = (height + chunk_height - 1) // chunk_height
    expected_chunks_x = (width + chunk_width - 1) // chunk_width
    expected_rows = expected_chunks_y * expected_chunks_x
    assert loaded_tab.num_rows == expected_rows

    # Verify the total pixels and that each chunk loaded successfully
    total_pixels = 0
    for i in range(loaded_tab.num_rows):
        raster = loaded_tab["raster"][i].as_py()
        band = raster.bands[0]
        chunk_data = band.to_numpy()
        # bool is stored as uint8 in raster bands
        expected_dtype = np.uint8 if numpy_dtype == "bool" else np.dtype(numpy_dtype)
        assert chunk_data.dtype == expected_dtype
        total_pixels += chunk_data.size

    assert total_pixels == width * height


# Each numpy dtype below maps to a different `BandDataType` arm in
# `rust/sedona-raster-zarr/src/dtype.rs::zarr_to_band_data_type`.
@pytest.mark.parametrize(
    "numpy_dtype",
    [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_dtype_mapping_roundtrips(tmp_path, numpy_dtype):
    root = zarr.open_group(str(tmp_path), mode="w")
    arr = root.create_array(
        "temperature",
        shape=(2, 2),
        chunks=(1, 2),
        dtype=numpy_dtype,
        dimension_names=["y", "x"],
    )
    arr[:] = np.ones((2, 2), dtype=numpy_dtype)

    con = sedonadb.connect()
    df = con.read(f"file://{tmp_path}", format=sedonadb_zarr.Zarr())
    tab = df.to_arrow_table()
    assert tab.num_rows == 2

    # We can't extract data because these are OutDB refs
    with pytest.raises(
        ValueError, match="Can't extract buffer from a reference to external data"
    ):
        tab["raster"][0].as_py().bands[0].to_numpy()
