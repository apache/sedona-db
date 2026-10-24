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

import pandas as pd
import pyarrow as pa
import pytest
import sedonadb
from sedonadb import udf


def some_udf(arg0, arg1):
    arg0, arg1 = (
        pa.array(arg0.to_array()).to_pylist(),
        pa.array(arg1.to_array()).to_pylist(),
    )
    return pa.array(
        (f"{item0} / {item1}".encode() for item0, item1 in zip(arg0, arg1)),
        pa.binary(),
    )


def test_udf_matchers(con):
    udf_impl = udf.arrow_udf(pa.binary(), [udf.STRING, udf.NUMERIC])(some_udf)
    assert udf_impl._name == "some_udf"

    con.register(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql("SELECT some_udf('abcd', 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )


@pytest.mark.parametrize(
    ("matcher", "expected_type"),
    [
        (udf.BINARY, pa.binary()),
        (udf.BOOLEAN, pa.bool_()),
        (udf.NUMERIC, pa.float64()),
        (udf.STRING, pa.string()),
    ],
)
def test_type_matcher_arrow_c_schema(matcher, expected_type):
    assert isinstance(matcher, udf.TypeMatcher)
    field = pa.Field._import_from_c_capsule(matcher.__arrow_c_schema__())
    assert field.type == expected_type


@pytest.mark.parametrize(
    ("matcher", "extension_name"),
    [
        (udf.GEOGRAPHY, b"geoarrow.wkb"),
        (udf.GEOMETRY, b"geoarrow.wkb"),
        (udf.RASTER, b"sedona.raster"),
    ],
)
def test_extension_type_matcher_arrow_c_schema(matcher, extension_name):
    field = pa.Field._import_from_c_capsule(matcher.__arrow_c_schema__())
    if isinstance(field.type, pa.ExtensionType):
        actual_extension_name = field.type.extension_name.encode()
    else:
        actual_extension_name = field.metadata[b"ARROW:extension:name"]
    assert actual_extension_name == extension_name


def test_large_wkb_geography_matcher_preserves_edge_type(con):
    import geoarrow.pyarrow as ga
    from shapely.geometry import Point

    geography_type = (
        ga.large_wkb().with_edge_type(ga.EdgeType.SPHERICAL).with_crs("OGC:CRS84")
    )
    table = pa.table({"geom": pa.array([Point(1, 2).wkb], type=geography_type)})
    df = con.create_data_frame(table)

    assert df.schema.field("geom").type.edge_type == ga.EdgeType.SPHERICAL

    @udf.arrow_udf(pa.bool_(), [udf.GEOGRAPHY])
    def is_spherical(geom, *, num_rows):
        assert geom.type.edge_type == ga.EdgeType.SPHERICAL
        return pa.array([True] * num_rows)

    con.register(is_spherical)
    df.to_view("large_wkb_geography", overwrite=True)
    assert con.sql(
        "SELECT is_spherical(geom) FROM large_wkb_geography"
    ).to_arrow_table().column(0).to_pylist() == [True]


def test_udf_types(con):
    udf_impl = udf.arrow_udf(pa.binary(), [pa.string(), pa.int64()])(some_udf)
    assert udf_impl._name == "some_udf"

    con.register(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql("SELECT some_udf('abcd', 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )


def test_udf_any_input(con):
    udf_impl = udf.arrow_udf(pa.binary())(some_udf)
    assert udf_impl._name == "some_udf"

    con.register(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql("SELECT some_udf('abcd', 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )


def test_udf_return_type_fn(con):
    udf_impl = udf.arrow_udf(lambda arg_types, arg_scalars: arg_types[0])(some_udf)
    assert udf_impl._name == "some_udf"

    con.register(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql("SELECT some_udf('abcd'::BYTEA, 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"b'abcd' / 123"]}),
    )


def test_udf_array_input(con):
    udf_impl = udf.arrow_udf(pa.binary(), [udf.STRING, udf.NUMERIC])(some_udf)
    assert udf_impl._name == "some_udf"

    con.register(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql(
            "SELECT some_udf(x, 123) as col FROM (VALUES ('a'), ('b'), ('c')) as t(x)"
        ).to_pandas(),
        pd.DataFrame({"col": [b"a / 123", b"b / 123", b"c / 123"]}),
    )


def test_python_udf(con):
    calls = []

    @udf.python_udf(pa.string(), [udf.STRING, udf.NUMERIC])
    def format_value(value, number):
        calls.append((value, number))
        if value is None:
            return None
        return f"{value} / {number}"

    assert format_value._name == "format_value"
    con.register(format_value)
    assert con.sql(
        "SELECT format_value(x, n) AS col "
        "FROM (VALUES ('a', 1), ('b', 2), ('c', 3), (NULL, 4)) AS t(x, n)"
    ).to_arrow_table()["col"].to_pylist() == ["a / 1", "b / 2", "c / 3", None]
    assert set(calls) == {("a", 1), ("b", 2), ("c", 3), (None, 4)}


def test_python_udf_uses_literal_result_conversion(con):
    import geoarrow.pyarrow as ga
    import shapely

    @udf.python_udf(ga.wkb(), [pa.float64(), pa.float64()])
    def make_point(x, y):
        if x is None:
            return None
        return shapely.Point(x, y)

    con.register(make_point)
    assert con.sql(
        "SELECT ST_AsText(make_point(x, y)) AS col "
        "FROM (VALUES (1.0, 2.0), (3.0, 4.0), (NULL, 5.0)) AS t(x, y)"
    ).to_arrow_table()["col"].to_pylist() == ["POINT(1 2)", "POINT(3 4)", None]


@pytest.mark.parametrize("srid", [0, 3857])
def test_python_udf_geometry_identity(con, srid):
    from geoarrow.pyarrow._scalar import WkbScalar

    @udf.python_udf(udf.GEOMETRY, [udf.GEOMETRY])
    def geom_identity(value):
        assert value is None or isinstance(value, WkbScalar)
        return value

    con.register(geom_identity)
    source = (
        f"SELECT ST_Point(x, y, {srid}) AS geom "
        "FROM (VALUES (NULL, NULL), (1.0, 2.0), (3.0, 4.0)) AS t(x, y)"
    )
    expected = con.sql(source).to_arrow_table()
    actual = con.sql(
        f"SELECT geom_identity(geom) AS geom FROM ({source})"
    ).to_arrow_table()
    assert actual.equals(expected)

    assert (
        con.sql(f"SELECT geom_identity(ST_Point(1, 2, {srid})) AS geom")
        .to_arrow_table()
        .equals(con.sql(f"SELECT ST_Point(1, 2, {srid}) AS geom").to_arrow_table())
    )


@pytest.mark.parametrize("geometry", [False, True])
def test_python_udf_input_iterator(geometry):
    import geoarrow.pyarrow as ga

    array = ga.as_wkb([None, "POINT (1 2)"]) if geometry else pa.array([None, 1])
    values = udf._python_values_from_arrow(array, array.type)
    assert iter(values) is values
    assert next(values) is None
    value = next(values)
    assert value == (array[1] if geometry else 1)
    with pytest.raises(StopIteration):
        next(values)


def test_python_udf_scalar_iterator():
    import itertools

    class ScalarInput:
        type = pa.field("", pa.list_(pa.int64()))

        def is_scalar(self):
            return True

        def __arrow_c_array__(self, requested_schema=None):
            return pa.array([[1, 2]]).__arrow_c_array__(requested_schema)

    values = udf._python_udf_values(ScalarInput(), 3)
    assert isinstance(values, itertools.repeat)
    first = next(values)
    assert first == [1, 2]
    assert next(values) is first
    assert next(values) is first
    with pytest.raises(StopIteration):
        next(values)


def test_python_udf_reuses_converted_scalar_object(con):
    values_seen = []

    @udf.python_udf(pa.int64())
    def add_list_length(value, number):
        values_seen.append(value)
        return len(value) + number

    con.register(add_list_length)
    assert con.sql(
        "SELECT add_list_length([10, 20], x) AS col FROM (VALUES (1), (2), (3)) AS t(x)"
    ).to_arrow_table()["col"].to_pylist() == [3, 4, 5]
    assert len(values_seen) == 3
    assert all(value is values_seen[0] for value in values_seen)


def test_python_udf_raster_round_trip(con):
    from sedonadb.raster import Raster

    received = []

    @udf.python_udf(
        udf.RASTER,
        [udf.RASTER],
        metadata={"needs_pixels": "true", "returns_bytes": "true"},
    )
    def raster_identity(value):
        received.append(value)
        assert isinstance(value, Raster)
        assert value.to_numpy().sum() == 0
        return value

    con.register(raster_identity)
    assert con.sql(
        "SELECT RS_Width(raster_identity("
        "RS_MakeEmptyRaster(1, 'B', 3, 2, 0.0, 0.0, 1.0)"
        ")) AS col"
    ).to_arrow_table()["col"].to_pylist() == [3]
    assert len(received) == 1


def test_udf_name():
    udf_impl = udf.arrow_udf(pa.binary(), name="foofy")(some_udf)
    assert udf_impl._name == "foofy"


def test_shapely_udf(con):
    import geoarrow.pyarrow as ga
    import numpy as np
    import shapely

    @udf.arrow_udf(ga.wkb(), [udf.GEOMETRY, udf.NUMERIC])
    def shapely_udf(geom, distance):
        geom_wkb = pa.array(geom.storage.to_array())
        distance = pa.array(distance.to_array())
        geom = shapely.from_wkb(geom_wkb)
        result_shapely = shapely.buffer(geom, distance)
        return pa.array(shapely.to_wkb(result_shapely))

    con.register(shapely_udf)

    pd.testing.assert_frame_equal(
        con.sql("SELECT ST_Area(shapely_udf(ST_Point(0, 0), 2.0)) as col").to_pandas(),
        pd.DataFrame({"col": [12.485780609032208]}),
    )

    # Ensure we can propagate a crs
    pd.testing.assert_frame_equal(
        con.sql(
            "SELECT ST_SRID(shapely_udf(ST_Point(0, 0, 3857), 2.0)) as col"
        ).to_pandas(),
        pd.DataFrame({"col": [3857]}, dtype=np.uint32),
    )

    # Ensure we can collect with >1 batch without hanging
    con.funcs.table.sd_random_geometry("Point", 20000).to_view("pts", overwrite=True)
    df = con.sql(
        "SELECT ST_Area(shapely_udf(ST_Point(0, 0), 2.0)) as col FROM pts"
    ).to_pandas()
    assert len(df) == 20000

    # Ensure we can execute with >1 batch without hanging
    count = con.sql(
        "SELECT ST_Area(shapely_udf(ST_Point(0, 0), 2.0)) as col FROM pts"
    ).execute()
    assert count == 20000


def test_py_sedona_value(con):
    @udf.arrow_udf(pa.int64())
    def fn_arg_only(arg):
        assert repr(arg) == "PySedonaValue Scalar Int64[1]"
        assert arg.is_scalar() is True
        assert repr(arg.type) == "SedonaType int64<Int64>"

        return pa.array(range(len(pa.array(arg))))

    con.register(fn_arg_only)
    con.sql("SELECT fn_arg_only(123)").to_arrow_table()


def test_udf_kwargs(con):
    @udf.arrow_udf(pa.int64())
    def fn_return_type(arg, *, return_type=None):
        assert repr(return_type) == "SedonaType int64<Int64>"
        return pa.array(range(len(pa.array(arg))))

    con.register(fn_return_type)
    con.sql("SELECT fn_return_type('123')").to_arrow_table()

    @udf.arrow_udf(pa.int64())
    def fn_num_rows(arg, *, num_rows=None):
        assert num_rows == 1
        return pa.array(range(len(pa.array(arg))))

    con.register(fn_num_rows)
    con.sql("SELECT fn_num_rows('123')").to_arrow_table()

    @udf.arrow_udf(pa.int64())
    def fn_num_rows_and_return_type(arg, *, num_rows=None, return_type=None):
        assert repr(return_type) == "SedonaType int64<Int64>"
        assert num_rows == 1
        return pa.array(range(len(pa.array(arg))))

    con.register(fn_num_rows_and_return_type)
    con.sql("SELECT fn_num_rows_and_return_type('123')").to_arrow_table()


def test_udf_bad_return_object(con):
    @udf.arrow_udf(pa.binary())
    def questionable_udf(arg):
        return None

    con.register(questionable_udf)
    with pytest.raises(
        sedonadb._lib.SedonaError,
        match="NoneType.*not iterable",
    ):
        con.sql("SELECT questionable_udf(123) as col").to_pandas()


def test_udf_cast_return_type(con):
    @udf.arrow_udf(pa.binary())
    def questionable_udf(arg):
        return pa.array(["abc"], pa.string())

    con.register(questionable_udf)
    result = con.sql("SELECT questionable_udf(123) as col").to_arrow_table()["col"]
    assert result.type == pa.binary()
    assert result.to_pylist() == [b"abc"]


@pytest.mark.parametrize("kwargs", ["neither", "return_type", "num_rows", "both"])
def test_arrow_udf_list_output(con, kwargs):
    def plain(arg):
        return pa.array(arg).to_pylist()

    def with_type(arg, *, return_type):
        assert pa.field(return_type).type == pa.int32()
        return plain(arg)

    def with_rows(arg, *, num_rows):
        assert num_rows == len(pa.array(arg))
        return plain(arg)

    def with_both(arg, *, return_type, num_rows):
        with_rows(arg, num_rows=num_rows)
        return with_type(arg, return_type=return_type)

    func = {
        "neither": plain,
        "return_type": with_type,
        "num_rows": with_rows,
        "both": with_both,
    }[kwargs]
    # Exercise a resolved return type, rather than only a fixed decorator type.
    decorated = udf.arrow_udf(
        lambda arg_types, scalar_args: pa.int32(), name="list_output"
    )(func)
    con.register(decorated)
    result = con.sql(
        "SELECT list_output(x) AS col FROM (VALUES (1), (NULL), (3)) AS t(x)"
    ).to_arrow_table()["col"]
    assert result.type == pa.int32()
    assert result.to_pylist() == [1, None, 3]


def test_arrow_udf_shapely_list_output(con):
    import geoarrow.pyarrow as ga
    import shapely

    @udf.arrow_udf(ga.large_wkb(), [pa.int64()])
    def points(arg):
        return [
            None if x is None else shapely.Point(x, 2)
            for x in pa.array(arg).to_pylist()
        ]

    con.register(points)
    result = con.sql(
        "SELECT ST_X(points(x)) AS col FROM (VALUES (1), (NULL), (3)) AS t(x)"
    ).to_arrow_table()["col"]
    assert result.to_pylist() == [1.0, None, 3.0]


def test_arrow_udf_raster_list_output(con):
    import numpy as np
    from sedonadb.raster import Raster

    raster = Raster.from_numpy(np.zeros((2, 3), dtype=np.uint8))

    @udf.arrow_udf(udf.RASTER, [pa.int64()])
    def rasters(arg):
        return [None if x is None else raster for x in pa.array(arg).to_pylist()]

    con.register(rasters)
    result = con.sql(
        "SELECT RS_Width(rasters(x)) AS col FROM (VALUES (1), (NULL), (3)) AS t(x)"
    ).to_arrow_table()["col"]
    assert result.to_pylist() == [3, None, 3]


def test_udf_bad_return_length(con):
    @udf.arrow_udf(pa.binary())
    def questionable_udf(arg):
        return pa.array([b"abc", b"def"], pa.binary())

    con.register(questionable_udf)
    with pytest.raises(
        sedonadb._lib.SedonaError,
        match="Expected result of user-defined function to return array of length 1 but got 2",
    ):
        con.sql("SELECT questionable_udf(123) as col").to_pandas()


def test_native_scalar_udf_export_import_roundtrip(con):
    # A SedonaDB built-in scalar function exposes its native overload kernels
    # via __sedonadb_scalar_udf__ (the export side of the plugin protocol).
    # Exporting those capsules and rebuilding a UDF under a new name must
    # produce a function whose output is identical to the built-in's, proving
    # the native kernels survive a full export -> capsule -> import roundtrip.
    from sedonadb.udf import sedona_native_scalar_udf

    st_asbinary = con.funcs.st_asbinary
    assert hasattr(st_asbinary, "__sedonadb_scalar_udf__")

    capsules = st_asbinary.__sedonadb_scalar_udf__()
    assert len(capsules) >= 1

    # Rebuild under a fresh name so it doesn't collide with the built-in, then
    # register it back into the same context.
    rebuilt = sedona_native_scalar_udf(capsules, name="rt_asbinary_roundtrip")
    con.register(rebuilt)

    # Run both the built-in and the reimported UDF over a real one-row table
    # column (not a literal) so the full execution path runs.
    con.sql("SELECT ST_Point(30.0, 10.0) AS geom").to_view(
        "rt_native_roundtrip", overwrite=True
    )
    expected = con.sql(
        "SELECT ST_AsBinary(geom) AS col FROM rt_native_roundtrip"
    ).to_arrow_table()
    actual = con.sql(
        "SELECT rt_asbinary_roundtrip(geom) AS col FROM rt_native_roundtrip"
    ).to_arrow_table()

    assert actual.equals(expected)


def test_native_scalar_udf_export_preserves_sedona_options(con):
    if "s2geography" not in sedonadb.__features__:
        pytest.skip("Python package built without s2geography")

    capsules = con.funcs.st_envelope.__sedonadb_scalar_udf__()
    con.register(udf.sedona_native_scalar_udf(capsules, name="rt_envelope"))

    expected = con.sql(
        "SELECT ST_Envelope(ST_GeogFromText('POINT (1 2)')) AS col"
    ).to_arrow_table()
    actual = con.sql(
        "SELECT rt_envelope(ST_GeogFromText('POINT (1 2)')) AS col"
    ).to_arrow_table()

    assert actual.equals(expected)


def test_native_scalar_udf_register_appends_overload(con):
    # Registering a native scalar UDF under a name already in use appends its
    # kernels as overloads rather than replacing the function: both the
    # previously registered signature and the newly registered one stay
    # dispatchable. Two built-ins with disjoint input types are exported and
    # re-registered under one shared name -- ST_AsBinary takes a geometry,
    # ST_GeomFromWKT takes a WKT string. If registration replaced rather than
    # appended, the geometry overload registered first would vanish and calling
    # it would raise "No kernel matching arguments".
    from sedonadb.udf import sedona_native_scalar_udf

    name = "rt_overload_probe"

    con.register(
        sedona_native_scalar_udf(
            con.funcs.st_asbinary.__sedonadb_scalar_udf__(), name=name
        )
    )
    con.register(
        sedona_native_scalar_udf(
            con.funcs.st_geomfromwkt.__sedonadb_scalar_udf__(), name=name
        )
    )

    # The geometry -> binary overload registered first is still reachable...
    assert (
        con.sql(f"SELECT {name}(ST_Point(30.0, 10.0)) AS col")
        .to_arrow_table()
        .equals(
            con.sql("SELECT ST_AsBinary(ST_Point(30.0, 10.0)) AS col").to_arrow_table()
        )
    )

    # ...and the WKT-string -> geometry overload appended second dispatches too.
    pd.testing.assert_frame_equal(
        con.sql(f"SELECT ST_AsText({name}('POINT (1 2)')) AS col").to_pandas(),
        pd.DataFrame({"col": ["POINT(1 2)"]}),
    )


def test_raster_udf_over_loaded_raster_as_pixel_argument(con, tmp_path):
    """A Python raster UDF over RS_EnsureLoaded(..), passed as the raster of
    functions that read pixels. The planner wraps the UDF's output in another
    RS_EnsureLoaded, which passes the already loaded pixels through. Two calls
    over the same expression exercise the common-subexpression path."""
    pytest.importorskip("rasterio")
    np = pytest.importorskip("numpy")
    from sedonadb.raster import Raster
    from sedonadb.raster_testing import write_geotiff

    path = tmp_path / "grid.tif"
    pixels = np.arange(1, 7, dtype="float64").reshape(1, 2, 3)
    write_geotiff(path, pixels, bbox=(0.0, 0.0, 3.0, 2.0))

    raster_type = con.sql("SELECT RS_Example() AS r").to_arrow_table()["r"].type

    @udf.arrow_udf(raster_type)
    def double_pixels(rast):
        out = []
        for item in pa.array(rast.to_array()):
            r = item.as_py()
            doubled = Raster.from_numpy(r.to_numpy()[0] * 2, transform=r.transform)
            out.append(pa.array(doubled))
        return pa.chunked_array(out, type=raster_type).combine_chunks()

    con.register(double_pixels)

    doubled = "double_pixels(RS_EnsureLoaded(RS_FromPath($1)))"
    got = con.sql(
        f"SELECT RS_SummaryStats({doubled}, 'mean', 1) AS mean, "
        f"RS_Value({doubled}, 2, 1, 1) AS v",
        params=(str(path),),
    ).to_arrow_table()
    assert got["mean"].to_pylist() == [7.0]
    assert got["v"].to_pylist() == [4.0]
