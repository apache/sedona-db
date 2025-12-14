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

    con.register_udf(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql("SELECT some_udf('abcd', 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )


def test_udf_types(con):
    udf_impl = udf.arrow_udf(pa.binary(), [pa.string(), pa.int64()])(some_udf)
    assert udf_impl._name == "some_udf"

    con.register_udf(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql("SELECT some_udf('abcd', 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )


def test_udf_any_input(con):
    udf_impl = udf.arrow_udf(pa.binary())(some_udf)
    assert udf_impl._name == "some_udf"

    con.register_udf(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql("SELECT some_udf('abcd', 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )


def test_udf_return_type_fn(con):
    udf_impl = udf.arrow_udf(lambda arg_types, arg_scalars: arg_types[0])(some_udf)
    assert udf_impl._name == "some_udf"

    con.register_udf(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql("SELECT some_udf('abcd'::BYTEA, 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"b'abcd' / 123"]}),
    )


def test_udf_array_input(con):
    udf_impl = udf.arrow_udf(pa.binary(), [udf.STRING, udf.NUMERIC])(some_udf)
    assert udf_impl._name == "some_udf"

    con.register_udf(udf_impl)
    pd.testing.assert_frame_equal(
        con.sql(
            "SELECT some_udf(x, 123) as col FROM (VALUES ('a'), ('b'), ('c')) as t(x)"
        ).to_pandas(),
        pd.DataFrame({"col": [b"a / 123", b"b / 123", b"c / 123"]}),
    )


def test_udf_name():
    udf_impl = udf.arrow_udf(pa.binary(), name="foofy")(some_udf)
    assert udf_impl._name == "foofy"


def test_shapely_udf(con):
    import shapely
    import geoarrow.pyarrow as ga
    import numpy as np

    @udf.arrow_udf(ga.wkb(), [udf.GEOMETRY, udf.NUMERIC])
    def shapely_udf(geom, distance):
        geom_wkb = pa.array(geom.storage.to_array())
        distance = pa.array(distance.to_array())
        geom = shapely.from_wkb(geom_wkb)
        result_shapely = shapely.buffer(geom, distance)
        return pa.array(shapely.to_wkb(result_shapely))

    con.register_udf(shapely_udf)

    pd.testing.assert_frame_equal(
        con.sql("SELECT ST_Area(shapely_udf(ST_Point(0, 0), 2.0)) as col").to_pandas(),
        pd.DataFrame({"col": [12.485780609032208]}),
    )

    # Ensure we can propagate a crs
    pd.testing.assert_frame_equal(
        con.sql(
            "SELECT ST_SRID(shapely_udf(ST_SetSRID(ST_Point(0, 0), 3857), 2.0)) as col"
        ).to_pandas(),
        pd.DataFrame({"col": [3857]}, dtype=np.uint32),
    )


def test_py_sedona_value(con):
    @udf.arrow_udf(pa.int64())
    def fn_arg_only(arg):
        assert repr(arg) == "PySedonaValue Scalar Int64[1]"
        assert arg.is_scalar() is True
        assert repr(arg.type) == "SedonaType int64<Int64>"

        return pa.array(range(len(pa.array(arg))))

    con.register_udf(fn_arg_only)
    con.sql("SELECT fn_arg_only(123)").to_arrow_table()


def test_udf_kwargs(con):
    @udf.arrow_udf(pa.int64())
    def fn_return_type(arg, *, return_type=None):
        assert repr(return_type) == "SedonaType int64<Int64>"
        return pa.array(range(len(pa.array(arg))))

    con.register_udf(fn_return_type)
    con.sql("SELECT fn_return_type('123')").to_arrow_table()

    @udf.arrow_udf(pa.int64())
    def fn_num_rows(arg, *, num_rows=None):
        assert num_rows == 1
        return pa.array(range(len(pa.array(arg))))

    con.register_udf(fn_num_rows)
    con.sql("SELECT fn_num_rows('123')").to_arrow_table()

    @udf.arrow_udf(pa.int64())
    def fn_num_rows_and_return_type(arg, *, num_rows=None, return_type=None):
        assert repr(return_type) == "SedonaType int64<Int64>"
        assert num_rows == 1
        return pa.array(range(len(pa.array(arg))))

    con.register_udf(fn_num_rows_and_return_type)
    con.sql("SELECT fn_num_rows_and_return_type('123')").to_arrow_table()


def test_udf_bad_return_object(con):
    @udf.arrow_udf(pa.binary())
    def questionable_udf(arg):
        return None

    con.register_udf(questionable_udf)
    with pytest.raises(
        ValueError,
        match="Expected result of user-defined function to return an object implementing __arrow_c_array__",
    ):
        con.sql("SELECT questionable_udf(123) as col").to_pandas()


def test_udf_bad_return_type(con):
    @udf.arrow_udf(pa.binary())
    def questionable_udf(arg):
        return pa.array(["abc"], pa.string())

    con.register_udf(questionable_udf)
    with pytest.raises(
        ValueError,
        match=(
            "Expected result of user-defined function to "
            "return array of type Binary or its storage "
            "but got Utf8"
        ),
    ):
        con.sql("SELECT questionable_udf(123) as col").to_pandas()


def test_udf_bad_return_length(con):
    @udf.arrow_udf(pa.binary())
    def questionable_udf(arg):
        return pa.array([b"abc", b"def"], pa.binary())

    con.register_udf(questionable_udf)
    with pytest.raises(
        ValueError,
        match="Expected result of user-defined function to return array of length 1 but got 2",
    ):
        con.sql("SELECT questionable_udf(123) as col").to_pandas()


def test_udf_datafusion_to_sedonadb(con):
    udf_impl = udf.arrow_udf(
        pa.binary(), [udf.STRING, udf.NUMERIC], name="some_external_udf"
    )(some_udf)

    class UdfWrapper:
        def __init__(self, obj):
            self.obj = obj

        def __datafusion_scalar_udf__(self):
            return self.obj.__datafusion_scalar_udf__()

    con.register_udf(UdfWrapper(udf_impl))
    pd.testing.assert_frame_equal(
        con.sql("SELECT some_external_udf('abcd', 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )


def test_udf_sedonadb_registry_function_to_datafusion(con):
    datafusion = pytest.importorskip("datafusion")
    udf_impl = udf.arrow_udf(pa.binary(), [udf.STRING, udf.NUMERIC])(some_udf)

    # Register with our session
    con.register_udf(udf_impl)

    # Create a datafusion session, fetch our udf and register with the other session
    datafusion_ctx = datafusion.SessionContext()
    datafusion_ctx.register_udf(
        datafusion.ScalarUDF.from_pycapsule(con._impl.scalar_udf("some_udf"))
    )

    # Can't quite use to_pandas() because there is a schema/batch nullability mismatch
    batches = datafusion_ctx.sql("SELECT some_udf('abcd', 123) as col").collect()
    assert len(batches) == 1
    pd.testing.assert_frame_equal(
        batches[0].to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )


def test_udf_sedonadb_to_datafusion():
    datafusion = pytest.importorskip("datafusion")
    udf_impl = udf.arrow_udf(pa.binary(), [udf.STRING, udf.NUMERIC])(some_udf)

    # Create a datafusion session, register udf_impl directly
    datafusion_ctx = datafusion.SessionContext()
    datafusion_ctx.register_udf(datafusion.ScalarUDF.from_pycapsule(udf_impl))

    # Can't quite use to_pandas() because there is a schema/batch nullability mismatch
    batches = datafusion_ctx.sql("SELECT some_udf('abcd', 123) as col").collect()
    assert len(batches) == 1
    pd.testing.assert_frame_equal(
        batches[0].to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )
