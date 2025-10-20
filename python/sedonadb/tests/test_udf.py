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
        match="Expected result of user-defined function to return array of type Binary but got Utf8",
    ):
        con.sql("SELECT questionable_udf(123) as col").to_pandas()


def test_udf_bad_return_length(con):
    @udf.arrow_udf(pa.binary())
    def questionable_udf(arg):
        return pa.array([b"abc", b"def"], pa.binary())

    con.register_udf(questionable_udf)
    with pytest.raises(
        ValueError,
        match="UDF questionable_udf returned a different number of rows than expected. Expected: 1, Got: 2.",
    ):
        con.sql("SELECT questionable_udf(123) as col").to_pandas()
