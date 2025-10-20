import pyarrow as pa
from sedonadb import udf
import pandas as pd


def test_basic_udf(con):
    @udf.arrow_udf(pa.binary(), ["string", "numeric"])
    def some_udf(arg0, arg1):
        arg0, arg1 = (
            pa.array(arg0.to_array()).to_pylist(),
            pa.array(arg1.to_array()).to_pylist(),
        )
        return pa.array(
            (f"{item0} / {item1}".encode() for item0, item1 in zip(arg0, arg1)),
            pa.binary(),
        )

    assert some_udf._name == "some_udf"

    con.register_udf(some_udf)

    pd.testing.assert_frame_equal(
        con.sql("SELECT some_udf('abcd', 123) as col").to_pandas(),
        pd.DataFrame({"col": [b"abcd / 123"]}),
    )
