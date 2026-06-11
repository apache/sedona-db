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

import inspect
import re
from typing import Any, List, Literal, Optional, Union

from sedonadb._lib import sedona_aggregate_udf, sedona_scalar_udf
from sedonadb.utility import sedona  # noqa: F401


class TypeMatcher(str):
    """Helper class to mark type matchers that can be used as the `input_types` for
    user-defined functions

    Note that the internal storage of the type matcher (currently a string) is
    arbitrary and may change in a future release. Use the constants provided by
    the `udf` module.
    """

    pass


def arrow_udf(
    return_type: Any,
    input_types: List[Union[TypeMatcher, Any]] = None,
    volatility: Literal["immutable", "stable", "volatile"] = "immutable",
    name: Optional[str] = None,
):
    """Generic Arrow-based user-defined scalar function decorator

    This decorator may be used to annotate a function that accepts arguments as
    Arrow array wrappers implementing the
    [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html).
    The annotated function must return a value of a consistent length of the
    appropriate type.

    !!! warning
        SedonaDB will call the provided function from multiple threads. Attempts
        to modify shared state from the body of the function may crash or cause
        unusual behaviour.

    SedonaDB Python UDFs are experimental and this interface may change based on
    user feedback.

    Args:
        return_type: One of
            - A data type (e.g., pyarrow.DataType, arro3.core.DataType, nanoarrow.Schema)
              if this function returns the same type regardless of its inputs.
            - A function of `arg_types` (list of data types) and `scalar_args` (list of
              optional scalars) that returns a data type. This function is also
              responsible for returning `None` if this function does not apply to the
              input types.
        input_types: One of
            - A list where each member is a data type or a `TypeMatcher`. The
              `udf.GEOMETRY` and `udf.GEOGRAPHY` type matchers are the most useful
              because otherwise the function will only match spatial data types whose
              coordinate reference system (CRS) also matches (i.e., based on simple
              equality). Using these type matchers will also ensure input CRS consistency
              and will automatically propagate input CRSes into the output.
            - `None`, indicating that this function can accept any number of arguments
              of any type. Usually this is paired with a functional `return_type` that
              dynamically computes a return type or returns `None` if the number or
              types of arguments do not match.
        volatility: Use "immutable" for functions whose output is always consistent
            for the same inputs (even between queries); use "stable" for functions
            whose output is always consistent for the same inputs but only within
            the same query, and use "volatile" for functions that generate random
            or otherwise non-deterministic output.
        name: An optional name for the UDF. If not given, it will be derived from
            the name of the provided function.

    Examples:

        >>> import pyarrow as pa
        >>> from sedonadb import udf
        >>> sd = sedona.db.connect()

        The simplest scalar UDF only specifies return types. This implies that
        the function can handle input of any type.

        >>> @udf.arrow_udf(pa.string())
        ... def some_udf(arg0, arg1):
        ...     arg0, arg1 = (
        ...         pa.array(arg0.to_array()).to_pylist(),
        ...         pa.array(arg1.to_array()).to_pylist(),
        ...     )
        ...     return pa.array(
        ...         (f"{item0} / {item1}" for item0, item1 in zip(arg0, arg1)),
        ...         pa.string(),
        ...     )
        ...
        >>> sd.register(some_udf)
        >>> sd.sql("SELECT some_udf(123, 'abc') as col").show()
        ┌───────────┐
        │    col    │
        │    utf8   │
        ╞═══════════╡
        │ 123 / abc │
        └───────────┘

        Use the `TypeMatcher` constants where possible to specify input.
        This ensures that the function can handle the usual range of input
        types that might exist for a given input.

        >>> @udf.arrow_udf(pa.int64(), [udf.STRING])
        ... def char_count(arg0):
        ...     arg0 = pa.array(arg0.to_array())
        ...
        ...     return pa.array(
        ...         (len(item) for item in arg0.to_pylist()),
        ...         pa.int64()
        ...     )
        ...
        >>> sd.register(char_count)
        >>> sd.sql("SELECT char_count('abcde') as col").show()
        ┌───────┐
        │  col  │
        │ int64 │
        ╞═══════╡
        │     5 │
        └───────┘

        In this case, the type matcher ensures we can also use the function
        for string view input which is the usual type SedonaDB emits when
        reading Parquet files.

        >>> sd.sql("SELECT char_count(arrow_cast('abcde', 'Utf8View')) as col").show()
        ┌───────┐
        │  col  │
        │ int64 │
        ╞═══════╡
        │     5 │
        └───────┘

        Geometry UDFs are best written using Shapely because pyproj (including its use
        in GeoPandas) is not thread safe and can crash when attempting to look up
        CRSes when importing an Arrow array. The UDF framework supports returning
        geometry storage to make this possible. Coordinate reference system metadata
        is propagated automatically from the input.

        >>> import shapely
        >>> import geoarrow.pyarrow as ga
        >>> @udf.arrow_udf(ga.wkb(), [udf.GEOMETRY, udf.NUMERIC])
        ... def shapely_udf(geom, distance):
        ...     geom_wkb = pa.array(geom.storage.to_array())
        ...     distance = pa.array(distance.to_array())
        ...     geom = shapely.from_wkb(geom_wkb)
        ...     result_shapely = shapely.buffer(geom, distance)
        ...     return pa.array(shapely.to_wkb(result_shapely))
        ...
        >>>
        >>> sd.register(shapely_udf)
        >>> sd.sql("SELECT ST_SRID(shapely_udf(ST_Point(0, 0), 2.0)) as col").show()
        ┌────────┐
        │   col  │
        │ uint32 │
        ╞════════╡
        │      0 │
        └────────┘

        >>> sd.sql("SELECT ST_SRID(shapely_udf(ST_Point(0, 0, 3857), 2.0)) as col").show()
        ┌────────┐
        │   col  │
        │ uint32 │
        ╞════════╡
        │   3857 │
        └────────┘

        Annotated functions may also declare keyword arguments `return_type` and/or `num_rows`,
        which will be passed the appropriate value by the UDF framework. This facilitates writing
        generic UDFs and/or UDFs with no arguments.

        >>> import numpy as np
        >>> def random_impl(return_type, num_rows):
        ...     pa_type = pa.field(return_type).type
        ...     return pa.array(np.random.random(num_rows), pa_type)
        ...
        >>> @udf.arrow_udf(pa.float32(), [])
        ... def random_f32(*, return_type=None, num_rows=None):
        ...     return random_impl(return_type, num_rows)
        ...
        >>> @udf.arrow_udf(pa.float64(), [])
        ... def random_f64(*, return_type=None, num_rows=None):
        ...     return random_impl(return_type, num_rows)
        ...
        >>> np.random.seed(487)
        >>> sd.register(random_f32)
        >>> sd.register(random_f64)
        >>> sd.sql("SELECT random_f32() AS f32, random_f64() as f64;").show()
        ┌────────────┬─────────────────────┐
        │     f32    ┆         f64         │
        │   float32  ┆       float64       │
        ╞════════════╪═════════════════════╡
        │ 0.35385555 ┆ 0.24793247139474195 │
        └────────────┴─────────────────────┘

    """

    def decorator(func):
        kwarg_names = _callable_kwarg_only_names(func)
        if "return_type" in kwarg_names and "num_rows" in kwarg_names:

            def func_wrapper(args, return_type, num_rows):
                return func(*args, return_type=return_type, num_rows=num_rows)
        elif "return_type" in kwarg_names:

            def func_wrapper(args, return_type, num_rows):
                return func(*args, return_type=return_type)
        elif "num_rows" in kwarg_names:

            def func_wrapper(args, return_type, num_rows):
                return func(*args, num_rows=num_rows)
        else:

            def func_wrapper(args, return_type, num_rows):
                return func(*args)

        name_arg = func.__name__ if name is None and hasattr(func, "__name__") else name
        return ScalarUdfImpl(
            func_wrapper, return_type, input_types, volatility, name_arg
        )

    return decorator


BINARY: TypeMatcher = "binary"
"""Match any binary argument (i.e., binary, binary view, large binary,
fixed-size binary)"""

BOOLEAN: TypeMatcher = "boolean"
"""Match a boolean argument"""

GEOGRAPHY: TypeMatcher = "geography"
"""Match a geography argument"""

GEOMETRY: TypeMatcher = "geometry"
"""Match a geometry argument"""

NUMERIC: TypeMatcher = "numeric"
"""Match any numeric argument"""

STRING: TypeMatcher = "string"
"""Match any string argument (i.e., string, string view, large string)"""


class ScalarUdfImpl:
    """Scalar user-defined function wrapper

    This class is a wrapper class used as the return value for user-defined
    function constructors. This wrapper allows the UDF to be registered with
    a SedonaDB context or any context that accepts DataFusion Python
    Scalar UDFs. This object is not intended to be used to call a UDF.
    """

    def __init__(
        self,
        invoke_batch,
        return_type,
        input_types=None,
        volatility: Literal["immutable", "stable", "volatile"] = "immutable",
        name: Optional[str] = None,
    ):
        # If the input_types are None, the return_type must be callable when passed
        # to the internals. In the Python API we allow a data type as the return type
        # to the argument easier to understand, which means we may have to wrap
        # it in a callable here.
        if input_types is None and not callable(return_type):

            def return_type_impl(*args, **kwargs):
                return return_type

            self._return_type = return_type_impl
        else:
            self._return_type = return_type

        self._invoke_batch = invoke_batch
        self._input_types = input_types
        if name is None and hasattr(invoke_batch, "__name__"):
            self._name = invoke_batch.__name__
        else:
            self._name = name

        self._volatility = volatility

    def __sedonadb_internal_udf__(self):
        return sedona_scalar_udf(
            self._invoke_batch,
            self._return_type,
            self._input_types,
            self._volatility,
            self._name,
        )


def _callable_kwarg_only_names(f):
    sig = inspect.signature(f)
    return [
        k for k, p in sig.parameters.items() if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]


def _camel_to_snake(name: str) -> str:
    """Convert a `CamelCase` identifier to `snake_case`.

    `MyMean` -> `my_mean`, `STMean` -> `st_mean`, `my_mean` -> `my_mean`.
    """
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def arrow_aggregate_udf(
    return_type: Any,
    input_types: List[Union[TypeMatcher, Any]],
    state_types: List[Any],
    *,
    volatility: Literal["immutable", "stable", "volatile"] = "immutable",
    name: Optional[str] = None,
):
    """Decorator for Python-implemented aggregate UDFs.

    Decorates a class whose instances act as a stateful accumulator. Each
    grouped/global aggregation builds one instance per partial-state slot
    and then merges them.

    !!! warning
        SedonaDB Python UDFs are experimental and this interface may change
        based on user feedback.

    The decorated class must define:

    - `__init__(self)`: build a fresh accumulator (no arguments).
    - `update(self, *arrays)`: receives one `pa.Array` per declared input
      column, splatted positionally (single input today; the array can
      contain nulls).
    - `state(self) -> tuple`: serialize the accumulator into a tuple of
      Python values matching `state_types` in order.
    - `merge(self, *arrays)`: receives one `pa.Array` per element of
      `state_types`, splatted positionally. Each array has N rows for N
      partial states being merged in.
    - `evaluate(self)`: return the final scalar Python value (or `None`
      for SQL NULL) matching `return_type`.

    Args:
        return_type: A pyarrow data type for the final aggregate result.
            Must be a concrete pyarrow type — `TypeMatcher` constants are
            not accepted here.
        input_types: One or more types describing the columns the
            aggregate consumes. Each entry is either a `TypeMatcher`
            constant (`udf.NUMERIC`, `udf.GEOMETRY`, …) or a concrete
            pyarrow type.
        state_types: A list of concrete pyarrow types describing the
            serialized state. The length must match the tuple returned by
            `state()`.
        volatility: `"immutable"` (default), `"stable"`, or `"volatile"`.
        name: SQL-visible name. Defaults to the decorated class name
            converted from `CamelCase` to `snake_case` (so a `MyMean`
            class is callable as `my_mean`).

    Examples:

        >>> import pyarrow as pa
        >>> import pandas as pd
        >>> import sedonadb
        >>> from sedonadb import udf
        >>> sd = sedonadb.connect()
        >>>
        >>> @udf.arrow_aggregate_udf(
        ...     return_type=pa.float64(),
        ...     input_types=[udf.NUMERIC],
        ...     state_types=[pa.float64(), pa.int64()],
        ... )
        ... class my_mean:
        ...     def __init__(self):
        ...         self.total = 0.0
        ...         self.count = 0
        ...     def update(self, batch):
        ...         for v in batch:
        ...             if v.is_valid:
        ...                 self.total += float(v.as_py())
        ...                 self.count += 1
        ...     def state(self):
        ...         return (self.total, self.count)
        ...     def merge(self, totals, counts):
        ...         for i in range(len(totals)):
        ...             self.total += totals[i].as_py()
        ...             self.count += counts[i].as_py()
        ...     def evaluate(self):
        ...         return None if self.count == 0 else self.total / self.count
        ...
        >>> sd.register(my_mean)
        >>> sd.create_data_frame(
        ...     pd.DataFrame({"k": ["a", "a", "b"], "v": [1.0, 3.0, 7.0]})
        ... ).to_view("t", overwrite=True)
        >>> sd.sql("SELECT k, my_mean(v) AS m FROM t GROUP BY k ORDER BY k").show()
        ┌──────┬─────────┐
        │   k  ┆    m    │
        │ utf8 ┆ float64 │
        ╞══════╪═════════╡
        │ a    ┆     2.0 │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
        │ b    ┆     7.0 │
        └──────┴─────────┘
    """

    def decorator(cls):
        return AggregateUdfImpl(
            cls, return_type, input_types, state_types, volatility, name
        )

    return decorator


class AggregateUdfImpl:
    """Aggregate user-defined function wrapper.

    Returned by [`arrow_aggregate_udf`][sedonadb.udf.arrow_aggregate_udf].
    Holds the user's class plus the type schemas; on registration, builds
    a factory that produces a per-accumulator `_AccumulatorWrapper`
    bridging the user's class to the Rust `Accumulator` trait.
    """

    def __init__(
        self,
        user_cls,
        return_type,
        input_types,
        state_types,
        volatility: Literal["immutable", "stable", "volatile"] = "immutable",
        name: Optional[str] = None,
    ):
        self._user_cls = user_cls
        self._return_type = return_type
        self._input_types = input_types
        self._state_types = state_types
        self._volatility = volatility
        if name is None and hasattr(user_cls, "__name__"):
            # Linters push users toward CamelCase class names, but function
            # lookups (con.funcs.<name> and SQL) resolve against a lowercased
            # name. Derive a snake_case default so `MyMean` is callable as
            # `my_mean`.
            self._name = _camel_to_snake(user_cls.__name__)
        else:
            self._name = name

    def __sedonadb_internal_aggregate_udf__(self):
        # The factory closure is invoked by the Rust kernel once per
        # accumulator instance; it captures `self` (which only holds the
        # user class and small type lists).
        def factory():
            return _AccumulatorWrapper(
                self._user_cls(), self._return_type, self._state_types
            )

        return sedona_aggregate_udf(
            factory,
            self._return_type,
            self._input_types,
            self._state_types,
            self._volatility,
            self._name,
        )


class _AccumulatorWrapper:
    """Bridges a user accumulator class to the Rust `Accumulator` calls.

    Translates each Rust → Python crossing:

    - `update(args)` / `merge(args)`: `args` is a tuple of zero-copy
      Arrow-array handles (PySedonaValue); we convert each to a real
      `pa.Array` and splat them positionally into the user's method.
    - `evaluate()`: wraps the user's scalar return value as a 1-element
      `pa.Array` of `return_type` so the Rust side can extract a
      `ScalarValue` via the C-data interface.
    - `state()`: wraps each scalar in the user's tuple as a 1-element
      `pa.Array` of the matching `state_types[i]`.
    """

    def __init__(self, user_instance, return_type, state_types):
        self._user = user_instance
        self._return_type = return_type
        self._state_types = state_types

    @staticmethod
    def _to_arrays(args):
        import pyarrow as pa

        # `.to_array()` forces the Array variant before the C-data import,
        # matching the conversion the scalar-UDF examples use.
        return [pa.array(a.to_array()) for a in args]

    @staticmethod
    def _wrap_scalar(value, pa_type):
        import pyarrow as pa

        return pa.array([value], type=pa_type)

    def update(self, args):
        self._user.update(*self._to_arrays(args))

    def merge(self, args):
        self._user.merge(*self._to_arrays(args))

    def evaluate(self):
        return self._wrap_scalar(self._user.evaluate(), self._return_type)

    def state(self):
        values = self._user.state()
        if not isinstance(values, (tuple, list)):
            values = (values,)
        if len(values) != len(self._state_types):
            raise ValueError(
                f"Aggregate UDF state() returned {len(values)} elements, "
                f"expected {len(self._state_types)} per state_types"
            )
        return tuple(self._wrap_scalar(v, t) for v, t in zip(values, self._state_types))
