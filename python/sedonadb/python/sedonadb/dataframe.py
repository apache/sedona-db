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

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Union

from sedonadb.expr import Expr, SortExpr
from sedonadb.expr import Literal as _SedonaLit
from sedonadb.expr import col as _col
from sedonadb.expr.expression import collect_exprs
from sedonadb.utility import sedona  # noqa: F401

if TYPE_CHECKING:
    import geopandas
    import pandas
    import pyarrow as pa


class DataFrame:
    """Representation of a (lazy) collection of columns

    This object is usually constructed from `sd = sedona.db.connect()`
    by importing an object with `sd.create_data_frame()`, reading a file
    with `sd.read_parquet()`/`sd.read_pyogrio()`, or executing SQL with
    `sd.sql()`. Once created, a DataFrame can be modified using the Python
    API (e.g., `.select()`, `.filter()`, `.sort()`, `.limit()`) or by
    creating a temporary view with `.to_view("name")` and querying the
    resulting view using `sd.sql()`. The Python API aims to provide
    a minimal subset of functionality derived primarily from Ibis and
    DuckDB's relational APIs.

    Examples:

        >>> sd = sedona.db.connect()
        >>> sd.options.interactive = True
        >>> df = sd.sql("SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t(x, y)")
        >>> df.limit(2)
        ┌───────┬──────┐
        │   x   ┆   y  │
        │ int64 ┆ utf8 │
        ╞═══════╪══════╡
        │     1 ┆ a    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │     2 ┆ b    │
        └───────┴──────┘

        Columns can be specified as part of the Python API in several ways:

        >>> df.select("y", z=df.x + 1)
        ┌──────┬───────┐
        │   y  ┆   z   │
        │ utf8 ┆ int64 │
        ╞══════╪═══════╡
        │ a    ┆     2 │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ b    ┆     3 │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ c    ┆     4 │
        └──────┴───────┘
        >>> df.select("y", z=df["x"] + 1)
        ┌──────┬───────┐
        │   y  ┆   z   │
        │ utf8 ┆ int64 │
        ╞══════╪═══════╡
        │ a    ┆     2 │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ b    ┆     3 │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ c    ┆     4 │
        └──────┴───────┘
        >>> df.select("y", z=sd.col("x") + 1)
        ┌──────┬───────┐
        │   y  ┆   z   │
        │ utf8 ┆ int64 │
        ╞══════╪═══════╡
        │ a    ┆     2 │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ b    ┆     3 │
        ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ c    ┆     4 │
        └──────┴───────┘

        Literals can be specified explicitly using `sd.lit(obj)` but can also
        be used directly in function calls or comparisons.

        >>> df.filter(df.x > 1)
        ┌───────┬──────┐
        │   x   ┆   y  │
        │ int64 ┆ utf8 │
        ╞═══════╪══════╡
        │     2 ┆ b    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │     3 ┆ c    │
        └───────┴──────┘
        >>> df.filter(df.x > sd.lit(1))
        ┌───────┬──────┐
        │   x   ┆   y  │
        │ int64 ┆ utf8 │
        ╞═══════╪══════╡
        │     2 ┆ b    │
        ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
        │     3 ┆ c    │
        └───────┴──────┘

        Supported literals include any Python object supported by pyarrow
        in addition to GeoPandas, Shapely, and PyProj objects for use in
        geometry functions.

        Functions are available from `sd.funcs`. Assigning this to a local
        variable generally leads to better autocomplete but is not necessary.
        In addition to standard DataFusion scalar and aggregate functions,
        a number of spatial functions are provided. See the
        [SedonaDB SQL Reference](https://sedona.apache.org/sedonadb/latest/reference/sql/)
        for a list of supported functions.

        >>> f = sd.funcs
        >>> df.select(geometry=f.st_point(df.x, df.x + 1))
        ┌────────────┐
        │  geometry  │
        │  geometry  │
        ╞════════════╡
        │ POINT(1 2) │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ POINT(2 3) │
        ├╌╌╌╌╌╌╌╌╌╌╌╌┤
        │ POINT(3 4) │
        └────────────┘
    """

    def __init__(self, ctx, impl):
        self._ctx = ctx
        self._impl = impl

    @property
    def schema(self):
        """Return the column names and data types

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT 1 as one")
            >>> df.schema
            SedonaSchema with 1 field:
              one: non-nullable int64<Int64>
            >>> df.schema.field(0)
            SedonaField one: non-nullable int64<Int64>
            >>> df.schema.field(0).name, df.schema.field(0).type
            ('one', SedonaType int64<Int64>)
        """
        return self._impl.schema()

    @property
    def columns(self) -> list[str]:
        """Return a list of column names"""
        return self._impl.columns()

    def head(self, n: int = 5) -> "DataFrame":
        """Limit result to the first n rows

        Note that this is non-deterministic for many queries.

        Args:
            n: The number of rows to return

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
            >>> df.head(1).show()
            ┌──────┐
            │  val │
            │ utf8 │
            ╞══════╡
            │ one  │
            └──────┘
        """
        return self.limit(n)

    def alias(self, name: str) -> "DataFrame":
        """Qualify all columns of this DataFrame with a given name

        Returns a DataFrame where all columns are qualified to disambiguate
        references in join expressions. This is the equivalent of aliasing a subquery
        in SQL (`(SELECT * FROM df) AS name`).
        """
        return DataFrame(self._ctx, self._impl.alias(name))

    def __getitem__(self, key: Union[str, int]) -> Expr:
        """Reference a single column by name or position.

        Returns an `Expr` referencing the requested column. The
        return-type is always `Expr` (no `DataFrame ⏐ Expr` union) so that
        IDEs and type-aware tools can resolve `df["x"].<method>` cleanly.

        For row filtering use `df.filter(predicate)`. For multi-column
        projection use `df.select(*cols)`.

        Args:
            key: A column name (`str`) or a 0-based column position
                (`int`, negative indices count from the end).

        Raises:
            KeyError: A string key that does not match any column. The
                error message lists the available columns.
            IndexError: An integer index outside the column range.
            TypeError: Any other key type, including `bool`, `slice`,
                `list`, and `Expr`. The message points at `select` or
                `filter` for those use cases.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES (1, 10), (2, 20)) AS t(x, y)").alias("t")
            >>> df["x"]
            Expr(t.x)
            >>> df[1]
            Expr(t.y)
            >>> df[-1]
            Expr(t.y)
        """
        # `bool` is a subclass of `int`, so guard explicitly — otherwise
        # `df[True]` would silently mean `df[1]`.
        if isinstance(key, bool):
            raise TypeError(
                "DataFrame indexing is not supported for bool. "
                "Use df['name'] or df[i] for a column expression; "
                "use df.select(*cols) for multi-column projection "
                "and df.filter(expr) for row filtering."
            )
        if isinstance(key, list):
            raise TypeError(
                "DataFrame indexing with a list is not supported. "
                "Use df.select(*cols) for multi-column projection."
            )
        if isinstance(key, Expr):
            raise TypeError(
                "DataFrame indexing with an Expr is not supported. "
                "Use df.filter(expr) for row filtering."
            )
        if isinstance(key, slice):
            raise TypeError(
                "DataFrame slicing is not supported. "
                "Use df.limit(n) or df.limit(n, offset=k)."
            )

        inner_expr = self._impl.qualified_column_expr(key)
        return Expr(inner_expr, self._ctx)

    def __getattr__(self, name):
        """Syntactic sugar for column access

        Allows columns to be accessed like `t.geometry` for columns
        that do not collide with existing attributes. Programmatic usage
        should use `sd.col()` or `t["col"]`.
        """
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(str(e))

    def __dir__(self):
        """List attributes of this object

        This is primarily intended to power autocomplete, so columns
        are placed first.
        """
        return self.columns + super().__dir__()

    def _ipython_key_completions_(self):
        """Enable tab completion for df["col"] in IPython/Jupyter."""
        return self.columns

    def select(
        self,
        *exprs: Union[Expr, str, _SedonaLit],
        **kwargs: Union[Expr, str, _SedonaLit],
    ) -> "DataFrame":
        """Project a set of columns or expressions.

        Returns a new lazy `DataFrame` whose columns are exactly the
        projection. Column-name strings are converted to column references
        via `sedonadb.expr.col` internally, so `df.select("x", "y")` and
        `df.select(col("x"), col("y"))` produce the same plan. Literals
        produced by `sedonadb.expr.lit()` are also accepted; use
        `lit(value).alias(name)` to give the literal column a name.

        Keyword arguments provide a shorthand for aliasing: the key becomes
        the output column name and the value is the expression. For example,
        `df.select(z=df.x + 1)` is equivalent to `df.select((df.x + 1).alias("z"))`.

        Args:
            *exprs: Zero or more positional arguments. Each argument is either
                a column name (`str`), a `sedonadb.expr.Expr`, or a
                `sedonadb.expr.Literal`.
            **kwargs: Zero or more keyword arguments where each key is the
                desired output column name and each value is a column name
                (`str`), an `Expr`, or a `Literal`.

        Note:
            At least one positional or keyword argument is required.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT 1 AS a, 2 AS b")
            >>> df.select("a", (df.b + 1).alias("b_plus_1")).show()
            ┌───────┬──────────┐
            │   a   ┆ b_plus_1 │
            │ int64 ┆   int64  │
            ╞═══════╪══════════╡
            │     1 ┆        3 │
            └───────┴──────────┘

            >>> df.select("a", b_plus_1=df.b + 1).show()
            ┌───────┬──────────┐
            │   a   ┆ b_plus_1 │
            │ int64 ┆   int64  │
            ╞═══════╪══════════╡
            │     1 ┆        3 │
            └───────┴──────────┘
        """
        if not exprs and not kwargs:
            raise ValueError("select() requires at least one column or expression")

        coerced = collect_exprs("select", exprs, kwargs)
        return DataFrame(self._ctx, self._impl.select([e._impl for e in coerced]))

    def filter(self, *exprs: Expr) -> "DataFrame":
        """Filter rows by one or more boolean expressions.

        Multiple expressions are combined with logical AND, so
        `df.filter(a, b)` is equivalent to `df.filter(a & b)` and to
        `df.filter(a).filter(b)` (the planner sees one conjunction in
        the first two forms and two filter nodes in the third).

        Only `Expr` arguments are accepted. Strings are not interpreted
        as SQL predicates (that is a separate feature). Bare `Literal`
        values are also rejected — `filter(lit(True))` is almost
        certainly a typo; if you really mean a constant predicate, wrap
        a column expression like `col("flag") == lit(True)`.

        Args:
            *exprs: One or more boolean `sedonadb.expr.Expr` predicates.
                At least one argument is required.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES (1), (2), (3), (4)) AS t(x)")
            >>> df.filter(df.x > 2).show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     3 │
            ├╌╌╌╌╌╌╌┤
            │     4 │
            └───────┘
        """
        if not exprs:
            raise ValueError("filter() requires at least one predicate")

        for e in exprs:
            if isinstance(e, _SedonaLit):
                raise TypeError(
                    "filter() does not accept Literal arguments. "
                    "filter(lit(value)) is almost always a typo; if you really "
                    "want a constant predicate, wrap a column expression like "
                    "col('flag') == lit(value)."
                )
            if not isinstance(e, Expr):
                raise TypeError(
                    f"filter() expects Expr arguments, got {type(e).__name__}"
                )

        return DataFrame(
            self._ctx,
            self._impl.filter([e._impl for e in exprs]),
        )

    def sort(self, *keys: Union[str, Expr, SortExpr]) -> "DataFrame":
        """Sort rows by one or more keys.

        Each argument is either a column name (`str`), an `Expr`, or a
        `SortExpr` (built via `Expr.asc()` / `Expr.desc()` or
        `sedonadb.expr.sort_expr(...)`). Strings and bare `Expr`s
        auto-promote to ascending sort keys with nulls placed last; for
        descending order or null-first placement, use the explicit
        `SortExpr` forms.

        Null placement defaults to "nulls last" for both ascending and
        descending sorts (overriding DataFusion's SQL-style nulls-first-
        on-descending). Override via `sort_expr(expr, asc=..., nulls_first=...)`
        for the SQL behavior.

        Args:
            *keys: One or more sort keys, in order of priority. At least
                one is required.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES (3), (1), (2)) AS t(x)")
            >>> df.sort("x").show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     1 │
            ├╌╌╌╌╌╌╌┤
            │     2 │
            ├╌╌╌╌╌╌╌┤
            │     3 │
            └───────┘
            >>> df.sort(sd.col("x").desc()).show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     3 │
            ├╌╌╌╌╌╌╌┤
            │     2 │
            ├╌╌╌╌╌╌╌┤
            │     1 │
            └───────┘
        """
        if not keys:
            raise ValueError("sort() requires at least one sort key")

        coerced: List = []
        for k in keys:
            if isinstance(k, SortExpr):
                coerced.append(k._impl)
            elif isinstance(k, Expr):
                # Default direction is ascending, nulls last.
                coerced.append(k.asc()._impl)
            elif isinstance(k, str):
                coerced.append(_col(k).asc()._impl)
            else:
                raise TypeError(
                    f"sort() expects str, Expr, or SortExpr arguments, "
                    f"got {type(k).__name__}"
                )

        return DataFrame(self._ctx, self._impl.sort(coerced))

    def drop(self, *cols: str) -> "DataFrame":
        """Drop the named columns.

        Returns a new lazy `DataFrame` with each named column removed.
        Only column-name strings are accepted; expression arguments are
        rejected because "drop a computed expression" has no meaning at
        the schema level. Unknown column names raise a `SedonaError` at
        plan-build time, with the list of valid field names included in
        the message.

        Args:
            *cols: One or more column names to drop. At least one is
                required.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT 1 AS a, 2 AS b, 3 AS c")
            >>> df.drop("b").show()
            ┌───────┬───────┐
            │   a   ┆   c   │
            │ int64 ┆ int64 │
            ╞═══════╪═══════╡
            │     1 ┆     3 │
            └───────┴───────┘
        """
        if not cols:
            raise ValueError("drop() requires at least one column name")

        for c in cols:
            if not isinstance(c, str):
                raise TypeError(f"drop() expects str arguments, got {type(c).__name__}")

        # DataFusion's `drop_columns` silently ignores names not in the
        # schema — that hides typos. Validate Python-side so the user
        # gets an immediate KeyError listing the available columns.
        columns = self._impl.columns()
        unknown = [c for c in cols if c not in columns]
        if unknown:
            raise KeyError(
                f"Column(s) {unknown} not found. Available columns: {columns}"
            )

        return DataFrame(self._ctx, self._impl.drop_columns(list(cols)))

    def unnest(self, *columns: str) -> "DataFrame":
        """Expand list/array column(s) so each element becomes its own row.

        This is the relational form of `pandas`/`GeoPandas` `explode()`: for
        each input row, a list-typed column is expanded to one output row per
        element, with the other columns repeated. When several columns are
        given, they are unnested in parallel (position by position).

        A common spatial use is flattening a multi-geometry: `ST_Dump(geom)`
        returns a list of parts, and `unnest()` turns each part into its own
        row.

        Args:
            *columns: One or more list/array column names to expand. At least
                one is required.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT 'a' AS label, [10, 20, 30] AS vals")
            >>> df.unnest("vals").show()
            ┌───────┬───────┐
            │ label ┆  vals │
            │  utf8 ┆ int64 │
            ╞═══════╪═══════╡
            │ a     ┆    10 │
            ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
            │ a     ┆    20 │
            ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
            │ a     ┆    30 │
            └───────┴───────┘

            Explode a multi-geometry into one row per part: dump the parts with
            `ST_Dump`, `unnest` the resulting list, then read the geometry out
            of the `{path, geom}` struct each element carries.

            >>> multi = sd.sql("SELECT ST_GeomFromText('MULTIPOINT (0 0, 1 1)') AS geometry")
            >>> dumped = multi.select(multi["geometry"].geo.dump().alias("dump")).unnest("dump")
            >>> dumped.select(dumped["dump"]["geom"].alias("geometry")).show()
            ┌────────────┐
            │  geometry  │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 0) │
            ├╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ POINT(1 1) │
            └────────────┘
        """
        if not columns:
            raise ValueError("unnest() requires at least one column name")

        for c in columns:
            if not isinstance(c, str):
                raise TypeError(
                    f"unnest() expects str arguments, got {type(c).__name__}"
                )

        return DataFrame(self._ctx, self._impl.unnest(list(columns)))

    def agg(self, *exprs: Expr, **named_exprs: Expr) -> "DataFrame":
        """Aggregate the entire DataFrame to a single row.

        Aggregate expressions can be passed positionally or as keyword
        arguments. With keyword arguments the keyword becomes the
        output column name — `df.agg(total=sd.funcs.sum(sd.col("x")))`
        is shorthand for
        `df.agg(sd.funcs.sum(sd.col("x")).alias("total"))`. The two
        forms can be mixed in a single call.

        Args:
            *exprs: Positional aggregate expressions.
            **named_exprs: Keyword aggregate expressions; each keyword
                is applied as the output alias of the corresponding
                expression.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES (1), (2), (3), (4)) AS t(x)")
            >>> df.agg(sd.funcs.sum(sd.col("x")).alias("total")).show()
            ┌───────┐
            │ total │
            │ int64 │
            ╞═══════╡
            │    10 │
            └───────┘
            >>> df.agg(total=sd.funcs.sum(sd.col("x"))).show()
            ┌───────┐
            │ total │
            │ int64 │
            ╞═══════╡
            │    10 │
            └───────┘
        """
        if not exprs and not named_exprs:
            raise ValueError("agg() requires at least one aggregate expression")

        for e in exprs:
            if not isinstance(e, Expr):
                raise TypeError(f"agg() expects Expr arguments, got {type(e).__name__}")

        all_exprs: List[Expr] = list(exprs)
        for name, e in named_exprs.items():
            if not isinstance(e, Expr):
                raise TypeError(
                    f"agg() expects Expr keyword values, got {type(e).__name__} "
                    f"for keyword {name!r}"
                )
            all_exprs.append(e.alias(name))

        return DataFrame(
            self._ctx,
            self._impl.aggregate([], [e._impl for e in all_exprs]),
        )

    def group_by(self, *keys: Union[str, Expr]) -> "GroupedDataFrame":
        """Group rows by one or more keys for aggregation.

        Returns a `GroupedDataFrame` whose `.agg(...)` method runs the
        aggregation. Strings are auto-promoted to column references
        (same pattern as `sort`); arbitrary `Expr` values are accepted
        as computed group keys.

        Args:
            *keys: One or more `str` column names or `Expr` group keys.
                At least one is required.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql(
            ...     "SELECT * FROM (VALUES ('a', 1), ('a', 2), ('b', 3)) AS t(k, v)"
            ... )
            >>> df.group_by("k").agg(total=sd.funcs.sum(sd.col("v"))).sort("k").show()
            ┌──────┬───────┐
            │   k  ┆ total │
            │ utf8 ┆ int64 │
            ╞══════╪═══════╡
            │ a    ┆     3 │
            ├╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
            │ b    ┆     3 │
            └──────┴───────┘
        """
        if not keys:
            raise ValueError("group_by() requires at least one key")

        coerced: List[Expr] = []
        for k in keys:
            if isinstance(k, Expr):
                coerced.append(k)
            elif isinstance(k, str):
                coerced.append(_col(k))
            else:
                raise TypeError(
                    f"group_by() expects str or Expr arguments, got {type(k).__name__}"
                )

        return GroupedDataFrame(self, coerced)

    def join(
        self,
        other: "DataFrame",
        on: Union[str, List[str], Expr, List[Expr]],
        how: Literal[
            "inner",
            "left",
            "right",
            "outer",
            "full",
            "left_semi",
            "semi",
            "left_anti",
            "anti",
            "right_semi",
            "right_anti",
        ] = "inner",
    ) -> "DataFrame":
        """Join two DataFrames.

        `on` accepts either common column names or arbitrary boolean
        predicates:

        - **Column names** (`str` or `list[str]`): the named column(s)
          must exist on both sides. Result has a single copy of each
          join key — matching pandas / Polars / PySpark output shape.
        - **Predicate expressions** (`Expr` or `list[Expr]`): each Expr
          is a boolean predicate combining columns from both sides
          (e.g. `left.k == right.k`, or `f.st_intersects(left.g, right.g)`).
          Result keeps both sides' columns verbatim — disambiguate
          with `df.alias(...)` on either side.

        Args:
            other: The right-hand DataFrame to join against.
            on: Join key(s). A column name (`str`), a list of column
                names, a single boolean `Expr`, or a list of boolean
                `Expr`s combined with logical AND.
            how: Join type. Canonical: `"inner"` (default), `"left"`,
                `"right"`, `"outer"`, `"left_semi"`, `"left_anti"`,
                `"right_semi"`, `"right_anti"`. PySpark aliases also
                accepted: `"full"` (= outer), `"semi"` (= left_semi),
                `"anti"` (= left_anti).

        Examples:

            >>> sd = sedona.db.connect()
            >>> left = sd.sql(
            ...     "SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(k, v)"
            ... )
            >>> right = sd.sql(
            ...     "SELECT * FROM (VALUES (1, 'x'), (2, 'y'), (3, 'z')) AS t(k, w)"
            ... )
            >>> left.join(right, on="k").sort("k").show()
            ┌───────┬──────┬──────┐
            │   k   ┆   v  ┆   w  │
            │ int64 ┆ utf8 ┆ utf8 │
            ╞═══════╪══════╪══════╡
            │     1 ┆ a    ┆ x    │
            ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
            │     2 ┆ b    ┆ y    │
            └───────┴──────┴──────┘
        """
        if not isinstance(other, DataFrame):
            raise TypeError(
                f"join() expects a DataFrame as the first argument, "
                f"got {type(other).__name__}"
            )

        HOW_ALIASES = {"semi": "left_semi", "anti": "left_anti", "full": "outer"}
        valid_canonical = {
            "inner",
            "left",
            "right",
            "outer",
            "left_semi",
            "left_anti",
            "right_semi",
            "right_anti",
        }
        canonical_how = HOW_ALIASES.get(how, how)
        if canonical_how not in valid_canonical:
            accepted = sorted(valid_canonical | set(HOW_ALIASES))
            raise ValueError(f"join() `how` must be one of {accepted}, got {how!r}")

        if isinstance(on, (str, Expr)):
            on_list = [on]
        elif isinstance(on, list):
            on_list = on
        else:
            raise TypeError(
                f"join() `on` expects str, Expr, or a list of either, "
                f"got {type(on).__name__}"
            )

        if not on_list:
            raise ValueError("join() requires at least one element in `on`")

        is_str_keys = all(isinstance(x, str) for x in on_list)
        is_expr_keys = all(isinstance(x, Expr) for x in on_list)
        if not (is_str_keys or is_expr_keys):
            raise TypeError(
                "join() `on` list must contain only str or only Expr, not a mix"
            )

        if is_expr_keys:
            joined_impl = self._impl.join_on(
                other._impl, [p._impl for p in on_list], canonical_how
            )
            return DataFrame(self._ctx, joined_impl)

        # String-keys path: alias both sides, synthesize equi-join
        # predicates from the qualified columns, then project to dedupe
        # the join keys so the output shape matches pandas / PySpark
        # rather than DataFusion's keep-both-copies default.
        LEFT_ALIAS = "_sd_join_left_"
        RIGHT_ALIAS = "_sd_join_right_"
        left_cols = self._impl.columns()
        right_cols = other._impl.columns()

        missing_left = [k for k in on_list if k not in left_cols]
        missing_right = [k for k in on_list if k not in right_cols]
        if missing_left or missing_right:
            raise KeyError(
                f"Join keys missing — left: {missing_left}, "
                f"right: {missing_right}. "
                f"Left columns: {left_cols}; right columns: {right_cols}"
            )

        left_aliased = self.alias(LEFT_ALIAS)
        right_aliased = other.alias(RIGHT_ALIAS)

        predicates = [(left_aliased[k] == right_aliased[k])._impl for k in on_list]
        joined_impl = left_aliased._impl.join_on(
            right_aliased._impl, predicates, canonical_how
        )

        if canonical_how in ("left_semi", "left_anti"):
            projection = [left_aliased[c]._impl for c in left_cols]
        elif canonical_how in ("right_semi", "right_anti"):
            projection = [right_aliased[c]._impl for c in right_cols]
        else:
            # For the unified key column, pick the side that is always
            # populated: right join takes from the right, outer COALESCEs
            # so unmatched-on-either-side rows still carry a key value.
            key_set = set(on_list)
            projection = []
            for c in left_cols:
                if c in key_set and canonical_how == "right":
                    projection.append(right_aliased[c]._impl)
                elif c in key_set and canonical_how == "outer":
                    coalesced = self._ctx.funcs.coalesce(
                        left_aliased[c], right_aliased[c]
                    ).alias(c)
                    projection.append(coalesced._impl)
                else:
                    projection.append(left_aliased[c]._impl)
            for c in right_cols:
                if c not in key_set:
                    projection.append(right_aliased[c]._impl)

        return DataFrame(self._ctx, joined_impl.select(projection))

    def cross_join(self, other: "DataFrame") -> "DataFrame":
        """Cartesian product of two DataFrames.

        Returns a DataFrame containing every pair of rows from `self`
        and `other`; the row count is the product of the two input
        row counts. Both sides' columns are kept verbatim —
        disambiguate with `df.alias(...)` on either side if column
        names collide.

        Args:
            other: The right-hand DataFrame.

        Examples:

            >>> sd = sedona.db.connect()
            >>> left = sd.sql("SELECT * FROM (VALUES (1), (2)) AS t(x)")
            >>> right = sd.sql("SELECT * FROM (VALUES ('a'), ('b')) AS t(y)")
            >>> left.cross_join(right).sort("x", "y").show()
            ┌───────┬──────┐
            │   x   ┆   y  │
            │ int64 ┆ utf8 │
            ╞═══════╪══════╡
            │     1 ┆ a    │
            ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
            │     1 ┆ b    │
            ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
            │     2 ┆ a    │
            ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
            │     2 ┆ b    │
            └───────┴──────┘
        """
        if not isinstance(other, DataFrame):
            raise TypeError(
                f"cross_join() expects a DataFrame as the first argument, "
                f"got {type(other).__name__}"
            )
        return DataFrame(self._ctx, self._impl.cross_join(other._impl))

    def distinct(self) -> "DataFrame":
        """Remove duplicate rows, comparing all columns.

        Returns a new lazy DataFrame keeping one row from each set of
        rows that are equal across every column (SQL `SELECT DISTINCT`).

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql(
            ...     "SELECT * FROM (VALUES (1), (1), (2)) AS t(x)"
            ... )
            >>> df.distinct().sort("x").show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     1 │
            ├╌╌╌╌╌╌╌┤
            │     2 │
            └───────┘
        """
        return DataFrame(self._ctx, self._impl.distinct())

    def distinct_on(self, *cols: Union[str, Expr]) -> "DataFrame":
        """Remove duplicate rows, comparing only the given key columns.

        Keeps one row for each distinct combination of `cols` (SQL
        `DISTINCT ON`) and returns all columns. Which row survives for each
        key is unspecified and not controllable today — there is no row
        ordering applied, and a preceding `.sort(...)` is not guaranteed to
        be respected. For a deterministic result, ensure the non-key
        columns are the same for all rows that share a key.

        Args:
            *cols: One or more key columns (`str` names or `Expr`). At
                least one is required.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql(
            ...     "SELECT * FROM (VALUES (1, 'a'), (1, 'a'), (2, 'b')) AS t(k, v)"
            ... )
            >>> df.distinct_on("k").sort("k").show()
            ┌───────┬──────┐
            │   k   ┆   v  │
            │ int64 ┆ utf8 │
            ╞═══════╪══════╡
            │     1 ┆ a    │
            ├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
            │     2 ┆ b    │
            └───────┴──────┘
        """
        if not cols:
            raise ValueError("distinct_on() requires at least one column")

        coerced = []
        for c in cols:
            if isinstance(c, Expr):
                coerced.append(c._impl)
            elif isinstance(c, str):
                coerced.append(_col(c)._impl)
            else:
                raise TypeError(
                    f"distinct_on() expects str or Expr arguments, "
                    f"got {type(c).__name__}"
                )

        return DataFrame(self._ctx, self._impl.distinct_on(coerced))

    def _check_set_op_compatible(self, other: "DataFrame", method: str) -> None:
        # Both inputs must line up by column name (in order) so a positional
        # set operation can't silently misalign differently-named columns —
        # the classic footgun. Callers that want a positional set operation
        # on differently-named columns opt in by aliasing/selecting to align
        # the names first.
        if not isinstance(other, DataFrame):
            raise TypeError(
                f"{method}() expects a DataFrame as the first argument, "
                f"got {type(other).__name__}"
            )
        if self.columns != other.columns:
            raise ValueError(
                f"{method}() requires both DataFrames to have the same column "
                f"names in the same order; got {self.columns} and "
                f"{other.columns}. Align them first (e.g. with select) if you "
                f"intend a positional union."
            )

    def union(self, other: "DataFrame") -> "DataFrame":
        """Concatenate two DataFrames vertically, keeping duplicate rows.

        This is SQL `UNION ALL` (and matches PySpark's `union`): duplicate
        rows are preserved. Use `union_distinct` for SQL `UNION` semantics.
        Both DataFrames must have the same column names in the same order;
        otherwise an error is raised. (To union differently-named columns
        positionally, align the names first, e.g. with `select`.)

        Args:
            other: The DataFrame to append. Must have the same column names,
                in the same order, as this DataFrame.

        Examples:

            >>> sd = sedona.db.connect()
            >>> a = sd.sql("SELECT * FROM (VALUES (1), (2)) AS t(x)")
            >>> b = sd.sql("SELECT * FROM (VALUES (2), (3)) AS t(x)")
            >>> a.union(b).sort("x").show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     1 │
            ├╌╌╌╌╌╌╌┤
            │     2 │
            ├╌╌╌╌╌╌╌┤
            │     2 │
            ├╌╌╌╌╌╌╌┤
            │     3 │
            └───────┘
        """
        self._check_set_op_compatible(other, "union")
        return DataFrame(self._ctx, self._impl.union(other._impl))

    def union_distinct(self, other: "DataFrame") -> "DataFrame":
        """Concatenate two DataFrames vertically and drop duplicate rows.

        This is SQL `UNION`: the result has no duplicate rows (comparing
        all columns). Use `union` to keep duplicates (`UNION ALL`). Both
        DataFrames must have the same column names in the same order;
        otherwise an error is raised. (To union differently-named columns
        positionally, align the names first, e.g. with `select`.)

        Args:
            other: The DataFrame to append. Must have the same column names,
                in the same order, as this DataFrame.

        Examples:

            >>> sd = sedona.db.connect()
            >>> a = sd.sql("SELECT * FROM (VALUES (1), (2)) AS t(x)")
            >>> b = sd.sql("SELECT * FROM (VALUES (2), (3)) AS t(x)")
            >>> a.union_distinct(b).sort("x").show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     1 │
            ├╌╌╌╌╌╌╌┤
            │     2 │
            ├╌╌╌╌╌╌╌┤
            │     3 │
            └───────┘
        """
        self._check_set_op_compatible(other, "union_distinct")
        return DataFrame(self._ctx, self._impl.union_distinct(other._impl))

    def intersect(self, other: "DataFrame") -> "DataFrame":
        """Rows present in both DataFrames, keeping duplicate rows.

        This is SQL `INTERSECT ALL`: a row that appears `m` times on the
        left and `n` times on the right appears `min(m, n)` times in the
        result. Use `intersect_distinct` for SQL `INTERSECT` semantics.
        Both DataFrames must have the same column names in the same order.

        Args:
            other: The DataFrame to intersect with. Must have the same
                column names, in the same order, as this DataFrame.

        Examples:

            >>> sd = sedona.db.connect()
            >>> a = sd.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(x)")
            >>> b = sd.sql("SELECT * FROM (VALUES (2), (3), (4)) AS t(x)")
            >>> a.intersect(b).sort("x").show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     2 │
            ├╌╌╌╌╌╌╌┤
            │     3 │
            └───────┘
        """
        self._check_set_op_compatible(other, "intersect")
        return DataFrame(self._ctx, self._impl.intersect(other._impl))

    def intersect_distinct(self, other: "DataFrame") -> "DataFrame":
        """Distinct rows present in both DataFrames.

        This is SQL `INTERSECT`: the result has no duplicate rows. Use
        `intersect` to preserve multiplicity (`INTERSECT ALL`). Both
        DataFrames must have the same column names in the same order.

        Args:
            other: The DataFrame to intersect with. Must have the same
                column names, in the same order, as this DataFrame.
        """
        self._check_set_op_compatible(other, "intersect_distinct")
        return DataFrame(self._ctx, self._impl.intersect_distinct(other._impl))

    def except_distinct(self, other: "DataFrame") -> "DataFrame":
        """Distinct rows in this DataFrame that are not in `other`.

        This is SQL `EXCEPT`: the result has no duplicate rows. Both
        DataFrames must have the same column names in the same order.

        There is intentionally no `except`/`except_all` counterpart:
        multiplicity-preserving `EXCEPT ALL` is not supported by the
        underlying engine (a true anti-difference would instead be a
        `join(how="anti")`), so only the distinct variant is exposed.

        Args:
            other: The DataFrame whose rows to subtract. Must have the same
                column names, in the same order, as this DataFrame.

        Examples:

            >>> sd = sedona.db.connect()
            >>> a = sd.sql("SELECT * FROM (VALUES (1), (2), (3)) AS t(x)")
            >>> b = sd.sql("SELECT * FROM (VALUES (2), (3), (4)) AS t(x)")
            >>> a.except_distinct(b).sort("x").show()
            ┌───────┐
            │   x   │
            │ int64 │
            ╞═══════╡
            │     1 │
            └───────┘
        """
        self._check_set_op_compatible(other, "except_distinct")
        return DataFrame(self._ctx, self._impl.except_distinct(other._impl))

    def mutate(
        self,
        *exprs: Union[Expr, str, _SedonaLit],
        **named_exprs: Union[Expr, str, _SedonaLit],
    ) -> "DataFrame":
        """Add or replace columns, keeping all existing ones.

        Adds each given column (Ibis / dplyr `mutate`). Positional and
        keyword arguments follow the same rules as `select`: a positional
        value is used as-is (its own output name applies), a keyword value
        is aliased to the keyword. A column whose output name matches an
        existing column replaces it in place; genuinely new columns are
        appended in the order given. Unlike `select`, you don't re-list the
        columns you want to keep — `mutate` keeps them all.

        Args:
            *exprs: Positional column definitions (`Expr` / column-name
                `str` / `lit()` literal), used with their own output names.
            **named_exprs: Keyword column definitions, aliased to the key.
                At least one positional or keyword argument is required.

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT 1 AS a, 2 AS b")
            >>> df.mutate(c=df.a + df.b, b=df.b * 10).show()
            ┌───────┬───────┬───────┐
            │   a   ┆   b   ┆   c   │
            │ int64 ┆ int64 ┆ int64 │
            ╞═══════╪═══════╪═══════╡
            │     1 ┆    20 ┆     3 │
            └───────┴───────┴───────┘
        """
        if not exprs and not named_exprs:
            raise ValueError("mutate() requires at least one column")

        # Coerce with the same rules as select, then key each result by its
        # output name so an existing column can be replaced in place.
        coerced = collect_exprs("mutate", exprs, named_exprs)
        by_name: Dict[str, Expr] = {}
        order: List[str] = []
        for e in coerced:
            name = e._output_name()
            if name not in by_name:
                order.append(name)
            by_name[name] = e

        # Single projection: existing columns (replaced in place where a
        # mutation matches the name), then the new columns in input order.
        current = self._impl.columns()
        projection = [by_name[c] if c in by_name else _col(c) for c in current]
        projection += [by_name[name] for name in order if name not in current]

        return DataFrame(self._ctx, self._impl.select([e._impl for e in projection]))

    def rename(self, *args: Any, **new_to_old: str) -> "DataFrame":
        """Rename columns, keeping all others and their order.

        Follows the Ibis / dplyr convention: each keyword is the **new**
        name and its value is the existing (**old**) name — i.e.
        `df.rename(new_name="old_name")`. This is the reverse of pandas /
        Polars, which map old→new; passing a dict raises a clear error
        pointing at the keyword form.

        Args:
            **new_to_old: `new_name="old_name"` pairs. At least one is
                required.

        Raises:
            KeyError: If an old column name doesn't exist (the message
                lists the available columns).

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT 1 AS a, 2 AS b")
            >>> df.rename(c="b").show()
            ┌───────┬───────┐
            │   a   ┆   c   │
            │ int64 ┆ int64 │
            ╞═══════╪═══════╡
            │     1 ┆     2 │
            └───────┴───────┘
        """
        # The common pandas/Polars habit is `rename({"old": "new"})`; catch a
        # positional (dict) argument and redirect to the keyword form. A clear,
        # actionable message so it's an easy fix (including for LLMs iterating).
        if args:
            hint = ""
            if len(args) == 1 and isinstance(args[0], dict):
                pairs = ", ".join(f'{v}="{k}"' for k, v in args[0].items())
                hint = f" Rewrite {args[0]!r} as rename({pairs})."
            raise TypeError(
                'rename() takes new_name="old_name" keyword arguments '
                "(Ibis/dplyr style), not positional/dict arguments — this is "
                "the reverse of the pandas/Polars {old: new} mapping." + hint
            )

        if not new_to_old:
            raise ValueError('rename() requires at least one new_name="old_name" pair')

        old_to_new: Dict[str, str] = {}
        for new_name, old_name in new_to_old.items():
            if not isinstance(old_name, str):
                raise TypeError(
                    'rename() takes new_name="old_name" keyword pairs '
                    "(the value is the existing column name as a str); got "
                    f"{type(old_name).__name__} for '{new_name}'"
                )
            old_to_new[old_name] = new_name

        columns = self._impl.columns()
        missing = [old for old in old_to_new if old not in columns]
        if missing:
            raise KeyError(
                f"Column(s) {missing} not found. Available columns: {columns}"
            )

        projection = [
            _col(c).alias(old_to_new[c]) if c in old_to_new else _col(c)
            for c in columns
        ]
        return DataFrame(self._ctx, self._impl.select([e._impl for e in projection]))

    def limit(self, n: Optional[int], /, *, offset: int = 0) -> "DataFrame":
        """Limit result to n rows starting at offset

        Note that this is non-deterministic for many queries.

        Args:
            n: The number of rows to return
            offset: The number of rows to skip (optional)

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
            >>> df.limit(1).show()
            ┌──────┐
            │  val │
            │ utf8 │
            ╞══════╡
            │ one  │
            └──────┘

            >>> df.limit(1, offset=2).show()
            ┌───────┐
            │  val  │
            │  utf8 │
            ╞═══════╡
            │ three │
            └───────┘

        """
        return DataFrame(self._ctx, self._impl.limit(n, offset))

    def execute(self) -> None:
        """Execute the plan represented by this DataFrame

        This will execute the query without collecting results into memory,
        which is useful for executing SQL statements like SET, CREATE VIEW,
        and CREATE EXTERNAL TABLE.

        Note that this is functionally similar to `.count()` except it does
        not apply any optimizations (e.g., does not use statistics to avoid
        reading data to calculate a count).

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("CREATE OR REPLACE VIEW temp_view AS SELECT 1 as one").execute()
            0
            >>> sd.view("temp_view").show()
            ┌───────┐
            │  one  │
            │ int64 │
            ╞═══════╡
            │     1 │
            └───────┘
        """
        return self._impl.execute()

    def count(self) -> int:
        """Compute the number of rows in this DataFrame

        Examples:

            >>> sd = sedona.db.connect()
            >>> df = sd.sql("SELECT * FROM (VALUES ('one'), ('two'), ('three')) AS t(val)")
            >>> df.count()
            3

        """
        return self._impl.count()

    def with_params(self, *args: List[Any], **kwargs: Dict[str, Any]):
        """Replace unbound parameters in this query

        For DataFrames that represent a logical plan that contains parameters (e.g.,
        a SQL query of `SELECT $1 + 2`), replace parameters with concrete values.
        See `lit()` for a list of supported Python objects.

        Args:
            args: Values to bind to positional parameters (e.g., `$1`, `$2`, `$3`)
            kwargs: Values to bind to named parameters (e.g., `$my_param`). Note that
                positional and named parameters cannot currently be mixed (i.e.,
                parameters must be all positional or all named).

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT $1 + 2 AS c").with_params(100).show()
            ┌───────┐
            │   c   │
            │ int64 │
            ╞═══════╡
            │   102 │
            └───────┘
            >>> sd.sql("SELECT $my_param + 2 AS c").with_params(my_param=100).show()
            ┌───────┐
            │   c   │
            │ int64 │
            ╞═══════╡
            │   102 │
            └───────┘

        """
        from sedonadb.expr.literal import lit

        positional_params = [lit(arg) for arg in args]
        named_params = {k: lit(param) for k, param in kwargs.items()}

        return DataFrame(
            self._ctx,
            self._impl.with_params(positional_params, named_params),
        )

    def __arrow_c_schema__(self):
        """ArrowSchema PyCapsule interface

        Returns a PyCapsule wrapping an Arrow C Schema for interoperability
        with libraries that understand Arrow C data types. See the
        [Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
        for more details.
        """
        return self._impl.schema().__arrow_c_schema__()

    def __arrow_c_stream__(self, requested_schema: Any = None):
        """ArrowArrayStream Stream PyCapsule interface

        Returns a PyCapsule wrapping an Arrow C ArrayStream for interoperability
        with libraries that understand Arrow C data types. See the
        [Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
        for more details.

        Args:
            requested_schema: A PyCapsule representing the desired output schema.
        """
        return self._impl.to_stream(self._ctx._impl, simplify=False).__arrow_c_stream__(
            requested_schema=requested_schema
        )

    def to_arrow_reader(self, *, simplify: bool = False) -> "pa.RecordBatchReader":
        """Execute and stream results as a PyArrow RecordBatchReader

        Executes the logical plan represented by this object and returns a
        PyArrow RecordBatchReader. This requires that pyarrow is installed.

        Args:
            simplify: Use `True` to simplify Arrow storage types at the export
                boundary, for example `Utf8View` to `Utf8` and `BinaryView` to
                `Binary`.

        Examples:

            >>> sd = sedona.db.connect()
            >>> reader = sd.sql(
            ...     "SELECT ST_Point(0, 1) as geometry"
            ... ).to_arrow_reader()
            >>> reader.read_all()
            pyarrow.Table
            geometry: extension<geoarrow.wkb<WkbType>> not null
            ----
            geometry: [[01010000000000000000000000000000000000F03F]]

        """
        import pyarrow as pa
        from sedonadb.utility import register_pyarrow_extension_types

        register_pyarrow_extension_types()

        return pa.RecordBatchReader.from_stream(
            self._impl.to_stream(self._ctx._impl, simplify=simplify)
        )

    def arrow(self, *, simplify: bool = False) -> "pa.RecordBatchReader":
        """Alias of `to_arrow_reader()`"""
        return self.to_arrow_reader(simplify=simplify)

    def to_view(self, name: str, overwrite: bool = False):
        """Create a view based on the query represented by this object

        Registers this logical plan as a named view with the underlying context
        such that it can be referred to in SQL.

        Args:
            name: The name to which this query should be referred
            overwrite: Use `True` to overwrite an existing view of this name

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geom").to_view("foofy")
            >>> sd.view("foofy").show()
            ┌────────────┐
            │    geom    │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        self._impl.to_view(self._ctx._impl, name, overwrite)

    def to_memtable(self) -> "DataFrame":
        """Collect a data frame into a memtable

        Executes the logical plan represented by this object and returns a
        DataFrame representing it.

        Does not guarantee ordering of rows.  Use `to_arrow_table()` if
        ordering is needed.

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geom").to_memtable().show()
            ┌────────────┐
            │    geom    │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        return DataFrame(self._ctx, self._impl.to_memtable(self._ctx._impl))

    def __sedonadb_table_provider__(self):
        return self._impl.__sedonadb_table_provider__(self._ctx._impl)

    def to_arrow_table(self, schema: Any = None) -> "pa.Table":
        """Execute and collect results as a PyArrow Table

        Executes the logical plan represented by this object and returns a
        PyArrow Table. This requires that pyarrow is installed.

        Args:
            schema: The requested output schema or `None` to use the inferred
                schema.

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geometry").to_arrow_table()
            pyarrow.Table
            geometry: extension<geoarrow.wkb<WkbType>> not null
            ----
            geometry: [[01010000000000000000000000000000000000F03F]]

        """
        import pyarrow as pa
        from sedonadb.utility import register_pyarrow_extension_types

        register_pyarrow_extension_types()

        # Collects all batches into an object that exposes __arrow_c_stream__()
        batches = self._impl.to_batches(schema)
        return pa.table(batches)

    def to_pandas(
        self, geometry: Optional[str] = None
    ) -> Union["pandas.DataFrame", "geopandas.GeoDataFrame"]:
        """Execute and collect results as a pandas DataFrame or GeoDataFrame

        If this data frame contains geometry columns, collect results as a
        single [`geopandas.GeoDataFrame`][]. Otherwise, collect results as a
        [`pandas.DataFrame`][].

        Args:
            geometry: If specified, the name of the column to use for the default
                geometry column. If not specified, this is inferred as the column
                named "geometry", the column named "geography", or the first
                column with a spatial data type (in that order).

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geometry").to_pandas()
                  geometry
            0  POINT (0 1)

        """
        table = self.to_arrow_table()

        if geometry is None:
            geometry = self._impl.primary_geometry_column()

        if geometry:
            from geopandas import GeoDataFrame

            return GeoDataFrame.from_arrow(table, geometry=geometry)
        else:
            return table.to_pandas()

    def to_parquet(
        self,
        path: Union[str, Path],
        *,
        options: Optional[Dict[str, Any]] = None,
        partition_by: Optional[Union[str, Iterable[str]]] = None,
        sort_by: Optional[Union[str, Iterable[str]]] = None,
        single_file_output: Optional[bool] = None,
        geoparquet_version: Literal["1.0", "1.1", "2.0", "none", None] = None,
        overwrite_bbox_columns: Optional[bool] = None,
        max_row_group_size: Optional[int] = None,
        compression: Optional[str] = None,
    ):
        """Write this DataFrame to one or more (Geo)Parquet files

        For input that contains geometry columns, GeoParquet metadata is written
        such that suitable readers can recreate Geometry/Geography types when
        reading the output and potentially read fewer row groups when only a
        subset of the file is needed for a given query.

        Args:
            path: A filename or directory to which parquet file(s) should be written.
            options: Key/value options to be used when constructing a parquet writer.
                Common options are exposed as other arguments to `to_parquet()`; however,
                this argument allows setting any DataFusion Parquet writer option. If
                an option is specified here and by an argument to this function, the
                value specified as a keyword argument takes precedence.
            partition_by: A vector of column names to partition by. If non-empty,
                applies hive-style partitioning to the output.
            sort_by: A vector of column names to sort by. Currently only ascending
                sort is supported.
            single_file_output: Use True or False to force writing a single Parquet
                file vs. writing one file per partition to a directory. By default,
                a single file is written if `partition_by` is unspecified and
                `path` ends with `.parquet`.
            geoparquet_version: GeoParquet metadata version to write if output contains
                one or more geometry columns. The default (1.0) is the most widely
                supported and will result in geometry columns being recognized in many
                readers; however, only includes statistics at the file level.

                Use GeoParquet 1.1 to compute an additional bounding box column
                for every geometry column in the output: some readers can use these columns
                to prune row groups when files contain an effective spatial ordering.
                The extra columns will appear just before their geometry column and
                will be named "[geom_col_name]_bbox" for all geometry columns except
                "geometry", whose bounding box column name is just "bbox".

                Use GeoParquet 2.0 to write compatible GeoParquet metadata with
                Parquet-native Geometry and/or Geography data types; use "none" to omit
                GeoParquet metadata completely.
            overwrite_bbox_columns: Use `True` to overwrite any bounding box columns
                that already exist in the input. This is useful in a read -> modify
                -> write scenario to ensure these columns are up-to-date. If `False`
                (the default), an error will be raised if a bbox column already exists.
            max_row_group_size: Target maximum number of rows in each row group. Defaults
                to the global configuration value (1M rows).
            compression: Sets the Parquet compression codec. Valid values are: uncompressed,
                snappy, gzip(level), brotli(level), lz4, zstd(level), and lz4_raw. Defaults
                to the global configuration value (zstd(3)).

        Examples:

            >>> import tempfile
            >>> sd = sedona.db.connect()
            >>> td = tempfile.TemporaryDirectory()
            >>> url = "https://github.com/apache/sedona-testing/raw/refs/heads/main/data/parquet/geoparquet-1.1.0.parquet"
            >>> sd.read_parquet(url).to_parquet(f"{td.name}/tmp.parquet")

        """

        path = Path(path)

        if options is not None:
            options = {k: str(v) for k, v in options.items()}
        else:
            options = {}

        if max_row_group_size is not None:
            options["max_row_group_size"] = str(max_row_group_size)

        if compression is not None:
            options["compression"] = str(compression)

        if geoparquet_version is not None:
            options["geoparquet_version"] = str(geoparquet_version)

        if overwrite_bbox_columns is not None:
            options["overwrite_bbox_columns"] = str(overwrite_bbox_columns)

        if single_file_output is None:
            single_file_output = partition_by is None and str(path).endswith(".parquet")

        if isinstance(partition_by, str):
            partition_by = [partition_by]
        elif partition_by is not None:
            partition_by = list(partition_by)
        else:
            partition_by = []

        if isinstance(sort_by, str):
            sort_by = [sort_by]
        elif sort_by is not None:
            sort_by = list(sort_by)
        else:
            sort_by = []

        self._impl.to_parquet(
            self._ctx._impl,
            str(path),
            options,
            partition_by,
            sort_by,
            single_file_output,
        )

    def to_pyogrio(
        self,
        path: Union[str, Path, io.BytesIO],
        *,
        driver: Optional[str] = None,
        geometry_type: Optional[str] = None,
        geometry_name: Optional[str] = None,
        crs: Optional[str] = None,
        append: bool = False,
        **kwargs: Dict[str, Any],
    ):
        """Write using GDAL/OGR via pyogrio

        Writes this DataFrame batchwise to a file using GDAL/OGR using the
        implementation provided by the pyogrio package. This is the same backend
        used by GeoPandas and this function is a light wrapper around
        `pyogrio.raw.write_arrow()` that fills in default values using
        information available to the DataFrame (e.g., geometry column and CRS).

        Args:
            path: An output path or `BytesIO` output buffer.
            driver: An explicit GDAL OGR driver. Usually inferred from `path` but
                must be provided if path is a `BytesIO`. Not all drivers support
                writing to `BytesIO`.
            geometry_type: A GeoJSON-style geometry type or `None` to provide an
                inferred default value (which may be `"Unknown"`). This is required
                to write some types of output (e.g. Shapefiles) and may provide
                files that are more efficiently read.
            geometry_name: The column to write as the primary geometry column. If
                `None`, the name of the geometry column will be inferred.
            crs: An optional string overriding the CRS of `geometry_name`.
            append: Use `True` to append to the file for drivers that support
                appending.
            kwargs: Extra arguments passed to `pyogrio.raw.write_arrow()`.

        Examples:

            >>> import tempfile
            >>> sd = sedona.db.connect()
            >>> td = tempfile.TemporaryDirectory()
            >>> sd.sql("SELECT ST_Point(0, 1, 3857)").to_pyogrio(f"{td.name}/tmp.fgb")
            >>> sd.read_pyogrio(f"{td.name}/tmp.fgb").show()
            ┌──────────────┐
            │ wkb_geometry │
            │   geometry   │
            ╞══════════════╡
            │ POINT(0 1)   │
            └──────────────┘
        """
        if geometry_name is None:
            geometry_name = self._impl.primary_geometry_column()

        if crs is None and geometry_name is not None:
            inferred_crs = self.schema.field(geometry_name).type.crs
            crs = None if inferred_crs is None else inferred_crs.to_json()

        if geometry_type is None:
            # This is required for pyogrio.raw.write_arrow(). We could try harder
            # to infer this because some drivers need this information.
            geometry_type = "Unknown"

        if isinstance(path, Path):
            path = str(path)

        if isinstance(path, io.BytesIO) and driver is None:
            raise ValueError("driver must be provided when path is a BytesIO")

        # There may be more endings worth special-casing here but zipped FlatGeoBuf
        # is particularly useful and isn't automatically recognized
        if driver is None and isinstance(path, str) and path.endswith(".fgb.zip"):
            driver = "FlatGeoBuf"

        # GDAL does not support newer Arrow types like string views util 3.14, so we export a
        # reader with simpler types here
        self_simplified = self._impl.to_stream(self._ctx._impl, simplify=True)

        # Writer: pyogrio.write_arrow() via Cython ogr_write_arrow()
        # https://github.com/geopandas/pyogrio/blob/3b2d40273b501c10ecf46cbd37c6e555754c89af/pyogrio/raw.py#L755-L897
        # https://github.com/geopandas/pyogrio/blob/3b2d40273b501c10ecf46cbd37c6e555754c89af/pyogrio/_io.pyx#L2858-L2980
        import pyogrio.raw

        pyogrio.raw.write_arrow(
            self_simplified,
            path,
            driver=driver,
            geometry_type=geometry_type,
            geometry_name=geometry_name,
            crs=crs,
            append=append,
            **kwargs,
        )

    def show(
        self,
        limit: Optional[int] = 10,
        width: Optional[int] = None,
        ascii: bool = False,
    ) -> str:
        """Print the first limit rows to the console

        Args:
            limit: The number of rows to display. Using None will display the
                entire table which may result in very large output.
            width: The number of characters to use to display the output.
                If None, uses `Options.width` or detects the value from the
                current terminal if available. The default width is 100 characters
                if a width is not set by another mechanism.
            ascii: Use True to disable UTF-8 characters in the output.

        Examples:

            >>> sd = sedona.db.connect()
            >>> sd.sql("SELECT ST_Point(0, 1) as geometry").show()
            ┌────────────┐
            │  geometry  │
            │  geometry  │
            ╞════════════╡
            │ POINT(0 1) │
            └────────────┘

        """
        width = self._out_width(width)
        print(self._impl.show(self._ctx._impl, limit, width, ascii), end="")

    def explain(
        self,
        type: str = "standard",
        format: str = "indent",
    ) -> "DataFrame":
        """Return the execution plan for this DataFrame as a DataFrame

        Retrieves the logical and physical execution plans that will be used to
        compute this DataFrame. This is useful for understanding query
        performance and optimization.

        Args:
            type: The type of explain plan to generate. Supported values are:
                "standard" (default) - shows logical and physical plans,
                "extended" - includes additional query optimization details,
                "analyze" - executes the plan and reports actual metrics.
            format: The format to use for displaying the plan. Supported formats are
                "indent" (default), "tree", "pgjson" and "graphviz".

        Returns:
            A DataFrame containing the execution plan information with columns
            'plan_type' and 'plan'.

        Examples:

            >>> import sedonadb
            >>> con = sedonadb.connect()
            >>> df = con.sql("SELECT 1 as one")
            >>> df.explain().show()
            ┌───────────────┬─────────────────────────────────┐
            │   plan_type   ┆               plan              │
            │      utf8     ┆               utf8              │
            ╞═══════════════╪═════════════════════════════════╡
            │ logical_plan  ┆ Projection: Int64(1) AS one     │
            │               ┆   EmptyRelation: rows=1         │
            ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ physical_plan ┆ ProjectionExec: expr=[1 as one] │
            │               ┆   PlaceholderRowExec            │
            │               ┆                                 │
            └───────────────┴─────────────────────────────────┘
        """
        return DataFrame(self._ctx, self._impl.explain(type, format))

    def __repr__(self) -> str:
        if self._ctx.options.interactive:
            width = self._out_width()
            return self._impl.show(self._ctx._impl, 10, width, ascii=False).strip()
        else:
            return super().__repr__()

    def __len__(self):
        raise ValueError(
            "Can't compute len() of a lazy SedonaDB DataFrame. "
            "Use .count() to execute and resolve the number of rows."
        )

    def _simplify_storage_types(self):
        return DataFrame(self._ctx, self._impl.simplify_storage_types(self._ctx._impl))

    def _out_width(self, width=None) -> int:
        if width is None:
            width = self._ctx.options.width

        if width is None:
            import shutil

            width, _ = shutil.get_terminal_size(fallback=(100, 24))

        return width


def _create_data_frame(ctx, obj, schema) -> DataFrame:
    """Create a DataFrame (internal)

    This is defined here because we need it in future dataframe methods like
    inner_join() (as well as context methods like create_data_frame()). This
    handles interpreting an arbitrary object as a DataFrame.
    """
    # If we're dealing with an anonymous data frame on the same context,
    # just return it. Otherwise, fall back to the default interpretation
    # (which uses __sedonadb_table_provider__).
    if isinstance(obj, DataFrame) and obj._ctx is ctx and schema is None:
        return obj

    # We special case a few object types where collecting the __arrow_c_stream__
    # up front provides a better user experience. These are objects where the
    # cost of converting to Arrow is cheap and it makes more sense to do it once.
    # This includes geopandas/pandas DataFrames, pyarrow tables, and Polars tables.
    type_name = _qualified_type_name(obj)
    if type_name in SPECIAL_CASED_SCANS:
        return SPECIAL_CASED_SCANS[type_name](ctx, obj, schema)

    # The default implementation handles objects that implement
    # __sedonadb_table_provider__ or __arrow_c_stream__. For objects implementing
    # __arrow_c_stream__, this currently will only work for a single scan (i.e.,
    # the returned data frame can't be previewed before the query is computed).
    return _scan_default(ctx, obj, schema)


def _scan_default(ctx, obj, schema):
    impl = ctx._impl.create_data_frame(obj, schema)
    return DataFrame(ctx, impl)


def _scan_collected_default(ctx, obj, schema):
    return _scan_default(ctx, obj, schema).to_memtable()


class GroupedDataFrame:
    """A `DataFrame` partitioned by one or more group keys.

    Produced by `DataFrame.group_by(...)`. The class exists as a step
    in the chain to simplify aggregation expressions.
    """

    __slots__ = ("_df", "_group_exprs")

    def __init__(self, df: DataFrame, group_exprs: List[Expr]):
        self._df = df
        self._group_exprs = group_exprs

    def agg(self, *exprs: Expr, **named_exprs: Expr) -> DataFrame:
        """Aggregate within each group.

        Same signature as `DataFrame.agg`: positional aggregate `Expr`s
        and/or keyword aggregates where the keyword is the output
        column name.

        Args:
            *exprs: Positional aggregate expressions.
            **named_exprs: Keyword aggregate expressions; each keyword
                becomes the output alias.
        """
        if not exprs and not named_exprs:
            raise ValueError("agg() requires at least one aggregate expression")

        for e in exprs:
            if not isinstance(e, Expr):
                raise TypeError(f"agg() expects Expr arguments, got {type(e).__name__}")

        all_exprs: List[Expr] = list(exprs)
        for name, e in named_exprs.items():
            if not isinstance(e, Expr):
                raise TypeError(
                    f"agg() expects Expr keyword values, got {type(e).__name__} "
                    f"for keyword {name!r}"
                )
            all_exprs.append(e.alias(name))

        return DataFrame(
            self._df._ctx,
            self._df._impl.aggregate(
                [g._impl for g in self._group_exprs],
                [e._impl for e in all_exprs],
            ),
        )


def _scan_geopandas(ctx, obj, schema):
    return _scan_collected_default(ctx, obj.to_arrow(geometry_encoding="WKB"), schema)


def _qualified_type_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


SPECIAL_CASED_SCANS = {
    "pyarrow.lib.Table": _scan_collected_default,
    # pandas < 3.0
    "pandas.core.frame.DataFrame": _scan_collected_default,
    # pandas >= 3.0
    "pandas.DataFrame": _scan_collected_default,
    "geopandas.geodataframe.GeoDataFrame": _scan_geopandas,
    "polars.dataframe.frame.DataFrame": _scan_collected_default,
}
