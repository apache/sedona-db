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
"""The `.str` and `.dt` accessors of a Series, in the shape of pandas'.

Each method builds an expression over the column, so the result is a Series
of the same frame and nothing is computed until it is collected. The engine's
string and date functions differ from Python's in many small ways (whitespace,
padding that truncates, Sunday-first weekdays, ...); each method says what it
builds on where the difference matters.
"""

import numbers
import re

import pyarrow as pa
from sedonadb.expr import lit

from sedonadb_geopandas._series import (
    Series,
    _choose,
    _kind,
    _operand,
    _pandas_missing,
)

# Default for `na`, telling "not given" from None, a missing value.
_NO_DEFAULT = object()


def _missing_answer(dtype, accessor, method, *args, **kwargs):
    """pandas' answer to `<accessor>.<method>(*args, **kwargs)` for a missing
    value of `dtype`: a boolean, or None if it is missing.

    pandas 2 leaves a missing value missing in a boolean result, while
    pandas 3 makes it False for its strings, and either makes it `na` when
    given; asking the installed pandas keeps the answer in step with what
    `to_pandas()` returns. Only a boolean answer fits a boolean column.
    """
    import numpy as np
    import pandas as pd

    missing = _pandas_missing(dtype)
    if missing is None:
        return None
    answer = getattr(getattr(missing, accessor), method)
    # A method, or a property such as `dt.is_month_start`.
    answer = (answer(*args, **kwargs) if callable(answer) else answer).iloc[0]
    if pd.isna(answer):
        return None
    if not isinstance(answer, (bool, np.bool_)):
        raise TypeError(f"na must be a boolean or a missing value, not {answer!r}")
    return bool(answer)


def _answered(series, expr, answer):
    """A boolean result `expr` as a Series, with `answer` for a missing value
    (None to leave it missing)."""
    if answer is not None:
        expr = expr.funcs.coalesce(lit(answer))
    return Series(series._df, expr, series._name)


# -- regular expressions -----------------------------------------------------

# The flags of Python's `re` that the engine's regular expressions spell as
# inline flags. re.UNICODE is how a str pattern always matches there anyway.
_INLINE_FLAGS = {
    re.IGNORECASE: "i",
    re.MULTILINE: "m",
    re.DOTALL: "s",
    re.VERBOSE: "x",
}

# Python's `\Z`, the end of the string, unescaped: the engine spells it `\z`.
_END_ANCHOR = re.compile(r"(?<!\\)((?:\\\\)*)\\Z")

# The characters the engine's regular expressions give a meaning (its
# `regex::escape` escapes these).
_META = set("\\.+*?()|[]{}^$#&-~")


def _pattern_and_flags(pat, flags, case, method):
    """The pattern text and `re` flags of `pat`, a string or a compiled
    pattern, with `case=False` as IGNORECASE."""
    if isinstance(pat, re.Pattern):
        flags |= pat.flags
        pat = pat.pattern
    if not isinstance(pat, str):
        raise TypeError(f"{method}() needs a string pattern, not {type(pat).__name__}")
    if not isinstance(flags, numbers.Integral):
        raise TypeError(f"flags must be an integer, not {type(flags).__name__}")
    if case is False:
        flags |= re.IGNORECASE
    return pat, flags & ~re.UNICODE


def _engine_regex(pat, flags=0, case=True, method="contains"):
    """`pat` (Python's `re` syntax), with `flags`, as a pattern of the engine's
    regular expressions (Rust's `regex` crate).

    The syntax is shared but for what the engine lacks (lookaround,
    backreferences: it refuses those) and `\\Z`, which it spells `\\z`. Flags
    become inline flags; ones it has no counterpart for are refused.
    """
    pat, flags = _pattern_and_flags(pat, flags, case, method)
    inline = ""
    for flag, letter in _INLINE_FLAGS.items():
        if flags & flag:
            inline += letter
            flags &= ~flag
    if flags:
        raise NotImplementedError(
            f"{method}() can't use the flags {re.RegexFlag(flags)!r}: the "
            f"engine's regular expressions have no counterpart (re.ASCII and "
            f"re.LOCALE included)"
        )
    pat = _END_ANCHOR.sub(r"\1\\z", pat)
    return f"(?{inline}){pat}" if inline else pat


def _escaped(text):
    """`text` as a pattern of the engine's regular expressions matching it
    literally."""
    return "".join(f"\\{c}" if c in _META else c for c in text)


# The escapes a replacement template of Python's `re` reads as characters.
_TEMPLATE_ESCAPES = {
    "\\": "\\",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}


def _engine_template(repl):
    """A replacement template of Python's `re` (`\\1`, `\\g<1>`, `\\g<name>`)
    in the engine's syntax (`${1}`, `${name}`), the rest of it literal.

    Octal escapes are not supported; an unknown escape of an ASCII letter is
    an error, as in Python, and of any other character is kept as written.
    """
    out = []
    i = 0
    while i < len(repl):
        c = repl[i]
        if c == "$":
            out.append("$$")
            i += 1
            continue
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 == len(repl):
            raise re.error("bad escape (end of pattern)", repl, i)
        nxt = repl[i + 1]
        if nxt == "g":
            match = re.match(r"<([^>]*)>", repl[i + 2 :])
            if match is None or not match.group(1):
                raise re.error("missing group name", repl, i)
            out.append("${" + match.group(1) + "}")
            i += 2 + match.end()
        elif nxt.isdigit():
            if nxt == "0" or re.match(r"[0-7]{3}", repl[i + 1 : i + 4]):
                raise NotImplementedError(
                    "replace() does not support octal escapes in the replacement"
                )
            # A group number has one or two digits, as in Python.
            digits = re.match(r"\d\d?", repl[i + 1 :]).group(0)
            out.append("${" + digits + "}")
            i += 1 + len(digits)
        elif nxt in _TEMPLATE_ESCAPES:
            out.append(_TEMPLATE_ESCAPES[nxt])
            i += 2
        elif nxt.isascii() and nxt.isalpha():
            raise re.error(f"bad escape \\{nxt}", repl, i)
        else:
            # Kept as written, backslash and all.
            out.append("\\" + ("$$" if nxt == "$" else nxt))
            i += 2
    return "".join(out)


# -- strings -----------------------------------------------------------------


class StringMethods:
    """String functions for a Series of strings, as pandas' `Series.str`.

    **EXPERIMENTAL.** Each method computes its result in the engine, as a
    Series of the same frame; a missing value stays missing, but for the
    boolean methods, which answer for it as pandas does (with pandas 3,
    False). Lengths and positions count characters (code points), as Python
    does.

    Regular expressions are the engine's (Rust's `regex` crate): Python's
    syntax for most patterns, but without lookaround or backreferences, with
    `$` matching at the end of a string only (Python's also matches before a
    final newline), with classes such as `\\w` as Unicode defines them, and
    without an empty match right after another match (`x*` matches "xy"
    twice, not three times), as pandas 3's pyarrow-backed strings have it.
    """

    def __init__(self, series):
        dtype = series._dtype()
        if _kind(dtype) != "string" and not pa.types.is_null(dtype):
            raise AttributeError("Can only use .str accessor with string values!")
        self._series = series
        # A string result has this column's string type; a column of only
        # missing values (Arrow's null type), which pandas reads as Python
        # objects, keeps its type, though its values are read as strings.
        self._type = dtype
        values = series._decoded()
        self._values = values.cast(pa.string()) if pa.types.is_null(dtype) else values

    @property
    def _ctx(self):
        return self._series._df._ctx

    def _wrap(self, expr):
        return Series(self._series._df, expr, self._series._name)

    def _text(self, expr):
        """A string result as a Series, in this column's string type: the
        engine's functions answer some in another (`substr()` in views). A
        column of the null type has nothing but missing values to give."""
        if pa.types.is_null(self._type):
            return self._wrap(self._series._decoded())
        return self._wrap(expr.cast(self._type))

    def _count(self, expr):
        """A count or position as a Series of int64, as pandas has them."""
        return self._wrap(expr.cast(pa.int64()))

    def _boolean(self, expr, method, *args, na=_NO_DEFAULT):
        """A boolean result as a Series, missing values answered as pandas
        answers `method(*args, na=na)` for them.

        A dictionary-encoded column is a categorical in pandas, which answers
        as its categories' strings do: as the decoded type does.
        """
        kwargs = {} if na is _NO_DEFAULT else {"na": na}
        answer = _missing_answer(self._type, "str", method, *args, **kwargs)
        return _answered(self._series, expr, answer)

    def _length(self):
        return self._values.funcs.character_length().cast(pa.int64())

    def _present(self, expr, dtype):
        """`expr` (of `dtype`) where this column's value is present, and
        missing where it is missing: some of the engine's functions answer a
        missing value (`concat()` skips it, `regexp_count()` counts 0)."""
        return _choose(self._values.is_not_null(), expr, lit(pa.scalar(None, dtype)))

    # -- size and case ---------------------------------------------------------
    def len(self):
        """The number of characters of each value."""
        return self._count(self._length())

    def lower(self):
        """Each value in lower case, by Unicode's full case mapping, as Python's
        `str.lower()`."""
        return self._text(self._values.funcs.lower())

    def upper(self):
        """Each value in upper case, by Unicode's full case mapping ("ß" becomes
        "SS"), as Python's `str.upper()`."""
        return self._text(self._values.funcs.upper())

    # -- trimming --------------------------------------------------------------
    def strip(self, to_strip=None):
        """Each value without leading and trailing whitespace, or without the
        characters of `to_strip` there, as Python's `str.strip()`."""
        return self._stripped(to_strip, "btrim", r"^{0}+|{0}+$")

    def lstrip(self, to_strip=None):
        """Each value without leading whitespace (or `to_strip` characters)."""
        return self._stripped(to_strip, "ltrim", r"^{0}+")

    def rstrip(self, to_strip=None):
        """Each value without trailing whitespace (or `to_strip` characters)."""
        return self._stripped(to_strip, "rtrim", r"{0}+$")

    def _stripped(self, to_strip, trim, ends):
        if to_strip is not None:
            if not isinstance(to_strip, str):
                raise TypeError(
                    f"to_strip must be a string or None, not {type(to_strip).__name__}"
                )
            return self._text(getattr(self._values.funcs, trim)(lit(to_strip)))
        # Python's whitespace: the engine's trim functions strip spaces only,
        # and its \s lacks the information separators \x1c-\x1f, which
        # Python's str.isspace() counts.
        pattern = ends.format(r"[\s\x1c-\x1f]")
        return self._text(
            self._values.funcs.regexp_replace(lit(pattern), lit(""), lit("g"))
        )

    def removeprefix(self, prefix):
        """Each value without `prefix` where it starts with it."""
        if not isinstance(prefix, str):
            raise TypeError(f"prefix must be a string, not {type(prefix).__name__}")
        starts = self._values.funcs.starts_with(lit(prefix)).cast(pa.int64())
        return self._text(self._values.funcs.substr(starts * lit(len(prefix)) + lit(1)))

    def removesuffix(self, suffix):
        """Each value without `suffix` where it ends with it."""
        if not isinstance(suffix, str):
            raise TypeError(f"suffix must be a string, not {type(suffix).__name__}")
        ends = self._values.funcs.ends_with(lit(suffix)).cast(pa.int64())
        kept = self._length() - ends * lit(len(suffix))
        return self._text(self._values.funcs.substr(lit(1), kept))

    # -- tests -----------------------------------------------------------------
    def startswith(self, pat, na=_NO_DEFAULT):
        """Whether each value starts with `pat`, a string or a tuple of them
        (any of which)."""
        return self._affix("startswith", "starts_with", pat, na)

    def endswith(self, pat, na=_NO_DEFAULT):
        """Whether each value ends with `pat`, a string or a tuple of them."""
        return self._affix("endswith", "ends_with", pat, na)

    def _affix(self, method, function, pat, na):
        pats = (pat,) if isinstance(pat, str) else pat
        if not isinstance(pats, tuple) or not all(isinstance(p, str) for p in pats):
            raise TypeError(f"expected a string or tuple, not {type(pat).__name__}")
        tests = [getattr(self._values.funcs, function)(lit(p)) for p in pats]
        if tests:
            expr = tests[0]
            for test in tests[1:]:
                expr = expr | test
        else:
            # No affix: False for a present value, missing for a missing one.
            expr = self._values.is_null().funcs.nullif(lit(True))
        return self._boolean(expr, method, "", na=na)

    def contains(self, pat, case=True, flags=0, na=_NO_DEFAULT, regex=True):
        """Whether each value contains `pat`, a regular expression (or a
        string, if `regex=False`), ignoring case if `case=False`."""
        if regex:
            pattern = _engine_regex(pat, flags, case, "contains")
            expr = self._values.funcs.regexp_like(lit(pattern))
        else:
            if not isinstance(pat, str):
                raise TypeError(
                    f"contains() with regex=False needs a string, not "
                    f"{type(pat).__name__}"
                )
            if case:
                expr = self._values.funcs.contains(lit(pat))
            else:
                expr = self._values.funcs.regexp_like(lit("(?i)" + _escaped(pat)))
        return self._boolean(expr, "contains", "", na=na)

    def match(self, pat, case=True, flags=0, na=_NO_DEFAULT):
        """Whether a match of the regular expression `pat` starts each value,
        as `re.match()`."""
        pattern = _engine_regex(pat, flags, case, "match")
        # `\A` is the start of the string whatever the flags, as re.match()
        # anchors there with re.MULTILINE too; flags inside the group hold
        # for all of it.
        expr = self._values.funcs.regexp_like(lit(rf"\A(?:{pattern})"))
        return self._boolean(expr, "match", "", na=na)

    def fullmatch(self, pat, case=True, flags=0, na=_NO_DEFAULT):
        """Whether the regular expression `pat` matches each whole value, as
        `re.fullmatch()`."""
        pattern = _engine_regex(pat, flags, case, "fullmatch")
        expr = self._values.funcs.regexp_like(lit(rf"\A(?:{pattern})\z"))
        return self._boolean(expr, "fullmatch", "", na=na)

    def count(self, pat, flags=0):
        """The number of non-overlapping matches of the regular expression
        `pat` in each value."""
        pattern = _engine_regex(pat, flags, True, "count")
        counted = self._values.funcs.regexp_count(lit(pattern))
        return self._count(self._present(counted.cast(pa.int64()), pa.int64()))

    def find(self, sub, start=0, end=None):
        """The position of the first `sub` in each value, -1 if none."""
        self._check_find(sub, start, end, "find")
        return self._count(self._values.funcs.strpos(lit(sub)) - lit(1))

    def rfind(self, sub, start=0, end=None):
        """The position of the last `sub` in each value, -1 if none."""
        self._check_find(sub, start, end, "rfind")
        # The first match in the reversed value, counted from the end: the
        # engine has no search from the right.
        found = self._values.funcs.reverse().funcs.strpos(lit(sub[::-1]))
        hit = (found > lit(0)).cast(pa.int64())
        position = self._length() - lit(len(sub)) + lit(2) - found
        return self._count(hit * position - lit(1))

    @staticmethod
    def _check_find(sub, start, end, method):
        if not isinstance(sub, str):
            raise TypeError(f"expected a string object, not {type(sub).__name__}")
        if start != 0 or end is not None:
            raise NotImplementedError(f"{method}() does not support start or end")

    # -- replacing -------------------------------------------------------------
    def replace(self, pat, repl=None, n=-1, case=None, flags=0, regex=False):
        """Each value with `pat` replaced by `repl`, as pandas does: `pat` is
        a string (a regular expression if `regex=True`) or a dict of patterns
        to their replacements, applied in turn.

        With a regular expression, `repl` can refer to groups as Python's `re`
        does (`\\1`, `\\g<name>`). `n` is the number of replacements: all of
        them if negative, the first only if 1, none if 0.
        """
        pairs = pat.items() if isinstance(pat, dict) else [(pat, repl)]
        if isinstance(pat, dict) and repl is not None:
            raise ValueError("repl cannot be used when pat is a dictionary")
        if isinstance(n, bool) or not isinstance(n, numbers.Integral):
            raise TypeError(f"n must be an integer, not {type(n).__name__}")
        if n not in (0, 1) and n >= 0:
            raise NotImplementedError(
                f"replace() supports n=-1 (all), n=1 (the first) or n=0 (none), "
                f"got n={n}"
            )
        result = self
        for key, value in pairs:
            result = StringMethods(result._replaced(key, value, n, case, flags, regex))
        return result._series

    def _replaced(self, pat, repl, n, case, flags, regex):
        if callable(repl):
            raise NotImplementedError("replace() does not support a callable repl")
        if not isinstance(repl, str):
            raise TypeError("repl must be a string or callable")
        if isinstance(pat, re.Pattern):
            if not regex:
                raise ValueError(
                    "Cannot use a compiled regex as replacement pattern with regex=False"
                )
            if case is not None or flags != 0:
                raise ValueError(
                    "case and flags cannot be set when pat is a compiled regex"
                )
        if n == 0:
            return self._series
        if case is None:
            case = True
        if regex:
            pattern = _engine_regex(pat, flags, case, "replace")
            template = _engine_template(repl)
        else:
            # A literal pattern: of the flags, only re.IGNORECASE changes what
            # one matches, as in pandas.
            pat, flags = _pattern_and_flags(pat, flags, case, "replace")
            ignore_case = bool(flags & re.IGNORECASE)
            if not ignore_case and n < 0:
                return self._text(self._values.funcs.replace(lit(pat), lit(repl)))
            pattern = ("(?i)" if ignore_case else "") + _escaped(pat)
            template = repl.replace("$", "$$")
        arguments = [lit(pattern), lit(template)]
        if n < 0:
            arguments.append(lit("g"))
        return self._text(self._values.funcs.regexp_replace(*arguments))

    # -- slicing ---------------------------------------------------------------
    def slice(self, start=None, stop=None, step=None):
        """Each value sliced as Python slices a string (`value[start:stop]`).

        `step` is 1, or -1 over the whole value, which reverses it.
        """
        for name, value in (("start", start), ("stop", stop), ("step", step)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, numbers.Integral)
            ):
                raise TypeError(f"slice {name} must be an integer or None")
        if step == -1 and start is None and stop is None:
            return self._text(self._values.funcs.reverse())
        if step not in (None, 1):
            raise NotImplementedError(
                "slice() supports a step of 1, or -1 over the whole value"
            )
        first = self._bound(0 if start is None else start)
        position = lit(first + 1) if isinstance(first, int) else first + lit(1)
        if stop is None:
            return self._text(self._values.funcs.substr(position))
        last = self._bound(stop)
        if isinstance(first, int) and isinstance(last, int):
            count = lit(max(last - first, 0))
        else:
            count = (last - first).funcs.greatest(lit(0))
        return self._text(self._values.funcs.substr(position, count))

    def _bound(self, index):
        """A slice bound as a position from 0: a number, or for one from the
        end an expression of the value's length. A position past the end
        needs no limit, as `substr()` stops there."""
        if index >= 0:
            return int(index)
        return (self._length() + lit(int(index))).funcs.greatest(lit(0))

    def get(self, i):
        """The character at position `i` of each value (from the end if
        negative), missing if there is none."""
        if isinstance(i, bool) or not isinstance(i, numbers.Integral):
            raise TypeError(f"get() takes an integer position, not {i!r}")
        # A position before the first character gives an empty string, as
        # does one past the last: no character there.
        position = lit(int(i) + 1) if i >= 0 else self._length() + lit(int(i) + 1)
        char = self._values.funcs.substr(position, lit(1))
        return self._text(char.funcs.nullif(lit("")))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.slice(key.start, key.stop, key.step)
        return self.get(key)

    # -- padding ---------------------------------------------------------------
    def pad(self, width, side="left", fillchar=" "):
        """Each value padded with `fillchar` to `width` characters, on the
        `side` "left", "right", or "both"; a longer value is unchanged."""
        if not isinstance(width, numbers.Integral) or isinstance(width, bool):
            raise TypeError(
                f"width must be of integer type, not {type(width).__name__}"
            )
        if not isinstance(fillchar, str):
            raise TypeError(
                f"fillchar must be a character, not {type(fillchar).__name__}"
            )
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        if side not in ("left", "right", "both"):
            raise ValueError("Invalid side")
        width = int(width)
        length = self._length()
        # The engine's lpad()/rpad() cut a longer value down to the width.
        full = length.funcs.greatest(lit(width))
        fill = lit(fillchar)
        if side == "left":
            return self._text(self._values.funcs.lpad(full, fill))
        if side == "right":
            return self._text(self._values.funcs.rpad(full, fill))
        # As Python's str.center(): the odd padding character goes on the
        # left when the width is odd, and on the right when it is even.
        total = (lit(width) - length).funcs.greatest(lit(0))
        left = (total + lit(width % 2)) / lit(2)
        padded = self._values.funcs.lpad(length + left, fill)
        return self._text(padded.funcs.rpad(full, fill))

    def ljust(self, width, fillchar=" "):
        """Each value padded on the right to `width` characters."""
        return self.pad(width, side="right", fillchar=fillchar)

    def rjust(self, width, fillchar=" "):
        """Each value padded on the left to `width` characters."""
        return self.pad(width, side="left", fillchar=fillchar)

    def center(self, width, fillchar=" "):
        """Each value padded on both sides to `width` characters."""
        return self.pad(width, side="both", fillchar=fillchar)

    def zfill(self, width):
        """Each value padded with zeros on the left to `width` characters,
        after a leading sign, as Python's `str.zfill()`."""
        if not isinstance(width, numbers.Integral) or isinstance(width, bool):
            raise TypeError(
                f"width must be of integer type, not {type(width).__name__}"
            )
        width = int(width)
        signed = (
            self._values.funcs.starts_with(lit("+"))
            | self._values.funcs.starts_with(lit("-"))
        ).cast(pa.int64())
        sign = self._values.funcs.substr(lit(1), signed)
        rest = self._values.funcs.substr(signed + lit(1))
        zeros = self._ctx.lit("0").funcs.repeat(
            (lit(width) - self._length()).funcs.greatest(lit(0))
        )
        padded = sign.funcs.concat(zeros, rest)
        return self._text(self._present(padded, pa.string()))

    def repeat(self, repeats):
        """Each value repeated `repeats` times."""
        if not isinstance(repeats, numbers.Integral) or isinstance(repeats, bool):
            raise NotImplementedError("repeat() supports a single integer count")
        return self._text(self._values.funcs.repeat(lit(int(repeats))))

    # -- concatenating ---------------------------------------------------------
    def cat(self, others=None, sep=None, na_rep=None, join="left"):
        """Each value followed by the values of `others` (a Series of strings
        of this frame, or a list of them), separated by `sep`.

        A missing value makes the result missing, or is written as `na_rep`.
        There is no index, so `join` changes nothing. Without `others`, pandas
        joins all the values into one string, in row order, which a lazy frame
        does not have.
        """
        if others is None:
            raise NotImplementedError(
                "cat() without others joins values in row order, which a lazy "
                "frame does not have"
            )
        if join not in ("left", "right", "outer", "inner"):
            raise ValueError(
                f"join must be 'left', 'right', 'outer' or 'inner', not {join!r}"
            )
        columns = [others] if isinstance(others, Series) else others
        if not isinstance(columns, (list, tuple)) or not all(
            isinstance(column, Series) for column in columns
        ):
            raise TypeError(
                "cat() takes Series of this frame as others (there is no row alignment)"
            )
        if sep is None:
            sep = ""
        if not isinstance(sep, str):
            raise TypeError(f"sep must be a string, not {type(sep).__name__}")
        pieces = [self._values]
        for column in columns:
            expr = _operand(self._series._df, column)
            dtype = column._dtype()
            if _kind(dtype) != "string" and not pa.types.is_null(dtype):
                raise TypeError(f"cat() joins strings, not {dtype}")
            pieces.append(expr.cast(pa.string()))
        if na_rep is not None:
            if not isinstance(na_rep, str):
                raise TypeError(f"na_rep must be a string, not {type(na_rep).__name__}")
            pieces = [piece.funcs.coalesce(lit(na_rep)) for piece in pieces]
        joined = pieces[0]
        for piece in pieces[1:]:
            joined = joined.funcs.concat(lit(sep), piece)
        if na_rep is None:
            # The engine's concat() skips a missing piece; pandas makes the
            # result missing.
            present = pieces[0].is_not_null()
            for piece in pieces[1:]:
                present = present & piece.is_not_null()
            joined = _choose(present, joined, lit(pa.scalar(None, pa.string())))
        return self._text(joined)


# -- dates and times -----------------------------------------------------------

_UNIT_NANOS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}

_DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

_MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def datetime_properties(series):
    """The `.dt` accessor of `series`: for timestamps or durations."""
    dtype = series._dtype()
    if pa.types.is_timestamp(dtype):
        return DatetimeProperties(series)
    if pa.types.is_duration(dtype):
        return TimedeltaProperties(series)
    raise AttributeError("Can only use .dt accessor with datetimelike values")


def _remainder(value, n):
    """`value` modulo `n` (a positive number), from 0 to `n` - 1, as Python's
    `%`: the engine's integer division truncates toward zero, so its
    remainder takes the sign of `value`."""
    remainder = value - (value / lit(n)) * lit(n)
    return remainder + lit(n) * (remainder < lit(0)).cast(pa.int64())


def _quotient(value, n):
    """`value` divided by `n` (a positive number), rounded down, as Python's
    `//`."""
    return (value - _remainder(value, n)) / lit(n)


def _names(ctx, names, index, present):
    """The name at the 0-based `index` (an expression) of `names`, where
    `present` (never null) is true, and missing elsewhere: the engine's
    array_element() reads a missing position as the first."""
    array = ctx.lit(names[0]).funcs.make_array(*[lit(name) for name in names[1:]])
    name = array.funcs.array_element(index + lit(1))
    return _choose(present, name, lit(pa.scalar(None, pa.string())))


def _check_raise(ambiguous, nonexistent):
    """pandas' `ambiguous` and `nonexistent`, as the engine localizes a wall
    time: an ambiguous or nonexistent one fails."""
    if not (isinstance(ambiguous, str) and ambiguous == "raise"):
        raise NotImplementedError(f"ambiguous={ambiguous!r} is not supported")
    if not (isinstance(nonexistent, str) and nonexistent == "raise"):
        raise NotImplementedError(f"nonexistent={nonexistent!r} is not supported")


def _time_zone(tz):
    """`tz`, a name or a tzinfo, as the name an Arrow timestamp type holds,
    checked as pandas checks it."""
    if isinstance(tz, str):
        pa.lib.string_to_tzinfo(tz)
        return tz
    return pa.lib.tzinfo_to_string(tz)


class DatetimeProperties:
    """Date and time properties of a Series of timestamps, as pandas'
    `Series.dt`.

    **EXPERIMENTAL.** Each is computed in the engine, as a Series of the same
    frame. For timestamps with a time zone, properties and rounding are of
    the wall time there, as in pandas; a missing timestamp gives a missing
    value, but for the `is_*` tests, which answer False for it.
    """

    def __init__(self, series):
        self._series = series
        self._type = series._dtype()
        self._values = series._decoded()
        # The wall time: naive timestamps already are.
        if self._type.tz is None:
            self._local = self._values
        else:
            self._local = self._values.funcs.to_local_time()

    @property
    def _ctx(self):
        return self._series._df._ctx

    def _wrap(self, expr):
        return Series(self._series._df, expr, self._series._name)

    def _part(self, part):
        """A field of the wall time, as the engine's date_part() reads it."""
        return self._ctx.lit(part).funcs.date_part(self._local)

    def _boolean(self, expr, name):
        answer = _missing_answer(self._type, "dt", name)
        return _answered(self._series, expr, answer)

    # -- fields ----------------------------------------------------------------
    @property
    def year(self):
        return self._wrap(self._part("year"))

    @property
    def month(self):
        return self._wrap(self._part("month"))

    @property
    def day(self):
        return self._wrap(self._part("day"))

    @property
    def hour(self):
        return self._wrap(self._part("hour"))

    @property
    def minute(self):
        return self._wrap(self._part("minute"))

    @property
    def second(self):
        return self._wrap(self._part("second"))

    @property
    def microsecond(self):
        # The engine's microsecond field counts the second's too.
        micros = self._part("microsecond")
        return self._wrap(_remainder(micros, 10**6).cast(pa.int32()))

    @property
    def nanosecond(self):
        # The nanoseconds past the microsecond, as pandas has them; the
        # engine's field counts the second's too.
        nanos = self._part("nanosecond")
        return self._wrap(_remainder(nanos, 1000).cast(pa.int32()))

    @property
    def day_of_week(self):
        """The day of the week, Monday as 0, as in pandas (the engine counts
        from Sunday)."""
        return self._wrap(_remainder(self._part("dow") + lit(6), 7).cast(pa.int32()))

    # pandas 3.1 deprecates these names; pandas 2 code uses them.
    dayofweek = day_of_week
    weekday = day_of_week

    @property
    def day_of_year(self):
        return self._wrap(self._part("doy"))

    dayofyear = day_of_year

    @property
    def quarter(self):
        return self._wrap(self._part("quarter"))

    def _leap(self):
        year = self._part("year").cast(pa.int64())
        return (_remainder(year, 4) == lit(0)) & (
            (_remainder(year, 100) != lit(0)) | (_remainder(year, 400) == lit(0))
        )

    def _days_in_month(self):
        month = self._part("month").cast(pa.int64())
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        lengths = self._ctx.lit(days[0]).funcs.make_array(*[lit(d) for d in days[1:]])
        february = (self._leap() & (month == lit(2))).cast(pa.int64())
        # Missing for a missing timestamp through `february`, though
        # array_element() reads a missing month as the first.
        return lengths.funcs.array_element(month) + february

    @property
    def days_in_month(self):
        return self._wrap(self._days_in_month().cast(pa.int32()))

    daysinmonth = days_in_month

    @property
    def is_leap_year(self):
        return self._boolean(self._leap(), "is_leap_year")

    @property
    def is_month_start(self):
        return self._boolean(self._part("day") == lit(1), "is_month_start")

    @property
    def is_month_end(self):
        end = self._part("day").cast(pa.int64()) == self._days_in_month()
        return self._boolean(end, "is_month_end")

    @property
    def is_quarter_start(self):
        start = (self._part("day") == lit(1)) & self._part("month").isin([1, 4, 7, 10])
        return self._boolean(start, "is_quarter_start")

    @property
    def is_quarter_end(self):
        end = self._part("day").cast(pa.int64()) == self._days_in_month()
        end = end & self._part("month").isin([3, 6, 9, 12])
        return self._boolean(end, "is_quarter_end")

    @property
    def is_year_start(self):
        start = (self._part("month") == lit(1)) & (self._part("day") == lit(1))
        return self._boolean(start, "is_year_start")

    @property
    def is_year_end(self):
        end = (self._part("month") == lit(12)) & (self._part("day") == lit(31))
        return self._boolean(end, "is_year_end")

    def day_name(self, locale=None):
        """The English name of the day of the week."""
        if locale is not None:
            raise NotImplementedError("day_name() supports the English names only")
        day = self.day_of_week._expr
        present = self._values.is_not_null()
        return self._wrap(_names(self._ctx, _DAY_NAMES, day, present))

    def month_name(self, locale=None):
        """The English name of the month."""
        if locale is not None:
            raise NotImplementedError("month_name() supports the English names only")
        month = self._part("month").cast(pa.int64()) - lit(1)
        present = self._values.is_not_null()
        return self._wrap(_names(self._ctx, _MONTH_NAMES, month, present))

    @property
    def date(self):
        """The date of the wall time, as Python dates."""
        return self._wrap(self._local.cast(pa.date32()))

    @property
    def time(self):
        """The time of day of the wall time, as Python times."""
        unit = "ns" if self._type.unit == "ns" else "us"
        return self._wrap(self._local.cast(pa.time64(unit)))

    @property
    def tz(self):
        """The time zone, as pandas has it (None for naive timestamps)."""
        return getattr(self._series.dtype, "tz", None)

    @property
    def unit(self):
        return self._type.unit

    # -- rounding --------------------------------------------------------------
    def floor(self, freq, ambiguous="raise", nonexistent="raise"):
        """Each timestamp rounded down to a multiple of `freq`, a fixed
        frequency ("h", "15min", "D", ...), counted from 1970-01-01."""
        return self._rounded(freq, "floor", ambiguous, nonexistent)

    def ceil(self, freq, ambiguous="raise", nonexistent="raise"):
        """Each timestamp rounded up to a multiple of `freq`."""
        return self._rounded(freq, "ceil", ambiguous, nonexistent)

    def round(self, freq, ambiguous="raise", nonexistent="raise"):
        """Each timestamp rounded to the nearest multiple of `freq`, half to
        even, as in pandas."""
        return self._rounded(freq, "round", ambiguous, nonexistent)

    def normalize(self):
        """Each timestamp at midnight of its day (of its wall time)."""
        return self._rounded("D", "floor", "raise", "raise")

    def _rounded(self, freq, mode, ambiguous, nonexistent):
        """Rounded as pandas rounds: the wall time's count of units, to a
        multiple of `freq` in units, then localized again, which fails for a
        wall time that is ambiguous or does not exist there, as pandas does
        by default."""
        from pandas.tseries.frequencies import to_offset

        _check_raise(ambiguous, nonexistent)
        # A ValueError for a frequency that is not fixed, as in pandas.
        nanos = to_offset(freq).nanos
        step = nanos // _UNIT_NANOS[self._type.unit]
        if step <= 0:
            # A frequency finer than the unit changes nothing.
            return self._wrap(self._values)
        ticks = self._local.cast(pa.int64())
        remainder = _remainder(ticks, step)
        floor = ticks - remainder
        if mode == "floor":
            ticks = floor
        elif mode == "ceil":
            ticks = floor + lit(step) * (remainder > lit(0)).cast(pa.int64())
        else:
            twice = remainder * lit(2)
            odd = _remainder(floor / lit(step), 2) == lit(1)
            up = (twice > lit(step)) | ((twice == lit(step)) & odd)
            ticks = floor + lit(step) * up.cast(pa.int64())
        local = ticks.cast(pa.timestamp(self._type.unit))
        return self._wrap(self._localized(local, self._type.tz))

    def _localized(self, local, tz):
        """A wall time as a timestamp in `tz` (None: as it is)."""
        if tz is None:
            return local
        return local.cast(pa.timestamp(self._type.unit, tz))

    # -- time zones ------------------------------------------------------------
    def tz_localize(self, tz, ambiguous="raise", nonexistent="raise"):
        """Naive timestamps as wall times in `tz`, or, with `tz=None`, these
        timestamps' wall times without their time zone.

        A wall time that is ambiguous or does not exist in `tz` fails, as in
        pandas by default.
        """
        _check_raise(ambiguous, nonexistent)
        if tz is None:
            return self._wrap(self._local)
        if self._type.tz is not None:
            raise TypeError("Already tz-aware, use tz_convert to convert.")
        return self._wrap(self._localized(self._values, _time_zone(tz)))

    def tz_convert(self, tz):
        """These timestamps in the time zone `tz` (or, with `tz=None`, in UTC
        without a time zone)."""
        if self._type.tz is None:
            raise TypeError(
                "Cannot convert tz-naive timestamps, use tz_localize to localize"
            )
        zone = None if tz is None else _time_zone(tz)
        return self._wrap(self._values.cast(pa.timestamp(self._type.unit, zone)))


class TimedeltaProperties:
    """The parts of a Series of durations, as pandas' `Series.dt` for
    timedeltas.

    **EXPERIMENTAL.** As in pandas, the days are rounded down, so the other
    parts of a negative duration are not negative (-1 second is -1 day and
    86399 seconds).
    """

    def __init__(self, series):
        self._series = series
        self._type = series._dtype()
        self._ticks = series._decoded().cast(pa.int64())
        self._per_second = 10**9 // _UNIT_NANOS[self._type.unit]

    def _wrap(self, expr):
        return Series(self._series._df, expr, self._series._name)

    @property
    def days(self):
        return self._wrap(_quotient(self._ticks, 86400 * self._per_second))

    @property
    def seconds(self):
        """The seconds past the days, from 0 to 86399."""
        past = _remainder(self._ticks, 86400 * self._per_second)
        return self._wrap(past / lit(self._per_second))

    def _subsecond(self):
        """The nanoseconds past the second."""
        past = _remainder(self._ticks, self._per_second)
        return past * lit(_UNIT_NANOS[self._type.unit])

    @property
    def microseconds(self):
        """The microseconds past the second, from 0 to 999999."""
        return self._wrap(self._subsecond() / lit(1000))

    @property
    def nanoseconds(self):
        """The nanoseconds past the microsecond, from 0 to 999."""
        return self._wrap(_remainder(self._subsecond(), 1000))

    def total_seconds(self):
        """Each duration in seconds, as a float."""
        return self._wrap(self._ticks.cast(pa.float64()) / lit(float(self._per_second)))

    @property
    def unit(self):
        return self._type.unit
