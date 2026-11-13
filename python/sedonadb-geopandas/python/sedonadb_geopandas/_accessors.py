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
from sedonadb_geopandas._temporal import _overflow_check

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


def _literal_template(text):
    """`text` as a replacement template of the engine's `regexp_replace()`
    that writes it as it is.

    The engine reads `$` as the start of a group (`$$` writes a `$`), and
    first reads one or two backslashes before digits as a group too (`\\1`
    as `${1}`): so each backslash is followed by `${}`, a group no pattern
    can have, which writes nothing.
    """
    return text.replace("$", "$$").replace("\\", "\\${}")


def _engine_groups(pattern):
    """The capture groups of `pattern`, of the engine's syntax, in order:
    each one's name, or None for a group without one; None for a pattern the
    engine refuses (unbalanced, or with lookaround).

    This reads the pattern as the engine does (Rust's `regex-syntax`), as far
    as groups go: escapes, character classes (nested ones, and a `]` first in
    one, included), and in verbose mode, which the `x` flag sets until the
    end of its group, whitespace and comments, which it skips in classes too.
    """
    verbose = [False]  # of each open group
    groups = []

    def skipped(i):
        # Past the whitespace (Unicode's: Python's but for the information
        # separators) and comments verbose mode skips.
        while verbose[-1] and i < len(pattern):
            if pattern[i] == "#":
                end = pattern.find("\n", i)
                i = len(pattern) if end < 0 else end + 1
            elif pattern[i].isspace() and pattern[i] not in "\x1c\x1d\x1e\x1f":
                i += 1
            else:
                break
        return i

    def item(i):
        # Past a character or an escape (what follows one, such as `{L}` of
        # `\\p{L}`, can't open or close a group or class).
        return i + 2 if pattern[i] == "\\" else i + 1

    def klass(i):
        # Past the class opening at i, or None if it is not closed.
        depth = 0
        opening = True
        while True:
            if opening:
                depth += 1
                i = skipped(i + 1)
                if pattern.startswith("^", i):
                    i = skipped(i + 1)
                literal_bracket = True
                while pattern.startswith("-", i):
                    i = skipped(i + 1)
                    literal_bracket = False
                if literal_bracket and pattern.startswith("]", i):
                    i += 1
                opening = False
            i = skipped(i)
            if i >= len(pattern):
                return None
            if pattern[i] == "[":
                # A nested class (or one such as `[:alpha:]`, alike here).
                opening = True
            elif pattern[i] == "]":
                depth -= 1
                i += 1
                if depth == 0:
                    return i
            elif pattern.startswith(("&&", "--", "~~"), i):
                i += 2
            else:
                i = item(i)
                # A range, whose end can be any character, `[` included.
                dash = skipped(i)
                if pattern.startswith("-", dash):
                    after = skipped(dash + 1)
                    if after < len(pattern) and pattern[after] not in "]-":
                        i = item(after)

    i = 0
    while True:
        i = skipped(i)
        if i >= len(pattern):
            break
        c = pattern[i]
        if c == "\\":
            i = item(i)
        elif c == "[":
            i = klass(i)
            if i is None:
                return None
        elif c == "(":
            i = skipped(i + 1)
            if pattern.startswith(("?=", "?!", "?<=", "?<!"), i):
                return None
            prefix = next((p for p in ("?P<", "?<") if pattern.startswith(p, i)), None)
            if prefix is not None:
                end = pattern.find(">", i)
                if end < 0:
                    return None
                groups.append(pattern[i + len(prefix) : end])
                verbose.append(verbose[-1])
                i = end + 1
            elif pattern.startswith("?", i):
                # Flags, until the end of this group or of a new one.
                end = i + 1
                on, negated = verbose[-1], False
                while end < len(pattern) and pattern[end] not in ":)":
                    if pattern[end] == "-":
                        negated = True
                    elif pattern[end] == "x":
                        on = not negated
                    end += 1
                if end == len(pattern):
                    return None
                if pattern[end] == ":":
                    verbose.append(on)
                else:
                    verbose[-1] = on
                i = end + 1
            else:
                groups.append(None)
                verbose.append(verbose[-1])
        elif c == ")":
            if len(verbose) == 1:
                return None
            verbose.pop()
            i += 1
        else:
            i += 1
    return groups if len(verbose) == 1 else None


def _check_groups(pattern, repl):
    """Python's error for a replacement template `repl` that refers to a
    group `pattern` (of the engine's syntax) does not have, as `re.sub()`
    (and pandas) raises: the engine would replace it with nothing.

    The groups are the engine's, which Python can't always read (`\\p{L}`)
    or reads differently (`[[]()]]` is a nested class there); Python checks
    the template, which it reads whole before looking for a match, against a
    pattern of the same groups.
    """
    groups = _engine_groups(pattern)
    if groups is None:
        # The engine refuses the pattern itself.
        return
    # A name Python can't have (the engine's can hold `.`, `[` and `]`) is
    # one no template can refer to, as Python reads templates.
    alike = "".join(
        f"(?P<{name}>)" if name is not None and name.isidentifier() else "()"
        for name in groups
    )
    try:
        compiled = re.compile(alike)
    except re.error:
        # A name used twice, which the engine refuses too.
        return
    compiled.sub(repl, "")


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
        if c != "\\":
            out.append(_literal_template(c))
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
            out.append(_literal_template(_TEMPLATE_ESCAPES[nxt]))
            i += 2
        elif nxt.isascii() and nxt.isalpha():
            raise re.error(f"bad escape \\{nxt}", repl, i)
        else:
            # Kept as written, backslash and all.
            out.append(_literal_template("\\" + nxt))
            i += 2
    return "".join(out)


# -- strings -----------------------------------------------------------------


class StringMethods:
    """String functions for a Series of strings, as pandas' `Series.str`.

    **EXPERIMENTAL.** Each method computes its result in the engine, as a
    Series of the same frame; a missing value stays missing, but for the
    boolean methods, which answer for it as pandas does (with pandas 3,
    False), and `cat()` with `na_rep`, which writes it. Lengths and
    positions count characters (code points), as Python does.

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
        # objects, keeps its type, though its values are read as strings
        # (but for cat() with na_rep, whose result is strings).
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
                # As pandas compares Python strings ignoring case: both in
                # upper case, by the full case mapping ("ß" contains "ss").
                upper = self._values.funcs.upper()
                expr = upper.funcs.contains(lit(pat.upper()))
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
            _check_groups(pattern, repl)
            template = _engine_template(repl)
        else:
            # A literal pattern: of the flags, only re.IGNORECASE changes what
            # one matches, as in pandas.
            pat, flags = _pattern_and_flags(pat, flags, case, "replace")
            ignore_case = bool(flags & re.IGNORECASE)
            # The engine's replace() leaves a value as it is for an empty
            # pattern, as PostgreSQL does; Python's puts `repl` around every
            # character, as the regular expression does.
            if not ignore_case and n < 0 and pat:
                return self._text(self._values.funcs.replace(lit(pat), lit(repl)))
            pattern = ("(?i)" if ignore_case else "") + _escaped(pat)
            template = _literal_template(repl)
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
        # A negative width pads nothing, as in Python; the engine's lpad() and
        # rpad() panic at a negative length (a missing value's, here).
        width = max(int(width), 0)
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
        joined = self._ctx.lit(sep).funcs.concat_ws(*pieces)
        if na_rep is None:
            # The engine's concat_ws() skips a missing piece; pandas makes the
            # result missing.
            present = pieces[0].is_not_null()
            for piece in pieces[1:]:
                present = present & piece.is_not_null()
            joined = _choose(present, joined, lit(pa.scalar(None, pa.string())))
        elif pa.types.is_null(self._type):
            # Strings, written with na_rep, though this column has none.
            return self._wrap(joined)
        return self._text(joined)


# -- dates and times -----------------------------------------------------------

_UNIT_NANOS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}

# The last tick of a timestamp, either way: int64's least is pandas' NaT.
_MAX_TICKS = 2**63 - 1

# The first and last second the engine reads a date of, from 1970:
# -262143-01-01 00:00:00 and 262142-12-31 23:59:59. Seconds, milliseconds and
# microseconds reach past them; nanoseconds don't.
_FIRST_SECOND = -8_334_601_228_800
_LAST_SECOND = 8_210_266_876_799
_PAST_DATES = "a timestamp is past the dates the engine reads (years -262143 to 262142)"


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
    `//`: the engine's quotient, rounded toward zero, less one where that
    leaves a negative remainder, so that nothing past int64 is computed."""
    truncated = value / lit(n)
    below = (value - truncated * lit(n)) < lit(0)
    return truncated - below.cast(pa.int64())


def _step(freq, unit):
    """The fixed frequency `freq` as the number of `unit`s to round to, or 0
    if it divides the unit (rounding to it changes nothing), as pandas 3
    reads it.

    A frequency that is neither a multiple of the unit nor divides it is an
    error (pandas 2 rounds to a whole number of units instead), as is a
    negative one (pandas rounds the other way for it).
    """
    from pandas.tseries.frequencies import to_offset

    offset = to_offset(freq)
    # A ValueError for a frequency that is not fixed, as in pandas.
    nanos = offset.nanos
    per_tick = _UNIT_NANOS[unit]
    if nanos < 0:
        raise ValueError(f"freq={offset!r} is negative: rounding needs a positive one")
    if nanos % per_tick == 0:
        step = nanos // per_tick
    elif per_tick % nanos == 0:
        step = 0
    else:
        raise ValueError(
            f"freq={offset!r} is incompatible with unit={unit}. Use a lower freq "
            f"or a higher unit instead."
        )
    if step > _MAX_TICKS:
        raise OverflowError(f"Cannot cast {offset!r} to unit={unit} without overflow.")
    return step


def _check_raise(ambiguous, nonexistent):
    """pandas' `ambiguous` and `nonexistent`, as the engine localizes a wall
    time: an ambiguous or nonexistent one fails."""
    if not (isinstance(ambiguous, str) and ambiguous == "raise"):
        raise NotImplementedError(f"ambiguous={ambiguous!r} is not supported")
    if not (isinstance(nonexistent, str) and nonexistent == "raise"):
        raise NotImplementedError(f"nonexistent={nonexistent!r} is not supported")


def _time_zone(tz):
    """`tz`, a name or a tzinfo, as the name an Arrow timestamp type holds,
    read as pandas reads it ("utc" is UTC), with pandas' error for a zone it
    doesn't know (pyarrow's, without pytz, names no zone)."""
    import pandas as pd

    return pa.lib.tzinfo_to_string(pd.DatetimeTZDtype("ns", tz).tz)


class DatetimeProperties:
    """Date and time properties of a Series of timestamps, as pandas'
    `Series.dt`.

    **EXPERIMENTAL.** Each is computed in the engine, as a Series of the same
    frame. For timestamps with a time zone, properties and rounding are of
    the wall time there, as in pandas; a missing timestamp gives a missing
    value, but for the `is_*` tests, which answer False for it. A timestamp
    past the years the engine reads dates of, -262143 to 262142, raises for
    its fields, names, date and time, and for its wall time where it has a
    time zone; a naive one can still be rounded.
    """

    def __init__(self, series):
        self._series = series
        self._type = series._dtype()
        self._values = series._decoded()
        # The wall time: naive timestamps already are. `_dated` is the wall
        # time for what reads its date, which would be missing past the
        # engine's dates; rounding takes `_local`, whose many uses would each
        # repeat the check.
        dated = self._within_dates(self._values)
        if self._type.tz is None:
            self._local = self._values
            self._dated = dated
        else:
            self._local = self._values.funcs.to_local_time()
            self._dated = dated.funcs.to_local_time()

    def _within_dates(self, values):
        """`values`, raising for one past the dates the engine reads."""
        per_second = 10**9 // _UNIT_NANOS[self._type.unit]
        first = _FIRST_SECOND * per_second
        last = (_LAST_SECOND + 1) * per_second - 1
        if first < -_MAX_TICKS and last > _MAX_TICKS:
            return values
        # Within a day of either end, a time zone can still move the wall
        # time past it, which fails with the engine's own error.
        ticks = values.cast(pa.int64())
        in_range = (ticks >= lit(first)) & (ticks <= lit(last))
        return (ticks + _overflow_check(in_range, _PAST_DATES)).cast(self._type)

    @property
    def _ctx(self):
        return self._series._df._ctx

    def _wrap(self, expr):
        return Series(self._series._df, expr, self._series._name)

    def _part(self, part):
        """A field of the wall time, as the engine's date_part() reads it."""
        return self._ctx.lit(part).funcs.date_part(self._dated)

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
        return self._wrap(self._dated.funcs.to_char(lit("%A")))

    def month_name(self, locale=None):
        """The English name of the month."""
        if locale is not None:
            raise NotImplementedError("month_name() supports the English names only")
        return self._wrap(self._dated.funcs.to_char(lit("%B")))

    @property
    def date(self):
        """The date of the wall time, as Python dates."""
        return self._wrap(self._dated.cast(pa.date32()))

    @property
    def time(self):
        """The time of day of the wall time, as Python times: to the
        microsecond, as Python's are (pandas drops the nanoseconds too), where
        a time of nanoseconds would not convert to one."""
        return self._wrap(self._dated.cast(pa.time64("us")))

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
        _check_raise(ambiguous, nonexistent)
        unit = self._type.unit
        step = _step(freq, unit)
        if step == 0:
            # A frequency that divides the unit changes nothing.
            return self._wrap(self._values)
        ticks = self._local.cast(pa.int64())
        remainder = _remainder(ticks, step)
        # How far each value moves: from -step to step, where the moved value
        # can be past int64.
        if mode == "floor":
            delta = 0 - remainder
        elif mode == "ceil":
            delta = (lit(step) - remainder) * (remainder > lit(0)).cast(pa.int64())
        else:
            rest = lit(step) - remainder
            odd = _remainder(_quotient(ticks, step), 2) == lit(1)
            up = (remainder > rest) | ((remainder == rest) & odd)
            delta = lit(step) * up.cast(pa.int64()) - remainder
        # An error past the last timestamp either way, as pandas raises,
        # where the sum would wrap around to the other end.
        in_range = (ticks <= lit(_MAX_TICKS) - delta.funcs.greatest(lit(0))) & (
            ticks >= lit(-_MAX_TICKS) - delta.funcs.least(lit(0))
        )
        check = _overflow_check(
            in_range, f"rounding to {freq} overflowed the timestamps"
        )
        local = (ticks + delta + check).cast(pa.timestamp(unit))
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
        return self._wrap(self._localized(self._dated, _time_zone(tz)))

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
        return self._wrap((past / lit(self._per_second)).cast(pa.int32()))

    def _subsecond(self):
        """The nanoseconds past the second."""
        past = _remainder(self._ticks, self._per_second)
        return past * lit(_UNIT_NANOS[self._type.unit])

    @property
    def microseconds(self):
        """The microseconds past the second, from 0 to 999999."""
        return self._wrap((self._subsecond() / lit(1000)).cast(pa.int32()))

    @property
    def nanoseconds(self):
        """The nanoseconds past the microsecond, from 0 to 999."""
        return self._wrap(_remainder(self._subsecond(), 1000).cast(pa.int32()))

    def total_seconds(self):
        """Each duration in seconds, as a float."""
        return self._wrap(self._ticks.cast(pa.float64()) / lit(float(self._per_second)))

    @property
    def unit(self):
        return self._type.unit
