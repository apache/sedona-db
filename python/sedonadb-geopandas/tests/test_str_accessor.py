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
"""The `.str` accessor, compared against pandas."""

import re
import warnings

import pandas as pd
import pyarrow as pa
import pytest

import sedonadb_geopandas as sgpd

# Missing and empty values, non-ASCII text, signs, regular-expression
# characters, and a newline.
STRINGS = [
    "Ab c",
    None,
    "",
    "aXa",
    "  padded\t",
    "héllo wörld",
    "-12",
    "+7",
    "a.b*c$",
    "line\nend",
    "xyzzy",
]

TYPES = {
    "string": pa.string(),
    "large_string": pa.large_string(),
    "string_view": pa.string_view(),
}


def _series(values, type=pa.string()):
    table = pa.table({"s": pa.array(values, type)})
    return sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))["s"]


def _same(got, expected):
    # Values, and missing values whichever their marker.
    got, expected = list(got), list(expected)
    if len(got) != len(expected):
        return False
    for a, b in zip(got, expected):
        if pd.isna(a) or pd.isna(b):
            if not (pd.isna(a) and pd.isna(b)):
                return False
        elif a != b:
            return False
    return True


def _check(ours, theirs):
    got = ours.to_pandas()
    assert _same(got, theirs), (got.tolist(), theirs.tolist())
    assert got.dtype == theirs.dtype, (got.dtype, theirs.dtype)
    assert got.name == theirs.name


def _compare(method, *args, values=STRINGS, type=pa.string(), **kwargs):
    series = _series(values, type)
    ours = getattr(series.str, method)(*args, **kwargs)
    theirs = getattr(series.to_pandas().str, method)(*args, **kwargs)
    _check(ours, theirs)


CASES = [
    ("len", (), {}),
    ("lower", (), {}),
    ("upper", (), {}),
    ("strip", (), {}),
    ("lstrip", (), {}),
    ("rstrip", (), {}),
    ("strip", ("a",), {}),
    ("lstrip", (" Ab",), {}),
    ("rstrip", ("cz",), {}),
    ("strip", ("",), {}),
    ("startswith", ("a",), {}),
    ("endswith", ("c",), {}),
    ("startswith", (("A", "+", "-"),), {}),
    ("endswith", (("$", "y"),), {}),
    ("startswith", ((),), {}),
    ("startswith", ("a",), {"na": False}),
    ("endswith", ("a",), {"na": True}),
    ("contains", ("a",), {}),
    ("contains", ("A",), {"case": False}),
    ("contains", ("l+",), {}),
    ("contains", (r"\d",), {}),
    ("contains", (".",), {"regex": False}),
    ("contains", ("*c",), {"regex": False}),
    ("contains", ("AB",), {"regex": False, "case": False}),
    ("contains", ("a",), {"na": False}),
    ("contains", ("a",), {"na": True}),
    ("contains", ("^line$",), {"flags": re.MULTILINE}),
    ("contains", ("e.e",), {"flags": re.DOTALL}),
    ("contains", (r"c\Z",), {}),
    ("match", ("a",), {}),
    ("match", ("[a-z]+",), {}),
    ("match", ("x",), {"case": False}),
    ("match", ("end",), {"flags": re.MULTILINE}),
    ("match", ("a",), {"na": False}),
    ("fullmatch", ("a.a",), {}),
    ("fullmatch", (r"[+-]\d+",), {}),
    ("fullmatch", ("xYZZY",), {"case": False}),
    ("fullmatch", ("a|aXa",), {}),
    ("count", ("a",), {}),
    ("count", ("[a-c]",), {}),
    ("count", ("A",), {"flags": re.IGNORECASE}),
    ("find", ("a",), {}),
    ("find", ("",), {}),
    ("find", ("l",), {}),
    ("rfind", ("a",), {}),
    ("rfind", ("",), {}),
    ("rfind", ("zy",), {}),
    ("replace", ("a", "zz"), {}),
    ("replace", (".", "!"), {}),
    ("replace", ("", "-"), {}),
    ("replace", ("a", "$0"), {}),
    ("replace", ("a", r"\1"), {}),
    ("replace", ("a", "z"), {"n": 1}),
    ("replace", ("a", "z"), {"n": 0}),
    ("replace", ("A", "z"), {"case": False}),
    ("replace", ("[ab]", "z"), {"regex": True}),
    ("replace", ("(a)(.)", r"\2\1"), {"regex": True}),
    ("replace", ("(?P<first>a)", r"<\g<first>>"), {"regex": True}),
    ("replace", ("(a)", r"\g<1>$"), {"regex": True}),
    ("replace", ("a", "z"), {"regex": True, "n": 1}),
    ("replace", ("", "-"), {"regex": True}),
    ("replace", ("A", "_"), {"regex": True, "flags": re.IGNORECASE}),
    ("slice", (), {}),
    ("slice", (1,), {}),
    ("slice", (1, 3), {}),
    ("slice", (-3,), {}),
    ("slice", (None, -1), {}),
    ("slice", (-4, -1), {}),
    ("slice", (2, 1), {}),
    ("slice", (-100, 100), {}),
    ("slice", (None, None, -1), {}),
    ("get", (0,), {}),
    ("get", (1,), {}),
    ("get", (-1,), {}),
    ("get", (-3,), {}),
    ("pad", (6,), {}),
    ("pad", (6,), {"side": "right", "fillchar": "*"}),
    ("pad", (7,), {"side": "both", "fillchar": "-"}),
    ("pad", (0,), {}),
    ("ljust", (5, "."), {}),
    ("rjust", (5,), {}),
    ("center", (5, "*"), {}),
    ("center", (6, "*"), {}),
    ("center", (-1,), {}),
    ("zfill", (4,), {}),
    ("zfill", (0,), {}),
    ("repeat", (2,), {}),
    ("repeat", (0,), {}),
    ("removeprefix", ("a",), {}),
    ("removeprefix", ("",), {}),
    ("removeprefix", ("aXa",), {}),
    ("removesuffix", ("a",), {}),
    ("removesuffix", ("d",), {}),
]


@pytest.mark.parametrize(
    ("method", "args", "kwargs"),
    CASES,
    ids=[f"{m}{args}{kwargs}" for m, args, kwargs in CASES],
)
def test_string_methods_match_pandas(method, args, kwargs):
    _compare(method, *args, **kwargs)


@pytest.mark.parametrize("type", TYPES.values(), ids=TYPES.keys())
@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("len", ()),
        ("upper", ()),
        ("contains", ("a",)),
        ("replace", ("a", "z")),
        ("slice", (1, 3)),
        ("get", (-1,)),
        ("zfill", (4,)),
        ("removesuffix", ("a",)),
    ],
)
def test_string_types(method, args, type):
    # A string result has the column's own string type.
    _compare(method, *args, type=type)


def test_dictionary_encoded_strings():
    # pandas reads a dictionary-encoded column as a categorical; its string
    # methods answer as they do for the categories' values.
    values = pa.array(STRINGS).dictionary_encode()
    series = _series(values, values.type)
    for method, args in [("lower", ()), ("len", ()), ("contains", ("a",))]:
        ours = getattr(series.str, method)(*args).to_pandas()
        theirs = getattr(series.to_pandas().str, method)(*args)
        assert _same(ours, theirs), (ours.tolist(), theirs.tolist())


def test_strip_python_whitespace():
    # Python's whitespace, which is more than the engine's: the information
    # separators \x1c-\x1f too. Characters that are not whitespace stay.
    spaces = [chr(c) for c in range(0x3001) if chr(c).isspace()]
    others = ["᠎", "​", "﻿", "x"]
    values = [f"{c}a{c}" for c in spaces + others] + ["".join(spaces) + "a"]
    for method in ("strip", "lstrip", "rstrip"):
        _compare(method, values=values)


def test_case_and_repeats_as_python():
    # pandas 3's pyarrow-backed strings lower-case a final sigma as any other
    # and refuse a negative count, where Python does not.
    values = ["ΟΣ", "ΟΣΑ", "AΣ A", "İ", "ẞ", ""]
    series = _series(values)
    assert series.str.lower().to_pandas().tolist() == [v.lower() for v in values]
    assert series.str.repeat(-1).to_pandas().tolist() == [v * -1 for v in values]


def test_count_of_empty_matches():
    # An empty pattern matches before each character and at the end, as in
    # Python's re; pandas 3 counts the bytes of non-ASCII text instead.
    series = _series(STRINGS)
    expected = [None if v is None else len(re.findall("", v)) for v in STRINGS]
    assert _same(series.str.count("").to_pandas(), expected)


@pytest.mark.parametrize(
    "repl",
    [
        r"\$",
        r"\\",
        r"x\-y",
        r"\n",
        "$1",
        r"\1\1",
        r"[\g<0>]",
        r"\10",
        # A backslash before a digit, and before a group.
        r"\\1",
        r"\\10",
        r"\\\1",
        r"\\\\1",
        r"\\\g<1>",
    ],
)
def test_replacement_templates_as_python(repl):
    # As Python's re.sub() reads a template: pandas 3 hands some of these to
    # pyarrow, which refuses them.
    values = ["a-b", None, "xaay", "abcdefghijk"]
    pattern = "(a)(b)?(c)?(d)?(e)?(f)?(g)?(h)?(i)?(j)?"
    series = _series(values).str.replace(pattern, repl, regex=True)
    expected = [None if v is None else re.sub(pattern, repl, v) for v in values]
    assert _same(series.to_pandas(), expected), series.to_pandas().tolist()


@pytest.mark.parametrize("repl", [r"\1", r"\\2", r"x\\", "$1", r"\g<0>"])
@pytest.mark.parametrize(("n", "case"), [(-1, True), (1, True), (-1, False)])
def test_replacements_of_text_as_written(repl, n, case):
    # Without a regular expression, the replacement is the text itself,
    # backslashes and all, however the engine replaces it. (pandas reads it
    # as a template when ignoring case, where it uses re.sub().)
    values = ["a", None, "bAa", ""]
    series = _series(values).str.replace("a", repl, n=n, case=case, regex=False)
    pattern = re.compile("a", 0 if case else re.IGNORECASE)
    expected = [
        None if v is None else pattern.sub(lambda m: repl, v, count=max(n, 0))
        for v in values
    ]
    assert _same(series.to_pandas(), expected), series.to_pandas().tolist()


@pytest.mark.parametrize("repl", [r"\2", r"\g<2>", r"\g<missing>", r"x\g<1>\3"])
def test_replacement_groups_the_pattern_lacks(repl):
    # Python's error, where the engine would replace the group with nothing.
    # pandas raises too (pyarrow's error in pandas 3, for some of these).
    series = _series(STRINGS)
    with pytest.raises((re.error, IndexError)) as python:
        re.sub("(a)", repl, "a")
    with pytest.raises(type(python.value), match=re.escape(str(python.value))):
        series.str.replace("(a)", repl, regex=True)
    with pytest.raises(Exception):
        series.to_pandas().str.replace("(a)", repl, regex=True)


def test_compiled_patterns():
    # A compiled pattern's flags hold, as in Python's re (pandas 2.3.2 drops
    # them for match() and fullmatch()).
    values = [v for v in STRINGS if v is not None]
    series = _series(values)
    pattern = re.compile("a", re.IGNORECASE)
    for method, expected in [
        ("contains", [pattern.search(v) is not None for v in values]),
        ("match", [pattern.match(v) is not None for v in values]),
        ("fullmatch", [pattern.fullmatch(v) is not None for v in values]),
        ("count", [len(pattern.findall(v)) for v in values]),
    ]:
        got = getattr(series.str, method)(pattern).to_pandas()
        assert got.tolist() == expected, (method, got.tolist())
    pairs = re.compile("x(.)", re.IGNORECASE)
    got = series.str.replace(pairs, r"<\1>", regex=True).to_pandas()
    assert got.tolist() == [pairs.sub(r"<\1>", v) for v in values]


def test_contains_text_ignoring_case():
    # As pandas compares Python strings: both in upper case, so "ß" contains
    # "ss" (pandas 3's pyarrow-backed strings fold one character at a time).
    values = ["ß", "Straße", "ẞ", "ss", "SS", "İ", "ﬃ", "ΣΑΣ", "σας", "a.b"]
    series = _series(values)
    for pat in ["ss", "SS", "ß", "i", "ffi", "σ", "ς", ".", ""]:
        got = series.str.contains(pat, case=False, regex=False).to_pandas()
        expected = [pat.upper() in v.upper() for v in values]
        assert got.tolist() == expected, (pat, got.tolist())


def test_slicing_with_brackets():
    series = _series(STRINGS)
    expected = series.to_pandas()
    _check(series.str[1:3], expected.str[1:3])
    _check(series.str[-1], expected.str[-1])
    _check(series.str[::-1], expected.str[::-1])


def test_slices_as_python_slices():
    # Every start and stop, both signs, beyond the ends too.
    values = ["", "a", "ab", "abcdef", "héllo"]
    bounds = [None, -7, -3, -1, 0, 1, 2, 6, 9]
    for start in bounds:
        for stop in bounds:
            _compare("slice", start, stop, values=values)


def test_padding_widths():
    values = ["", "a", "ab", "abc", "-1", "+12", "-"]
    for width in range(0, 8):
        for method in ("rjust", "ljust", "center", "zfill"):
            _compare(method, width, values=values)


def test_cat_with_other_columns():
    table = pa.table({"a": ["x", None, "z", ""], "b": ["1", "2", None, ""]})
    gdf = sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))
    frame = gdf.to_pandas()
    _check(gdf["a"].str.cat(gdf["b"]), frame["a"].str.cat(frame["b"]))
    _check(gdf["a"].str.cat(gdf["b"], sep="-"), frame["a"].str.cat(frame["b"], sep="-"))
    _check(
        gdf["a"].str.cat([gdf["b"], gdf["a"]], sep="/", na_rep="?"),
        frame["a"].str.cat([frame["b"], frame["a"]], sep="/", na_rep="?"),
    )


def test_cat_with_a_column_of_missing_values():
    # Of the null type, on either side: written as na_rep, if given.
    table = pa.table({"a": pa.array([None, None], pa.null()), "b": ["x", None]})
    gdf = sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))
    frame = gdf.to_pandas()
    for left, right in [("a", "b"), ("b", "a")]:
        for na_rep in (None, "?"):
            ours = gdf[left].str.cat(gdf[right], sep="-", na_rep=na_rep).to_pandas()
            theirs = frame[left].str.cat(frame[right], sep="-", na_rep=na_rep)
            assert _same(ours, theirs), (left, na_rep, ours.tolist(), theirs.tolist())


def test_cat_needs_columns_of_this_frame():
    series = _series(STRINGS)
    with pytest.raises(NotImplementedError, match="row order"):
        series.str.cat()
    with pytest.raises(ValueError, match="different DataFrames"):
        series.str.cat(_series(STRINGS))
    with pytest.raises(TypeError, match="row alignment"):
        series.str.cat(["a"] * len(STRINGS))
    numbers = sgpd.GeoDataFrame(
        sgpd.default_context().create_data_frame(
            pa.table({"s": STRINGS, "i": range(11)})
        )
    )
    with pytest.raises(TypeError, match="strings"):
        numbers["s"].str.cat(numbers["i"])


def test_methods_chain_and_mask():
    gdf = sgpd.GeoDataFrame(
        sgpd.default_context().create_data_frame(
            pa.table({"s": STRINGS, "i": list(range(len(STRINGS)))})
        )
    )
    frame = gdf.to_pandas()
    _check(gdf["s"].str.strip().str.len(), frame["s"].str.strip().str.len())
    ours = gdf[gdf["s"].str.contains("a", na=False)].to_pandas()
    theirs = frame[frame["s"].str.contains("a", na=False)]
    assert ours["i"].tolist() == theirs["i"].tolist()


def test_only_for_strings():
    table = pa.table({"i": [1, 2]})
    gdf = sgpd.GeoDataFrame(sgpd.default_context().create_data_frame(table))
    with pytest.raises(AttributeError, match="string values"):
        gdf["i"].str
    assert not hasattr(gdf["i"], "str")
    points = sgpd.GeoDataFrame(
        sgpd.default_context().sql("SELECT ST_Point(0, 1) AS geometry")
    )
    with pytest.raises(AttributeError, match="string values"):
        points.geometry.str


def test_a_column_of_missing_values():
    # pandas reads a column of only missing values as strings: so does this.
    _compare("upper", values=[None, None], type=pa.null())
    _compare("contains", "a", values=[None, None], type=pa.null())


def test_unsupported_arguments():
    series = _series(STRINGS)
    with pytest.raises(NotImplementedError, match="start"):
        series.str.find("a", 1)
    with pytest.raises(NotImplementedError, match="step"):
        series.str.slice(0, 4, 2)
    with pytest.raises(NotImplementedError, match="callable"):
        series.str.replace("a", lambda m: "z", regex=True)
    with pytest.raises(NotImplementedError, match="n="):
        series.str.replace("a", "z", n=2)
    with pytest.raises(NotImplementedError, match="ASCII"):
        series.str.contains("a", flags=re.ASCII)
    # pandas 3 refuses such an `na` itself; pandas 2 would make the column
    # one of Python objects (and warns that it will refuse it).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises((TypeError, ValueError), match="na"):
            series.str.contains("a", na="x")


def test_argument_errors_as_pandas():
    series = _series(STRINGS)
    expected = series.to_pandas()
    for call in (
        lambda s: s.str.startswith(1),
        lambda s: s.str.pad(3.5),
        lambda s: s.str.pad(3, fillchar="ab"),
        lambda s: s.str.zfill("3"),
        lambda s: s.str.replace("a", 1),
    ):
        with pytest.raises(TypeError):
            call(expected)
        with pytest.raises(TypeError):
            call(series)
    with pytest.raises(ValueError):
        expected.str.pad(3, side="middle")
    with pytest.raises(ValueError):
        series.str.pad(3, side="middle")
    with pytest.raises(ValueError):
        series.str.replace(re.compile("a"), "z", regex=False)
