import pytest
from sedonadb.testing import geom_or_null, PostGIS, SedonaDB, val_or_null


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", 0.0),
        ("LINESTRING EMPTY", 0.0),
        ("POLYGON EMPTY", 0.0),
        ("MULTIPOINT EMPTY", 0.0),
        ("MULTILINESTRING EMPTY", 0.0),
        ("MULTIPOLYGON EMPTY", 0.0),
        ("GEOMETRYCOLLECTION EMPTY", 0.0),
        ("POINT (5 2)", 0.0),
        ("MULTIPOINT ((0 0), (1 1))", 0.0),
        ("LINESTRING (0 0, 1 1)", 0.0),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", 0.0),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 1.0),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            2.0,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1), GEOMETRYCOLLECTION (POLYGON ((0 0, -1 0, -1 -1, 0 -1, 0 0))))",
            2.0,
        ),
    ],
)
def test_st_area(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Area({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        ("POINT (1 1)", b"0101000000000000000000F03F000000000000F03F"),
        ("POINT EMPTY", b"0101000000000000000000F87F000000000000F87F"),
        (
            "LINESTRING (0 0, 1 2, 3 4)",
            b"01020000000300000000000000000000000000000000000000000000000000F03F000000000000004000000000000008400000000000001040",
        ),
        ("LINESTRING EMPTY", b"010200000000000000"),
        (
            "POINT ZM (0 0 0 0)",
            b"01B90B00000000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            b"0107000000020000000101000000000000000000000000000000000000000103000000010000000500000000000000000000000000000000000000000000000000F03F0000000000000000000000000000F03F000000000000F03F0000000000000000000000000000F03F00000000000000000000000000000000",
        ),
    ],
)
def test_st_asbinary(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_AsBinary({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        # Arrow-rs returns POINT (nan nan) instead of POINT EMPTY
        ("POINT EMPTY", "POINT (nan nan)"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("POLYGON EMPTY", "POLYGON EMPTY"),
        ("MULTIPOINT EMPTY", "MULTIPOINT EMPTY"),
        ("MULTILINESTRING EMPTY", "MULTILINESTRING EMPTY"),
        ("MULTIPOLYGON EMPTY", "MULTIPOLYGON EMPTY"),
        ("GEOMETRYCOLLECTION EMPTY", "GEOMETRYCOLLECTION EMPTY"),
        ("POINT (1 1)", "POINT (1 1)"),
        ("LINESTRING (0 0, 1 1)", "LINESTRING (0 0, 1 1)"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
        ("MULTIPOINT ((0 0), (1 1))", "MULTIPOINT (0 0, 1 1)"),
        (
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
        ),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1), GEOMETRYCOLLECTION (POLYGON ((0 0, -1 0, -1 -1, 0 -1, 0 0))))",
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1), GEOMETRYCOLLECTION (POLYGON ((0 0, -1 0, -1 -1, 0 -1, 0 0))))",
        ),
        ("POINT Z (0 0 0)", "POINT Z (0 0 0)"),
        ("POINT ZM (0 0 0 0)", "POINT ZM (0 0 0 0)"),
    ],
)
def test_st_astext(eng, geom, expected):
    eng = eng.create_or_skip()
    # Engines vary on where they place spaces in the text, so we convert back to geometry to compare
    eng.assert_query_result(
        f"SELECT ST_GeomFromText(ST_AsText({geom_or_null(geom)}))", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "dist", "expected_area"),
    [
        (None, None, None),
        (None, 1.0, None),
        ("POINT (1 1)", None, None),
        ("POINT (1 1)", 0.0, 0),
        ("POINT EMPTY", 1.0, 0),
        ("LINESTRING EMPTY", 1.0, 0),
        ("POLYGON EMPTY", 1.0, 0),
        ("POINT (0 0)", 1.0, 3.121445152258052),
        ("POINT (0 0)", 2.0, 12.485780609032208),
        ("LINESTRING (0 0, 1 1)", 1.0, 5.949872277004242),
        ("LINESTRING (0 0, 1 1)", 2.0, 18.14263485852459),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 2.0, 21.48578060903221),
        ("MULTIPOINT ((0 0), (1 1))", 1.0, 5.682167728387077),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            1.0,
            8.121445152256216,
        ),
    ],
)
def test_st_buffer(eng, geom, dist, expected_area):
    eng = eng.create_or_skip()

    eng.assert_query_result(
        f"SELECT ST_Area(ST_Buffer({geom_or_null(geom)}, {val_or_null(dist)}))",
        expected_area,
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT (0 0)", "POINT (0 0)"),
        ("LINESTRING (0 0, 1 1)", "POINT (0.5 0.5)"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POINT (0.5 0.5)"),
        ("MULTIPOINT ((0 0), (1 1))", "POINT (0.5 0.5)"),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", "POINT (1 1)"),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "POINT (0.5 0.5)",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "POINT (0.5 0.5)",
        ),
        # Failing: issue with testing code: geoarrow-c rendering POINT (nan, nan)
        # instead of POINT EMPTY
        # https://github.com/geoarrow/geoarrow-c/issues/143
        ("POINT EMPTY", "POINT (nan nan)"),
        ("LINESTRING EMPTY", "POINT (nan nan)"),
        ("POLYGON EMPTY", "POINT (nan nan)"),
        ("MULTIPOINT EMPTY", "POINT (nan nan)"),
        ("MULTILINESTRING EMPTY", "POINT (nan nan)"),
        ("MULTIPOLYGON EMPTY", "POINT (nan nan)"),
        ("GEOMETRYCOLLECTION EMPTY", "POINT (nan nan)"),
    ],
)
def test_st_centroid(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Centroid({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", 0),
        ("LINESTRING EMPTY", 1),
        ("POLYGON EMPTY", 2),
        ("MULTIPOINT EMPTY", 0),
        ("MULTILINESTRING EMPTY", 1),
        ("MULTIPOLYGON EMPTY", 2),
        ("GEOMETRYCOLLECTION EMPTY", 0),
        ("POINT (0 0)", 0),
        ("LINESTRING (0 0, 1 1)", 1),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 2),
        ("MULTIPOINT ((0 0), (1 1))", 0),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", 1),
        ("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))", 2),
        # Ensure GeometryCollections with different nested geometries are handled correctly
        ("GEOMETRYCOLLECTION (POINT (0 0))", 0),
        ("GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))", 1),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            2,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1)))",
            1,
        ),
        ("POINT Z (0 0 0)", 0),
        ("POINT ZM (0 0 0 0)", 0),
    ],
)
def test_st_dimension(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Dimension({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        # Failing: issue with testing code: geoarrow-c rendering POINT (nan, nan)
        # instead of POINT EMPTY
        # https://github.com/geoarrow/geoarrow-c/issues/143
        ("POINT EMPTY", "POINT (nan nan)"),
        ("POLYGON EMPTY", "POLYGON EMPTY"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("MULTIPOINT EMPTY", "MULTIPOINT EMPTY"),
        ("MULTILINESTRING EMPTY", "MULTILINESTRING EMPTY"),
        ("MULTIPOLYGON EMPTY", "MULTIPOLYGON EMPTY"),
        ("GEOMETRYCOLLECTION EMPTY", "GEOMETRYCOLLECTION EMPTY"),
        ("POINT (0 0)", "POINT (0 0)"),
        ("LINESTRING (0 0, 1 1)", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
        ("LINESTRING (0 0, 0 1)", "LINESTRING (0 0, 0 1)"),
        ("MULTIPOINT ((0 0), (1 1))", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
            "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
        ),
    ],
)
def test_st_envelope(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Envelope({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", "ST_Point"),
        ("POLYGON EMPTY", "ST_Polygon"),
        ("LINESTRING EMPTY", "ST_LineString"),
        ("MULTIPOINT EMPTY", "ST_MultiPoint"),
        ("MULTILINESTRING EMPTY", "ST_MultiLineString"),
        ("MULTIPOLYGON EMPTY", "ST_MultiPolygon"),
        ("GEOMETRYCOLLECTION EMPTY", "ST_GeometryCollection"),
        ("POINT (0 0)", "ST_Point"),
        ("LINESTRING (0 0, 1 1)", "ST_LineString"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "ST_Polygon"),
        ("MULTIPOINT ((0 0), (1 1))", "ST_MultiPoint"),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", "ST_MultiLineString"),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            "ST_MultiPolygon",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
            "ST_GeometryCollection",
        ),
    ],
)
def test_st_geometrytype(eng, geom, expected):
    if eng == PostGIS and geom is None:
        # PostGIS ST_GeometryType doesn't work unless we explicitly cast to null
        arg = "NULL::geometry"
    else:
        arg = geom_or_null(geom)

    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_GeometryType({arg})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("wkt", "expected"),
    [
        (None, None),
        ("POINT (0 0)", "POINT (0 0)"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("MULTIPOINT ((0 0), (1 1))", "MULTIPOINT (0 0, 1 1)"),
        ("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
    ],
)
def test_st_geogfromtext(eng, wkt, expected):
    if wkt is not None:
        wkt = f"'{wkt}'"
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_GeogFromText({val_or_null(wkt)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("wkt", "expected"),
    [
        (None, None),
        ("POINT (0 0)", "POINT (0 0)"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("MULTIPOINT ((0 0), (1 1))", "MULTIPOINT (0 0, 1 1)"),
        ("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
    ],
)
def test_st_geomfromtext(eng, wkt, expected):
    if wkt is not None:
        wkt = f"'{wkt}'"
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_GeomFromText({val_or_null(wkt)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        ("POINT (1 1)", "POINT (1 1)"),
        ("POINT EMPTY", "POINT (nan nan)"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("POLYGON EMPTY", "POLYGON EMPTY"),
        ("GEOMETRYCOLLECTION EMPTY", "GEOMETRYCOLLECTION EMPTY"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
        (
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
        ),
    ],
)
def test_st_geogfromwkb(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_GeogFromWKB(ST_AsBinary({geom_or_null(geom)}))", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        ("POINT (1 1)", "POINT (1 1)"),
        ("POINT EMPTY", "POINT (nan nan)"),
        ("LINESTRING EMPTY", "LINESTRING EMPTY"),
        ("POLYGON EMPTY", "POLYGON EMPTY"),
        ("GEOMETRYCOLLECTION EMPTY", "GEOMETRYCOLLECTION EMPTY"),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
        (
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
            "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))",
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))",
        ),
    ],
)
def test_st_geomfromwkb(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_GeomFromWKB(ST_AsBinary({geom_or_null(geom)}))", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", False),
        ("POINT Z EMPTY", True),
        ("POINT M EMPTY", False),
        ("POINT ZM EMPTY", True),
        ("POINT Z (0 0 0)", True),
        ("POINT M (0 0 0)", False),
        ("POINT ZM (0 0 0 0)", True),
        ("LINESTRING EMPTY", False),
        ("LINESTRING Z EMPTY", True),
        ("LINESTRING Z (0 0 0, 1 1 1)", True),
        ("POLYGON EMPTY", False),
        ("MULTIPOINT ((0 0), (1 1))", False),
        ("GEOMETRYCOLLECTION EMPTY", False),
        ("GEOMETRYCOLLECTION (POINT Z (0 0 0))", True),
        ("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT Z (0 0 0)))", True),
    ],
)
def test_st_hasz(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_HasZ({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", False),
        ("POINT Z EMPTY", False),
        ("POINT M EMPTY", True),
        ("POINT ZM EMPTY", True),
        ("POINT Z (0 0 0)", False),
        ("POINT M (0 0 0)", True),
        ("POINT ZM (0 0 0 0)", True),
        ("LINESTRING EMPTY", False),
        ("LINESTRING M EMPTY", True),
        ("LINESTRING M (0 0 0, 1 1 1)", True),
        ("POLYGON EMPTY", False),
        ("MULTIPOINT ((0 0), (1 1))", False),
        ("GEOMETRYCOLLECTION EMPTY", False),
        ("GEOMETRYCOLLECTION (POINT M (0 0 0))", True),
        ("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT M (0 0 0)))", True),
    ],
)
def test_st_hasm(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_HasM({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", True),
        ("LINESTRING EMPTY", True),
        ("POLYGON EMPTY", True),
        ("MULTIPOINT EMPTY", True),
        ("MULTILINESTRING EMPTY", True),
        ("MULTIPOLYGON EMPTY", True),
        ("GEOMETRYCOLLECTION EMPTY", True),
        ("POINT (0 0)", False),
        ("LINESTRING (0 0, 1 1)", False),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", False),
        ("MULTIPOINT ((0 0), (1 1))", False),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", False),
        (
            "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            False,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            False,
        ),
    ],
)
def test_st_isempty(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_IsEmpty({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", 0),
        ("LINESTRING EMPTY", 0),
        ("POINT (0 0)", 0),
        ("LINESTRING (0 0, 0 1)", 1),
        ("MULTIPOINT ((0 0), (1 1))", 0),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", 2.8284271247461903),
        # Polygons contribute 0 because perimeters aren't included in the length calculation
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 0),
        ("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))", 0),
        (
            "GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            2.8284271247461903,
        ),
    ],
)
def test_st_length(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Length({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", 0),
        ("LINESTRING EMPTY", 0),
        ("POINT (0 0)", 0),
        ("LINESTRING (0 0, 0 1)", 0),
        ("MULTIPOINT ((0 0), (1 1))", 0),
        ("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))", 0),
        # Polygons contribute 0 because perimeters aren't included in the length calculation
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 4),
        ("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)))", 8),
        (
            "GEOMETRYCOLLECTION (POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))",
            8,
        ),
    ],
)
def test_st_perimeter(eng, geom, expected):
    if eng == PostGIS and geom is None:
        # PostGIS ST_Perimeter doesn't work unless we explicitly cast to null
        arg = "NULL::geometry"
    else:
        arg = geom_or_null(geom)

    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Perimeter({arg})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (None, None, None),
        (1, None, None),
        (None, 1, None),
        (1, 1, "POINT (1 1)"),
        (1.0, 1.0, "POINT (1 1)"),
        (10, -1.5, "POINT (10 -1.5)"),
    ],
)
def test_st_geogpoint(eng, x, y, expected):
    eng = eng.create_or_skip()
    if eng == SedonaDB:
        eng.assert_query_result(
            f"SELECT ST_GeogPoint({val_or_null(x)}, {val_or_null(y)})", expected
        )
    else:
        eng.assert_query_result(
            f"SELECT ST_Point({val_or_null(x)}, {val_or_null(y)}) as geography",
            expected,
        )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (None, None, None),
        (1, None, None),
        (None, 1, None),
        (1, 1, "POINT (1 1)"),
        (1.0, 1.0, "POINT (1 1)"),
        (10, -1.5, "POINT (10 -1.5)"),
    ],
)
def test_st_point(eng, x, y, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(
        f"SELECT ST_Point({val_or_null(x)}, {val_or_null(y)})", expected
    )


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT Z EMPTY", None),
        ("POINT M EMPTY", None),
        ("POINT ZM EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", None),
        ("POINT M (1 2 3)", 3),
        ("POINT ZM (1 2 3 -4)", -4),
    ],
)
def test_st_m(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_M({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT Z EMPTY", None),
        ("POINT M EMPTY", None),
        ("POINT ZM EMPTY", None),
        ("POINT (1 2)", 1),
        ("POINT Z (1 2 3)", 1),
        ("POINT M (1 2 3)", 1),
        ("POINT ZM (1 2 3 -4)", 1),
    ],
)
def test_st_x(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_X({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT Z EMPTY", None),
        ("POINT M EMPTY", None),
        ("POINT ZM EMPTY", None),
        ("POINT (1 2)", 2),
        ("POINT Z (1 2 3)", 2),
        ("POINT M (1 2 3)", 2),
        ("POINT ZM (1 2 3 -4)", 2),
    ],
)
def test_st_y(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Y({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT Z EMPTY", None),
        ("POINT M EMPTY", None),
        ("POINT ZM EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", 3),
        ("POINT M (1 2 3)", None),
        ("POINT ZM (1 2 3 -4)", 3),
    ],
)
def test_st_z(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_Z({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", 1),
        ("POINT Z (1 2 3)", 1),
        ("POINT M (1 2 3)", 1),
        ("POINT ZM (1 2 3 4)", 1),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", 0),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 0),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            1,
        ),
    ],
)
def test_st_xmin(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_XMin({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", 2),
        ("POINT Z (1 2 3)", 2),
        ("POINT M (1 2 3)", 2),
        ("POINT ZM (1 2 3 4)", 2),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", 0),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 0),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            2,
        ),
    ],
)
def test_st_ymin(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_YMin({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", 1),
        ("POINT Z (1 2 3)", 1),
        ("POINT M (1 2 3)", 1),
        ("POINT ZM (1 2 3 4)", 1),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", 3),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 1),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            1,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            7,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            10,
        ),
    ],
)
def test_st_xmax(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_XMax({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", 2),
        ("POINT Z (1 2 3)", 2),
        ("POINT M (1 2 3)", 2),
        ("POINT ZM (1 2 3 4)", 2),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", 3),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 1),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            1,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            8,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            10,
        ),
    ],
)
def test_st_ymax(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_YMax({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", 3),
        ("POINT M (1 2 3)", None),
        ("POINT ZM (1 2 3 4)", 3),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            None,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            None,
        ),
    ],
)
def test_st_zmin(eng, geom, expected):
    # PostGIS returns 0 instead of null for non-empty geometries that don't have a Z coordinate
    if eng == PostGIS and (
        expected is None and geom is not None and "EMPTY" not in geom
    ):
        expected = 0

    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_ZMin({geom_or_null(geom)})", expected)


@pytest.mark.parametrize("eng", [SedonaDB, PostGIS])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", 3),
        ("POINT M (1 2 3)", None),
        ("POINT ZM (1 2 3 4)", 3),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            None,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            9,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            None,
        ),
    ],
)
def test_st_zmax(eng, geom, expected):
    # PostGIS returns 0 instead of null for non-empty geometries that don't have a Z coordinate
    if eng == PostGIS and (
        expected is None and geom is not None and "EMPTY" not in geom
    ):
        expected = 0

    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_ZMax({geom_or_null(geom)})", expected)


# Note: PostGIS doesn't support MMin/MMax, so we only test SedonaDB
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", None),
        ("POINT M (1 2 3)", 3),
        ("POINT ZM (1 2 3 4)", 4),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            None,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            0,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            None,
        ),
    ],
)
def test_st_mmin(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_MMin({geom_or_null(geom)})", expected)


# Note: PostGIS doesn't support MMin/MMax, so we only test SedonaDB
@pytest.mark.parametrize("eng", [SedonaDB])
@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (None, None),
        ("POINT EMPTY", None),
        ("POINT (1 2)", None),
        ("POINT Z (1 2 3)", None),
        ("POINT M (1 2 3)", 3),
        ("POINT ZM (1 2 3 4)", 4),
        ("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", None),
        ("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", None),
        (
            "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)), LINESTRING (0 0, 1 1))",
            None,
        ),
        (
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (3 4 5 6, 7 8 9 10), POLYGON ZM ((0 0 0 0, 1 0 0 0, 0 1 0 0, 0 0 0 0)))",
            10,
        ),
        (
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6), GEOMETRYCOLLECTION (POINT (10 10)))",
            None,
        ),
    ],
)
def test_st_mmax(eng, geom, expected):
    eng = eng.create_or_skip()
    eng.assert_query_result(f"SELECT ST_MMax({geom_or_null(geom)})", expected)
