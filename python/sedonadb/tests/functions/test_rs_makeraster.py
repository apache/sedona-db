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

"""RS_MakeRaster: an out-db raster reference built from catalog metadata.

The Rust unit tests pin the exact output (transform, CRS, out-db URI) per
kernel; these tests check the raster against RS_FromPath of the same file,
including lazy pixel loading through the out-db loader, and that building and
inspecting one never touches the file.
"""

import pytest

METADATA = """
    RS_Width({r}) AS width,
    RS_Height({r}) AS height,
    RS_GeoReference({r}) AS georeference,
    RS_SRID({r}) AS srid,
    RS_NumBands({r}) AS num_bands,
    RS_BandPixelType({r}, 1) AS pixel_type
"""


def _affine_args(path, band=None):
    """RS_MakeRaster's affine-form arguments for `path`, read with rasterio the
    way a catalog records them (STAC `proj:shape`, `proj:transform`,
    `proj:code` and `data_type`)."""
    rasterio = pytest.importorskip("rasterio")
    with rasterio.open(path) as src:
        t = src.transform
        b = 1 if band is None else band
        args = [
            f"'{path}'",
            f"'{src.dtypes[b - 1]}'",
            str(src.width),
            str(src.height),
            repr(t.c),
            repr(t.f),
            repr(t.a),
            repr(t.e),
            repr(t.b),
            repr(t.d),
            str(src.crs.to_epsg()),
        ]
    if band is not None:
        args.append(str(band))
    return ", ".join(args)


def _explain(con, sql):
    rows = con.sql(f"EXPLAIN {sql}").to_arrow_table().to_pylist()
    return "\n".join(row["plan"] for row in rows)


def _row(con, sql):
    return con.sql(sql).to_arrow_table().to_pylist()[0]


@pytest.mark.parametrize("name", ["test1.tiff", "test4.tiff", "sentinel2.tif"])
def test_matches_rs_frompath(con, sedona_testing, name):
    """A raster built from a file's metadata is the raster RS_FromPath reads
    from its header, and functions that need pixels load the same values
    through the out-db loader."""
    path = sedona_testing / "data/raster" / name
    made = f"RS_MakeRaster({_affine_args(path)})"
    read = f"RS_FromPath('{path}')"

    assert _row(con, f"SELECT {METADATA.format(r=made)}") == _row(
        con, f"SELECT {METADATA.format(r=read)}"
    )
    assert _row(con, f"SELECT RS_BandPath({made}) AS p") == _row(
        con, f"SELECT RS_BandPath({read}) AS p"
    )

    # The world coordinate of a pixel near the middle of the raster.
    center = _row(
        con,
        f"""
        SELECT
            RS_RasterToWorldCoordX({read}, RS_Width({read}) / 2, RS_Height({read}) / 3) AS cx,
            RS_RasterToWorldCoordY({read}, RS_Width({read}) / 2, RS_Height({read}) / 3) AS cy,
            RS_SRID({read}) AS srid
        """,
    )
    point = (
        f"ST_GeomFromText('POINT ({center['cx']} {center['cy']})', "
        f"'EPSG:{center['srid']}')"
    )
    pixels = """
        RS_SummaryStats({r}, 'sum', 1, false) AS sum,
        RS_SummaryStats({r}, 'mean', 1, false) AS mean,
        RS_SummaryStats({r}, 'max', 1, false) AS max,
        RS_Value({r}, {point}) AS value
    """
    made_pixels = _row(con, "SELECT " + pixels.format(r=made, point=point))
    read_pixels = _row(con, "SELECT " + pixels.format(r=read, point=point))
    assert made_pixels == read_pixels
    assert made_pixels["value"] is not None
    assert made_pixels["max"] > 0


def test_extent_form_matches_rs_frompath(con, sedona_testing):
    """test4.tiff is a north-up 10 x 10 grid over (0, 0)-(10, 10) in EPSG:4326,
    so the extent form describes it exactly."""
    path = sedona_testing / "data/raster/test4.tiff"
    made = (
        f"RS_MakeRaster('{path}', 'uint8', 10, 10, ST_MakeEnvelope(0, 0, 10, 10, 4326))"
    )
    read = f"RS_FromPath('{path}')"
    assert _row(con, f"SELECT {METADATA.format(r=made)}") == _row(
        con, f"SELECT {METADATA.format(r=read)}"
    )
    stats = "RS_SummaryStats({r}, 'sum', 1, false) AS sum"
    assert _row(con, f"SELECT {stats.format(r=made)}") == _row(
        con, f"SELECT {stats.format(r=read)}"
    )


def test_crs_string_and_srid_forms_agree(con, sedona_testing):
    path = sedona_testing / "data/raster/sentinel2.tif"
    srid_form = f"RS_MakeRaster({_affine_args(path)})"
    crs_form = srid_form.replace(", 32614)", ", 'EPSG:32614')")
    assert crs_form != srid_form

    probe = f"""
        {METADATA.format(r="{r}")},
        RS_SummaryStats({{r}}, 'sum', 1, false) AS sum,
        ST_AsText(RS_Envelope({{r}})) AS envelope
    """
    assert _row(con, "SELECT " + probe.format(r=crs_form)) == _row(
        con, "SELECT " + probe.format(r=srid_form)
    )
    assert _row(con, f"SELECT RS_SRID({crs_form}) AS srid") == {"srid": 32614}

    # A PROJJSON CRS is accepted too, and resolves to the same SRID.
    projjson = _row(con, f"SELECT RS_CRS(RS_FromPath('{path}')) AS crs")["crs"]
    assert projjson.startswith("{")
    projjson_form = srid_form.replace(", 32614)", ", $1)")
    got = (
        con.sql(f"SELECT RS_SRID({projjson_form}) AS srid", params=(projjson,))
        .to_arrow_table()
        .to_pylist()
    )
    assert got == [{"srid": 32614}]


def test_band_selects_the_source_band(con, sedona_testing):
    """test3.tif has four uint16 bands; band=2 references and reads band 2."""
    path = sedona_testing / "data/raster/test3.tif"
    made = f"RS_MakeRaster({_affine_args(path, band=2)})"
    read = f"RS_FromPath('{path}')"

    made_sum = _row(con, f"SELECT RS_SummaryStats({made}, 'sum', 1, false) AS s")
    read_sums = [
        _row(con, f"SELECT RS_SummaryStats({read}, 'sum', {b}, false) AS s")["s"]
        for b in (1, 2)
    ]
    assert made_sum["s"] == read_sums[1]
    # The bands differ, so this would catch band 1 being read instead.
    assert read_sums[0] != read_sums[1]


def test_null_arguments_yield_null(con):
    affine = "'/a.tif', 'uint8', 2, 2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0"
    extent = "ST_MakeEnvelope(0, 0, 2, 2, 4326)"
    for expr in [
        f"RS_MakeRaster(CAST(NULL AS VARCHAR), 'uint8', 2, 2, {extent})",
        f"RS_MakeRaster('/a.tif', CAST(NULL AS VARCHAR), 2, 2, {extent})",
        f"RS_MakeRaster('/a.tif', 'uint8', CAST(NULL AS INT), 2, {extent})",
        "RS_MakeRaster('/a.tif', 'uint8', 2, 2, ST_GeomFromText(NULL))",
        f"RS_MakeRaster('/a.tif', 'uint8', 2, 2, {extent}, CAST(NULL AS INT))",
        f"RS_MakeRaster({affine}, CAST(NULL AS INT))",
        f"RS_MakeRaster({affine}, CAST(NULL AS VARCHAR))",
        f"RS_MakeRaster({affine}, 'EPSG:4326', CAST(NULL AS INT))",
        "RS_MakeRaster('/a.tif', 'uint8', 2, 2, CAST(NULL AS DOUBLE), 2.0, 1.0, "
        "-1.0, 0.0, 0.0, 4326)",
    ]:
        got = con.sql(f"SELECT {expr} IS NULL AS n").to_arrow_table().to_pylist()
        assert got == [{"n": True}], expr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            "'/a.tif', 'uint8', 0, 2, ST_MakeEnvelope(0, 0, 2, 2)",
            "width and height must be positive",
        ),
        (
            "'/a.tif', 'uint8', 2, -2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0, 4326",
            "width and height must be positive",
        ),
        (
            "'/a.tif', 'uint8', 2, 2, ST_MakeEnvelope(0, 0, 2, 2), 0",
            "band must be a 1-based band index",
        ),
        (
            "'/a.tif', 'complex128', 2, 2, ST_MakeEnvelope(0, 0, 2, 2)",
            "Unsupported pixelType",
        ),
        (
            "'/a.tif', 'uint8', 2, 2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0, 'not a crs'",
            "invalid crs 'not a crs'",
        ),
        (
            "'/a.tif', 'uint8', 2, 2, ST_GeomFromText('POINT (1 1)')",
            "positive width and height",
        ),
    ],
)
def test_invalid_arguments(con, args, message):
    with pytest.raises(Exception, match=message):
        con.sql(f"SELECT RS_MakeRaster({args})").to_arrow_table()


def test_no_file_access(con, tmp_path):
    """Nothing reads the file until a function needs pixels: with a path that
    does not exist, metadata functions (and the metadata-only setter
    RS_SetBandNoDataValue) succeed and only pixel reads fail."""
    path = tmp_path / "does-not-exist.tif"
    r = (
        f"RS_MakeRaster('{path}', 'uint16', 100, 50, "
        "500000.0, 4100000.0, 10.0, -10.0, 0.0, 0.0, 'EPSG:32610')"
    )
    got = _row(
        con,
        f"""
        SELECT
            RS_Width({r}) AS width,
            RS_Height({r}) AS height,
            RS_SRID({r}) AS srid,
            ST_AsText(RS_Envelope({r})) AS envelope,
            RS_Intersects({r}, ST_GeomFromText('POINT (500500 4099900)', 'EPSG:32610')) AS hit,
            RS_Intersects({r}, ST_GeomFromText('POINT (0 0)', 'EPSG:32610')) AS miss,
            RS_BandPath({r}) AS band_path,
            RS_BandNoDataValue(RS_SetBandNoDataValue({r}, 1, 0), 1) AS nodata
        """,
    )
    assert got == {
        "width": 100,
        "height": 50,
        "srid": 32610,
        "envelope": "POLYGON((500000 4099500,501000 4099500,501000 4100000,"
        "500000 4100000,500000 4099500))",
        "hit": True,
        "miss": False,
        "band_path": str(path),
        "nodata": 0.0,
    }
    assert not path.exists()

    with pytest.raises(Exception, match="does-not-exist.tif"):
        con.sql(f"SELECT RS_SummaryStats({r}, 'sum')").to_arrow_table()


def test_spatial_join_and_ensure_loaded_plans(con, sedona_testing):
    """RS_MakeRaster plugs into the raster planner paths like RS_FromPath: a
    raster-vector join on RS_Intersects plans a spatial join, and a pixel
    function gets RS_EnsureLoaded injected ahead of it."""
    path = sedona_testing / "data/raster/test4.tiff"
    con.sql(
        f"""
        SELECT RS_MakeRaster('{path}', 'uint8', 10, 10, ST_MakeEnvelope(0, 0, 10, 10, 4326)) AS r
        """
    ).to_view("test_rs_makeraster_rasters", overwrite=True)
    con.sql(
        """
        SELECT * FROM (VALUES
            (1, ST_GeomFromText('POINT (5 5)', 4326)),
            (2, ST_GeomFromText('POINT (50 50)', 4326))
        ) AS t(id, geom)
        """
    ).to_view("test_rs_makeraster_points", overwrite=True)

    join = """
        SELECT p.id, RS_Value(r.r, p.geom) AS v
        FROM test_rs_makeraster_rasters r
        JOIN test_rs_makeraster_points p ON RS_Intersects(r.r, p.geom)
    """
    plan = _explain(con, join)
    assert "SpatialJoinExec" in plan, plan
    rows = con.sql(join).to_arrow_table().to_pylist()
    assert [row["id"] for row in rows] == [1]
    assert rows[0]["v"] is not None

    stats = "SELECT RS_SummaryStats(r, 'sum') AS s FROM test_rs_makeraster_rasters"
    plan = _explain(con, stats)
    assert "rs_ensureloaded" in plan.lower(), plan
