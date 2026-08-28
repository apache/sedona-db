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
"""Sedona Spark as a :class:`~sedonadb.testing.DBEngine`.

The compatibility target for SedonaDB's SQL surface is Sedona Spark, so parity
tests broadcast one shared SQL string to both engines and compare strictly —
the same pattern the geometry suite uses for SedonaDB vs PostGIS. This lives in
its own module because the engine needs pyspark, a JVM, and network access,
none of which the core :mod:`sedonadb.testing` module should pull in.

Bootstrapping a ``SparkSession`` (downloading the Sedona jars from Maven, JVM
startup) costs tens of seconds, so these tests are opt-in: :meth:`create_or_skip`
skips — even under ``SEDONADB_PYTHON_NO_SKIP_TESTS`` — unless
``SEDONADB_RUN_SPARK_TESTS`` is set, in which case construction failures
propagate.

Requires Spark 4.0+ (results travel out of the JVM through ``DataFrame.toArrow``,
which preserves nulls Arrow-natively — unlike ``toPandas``, which would coerce a
nodata ``None`` to ``NaN`` and mask exactly the value under test).
"""

import os

import pyarrow as pa

from sedonadb.testing import DBEngine

# Sedona jar coordinates. Override with SEDONADB_SEDONA_SPARK_PACKAGES (full
# Maven coordinates) when testing against a different Sedona release.
SEDONA_SPARK_VERSION = "1.9.0"
GEOTOOLS_WRAPPER_VERSION = "1.9.0-33.5"

# ``result_to_table`` collects through ``DataFrame.toArrow``, which landed in
# Spark 4.0. That floor is a property of this harness rather than of the jars,
# so it holds even when SEDONADB_SEDONA_SPARK_PACKAGES overrides the
# coordinates.
MIN_PYSPARK_VERSION = (4, 0)


class SedonaSpark(DBEngine):
    """Runs Sedona Spark SQL — the compatibility-target dialect.

    One local ``SparkSession`` is bootstrapped per process and shared across
    engine instances. Rasters are read from GeoTIFF files with
    ``RS_FromGeoTiff`` over a ``binaryFile`` scan (see :meth:`create_raster_view`).
    """

    _spark = None

    def __init__(self):
        self._session = self._ensure_session()

    @classmethod
    def name(cls) -> str:
        return "sedona-spark"

    @classmethod
    def install_hint(cls) -> str:
        return (
            "- Run `pip install 'pyspark>=4.0' apache-sedona` (needs a JVM; the "
            "first run downloads the Sedona jars from Maven)\n"
            "- Set SEDONADB_RUN_SPARK_TESTS=true to opt in"
        )

    @classmethod
    def create_or_skip(cls, *args, **kwargs) -> "SedonaSpark":
        import pytest

        if os.environ.get("SEDONADB_RUN_SPARK_TESTS", "false") not in ("true", "1"):
            pytest.skip("Sedona Spark parity tests are opt-in:\n" + cls.install_hint())
        return cls(*args, **kwargs)

    @classmethod
    def _ensure_session(cls):
        if SedonaSpark._spark is None:
            cls._check_pyspark_version()
            from sedona.spark import SedonaContext

            config = (
                SedonaContext.builder()
                .master("local[2]")
                .appName("sedonadb-spark-parity")
                .config("spark.jars.packages", cls._packages())
                .config("spark.jars.ivy", cls._ivy_dir())
                .config("spark.ui.enabled", "false")
                .getOrCreate()
            )
            SedonaSpark._spark = SedonaContext.create(config)
        return SedonaSpark._spark

    @staticmethod
    def _pyspark_version() -> tuple:
        import pyspark

        return tuple(int(part) for part in pyspark.__version__.split(".")[:2])

    @classmethod
    def _check_pyspark_version(cls) -> None:
        """Reject a pyspark that predates the Arrow collect path.

        Without this a 3.5 install selects jars and builds a session happily,
        then dies with an ``AttributeError`` on ``toArrow`` inside the first
        comparison — a failure far from its cause. Checked at session setup, not
        in :meth:`_packages`, so an explicit SEDONADB_SEDONA_SPARK_PACKAGES
        override cannot route around it.
        """
        import pyspark

        if cls._pyspark_version() < MIN_PYSPARK_VERSION:
            minimum = ".".join(str(part) for part in MIN_PYSPARK_VERSION)
            raise RuntimeError(
                f"Sedona Spark parity tests need pyspark >= {minimum} for "
                f"DataFrame.toArrow; found {pyspark.__version__}. Run "
                f"`pip install 'pyspark>={minimum}'`"
            )

    @classmethod
    def _packages(cls) -> str:
        env = os.environ.get("SEDONADB_SEDONA_SPARK_PACKAGES")
        if env:
            return env
        major, minor = cls._pyspark_version()
        # Sedona publishes per-Spark-minor artifacts, and 1.9.0's only Spark 4
        # build is 4.0 (Scala 2.13); _check_pyspark_version has already rejected
        # everything below that. A pyspark newer than the newest published
        # artifact gets the newest, which usually loads — override the
        # coordinates if it doesn't.
        known = ("4.0",)
        spark_suffix = f"{major}.{minor}"
        if spark_suffix not in known:
            spark_suffix = known[-1]
        return (
            f"org.apache.sedona:sedona-spark-shaded-{spark_suffix}_2.13:"
            f"{SEDONA_SPARK_VERSION},"
            f"org.datasyslab:geotools-wrapper:{GEOTOOLS_WRAPPER_VERSION}"
        )

    @staticmethod
    def _ivy_dir() -> str:
        """Directory Ivy resolves ``spark.jars.packages`` into.

        Pinned so CI can cache the downloaded jars: newer Ivy releases (bundled
        with newer Spark) moved the default location, silently breaking a cache
        keyed on the old path. Override with SEDONADB_SPARK_IVY_DIR.
        """
        return os.environ.get(
            "SEDONADB_SPARK_IVY_DIR",
            os.path.join(os.path.expanduser("~"), ".ivy2"),
        )

    def create_raster_view(self, name, path) -> "SedonaSpark":
        self._session.read.format("binaryFile").load(str(path)).selectExpr(
            "RS_FromGeoTiff(content) AS rast"
        ).createOrReplaceTempView(name)
        return self

    def execute_and_collect(self, query):
        return self._session.sql(query)

    def result_to_table(self, result) -> pa.Table:
        # toArrow() preserves nulls Arrow-natively; toPandas() would turn a
        # nodata None into a float NaN and mask the value under test.
        return result.toArrow()

    def decode_raster_result(self, sql):
        from sedonadb.raster_testing import DecodedRaster, decode_geotiff_bytes

        # A raster can't leave the JVM as a native column, so transport it as
        # GeoTIFF bytes. geotools' writer refuses a CRS-less coverage, so stamp
        # an arbitrary SRID first — transport-only, it doesn't touch pixels or
        # the geotransform (and DecodedRaster carries no CRS to compare). Nodata
        # is read separately through RS_BandNoDataValue so the comparison sees
        # the engine's own claim rather than the GeoTIFF writer's encoding of it.
        result = self._session.sql(sql).toDF("r").cache()
        try:
            head = result.selectExpr(
                "RS_AsGeoTiff(RS_SetSRID(r, 3857)) AS t", "RS_NumBands(r) AS n"
            ).first()
            if head is None or head.t is None:
                return None
            decoded = decode_geotiff_bytes(bytes(head.t))
            nodata = result.selectExpr(
                *[f"RS_BandNoDataValue(r, {b}) AS nd{b}" for b in range(1, head.n + 1)]
            ).first()
            return DecodedRaster(decoded.pixels, decoded.gdal_transform, list(nodata))
        finally:
            result.unpersist()
