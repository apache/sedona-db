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

"""Cross-engine parity testing for raster (`RS_*`) functions.

The raster sibling of the `DBEngine` framework in `sedonadb.testing`.
Geometry parity broadcasts one SQL string to every engine; raster engines
cannot share SQL — rasterio is not a SQL engine, and raster SQL dialects
disagree on function names and argument order — so parity here is
operation-level instead: `RasterEngine` exposes one method per operation
(`clip()`, `value()`, ...) returning plain decoded values, and each
engine translates the operation into its own invocation.

Comparisons are asymmetric: `SedonaDB` is always the **subject** — the
engine whose behavior the tests exist to pin — and the other engines are
**comparators** it is checked against. `Rasterio` reconstructs each
operation from library primitives (where an operation has engine-specific
policy, like the fill outside a rasterized geometry, the reconstruction
resolves it to the subject's policy); `SedonaSpark` runs the Sedona Spark
SQL dialect, the compatibility target. Comparator↔comparator agreement is
never asserted. Where the subject and a comparator are known to disagree,
the test module declares a `Deviation`: the case still runs and is marked
as a strict expected failure, so the ledger entry itself fails the suite
the day the engines converge. Operations the subject does not implement
skip their module wholesale (`SedonaDB.implements`) until it does.

Test fixtures follow one pattern: define the raster once as a numpy array +
GDAL geotransform, write it to a GeoTIFF with `write_geotiff`, and hand the
same file to every engine. Fixtures stay CRS-less (no CRS on either side, so
nothing reprojects and results stay bit-comparable) except where an engine's
encoder requires a real one — then every side carries the same CRS.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Tuple

import numpy as np
import pyarrow as pa

# Maven coordinates for the Sedona Spark jars downloaded by `SedonaSpark`.
# The artifact's Spark/Scala suffix is derived from the installed pyspark;
# override the whole packages list with SEDONADB_SEDONA_SPARK_PACKAGES when
# the derivation doesn't match your environment.
SEDONA_SPARK_VERSION = "1.9.0"
GEOTOOLS_WRAPPER_VERSION = "1.9.0-33.5"


@dataclass
class DecodedRaster:
    """One raster decoded to plain values.

    `pixels` is `(band, rows, cols)`, `gdal_transform` is GDAL-order
    `(origin_x, scale_x, skew_x, origin_y, skew_y, scale_y)`, and `nodata`
    holds one sentinel per band (unpacked in the band's dtype).
    `compression` is the codec name of the decoded container when one was
    read (GeoTIFF decodes only); it is carried for encoder tests to assert
    on and deliberately not part of `assert_decoded_equal`.
    """

    pixels: "np.ndarray"
    gdal_transform: Tuple[float, ...]
    nodata: List[Any]
    compression: Optional[str] = None


@dataclass
class Deviation:
    """One known behavioral difference between the subject and a comparator.

    Declared next to the cases it covers in a test module and applied with
    `expect_deviations`. `matches` receives the test's parametrization dict
    and selects the cases the deviation covers (None covers every case of
    the operation). `kind` is "xfail" when both engines compute an answer
    and the answers differ — enforced as a *strict* expected failure, so the
    entry itself fails the suite when the engines converge and the ledger
    can't go stale — or "skip" when the comparator cannot run the case at
    all (it raises where the subject computes).
    """

    comparator: type
    operation: str
    reason: str
    matches: Optional[Callable[[dict], bool]] = None
    kind: str = "xfail"


def expect_deviations(request, comparator, operation: str, deviations) -> None:
    """Arm the ledger entries matching this test invocation.

    Call first in a parity test body, before invoking the engines. Matching
    "skip" entries skip immediately; matching "xfail" entries mark the test
    as a strict expected failure constrained to `AssertionError`, so a
    comparator exception or harness bug still fails loudly and convergence
    surfaces as XPASS(strict).
    """
    import pytest

    params = getattr(getattr(request.node, "callspec", None), "params", {})
    for deviation in deviations:
        if not isinstance(comparator, deviation.comparator):
            continue
        if deviation.operation != operation:
            continue
        if deviation.matches is not None and not deviation.matches(params):
            continue
        if deviation.kind == "skip":
            pytest.skip(f"{type(comparator).name()} deviation: {deviation.reason}")
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True, raises=AssertionError, reason=deviation.reason
            )
        )


class RasterEngine:
    """Executes raster operations on one engine for cross-engine comparison.

    Operations take the path of a GeoTIFF fixture so the same file can be
    handed to every engine. Conventions are the ones Sedona (DB and Spark)
    and PostGIS share: bands and pixel `(col, row)` indices are 1-based and
    world coordinates are in the raster's CRS. Geometry arguments are WKT;
    geometry results are shapely objects.
    """

    @classmethod
    def name(cls) -> str:
        """A short identifier used in parametrized test ids and messages."""
        return cls.__name__.lower()

    @classmethod
    def install_hint(cls) -> str:
        """A short setup hint appended to skip messages when this engine
        can't be built."""
        return ""

    @classmethod
    def implements(cls, operation: str) -> bool:
        """Whether this engine overrides `operation` (the base raises)."""
        return getattr(cls, operation) is not getattr(RasterEngine, operation)

    @classmethod
    def create_or_skip(cls, *args, **kwargs):
        """Create this engine, or skip the calling test if it can't be built.

        If `SEDONADB_PYTHON_NO_SKIP_TESTS` is set the failure propagates
        instead, so CI can't silently skip.
        """
        try:
            return cls(*args, **kwargs)
        except Exception as e:
            if os.environ.get("SEDONADB_PYTHON_NO_SKIP_TESTS", "false") in (
                "true",
                "1",
            ):
                raise
            import pytest

            pytest.skip(
                f"Can't create {cls.__name__} raster engine: {e}\n{cls.install_hint()}"
            )

    def close(self):
        """Release engine resources — base implementation does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _not_implemented(self, operation: str):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement {operation}"
        )

    def clip(
        self,
        path,
        geometry_wkt: str,
        *,
        band: int = 0,
        all_touched: bool = False,
        nodata: Optional[float] = None,
        crop: bool = True,
    ) -> Optional[DecodedRaster]:
        """Clip the GeoTIFF at `path` to a WKT geometry.

        `band` 0 selects all bands, otherwise the 1-based band. Returns None
        when the geometry selects no pixels and the engine's lenient behavior
        yields a NULL/absent result.
        """
        self._not_implemented("clip")

    def value(self, path, x: float, y: float, *, band: int = 1) -> Optional[float]:
        """Sample the band's value at world coordinates `(x, y)`.

        Returns None when the point falls outside the raster extent or the
        sampled value equals the band's nodata.
        """
        self._not_implemented("value")

    def values(
        self, path, points: List[Tuple[float, float]], *, band: int = 1
    ) -> List[Optional[float]]:
        """Sample the band at several world points in one call.

        One result per input point, in input order, with the same None rules
        as `value`. Engines differ in how a point collection is spelled
        (SedonaDB takes a MULTIPOINT, Sedona Spark an array of points); the
        operation takes plain `(x, y)` tuples and each engine translates.
        """
        self._not_implemented("values")

    def band_nodata(self, path, *, band: int = 1) -> Optional[float]:
        """The band's nodata value, or None when the band doesn't define one."""
        self._not_implemented("band_nodata")

    def pixel_as_point(self, path, col: int, row: int) -> Tuple[float, float]:
        """World coordinates of the upper-left corner of pixel `(col, row)`."""
        self._not_implemented("pixel_as_point")

    def pixel_as_centroid(self, path, col: int, row: int) -> Tuple[float, float]:
        """World coordinates of the center of pixel `(col, row)`."""
        self._not_implemented("pixel_as_centroid")

    def pixel_as_polygon(self, path, col: int, row: int):
        """The bounding polygon of pixel `(col, row)` as a shapely Polygon."""
        self._not_implemented("pixel_as_polygon")

    def as_raster(
        self,
        geometry_wkt: str,
        path,
        pixel_type: str,
        *,
        all_touched: bool = False,
        burn_value: float = 1.0,
        nodata: Optional[float] = None,
        use_geometry_extent: bool = True,
    ) -> DecodedRaster:
        """Rasterize a WKT geometry on the grid of the reference GeoTIFF.

        `pixel_type` is a numpy dtype name from the set every engine supports:
        uint8, uint16, int16, int32, float32, float64. Pixels inside the
        geometry get `burn_value`, pixels outside get `nodata` (0 when None).
        With `use_geometry_extent` the output grid is the geometry's envelope
        snapped to the reference grid, otherwise the full reference grid.
        """
        self._not_implemented("as_raster")

    def resample(
        self, path, *, width: int, height: int, algorithm: str = "nearestneighbor"
    ) -> DecodedRaster:
        """Resample the raster to `width` x `height` pixels over the same extent.

        `algorithm` is one of nearestneighbor, bilinear, bicubic
        (case-insensitive; each engine maps to its own resampler names).
        """
        self._not_implemented("resample")

    def zonal_stats(
        self,
        path,
        geometry_wkt: str,
        *,
        band: Optional[int] = None,
        stat: str = "mean",
        all_touched: bool = False,
        exclude_nodata: bool = True,
        lenient: bool = True,
    ) -> Optional[float]:
        """One summary statistic over the band's pixels inside a WKT zone.

        Which of `RS_ZonalStats`'s positional overloads runs is chosen by which
        optional parameters are set: `band=None` omits the band argument (the
        3-argument `(raster, zone, stat_type)` overload, which resolves to band
        1 on a single-band raster); a set `band` selects the 4-argument overload;
        `all_touched`, then `exclude_nodata`, then `lenient` extend it to the 5-,
        6-, and 7-argument overloads.

        `stat` is one of count, sum, mean, median, mode, min, max, or the sample
        (ddof=1) stddev and variance. With `exclude_nodata` (the default) pixels
        equal to the band's nodata are skipped. Pixel selection follows the same
        centroid/all_touched rule as `clip`. With `lenient` (the default) a zone
        that misses the raster yields None instead of an error.
        """
        self._not_implemented("zonal_stats")

    def zonal_stats_all(
        self,
        path,
        geometry_wkt: str,
        *,
        band: Optional[int] = None,
        all_touched: bool = False,
        exclude_nodata: bool = True,
        lenient: bool = True,
    ) -> Optional[Mapping[str, Any]]:
        """Every summary statistic over the band's pixels inside a WKT zone.

        The `RS_ZonalStatsAll` overload ladder mirrors `zonal_stats` minus the
        `stat_type` argument: `(raster, zone)`, `(raster, zone, band)`, then
        `+all_touched`, `+exclude_nodata`, `+lenient`. Returns a mapping of the
        struct fields in `ZONAL_ALL_FIELDS` order — `count` (an integer pixel
        count) then the Float64 statistics — or None when `lenient` and the zone
        misses the raster.
        """
        self._not_implemented("zonal_stats_all")

    def tile_explode(
        self, path, tile_width: int, tile_height: int
    ) -> List[Tuple[int, int, DecodedRaster]]:
        """Split the raster into a grid of tiles.

        Returns `(tile_x, tile_y, tile)` triples sorted by `(tile_y, tile_x)`
        with 0-based tile indices. Edge tiles keep their partial size (no
        nodata padding); every tile keeps all bands and the band nodata.
        """
        self._not_implemented("tile_explode")

    def as_geotiff(
        self,
        path,
        *,
        compression: Optional[str] = None,
        quality: Optional[float] = None,
    ) -> DecodedRaster:
        """Load the raster, encode it back to GeoTIFF, and decode the bytes.

        Parity is content preservation: pixels, geotransform, and nodata must
        survive the engine's encoder regardless of `compression` (None,
        Deflate, LZW, PackBits — lossless codecs only) and `quality` (a
        0.0-1.0 fraction, only meaningful for lossy codecs).
        """
        self._not_implemented("as_geotiff")

    def from_binary(self, data: bytes) -> DecodedRaster:
        """Decode GeoTIFF bytes into the engine's raster type and back out.

        Pins the binary-input constructor (RS_FromGDALRaster in SedonaDB,
        RS_FromGeoTiff in Sedona Spark) against the file-based path: the
        decoded content must match what rasterio reads from the same bytes.
        """
        self._not_implemented("from_binary")


class DialectEngine(RasterEngine):
    """A `RasterEngine` that runs the RS_* function under test as SQL.

    SedonaDB and Sedona Spark are both dialect engines: an operation is one
    RS_* call, and the compatibility contract the parity tests exist to protect
    is that the identical call runs on both ("Spark SQL runs unchanged in
    SedonaDB"). To make that contract structural rather than a coincidence, the
    argument string is built in exactly one place — the `*_expr` generators
    below — and both dialects execute whatever those return. A dialect subclass
    supplies only its per-engine I/O: how a GeoTIFF path becomes a `rast` column
    and how a scalar / struct result is decoded (`_run_scalar` / `_run_struct`).
    The two dialects therefore cannot emit different SQL for the same operation.

    Only RS_ZonalStats / RS_ZonalStatsAll are routed through this shared surface;
    the other operations keep their existing per-dialect implementations.
    """

    def _run_scalar(self, path, expr: str) -> Optional[float]:
        """Execute a scalar-valued RS_* `expr` over a `rast` column from `path`."""
        self._not_implemented("_run_scalar")

    def _run_struct(self, path, expr: str) -> Optional[Mapping[str, Any]]:
        """Execute a struct-valued RS_* `expr` over a `rast` column from `path`,
        returning a mapping in `ZONAL_ALL_FIELDS` order (None for a NULL result)."""
        self._not_implemented("_run_struct")

    @staticmethod
    def _zonal_args(
        geometry_wkt, *, band, stat, all_touched, exclude_nodata, lenient
    ) -> List[str]:
        """The RS_ZonalStats / RS_ZonalStatsAll positional argument list, as SQL
        text, following Apache Sedona Spark's overload ladder verbatim.

        The two required arguments are the `rast` column and an
        `ST_GeomFromText('...')` zone (a constructor both dialects spell the same
        way). `band` is appended when set; `stat` (a statistic name, or None for
        RS_ZonalStatsAll, which has no stat argument) follows it. The optional
        `all_touched`/`exclude_nodata`/`lenient` flags append the shortest suffix
        that covers every non-default value — and, since they are the 5th+
        arguments, require an explicit `band`."""
        args = ["rast", f"ST_GeomFromText('{geometry_wkt}')"]
        if band is not None:
            args.append(str(int(band)))
        if stat is not None:
            args.append(f"'{stat}'")
        flags = [all_touched, exclude_nodata, lenient]
        defaults = [False, True, True]
        depth = 0
        for i, (value, default) in enumerate(zip(flags, defaults)):
            if value != default:
                depth = i + 1
        if depth:
            if band is None:
                raise ValueError(
                    "RS_ZonalStats needs an explicit band to pass the "
                    "all_touched/exclude_nodata/lenient flags"
                )
            args += [str(bool(flags[i])).lower() for i in range(depth)]
        return args

    @classmethod
    def zonal_stats_expr(
        cls,
        geometry_wkt,
        *,
        band=None,
        stat="mean",
        all_touched=False,
        exclude_nodata=True,
        lenient=True,
    ) -> str:
        """The exact `RS_ZonalStats(...)` SQL fragment both dialects run."""
        args = cls._zonal_args(
            geometry_wkt,
            band=band,
            stat=stat,
            all_touched=all_touched,
            exclude_nodata=exclude_nodata,
            lenient=lenient,
        )
        return f"RS_ZonalStats({', '.join(args)})"

    @classmethod
    def zonal_stats_all_expr(
        cls,
        geometry_wkt,
        *,
        band=None,
        all_touched=False,
        exclude_nodata=True,
        lenient=True,
    ) -> str:
        """The exact `RS_ZonalStatsAll(...)` SQL fragment both dialects run."""
        args = cls._zonal_args(
            geometry_wkt,
            band=band,
            stat=None,
            all_touched=all_touched,
            exclude_nodata=exclude_nodata,
            lenient=lenient,
        )
        return f"RS_ZonalStatsAll({', '.join(args)})"

    def zonal_stats(
        self,
        path,
        geometry_wkt,
        *,
        band=None,
        stat="mean",
        all_touched=False,
        exclude_nodata=True,
        lenient=True,
    ):
        return self._run_scalar(
            path,
            self.zonal_stats_expr(
                geometry_wkt,
                band=band,
                stat=stat,
                all_touched=all_touched,
                exclude_nodata=exclude_nodata,
                lenient=lenient,
            ),
        )

    def zonal_stats_all(
        self,
        path,
        geometry_wkt,
        *,
        band=None,
        all_touched=False,
        exclude_nodata=True,
        lenient=True,
    ):
        return self._run_struct(
            path,
            self.zonal_stats_all_expr(
                geometry_wkt,
                band=band,
                all_touched=all_touched,
                exclude_nodata=exclude_nodata,
                lenient=lenient,
            ),
        )


class SedonaDB(DialectEngine):
    """Runs `RS_*` on a SedonaDB connection — the engine under test."""

    def __init__(self, con=None):
        import sedonadb

        self._con = con if con is not None else sedonadb.connect()

    @classmethod
    def create_or_skip(cls, *args, **kwargs):
        # Never skip on the engine under test: a construction failure here is
        # a bug, not a missing optional backend.
        return cls(*args, **kwargs)

    @classmethod
    def registers_function(cls, probe_sql: str) -> bool:
        """Whether this SedonaDB build registers the RS_* function `probe_sql`
        calls — the runtime gate for a parity module.

        The `DialectEngine` harness methods are pure SQL construction and exist
        whether or not the underlying function is compiled in, so a module can't
        gate on the method's presence. It runs a trivial call instead: a
        planning-time "Invalid function" error means the build predates the
        function (skip the module); any other outcome — success, or an
        argument/CRS error from the probe call — means it is registered."""
        try:
            cls()._con.sql(probe_sql).to_arrow_table()
        except Exception as e:
            return "invalid function" not in str(e).lower()
        return True

    def _rast_view(self, path) -> str:
        """Register the GeoTIFF at `path` as a one-row view exposing a `rast`
        column and return a query prefix that reads from it.

        `rast` is derived from a table column (`RS_FromPath(path)`), not a
        literal, so a kernel over it runs its real array path — literals
        constant-fold. The view name is fixed and overwritten each call; the
        session connection is reused across the suite."""
        name = "_zonal_parity_input"
        self._con.create_data_frame(
            pa.table({"path": pa.array([str(path)], type=pa.utf8())})
        ).to_view(name, overwrite=True)
        return f"(SELECT RS_FromPath(path) AS rast FROM {name})"

    def _run_scalar(self, path, expr):
        result = self._con.sql(
            f"SELECT {expr} AS v FROM {self._rast_view(path)}"
        ).to_arrow_table()["v"]
        return result[0].as_py()

    def _run_struct(self, path, expr):
        result = self._con.sql(
            f"SELECT {expr} AS r FROM {self._rast_view(path)}"
        ).to_arrow_table()["r"]
        return result[0].as_py() if result[0].is_valid else None

    def clip(
        self, path, geometry_wkt, *, band=0, all_touched=False, nodata=None, crop=True
    ):
        (result,) = self.clip_rows(
            path,
            [
                {
                    "wkt": geometry_wkt,
                    "band": band,
                    "all_touched": all_touched,
                    "nodata": nodata,
                    "crop": crop,
                }
            ],
        )
        return result

    def clip_rows(self, path, rows) -> List[Optional[DecodedRaster]]:
        """Run `RS_Clip` once over an N-row table, one row per parameter combo.

        Options travel as table columns rather than literals so the kernel
        executes its real array path (literals constant-fold). Each row is a
        dict with keys `wkt`, `band`, `all_touched`, `nodata` (None = use the
        band's own), and `crop`. Returns one `DecodedRaster` (or None for NULL
        rows) per input row, in input order.
        """
        table = pa.table(
            {
                "idx": pa.array(range(len(rows)), type=pa.int64()),
                "path": pa.array([str(path)] * len(rows), type=pa.utf8()),
                "wkt": pa.array([r["wkt"] for r in rows], type=pa.utf8()),
                "band": pa.array([r["band"] for r in rows], type=pa.int32()),
                "all_touched": pa.array(
                    [r["all_touched"] for r in rows], type=pa.bool_()
                ),
                "nodata": pa.array([r["nodata"] for r in rows], type=pa.float64()),
                "crop": pa.array([r["crop"] for r in rows], type=pa.bool_()),
            }
        )
        df = self._con.create_data_frame(table)
        result = (
            df.select(
                "idx",
                r=df.path.funcs.rs_frompath().funcs.rs_clip(
                    df.band,
                    self._con.funcs.st_geomfromtext(df.wkt),
                    df.all_touched,
                    df.nodata,
                    df.crop,
                ),
            )
            .sort("idx")
            .to_arrow_table()["r"]
        )
        return [decode_raster(result[i]) for i in range(len(result))]

    def _one_row_df(self, columns):
        """A one-row DataFrame from `{name: (value, pa_type)}` — arguments
        travel as table columns so kernels run their real array path
        (literals constant-fold)."""
        return self._con.create_data_frame(
            pa.table(
                {
                    name: pa.array([value], type=t)
                    for name, (value, t) in columns.items()
                }
            )
        )

    def value(self, path, x, y, *, band=1):
        df = self._one_row_df(
            {
                "path": (str(path), pa.utf8()),
                "x": (float(x), pa.float64()),
                "y": (float(y), pa.float64()),
                "band": (int(band), pa.int32()),
            }
        )
        result = df.select(
            v=df.path.funcs.rs_frompath().funcs.rs_value(
                self._con.funcs.st_point(df.x, df.y), df.band
            )
        ).to_arrow_table()["v"]
        return result[0].as_py()

    def band_nodata(self, path, *, band=1):
        df = self._one_row_df(
            {"path": (str(path), pa.utf8()), "band": (int(band), pa.int32())}
        )
        result = df.select(
            v=df.path.funcs.rs_frompath().funcs.rs_bandnodatavalue(df.band)
        ).to_arrow_table()["v"]
        return result[0].as_py()

    def _pixel_geometry(self, function, path, col, row):
        import shapely

        df = self._one_row_df(
            {
                "path": (str(path), pa.utf8()),
                "col": (int(col), pa.int32()),
                "row": (int(row), pa.int32()),
            }
        )
        raster = df.path.funcs.rs_frompath()
        expr = getattr(raster.funcs, function)(df.col, df.row)
        wkt = df.select(g=expr.funcs.st_astext()).to_arrow_table()["g"][0].as_py()
        return shapely.from_wkt(wkt)

    def pixel_as_point(self, path, col, row):
        point = self._pixel_geometry("rs_pixelaspoint", path, col, row)
        return (point.x, point.y)

    def pixel_as_centroid(self, path, col, row):
        point = self._pixel_geometry("rs_pixelascentroid", path, col, row)
        return (point.x, point.y)

    def pixel_as_polygon(self, path, col, row):
        return self._pixel_geometry("rs_pixelaspolygon", path, col, row)

    def as_raster(
        self,
        geometry_wkt,
        path,
        pixel_type,
        *,
        all_touched=False,
        burn_value=1.0,
        nodata=None,
        use_geometry_extent=True,
    ):
        df = self._one_row_df(
            {
                "path": (str(path), pa.utf8()),
                "wkt": (geometry_wkt, pa.utf8()),
                "pixel_type": (pixel_type, pa.utf8()),
                "all_touched": (all_touched, pa.bool_()),
                "burn": (float(burn_value), pa.float64()),
                "nodata": (None if nodata is None else float(nodata), pa.float64()),
                "extent": (use_geometry_extent, pa.bool_()),
            }
        )
        result = df.select(
            r=self._con.funcs.rs_asraster(
                self._con.funcs.st_geomfromtext(df.wkt),
                df.path.funcs.rs_frompath(),
                df.pixel_type,
                df.all_touched,
                df.burn,
                df.nodata,
                df.extent,
            )
        ).to_arrow_table()["r"]
        return decode_raster(result[0])


class SedonaSpark(DialectEngine):
    """Runs Sedona Spark SQL `RS_*` functions — the compatibility-target dialect.

    Bootstraps one local SparkSession per process with
    `sedona.spark.SedonaContext`, downloading the Sedona jars from Maven on
    first use. Because that needs pyspark, a JVM, network access, and tens of
    seconds of startup, these tests are opt-in: `create_or_skip` skips —
    even under `SEDONADB_PYTHON_NO_SKIP_TESTS` — unless
    `SEDONADB_RUN_SPARK_TESTS` is set, in which case construction failures
    propagate.

    Rasters travel out of Spark as GeoTIFF bytes (`RS_AsGeoTiff`) decoded
    with rasterio, except band nodata, which is read through
    `RS_BandNoDataValue` so the comparison sees the engine's own claim rather
    than the GeoTIFF writer's encoding of it. Geometry results travel as WKT.

    Sedona Spark treats a geometry without an SRID as EPSG:4326 and
    reprojects it into the raster's CRS, so geometry-taking operations only
    behave identically across engines when the fixtures are CRS-less (then
    nothing reprojects anywhere).
    """

    # numpy dtype name -> the Sedona Spark pixel-type code (java.awt sample
    # types); the intersection of the two engines' band types.
    PIXEL_TYPES = {
        "uint8": "B",
        "int16": "S",
        "uint16": "US",
        "int32": "I",
        "float32": "F",
        "float64": "D",
    }

    _spark = None

    def __init__(self):
        import rasterio  # noqa: F401 — availability probe: results decode via rasterio
        import shapely  # noqa: F401

        self._session = self._ensure_session()

    @classmethod
    def install_hint(cls) -> str:
        return (
            "- Run `pip install pyspark apache-sedona` (needs a JVM; the first "
            "run downloads the Sedona jars from Maven)\n"
            "- Set SEDONADB_RUN_SPARK_TESTS=true to opt in"
        )

    @classmethod
    def create_or_skip(cls, *args, **kwargs):
        import pytest

        if os.environ.get("SEDONADB_RUN_SPARK_TESTS", "false") not in ("true", "1"):
            pytest.skip("Sedona Spark parity tests are opt-in:\n" + cls.install_hint())
        return cls(*args, **kwargs)

    @classmethod
    def _ensure_session(cls):
        if SedonaSpark._spark is None:
            from sedona.spark import SedonaContext

            config = (
                SedonaContext.builder()
                .master("local[2]")
                .appName("sedonadb-raster-parity")
                .config("spark.jars.packages", cls._packages())
                .config("spark.jars.ivy", cls._ivy_dir())
                .config("spark.ui.enabled", "false")
                .getOrCreate()
            )
            SedonaSpark._spark = SedonaContext.create(config)
        return SedonaSpark._spark

    @staticmethod
    def _packages() -> str:
        env = os.environ.get("SEDONADB_SEDONA_SPARK_PACKAGES")
        if env:
            return env
        import pyspark

        major, minor = (int(part) for part in pyspark.__version__.split(".")[:2])
        # Sedona publishes per-Spark-minor artifacts (Spark 3.5+ for this
        # Sedona line). A pyspark older than every published artifact is an
        # error — a newer jar on an older runtime fails at class load. A
        # pyspark newer than the newest artifact tries the newest one, which
        # usually loads; override the coordinates if it doesn't.
        known = ("3.5", "4.0")
        spark_suffix = f"{major}.{minor}"
        if spark_suffix not in known:
            if (major, minor) < (3, 5):
                raise RuntimeError(
                    f"No Sedona {SEDONA_SPARK_VERSION} artifact supports pyspark "
                    f"{pyspark.__version__}; install pyspark >= 3.5 or set "
                    "SEDONADB_SEDONA_SPARK_PACKAGES to explicit Maven coordinates"
                )
            spark_suffix = known[-1]
        scala_suffix = "2.13" if spark_suffix == "4.0" else "2.12"
        return (
            f"org.apache.sedona:sedona-spark-shaded-{spark_suffix}_{scala_suffix}:"
            f"{SEDONA_SPARK_VERSION},"
            f"org.datasyslab:geotools-wrapper:{GEOTOOLS_WRAPPER_VERSION}"
        )

    @staticmethod
    def _ivy_dir() -> str:
        """Directory Ivy resolves `spark.jars.packages` into.

        Pinned so CI can cache the downloaded jars: newer Ivy releases (bundled
        with newer Spark) moved the default location, silently breaking a cache
        keyed on the old path. Override with SEDONADB_SPARK_IVY_DIR.
        """
        return os.environ.get(
            "SEDONADB_SPARK_IVY_DIR",
            os.path.join(os.path.expanduser("~"), ".ivy2"),
        )

    def _raster_df(self, path):
        return (
            self._session.read.format("binaryFile")
            .load(str(path))
            .selectExpr("RS_FromGeoTiff(content) AS rast")
        )

    def _scalar(self, df, expr):
        return df.selectExpr(f"{expr} AS v").first().v

    @staticmethod
    def _transport(expr: str) -> str:
        """GeoTIFF-encode a raster expression for the trip out of the JVM.

        geotools' GeoTIFF writer refuses a CRS-less (engineering-CRS)
        coverage, so the raster is stamped with an arbitrary SRID first. The
        stamp is transport-only: pixels, geotransform, and nodata — the
        values under comparison — are unaffected.
        """
        return f"RS_AsGeoTiff(RS_SetSRID({expr}, 3857))"

    def _decode_expr(self, df, expr) -> Optional[DecodedRaster]:
        """Evaluate a raster-valued SQL expression and decode the result."""
        # Two actions follow (transport + per-band nodata); cache so the
        # operation under test executes once, not once per action, and
        # unpersist so the suite doesn't accumulate cached blocks in the
        # long-lived session.
        result = df.selectExpr(f"{expr} AS r").cache()
        try:
            row = result.selectExpr(
                f"{self._transport('r')} AS t", "RS_NumBands(r) AS n"
            ).first()
            if row is None or row.t is None:
                return None
            decoded = decode_geotiff_bytes(bytes(row.t))
            nodata = result.selectExpr(
                *[f"RS_BandNoDataValue(r, {b}) AS nd{b}" for b in range(1, row.n + 1)]
            ).first()
            return DecodedRaster(decoded.pixels, decoded.gdal_transform, list(nodata))
        finally:
            result.unpersist()

    def _point_of(self, df, expr) -> Tuple[float, float]:
        import shapely

        point = shapely.from_wkt(self._scalar(df, f"ST_AsText({expr})"))
        return (point.x, point.y)

    def clip(
        self, path, geometry_wkt, *, band=0, all_touched=False, nodata=None, crop=True
    ):
        # A SQL NULL argument nulls the whole expression (Sedona's
        # InferredExpression), so optional arguments select the overload
        # arity instead of passing NULL. The ladder puts noDataValue before
        # crop, so a non-default crop requires an explicit nodata.
        args = f"rast, {int(band)}, ST_GeomFromText('{geometry_wkt}'), {str(all_touched).lower()}"
        if nodata is not None:
            args += f", {float(nodata)!r}, {str(crop).lower()}"
        elif not crop:
            raise ValueError(
                "Sedona Spark's RS_Clip cannot express crop=False without an "
                "explicit nodata"
            )
        return self._decode_expr(self._raster_df(path), f"RS_Clip({args})")

    def value(self, path, x, y, *, band=1):
        return self._scalar(
            self._raster_df(path),
            f"RS_Value(rast, ST_GeomFromText('POINT ({float(x)!r} {float(y)!r})'), {int(band)})",
        )

    def values(self, path, points, *, band=1):
        point_exprs = ", ".join(
            f"ST_GeomFromText('POINT ({float(x)!r} {float(y)!r})')" for x, y in points
        )
        return list(
            self._scalar(
                self._raster_df(path),
                f"RS_Values(rast, array({point_exprs}), {band})",
            )
        )

    def band_nodata(self, path, *, band=1):
        return self._scalar(self._raster_df(path), f"RS_BandNoDataValue(rast, {band})")

    def pixel_as_point(self, path, col, row):
        return self._point_of(
            self._raster_df(path), f"RS_PixelAsPoint(rast, {col}, {row})"
        )

    def pixel_as_centroid(self, path, col, row):
        return self._point_of(
            self._raster_df(path), f"RS_PixelAsCentroid(rast, {col}, {row})"
        )

    def pixel_as_polygon(self, path, col, row):
        import shapely

        return shapely.from_wkt(
            self._scalar(
                self._raster_df(path),
                f"ST_AsText(RS_PixelAsPolygon(rast, {col}, {row}))",
            )
        )

    def as_raster(
        self,
        geometry_wkt,
        path,
        pixel_type,
        *,
        all_touched=False,
        burn_value=1.0,
        nodata=None,
        use_geometry_extent=True,
    ):
        # Optional arguments select the overload arity (a SQL NULL would null
        # the whole expression); useGeometryExtent comes after noDataValue in
        # the ladder, so leaving it default requires an explicit nodata.
        args = (
            f"ST_GeomFromText('{geometry_wkt}'), rast, "
            f"'{self.PIXEL_TYPES[pixel_type]}', {str(all_touched).lower()}, "
            f"{float(burn_value)!r}"
        )
        if nodata is not None:
            args += f", {float(nodata)!r}, {str(use_geometry_extent).lower()}"
        elif not use_geometry_extent:
            raise ValueError(
                "Sedona Spark's RS_AsRaster cannot express "
                "use_geometry_extent=False without an explicit nodata"
            )
        return self._decode_expr(self._raster_df(path), f"RS_AsRaster({args})")

    def resample(self, path, *, width, height, algorithm="nearestneighbor"):
        return self._decode_expr(
            self._raster_df(path),
            f"RS_Resample(rast, CAST({width} AS DOUBLE), CAST({height} AS DOUBLE), "
            f"false, '{algorithm}')",
        )

    def _run_scalar(self, path, expr):
        return self._scalar(self._raster_df(path), expr)

    def _run_struct(self, path, expr):
        row = self._scalar(self._raster_df(path), expr)
        if row is None:
            return None
        # Sedona Spark returns a uniform Double[] in field order (so its count is
        # a float, not the Int64 SedonaDB records); the dict comparison is by
        # value, so the numeric count still matches.
        return dict(zip(ZONAL_ALL_FIELDS, row))

    def tile_explode(self, path, tile_width, tile_height):
        df = self._raster_df(path)
        num_bands = int(self._scalar(df, "RS_NumBands(rast)"))
        tiles = df.selectExpr(
            f"RS_TileExplode(rast, {int(tile_width)}, {int(tile_height)}) AS (x, y, tile)"
        ).selectExpr(
            "x",
            "y",
            f"{self._transport('tile')} AS t",
            *[
                f"RS_BandNoDataValue(tile, {b}) AS nd{b}"
                for b in range(1, num_bands + 1)
            ],
        )
        out = []
        for row in tiles.collect():
            decoded = decode_geotiff_bytes(bytes(row.t))
            nodata = [row[f"nd{b}"] for b in range(1, num_bands + 1)]
            out.append(
                (
                    row.x,
                    row.y,
                    DecodedRaster(decoded.pixels, decoded.gdal_transform, nodata),
                )
            )
        return sorted(out, key=lambda t: (t[1], t[0]))

    def as_geotiff(self, path, *, compression=None, quality=None):
        if compression is None:
            expr = "RS_AsGeoTiff(rast)"
        else:
            expr = f"RS_AsGeoTiff(rast, '{compression}', {float(quality)!r})"
        return decode_geotiff_bytes(bytes(self._scalar(self._raster_df(path), expr)))

    def from_binary(self, data):
        # The bytes go through a temporary file and the binaryFile reader —
        # the same JVM-side route as every other load — because shipping
        # them through createDataFrame would involve Python workers, which
        # depend on the local Spark install's interpreter setup.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "from_binary.tiff")
            with open(path, "wb") as f:
                f.write(data)
            return self._decode_expr(self._raster_df(path), "rast")


class Rasterio(RasterEngine):
    """Reference implementation composed from rasterio primitives.

    The window is `rasterio.features.geometry_window` (geometry bounds ∩
    raster extent, snapped outward to the grid), the selection is
    `rasterio.features.geometry_mask`, and pixels compose as "inside the
    geometry: source value verbatim; outside: `nodata`". This is deliberately
    not `rasterio.mask.mask`, which additionally reads the source masked and
    so remaps pixels *valued* at the band nodata to the output nodata — a
    policy Sedona (Spark and DB) does not follow.

    As a reference engine it takes no position on nodata precedence: `nodata`
    must be the already-resolved fill value, so `clip()` errors when it is
    None rather than guessing.

    One caveat on independence: both this engine and SedonaDB ultimately burn
    geometries with GDAL's rasterizer (rasterio bundles its own GDAL), so the
    pixel-center/all_touched selection rule itself is shared. The genuinely
    independent implementations under comparison are everything on top of the
    burn: window snapping, crop copy, transform shift, nodata handling, and
    band handling.
    """

    def __init__(self):
        import rasterio  # noqa: F401 — availability probe for create_or_skip
        import shapely  # noqa: F401

    @classmethod
    def install_hint(cls) -> str:
        return "- Run `pip install rasterio shapely`"

    def clip(
        self, path, geometry_wkt, *, band=0, all_touched=False, nodata=None, crop=True
    ):
        import rasterio
        import rasterio.features
        import rasterio.windows
        import shapely

        if nodata is None:
            raise ValueError(
                "Rasterio is a reference engine: pass the resolved nodata fill"
            )

        geom = shapely.from_wkt(geometry_wkt)
        with rasterio.open(str(path)) as src:
            if band < 0 or band > src.count:
                raise ValueError(
                    f"band {band} out of range for a {src.count}-band raster"
                )
            if crop:
                window = rasterio.features.geometry_window(src, [geom])
            else:
                window = rasterio.windows.Window(0, 0, src.width, src.height)
            transform = src.window_transform(window)
            data = src.read(window=window)
            inside = rasterio.features.geometry_mask(
                [geom],
                out_shape=(data.shape[1], data.shape[2]),
                transform=transform,
                all_touched=all_touched,
                invert=True,
            )
            # numpy raises for out-of-range fills but silently truncates
            # fractional ones (250.7 -> 250 on uint8); a lossy fill would make
            # the reference mirror the very bug it exists to catch.
            fill = np.asarray(nodata, dtype=data.dtype)
            if float(fill) != float(nodata):
                raise ValueError(
                    f"nodata {nodata} is not exactly representable as {data.dtype}"
                )
            pixels = np.where(inside, data, fill)

        if band != 0:
            pixels = pixels[band - 1 : band]
        return DecodedRaster(pixels, tuple(transform.to_gdal()), [nodata] * len(pixels))

    def value(self, path, x, y, *, band=1):
        return self.values(path, [(x, y)], band=band)[0]

    def values(self, path, points, *, band=1):
        import rasterio

        with rasterio.open(str(path)) as src:
            data = src.read(band)
            nodata = src.nodatavals[band - 1]
            out = []
            for x, y in points:
                # index() floors through the inverse transform, so a pixel
                # owns its upper-left edges — the same ownership rule the
                # dialects use. Fixture points avoid pixel boundaries anyway.
                row, col = src.index(x, y)
                if not (0 <= row < src.height and 0 <= col < src.width):
                    out.append(None)
                    continue
                sampled = data[row, col]
                out.append(None if _is_nodata(sampled, nodata) else float(sampled))
        return out

    def band_nodata(self, path, *, band=1):
        import rasterio

        with rasterio.open(str(path)) as src:
            return src.nodatavals[band - 1]

    def _pixel_corner(self, src, col, row):
        x, y = src.transform * (col - 1, row - 1)
        return (x, y)

    def pixel_as_point(self, path, col, row):
        import rasterio

        with rasterio.open(str(path)) as src:
            return self._pixel_corner(src, col, row)

    def pixel_as_centroid(self, path, col, row):
        import rasterio

        with rasterio.open(str(path)) as src:
            x, y = src.transform * (col - 0.5, row - 0.5)
            return (x, y)

    def pixel_as_polygon(self, path, col, row):
        import rasterio
        import shapely

        with rasterio.open(str(path)) as src:
            # Ring order matches both dialects: UL, UR, LR, LL (closed).
            corners = [
                (col - 1, row - 1),
                (col, row - 1),
                (col, row),
                (col - 1, row),
            ]
            return shapely.Polygon([src.transform * corner for corner in corners])

    def as_raster(
        self,
        geometry_wkt,
        path,
        pixel_type,
        *,
        all_touched=False,
        burn_value=1.0,
        nodata=None,
        use_geometry_extent=True,
    ):
        """Rasterize with `rasterio.features.rasterize`.

        Pixels outside the geometry are filled with the subject's policy —
        SedonaDB initializes the output grid with the nodata value (0 when
        none is given). Comparators with a different fill policy (Sedona
        Spark burns into zeros and records nodata as metadata only) carry a
        `Deviation` entry in the test modules instead of bending this
        reconstruction.
        """
        import rasterio
        import rasterio.features
        import shapely

        geom = shapely.from_wkt(geometry_wkt)
        with rasterio.open(str(path)) as src:
            if use_geometry_extent:
                window = rasterio.features.geometry_window(src, [geom])
                transform = src.window_transform(window)
                shape = (int(window.height), int(window.width))
            else:
                transform = src.transform
                shape = (src.height, src.width)

        fill = 0.0 if nodata is None else nodata
        for name, value in [("fill", fill), ("burn_value", burn_value)]:
            if np.asarray(value, dtype=pixel_type) != np.asarray(
                value, dtype="float64"
            ):
                raise ValueError(
                    f"{name} {value} is not exactly representable as {pixel_type}"
                )
        pixels = rasterio.features.rasterize(
            [(geom, burn_value)],
            out_shape=shape,
            transform=transform,
            fill=fill,
            all_touched=all_touched,
            dtype=pixel_type,
        )
        return DecodedRaster(pixels[np.newaxis], tuple(transform.to_gdal()), [nodata])

    def resample(self, path, *, width, height, algorithm="nearestneighbor"):
        import rasterio
        import rasterio.crs
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        resampling = {
            "nearestneighbor": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "bicubic": Resampling.cubic,
        }[algorithm.lower()]
        with rasterio.open(str(path)) as src:
            source = src.read()
            dst_transform = src.transform * src.transform.scale(
                src.width / width, src.height / height
            )
            # reproject() insists on a CRS; with the same one on both sides
            # nothing reprojects, so an arbitrary CRS keeps a CRS-less
            # fixture a pure grid resample.
            crs = src.crs or rasterio.crs.CRS.from_epsg(3857)
            destination = np.zeros((src.count, height, width), dtype=source.dtype)
            reproject(
                source,
                destination,
                src_transform=src.transform,
                src_crs=crs,
                dst_transform=dst_transform,
                dst_crs=crs,
                resampling=resampling,
            )
            return DecodedRaster(
                destination, tuple(dst_transform.to_gdal()), list(src.nodatavals)
            )

    def zonal_stats(
        self,
        path,
        geometry_wkt,
        *,
        band=None,
        stat="mean",
        all_touched=False,
        exclude_nodata=True,
        lenient=True,
    ):
        values = self._zone_values(
            path, geometry_wkt, band, all_touched, exclude_nodata
        )
        if values.size == 0:
            # An empty selection makes count 0 and every other statistic NULL
            # (Sedona's empty-zone shortcut); comparisons here use zones that
            # select pixels, so this only guards a degenerate case.
            return 0.0 if stat == "count" else None
        return float(_STAT_REDUCERS[stat](values))

    def zonal_stats_all(
        self,
        path,
        geometry_wkt,
        *,
        band=None,
        all_touched=False,
        exclude_nodata=True,
        lenient=True,
    ):
        values = self._zone_values(
            path, geometry_wkt, band, all_touched, exclude_nodata
        )
        return _reduce_all_stats(values)

    def _zone_values(self, path, geometry_wkt, band, all_touched, exclude_nodata):
        """The band's pixel values selected by the zone, in float64.

        `band=None` resolves to band 1 (matching the subject's single-band
        default for the band-less overload). Pixels equal to the band nodata are
        dropped only when `exclude_nodata`."""
        import rasterio
        import rasterio.features
        import shapely

        b = 1 if band is None else band
        geom = shapely.from_wkt(geometry_wkt)
        with rasterio.open(str(path)) as src:
            data = src.read(b)
            nodata = src.nodatavals[b - 1]
            inside = rasterio.features.geometry_mask(
                [geom],
                out_shape=(src.height, src.width),
                transform=src.transform,
                all_touched=all_touched,
                invert=True,
            )
        values = data[inside].astype(np.float64)
        if exclude_nodata and nodata is not None:
            if math.isnan(nodata):
                values = values[~np.isnan(values)]
            else:
                values = values[values != nodata]
        return values

    def tile_explode(self, path, tile_width, tile_height):
        import rasterio
        from rasterio.windows import Window

        out = []
        with rasterio.open(str(path)) as src:
            for tile_y, row_off in enumerate(range(0, src.height, tile_height)):
                for tile_x, col_off in enumerate(range(0, src.width, tile_width)):
                    window = Window(
                        col_off,
                        row_off,
                        min(tile_width, src.width - col_off),
                        min(tile_height, src.height - row_off),
                    )
                    out.append(
                        (
                            tile_x,
                            tile_y,
                            DecodedRaster(
                                src.read(window=window),
                                tuple(src.window_transform(window).to_gdal()),
                                list(src.nodatavals),
                            ),
                        )
                    )
        return out

    def as_geotiff(self, path):
        # The reference for an encode round-trip is the source content
        # itself: lossless codecs must preserve pixels, transform, and
        # nodata bit for bit, so compression options don't change the
        # expectation and this override doesn't take them.
        return decode_geotiff(path)

    def from_binary(self, data):
        return decode_geotiff_bytes(data)


# RS_ZonalStatsAll struct fields, in the order the Rust kernel emits them
# (count first, then the Float64 statistics). Both the Sedona Spark decoder and
# the parity test read the struct through this list, so it is defined once.
ZONAL_ALL_FIELDS = [
    "count",
    "sum",
    "mean",
    "median",
    "mode",
    "stddev",
    "variance",
    "min",
    "max",
]


def _mode(values: "np.ndarray") -> float:
    """The most frequent value, breaking ties toward the larger (matching
    Sedona's `StatUtils.mode`). `np.unique` returns values ascending, so the
    max of the tied-most-frequent values is the larger mode."""
    uniques, counts = np.unique(values, return_counts=True)
    return float(uniques[counts == counts.max()].max())


# RS_ZonalStats statistic name -> reducer over the selected float64 values.
# stddev/variance are the sample (ddof=1) statistics Sedona computes; a single
# value yields variance 0. count is an integer pixel count.
_STAT_REDUCERS = {
    "count": lambda v: int(v.size),
    "sum": np.sum,
    "mean": np.mean,
    "median": np.median,
    "mode": _mode,
    "stddev": lambda v: v.std(ddof=1) if v.size > 1 else 0.0,
    "variance": lambda v: v.var(ddof=1) if v.size > 1 else 0.0,
    "min": np.min,
    "max": np.max,
}


def _reduce_all_stats(values: "np.ndarray") -> Optional[Mapping[str, Any]]:
    """Every RS_ZonalStatsAll field over the selected values.

    An empty selection is `count = 0` with every other field None (Sedona's
    empty-zone shortcut). `count` is an integer; the rest are floats."""
    n = int(values.size)
    if n == 0:
        return {field: (0 if field == "count" else None) for field in ZONAL_ALL_FIELDS}
    variance = float(values.var(ddof=1)) if n > 1 else 0.0
    return {
        "count": n,
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mode": _mode(values),
        "stddev": math.sqrt(variance),
        "variance": variance,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _is_nodata(sampled, nodata) -> bool:
    """Whether a sampled value equals the band nodata, NaN-aware (a NaN
    sentinel matches NaN pixels, which bare `==` never would)."""
    if nodata is None:
        return False
    if math.isnan(nodata):
        return bool(np.isnan(sampled))
    return bool(sampled == nodata)


def assert_decoded_equal(got: DecodedRaster, expected: DecodedRaster, *, context=""):
    """Strict raster comparison: exact pixels and dtype, geotransform to
    1e-12, nodata by value (None must match None, NaN matches NaN).
    `compression` is decode metadata, not content, and is not compared."""
    assert got is not None, f"got no raster: {context}"
    assert expected is not None, f"expected no raster: {context}"
    assert got.pixels.dtype == expected.pixels.dtype, context
    np.testing.assert_array_equal(got.pixels, expected.pixels, err_msg=str(context))
    assert got.gdal_transform == approx_geotransform(expected.gdal_transform), context
    assert len(got.nodata) == len(expected.nodata), context
    for got_nodata, expected_nodata in zip(got.nodata, expected.nodata):
        if expected_nodata is None:
            assert got_nodata is None, context
        elif isinstance(expected_nodata, float) and math.isnan(expected_nodata):
            assert got_nodata is not None and math.isnan(got_nodata), context
        else:
            assert got_nodata == expected_nodata, context


def approx_geotransform(value):
    """pytest.approx tight enough that only real georeferencing bugs pass it."""
    import pytest

    return pytest.approx(value, rel=1e-12, abs=1e-12)


def decode_raster(scalar) -> Optional[DecodedRaster]:
    """Decode one `sedona.raster` Arrow scalar to a `DecodedRaster` (None if NULL)."""
    if not scalar.is_valid:
        return None
    raster = scalar.as_py()
    return DecodedRaster(
        raster.to_numpy(),
        tuple(raster.transform),
        [band.nodata for band in raster.bands],
    )


def decode_geotiff(path) -> DecodedRaster:
    """Decode a GeoTIFF file to a `DecodedRaster` with rasterio."""
    with open(path, "rb") as f:
        return decode_geotiff_bytes(f.read())


def decode_geotiff_bytes(data: bytes) -> DecodedRaster:
    """Decode in-memory GeoTIFF bytes to a `DecodedRaster` with rasterio."""
    from rasterio.io import MemoryFile

    with MemoryFile(bytes(data)) as mem, mem.open() as src:
        return DecodedRaster(
            src.read(),
            tuple(src.transform.to_gdal()),
            list(src.nodatavals),
            compression=src.compression.value if src.compression else None,
        )


def write_geotiff(
    path, data: "np.ndarray", *, gdal_transform, nodata=None, crs=None
) -> None:
    """Write a `(bands, height, width)` array as a GeoTIFF.

    `gdal_transform` is GDAL-order `(origin_x, scale_x, skew_x, origin_y,
    skew_y, scale_y)`; `nodata` (optional) becomes the per-band nodata of
    every band. `crs` (optional) is any CRS rasterio accepts; parity fixtures
    stay CRS-less unless an engine requires one, and then use the same CRS
    everywhere so nothing reprojects.
    """
    import rasterio
    from rasterio.transform import Affine

    bands, height, width = data.shape
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=str(data.dtype),
        transform=Affine.from_gdal(*gdal_transform),
        nodata=nodata,
        crs=crs,
    ) as dst:
        dst.write(data)


def dtype_min(dtype):
    """The minimum representable value of a numpy dtype — SedonaDB's default
    nodata sentinel when neither an explicit value nor a band nodata exists."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return float(np.finfo(dtype).min)
    return int(np.iinfo(dtype).min)


def random_raster_data(
    dtype,
    *,
    bands: int,
    height: int,
    width: int,
    seed: int = 42,
    plants: Optional[Mapping[Tuple[int, int], Any]] = None,
) -> "np.ndarray":
    """Random `(bands, height, width)` pixels with adversarial values planted.

    The dtype extremes always go in opposite corners (values that must
    round-trip through any operation that keeps them, and be overwritten by
    any that fills them). `plants` maps `(row, col)` to a value written into
    every band — use it to place values the test's geometry or nodata choices
    make meaningful.
    """
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        data = ((rng.random((bands, height, width)) - 0.5) * 200.0).astype(dtype)
        info = np.finfo(dtype)
    else:
        info = np.iinfo(dtype)
        data = rng.integers(
            info.min, info.max, size=(bands, height, width), dtype=dtype, endpoint=True
        )
    data[:, 0, 0] = info.max
    data[:, -1, -1] = info.min
    for (row, col), value in (plants or {}).items():
        data[:, row, col] = value
    return data
