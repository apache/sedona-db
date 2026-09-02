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
import multiprocessing
import tempfile
import threading
import time
import traceback
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geoarrow.pyarrow as ga
import geopandas
import geopandas.testing
import pandas as pd
import pyarrow as pa
import pytest
import sedonadb
from sedonadb.datasource import PyogrioFormatSpec
import shapely


def test_read_ogr_projection(con):
    n = 1024
    series = geopandas.GeoSeries.from_xy(
        list(range(n)), list(range(1, n + 1)), crs="EPSG:3857"
    )
    gdf = geopandas.GeoDataFrame({"idx": list(range(n)), "wkb_geometry": series})
    gdf = gdf.set_geometry(gdf["wkb_geometry"])

    with tempfile.TemporaryDirectory() as td:
        temp_fgb_path = f"{td}/temp.fgb"
        gdf.to_file(temp_fgb_path)
        con.read_pyogrio(temp_fgb_path).to_view("test_fgb", overwrite=True)

        # With no projection
        geopandas.testing.assert_geodataframe_equal(
            con.sql("SELECT * FROM test_fgb ORDER BY idx").to_pandas(), gdf
        )

        # With only not geometry selected
        pd.testing.assert_frame_equal(
            con.sql("SELECT idx FROM test_fgb ORDER BY idx").to_pandas(),
            gdf.filter(["idx"]),
        )

        # With reversed columns
        pd.testing.assert_frame_equal(
            con.sql("SELECT wkb_geometry, idx FROM test_fgb ORDER BY idx").to_pandas(),
            gdf.filter(["wkb_geometry", "idx"]),
        )


def test_read_ogr_multi_file(con):
    n = 1024 * 16
    partitions = [f"part_{c}" for c in "abcdefghijklmnop"]
    series = geopandas.GeoSeries.from_xy(
        list(range(n)), list(range(1, n + 1)), crs="EPSG:3857"
    )
    gdf = geopandas.GeoDataFrame(
        {
            "idx": list(range(n)),
            "partition": [partitions[i % len(partitions)] for i in range(n)],
            "wkb_geometry": series,
        }
    )
    gdf = gdf.set_geometry(gdf["wkb_geometry"])

    with tempfile.TemporaryDirectory() as td:
        # Create partitioned files by writing Parquet first and translating
        # one file at a time. We need to cast partition in pandas>=3.0 because
        # the default translation of a string column is LargeUtf8 and this is not
        # currently supported by DataFusion partition_by.
        con.create_data_frame(gdf).to_view("tmp_gdf", overwrite=True)
        con.sql(
            """SELECT idx, partition::VARCHAR AS partition, wkb_geometry FROM tmp_gdf"""
        ).to_parquet(td, partition_by="partition")
        for parquet_path in Path(td).rglob("*.parquet"):
            fgb_path = str(parquet_path).replace(".parquet", ".fgb")
            con.read_parquet(parquet_path).to_pandas().to_file(fgb_path)

        # Reading a directory while specifying the extension should work
        # Partition columns are auto-discovered from the directory structure
        con.read_pyogrio(f"{td}", extension="fgb").to_view(
            "gdf_from_dir", overwrite=True
        )
        geopandas.testing.assert_geodataframe_equal(
            con.sql(
                "SELECT idx, partition, wkb_geometry FROM gdf_from_dir ORDER BY idx"
            ).to_pandas(),
            gdf,
        )

        # Reading using a glob without specifying the extension should work
        con.read_pyogrio(f"{td}/**/*.fgb").to_view("gdf_from_glob", overwrite=True)
        geopandas.testing.assert_geodataframe_equal(
            con.sql(
                "SELECT idx, partition, wkb_geometry FROM gdf_from_glob ORDER BY idx"
            ).to_pandas(),
            gdf,
        )


def test_read_ogr_filter(con):
    n = 1024
    series = geopandas.GeoSeries.from_xy(
        list(range(n)), list(range(1, n + 1)), crs="EPSG:3857"
    )
    gdf = geopandas.GeoDataFrame({"idx": list(range(n)), "wkb_geometry": series})
    gdf = gdf.set_geometry(gdf["wkb_geometry"])

    with tempfile.TemporaryDirectory() as td:
        temp_fgb_path = f"{td}/temp.fgb"
        gdf.to_file(temp_fgb_path)
        con.read_pyogrio(temp_fgb_path).to_view("test_fgb", overwrite=True)

        # With something that should trigger a bounding box filter
        geopandas.testing.assert_geodataframe_equal(
            con.sql(
                """
                SELECT * FROM test_fgb
                WHERE ST_Equals(wkb_geometry, ST_Point(1, 2, 3857))
                """
            ).to_pandas(),
            gdf[gdf.geometry.geom_equals(shapely.Point(1, 2))].reset_index(drop=True),
        )


def test_read_ogr_layer_selection(con):
    series = geopandas.GeoSeries.from_xy([0, 1], [1, 2], crs="EPSG:3857")
    gdf = geopandas.GeoDataFrame({"val": ["a", "b"], "geom": series})
    gdf = gdf.set_geometry(gdf["geom"])

    with tempfile.TemporaryDirectory() as td:
        gpkg_path = f"{td}/test.gpkg"
        gdf.to_file(gpkg_path, layer="my_layer")

        # Reading with the correct layer name should work
        geopandas.testing.assert_geodataframe_equal(
            con.read_pyogrio(gpkg_path, options={"layer": "my_layer"}).to_pandas(),
            gdf,
        )


def test_read_ogr_path_suffix(con):
    series = geopandas.GeoSeries.from_xy([0, 1], [1, 2], crs="EPSG:3857")
    gdf = geopandas.GeoDataFrame({"val": ["a", "b"], "geom": series})
    gdf = gdf.set_geometry(gdf["geom"])

    with tempfile.TemporaryDirectory() as td:
        gpkg_path = f"{td}/data.gpkg"
        gdf.to_file(gpkg_path)

        zip_path = f"{td}/archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(gpkg_path, "nested/data.gpkg")

        geopandas.testing.assert_geodataframe_equal(
            con.read_pyogrio(
                zip_path, options={"path_suffix": "nested/data.gpkg"}
            ).to_pandas(),
            gdf,
        )


def test_read_ogr_file_not_found(con):
    with pytest.raises(
        sedonadb._lib.SedonaError, match="Can't infer schema for zero objects"
    ):
        con.read_pyogrio("this/is/not/a/directory")

    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(
            sedonadb._lib.SedonaError, match="Can't infer schema for zero objects"
        ):
            con.read_pyogrio(Path(td) / "file_does_not_exist")


def test_write_ogr(con):
    with tempfile.TemporaryDirectory() as td:
        # Basic write with defaults
        df = con.sql("SELECT ST_Point(0, 1, 3857)")
        expected = geopandas.GeoDataFrame(
            {"geometry": geopandas.GeoSeries.from_wkt(["POINT (0 1)"], crs=3857)}
        )

        df.to_pyogrio(f"{td}/foofy.fgb")
        geopandas.testing.assert_geodataframe_equal(
            geopandas.read_file(f"{td}/foofy.fgb"), expected
        )

        # Ensure Path input works
        df.to_pyogrio(Path(f"{td}/foofy.fgb"))
        geopandas.testing.assert_geodataframe_equal(
            geopandas.read_file(f"{td}/foofy.fgb"), expected
        )

        # Ensure zipped FlatGeoBuf doesn't require specifying the driver
        df.to_pyogrio(Path(f"{td}/foofy.fgb.zip"))
        geopandas.testing.assert_geodataframe_equal(
            geopandas.read_file(f"{td}/foofy.fgb.zip"), expected
        )

        # Ensure inferred CRS that is None works
        # pyogrio warns for the case where a CRS is None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            con.sql("SELECT ST_Point(0, 1)").to_pyogrio(f"{td}/foofy.fgb")
            expected = geopandas.GeoDataFrame(
                {"geometry": geopandas.GeoSeries.from_wkt(["POINT (0 1)"])}
            )
            geopandas.testing.assert_geodataframe_equal(
                geopandas.read_file(f"{td}/foofy.fgb"), expected
            )


def test_write_ogr_buffer(con):
    buf = io.BytesIO()
    df = con.sql("SELECT ST_Point(0, 1, 3857)")
    expected = geopandas.GeoDataFrame(
        {"geometry": geopandas.GeoSeries.from_wkt(["POINT (0 1)"], crs=3857)}
    )

    df.to_pyogrio(buf, driver="FlatGeoBuf")
    geopandas.testing.assert_geodataframe_equal(
        geopandas.read_file(buf.getvalue()), expected
    )

    # Ensure reasonable error if driver is not specified
    with pytest.raises(ValueError, match="driver must be provided"):
        df.to_pyogrio(buf)


def test_write_ogr_no_geometry(con):
    with tempfile.TemporaryDirectory() as td:
        df = con.sql("SELECT 'one' as one")
        expected = pd.DataFrame({"one": ["one"]})

        df.to_pyogrio(f"{td}/foofy.csv")
        pd.testing.assert_frame_equal(pd.read_csv(f"{td}/foofy.csv"), expected)


def test_write_ogr_many_batches(con):
    # Check with a non-trivial number of batches
    con.funcs.table.sd_random_geometry("MultiLineString", 50000, seed=4837).to_view(
        "pyogrio_test"
    )
    df = con.sql(
        """
        SELECT id, ST_SetCrs(geometry, 'EPSG:4326') AS geometry
        FROM pyogrio_test
        ORDER BY id
        """
    )
    expected = df.to_pandas()

    with tempfile.TemporaryDirectory() as td:
        df.to_pyogrio(f"{td}/foofy.gpkg")
        geopandas.testing.assert_geodataframe_equal(
            geopandas.read_file(f"{td}/foofy.gpkg"), expected
        )


def test_write_ogr_from_view_types(con):
    # Check that we can write something with view types (even though it is read back
    # as the simplified type)
    wkb_array = ga.with_crs(ga.as_wkb(["POINT (0 1)", "POINT (1 2)"]), ga.OGC_CRS84)
    wkb_view_array = (
        ga.wkb_view()
        .with_crs(ga.OGC_CRS84)
        .wrap_array(wkb_array.storage.cast(pa.binary_view()))
    )
    tab_simple = pa.table(
        {"string_col": pa.array(["one", "two"], pa.string()), "wkb_geometry": wkb_array}
    )
    tab = pa.table(
        {
            "string_col": pa.array(["one", "two"], pa.string_view()),
            "wkb_geometry": wkb_view_array,
        }
    )

    with tempfile.TemporaryDirectory() as td:
        con.create_data_frame(tab).to_pyogrio(f"{td}/foofy.fgb")
        tab_roundtrip = con.read_pyogrio(f"{td}/foofy.fgb").to_arrow_table()
        assert tab_roundtrip.sort_by("string_col") == tab_simple


def test_read_ogr_partitioned(con):
    n = 100
    series = geopandas.GeoSeries.from_xy(
        list(range(n)), list(range(1, n + 1)), crs="EPSG:3857"
    )
    gdf = geopandas.GeoDataFrame(
        {
            "idx": list(range(n)),
            "grp": [str(i // 10) for i in range(n)],
            "wkb_geometry": series,
        }
    )
    gdf = gdf.set_geometry(gdf["wkb_geometry"])

    with tempfile.TemporaryDirectory() as td:
        # Write partitioned FGB files using hive-style directories because
        # write_pyogrio doesn't support writing partitions yet
        for grp_val in gdf["grp"].unique():
            grp_dir = Path(td) / f"grp={grp_val}"
            grp_dir.mkdir()
            subset = gdf[gdf["grp"] == grp_val].drop(columns=["grp"])
            subset.to_file(grp_dir / "data.fgb")

        # Test auto-discovery of partition columns with the default value
        # (which is to autodiscover partitions)
        con.read_pyogrio(td, extension="fgb").to_view(
            "partitioned_auto", overwrite=True
        )
        geopandas.testing.assert_geodataframe_equal(
            con.sql(
                "SELECT idx, grp, wkb_geometry FROM partitioned_auto ORDER BY idx"
            ).to_pandas(),
            gdf,
        )

        # Test auto-discovery of partition columns (partitioning=None)
        con.read_pyogrio(td, extension="fgb", partitioning=None).to_view(
            "partitioned_auto", overwrite=True
        )
        geopandas.testing.assert_geodataframe_equal(
            con.sql(
                "SELECT idx, grp, wkb_geometry FROM partitioned_auto ORDER BY idx"
            ).to_pandas(),
            gdf,
        )

        # Test explicit partitioning specification (list)
        con.read_pyogrio(td, extension="fgb", partitioning=["grp"]).to_view(
            "partitioned_explicit", overwrite=True
        )
        geopandas.testing.assert_geodataframe_equal(
            con.sql(
                "SELECT idx, grp, wkb_geometry FROM partitioned_explicit ORDER BY idx"
            ).to_pandas(),
            gdf,
        )

        # Test explicit partitioning specification (str)
        con.read_pyogrio(td, extension="fgb", partitioning="grp").to_view(
            "partitioned_explicit", overwrite=True
        )
        geopandas.testing.assert_geodataframe_equal(
            con.sql(
                "SELECT idx, grp, wkb_geometry FROM partitioned_explicit ORDER BY idx"
            ).to_pandas(),
            gdf,
        )

        # Test partitioning=[] disables auto-discovery
        con.read_pyogrio(td, extension="fgb", partitioning=[]).to_view(
            "partitioned_disabled", overwrite=True
        )
        geopandas.testing.assert_geodataframe_equal(
            con.sql("SELECT * FROM partitioned_disabled ORDER BY idx").to_pandas(),
            gdf.filter(["idx", "wkb_geometry"]),
        )


def test_pyogrio_format_register():
    # Create a dedicated connection here because we're about to modify options
    sd = sedonadb.connect()
    sd.register(PyogrioFormatSpec("fgb"))

    n = 1024
    series = geopandas.GeoSeries.from_xy(
        list(range(n)), list(range(1, n + 1)), crs="EPSG:3857"
    )
    gdf = geopandas.GeoDataFrame({"idx": list(range(n)), "wkb_geometry": series})
    gdf = gdf.set_geometry(gdf["wkb_geometry"])

    with tempfile.TemporaryDirectory() as td:
        temp_fgb_path = f"{td}/temp.fgb"
        gdf.to_file(temp_fgb_path)

        # Should be able to SELECT * from 'file' after registering the format
        df = sd.sql(f"SELECT * FROM '{temp_fgb_path}' ORDER BY idx")
        geopandas.testing.assert_geodataframe_equal(df.to_pandas(), gdf)


class _NativePullBoundary:
    def __init__(self):
        self._barrier = threading.Barrier(2, timeout=10)
        self._lock = threading.Lock()
        self.pull_intervals = {}

    def read_next_batch(self, source, reader):
        # Rendezvous immediately before each real pyogrio Arrow batch pull.
        # The interval covers the native read_next_batch() call itself, so an
        # overlap is evidence that two independent GDAL readers were active at
        # the same time rather than merely that their Python shelters coexisted.
        self._barrier.wait()
        started = time.perf_counter_ns()
        batch = reader.read_next_batch()
        finished = time.perf_counter_ns()

        with self._lock:
            self.pull_intervals.setdefault(source, []).append((started, finished))
        return batch

    def batch_pulls(self):
        with self._lock:
            return {
                source: len(intervals)
                for source, intervals in self.pull_intervals.items()
            }

    def native_read_overlaps(self):
        with self._lock:
            sources = list(self.pull_intervals)
            if len(sources) != 2:
                return 0
            left_intervals = self.pull_intervals[sources[0]]
            right_intervals = self.pull_intervals[sources[1]]

        return sum(
            max(left_start, right_start) < min(left_end, right_end)
            for left_start, left_end in left_intervals
            for right_start, right_end in right_intervals
        )


class _NativePullBoundaryReader:
    def __init__(self, inner, boundary, source):
        native_reader = pa.RecordBatchReader.from_stream(inner)
        self._reader = pa.RecordBatchReader.from_batches(
            native_reader.schema,
            self._read_batches(boundary, source, native_reader, inner),
        )

    @staticmethod
    def _read_batches(boundary, source, reader, shelter):
        try:
            while True:
                try:
                    yield boundary.read_next_batch(source, reader)
                except StopIteration:
                    return
        finally:
            # Keep the original pyogrio shelter strongly referenced until its
            # imported native Arrow stream is closed. This preserves the same
            # stream-before-context cleanup order as the production bridge.
            try:
                reader.close()
            finally:
                del shelter

    def __arrow_c_stream__(self, requested_schema=None):
        return self._reader.__arrow_c_stream__(requested_schema)


def _run_independent_pyogrio_scans(result_sender, paths, expected_values):
    boundary = _NativePullBoundary()
    original_open_reader = PyogrioFormatSpec.open_reader
    executor = None
    futures = []

    def open_reader(format_spec, args):
        inner = original_open_reader(format_spec, args)
        if args.file_schema is None:
            # Schema inference does not pull batches. Leave its real shelter
            # untouched and instrument only the execution reader.
            return inner
        return _NativePullBoundaryReader(inner, boundary, args.src.to_url())

    try:
        PyogrioFormatSpec.open_reader = open_reader
        connections = [sedonadb.connect(), sedonadb.connect()]
        assert len(connections) == len(paths) == len(expected_values) == 2
        for connection in connections:
            connection.sql("SET datafusion.execution.batch_size TO 64").execute()

        def scan(connection, path):
            return connection.read_pyogrio(path).to_arrow_table()

        executor = ThreadPoolExecutor(max_workers=2)
        futures = [
            executor.submit(scan, connections[index], paths[index])
            for index in range(2)
        ]
        tables = [future.result() for future in futures]
        assert len(futures) == len(tables) == 2

        for index in range(2):
            table = tables[index]
            values = expected_values[index]
            assert table.num_rows == len(values)
            assert sorted(table.column("idx").to_pylist()) == values
        result_sender.send(
            {
                "ok": True,
                "native_batch_pulls": boundary.batch_pulls(),
                "native_read_overlaps": boundary.native_read_overlaps(),
            }
        )
    except BaseException:
        result_sender.send({"ok": False, "error": traceback.format_exc()})
    finally:
        PyogrioFormatSpec.open_reader = original_open_reader
        if executor is not None:
            executor.shutdown(wait=all(future.done() for future in futures))
        result_sender.close()


def _run_paused_pyogrio_readers(result_sender, paths, expected_values):
    try:
        connections = [sedonadb.connect(), sedonadb.connect()]
        for connection in connections:
            connection.sql("SET datafusion.execution.batch_size TO 64").execute()

        first_reader = connections[0].read_pyogrio(paths[0]).to_arrow_reader()
        first_batch = first_reader.read_next_batch()
        assert 0 < first_batch.num_rows < len(expected_values[0])

        # Keep first_reader and its scan-local lifecycle guard alive while an
        # independent connection opens and drains another real GDAL reader.
        second_reader = connections[1].read_pyogrio(paths[1]).to_arrow_reader()
        second_table = second_reader.read_all()
        first_remainder = first_reader.read_all()
        first_values = first_batch.column("idx").to_pylist()
        first_values.extend(first_remainder.column("idx").to_pylist())
        second_values = second_table.column("idx").to_pylist()
        assert sorted(first_values) == expected_values[0]
        assert sorted(second_values) == expected_values[1]

        result_sender.send(
            {
                "ok": True,
                "first_batch_rows": first_batch.num_rows,
                "row_counts": [len(first_values), len(second_values)],
            }
        )
    except BaseException:
        result_sender.send({"ok": False, "error": traceback.format_exc()})
    finally:
        result_sender.close()


def _block_child_process(result_sender):
    threading.Event().wait()


def _exit_child_process(result_sender):
    raise SystemExit(17)


def _terminate_child(process):
    process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join(5)


def _run_child_process(target, *args, timeout):
    context = multiprocessing.get_context("spawn")
    result_receiver, result_sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(result_sender, *args))
    process.start()
    result_sender.close()
    try:
        process.join(timeout)
        if process.is_alive():
            _terminate_child(process)
            raise TimeoutError(f"child process exceeded {timeout} seconds")

        child_result = None
        try:
            if result_receiver.poll():
                child_result = result_receiver.recv()
        except (BrokenPipeError, EOFError):
            pass
        if process.exitcode != 0:
            raise AssertionError(
                f"child process exited with {process.exitcode}: {child_result}"
            )
        if child_result is None:
            raise AssertionError("child process exited without reporting a result")
        return child_result
    finally:
        if process.is_alive():
            _terminate_child(process)
        result_receiver.close()


def _write_pyogrio_pair(directory, extension, expected_values):
    paths = []
    for index, side in enumerate(("left", "right")):
        values = expected_values[index]
        path = Path(directory) / f"{side}.{extension}"
        geopandas.GeoDataFrame(
            {"idx": values},
            geometry=geopandas.GeoSeries.from_xy(values, values, crs="EPSG:4326"),
        ).to_file(path)
        paths.append(str(path))
    return paths


def test_independent_scans_child_timeout_is_bounded():
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="exceeded 0.2 seconds"):
        _run_child_process(_block_child_process, timeout=0.2)
    assert time.monotonic() - started < 5


def test_child_process_nonzero_exit_is_reported():
    with pytest.raises(AssertionError, match="child process exited with 17: None"):
        _run_child_process(_exit_child_process, timeout=5)


@pytest.mark.parametrize("extension", ["fgb", "gpkg"])
def test_independent_scans_from_threads(extension):
    pytest.importorskip("pyogrio")
    expected_values = (list(range(2048)), list(range(10000, 12048)))

    with tempfile.TemporaryDirectory() as td:
        paths = _write_pyogrio_pair(td, extension, expected_values)
        result = _run_child_process(
            _run_independent_pyogrio_scans, paths, expected_values, timeout=30
        )

    assert result["ok"], result["error"]
    assert {Path(source).name for source in result["native_batch_pulls"]} == {
        f"left.{extension}",
        f"right.{extension}",
    }
    assert all(count > 1 for count in result["native_batch_pulls"].values())
    assert result["native_read_overlaps"] >= 1


@pytest.mark.parametrize("extension", ["fgb", "gpkg"])
def test_independent_reader_progress_while_first_reader_is_paused(extension):
    pytest.importorskip("pyogrio")
    expected_values = (list(range(2048)), list(range(10000, 12048)))

    with tempfile.TemporaryDirectory() as td:
        paths = _write_pyogrio_pair(td, extension, expected_values)
        result = _run_child_process(
            _run_paused_pyogrio_readers, paths, expected_values, timeout=15
        )

    assert result["ok"], result["error"]
    assert result["first_batch_rows"] < len(expected_values[0])
    assert result["row_counts"] == [len(values) for values in expected_values]


# The geometry-column name is GDAL's OGR reader's, not ours: fgb/geojson/shp
# store no named geometry field, so GDAL falls back to `wkb_geometry`, whereas
# GeoPackage persists a named geometry column (GDAL defaults it to `geom`).
@pytest.mark.parametrize(
    ("extension", "geometry_column"),
    [
        ("fgb", "wkb_geometry"),
        ("gpkg", "geom"),
        ("geojson", "wkb_geometry"),
        ("shp", "wkb_geometry"),
    ],
)
def test_url_table_autoregistered(extension, geometry_column):
    # The common single-file OGR formats are auto-registered when a context is
    # created, so a bare file URL resolves as a table without a manual
    # register() call (the way GeoParquet already does). A fresh connection is
    # used so this exercises context-creation wiring rather than any state left
    # on the shared session fixture.
    pytest.importorskip("pyogrio")
    sd = sedonadb.connect()

    gdf = geopandas.GeoDataFrame(
        {"idx": [0, 1, 2]},
        geometry=geopandas.GeoSeries.from_xy([0, 1, 2], [1, 2, 3], crs="EPSG:4326"),
    )

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"data.{extension}"
        gdf.to_file(path)

        # to_arrow_table() runs the full execution path.
        table = sd.sql(f"SELECT * FROM '{path.as_uri()}' ORDER BY idx").to_arrow_table()
        assert table.num_rows == 3
        assert table.column_names == ["idx", geometry_column]
        assert table.column("idx").to_pylist() == [0, 1, 2]

        # Pass-through geometry values round-trip exactly (no reprojection).
        result = sd.sql(f"SELECT * FROM '{path.as_uri()}' ORDER BY idx").to_pandas()
        assert result.geometry.tolist() == gdf.geometry.tolist()


def test_url_table_smoke_bare_path(con):
    # SQL-text smoke covering the plain (non file://) path form the SQL URL
    # table also accepts.
    pytest.importorskip("pyogrio")

    gdf = geopandas.GeoDataFrame(
        {"idx": [0, 1, 2]},
        geometry=geopandas.GeoSeries.from_xy([0, 1, 2], [1, 2, 3], crs="EPSG:4326"),
    )

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/data.fgb"
        gdf.to_file(path)

        table = con.sql(f"SELECT * FROM '{path}' ORDER BY idx").to_arrow_table()
        assert table.num_rows == 3
        assert table.column("idx").to_pylist() == [0, 1, 2]
