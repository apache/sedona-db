// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! RS_FromPath UDF - Load out-db raster from file path.
//!
//! `RS_FromPath` opens each file to read its header (grid, CRS, band types and
//! nodata); it reads no pixels. On object storage each open is a chain of
//! round trips, so a batch's files are opened concurrently on short-lived
//! threads, each open holding a permit of the session's [`RasterIoBudget`]
//! (`sedona.raster.io_concurrency`). The GDAL loader's pixel reads draw on the
//! same budget, so the cap is on header opens and pixel reads together, across
//! every partition of the session. The result is built on the calling thread
//! in row order, and a failing batch reports the error of its first failing
//! row, exactly as when the files were opened one at a time.
//!
//! The function stays a synchronous scalar UDF on purpose. An async UDF nests
//! under the `RS_EnsureLoaded` call the planner injects around raster
//! arguments of pixel-reading functions (`RS_SummaryStats(RS_FromPath(p))`),
//! which DataFusion cannot hoist (apache/datafusion#20031), and it cannot be
//! evaluated inside a spatial join's predicate. Instead, a call that fans out
//! hands its async worker over to the runtime with
//! `tokio::task::block_in_place` while it waits, so other tasks keep running.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::Array;
use arrow_schema::DataType;
use datafusion_common::cast::as_string_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_functions::executor::WkbBytesExecutor;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::io_budget::RasterIoBudget;
use sedona_raster::raster_loader::io_budget_from_config;
use sedona_schema::datatypes::{RASTER, SedonaType};
use sedona_schema::matchers::ArgMatcher;

use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::configure_thread_local_options;
use crate::utils::read_outdb_header;

pub fn rs_frompath_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_frompath",
        vec![Arc::new(RsFromPath::default())],
        Volatility::Volatile,
    )
}

#[derive(Debug, Default)]
pub(crate) struct RsFromPath {
    /// The budget used when the session has none to offer (a call without
    /// `ConfigOptions`, or a bare DataFusion context). A session's own
    /// budget, shared with its GDAL loader, takes precedence.
    budget: RasterIoBudget,
}

impl RsFromPath {
    #[cfg(test)]
    pub(crate) fn with_budget(budget: RasterIoBudget) -> Self {
        Self { budget }
    }
}

impl SedonaScalarKernel for RsFromPath {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(vec![ArgMatcher::is_string()], RASTER).match_args(args)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let executor = WkbBytesExecutor::new(arg_types, args);

        let paths = args[0]
            .cast_to(&DataType::Utf8, None)?
            .into_array_of_size(executor.num_iterations())?;
        let path_array = as_string_array(&paths)?;

        // Open each distinct path once: after a join a batch often names the
        // same file in many rows. Distinct paths are numbered in order of
        // first appearance, so the first failing row is the failing path with
        // the lowest number.
        let mut distinct: Vec<&str> = Vec::new();
        let mut index_of: HashMap<&str, usize> = HashMap::new();
        let rows: Vec<Option<usize>> = path_array
            .iter()
            .map(|path_opt| {
                path_opt.map(|path| {
                    *index_of.entry(path).or_insert_with(|| {
                        distinct.push(path);
                        distinct.len() - 1
                    })
                })
            })
            .collect();

        let open = |path: &str| {
            with_gdal(|gdal| {
                configure_thread_local_options(gdal, config_options)?;
                read_outdb_header(gdal, path)
            })
        };
        let budget = config_options
            .and_then(io_budget_from_config)
            .unwrap_or_else(|| self.budget.clone());
        let headers = open_all(&distinct, &budget, open)?;

        let mut builder = RasterBuilder::new(path_array.len());
        for row in rows {
            match row {
                Some(idx) => headers[idx].append_to(distinct[idx], &mut builder)?,
                None => builder.append_null()?,
            }
        }

        let result: Arc<dyn Array> = Arc::new(builder.finish()?);
        executor.finish(result)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("Should not be called because invoke_batch_from_args() is implemented")
    }
}

/// `open` every path under `budget` from wherever a scalar UDF is evaluated,
/// without stalling an async runtime, and return the results in `paths`
/// order. Error semantics are those of [`read_concurrently`].
fn open_all<T, F>(paths: &[&str], budget: &RasterIoBudget, open: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(&str) -> Result<T> + Sync,
{
    // A lone path with a permit free opens right here, with nothing to wait
    // for and no thread to hand off.
    if let [path] = paths
        && let Some(_permit) = budget.try_acquire()
    {
        return Ok(vec![open(path)?]);
    }
    match tokio::runtime::Handle::try_current().map(|handle| handle.runtime_flavor()) {
        // On a multi-threaded worker, waiting for permits and threads would
        // stall the worker's queued tasks: hand them to another thread for
        // the duration.
        Ok(tokio::runtime::RuntimeFlavor::MultiThread) => {
            tokio::task::block_in_place(|| read_concurrently(paths, budget, open))
        }
        // No runtime: nothing on this thread can be stalled.
        Err(_) => read_concurrently(paths, budget, open),
        // A current-thread runtime (or a blocking thread of one).
        Ok(_) => read_one_at_a_time(paths, budget, open),
    }
}

/// `open` each path in order on the calling thread, never waiting for the
/// budget. For a current-thread runtime, whose only worker may be this
/// thread: waiting there could deadlock, because the budget may hand a
/// released permit to one of the runtime's own tasks (a loader read about to
/// start), which cannot run until this call returns. An open holds a permit
/// when one is free and goes ahead without one otherwise, so on such a
/// runtime the cap can be exceeded by this one open.
fn read_one_at_a_time<T, F>(paths: &[&str], budget: &RasterIoBudget, open: F) -> Result<Vec<T>>
where
    F: Fn(&str) -> Result<T>,
{
    paths
        .iter()
        .map(|path| {
            let _permit = budget.try_acquire();
            open(path)
        })
        .collect()
}

/// `open` every path, at most `budget.limit()` at once across all users of
/// `budget`, and return the results in `paths` order. Blocks for permits, so
/// call it where blocking is allowed (see [`open_all`]).
///
/// Error semantics match opening the paths one at a time in order: the error
/// returned is that of the first failing path, and no path after it is
/// started once the failure is known (paths already in flight finish and are
/// discarded). Paths are started in order, so every path before the first
/// failure has run. All threads are joined before returning, so no open
/// outlives the call, whether it succeeds, fails or panics.
///
/// The calling thread takes part in the work; up to `budget.limit() - 1`
/// helper threads join it. Each thread holds at most one permit at a time,
/// and none while it waits for the next. A helper that cannot be spawned only
/// lowers the parallelism.
pub(crate) fn read_concurrently<T, F>(
    paths: &[&str],
    budget: &RasterIoBudget,
    open: F,
) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(&str) -> Result<T> + Sync,
{
    let n = paths.len();
    let next = AtomicUsize::new(0);
    let first_error = AtomicUsize::new(usize::MAX);
    let slots: Vec<Mutex<Option<Result<T>>>> = (0..n).map(|_| Mutex::new(None)).collect();

    let work = || {
        loop {
            let idx = next.fetch_add(1, Ordering::SeqCst);
            if idx >= n || idx > first_error.load(Ordering::SeqCst) {
                return;
            }
            let _permit = budget.acquire_blocking();
            // A failure may have landed while this thread waited for budget.
            if idx > first_error.load(Ordering::SeqCst) {
                return;
            }
            let result = open(paths[idx]);
            if result.is_err() {
                first_error.fetch_min(idx, Ordering::SeqCst);
            }
            // Each index is claimed by exactly one thread.
            *slots[idx].lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
        }
    };

    let helpers = budget.limit().min(n).saturating_sub(1);
    std::thread::scope(|scope| {
        for _ in 0..helpers {
            if std::thread::Builder::new()
                .name("rs_frompath".to_string())
                .spawn_scoped(scope, work)
                .is_err()
            {
                break;
            }
        }
        work();
    });

    let mut out = Vec::with_capacity(n);
    for (idx, slot) in slots.into_iter().enumerate() {
        match slot.into_inner().unwrap_or_else(PoisonError::into_inner) {
            Some(result) => out.push(result?),
            // Unreachable: every index before the first error ran, and the
            // loop returns at the first error.
            None => {
                return Err(sedona_internal_datafusion_err!(
                    "RS_FromPath: path {idx} of {n} was never opened"
                ));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{StringArray, StructArray};
    use datafusion_common::ScalarValue;
    use datafusion_common::cast::as_struct_array;
    use datafusion_expr::ScalarUDFImpl;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::io_budget::DEFAULT_RASTER_IO_CONCURRENCY;
    use sedona_raster::traits::RasterRef;
    use sedona_testing::data::test_raster;

    #[test]
    fn test_rs_from_path_udf_name() {
        assert_eq!(rs_frompath_udf().name(), "rs_frompath");
    }

    fn assert_raster_dimensions(
        result: &ColumnarValue,
        expected_len: usize,
        width: i64,
        height: i64,
    ) {
        fn assert_struct_array_dimensions(
            struct_arr: &StructArray,
            expected_len: usize,
            width: i64,
            height: i64,
        ) {
            let raster_array = RasterStructArray::try_new(struct_arr).unwrap();
            assert_eq!(raster_array.len(), expected_len);

            for idx in 0..expected_len {
                let raster = raster_array.get(idx).unwrap();
                assert_eq!(raster.width().unwrap(), width);
                assert_eq!(raster.height().unwrap(), height);
            }
        }

        match result {
            ColumnarValue::Array(arr) => {
                let struct_arr = as_struct_array(arr).unwrap();
                assert_struct_array_dimensions(struct_arr, expected_len, width, height);
            }
            ColumnarValue::Scalar(ScalarValue::Struct(struct_arr)) => {
                assert_struct_array_dimensions(struct_arr, expected_len, width, height);
            }
            other => panic!("Unexpected result: {other:?}"),
        }
    }

    #[test]
    fn test_invoke_rs_from_path() {
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let paths = Arc::new(StringArray::from(vec![path.as_str()]));
        let input = ColumnarValue::Array(paths);

        let kernel = RsFromPath::default();
        let result = kernel
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .expect("Should invoke successfully");

        assert_raster_dimensions(&result, 1, 10, 10);

        let scalar_input = ColumnarValue::Scalar(ScalarValue::Utf8(Some(path.clone())));
        let scalar_result = kernel
            .invoke_batch_from_args(
                &[],
                &[scalar_input],
                &SedonaType::Arrow(DataType::Null),
                0,
                None,
            )
            .expect("Should invoke successfully for scalar path");

        assert_raster_dimensions(&scalar_result, 1, 10, 10);

        let multi_paths = Arc::new(StringArray::from(vec![path.as_str(), path.as_str()]));
        let multi_result = kernel
            .invoke_batch_from_args(
                &[],
                &[ColumnarValue::Array(multi_paths)],
                &SedonaType::Arrow(DataType::Null),
                0,
                None,
            )
            .expect("Should invoke successfully for multiple paths");

        assert_raster_dimensions(&multi_result, 2, 10, 10);

        let empty_paths = Arc::new(StringArray::from(Vec::<&str>::new()));
        let empty_result = kernel
            .invoke_batch_from_args(
                &[],
                &[ColumnarValue::Array(empty_paths)],
                &SedonaType::Arrow(DataType::Null),
                0,
                None,
            )
            .expect("Should invoke successfully for empty paths");

        match empty_result {
            ColumnarValue::Array(arr) => {
                let struct_arr = as_struct_array(&arr).unwrap();
                assert_eq!(struct_arr.len(), 0);
            }
            other => panic!("Expected empty array result, got {other:?}"),
        }
    }

    #[test]
    fn test_invoke_rs_from_path_propagates_nulls() {
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let input =
            ColumnarValue::Array(Arc::new(StringArray::from(vec![Some(path.as_str()), None])));

        let result = RsFromPath::default()
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .expect("Should invoke successfully for null-containing input");

        match result {
            ColumnarValue::Array(arr) => {
                let struct_arr = as_struct_array(&arr).unwrap();
                assert_eq!(struct_arr.len(), 2);
                assert!(!struct_arr.is_null(0));
                assert!(struct_arr.is_null(1));

                let raster_array = RasterStructArray::try_new(struct_arr).unwrap();
                let raster = raster_array.get(0).unwrap();
                assert_eq!(raster.width().unwrap(), 10);
                assert_eq!(raster.height().unwrap(), 10);
            }
            other => panic!("Expected array result, got {other:?}"),
        }
    }

    #[test]
    fn test_invoke_rs_from_path_invalid_path_errors() {
        let missing_path = "/definitely/missing/rs_from_path_test.tif";
        let input = ColumnarValue::Scalar(ScalarValue::Utf8(Some(missing_path.to_string())));

        let err = RsFromPath::default()
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .expect_err("Missing path should return an error");

        let err_message = err.to_string();
        assert!(err_message.contains(&format!(
            "Failed to open raster file '{}' (GDAL path '{}')",
            missing_path, missing_path
        )));
    }

    #[test]
    fn test_invoke_rs_from_path_scalar_ignores_num_rows_for_shape() {
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let result = RsFromPath::default()
            .invoke_batch_from_args(
                &[],
                &[ColumnarValue::Scalar(ScalarValue::Utf8(Some(path)))],
                &SedonaType::Arrow(DataType::Null),
                32,
                None,
            )
            .expect("Should invoke successfully for scalar path with larger num_rows");

        assert!(matches!(result, ColumnarValue::Scalar(_)));
        assert_raster_dimensions(&result, 1, 10, 10);
    }

    // ---- concurrent header reads -------------------------------------------

    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use arrow_array::ArrayRef;
    use datafusion_common::cast::as_string_view_array;
    use datafusion_common::exec_err;

    /// Tracks how many `open` calls are in flight and the most ever seen.
    #[derive(Default)]
    struct InFlight {
        now: AtomicUsize,
        max: AtomicUsize,
        started: AtomicUsize,
    }

    impl InFlight {
        fn run<T>(&self, f: impl FnOnce() -> T) -> T {
            self.started.fetch_add(1, Ordering::SeqCst);
            let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
            self.max.fetch_max(now, Ordering::SeqCst);
            let out = f();
            self.now.fetch_sub(1, Ordering::SeqCst);
            out
        }
    }

    fn numbered_paths(n: usize) -> Vec<String> {
        (0..n).map(|i| i.to_string()).collect()
    }

    #[test]
    fn read_concurrently_keeps_input_order() {
        let owned = numbered_paths(40);
        let paths: Vec<&str> = owned.iter().map(String::as_str).collect();
        for size in [1, 4, DEFAULT_RASTER_IO_CONCURRENCY] {
            let budget = RasterIoBudget::new(size);
            // Early paths take longest, so they finish last.
            let out = read_concurrently(&paths, &budget, |p| {
                let i: u64 = p.parse().unwrap();
                std::thread::sleep(Duration::from_micros(40 - i));
                Ok(i)
            })
            .unwrap();
            assert_eq!(out, (0..40).collect::<Vec<u64>>(), "budget {size}");
            assert_eq!(budget.available(), size);
        }
    }

    #[test]
    fn read_concurrently_empty_input() {
        let budget = RasterIoBudget::new(8);
        let out: Vec<u8> = read_concurrently(&[], &budget, |_| unreachable!()).unwrap();
        assert!(out.is_empty());
        assert_eq!(budget.available(), 8);
    }

    #[test]
    fn read_concurrently_is_bounded_by_budget() {
        let owned = numbered_paths(48);
        let paths: Vec<&str> = owned.iter().map(String::as_str).collect();
        let budget = RasterIoBudget::new(8);
        let in_flight = InFlight::default();
        read_concurrently(&paths, &budget, |_| {
            in_flight.run(|| std::thread::sleep(Duration::from_millis(2)));
            Ok(())
        })
        .unwrap();
        let max = in_flight.max.load(Ordering::SeqCst);
        assert!(max <= 8, "{max} opens in flight with a budget of 8");
        assert!(max > 1, "opens never overlapped");
        assert_eq!(budget.available(), 8);
    }

    #[test]
    fn concurrent_callers_share_one_budget() {
        let owned = numbered_paths(16);
        let paths: Vec<&str> = owned.iter().map(String::as_str).collect();
        let budget = RasterIoBudget::new(2);
        let in_flight = InFlight::default();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    read_concurrently(&paths, &budget, |_| {
                        in_flight.run(|| std::thread::sleep(Duration::from_millis(1)));
                        Ok(())
                    })
                    .unwrap();
                });
            }
        });
        assert!(in_flight.max.load(Ordering::SeqCst) <= 2);
        assert_eq!(in_flight.started.load(Ordering::SeqCst), 8 * 16);
        assert_eq!(budget.available(), 2);
    }

    #[test]
    fn read_concurrently_reports_the_first_failing_path() {
        let owned = numbered_paths(20);
        let paths: Vec<&str> = owned.iter().map(String::as_str).collect();
        for size in [1, 3, DEFAULT_RASTER_IO_CONCURRENCY] {
            let budget = RasterIoBudget::new(size);
            // Path 5 fails slowly and path 12 fails at once, so with any
            // parallelism 12's error is known first; 5's must still win.
            let err = read_concurrently(&paths, &budget, |p| match p {
                "5" => {
                    std::thread::sleep(Duration::from_millis(20));
                    exec_err!("open failed: 5")
                }
                "12" => exec_err!("open failed: 12"),
                _ => Ok(()),
            })
            .unwrap_err();
            assert!(
                err.to_string().contains("open failed: 5"),
                "budget {size}: {err}"
            );
            assert_eq!(budget.available(), size, "budget {size} leaked permits");
        }
    }

    #[test]
    fn read_concurrently_stops_starting_paths_after_a_failure() {
        let owned = numbered_paths(50);
        let paths: Vec<&str> = owned.iter().map(String::as_str).collect();

        // One at a time this is exactly the serial behaviour: nothing after
        // the failing path is opened.
        let budget = RasterIoBudget::new(1);
        let in_flight = InFlight::default();
        read_concurrently(&paths, &budget, |p| {
            in_flight.run(|| if p == "2" { exec_err!("boom") } else { Ok(()) })
        })
        .unwrap_err();
        assert_eq!(in_flight.started.load(Ordering::SeqCst), 3);

        // With a budget, at most the paths already in flight finish.
        let budget = RasterIoBudget::new(4);
        let in_flight = InFlight::default();
        read_concurrently(&paths, &budget, |p| {
            in_flight.run(|| {
                if p == "0" {
                    exec_err!("boom")
                } else {
                    std::thread::sleep(Duration::from_millis(5));
                    Ok(())
                }
            })
        })
        .unwrap_err();
        assert!(in_flight.started.load(Ordering::SeqCst) < 10);
        assert_eq!(budget.available(), 4);
    }

    #[test]
    fn read_concurrently_returns_permits_when_an_open_panics() {
        let owned = numbered_paths(10);
        let paths: Vec<&str> = owned.iter().map(String::as_str).collect();
        let budget = RasterIoBudget::new(3);
        let finished = AtomicBool::new(false);
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            read_concurrently(&paths, &budget, |p| {
                if p == "4" {
                    panic!("open panicked");
                }
                Ok(())
            })
            .unwrap();
            finished.store(true, Ordering::SeqCst);
        }));
        assert!(unwound.is_err());
        assert!(!finished.load(Ordering::SeqCst));
        // Every thread was joined and every permit came back; the budget
        // still works.
        assert_eq!(budget.available(), 3);
        read_concurrently(&paths, &budget, |_| Ok(())).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn invoke_on_a_multi_thread_runtime_worker() {
        // The fan-out moves off the async worker with `block_in_place`,
        // which needs a multi-threaded runtime; check the call works there.
        let a = test_raster("test4.tiff").unwrap();
        let b = test_raster("test1.tiff").unwrap();
        let input = ColumnarValue::Array(Arc::new(StringArray::from(vec![
            a.as_str(),
            b.as_str(),
            a.as_str(),
        ])));
        let result = RsFromPath::default()
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .unwrap();
        let ColumnarValue::Array(arr) = result else {
            panic!("expected an array");
        };
        assert_eq!(arr.len(), 3);
    }

    /// The batch result matches appending each row on its own, one file at
    /// a time, as `RS_FromPath` used to: same rows, same order, nulls kept,
    /// repeated paths each get their own raster.
    #[test]
    fn concurrent_batch_matches_serial_rows() {
        use crate::utils::append_as_outdb_raster;

        let files: Vec<String> = ["test1.tiff", "test4.tiff", "test5.tiff", "sentinel2.tif"]
            .iter()
            .map(|f| test_raster(f).unwrap())
            .collect();
        let rows: Vec<Option<&str>> = (0..37)
            .map(|i| (i % 7 != 3).then(|| files[(i * 5) % files.len()].as_str()))
            .collect();

        let mut serial = RasterBuilder::new(rows.len());
        with_gdal(|gdal| {
            for row in &rows {
                match row {
                    Some(path) => append_as_outdb_raster(gdal, path, &mut serial)?,
                    None => serial.append_null()?,
                }
            }
            Ok(())
        })
        .unwrap();
        let serial: ArrayRef = Arc::new(serial.finish().unwrap());

        for size in [1, 2, DEFAULT_RASTER_IO_CONCURRENCY] {
            let input = ColumnarValue::Array(Arc::new(StringArray::from(rows.clone())));
            let ColumnarValue::Array(batch) = RsFromPath::with_budget(RasterIoBudget::new(size))
                .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
                .unwrap()
            else {
                panic!("expected an array");
            };
            assert_same_rows(&batch, &serial, size);
        }
    }

    /// Equal column by column, except that the CRS is compared by its
    /// top-level name. GDAL's PROJJSON for a file can depend on what the same
    /// thread opened earlier (`sentinel2.tif` renders its datum as a
    /// `datum` on a fresh thread, and as a `datum_ensemble` after
    /// `test4.tiff` was opened on it), and that is so with or without the
    /// fan-out, so the exact text is not what this test is about.
    fn assert_same_rows(batch: &ArrayRef, serial: &ArrayRef, size: usize) {
        fn crs_name(json: &str) -> &str {
            json.lines()
                .find(|line| line.trim_start().starts_with("\"name\""))
                .unwrap_or(json)
        }
        let (batch, serial) = (
            as_struct_array(batch).unwrap(),
            as_struct_array(serial).unwrap(),
        );
        assert_eq!(batch.nulls(), serial.nulls(), "budget {size}");
        for (name, (b, s)) in batch
            .column_names()
            .iter()
            .zip(batch.columns().iter().zip(serial.columns()))
        {
            if *name == "crs" {
                let (b, s) = (
                    as_string_view_array(b).unwrap(),
                    as_string_view_array(s).unwrap(),
                );
                assert_eq!(b.nulls(), s.nulls(), "budget {size}");
                for (row, (b, s)) in b.iter().zip(s.iter()).enumerate() {
                    assert_eq!(b.map(crs_name), s.map(crs_name), "budget {size}, row {row}");
                }
            } else {
                assert_eq!(b.as_ref(), s.as_ref(), "budget {size}, column {name}");
            }
        }
    }

    #[test]
    fn concurrent_batch_reports_the_first_missing_file() {
        let good = test_raster("test4.tiff").unwrap();
        let rows = vec![
            Some(good.as_str()),
            None,
            Some("/definitely/missing/first.tif"),
            Some(good.as_str()),
            Some("/definitely/missing/second.tif"),
        ];
        for size in [1, DEFAULT_RASTER_IO_CONCURRENCY] {
            let kernel = RsFromPath::with_budget(RasterIoBudget::new(size));
            let input = ColumnarValue::Array(Arc::new(StringArray::from(rows.clone())));
            let err = kernel
                .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("Failed to open raster file '/definitely/missing/first.tif'"),
                "budget {size}: {err}"
            );
            assert_eq!(kernel.budget.available(), size);
        }
    }

    /// With a session budget in the config, the call draws on it (resized
    /// to `sedona.raster.io_concurrency`) rather than the UDF's own.
    #[test]
    fn invoke_draws_on_the_session_budget() {
        use sedona_common::option::SedonaOptions;
        use sedona_raster::raster_loader::{RasterLoaderConfig, RasterLoaderRegistry};
        use std::sync::RwLock;

        let session = RasterIoBudget::default();
        let mut config = ConfigOptions::new();
        config.extensions.insert(SedonaOptions::default());
        config.extensions.insert(
            RasterLoaderConfig::from_handle(Arc::new(RwLock::new(RasterLoaderRegistry::new())))
                .with_io_budget(session.clone()),
        );
        config.set("sedona.raster.io_concurrency", "3").unwrap();

        let kernel = RsFromPath::default();
        let files: Vec<String> = ["test1.tiff", "test4.tiff", "test5.tiff"]
            .iter()
            .map(|f| test_raster(f).unwrap())
            .collect();
        let input = ColumnarValue::Array(Arc::new(StringArray::from(files.clone())));
        kernel
            .invoke_batch_from_args(
                &[],
                &[input],
                &SedonaType::Arrow(DataType::Null),
                0,
                Some(&config),
            )
            .unwrap();

        assert_eq!(session.limit(), 3);
        assert!(session.peak_in_use() >= 1);
        assert_eq!(session.in_use(), 0);
        assert_eq!(kernel.budget.peak_in_use(), 0);
    }

    /// On a current-thread runtime the call never waits for the budget: a
    /// released permit could be handed to a task of that same runtime, which
    /// cannot run while the call blocks its only worker. With the whole
    /// budget held elsewhere the batch still opens, one file at a time.
    #[test]
    fn current_thread_runtime_never_waits_for_the_budget() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let budget = RasterIoBudget::new(1);
        let kernel = RsFromPath::with_budget(budget.clone());
        let files: Vec<String> = ["test1.tiff", "test4.tiff", "test5.tiff"]
            .iter()
            .map(|f| test_raster(f).unwrap())
            .collect();
        runtime.block_on(async {
            for hold_the_budget in [false, true] {
                let held = hold_the_budget.then(|| budget.try_acquire().unwrap());
                let input = ColumnarValue::Array(Arc::new(StringArray::from(files.clone())));
                let ColumnarValue::Array(rasters) = kernel
                    .invoke_batch_from_args(
                        &[],
                        &[input],
                        &SedonaType::Arrow(DataType::Null),
                        0,
                        None,
                    )
                    .unwrap()
                else {
                    panic!("expected an array");
                };
                assert_eq!(rasters.len(), 3);
                drop(held);
            }
        });
        assert_eq!(budget.available(), 1);
        assert_eq!(budget.in_use(), 0);
    }
}
