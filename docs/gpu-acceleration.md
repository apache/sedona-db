<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# GPU Acceleration for Spatial Join

SedonaDB supports GPU-accelerated spatial joins with NVIDIA GPUs. The
key innovation is using NVIDIA RT cores for spatial query acceleration, which significantly reduces
candidate search cost and predicate evaluation.

> **Prerequisites:** GPU acceleration requires a SedonaDB build with GPU support, NVIDIA CUDA 12+
> on a GPU with compute capability 7.5 or higher.
> The feature works with or without dedicated RT cores. When RT cores are not available, NVIDA OptiX
> emulates ray tracing on CUDA cores.

## Why This Matters

Spatial joins are often bottlenecked by candidate generation and exact refinement. SedonaDB's GPU
path targets both phases:

- RT-core acceleration for high-throughput spatial filtering.
- A geometry-aware refinement pipeline, including a heavily optimized point-in-polygon (PIP) path.

## Filtering and Refining Stages

The GPU-based join follows SedonaDB's two-stage execution model:

1. **Filtering stage**
   - Runs on NVIDIA RT cores [1].
   - Quickly generates candidate geometry pairs that intersect.

2. **Refining stage**
   - Runs exact predicate checks on candidates.
   - For point-polygon geometry pairs, refinement is heavily optimized and accelerated with RT-core-backed
     ray-tracing techniques [2].
   - For other spatial join patterns, refinement runs on CUDA-core kernels. We will gradually expand the RT acceleration coverage in future releases.

## Supported Predicates and Fallback Behavior

GPU spatial join currently supports relation predicates:

- `ST_Intersects`
- `ST_Contains`
- `ST_Within`
- `ST_Covers`
- `ST_CoveredBy`
- `ST_Touches`
- `ST_Equals`

Not currently supported on the GPU path:

- Distance predicates (for example, `ST_DWithin`)
- KNN / KNN join predicates
- GeometryCollection

When `gpu.fallback_to_cpu = true` (default), unsupported predicates fall back to CPU spatial join.
When `gpu.fallback_to_cpu = false`, unsupported predicates fail the query.

## Install from Source with the GPU Feature

If you build the Python package from source, enable GPU at build time and configure the CUDA
environment before running `MATURIN_PEP517_ARGS="--features gpu" pip install`.

Common environment variables used by the GPU build:

- `CUDA_HOME`: points to your CUDA toolkit root.
- `CMAKE_CUDA_ARCHITECTURES`: CUDA SM targets (default falls back to `86;89` if not set). Change this according to your GPU architecture.

- `LIBGPUSPATIAL_LOGGING_LEVEL`: Logging level, including `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`.
- `GPUSPATIAL_PROFILING=ON`: Profiling mode (`ON`, `OFF`) to compile profiling instrumentation.

## Enable GPU Join with SQL `SET`

The GPU join is disabled by default even if you enabled the GPU feature at build time. To enable GPU acceleration for spatial joins, set the `gpu.enable` option to `true`:


```python
import sedonadb

ctx = sedonadb.connect()

ctx.sql("SET gpu.enable = true")
```

## Performance Tuning and Special Cautions

To keep the GPU efficiently utilized, use larger execution batches:


```python
ctx.sql("SET datafusion.execution.batch_size = 100000")
```

Important guidance:

- A large batch size (for example, `100000`) is often necessary for good GPU throughput.
- For highest GPU performance, spilling should be disabled (for example, set `sd.options.memory_limit = "unlimited"` before running queries).
- Small joins may not amortize GPU overhead and can be slower than CPU execution.
- Start with defaults for other `gpu.*` options, then tune based on measured workload behavior.

## GPU Options

The following session options are available under the `gpu.` prefix:

- `gpu.enable` (bool, default `false`): enable GPU spatial join.
- `gpu.concat_build` (bool, default `true`): concatenate geometry buffers before GPU processing.
- `gpu.device_id` (int, default `0`): CUDA device ID.
- `gpu.fallback_to_cpu` (bool, default `true`): use CPU path when GPU path is unavailable/unsupported.
- `gpu.use_memory_pool` (bool, default `true`): use CUDA memory pool.
- `gpu.memory_pool_init_percentage` (int, default `50`): initial CUDA memory pool size as a percentage of total GPU memory.
- `gpu.pipeline_batches` (int, default `1`): overlap parsing and refinement across batches.
- `gpu.compress_bvh` (bool, default `false`): compress BVH to reduce memory usage (can reduce performance).

Example:


```python
import sedonadb

ctx = sedonadb.connect()

ctx.sql("SET gpu.enable = true")
ctx.sql("SET gpu.device_id = 0")
ctx.sql("SET gpu.use_memory_pool = true")
ctx.sql("SET gpu.memory_pool_init_percentage = 60")
ctx.sql("SET gpu.pipeline_batches = 2")
ctx.sql("SET gpu.compress_bvh = false")
ctx.sql("SET datafusion.execution.batch_size = 100000")
```

## Example


```python
!nvidia-smi
```

    Mon Apr 20 19:26:08 2026
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA L4                      On  |   00000000:31:00.0 Off |                    0 |
    | N/A   36C    P8             19W /   72W |       0MiB /  23034MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+



```python
from huggingface_hub import snapshot_download
import os

snapshot_download(
  repo_id='apache-sedona/spatialbench',
  repo_type='dataset',
  local_dir='hf-data',
  allow_patterns=[
    "v0.1.0/sf1/zone/*",
    "v0.1.0/sf1/trip/*"],
)
```


    Downloading (incomplete total...): 0.00B [00:00, ?B/s]



    Fetching 8 files:   0%|          | 0/8 [00:00<?, ?it/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.





    '/mnt/data/sedona-db/docs/hf-data'




```python
import sedonadb

ctx = sedonadb.connect()
ctx.options.memory_limit = "unlimited"
```


```python
ctx.sql(f"""
    CREATE EXTERNAL TABLE zone
    STORED AS PARQUET
    LOCATION 'hf-data/v0.1.0/sf1/zone/'
""")
ctx.sql(f"""
    CREATE EXTERNAL TABLE trip
    STORED AS PARQUET
    LOCATION 'hf-data/v0.1.0/sf1/trip/'
""")
```




    <sedonadb.dataframe.DataFrame object at 0x7ff5213d96d0>




```python
import time
from tqdm.notebook import tqdm
from IPython.display import display, HTML
import ipywidgets as widgets

def interactive_spatial_benchmark(ctx, runs=6):
    query = """
    SELECT COUNT(*) AS cross_zone_trip_count
    FROM trip t
        JOIN zone pickup_zone
            ON ST_Within(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromWKB(pickup_zone.z_boundary))
        JOIN zone dropoff_zone
            ON ST_Within(ST_GeomFromWKB(t.t_dropoffloc), ST_GeomFromWKB(dropoff_zone.z_boundary))
    WHERE pickup_zone.z_zonekey != dropoff_zone.z_zonekey
    """

    modes = [("CPU", "false"), ("GPU", "true")]
    averages = {}

    # 1. Create a scrollable widget to catch the verbose `.show()` output
    log_output = widgets.Output(layout={'border': '1px solid #ccc', 'height': '150px', 'overflow_y': 'auto'})
    display(HTML("<h3>🚀 Running Spatial Benchmark...</h3>"))
    display(log_output)

    # 2. Execute the Benchmark
    for mode_name, gpu_flag in modes:
        ctx.sql(f"SET gpu.enable = {gpu_flag}")
        if gpu_flag:
            ctx.sql("SET datafusion.execution.batch_size = 2000000") # Increase batch size
        else:
            ctx.sql("SET datafusion.execution.batch_size = 8192") # Default
        execution_times = []

        # Display a Jupyter-native progress bar
        for i in tqdm(range(runs), desc=f"{mode_name} Executions"):
            start_time = time.time()

            result = ctx.sql(query)

            # Execute physical plan and catch the output inside our widget
            with log_output:
                print(f"--- {mode_name} Run {i+1} ---")
                result.show()

            elapsed = time.time() - start_time

            # Record everything except the first run (warmup)
            if i > 0:
                execution_times.append(elapsed)

        # Calculate average
        averages[mode_name] = sum(execution_times) / len(execution_times)

    # 3. Clean up the UI
    log_output.clear_output()
    log_output.layout.display = 'none'

    # 4. Generate Table Results
    cpu_avg = averages["CPU"]
    gpu_avg = averages["GPU"]

    # Calculate speedup (using CPU as the 1.0x baseline)
    cpu_speedup = 1.0
    gpu_speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0

    # Color code the GPU speedup (green if faster, red if slower)
    gpu_color = "green" if gpu_speedup >= 1.0 else "red"

    html_table = f"""
    <table style="width: 60%; text-align: center; border-collapse: collapse; font-family: sans-serif; margin-top: 15px;">
        <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
            <th style="padding: 12px; border: 1px solid #dee2e6;">Mode</th>
            <th style="padding: 12px; border: 1px solid #dee2e6;">Average Time (s)</th>
            <th style="padding: 12px; border: 1px solid #dee2e6;">Speedup</th>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #dee2e6;"><b>CPU</b> (Baseline)</td>
            <td style="padding: 12px; border: 1px solid #dee2e6;">{cpu_avg:.4f}</td>
            <td style="padding: 12px; border: 1px solid #dee2e6;">{cpu_speedup:.2f}x</td>
        </tr>
        <tr>
            <td style="padding: 12px; border: 1px solid #dee2e6;"><b>GPU</b></td>
            <td style="padding: 12px; border: 1px solid #dee2e6;">{gpu_avg:.4f}</td>
            <td style="padding: 12px; border: 1px solid #dee2e6; color: {gpu_color}; font-weight: bold;">{gpu_speedup:.2f}x</td>
        </tr>
    </table>
    """

    display(HTML("<h3>📊 Benchmark Results (Averaged over 5 runs)</h3>"))
    display(HTML(html_table))

# --- Execution Block ---
# Run the interactive function with your Sedona Context
interactive_spatial_benchmark(ctx)
```


<h3>🚀 Running Spatial Benchmark...</h3>



    Output(layout=Layout(border_bottom='1px solid #ccc', border_left='1px solid #ccc', border_right='1px solid #cc…



    CPU Executions:   0%|          | 0/6 [00:00<?, ?it/s]



    GPU Executions:   0%|          | 0/6 [00:00<?, ?it/s]



<h3>📊 Benchmark Results (Averaged over 5 runs)</h3>




<table style="width: 60%; text-align: center; border-collapse: collapse; font-family: sans-serif; margin-top: 15px;">
    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
        <th style="padding: 12px; border: 1px solid #dee2e6;">Mode</th>
        <th style="padding: 12px; border: 1px solid #dee2e6;">Average Time (s)</th>
        <th style="padding: 12px; border: 1px solid #dee2e6;">Speedup</th>
    </tr>
    <tr>
        <td style="padding: 12px; border: 1px solid #dee2e6;"><b>CPU</b> (Baseline)</td>
        <td style="padding: 12px; border: 1px solid #dee2e6;">21.5221</td>
        <td style="padding: 12px; border: 1px solid #dee2e6;">1.00x</td>
    </tr>
    <tr>
        <td style="padding: 12px; border: 1px solid #dee2e6;"><b>GPU</b></td>
        <td style="padding: 12px; border: 1px solid #dee2e6;">2.8889</td>
        <td style="padding: 12px; border: 1px solid #dee2e6; color: green; font-weight: bold;">7.45x</td>
    </tr>
</table>



## References

1. Geng, Liang, Rubao Lee, and Xiaodong Zhang. "Librts: A spatial indexing library by ray tracing." Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming. 2025.
2. Geng, Liang, Rubao Lee, and Xiaodong Zhang. "Rayjoin: Fast and precise spatial join." Proceedings of the 38th ACM International Conference on Supercomputing. 2024.
