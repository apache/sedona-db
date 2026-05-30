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
**Build from source**

For building the Python package from source, you should enable the GPU at build time and configure the CUDA
environment before running `MATURIN_PEP517_ARGS="--features='gpu,s2geography,pyo3/extension-module' pip install`.

Common environment variables used by the GPU build:

- `CUDA_HOME`: points to your CUDA toolkit root.
- `CMAKE_CUDA_ARCHITECTURES`: CUDA SM targets (default falls back to `native` to auto-detect the host GPU architecture). Change this according to your [GPU models](https://developer.nvidia.com/cuda/gpus).

- `LIBGPUSPATIAL_LOGGING_LEVEL`: Logging level, including `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`.
- `GPUSPATIAL_PROFILING=ON`: Profiling mode (`ON`, `OFF`) to compile profiling instrumentation.

**Using Docker**

We also provide a Dockerfile for the users. Make sure you have installed Docker, the NVIDIA Driver, and NVIDIA Container Toolkit by `sudo nvidia-ctk runtime configure --runtime=docker`. Then, you may run `docker build -f docker/sedonadb-gpu.dockerfile --build-arg CMAKE_CUDA_ARCHITECTURES="native" -t sedonadb-gpu .` to build an image (defaults to `native` to auto-detect the host's GPU architecture, or you can specify target architectures like `"86;89"`). Finally, you may run `docker run -it --rm --gpus all -p 8888:8888 sedonadb-gpu` to start a JupyterLab instance.

> [!NOTE]
> This Docker image is currently source-built directly from the repository's active codebase and is not tied to a released SedonaDB GPU wheel yet. This is to support development and early testing before an official SedonaDB GPU wheel is published.

## Enable GPU Join with SQL `SET`

The GPU join is disabled by default even if you enabled the GPU feature at build time. To enable GPU acceleration for spatial joins, set the `gpu.enable` option to `true`:


```python
import sedonadb

ctx = sedonadb.connect()

ctx.sql("SET gpu.enable = true").execute()
```




    <sedonadb.dataframe.DataFrame object at 0x70b85c1b7850>



## Performance Tuning and Special Cautions

To keep the GPU efficiently utilized, use larger execution batches:


```python
ctx.sql("SET datafusion.execution.batch_size = 100000").execute()
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
ctx.sql("SET datafusion.execution.batch_size = 100000").execute()
```

## Example

### Check GPU environment


```python
!nvidia-smi
```

    Tue May 26 20:49:41 2026
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
    +-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA L4                      On  |   00000000:31:00.0 Off |                    0 |
    | N/A   41C    P0             28W /   72W |   11610MiB /  23034MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+

    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |    0   N/A  N/A           20603      C   ...a3/envs/sedona/bin/python3.11      11602MiB |
    +-----------------------------------------------------------------------------------------+


### Install required packages for the example


```python
!pip install huggingface_hub rasterio pyogrio
```

### Download datasets


```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="apache-sedona/spatialbench",
    repo_type="dataset",
    local_dir="hf-data",
    allow_patterns=["v0.1.0/sf1/zone/*", "v0.1.0/sf1/trip/*"],
)
```


    Downloading (incomplete total...): 0.00B [00:00, ?B/s]



    Fetching 8 files:   0%|          | 0/8 [00:00<?, ?it/s]





    '/mnt/data/sedona-db/docs/hf-data'



### Create sedonadb context and load datasets


```python
import sedonadb

ctx = sedonadb.connect()
ctx.options.memory_limit = "unlimited"

ctx.sql("""
    CREATE EXTERNAL TABLE zone
    STORED AS PARQUET
    LOCATION 'hf-data/v0.1.0/sf1/zone/'
""")
ctx.sql("""
    CREATE EXTERNAL TABLE trip
    STORED AS PARQUET
    LOCATION 'hf-data/v0.1.0/sf1/trip/'
""").execute()
```




    0



### Define query


```python
# Define the query once to ensure both executions use the exact same logic
query = """
SELECT COUNT(*) AS cross_zone_trip_count
FROM trip t
    JOIN zone pickup_zone
        ON ST_Within(ST_GeomFromWKB(t.t_pickuploc), ST_GeomFromWKB(pickup_zone.z_boundary))
    JOIN zone dropoff_zone
        ON ST_Within(ST_GeomFromWKB(t.t_dropoffloc), ST_GeomFromWKB(dropoff_zone.z_boundary))
WHERE pickup_zone.z_zonekey != dropoff_zone.z_zonekey
"""
```

First, we will run this query using the standard CPU execution to establish our baseline correctness and get a sense of the standard execution time.


```python
import time

print("Executing on CPU (Baseline)...")
ctx.sql("SET gpu.enable = false")  # Disable the GPU feature
ctx.sql("SET datafusion.execution.batch_size = 8192")  # Default configuration

runs = 5
cpu_times = []
cpu_count = 0

for i in range(runs):
    start_time = time.time()
    cpu_df = ctx.sql(query).to_pandas()
    elapsed = time.time() - start_time

    cpu_times.append(elapsed)
    cpu_count = cpu_df.iloc[0, 0]
    print(f"  Run {i + 1}: {elapsed:.4f} seconds")

cpu_avg_time = sum(cpu_times) / runs
print("-" * 30)
print(f"CPU Average Time: {cpu_avg_time:.4f} seconds | Rows: {cpu_count}")
```

    Executing on CPU (Baseline)...
      Run 1: 7.7795 seconds
      Run 2: 6.8287 seconds
      Run 3: 7.0688 seconds
      Run 4: 7.2434 seconds
      Run 5: 7.1739 seconds
    ------------------------------
    CPU Average Time: 7.2189 seconds | Rows: 176391


Next, we enable GPU execution. We also increase the batch size, which is a crucial optimization step to ensure we are feeding enough data to the GPU to maximize its parallel processing capabilities.


```python
print("Executing on GPU (Accelerated)...")
ctx.sql("SET gpu.enable = true")
ctx.sql("SET datafusion.execution.batch_size = 1000000")  # Optimized for GPU throughput

runs = 5
gpu_times = []
gpu_count = 0

for i in range(runs):
    start_time = time.time()
    gpu_df = ctx.sql(query).to_pandas()
    elapsed = time.time() - start_time

    gpu_times.append(elapsed)
    gpu_count = gpu_df.iloc[0, 0]
    print(f"  Run {i + 1}: {elapsed:.4f} seconds")

gpu_avg_time = sum(gpu_times) / runs
print("-" * 30)
print(f"GPU Average Time: {gpu_avg_time:.4f} seconds | Rows: {gpu_count}")
```

    Executing on GPU (Accelerated)...
      Run 1: 3.2582 seconds
      Run 2: 3.3126 seconds
      Run 3: 3.0703 seconds
      Run 4: 3.2604 seconds
      Run 5: 3.1418 seconds
    ------------------------------
    GPU Average Time: 3.2087 seconds | Rows: 176391


### Compare Results


```python
# Validate correctness
match_status = "✅ Match" if cpu_count == gpu_count else "❌ Mismatch"

# Calculate speedup based on the new averages
speedup = cpu_avg_time / gpu_avg_time if gpu_avg_time > 0 else 0

print("--- Query Validation (Based on Averages) ---")
print(f"Row Count Validated : {gpu_count} ({match_status})")
print(f"Average CPU Time    : {cpu_avg_time:.4f}s")
print(f"Average GPU Time    : {gpu_avg_time:.4f}s")
print(f"Calculated Speedup  : {speedup:.2f}x")
```

    --- Query Validation (Based on Averages) ---
    Row Count Validated : 176391 (✅ Match)
    Average CPU Time    : 7.2189s
    Average GPU Time    : 3.2087s
    Calculated Speedup  : 2.25x


## References

1. Geng, Liang, Rubao Lee, and Xiaodong Zhang. "Librts: A spatial indexing library by ray tracing." Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming. 2025.
2. Geng, Liang, Rubao Lee, and Xiaodong Zhang. "Rayjoin: Fast and precise spatial join." Proceedings of the 38th ACM International Conference on Supercomputing. 2024.
