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

# sedona-spatial-join-gpu

GPU-accelerated spatial join execution for Apache SedonaDB.

## Overview

This package provides GPU-accelerated spatial joins that leverage CUDA for high-performance spatial operations. It integrates with DataFusion's execution engine to accelerate spatial join queries when GPU resources are available.

### Architecture

The GPU spatial join follows a **streaming architecture** that integrates seamlessly with DataFusion:

```
ParquetExec (left) ──┐
                     ├──> GpuSpatialJoinExec ──> Results
ParquetExec (right) ─┘
```

Unlike the CPU-based spatial join, the GPU implementation accepts child ExecutionPlan nodes and reads from their streams, making it composable with any DataFusion operator.

## Features

- **GPU-Accelerated Join**: Leverages CUDA for parallel spatial predicate evaluation
- **Streaming Integration**: Works with DataFusion's existing streaming infrastructure
- **Automatic Fallback**: Falls back to CPU when GPU is unavailable
- **Flexible Configuration**: Configurable device ID, batch size, and memory limits
- **Supported Predicates**: ST_Intersects, ST_Contains, ST_Within, ST_Covers, ST_CoveredBy, ST_Touches, ST_Equals

## Usage

### Prerequisites

**For GPU Acceleration:**
- CUDA Toolkit (11.0 or later)
- CUDA-capable GPU (compute capability 6.0+)
- Linux or Windows OS (macOS does not support CUDA)
- Build with `--features gpu` flag

**For Development Without GPU:**
- The package compiles and tests pass without GPU hardware
- Tests verify integration logic and API surface
- Actual GPU computation requires hardware (see Testing section below)

### Building

```bash
# Build with GPU support
cargo build --package sedona-spatial-join-gpu --features gpu

# Run tests
cargo test --package sedona-spatial-join-gpu --features gpu
```

### Configuration

GPU spatial join is disabled by default. Enable it via configuration:

```rust
use datafusion::prelude::*;
use sedona_common::option::add_sedona_option_extension;

let config = SessionConfig::new()
    .set_str("sedona.spatial_join.gpu.enable", "true")
    .set_str("sedona.spatial_join.gpu.device_id", "0")
    .set_str("sedona.spatial_join.gpu.batch_size", "8192");

let config = add_sedona_option_extension(config);
let ctx = SessionContext::new_with_config(config);
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `sedona.spatial_join.gpu.enable` | `false` | Enable GPU acceleration |
| `sedona.spatial_join.gpu.device_id` | `0` | GPU device ID to use |
| `sedona.spatial_join.gpu.batch_size` | `8192` | Batch size for processing |
| `sedona.spatial_join.gpu.fallback_to_cpu` | `true` | Fall back to CPU on GPU failure |
| `sedona.spatial_join.gpu.max_memory_mb` | `0` | Max GPU memory in MB (0=unlimited) |
| `sedona.spatial_join.gpu.min_rows_threshold` | `100000` | Minimum rows to use GPU |

## Testing

### Test Coverage

The test suite is divided into two categories:

#### 1. Structure and Integration Tests (No GPU Required)

These tests validate the API, integration with DataFusion, and error handling:

```bash
# Run unit tests (tests structure, not GPU functionality)
cargo test --package sedona-spatial-join-gpu

# Run integration tests (tests DataFusion integration)
cargo test --package sedona-spatial-join-gpu --test integration_test
```

**What these tests verify:**
- ✅ Execution plan creation and structure
- ✅ Schema combination logic
- ✅ Configuration parsing and defaults
- ✅ Stream state machine structure
- ✅ Error handling and fallback paths
- ✅ Geometry column detection
- ✅ Integration with DataFusion's ExecutionPlan trait

**What these tests DO NOT verify:**
- ❌ Actual GPU computation (CUDA kernels)
- ❌ GPU memory transfers
- ❌ Spatial predicate evaluation correctness on GPU
- ❌ Performance characteristics
- ❌ Multi-GPU coordination

#### 2. GPU Functional Tests (GPU Hardware Required)

These tests require an actual CUDA-capable GPU and can only run on Linux/Windows with CUDA toolkit installed:

```bash
# Run GPU functional tests (requires GPU hardware)
cargo test --package sedona-spatial-join-gpu --features gpu gpu_functional_tests

# Run on CI with GPU runner
cargo test --package sedona-spatial-join-gpu --features gpu -- --ignored
```

**Prerequisites for GPU tests:**
- CUDA-capable GPU (compute capability 6.0+)
- CUDA Toolkit 11.0 or later installed
- Linux or Windows OS (macOS not supported)
- GPU drivers properly configured

**What GPU tests verify:**
- ✅ Actual CUDA kernel execution
- ✅ Correctness of spatial join results
- ✅ GPU memory management
- ✅ Performance vs CPU baseline
- ✅ Multi-batch processing

### Running Tests Without GPU

On development machines without GPU (e.g., macOS), the standard tests will:
1. Compile successfully (libgpuspatial compiles without CUDA code)
2. Test the API surface and integration logic
3. Verify graceful degradation when GPU is unavailable
4. Pass without executing actual GPU code paths

This allows development and testing of the integration layer without GPU hardware.

### CI/CD Integration

GPU tests are automatically run via GitHub Actions on self-hosted runners with GPU support.

**Workflow**: `.github/workflows/rust-gpu.yml`

**Runner Requirements:**
- Self-hosted runner with CUDA-capable GPU
- Recommended: AWS EC2 g5.xlarge instance with Deep Learning AMI
- Labels: `[self-hosted, gpu, linux, cuda]`

**Setup Guide**: See [`docs/setup-gpu-ci-runner.md`](../../../docs/setup-gpu-ci-runner.md) for complete instructions on:
- Setting up AWS EC2 instance with GPU
- Installing CUDA toolkit and dependencies
- Configuring GitHub Actions runner
- Cost optimization tips
- Troubleshooting common issues

**Build Times** (g5.xlarge):
- libgpuspatial (CUDA): ~20-25 minutes (first build)
- GPU spatial join: ~2-3 minutes
- With caching: ~90% faster on subsequent builds

**Note:** GitHub-hosted runners do not provide GPU access. A self-hosted runner is required for actual GPU testing.
