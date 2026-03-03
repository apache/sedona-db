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

# Memory Management and Spilling

SedonaDB supports memory-limited execution with automatic spill-to-disk,
allowing you to process datasets that are larger than available memory. When a
memory limit is configured, operators that exceed their memory budget
automatically spill intermediate data to temporary files on disk and read them
back as needed.

## Configuring Memory Limits

Set `memory_limit` on the context options to cap the total memory available for
query execution. The limit accepts an integer (bytes) or a human-readable string
such as `"4gb"`, `"512m"`, or `"1.5g"`.

```python
import sedona.db

sd = sedona.db.connect()
sd.options.memory_limit = "4gb"
```

Without a memory limit, SedonaDB uses an unbounded memory pool and operators
can use as much memory as needed (until the process hits system limits). In
this mode, operators typically won't spill to disk because there is no memory
budget to enforce.

!!! note
    All runtime options (`memory_limit`, `memory_pool_type`, `temp_dir`,
    `unspillable_reserve_ratio`) must be set **before** the first query is
    executed. Once the first query runs, the internal execution context is
    created and these options become read-only.

## Memory Pool Types

The `memory_pool_type` option controls how the memory budget is distributed
among concurrent operators. Two pool types are available:

- **`"greedy"`** -- Grants memory reservations on a first-come-first-served
  basis. This is the default when no pool type is specified. Simple, but can
  lead to memory reservation failures under pressure -- one consumer may
  exhaust the pool before others get a chance to reserve memory.

- **`"fair"` (recommended)** -- Distributes memory fairly among spillable
  consumers and reserves a fraction of the pool for unspillable consumers.
  More stable under memory pressure and significantly less likely to cause
  reservation failures, at the cost of slightly lower utilization of the total
  reserved memory.

We recommend using `"fair"` whenever a memory limit is configured:

```python
sd = sedona.db.connect()
sd.options.memory_limit = "4gb"
sd.options.memory_pool_type = "fair"
```

!!! note
    `memory_pool_type` only takes effect when `memory_limit` is set.

### Unspillable reserve ratio

When using the `"fair"` pool, the `unspillable_reserve_ratio` option controls
the fraction of the memory pool reserved for unspillable consumers (operators
that cannot spill their memory to disk). It accepts a float between `0.0` and
`1.0` and defaults to `0.2` (20%) when not explicitly set.

```python
sd = sedona.db.connect()
sd.options.memory_limit = "8gb"
sd.options.memory_pool_type = "fair"
sd.options.unspillable_reserve_ratio = 0.3  # reserve 30% for unspillable consumers
```

## Temporary Directory for Spill Files

By default, DataFusion uses the system temporary directory for spill files. You
can override this with `temp_dir` to control where spill data is written -- for
example, to point to a larger or faster disk:

```python
sd = sedona.db.connect()
sd.options.memory_limit = "4gb"
sd.options.memory_pool_type = "fair"
sd.options.temp_dir = "/mnt/fast-ssd/sedona-spill"
```

## Full Example

```python
import sedona.db

sd = sedona.db.connect()

# Cap execution memory at 4 GB
sd.options.memory_limit = "4gb"

# Use the fair pool for stable memory distribution (recommended)
sd.options.memory_pool_type = "fair"

# Reserve 20% of the pool for unspillable consumers (default)
sd.options.unspillable_reserve_ratio = 0.2

# Write spill files to a dedicated directory
sd.options.temp_dir = "/tmp/sedona-spill"

# Now execute queries -- options are frozen after the first query runs
# Example: configure DataFusion settings and then run your workload
sd.sql("SET datafusion.execution.spill_compression = 'lz4_frame'").execute()

df = sd.sql("""
SELECT a.id, b.id
FROM a
JOIN b
  ON ST_Intersects(a.geom, b.geom)
""")
```

## Operators Supporting Memory Limits

When a memory limit is configured, the following operators automatically spill
intermediate data to disk when they exceed their memory budget:

In practice, this means memory limits and spilling can apply to both SedonaDB's
spatial operators and DataFusion's general-purpose operators used by common SQL
constructs:

**SedonaDB:**

- **Spatial joins** -- Both the build-side (index construction, partition
  collection) and probe-side (stream repartitioning) of SedonaDB's spatial
  joins support memory-pressure-driven spilling.

**DataFusion (physical operators):**

This list is not exhaustive. Many other DataFusion physical operators and
execution strategies may allocate memory through the same runtime memory pool
and may spill to disk when memory limits are enforced.

- **`ORDER BY` / sorted Top-K** (`SortExec`) -- External sort that
  spills sorted runs to disk when memory is exhausted, then merges them.
- **Many joins** (`HashJoinExec`) -- Hash join that spills hash table partitions
  to disk under memory pressure.
- **Sort-merge joins** (`SortMergeJoinExec`) -- Sort-merge join that spills
  buffered batches to disk when the memory limit is exceeded.
- **`GROUP BY` aggregations** (`AggregateExec`) -- Grouped aggregation that
  spills intermediate aggregation state to sorted spill files when memory is
  exhausted.

## Advanced DataFusion Configurations

DataFusion provides additional execution configurations that affect spill
behavior. These can be set via SQL `SET` statements after connecting:

!!! note
    `SET` is executed as a query. Configure `sd.options.*` runtime options (like
    `memory_limit` and `temp_dir`) before running any `SET` statements.

### Spill compression

By default, data is written to spill files uncompressed. Enabling compression
reduces the amount of disk I/O and disk space used at the cost of additional
CPU work. This is beneficial when disk I/O throughput is low or when disk space
is not large enough to hold uncompressed spill data.

```python
# Enable LZ4 compression for spill files
sd.sql("SET datafusion.execution.spill_compression = 'lz4_frame'").execute()
```

### Maximum temporary directory size

DataFusion limits the total size of temporary spill files to prevent unbounded
disk usage. The default limit is **100 GB**. If your workload needs to spill
more data than this, increase the limit:

```python
# Increase the spill directory size limit to 500 GB
sd.sql("SET datafusion.runtime.max_temp_directory_size = '500gb'").execute()
```

## System Configuration

### Maximum number of open files

Large workloads that spill heavily can create a large number of temporary files.
During a spatial join, each parallel execution thread may create one spill file
per spatial partition. The total number of open spill files can therefore reach
**parallelism x number of spatial partitions**. For example, on an 8-CPU host
running a spatial join that produces 500 spatial partitions, up to
**8 x 500 = 4,000** spill files may be open simultaneously -- far exceeding the
default per-process file descriptor limit.

The operating system's per-process file descriptor limit must be high enough to
accommodate this, otherwise queries will fail with "too many open files" errors.

**Linux:**

The default limit is typically 1024, which is easily exceeded by spill-heavy
workloads like the example above.

To raise the limit permanently, add the following to `/etc/security/limits.conf`:

```
*    soft    nofile    65535
*    hard    nofile    65535
```

Then log out and back in (or reboot) for the change to take effect. Verify with:

```bash
ulimit -n
```

**macOS:**

```bash
ulimit -n 65535
```

This affects the current shell session. Persistent/system-wide limits are OS
and configuration dependent; consult your macOS configuration and documentation
if you need to raise the hard limit.
