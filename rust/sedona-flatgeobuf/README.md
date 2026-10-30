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

# Native FlatGeobuf reader (draft)

`FlatGeobufFormatFactory` implements DataFusion's native `FileFormat` API,
using generic `ObjectStore` range reads and DataFusion's file metadata cache.
Enable Sedona's `fgb` feature for automatic registration; Python enables it by
default as a compiled capability. Python keeps pyogrio/GDAL as its `fgb` default;
use `sd.read(path, format="fgb_native")` to opt in to native reads.
`FlatGeobufFormatSpec` retains the earlier local external datasource adapter.

## Read contract

- Header-defined attributes precede the nullable `wkb_geometry` column, encoded as ISO WKB with
  GeoArrow metadata. Preserve Z/M dimensions and header CRS. JSON uses Utf8 with
  the `arrow.json` extension; RFC3339 DateTime strings are parsed to UTC
  microsecond timestamps (sub-microsecond precision is truncated). Invalid dates
  return an error.
- Produce only requested columns in bounded batches, including zero-column
  batches for count scans. Missing attributes remain null.
- Native scans read each file sequentially, skipping the RTree without loading
  feature offsets. DataFusion redistributes decoded batches with batch round robin
  when downstream work benefits from parallelism. Separate files can be read
  concurrently; a single file is not divided into independent byte ranges.
- Indexed and unindexed payloads use the same sequential framing decoder. Requests
  use 8 MiB buffers independently of Arrow row batch size, including features
  crossing request boundaries. Known feature counts are checked during iteration.
- Any registered ObjectStore can provide range reads. Curves and temporal T/TM
  dimensions, spatial pruning, writing and morselized scans are outside this draft.
  Feature-local column overrides and absent header column schemas are rejected explicitly.
- Reject invalid framing and unsupported schema types. Requested table fields
  are matched by name, missing fields become null, and supported Arrow casts adapt
  attribute types with conversion errors reported explicitly. Geometry/CRS
  metadata must remain compatible. Native metadata cache entries are validated against ObjectMeta and
  store identity. Arrow schemas are derived from cached FlatGeobuf headers using
  the requested geometry column option. Inputs must remain
  unchanged during a scan. Multi-file inferred schemas must merge successfully and retain consistent CRS.

The `metadata_size_hint` option controls the initial header prefix read (default
65536 bytes); incomplete headers are refetched before verified decoding.
Payload requests read at most 8 MiB at a time. A single feature may require
multiple requests and its complete frame must fit in memory before decoding.

The `geometry_column_name` format option renames the geometry field (default
`wkb_geometry`). Header conversion preserves duplicate names; DataFusion handles
name validation when the schema is registered.

## Verification

Run `cargo test -p sedona-flatgeobuf`. Tests generate their own FlatGeobuf files;
no third-party binary fixtures are bundled. Implementation and validation results
will be recorded in the draft PR after these checks complete.

The reader uses the
released FlatGeobuf crate to verify and decode each feature buffer, followed by
bounds-checked ISO WKB encoding. It does not require the earlier research fork.
Batch size bounds rows per batch; it is not a byte-memory limit or a zero-copy
claim. Native metadata contains only header information and the payload start.
The legacy local adapter alone retains per-feature offsets for its byte-range API.
The native cache uses DataFusion's configured memory limit; the local adapter
opens asynchronously and does not retain its own metadata cache.
This draft validates correctness, not a throughput improvement.
