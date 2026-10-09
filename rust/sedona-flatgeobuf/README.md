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
use `sd.read(path, format="fgb_native")` to opt in to native range reads.
`FlatGeobufFormatSpec` retains the earlier local external datasource adapter.

## Read contract

- Header-defined attributes precede the nullable `wkb_geometry` column, encoded as ISO WKB with
  GeoArrow metadata. Preserve Z/M dimensions and header CRS. JSON uses Utf8 with
  the `arrow.json` extension; RFC3339 DateTime strings are parsed to UTC
  microsecond timestamps (sub-microsecond precision is truncated). Invalid dates
  return an error.
- Produce only requested columns in bounded batches, including zero-column
  batches for count scans. Missing attributes remain null.
- Indexed files may be divided into arbitrary byte ranges. A feature belongs
  to the range containing the first byte of its size prefix. Metadata is shared
  across readers of the same file version, while feature streams are independent.
- Unindexed files use a serial fallback: the partition owning byte zero reads
  the entire file; other partitions yield no rows. Validate framing once before
  decoding; do not advertise parallel payload reads for these files.
- Any registered ObjectStore can provide range reads. Curves and temporal T/TM
  dimensions, spatial pruning, writing and morselized scans are outside this draft.
  Feature-local column overrides and absent header column schemas are rejected explicitly.
- Reject invalid framing, unsupported schema types and inconsistent requested
  schemas. Native metadata cache entries are validated against ObjectMeta and
  store identity. Arrow schemas are derived from cached FlatGeobuf headers using
  the requested geometry column option. Inputs must remain
  unchanged during a scan. Multi-file reads require consistent schemas and CRS.

The `metadata_size_hint` option controls the initial header prefix read (default
65536 bytes); incomplete headers are refetched before verified decoding.
Feature payload requests coalesce adjacent records up to an 8 MiB target; a
single indivisible feature may exceed this target.

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
claim. Index metadata uses one offset per feature and fetches only RTree leaves.
The native cache uses DataFusion's configured memory limit; the local adapter
opens asynchronously and does not retain its own metadata cache.
This draft validates correctness, not a throughput improvement.
