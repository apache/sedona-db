<!--
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements. See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# ObjectStore C interface, version 1

This interface passes a configured host ObjectStore into an independently built
format extension. It implements the read path: HEAD, conditional/ranged GET,
recursive listing (including offsets), and delimiter listing. Writes, multipart
uploads, copy, and delete return `NotSupported` in the Rust importer. FileFormat
registration and pointcloud packaging are separate work.

The ABI declarations are in `ffi.rs` and `../sedona_extension.h`. The public Rust
adapters are `ExportedObjectStore` and `ImportedObjectStore`.

## Ownership and execution

* A store has ABI version 1, capability bits, a thread-safe `call`, and `release`.
  Reserved fields must be zero. Unknown capability bits may be ignored; callers
  check the particular capability before dispatch. A raw C store is explicitly
  released; its Rust representation deliberately does not have a synchronous
  destructor. Successful import moves it and clears the caller's value. Failed
  import leaves ownership with the caller.
* Requests, their JSON bytes, and the call context are borrowed until `call`
  returns. Outputs and the error must initially be zero. The caller owns every
  populated output, even on failure. Callbacks must not unwind across C.
* A GET returns metadata and an optional single-use stream handle. HEAD has no
  stream. Other GETs and all listings must have a stream, including empty results.
  Metadata describes the same request that produced the body; there is no HEAD +
  GET emulation. All owned buffers have an allocator-side release callback.
  Static borrowed buffers may omit release if their lifetime covers the call's
  entire response consumption.
* `stream.start(handler)` returns zero after accepting the handler. It may invoke
  callbacks before returning. From acceptance, only the producer releases the
  handler, including cancellation and shutdown. On nonzero return it must not
  have invoked callbacks or retained the handler. The caller always releases the
  stream handle after start returns. The producer owns independent resources;
  releasing the started handle must not invalidate the producer.
* Stream callbacks follow the existing Arrow asynchronous device stream protocol.
  Only CPU arrays are accepted. Successful completion requires an explicit null
  task; errors terminate the stream. Handler release without completion is an
  error unless the consumer has abandoned the stream. Request/cancel must return
  promptly; extract must only extract ready data. The importer tolerates a stalled
  callback by abandoning its wait, not by freeing memory that callback still uses.
* The importer retains a library owner supplied by its caller, the store, and the
  operation's admission slot through pending callbacks and handler release. It
  copies wire data on the foreign executor before exposing it to queries. This
  intentionally avoids executing a foreign Arrow/buffer destructor on a query
  thread. A provider must also keep its own code loaded while its asynchronous
  tasks execute after calling handler release; consumer ownership is not a
  mechanism for safely unloading arbitrary running plugin threads.
  Zero-copy byte import can be considered separately with equivalent
  destruction isolation.

The default foreign executor has four threads and 256 admission slots shared by
all default imports within a linked copy of `sedona-extension`. Each imported store and each live operation holds one slot.
Cloning the Rust adapter does not allocate another store slot. Pending native
streams yield executor threads; no thread is dedicated to each stream. All
outbound foreign callbacks, stream polling/extraction, and result destruction run
on that executor. Inbound producer callbacks run on the provider's thread.

Admission fails with `WouldBlock` when full. Slots remain charged during cleanup,
including stuck callbacks and producers that never release their handlers. This
bounds accumulated abandoned calls; there is no unbounded replacement-thread
policy. Callers may share an explicitly configured `ObjectStoreExecutor` instead.

## Cancellation and deadlines

Every operation has independent cancellation state. Dropping a future or stream
signals local cancellation without calling foreign code or joining workers.
The worker subsequently cancels/drops its producer. A shared store is never
cancelled globally.

`ObjectStoreOptions` supplies a startup/call timeout, idle timeout, and total
operation timeout (defaults: 30 seconds, 30 seconds, 5 minutes). Admission,
queueing, the C call, and stream startup are covered by the startup budget. Calls
that expire before dispatch do not enter the provider. Deadline timers run on a
separate runtime, so exhausted foreign threads and callers without a Tokio runtime
still get timeouts. The provider receives a remaining millisecond budget and a
thread-safe cancellation callback; it must not retain this borrowed context.
Our exporter aborts its spawned GET future when that budget expires or the caller
cancels. It requires an actively driven provider runtime.

An idle timer starts when the consumer polls for data and resets after delivering
an item. Consumer think time/backpressure does not count as idle time. The total
operation budget includes startup and remains fixed across items. A caller that
needs a query-wide deadline should additionally drop the query's operations when
that deadline expires; this API does not discover a DataFusion query context.

Timeout guarantees consumer responsiveness, not termination of arbitrary native
code. Stalled callbacks may retain resources within the admission limit. A
provider violating pointer validity or calling back after release can corrupt the
process. Forcefully stopping such code requires process isolation.

## Request and response schemas

JSON is UTF-8, limited to 1 MiB per request or metadata response. Byte offsets,
lengths, and sizes are unsigned 64-bit integers. Dates are RFC3339 strings retaining
subsecond precision. Object paths retain `object_store::path::Path` semantics.
Unknown request fields/options are rejected rather than silently discarded.
Rust `GetOptions.extensions` is not portable and is rejected when nonempty.

GET request (all fields shown):

```json
{
  "operation": "get",
  "location": "path/to/object",
  "options": {
    "if_match": null,
    "if_none_match": null,
    "if_modified_since": null,
    "if_unmodified_since": null,
    "range": {"kind": "bounded", "start": 0, "end": 16},
    "version": null,
    "head": false
  }
}
```

Other range representations are `{"kind":"offset","offset":16}` and
`{"kind":"suffix","length":16}`; null requests the entire object. GET response:

```json
{
  "meta": {
    "location": "path/to/object",
    "last_modified": "2026-01-01T00:00:00Z",
    "size": 32,
    "e_tag": null,
    "version": null
  },
  "range_start": 0,
  "range_end": 16,
  "attributes": [["content_type", "application/octet-stream"]]
}
```

Attribute keys are `content_disposition`, `content_encoding`, `content_language`,
`content_type`, `cache_control`, `storage_class`, or `{"metadata":"user-key"}`.
Unknown attributes cause `NotSupported`. The returned range is checked against
object size, and body length must exactly match that range. Local-file payloads
are normalized into streamed byte chunks.

Listing request:

```json
{"operation":"list","prefix":"path","offset":null,"delimiter":false}
```

A non-null offset is exclusive. Delimiter listing must have a null offset.
Listing has an empty metadata response and a stream of entries. Object entries
are `{"kind":"object","meta":{...}}`, using the metadata object above. Common
prefixes are `{"kind":"prefix","path":"path/subdir"}` and occur only for
delimiter listings. The Rust adapter collects delimiter results because the
Rust trait returns a `ListResult`; the native backend may also materialize those
results. The FFI transfer itself is batched.

## Stream schema and limits

Both stream types use Arrow record batches with these fields, in this order:

| Name | Type | Nullable |
| --- | --- | --- |
| payload | Binary | yes |
| error | Utf8 | yes |

Schema metadata must contain exactly `sedona.object_store.version = "1"` and
`sedona.object_store.kind = "bytes"` or `"list"`. Each row has exactly one non-null
field. Byte payloads concatenate in order. List payloads contain one UTF-8 JSON
entry per row. There is at most 1 MiB of payload per batch and at most 128 rows.
The exporter splits large byte chunks and batches available listing entries
without waiting for a full batch. These limits bound wire batches, not allocations
already made by an underlying store or a collected `ListResult`.

The importer uses a request window of one and rejects more than two queued tasks.
Its outgoing locally owned batch channel has capacity one. Consumers must honor
request credit; the small queue allowance handles refill/extraction overlap.
A single-row error batch contains a null payload and a JSON error string:

```json
{"code":2,"path":"path/to/object","message":"object was not found"}
```

An error row terminates consumption and cancels the producer. Protocol-level
Arrow errors are separate from these typed storage errors. The same codes are
returned by `call`/`start`, with the error message in `SedonaCError` (the requested
path supplies the path for call errors):

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | Generic storage failure |
| 2 | Not found |
| 3 | Unsupported operation |
| 4 | Already exists |
| 5 | Precondition failed |
| 6 | Not modified |
| 7 | Permission denied |
| 8 | Unauthenticated |
| 9 | Invalid request, path, or response |
| 10 | Cancelled (`io::ErrorKind::Interrupted`) |
| 11 | Timed out (`io::ErrorKind::TimedOut`) |
| 12 | Capacity exhausted (`io::ErrorKind::WouldBlock`) |

The last four are represented by `object_store::Error::Generic` containing a typed
`std::io::Error` on import. Native `NotFound` and precondition categories remain
available to consumers such as format readers deciding whether a sidecar exists.
