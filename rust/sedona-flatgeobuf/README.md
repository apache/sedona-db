# Native FlatGeobuf reader (draft)

`FlatGeobufFormatSpec` implements SedonaDB's `ExternalFormatSpec` for local
FlatGeobuf files. Register it explicitly with `sedona_datasource::provider::external_table`.
It does not change the Python connection's existing Pyogrio registrations.

## Read contract

- Header-defined attributes precede the nullable `geometry` column, encoded as ISO WKB with
  GeoArrow metadata. Preserve Z/M dimensions and header CRS. JSON and DateTime
  properties remain UTF-8 strings; no timezone or JSON reinterpretation.
- Produce only requested columns in bounded batches, including zero-column
  batches for count scans. Missing attributes remain null.
- Indexed files may be divided into arbitrary byte ranges. A feature belongs
  to the range containing the first byte of its size prefix. Metadata is shared
  across readers of the same file version, while feature streams are independent.
- Unindexed files use a serial fallback: the partition owning byte zero reads
  the entire file; other partitions yield no rows. Validate framing once before
  decoding; do not advertise parallel payload reads for these files.
- Only local `file:` URLs are supported. Curves and temporal T/TM dimensions,
  remote object stores, spatial pruning and writing are outside this draft.
  Feature-local column overrides are rejected explicitly.
- Reject invalid framing, unsupported schema types and inconsistent requested
  schemas. A bounded metadata cache is invalidated when local file size or
  modification time changes. Inputs must remain unchanged during a scan.

## Verification

Run `cargo test -p sedona-flatgeobuf`. Tests generate their own FlatGeobuf files;
no third-party binary fixtures are bundled. Implementation and validation results
will be recorded in the draft PR after these checks complete.

## Explicit registration

```rust
use std::sync::Arc;
use datafusion::{prelude::SessionContext, datasource::listing::ListingTableUrl};
use sedona_datasource::provider::external_table;
use sedona_flatgeobuf::FlatGeobufFormatSpec;

// In an async function:
let context = SessionContext::new();
let table = external_table(
    Arc::new(FlatGeobufFormatSpec::default()),
    &context,
    vec![ListingTableUrl::parse("file:///data/roads.fgb")?],
    true,
    Some(vec![]),
).await?;
context.register_table("roads", table)?;
```

The crate documentation compiles this registration example. The reader uses the
released FlatGeobuf crate to verify and decode each feature buffer, followed by
bounds-checked ISO WKB encoding. It does not require the earlier research fork.
Batch size bounds rows per batch; it is not a byte-memory limit or a zero-copy
claim. Index metadata uses one offset per feature, retained for at most 16 files.
This draft validates correctness, not a throughput improvement.
