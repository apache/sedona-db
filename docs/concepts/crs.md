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

# Coordinate Reference Systems

A Coordinate Reference System (CRS) describes how the numeric coordinates in a
geometry, geography, or raster relate to locations on the Earth. Two datasets
whose coordinates look similar are only comparable if they share a CRS, so
SedonaDB tracks a CRS alongside every spatial value and uses it to decide
whether an operation is well defined.

This page explains how SedonaDB *represents* a CRS, how it decides whether two
CRSes are *equal*, and what that means when you combine data from different
sources.

## The three forms of a CRS

A CRS definition attached to a spatial value can take one of three forms.
SedonaDB recognizes and stores whichever form you provide:

- **An authority code** — a compact `AUTHORITY:CODE` string such as
  `EPSG:4326`, `EPSG:3857`, `ESRI:102005`, or `OGC:CRS84`. This is the most
  common and most portable form. A bare numeric code like `4269` is accepted and
  interpreted as `EPSG:4269`.
- **PROJJSON** — a JSON object following the
  [PROJJSON schema](https://proj.org/en/stable/specifications/projjson.html).
  This is the fully self-describing form used when writing GeoParquet and
  Iceberg.
- **A WKT string** — a WKT1 or WKT2 CRS definition (`PROJCRS[...]`,
  `GEOGCRS[...]`, `PROJCS[...]`, and similar). This is the form most often found
  in GeoTIFF and other GDAL-sourced files.

You choose the form when you set a CRS (for example with `ST_SetCRS` /
`RS_SetCRS`), and files you read carry whichever form they were written with.

## The definition is preserved verbatim

SedonaDB does **not** rewrite a CRS you give it. Whatever form you supply is the
form stored and the form you get back:

```sql
-- Set as an authority code, read back as that same authority code
SELECT ST_CRS(ST_SetCRS(ST_Point(0.25, 0.25), 'EPSG:3857'));
-- => EPSG:3857
```

A WKT or PROJJSON definition is kept in full — SedonaDB will **not** collapse it
down to an embedded `AUTHORITY:CODE` even when one is present. Nothing is
discarded on the way in, so a consumer that wants the compact form can ask for
it, but one that needs the full definition still has it.

Internally there are two views of the stored definition:

- the **round-trippable / PROJ-consumable** form (what `ST_CRS` and `RS_CRS`
  return, and what is handed to PROJ at transform time), and
- the **field-metadata** form (what is embedded in GeoArrow / GeoParquet
  metadata).

Both preserve the original authority code, PROJJSON, or WKT as given. The
**only** canonicalization SedonaDB applies is for the longitude/latitude WGS84
aliases: `EPSG:4326` and `OGC:CRS84` describe the same datum but imply different
axis orders, so in the field-metadata form both are written as `OGC:CRS84` to
keep the axis order (longitude, latitude) explicit for downstream readers. More
generally, SedonaDB always assumes longitude, latitude axis order regardless of
what a CRS declares — many geographic CRSes, `EPSG:4326` included, officially
specify latitude first, but coordinates are longitude-first everywhere and
`ST_Transform` normalizes every transform to that order. The PROJ-consumable
form still returns the authority code exactly as you set it.

## Column-level and row-level CRS

For **vector** data — geometry and geography — a CRS can attach at two levels.
A **column-level** CRS is a property of the whole column: the column carries a
single CRS in its type (and, on disk, in its field metadata), and every value
in the column shares it.
This is the form `ST_CRS` and `ST_SRID` read, the form the equality rules below
compare, and the one `ST_SetCRS` sets.

A **row-level** CRS instead travels with each value, which carries its own CRS
rather than inheriting one from the column. Internally this is the value paired
with a per-row CRS string.

Most vector functions accept either form. For a row-level input, SedonaDB
applies the CRS rules per value: a function whose result does not depend on the
CRS (such as `ST_Area`) simply uses the geometry; a function taking two or more
geometries requires their CRSes to be compatible; and a function that returns a
geometry carries the CRS through to its result. The important difference from a
column-level CRS is *when* compatibility is enforced — a column-level mismatch
is caught at query-planning time (see below), whereas a row-level CRS is not
known until the data flows, so it is reconciled while the query runs.

**Rasters do not have a column-level CRS.** A raster carries its own CRS as part
of the raster value, so every raster in a column is self-describing; `RS_CRS` /
`RS_SRID` read the CRS from each raster and `RS_SetCRS` sets it on individual
rasters.

## Equality: how SedonaDB compares two CRSes

Deciding whether two CRSes are "the same" is the subtle part, and SedonaDB's
rule is deliberately simple and conservative:

- **If both CRSes expose an authority code, the authority codes are compared.**
  The issuing authority is treated as the source of truth about identity, so
  `EPSG:3857` equals `EPSG:3857` regardless of how each side was originally
  spelled. (The `EPSG:4326` / `OGC:CRS84` lon/lat pair is the one built-in
  alias that compares equal.)
- **Otherwise, SedonaDB falls back to a structural / string comparison** of the
  definitions — matching PROJJSON objects structurally, or comparing WKT strings
  directly.

This comparison is intentionally **lenient**: it can *false-negative* (report
two definitions as different when a full geodetic analysis would call them
equivalent), but it will never *false-positive* (it never claims two genuinely
different CRSes are the same). SedonaDB keeps the PROJ library out of the core
schema layer entirely, so there is no semantic-equivalence check here. This is a
different trade-off from, for example, GeoPandas, which compares CRSes through
PROJ using an approximate, confidence-based similarity threshold.

### What this means for you

**Vector** operations that require their inputs to share a CRS — spatial joins,
`ST_Intersects`, distance predicates, and so on — **raise an error on a
mismatch** rather than silently reprojecting one side to match the other.
Silently reprojecting geometry coordinates is exactly the kind of hidden,
easy-to-miss behavior that produces wrong answers, so SedonaDB refuses to guess:

```
Error during planning: Mismatched CRS arguments: epsg:3857 vs epsg:4326
Use ST_Transform() or ST_SetSRID() to ensure arguments are compatible.
```

To fix a mismatch, make the CRSes agree before the operation:

- If the coordinates are in different systems, reproject one side with
  `ST_Transform` (which actually moves the coordinates).
- If a value is simply *missing* a CRS, or is labeled incorrectly, attach the
  right one with `ST_SetCRS` / `ST_SetSRID` (metadata only — no coordinates
  change).

Because equality can false-negative, two definitions that *are* equivalent but
are spelled differently (say, an authority code on one side and an
authority-less WKT on the other) may be reported as mismatched. The fix is the
same: normalize both sides to the same definition with `ST_Transform` or
`ST_SetCRS`.

### Raster joins are an exception

A spatial join involving a raster tests the raster's **footprint** (the polygon
through its corners), not its pixels. When a raster is joined against a geometry,
SedonaDB reconciles the two CRSes as follows:

- **They share a CRS, or both are CRS-less** — the footprints are compared
  directly, with no reprojection.
- **Both have a CRS but they differ** — SedonaDB reprojects the raster's
  footprint into the geometry's CRS and the join proceeds, rather than erroring
  as the vector path would. Reprojecting a footprint is far cheaper and safer
  than resampling the pixel grid.
- **Only one side has a CRS** — this is an error, just as on the vector path:
  SedonaDB will not reproject into or out of an unknown frame.

The reprojected case is a close approximation rather than an exact boundary:
reprojection is sometimes non-linear, so the footprint's straight edges may bow
into curves in the target CRS, and reprojecting only the four corners can
under-cover the true extent. SedonaDB handles this by **densifying** each edge —
adding interior points in the raster's own CRS, where the edges are still
straight — then reprojecting all of them, so the reprojected footprint follows
the curve.

One performance note: an accelerated spatial join relies on a spatial index
built over a single, column-level CRS. A geometry or raster column that carries
a *row-level* CRS can't use that index, so the join falls back to a slower
per-row evaluation — prefer a column-level CRS for join-heavy work.

## SRID vs CRS

An **SRID** (Spatial Reference Identifier) is just the numeric-code view of a
CRS. `ST_SRID` / `RS_SRID` return the integer, and `ST_SetSRID` / `RS_SetSRID`
set a CRS from one. SedonaDB maps between the two the obvious way: SRID `0`
means "no CRS", `4326` maps to `OGC:CRS84`, and any other value `N` maps to
`EPSG:N`. A CRS only has an SRID when it carries an EPSG (or lon/lat) authority
code; a purely custom WKT or PROJJSON definition with no authority has no SRID.

## Internals / design

A few deeper points, for the curious:

- **The core schema layer is PROJ-free and GDAL-free.** CRS representation,
  storage, and equality all live in the `sedona-schema` crate, which has no
  dependency on PROJ or GDAL. This keeps the low-level crate light and means
  equality never depends on a native geodetic library being present.
- **PROJ is used only at transform time.** Reprojection (`ST_Transform`) is
  performed by a pluggable CRS engine; the default is a PROJ-backed engine that
  lives in its own crate and is injected into the session. The stored CRS
  definition — including verbatim WKT — is handed to PROJ only when a transform
  actually runs.
- **A small in-crate WKT parser extracts identity without pulling in PROJ.** To
  read an authority code, SRID, or ellipsoid parameters out of a WKT definition
  (needed for equality and for spherical geography), `sedona-schema` uses a
  tiny, self-contained WKT-node parser rather than linking PROJ. It reads the
  authority only from the top-level `AUTHORITY[...]` / `ID[...]` tag, so a custom
  CRS is never misidentified by an authority tag buried on a nested unit or
  projection parameter. A WKT that is malformed or carries no top-level
  authority is still stored verbatim — it simply has no derived SRID.

## See also

- [Joining Spatial Data with Different Coordinate Systems](../crs-examples.md) —
  a worked example of the CRS-mismatch error and how to resolve it across a
  spatial join.
