<!--
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

# Release Notes

## SedonaDB 0.3.0

### Highlights

* GPU spatial join library integration
* LAS/LAZ point cloud format support
* Raster function expansion (RS_BandPath, RS_GeoReference, RS_NumBands, RS_ConvexHull, RS_SetSRID, RS_SetCRS)
* Partitioned KNN join for larger-than-memory datasets
* Auto-repartition probe side for balanced spatial join workload
* SedonaFairSpillPool memory pool with CLI memory limit support
* Parameter binding support in Python and R
* GDAL/OGR read and write support via pyogrio
* GeoParquet configuration options via SQL

### New Features

* feat(rust/sedona-pointcloud): Initial LAZ format support (#471)
* feat(rust/sedona-pointcloud): Add laz chunk statistics (#604)
* feat(rust/sedona-pointcloud): Add LAS support (#628)
* feat(c/sedona-libgpuspatial): Add Rust Wrapper (#586)
* feat(c/sedona-libgpuspatial): Refactoring GPU Spatial Join Library (#556)
* feat(rust/sedona-functions): Add raster display support to SD_Format (#591)
* feat(rust/sedona-functions): Implement ST_Force2D, ST_Force3D, ST_Force3DM, ST_Force4D (#606, #620)
* feat(rust/sedona-functions): Implement ST_KNN() simplified call (#667)
* feat(rust/sedona-functions): Add sd_simplifystorage utility (#650)
* feat(rust/sedona-functions): Add item SRID support to geometry constructors (#574)
* feat(rust/sedona-raster-functions): Add RS_BandPath (#603)
* feat(rust/sedona-raster-functions): Add RS_GeoReference (#601)
* feat(rust/sedona-raster-functions): Add RS_NumBands (#602)
* feat(rust/sedona-raster-functions): Add RS_SetSRID/RS_SetCRS with batch-local cache refactoring (#630)
* feat(rust/sedona-raster-functions): Add RS_ConvexHull and item-level CRS to RS_Envelope (#597)
* feat(rust/sedona-geoparquet): Ensure GeoParquet configuration options for read and write can be passed via SQL (#607)
* feat(rust/sedona-geoparquet): Support WKB validation in read_parquet() (#578)
* feat(rust/sedona-spatial-join): Support partitioned KNN join to handle larger than memory object side (#573)
* feat(rust/sedona-spatial-join): Auto-repartition probe side for balancing spatial join workload (#610)
* feat(rust/sedona-spatial-join): Automatic query-side filter pushdown for KNN joins (#641)
* feat(rust/sedona): Add SedonaFairSpillPool memory pool and CLI memory limit support (#599)
* feat(rust/sedona): Auto-configure spilled batch in-memory size threshold based on global memory limit (#680)
* feat(rust/sedona-schema): Improve deserialization of authority/code CRSes (#666)
* feat(rust/sedona-geometry): Let CrsTransform handle M coordinate (#619)
* feat(python/sedonadb): Implement parameter binding (#575)
* feat(python/sedonadb): Handle crs-like objects in parameterized queries (#660)
* feat(python/sedonadb): Expose memory pool and runtime configuration in Python bindings (#608)
* feat(python/sedonadb): Write GDAL/OGR via pyogrio (#632)
* feat(python/sedonadb): Add sort by geometry support in to_parquet() (#642)
* feat(r/sedonadb): Implement GDAL read via sf/arrow in R bindings (#670)
* feat(r/sedonadb): Add R bindings for parameterized queries (#662)
* feat(r/sedonadb): Port Python configuration options changes to R (#658)
* feat(r/sedonadb): Improve DataFrame API for R bindings (#651)
* feat(docs): Add PostGIS integration guide with GeoPandas and ADBC examples (#543)

### Bug Fixes

* fix(rust/sedona-spatial-join): Wrap probe-side repartition in ProbeShuffleExec to prevent optimizer stripping (#677)
* fix(rust/sedona-spatial-join): Prevent filter pushdown past KNN joins (#611)
* fix(rust/sedona-spatial-join): Reimplement planner for spatial join (#562)
* fix(rust/sedona-geoparquet): Ensure that GeoParquet files are always written with PROJJSON CRSes (#669)
* fix(rust/sedona-functions): Ensure WkbView types can be aggregated using the groups accumulator for ST_Envelope_Agg (#656)
* fix(rust/sedona-functions): Propagate NULL for scalar NULL SRID/CRS in ST_SetSRID/ST_SetCRS (#629)
* fix(rust/sedona): Fix panic when displaying very long content (#565)
* fix(rust/sedona): Fix false feature flag (#618)
* fix(raster): RS_Envelope returns axis-aligned bounding box for skewed rasters (#594)
* fix(c/sedona-geos): Support export of geometries with M values from GEOS (#640)
* fix(python/sedonadb): Add pandas < 3.0 Series entry to SPECIAL_CASED_LITERALS (#609)

### Improvements

* docs: Add memory management and spill configuration guide (#679)
* chore(docs/reference/sql): Migrate function documentation defined in Rust into Markdown (#616)
* chore(docs/reference): Migrate sql.md to new functions format (#585)
* chore(rust): Remove documentation field from SedonaScalarUDF and SedonaAggregateUDF (#633)
* refactor(rust/sedona-raster): Pre-downcast band metadata arrays in RasterStructArray (#588)
* refactor(rust/sedona-raster-functions): Extract CachedCrsToSRIDMapping and simplify SRID/CRS logic (#590)

## SedonaDB 0.2.0

### Highlights

* GeoParquet 1.1 write support
* GDAL/OGR format support via pyogrio in Python
* KNN join refactored with new geo-index trait and lock-free shared geometry cache
* New spatial functions: ST_Reverse, ST_Dump, ST_Translate, ST_Points, ST_NPoints, ST_PointN, ST_StartPoint, ST_EndPoint, ST_GeometryN, ST_Azimuth, ST_IsClosed, ST_IsCollection, ST_ZMFlag, ST_Buffer (full params)
* GEOS-backed functions: ST_IsValid, ST_IsValidReason, ST_IsRing, ST_IsSimple, ST_UnaryUnion, ST_Simplify, ST_SimplifyPreserveTopology, ST_Boundary, ST_Snap, ST_Polygonize, ST_MakeValid, ST_MinimumClearance, ST_MinimumClearanceLine
* Raster functions: RS_Width, RS_Height, RS_Example
* R package with FFI support, Parquet write, and runtime PROJ linking
* Python UDFs support
* Example Rust project
* Spatial predicate pruning with Covers filter for ST_Equals, overlaps, and crosses
* DataSourceExec metrics for spatial predicate pruning
* Ordering framework for geometry/geography types

### New Features

* feat(rust/sedona-geoparquet): GeoParquet 1.1 write support (#175)
* feat(rust/sedona-functions): Implement ST_Reverse, ST_Dump, ST_Translate, ST_Points, ST_NPoints, ST_PointN, ST_StartPoint, ST_EndPoint, ST_GeometryN, ST_IsClosed, ST_IsCollection, ST_ZMFlag, ST_Azimuth (#219, #257, #260, #265, #267, #269, #255, #245, #317, #183)
* feat(rust/sedona-functions): Add SRID argument to ST_Point() (#275)
* feat(rust/sedona-functions): Implement ordering framework for geometry/geography (#360)
* feat(rust/sedona-functions): Add ST_GeometryFromText alias for ST_GeomFromWKT (#213)
* feat(rust/sedona-functions): Implement ST_Crosses and ST_Overlaps predicates (#204)
* feat(c/sedona-geos): Implement ST_IsValid, ST_IsValidReason, ST_IsRing, ST_IsSimple, ST_UnaryUnion, ST_Simplify, ST_SimplifyPreserveTopology, ST_Boundary, ST_Snap, ST_Polygonize (scalar and aggregate), ST_MakeValid, ST_MinimumClearance, ST_MinimumClearanceLine, ST_Reverse (#229, #230, #231, #239, #234, #295, #298, #299, #286, #328, #312, #314, #316, #288)
* feat(c/sedona-geos): Plumb remaining parameters for ST_Buffer (#241)
* feat(rust/sedona-raster-functions): Add RS_Width, RS_Height, RS_Example (#268, #302, #307)
* feat(rust/sedona-raster-functions): Add affine transformation parameter functions (#311)
* feat(rust/sedona-expr): Use Covers filter for ST_Equals for more GeoParquet pruning (#216)
* feat(rust/sedona-expr): Implement SpatialFilter for overlaps and crosses for GeoParquet pruning (#217)
* feat(rust/sedona-datasource): Implement generic RecordBatchReader-based format (#251)
* feat(python/sedonadb): Implement GDAL/OGR formats via pyogrio (#283)
* feat(python/sedonadb): Implement Python UDFs (#228)
* feat(python/sedonadb): Implement DataFrame.columns (#226)
* feat(r/sedonadb): Add FFI support for ScalarUDF and TableProvider (#214)
* feat(r/sedonadb): Add sd_write_parquet() to R bindings (#210)
* feat(r/sedonadb): Add support for runtime linking of PROJ (#166)
* feat(examples/sedonadb-rust): Add example Rust project (#320)
* feat: Add metrics in DataSourceExec related to spatial predicate pruning (#173)
* feat: Refactor KNN join with new geo-index trait and lock-free shared geometry cache (#169)

### Bug Fixes

* fix(rust/sedona-geoparquet): Don't use ProjectionExec to create GeoParquet 1.1 bounding box columns (#398)
* fix(rust/sedona-expr): Fix GeoParquet pruning when number of final columns is less than the geometry column index (#385)
* fix(rust/sedona-expr): Resolve filter expression bounding box by name and not by index (#384)
* fix(python/sedonadb): Fix GDAL/OGR read on Windows (#371)
* fix(python/sedonadb): Ensure global Parquet options are considered on write (#367)
* fix(python/sedonadb): Fix failing test on MacOS wheel builds (#324)
* fix(rust/sedona-geoparquet): Ensure reading a Parquet file that doesn't exist errors (#366)
* fix(rust/sedona,python/sedonadb): Ensure empty batches are not included in RecordBatchReader output (#207)
* fix: Support projection pushdown for RecordBatchReader provider (#197)
* fix(r/sedonadb): Fix build on Windows for path restriction (#208)
* fix(r/sedonadb): Add -Wno-pedantic to avoid compile error (#181)
* fix(c/sedona-c2geography): Remove outdated feature cxx17 from abseil in vcpkg.json (#235)

### Improvements

* perf: Optimize RasterStructArray::get to avoid repeatedly extracting arrays from struct array (#306)
* perf(rust/sedona-geometry,rust/sedona-functions): Optimize st_has(z/m) using WKBBytesExecutor + Implement new WKBHeader (#171)
* chore: Upgrade Datafusion (v50) and Arrow (v56) dependencies (#237)
* chore: Add sedona-geo-traits-ext and sedona-geo-generic-alg to the workspace (#203, #194, #195)
* docs: Add delta lake documentation (#238)
* docs: Add benchmark results and fix typos (#142)

## SedonaDB 0.1.0

This is the initial release of SedonaDB, a native Rust implementation of Apache Sedona built on Apache DataFusion and Apache Arrow.

### Highlights

* Native Rust spatial query engine built on Apache DataFusion
* GeoParquet read support with spatial predicate pushdown
* Spatial join support (range join, KNN join)
* Core spatial functions (ST_Point, ST_Buffer, ST_Contains, ST_Intersects, ST_Within, ST_DWithin, ST_Distance, and more)
* Python bindings (SedonaContext, DataFrame API)
* R bindings
* S2 geography support
* CRS transformation via PROJ
