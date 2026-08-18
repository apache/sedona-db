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

## SedonaDB 0.4.1

### Highlights

* Experimental GeoPandas-compatible API package (sedonadb-geopandas)
* GDAL-backed raster processing functions (RS_Clip, RS_Resample, RS_Tile, RS_ZonalStats, RS_AsRaster, RS_Polygonize, RS_AsGeoTiff, RS_FromGDALRaster, RS_ReprojectMatch)
* Raster-vector spatial join (RS_Intersects, RS_Contains, RS_Within)
* Expanded Python DataFrame API (mutate, rename, unnest, union, intersect, except_, CSV and JSON read/write)
* New spatial functions: ST_BuildArea, ST_DelaunayTriangles, ST_ExteriorRing, ST_PointOnSurface, ST_HausdorffDistance, ST_LineLocatePoint, ST_MaxDistance, ST_ReducePrecision, ST_ConvexHull_Agg
* FFI table provider with filter pushdown across the FFI boundary
* Free-threaded Python support
* Point-coordinate fast paths for ST_Distance, ST_DWithin, ST_Azimuth, ST_X, ST_Y
* Memory limit and spatial join robustness fixes

### New Features

* feat(python/sedonadb-geopandas): GeoPandas-compatible API package (experimental) (#1052)
* feat(rust/sedona-raster-gdal): Add RS_Clip, RS_Resample, RS_ZonalStats/RS_ZonalStatsAll, RS_AsGeoTiff, RS_Polygonize, RS_AsRaster, RS_FromGDALRaster (#1000, #1064, #1066, #1028, #955, #956, #1030)
* feat: Add RS_Tile raster tiling function (#1065)
* feat: Add RS_ReprojectMatch (#1068)
* feat(rust/sedona-raster-functions): Add RS_Value and RS_Values for sampling raster values (#974, #1027)
* feat(rust/sedona-raster-functions): Add RS_SetGeoReference and RS_SetBandNoDataValue (#1003)
* feat(rust/sedona-raster): Persist and read non-identity band views (#1113)
* feat(rust/sedona-spatial-join-raster): Raster-vector spatial join (RS_Intersects/Contains/Within) (#1073)
* feat(rust/sedona-raster-zarr): Read CF spatial_ref GeoTransform georeferencing and derive transform from spatial:bbox (#975, #1009)
* feat: Implement ST_BuildArea, ST_DelaunayTriangles, ST_ExteriorRing, ST_PointOnSurface, ST_NumInteriorRing alias (#990)
* feat(c/sedona-geos): Add ST_HausdorffDistance (#1039)
* feat(c/sedona-geos): Add geometry support for ST_LineLocatePoint, ST_MaxDistance, ST_ReducePrecision (#994)
* feat(rust/sedona-geo): Add ST_ConvexHull_Agg (#1043)
* feat(rust/sedona-functions): Implement typed geometry constructors (ST_PointFromText and friends) (#993)
* feat(rust/sedona-functions): Add ST_NumPoints alias for ST_NPoints (#989)
* feat(c/sedona-s2geography): Parameterize S2_CoveringCellIds (min_level, max_level, max_cells) (#1023)
* feat(python/sedonadb): Add DataFrame.mutate and rename (#1010)
* feat(python/sedonadb): Add DataFrame.unnest (#1050)
* feat(python/sedonadb): Add DataFrame.union, union_distinct, intersect, intersect_distinct, except_ (#965, #1006)
* feat(python/sedonadb): Add DataFrame.to_csv and to_json (#1044)
* feat(python/sedonadb): Add read_csv and read_json (#1034)
* feat(python/sedonadb): Add a native scalar UDF import path for plugins (#1146)
* feat(python/sedonadb): Raster.from_numpy(bbox=, registration=) and zero-copy raster IO to/from numpy (#999, #1119)
* feat(c/sedona-extension): Add FFI table provider, exec plan, and synchronous stream exchange (#1004)
* feat(c/sedona-extension): Push Expr filters into the TableProvider across the FFI boundary (#1094)
* feat(sedona-datasource): Resolve directory-shaped formats as URL tables (#1125)
* feat(rust/sedona): Auto-register common OGR formats so file URLs work as tables (#1124)
* feat(rust/sedona-schema): SedonaType::UnrecognizedExtension for user-defined types (#1129)
* feat(r/sedonadb): Add SQL translation for nested field (#1171)
* feat(python/sedonadb): Upgrade PyO3 and enable free-threaded Python support (#1012, #1017)

### Bug Fixes

* fix(rust/sedona): Disable the default memory limit to avoid excessive spilling (#1155)
* fix(rust/sedona-spatial-join): Fix geography scalar functions failing over spatial join output (#1153)
* fix(rust/sedona-spatial-join): Handle empty left join build side (#1167)
* fix(c/sedona-s2geography): Make geography ring orientation less surprising for rings with crossing edges (#1143)
* fix(rust): Return zero distance for crossing linestrings (#1164)
* fix(rust/sedona-geoparquet): Handle intermediary columns in projection expressions in GeoParquet reader (#1116)
* fix(python/sedonadb): Share one Tokio runtime across contexts (#1128)
* fix(python/sedonadb): Free-threading-safe Tokio runtime teardown; re-enable cp314t macOS wheels (#1067)
* fix(python/sedonadb): Serialize pyogrio open/close (#1051)
* fix(sedona-gdal): Skip GDALClose during interpreter shutdown (#1135)
* fix(sedona-proj): Add missing PJ_INFO fields to match PROJ's ABI (#1127)
* fix(c/sedona-tg): Fix polygon hole containment and collection point containment boundary predicates (#1022, #1042)
* fix(rust/sedona-functions): Fix scalar iteration with num_iterations > 1 (#1041)
* fix(rust/sedona-raster-functions): Exact ESRI/NODE georeference for skewed rasters (#1049)
* fix(rust/sedona): Accept aws.allow_http and aws.session_token; reject dead AWS options (#1013)
* fix(python/sedonadb-geopandas): Add missing geoarrow-pyarrow dependency (#1134)
* fix(r/sedonadb): Fix build on Windows ARM64 (#1137)

### Improvements

* perf(rust): Point-coordinate fast paths for ST_Distance/ST_DWithin/ST_Azimuth (#1008)
* perf(rust/sedona-functions): Fast-path ST_X/ST_Y Point coordinates via read_point_xy (#1005)
* perf(rust/sedona-spatial-join-raster): Pin raster operand to the probe side (#1074)
* refactor(rust/sedona-common): Consolidate CrsEngine and Bounder into SedonaOptions (#1001)
* refactor(rust): Route ST_Transform and sd_order through the injected CRS engine (#1102)
* docs: Add Coordinate Reference Systems concept page (#1126)
* docs: Add missing geography kernels to SQL function reference pages (#1141)
* docs(r/sdplyr): Add README for the sdplyr package (#1147)
* ci(python/sedonadb-geopandas): Run tests, build wheels, and verify in releases (#1095)

## SedonaDB 0.4.0

### Highlights

* Packaging for conda-forge
* Python DataFrame API (select, filter, join, group_by, sort, and composable expressions)
* R dplyr interface (sdplyr)
* Expanded geography support (structural functions, accessors, envelopes, spatial join, GeoParquet pruning)
* GPU-accelerated spatial join via libgpuspatial integration
* GeoParquet 2.0/Parquet-native geometry and geography write support with partitioned datasets
* N-dimensional raster type with GDAL and Zarr support, lazy loading, and cloud storage backends
* New raster functions: RS_FromPath, RS_MetaData, RS_Contains, RS_Intersects, RS_Within, RS_IsEmpty
* New spatial functions: ST_Relate, ST_Normalize, ST_LineSubstring, ST_Segmentize, ST_TessellateGeom, ST_TessellateGeog

### New Features

* feat(python/sedonadb): Add Expr foundation, operator overloads, and context-aware piped expressions (#807, #823, #901)
* feat(python/sedonadb): Add DataFrame.select, filter/where, __getitem__, sort, drop, agg, group_by().agg(), join, cross_join, distinct/distinct_on (#832, #835, #846, #852, #859, #871, #887, #893, #908, #925, #961)
* feat(python/sedonadb): Add Python aggregate UDF decorator (#937)
* feat(python/sedonadb): Expose scalar and aggregate UDFs from context registry (#885)
* feat(python/sedonadb): Expose simplified Arrow stream export (#873)
* feat(python/sedonadb): Add Python GDAL configuration API (#689)
* feat(python/sedonadb): Support layer names and archive sub-paths for pyogrio sources (#778)
* feat(python/sedonadb): Enable Zarr read via sedonadb-zarr (#916)
* feat(rust/sedona,python/sedonadb): Support nested expressions in Python and SQL (#973)
* feat(r/sdplyr): Add sdplyr package (dplyr method implementations) with filter translation (#931, #972)
* feat(r/sedonafns): Add generated Sedona documentation as an R package (#851)
* feat(r/sedonadb): Add spatial join syntax, join type helpers, and join expression evaluation (#781, #814)
* feat(rust/sedona-functions,c/sedona-geos): Implement geography kernels for structural transformations and accessors (#844)
* feat(c/sedona-s2geography): Add ST_(X|Y)(Min|Max), ST_Envelope, ST_Envelope_Agg, and ST_Analyze_Agg implementations for geography (#850)
* feat(c/sedona-s2geography,rust/sedona-functions): Add ST_Segmentize, ST_TessellateGeom, ST_TessellateGeog (#867)
* feat(rust/sedona-functions,c/sedona-proj): Support geography in CRS/SRID functions (#848)
* feat(rust/sedona-functions): Add geography and CRS propagation to ST_Dump (#847)
* feat(rust/sedona-expr): Add pruning capability for geography type (#806)
* feat(rust/sedona-spatial-join-geography): Implement spatial join for geography type (#775)
* feat(rust/sedona-spatial-join-gpu): Integrate libgpuspatial into sedona-spatial-join (#722)
* feat(python/sedonadb): Enable GPU feature in Python package and add spatial join tests (#768)
* feat(docker): Publish multi-arch GPU image with multi-CUDA-arch support (#872, #909)
* feat(c/sedona-libgpuspatial): Interface, robustness, synchronization, and RMM upgrades (#717, #718, #719, #721, #767)
* feat(rust/sedona-raster): N-dimensional raster type extension and dimension query/manipulation functions (#749, #750)
* feat(c/sedona-gdal): Add crate with dynamically loaded GDAL bindings and wrapper utilities (#681, #695, #696, #697, #698, #699)
* feat(rust/sedona-raster-gdal): Add GDAL foundation library, in-db raster loading, RS_FromPath, RS_MetaData (#787, #811, #812, #831, #833)
* feat(rust): Lazy raster loading support for Zarr and GDAL (#886)
* feat(rust/sedona-raster): Zero-copy band data in RS_EnsureLoaded and zero-copy raster access from pyarrow arrays (#917, #942)
* feat(rust/sedona-raster-functions): Add RS_Contains, RS_Intersects, RS_Within, RS_IsEmpty (#615, #944)
* feat(rust/sedona-raster-gdal): Pass N-D rasters through the GDAL bridge via plane stacking (#928)
* feat(rust/sedona-raster-zarr,python/sedonadb-zarr): Add sedona-raster-zarr crate and sedona-zarr plugin (#858)
* feat(rust/sedona-raster-zarr): Cloud storage backends (S3, GCS, Azure, HTTP) via object_store (#888)
* feat(rust/sedona-raster-zarr): Georeferencing from coordinate arrays, CF/rioxarray CRS conventions, zlib codec (#954, #985, #987)
* feat(raster): Accept lat/lon and latitude/longitude spatial dimension names (#910)
* feat(rust/sedona-query-planner): Skip RS_EnsureLoaded on args returning loaded bytes (#979)
* feat(rust/sedona-geoparquet): Add GeoParquet 2.0/Parquet-native geometry and geography support to Parquet writer (#805)
* feat(rust/sedona-geoparquet,rust/sedona-datasource): Add support for partition columns and discovery (#906)
* feat(c/sedona-geos): Add ST_Relate implementation and boolean variant (#691, #741)
* feat(c/sedona-geos): Add ST_Normalize (#802)
* feat(rust/sedona-functions): Add ST_LineSubstring (#777)
* feat(rust/sedona-schema): Support WKT1/WKT2 CRS strings in deserialize_crs (#953)
* feat(rust/sedona-spatial-join): Make SpatialIndex a trait; configurable SpatialIndexBuilder and EvaluatedGeometryArray (#645, #737)
* feat(rust/sedona-spatial-join): Add config to disable spatial join reordering (#733)

### Bug Fixes

* fix(rust/sedona-query-planner): RS_EnsureLoaded idempotency, metadata preservation, and column-name preservation (#969, #976, #978)
* fix(rust): Clean up CRS string handling (preserve definitions, consolidate equality/SRID) (#962)
* fix(c/sedona-gdal): Resolve MEMDataset::Create on GDAL 3.13 and use MEMCreate C-API when available (#963)
* fix(python/sedonadb): Ensure PROJ and GDAL are detected on Windows + Conda (#980)
* fix(rust/sedona-pointcloud): Fix projection regression (#825)
* fix(rust/sedona-geo): Support array distances in ST_Buffer (#881)
* fix(rust/sedona-spatial-join): Improve evaluated batch memory accounting, implement EvaluatedGeometryArray::concat, fix customized join provider (#766, #784, #884)
* fix(rust/sedona-geoparquet): Cache metadata in inner ParquetOpener (#843)
* fix(rust/sedona-raster): Classify 0-element bands as InDb; make is_indb required (#929)
* fix(rust/sedona-raster-zarr): Discover child arrays via consolidated metadata; parse spatial:transform as affine order (#943, #950)
* fix(c/sedona-proj): Allow bound parameter for CRS argument of ST_Transform (#904)

### Improvements

* perf(rust/sedona-functions): Improve performance of ST_Reverse() (#912)
* perf(rust/sedona-spatial-join): Use row count first to decide join order (#725)
* refactor(rust/sedona-query-planner): Move query planner and utilities to dedicated crate (#735)
* refactor(c/sedona-s2geography): Move s2geography UDFs to extension ABI (#683)
* refactor(python/sedonadb): Refactor registration of extension components (#940)
* feat(rust/sedona-testing): Add ergonomic raster function test harness (#945)
* chore: Ensure workspace can be built and verified under GDAL 3.13 (#984)
* docs: Add GPU acceleration guide for spatial joins (#774)
* docs(examples): Add "Working with Zarr and NDArray data in SedonaDB" tutorial (#938)
* docs: Add conda installation method to docs (#786)
* docs: Add release notes for SedonaDB 0.1.0, 0.2.0, and 0.3.0 (#771)

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
