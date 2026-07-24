// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Optimized spatial joins with a raster operand.
//!
//! A raster/geometry spatial predicate (`RS_Intersects`, `RS_Contains`,
//! `RS_Within`) is accelerated by evaluating each raster into its footprint (the
//! convex hull of its four corners) as a WKB polygon plus a bounding rectangle.
//! The footprint is reprojected into the geometry operand's CRS so the R-tree
//! filter and the WKB refiner — both unchanged from the default planar spatial
//! join — compare footprints and geometries in a common CRS.
//!
//! Cross-CRS raster footprints are indexed and refined as the convex hull of the
//! raster's four reprojected corners; projection curvature along the edges is not
//! modeled, so for large-extent rasters reprojected between very different CRSs
//! the hull can slightly under-cover the true footprint (rare missed matches at
//! the extreme edges). Same-CRS joins are exact.

mod join_provider;
pub mod physical_planner;
