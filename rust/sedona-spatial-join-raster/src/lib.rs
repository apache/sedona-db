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
//! polygon through the raster's four corners) as a WKB polygon plus a bounding
//! rectangle. The footprint is reprojected into the geometry operand's CRS so the
//! R-tree filter and the WKB refiner — both unchanged from the default planar
//! spatial join — compare footprints and geometries in a common CRS.
//!
//! A cross-CRS raster footprint densifies each of its four edges (~10 interior
//! points) in the raster's own CRS, where the edges are exact straight lines,
//! then reprojects every densified point into the target CRS. The indexed and
//! refined footprint therefore follows the curved image of each edge rather than
//! chording straight across it. Same-CRS joins keep the exact four-corner
//! footprint (no reprojection, no densification). Antimeridian crossings and
//! geodesic edges are not modeled.

mod join_provider;
pub mod physical_planner;
