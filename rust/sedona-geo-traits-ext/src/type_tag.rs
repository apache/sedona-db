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
//! Geometry type tags for dispatching algorithm traits to the corresponding implementation

/// Marker trait implemented by all geometry type tags used for dispatch.
pub trait GeoTypeTag {}

/// Tag that identifies coordinate-like values.
pub struct CoordTag;
/// Tag that identifies point-like geometries.
pub struct PointTag;
/// Tag that identifies line-string-like geometries.
pub struct LineStringTag;
/// Tag that identifies polygon-like geometries.
pub struct PolygonTag;
/// Tag that identifies multi-point-like geometries.
pub struct MultiPointTag;
/// Tag that identifies multi-line-string-like geometries.
pub struct MultiLineStringTag;
/// Tag that identifies multi-polygon-like geometries.
pub struct MultiPolygonTag;
/// Tag that identifies geometry-collection-like geometries.
pub struct GeometryCollectionTag;
/// Tag that identifies generic geometry values.
pub struct GeometryTag;
/// Tag that identifies line-segment-like geometries.
pub struct LineTag;
/// Tag that identifies rectangle-like geometries.
pub struct RectTag;
/// Tag that identifies triangle-like geometries.
pub struct TriangleTag;

impl GeoTypeTag for CoordTag {}
impl GeoTypeTag for PointTag {}
impl GeoTypeTag for LineStringTag {}
impl GeoTypeTag for PolygonTag {}
impl GeoTypeTag for MultiPointTag {}
impl GeoTypeTag for MultiLineStringTag {}
impl GeoTypeTag for MultiPolygonTag {}
impl GeoTypeTag for GeometryCollectionTag {}
impl GeoTypeTag for GeometryTag {}
impl GeoTypeTag for LineTag {}
impl GeoTypeTag for RectTag {}
impl GeoTypeTag for TriangleTag {}

/// Helper trait implemented by extension traits to expose their geometry tag.
/// Each geometry type could only implement this trait once, so each geometry type
/// has one unique tag. This helps us work around the single-orphan rule of Rust
/// trait system and help us smoothly refactor the existing algorithms in georust/geo.
pub trait GeoTraitExtWithTypeTag {
    type Tag: GeoTypeTag;
}
