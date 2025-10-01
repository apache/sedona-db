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
// Geometry type tags for dispatching algorithm traits to the corresponding implementation

pub trait GeoTypeTag {}

pub struct CoordTag;
pub struct PointTag;
pub struct LineStringTag;
pub struct PolygonTag;
pub struct MultiPointTag;
pub struct MultiLineStringTag;
pub struct MultiPolygonTag;
pub struct GeometryCollectionTag;
pub struct GeometryTag;
pub struct LineTag;
pub struct RectTag;
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

pub trait GeoTraitExtWithTypeTag {
    type Tag: GeoTypeTag;
}
