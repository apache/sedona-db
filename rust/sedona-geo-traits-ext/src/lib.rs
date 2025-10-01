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
//! Extended traits for the `geo-traits` crate
//!
//! This crate extends the `geo-traits` crate with additional traits and
//! implementations. The goal is to provide a set of traits that are useful for
//! implementing algorithms on top of the `geo` crate. Most of the methods are
//! inspired by the `geo-types` crate, but are implemented as traits on the
//! `geo-traits` types. Some methods returns concrete types defined in `geo-types`,
//! these methods are only for computing tiny, intermediate results during
//! algorithm execution.
//!
//! The crate is designed to support migration of the `geo` crate to use the
//! traits defined in `geo-traits` by providing generic implementations of the
//! geospatial algorithms, rather than implementing algorithms on concrete types
//! defined in `geo-types`.
//!
//! The crate is currently under active development and the API is subject to
//! change.

pub use coord::CoordTraitExt;
pub use geometry::{GeometryTraitExt, GeometryTypeExt};
pub use geometry_collection::GeometryCollectionTraitExt;
pub use line::LineTraitExt;
pub use line_string::LineStringTraitExt;
pub use multi_line_string::MultiLineStringTraitExt;
pub use multi_point::MultiPointTraitExt;
pub use multi_polygon::MultiPolygonTraitExt;
pub use point::PointTraitExt;
pub use polygon::PolygonTraitExt;
pub use rect::RectTraitExt;
pub use triangle::TriangleTraitExt;

mod coord;
mod geometry;
mod geometry_collection;
mod line;
mod line_string;
mod multi_line_string;
mod multi_point;
mod multi_polygon;
mod point;
mod polygon;
mod rect;
mod triangle;

pub use type_tag::*;
mod type_tag;

pub mod wkb_ext;
