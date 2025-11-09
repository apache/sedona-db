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
//! Geometry type re-exports
//!
//! Ported (and contains copied code pattern) from the `geo` crate (geometry module) at commit
//! `5d667f844716a3d0a17aa60bc0a58528cb5808c3`:
//! <https://github.com/georust/geo/tree/5d667f844716a3d0a17aa60bc0a58528cb5808c3/geo/src>.
//! This file is a thin wrapper that publicly re-exports geometry types from `geo-types` to
//! mirror the upstream API surface. Original upstream code is dual-licensed under Apache-2.0 or MIT;
//! used here under Apache-2.0.
//! This module makes all geometry types available
pub use geo_types::geometry::*;
