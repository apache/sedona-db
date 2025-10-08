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
/// A well-known binary blob of MULTIPOINT (EMPTY)
///
/// The wkt crate's parser rejects this; however, it's a corner case that may show
/// up in WKB generated externally.
pub const MULTIPOINT_WITH_EMPTY_CHILD_WKB: [u8; 30] = [
    0x01, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
];

/// A well-known binary blob of MULTIPOINT ((1 2 3)) where outer dimension is specified for xy
/// while inner point's dimension is actually xyz
pub const MULTIPOINT_WITH_INFERRED_Z_DIMENSION_WKB: [u8; 38] = [
    0x01, // byte-order
    0x04, 0x00, 0x00, 0x00, // multipoint with xy-dimension specified
    0x01, 0x00, 0x00, 0x00, // 1 point
    // nested point geom
    0x01, // byte-order
    0xe9, 0x03, 0x00, 0x00, // point with xyz-dimension specified
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x-coordinate of point
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y-coordinate of point
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // z-coordinate of point
];
