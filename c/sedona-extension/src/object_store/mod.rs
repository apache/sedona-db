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

//! Read-only ObjectStore transport for independently built format extensions.
//!
//! Single-result C calls run on a bounded foreign executor. GET bodies and
//! listings use Arrow's asynchronous device stream protocol. Consumer deadlines
//! use a separate timer runtime, including when the caller has no Tokio runtime.
//! See `README.md` in this directory for the versioned wire and ownership contract.

mod bounded;
mod export;
pub mod ffi;
mod import;
mod wire;

pub use bounded::{ObjectStoreExecutor, ObjectStoreOptions};
pub use export::ExportedObjectStore;
pub use import::ImportedObjectStore;
