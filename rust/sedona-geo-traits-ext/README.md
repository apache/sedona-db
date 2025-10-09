<!-- Licensed to the Apache Software Foundation (ASF) under one
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
under the License. -->

# Geo-Traits Extended

This crate extends the `geo-traits` crate with additional traits and
implementations. The goal is to provide a set of traits that are useful for
implementing algorithms in `geo-generic-alg` crate. Most of the methods are
inspired by the `geo-types` crate, but are implemented as traits for the
`geo-traits` types. Some methods returns concrete types defined in `geo-types`,
these methods are only for computing tiny, intermediate results during
algorithm execution. By adding methods in `geo-types` to `geo-traits-ext`,
we can port algorithms in `geo` crate for concrete `geo-types` types to generic
`geo-traits-ext` types more easily.

`geo-traits-ext` traits also has an associated `Tag` type to workaround the
single orphan rule in Rust. For instance, we cannot write blanket `AreaTrait`
implementations for both `LineStringTrait` and `PolygonTrait` because we
cannot show that there would be no type implementing both `LineStringTrait` and
`PolygonTrait`. By adding an associated `Tag` type, we can write blanket
implementations for `AreaTrait<LineStringTag>` and `AreaTrait<PolygonTag>`, since
`AreaTrait<LineStringTag>` and `AreaTrait<PolygonTag>` are different types.
Please refer to the source code of `sedona-geo-generic-alg` for more details.
