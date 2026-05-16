# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# List files in here("../../docs/sql/reference/*.qmd") and extract title and params
# from frontmatter. Description section is currently the only section we pull.
# No examples yet. Parameter names for some types like geometry and geography are
# auto generated (e.g., geom, geog with description).
# For multiple kernels, typically the docs per parameter are shared but the types
# may have to be summarised because we only get one @param section
# For SeeAlso, link to the seondadb docs (e.g., https://sedona.apache.org/sedonadb/latest/reference/sql/st_asbinary/)
# record the hash of the file in a comment towards the top and only regenerate if the .qmd file changed
# hard code the default implemenations using the param names
# run air format at the end

# start by writing the script and rendering a few files (rerendering st_length, st_intersects, st_buffer are good candidates to start with)
