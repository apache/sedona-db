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


set -eu

main() {
    local -r source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local -r source_rpkg_dir="$(cd "${source_dir}/../" && pwd)"

    # Run the updater
    savvy-cli update "${source_rpkg_dir}"

    # Post-process files
    local -r api_h="${source_rpkg_dir}/src/rust/api.h"
    local -r init_c="${source_rpkg_dir}/src/init.c"
    local -r wrappers_r="${source_rpkg_dir}/R/000-wrappers.R"

    mv "${api_h}" "${api_h}.tmp"
    mv "${init_c}" "${init_c}.tmp"
    mv "${wrappers_r}" "${wrappers_r}.tmp"

    # Add license header to api.h
    echo "${LICENSE_C}" > "${api_h}"
    cat "${api_h}.tmp" >> "${api_h}"

    # Add license header, put includes on their own lines, and fix a typo in init.c
    echo "${LICENSE_C}" > "${init_c}"
    cat "${init_c}.tmp" >> "${init_c}"

    # Add license header to 000-wrappers.R
    echo "${LICENSE_R}" > "${wrappers_r}"
    cat "${wrappers_r}.tmp" >> "${wrappers_r}"

    # Run clang-format on the generated C files
    clang-format -i "${api_h}"
    clang-format -i "${init_c}"

    # Remove .tmp files
    rm "${api_h}.tmp" "${init_c}.tmp" "${wrappers_r}.tmp"
}

LICENSE_R='# Licensed to the Apache Software Foundation (ASF) under one
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
'
LICENSE_C='// Licensed to the Apache Software Foundation (ASF) under one
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
'

main
