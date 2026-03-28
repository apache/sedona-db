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

# =============================================================================
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# This function finds rmm and sets any additional necessary environment variables.
function(find_and_configure_rmm)
  include(${rapids-cmake-dir}/cpm/rmm.cmake)

  # Find or install RMM
  rapids_cpm_rmm(BUILD_EXPORT_SET gpuspatial-exports INSTALL_EXPORT_SET
                 gpuspatial-exports)

  # --- INJECT RMM HOTFIX ---
  # We only want to apply the patch if CPM actually fetched RMM from source.
  # If rmm_SOURCE_DIR is defined, it means it was downloaded into the build tree.
  message("rmm_SOURCE_DIR: ${rmm_SOURCE_DIR}")
  if(DEFINED rmm_SOURCE_DIR)
    set(RMM_PATCH_TARGET_DIR "${rmm_SOURCE_DIR}/cpp/include/rmm/mr")
    set(RMM_PATCH_MARKER "${rmm_SOURCE_DIR}/.rmm_race_condition_patched")

    # If the target directory exists and hasn't been patched yet
    if(EXISTS "${RMM_PATCH_TARGET_DIR}" AND NOT EXISTS "${RMM_PATCH_MARKER}")
      message(STATUS "RAPIDS-CMAKE BYPASS: Applying custom race-condition patch to RMM..."
      )

      execute_process(# Assuming the patch file is located in the root 'cmake/patches/' directory
                      COMMAND patch -p0 -i
                              "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/rmm_race_condition.patch"
                      WORKING_DIRECTORY "${RMM_PATCH_TARGET_DIR}"
                      RESULT_VARIABLE PATCH_RESULT)

      if(PATCH_RESULT EQUAL 0)
        file(TOUCH "${RMM_PATCH_MARKER}")
        message(STATUS "Successfully patched RMM tracking adaptor.")
      else()
        message(FATAL_ERROR "Failed to apply RMM patch. Check the patch file formatting.")
      endif()
    endif()
  endif()
  # -------------------------

endfunction()

find_and_configure_rmm()
