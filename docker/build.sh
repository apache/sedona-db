#!/bin/bash

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

# Build (and optionally publish) the SedonaDB GPU image.
#
# Usage:
#   docker/build.sh [BUILD_MODE] [IMAGE_TAG] [CMAKE_CUDA_ARCHITECTURES]
#
#   BUILD_MODE                 "local" (default) or "release".
#                              - local:   single-platform build loaded into the
#                                         local Docker daemon (no push).
#                              - release: multi-platform (amd64 + arm64) build
#                                         pushed straight to the registry.
#   IMAGE_TAG                  Full image reference. Default: apache/sedona:sedonadb-latest
#   CMAKE_CUDA_ARCHITECTURES   GPU compute capability passed to the build.
#                              Default: 86 (Ampere). Override for other GPUs,
#                              e.g. "90" (Hopper) or "75;80;86;90" for a fat binary.
#
# Examples:
#   # Local build for the current machine's architecture
#   docker/build.sh
#
#   # Multi-arch build + push to Docker Hub (run `docker login` first)
#   docker/build.sh release apache/sedona:sedonadb-latest
#
#   # Build for a Hopper GPU
#   docker/build.sh local apache/sedona:sedonadb-latest 90

set -euo pipefail

BUILD_MODE=${1:-local}
IMAGE_TAG=${2:-apache/sedona:sedonadb-latest}
CMAKE_CUDA_ARCHITECTURES=${3:-86}

# Run from the repository root so the build context (`.`) and the
# `docker/sedonadb-gpu.dockerfile` path resolve correctly regardless of the
# directory this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DOCKERFILE="docker/sedonadb-gpu.dockerfile"

echo "Build mode:   ${BUILD_MODE}"
echo "Image tag:    ${IMAGE_TAG}"
echo "CUDA arch:    ${CMAKE_CUDA_ARCHITECTURES}"
echo "Dockerfile:   ${DOCKERFILE}"
echo "Context:      ${REPO_ROOT}"

if [ "${BUILD_MODE}" = "local" ]; then
    # Single-platform build for the local environment, loaded into the daemon.
    docker buildx build --load \
        --build-arg CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -f "${DOCKERFILE}" \
        -t "${IMAGE_TAG}" .
elif [ "${BUILD_MODE}" = "release" ]; then
    # Cross-platform build pushed directly to the registry.
    # Requires a prior `docker login` and a buildx builder that supports
    # multi-platform output (e.g. `docker buildx create --use`).
    docker buildx build --platform linux/amd64,linux/arm64 \
        --progress=plain \
        --no-cache \
        --output type=registry \
        --build-arg CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
        -f "${DOCKERFILE}" \
        -t "${IMAGE_TAG}" .
else
    echo "Unknown BUILD_MODE '${BUILD_MODE}'. Use 'local' or 'release'." >&2
    exit 1
fi
