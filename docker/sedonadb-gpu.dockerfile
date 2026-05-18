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

FROM nvidia/cuda:13.1.2-devel-ubuntu24.04

# Install system dependencies (initial set; clang added from LLVM apt repo below)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    pkg-config \
    curl \
    python3 \
    python3-dev \
    python3-pip \
    gnupg \
    lsb-release \
    ca-certificates \
    wget \
    zip \
    unzip \
    tar \
    clang-18 \
    clang++-18 \
    libc++-18-dev \
    libc++abi-18-dev \
    lld-18 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Add Kitware APT repository and install the latest CMake (Updated for Ubuntu 24.04)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
 && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
 && apt-get update \
 && apt-get install -y cmake \
 && rm -rf /var/lib/apt/lists/*

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git /home/ubuntu/vcpkg
WORKDIR /home/ubuntu/vcpkg
RUN ./bootstrap-vcpkg.sh
RUN ./vcpkg install geos

# Copy your project
COPY . /workspace
WORKDIR /workspace

# Set environment variables for build
ENV VCPKG_ROOT=/home/ubuntu/vcpkg
ENV CMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
ENV PKG_CONFIG_PATH=$VCPKG_ROOT/installed/x64-linux/lib/pkgconfig
ENV CC=clang-18
ENV CXX=clang++-18
ENV CMAKE_CUDA_ARCHITECTURES="86"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pass features to maturin when pip builds the package via PEP517
ENV MATURIN_PEP517_ARGS="--features s2geography,gpu"

# Install maturin AND JupyterLab inside the virtual environment
RUN pip3 install maturin jupyterlab pyproj

# Install the project in editable mode (uses pyproject -> maturin under the hood)
RUN pip3 install -e "python/sedonadb" -vv

# Clean up the library path so the container uses the real driver at runtime
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Expose the default JupyterLab port
EXPOSE 8888

# Launch JupyterLab by default when the container starts
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
