<!---
  Licensed to the Apache Software Foundation (ASF) under one
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
  under the License.
-->

# libgpuspatial - A GPU-accelerated Geospatial Processing Library

***libgpuspatial*** currently supports the joining of large geospatial datasets using GPU acceleration.
It takes two inputs called "Build" and "Stream" from two ArrowArrays containing geometries in WKB format,
where "Build" is a smaller dataset that can be fit into the device memory and is built into an index,
and "Stream" can be a continuously incoming dataset that is streamed to find matches with the help of the index.
Currently, it supports the following geometries and join types:

Geometries:
- Point
- LineString
- Polygon
- MultiPoint
- MultiLineString
- MultiPolygon

For a given ArrowArray, a geometry type and its multiple variant can be co-existed in the same array. GeometryCollection has not been implemented yet.

***libgpuspatial*** supports the following spatial join types by computing DE-9IM (Dimensionally Extended Nine-Intersection Model) relations:

Spatial Join Types:
- Equals
- Disjoint
- Touches
- Contains
- Covers
- Intersects
- Within
- CoveredBy

## 1. Install dependencies

External dependencies:

- CUDA >= 12.0, Assuming you have CUDA installed.

- CMake >= 3.30.4

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.31.8/cmake-3.31.8-linux-x86_64.sh
bash cmake-3.31.8-linux-x86_64.sh --prefix=$HOME/.local --exclude-subdir --skip-license
```

- Arrow >= 20.0 (Optional, only needed if you want to build benchmarks)

```bash
wget "https://github.com/apache/arrow/releases/download/apache-arrow-20.0.0/apache-arrow-20.0.0.tar.gz"
sudo apt install libcurl4-openssl-dev libzstd-dev # dependencies for S3 support and NanoArrow
tar zxvf apache-arrow-20.0.0.tar.gz
cd apache-arrow-20.0.0/cpp
mkdir build && cd build
INSTALL_PATH=$HOME/.local
cmake -DARROW_S3=ON \
	  -DARROW_PARQUET=ON \
	  -DARROW_IPC=ON \
	  -DARROW_FILESYSTEM=ON \
	  -DARROW_WITH_SNAPPY=ON \
	  -DCMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
	  -DCMAKE_BUILD_TYPE=Release \
	  ..
make -j$(nproc)
make install
```



## 2. Build and install libgpuspatial

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=$HOME/.local \
      -DGPUSPATIAL_BUILD_TESTS=ON \
      -DGPUSPATIAL_BUILD_BENCHMARK=ON \
      ..
```

```cmake
# User's CMakeLists.txt

find_package(gpuspatial REQUIRED)

add_executable(my_app main.cpp)

# Link to gpuspatial
target_link_libraries(my_app PRIVATE gpuspatial::gpuspatial)

# Pass the shader path to the C++/CUDA code
target_compile_definitions(my_app PRIVATE
    GPUSPATIAL_SHADER_PATH="${gpuspatial_SHADER_DIR}"
)
```

## 3. Run benchmarks


```bash
aws configure sso --use-device-code
export AWS_DEFAULT_REGION=us-west-2
```

```bash
./build/benchmark -build_file wherobots-benchmark-prod/data/3rdparty-bench/postal-codes-sorted \
                  -stream_file wherobots-benchmark-prod/data/3rdparty-bench/osm-nodes-large-sorted-corrected \
                  -execution geos \
                  -limit 5
```
