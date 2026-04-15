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
from sedonadb import _lib
from sedonadb.context import connect, configure_proj, configure_gdal

__version__ = _lib.sedona_python_version()

__features__ = _lib.sedona_python_features()

__all__ = ["connect", "options"]

# Attempt to configure PROJ and GDAL on import. This will warn if PROJ
# or GDAL can't be configured but should never error. The auto-configured
# values can be overridden as long as configure_proj() is called before
# creating a transform and configure_gdal() is called before any
# GDAL-backed operation (e.g., raster I/O).
configure_proj("auto")
configure_gdal("auto")
