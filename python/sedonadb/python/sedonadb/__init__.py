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
import atexit
import os

from sedonadb import _lib
from sedonadb.context import (
    ConfigurationWarning,
    connect,
    configure_proj,
    configure_gdal,
)

__version__ = _lib.sedona_python_version()

__features__ = _lib.sedona_python_features()

__all__ = ["connect", "options", "ConfigurationWarning"]

# Attempt to configure PROJ and GDAL on import. This will warn (with a
# ConfigurationWarning) if PROJ or GDAL can't be configured but should never
# error. The auto-configured values can be overridden as long as
# configure_proj() is called before creating a transform and configure_gdal()
# is called before any GDAL-backed operation (e.g., raster I/O). Setting
# SEDONADB_AUTO_CONFIGURE=false skips this step (e.g., for lower-level
# extensions that don't need PROJ or GDAL or that configure them explicitly).
if os.environ.get("SEDONADB_AUTO_CONFIGURE", "true").strip().lower() not in (
    "false",
    "0",
    "no",
    "off",
):
    configure_proj("auto")
    configure_gdal("auto")

# From interpreter shutdown on, leave GDAL objects (e.g. datasets) unfreed
# instead of closing them. This guards Windows process exit, which unloads GDAL's
# library before the extension's thread-local destructors would close their
# datasets and abort the process; a Python atexit callback is the latest hook
# that reliably runs before that. GDAL itself keeps working, so atexit callbacks
# that run after this one can still use SedonaDB. Registered here so it runs
# before the extension is torn down.
atexit.register(_lib.leak_gdal_objects_on_drop)
