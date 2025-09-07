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
"""Scarf analytics utility functions."""

import os
import platform
import threading
import urllib.request


def make_scarf_call(language: str) -> None:
    """Make a call to Scarf for usage analytics.

    Args:
        language: The language identifier (e.g., 'python', 'adbc', 'dbapi')
    """

    def _scarf_request():
        try:
            # Check for user opt-out
            if (
                os.environ.get("SCARF_NO_ANALYTICS") is not None
                or os.environ.get("DO_NOT_TRACK") is not None
            ):
                return

            # Detect architecture and OS
            arch = platform.machine().lower().replace(" ", "_")
            os_name = platform.system().lower().replace(" ", "_")

            # Construct Scarf URL
            scarf_url = (
                f"https://sedona.gateway.scarf.sh/sedona-db/{arch}/{os_name}/{language}"
            )

            # Make the request in a non-blocking way
            urllib.request.urlopen(scarf_url, timeout=1)
        except Exception:
            # Silently ignore any errors - we don't want Scarf calls to break user code
            pass

    # Run in a separate thread to avoid blocking
    thread = threading.Thread(target=_scarf_request, daemon=True)
    thread.start()
