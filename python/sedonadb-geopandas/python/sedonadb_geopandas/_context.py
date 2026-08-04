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
"""Default SedonaDB context.

GeoPandas has no notion of a connection, so this package lazily creates a single
shared SedonaDB context the first time it is needed. Callers that want an
explicit context can always pass one through the public constructors.
"""

_DEFAULT = None


def default_context():
    """Return the process-wide default SedonaDB context, creating it if needed."""
    global _DEFAULT
    if _DEFAULT is None:
        import sedonadb

        _DEFAULT = sedonadb.connect()
    return _DEFAULT
