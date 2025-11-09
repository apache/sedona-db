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


class Sedona:
    """Mock sedona Python module

    The Apache Sedaona Python ecosystem is centered around the apache-sedona Python
    package which provides the `sedona` modules. To decouple the maintenance and
    provide fine-grained dependency control for projects that need it, sedonadb
    is distributed as a standalone package. This mock `sedona` module lets us write
    user-facing docstrings that use the documented install/import without requiring
    a circular dependency on apache-sedona for testing.
    """

    @property
    def db(self):
        import sedonadb

        return sedonadb

    @property
    def testing(self):
        import sedonadb.testing

        return sedonadb.testing

    @property
    def dbapi(self):
        import sedonadb.dbapi

        return sedonadb.dbapi


sedona = Sedona()
