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

import sedonadb
import difflib


with open("docs/reference/sql.md", "r") as f:
    lines = f.readlines()
    # Headers with `##` are the function names.
    st_funs_in_doc = [line[3:-1] for line in lines if line.startswith("## ")]


### Check if all the functions are documented

sd = sedonadb.connect()
df = sd.sql(r"""
SELECT DISTINCT routine_name
FROM information_schema.routines
WHERE routine_type = 'FUNCTION' AND regexp_like(routine_name, '^(st_|rs_)')
ORDER BY routine_name
""").to_pandas()
st_funs_in_impl_set = set(df["routine_name"].tolist())

st_funs_in_doc_set = set(f.lower() for f in st_funs_in_doc)

funs_only_in_impl = sorted(st_funs_in_impl_set - st_funs_in_doc_set)
funs_only_in_doc = sorted(st_funs_in_doc_set - st_funs_in_impl_set)

if funs_only_in_impl or funs_only_in_doc:
    print("\nFunctions only in implementation:\n  - ", end="")
    print("\n  - ".join(funs_only_in_impl))
    print("\nFunctions only in document:\n  - ", end="")
    print("\n  - ".join(funs_only_in_doc))
    print("\n")

    raise RuntimeError(
        "There are some mismatch between the SQL reference and the actual implementation!"
    )


### Check if the function order is sorted

if st_funs_in_doc != sorted(st_funs_in_doc):
    diff = difflib.unified_diff(
        st_funs_in_doc, sorted(st_funs_in_doc), fromfile="current", tofile="sorted"
    )

    print("\n".join(diff))

    raise RuntimeError("The SQL functions are not sorted in alphabetical order")
