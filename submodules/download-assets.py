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

import json
import urllib.request
import shutil

from pathlib import Path

HERE = Path(__file__).parent


def download_files_lazy(include):
    with open(HERE / "geoarrow-data" / "manifest.json") as f:
        manifest = json.load(f)

    for group in manifest["groups"]:
        group_name = group["name"]
        for file in group["files"]:
            url = file["url"]
            filename = Path(url).name
            local_path = HERE / "geoarrow-data" / group_name / "files" / filename
            if not path_match(local_path, include):
                continue

            if local_path.exists():
                print(f"Using cached '{filename}'")
            elif file["format"] in ("parquet", "geoparquet"):
                # Only download Parquet/GeoParquet versions of asset files to save space
                print(f"Downloading {url}")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(url) as fin, open(local_path, "wb") as fout:
                    shutil.copyfileobj(fin, fout)

    print("Done!")


def path_match(path, include):
    for pattern in include:
        if path.match(pattern):
            return True

    return False


if __name__ == "__main__":
    import sys

    include_patterns = sys.argv[1:]

    if not include_patterns:
        include_patterns = ["*"]

    download_files_lazy(include_patterns)
