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

import io
from pathlib import Path
import yaml

from _render_meta import render_meta


def render_listing(paths):
    """Lists meta information for zero or more filenames to function .qmd files

    Output:

    ```
    ## [ST_Name](st_name.md)

    <description>

    ## Usage

    return_type ST_Name(arg_name: arg_type)
    ```
    """
    for file in paths:
        raw_meta = read_frontmatter(file)
        link_href = file.name.replace(".qmd", ".md")
        print(f"\n## [{raw_meta['title']}]({link_href})\n\n")

        render_meta(raw_meta, level=2, arguments=False)


def read_frontmatter(path):
    frontmatter = io.StringIO()
    with open(path) as f:
        for line in f:
            if line.strip() == "---":
                break
        for line in f:
            if line.strip() == "---":
                break
            frontmatter.write(line)
    frontmatter.seek(0)
    return yaml.safe_load(frontmatter)


def collect_files(file_glob):
    return list(sorted(Path(__file__).parent.rglob(file_glob)))


if __name__ == "__main__":
    import argparse
    import sys
    import yaml

    parser = argparse.ArgumentParser(description="Render SedonaDB SQL function listing")
    parser.add_argument(
        "files",
        nargs="+",
        help="Files or globs of files whose frontmatter should be included in the listing",
    )
    args = parser.parse_args(sys.argv[1:])

    in_files: list[Path] = []
    for file_or_glob in args.files:
        in_files.extend(collect_files(file_or_glob))

    render_listing(in_files)
