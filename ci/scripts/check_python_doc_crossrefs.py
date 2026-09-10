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

"""Check that representative Python API cross-references rendered as links."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

EXPECTED_LINKS = {
    ("SedonaContext", "#sedonadb.context.SedonaContext"),
    ("connect()", "#sedonadb.context.connect"),
    (
        "create_data_frame()",
        "#sedonadb.context.SedonaContext.create_data_frame",
    ),
    ("read_parquet()", "#sedonadb.context.SedonaContext.read_parquet"),
    ("read_pyogrio()", "#sedonadb.context.SedonaContext.read_pyogrio"),
    ("sql()", "#sedonadb.context.SedonaContext.sql"),
    ("select()", "#sedonadb.dataframe.DataFrame.select"),
    ("filter()", "#sedonadb.dataframe.DataFrame.filter"),
    ("sort()", "#sedonadb.dataframe.DataFrame.sort"),
    ("limit()", "#sedonadb.dataframe.DataFrame.limit"),
    ("to_view()", "#sedonadb.dataframe.DataFrame.to_view"),
}


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: set[tuple[str, str]] = set()
        self.visible_text: list[str] = []
        self._href: str | None = None
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._link_text = []

    def handle_data(self, data: str) -> None:
        self.visible_text.append(data)
        if self._href is not None:
            self._link_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            text = " ".join("".join(self._link_text).split())
            self.links.add((text, self._href))
            self._href = None
            self._link_text = []


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} PYTHON_REFERENCE_HTML", file=sys.stderr)
        return 2

    html_path = Path(sys.argv[1])
    parser = LinkCollector()
    parser.feed(html_path.read_text(encoding="utf-8"))

    missing = sorted(EXPECTED_LINKS - parser.links)
    if missing:
        print("Python API cross-references did not render:", file=sys.stderr)
        for text, href in missing:
            print(f"  {text!r} -> {href}", file=sys.stderr)
        return 1

    visible_text = "".join(parser.visible_text)
    if "][sedonadb." in visible_text:
        print(
            "Unrendered Python API cross-reference markup found in generated HTML",
            file=sys.stderr,
        )
        return 1

    print(f"Validated {len(EXPECTED_LINKS)} Python API cross-references")
    return 0


if __name__ == "__main__":
    sys.exit(main())
