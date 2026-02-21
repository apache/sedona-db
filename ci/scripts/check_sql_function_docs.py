#!/usr/bin/env python3
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

import argparse
import json
import sys
from pathlib import Path

IGNORED_FUNCTIONS = {
    # Internal/unsupported for public docs
    "st_geomfromwkbunchecked",
}


def load_functions_from_stream(text):
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array from list-functions")
    return data


def validate_docs(functions, docs_dir):
    missing_docs = []
    missing_alias_mentions = []

    for item in functions:
        name = item.get("name")
        aliases = item.get("aliases", [])
        if not isinstance(name, str):
            raise ValueError(f"Invalid function item (name): {item!r}")
        if not isinstance(aliases, list) or not all(
            isinstance(a, str) for a in aliases
        ):
            raise ValueError(f"Invalid function item (aliases): {item!r}")
        if name in IGNORED_FUNCTIONS:
            continue

        qmd = docs_dir / f"{name}.qmd"
        if not qmd.exists():
            missing_docs.append(name)
            continue

        if aliases:
            text = qmd.read_text(encoding="utf-8").lower()
            for alias in aliases:
                if alias.lower() not in text:
                    missing_alias_mentions.append(f"{name}:{alias}")

    return missing_docs, missing_alias_mentions


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate that docs/reference/sql has a <function>.qmd for every "
            "function entry and that aliases are mentioned. Input JSON is "
            "expected to be emitted by `sedona-cli list-functions`."
        )
    )
    parser.add_argument(
        "--docs-dir",
        default="docs/reference/sql",
        help="Path to SQL docs directory (default: docs/reference/sql)",
    )
    parser.add_argument(
        "functions_json",
        help=(
            "Path to JSON file emitted by `sedona-cli list-functions`; "
            "use '-' to read from stdin"
        ),
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        raise ValueError(f"ERROR: docs directory not found: {docs_dir}")

    try:
        if args.functions_json == "-":
            raw_json = sys.stdin.read()
        else:
            raw_json = Path(args.functions_json).read_text(encoding="utf-8")
        functions = load_functions_from_stream(raw_json)
        missing_docs, missing_alias_mentions = validate_docs(functions, docs_dir)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Validation failed: {e}") from e

    source = "stdin" if args.functions_json == "-" else args.functions_json
    print(f"Checked {len(functions)} functions from: {source}")
    print(f"Docs directory: {docs_dir}")
    print(f"Missing docs: {len(missing_docs)}")
    print(f"Missing alias mentions: {len(missing_alias_mentions)}")

    if missing_docs:
        print("\nMissing .qmd files:")
        for name in missing_docs:
            print(f"- {name}")

    if missing_alias_mentions:
        print("\nMissing alias mentions (<function>:<alias>):")
        for item in missing_alias_mentions:
            print(f"- {item}")

    if missing_docs or missing_alias_mentions:
        return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
