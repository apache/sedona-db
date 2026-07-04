#!/usr/bin/env python3
#
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
import subprocess
import sys
from pathlib import Path


def cargo_metadata(workspace_root: Path) -> dict:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=workspace_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def is_publishable(package: dict) -> bool:
    # In cargo metadata, `publish = []` represents `publish = false`.
    return package.get("publish") != []


def package_path(workspace_root: Path, package: dict) -> str:
    manifest_path = Path(package["manifest_path"])
    return str(manifest_path.parent.relative_to(workspace_root))


def workspace_packages(metadata: dict) -> dict:
    workspace_ids = set(metadata["workspace_members"])
    return {
        package["id"]: package
        for package in metadata["packages"]
        if package["id"] in workspace_ids
    }


def workspace_graph(packages: dict) -> dict:
    by_name = {package["name"]: package["id"] for package in packages.values()}
    graph = {package_id: [] for package_id in packages}

    for package_id, package in packages.items():
        for dep in package["dependencies"]:
            if dep["name"] not in by_name:
                continue
            graph[package_id].append(by_name[dep["name"]])

    return graph


def find_cycles(graph: dict) -> list[list[str]]:
    visiting = set()
    visited = set()
    stack = []
    stack_index = {}
    cycles = []
    seen = set()

    def normalize(cycle: list[str]) -> tuple[str, ...]:
        without_repeat = cycle[:-1]
        min_pos = min(range(len(without_repeat)), key=lambda i: without_repeat[i])
        rotated = without_repeat[min_pos:] + without_repeat[:min_pos]
        return tuple(rotated)

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle = stack[stack_index[node] :] + [node]
            key = normalize(cycle)
            if key not in seen:
                seen.add(key)
                cycles.append(cycle)
            return

        visiting.add(node)
        stack_index[node] = len(stack)
        stack.append(node)

        for dep in graph[node]:
            visit(dep)

        stack.pop()
        stack_index.pop(node)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)

    return cycles


def publish_order(graph: dict, publishable_ids: set[str]) -> list[str]:
    ordered = []
    visited = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        visited.add(node)

        for dep in graph[node]:
            if dep in publishable_ids:
                visit(dep)

        if node in publishable_ids:
            ordered.append(node)

    for node in graph:
        visit(node)

    return ordered


def format_cycle(cycle: list[str], packages: dict) -> str:
    return " -> ".join(packages[node]["name"] for node in cycle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check SedonaDB cargo workspace dependencies and print publish order."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to the cargo workspace root.",
    )
    parser.add_argument(
        "--print-publish-order",
        action="store_true",
        help="Print publishable crate paths in dependency order.",
    )
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve()
    metadata = cargo_metadata(workspace_root)
    packages = workspace_packages(metadata)
    graph = workspace_graph(packages)

    cycles = find_cycles(graph)
    if cycles:
        print("Found circular workspace dependencies:", file=sys.stderr)
        for cycle in cycles:
            print(f"  - {format_cycle(cycle, packages)}", file=sys.stderr)
        return 1

    if args.print_publish_order:
        publishable_ids = {
            package_id
            for package_id, package in packages.items()
            if is_publishable(package)
        }
        ordered = publish_order(graph, publishable_ids)
        for package_id in ordered:
            package = packages[package_id]
            print(package_path(workspace_root, package))
    else:
        print("No circular workspace dependencies found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
