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

DEFAULT_ARG_NAMES = {
    "geometry": "geom",
    "geography": "geog",
    "raster": "rast",
}

DEFAULT_ARG_DESCRIPTIONS = {
    "geometry": "Input geometry",
    "geography": "Input geography",
    "raster": "Input rast",
}


def expand_arg(arg):
    if isinstance(arg, dict):
        pass
    else:
        arg = to_str(arg)
        arg = {
            "type": arg,
            "name": DEFAULT_ARG_NAMES[arg],
            "description": DEFAULT_ARG_DESCRIPTIONS[arg],
        }

    if "default" not in arg:
        arg["default"] = None

    return arg


def deduplicate_common_arg_combinations(expanded_args):
    all_names = [arg["name"] for arg in expanded_args]
    if all_names[:2] == ["geom", "geom"]:
        all_names[:2] = ["geomA", "geomB"]
    elif all_names[:2] == ["geog", "geog"]:
        all_names[:2] = ["geogA", "geogB"]

    return [
        {
            "name": new_name,
            "type": arg["type"],
            "description": arg["description"],
            "default": arg["default"],
        }
        for arg, new_name in zip(expanded_args, all_names)
    ]


def expand_args(args):
    args = [expand_arg(arg) for arg in args]
    return deduplicate_common_arg_combinations(args)


def render_description(description):
    print(to_str(description).strip())


def render_arg(arg):
    if arg["default"] is not None:
        return f"{arg['name']}: {arg['type']} = {arg['default']}"
    else:
        return f"{arg['name']}: {arg['type']}"


def render_usage(name, kernels):
    print("\n## Usage\n")
    print("\n```sql")
    for kernel in kernels:
        args = ", ".join(render_arg(arg) for arg in kernel["args"])
        print(f"{to_str(kernel['returns'])} {to_str(name)}({args})")
    print("```")


def render_args(kernels):
    expanded_args = {}
    for kernel in reversed(kernels):
        args_dict = {arg["name"]: arg for arg in kernel["args"]}
        expanded_args.update(args_dict)

    print("\n## Arguments\n")
    for arg in expanded_args.values():
        print(
            f"- **{to_str(arg['name'])}** ({to_str(arg['type'])}): {to_str(arg['description'])}"
        )


def render_all(raw_meta):
    if "description" in raw_meta:
        render_description(raw_meta["description"])

    if "kernels" in raw_meta:
        for kernel in raw_meta["kernels"]:
            kernel["args"] = expand_args(kernel["args"])

        render_usage(raw_meta["title"], raw_meta["kernels"])
        render_args(raw_meta["kernels"])


def to_str(v):
    if isinstance(v, str):
        return v

    out = io.StringIO()
    for ast_item in v:
        if ast_item["t"] == "Str":
            out.write(ast_item["c"])
        elif ast_item["t"] == "Space":
            out.write(" ")
        else:
            raise ValueError(f"Unhandled type in Pandoc ast convert: {v}")
    return out.getvalue()


if __name__ == "__main__":
    import argparse
    import sys
    import yaml

    parser = argparse.ArgumentParser(description="Render SedonaDB SQL function header")
    parser.add_argument(
        "meta",
        help="Function yaml metadata (e.g., frontmatter for a function doc page)",
    )

    args = parser.parse_args(sys.argv[1:])
    if args.meta == "-":
        args.meta = sys.stdin.read()

    with io.StringIO(args.meta) as f:
        raw_meta = yaml.safe_load(f)
        render_all(raw_meta)
