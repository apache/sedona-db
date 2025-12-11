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
        arg = {k: to_str(v) for k, v in arg.items()}
    else:
        arg = to_str(arg)
        arg = {
            "type": arg,
            "name": DEFAULT_ARG_NAMES[arg],
            "description": DEFAULT_ARG_DESCRIPTIONS.get(arg, None),
        }

    if "default" not in arg:
        arg["default"] = None
    if "description" not in arg:
        arg["description"] = None

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


def render_usage(name, kernels, level):
    print(f"\n{heading(level + 1)} Usage\n")
    print("\n```sql")
    for kernel in kernels:
        args = ", ".join(render_arg(arg) for arg in kernel["args"])
        print(f"{to_str(kernel['returns'])} {to_str(name)}({args})")
    print("```")


def render_args(kernels, level):
    try:
        expanded_args = {}
        for kernel in reversed(kernels):
            args_dict = {arg["name"]: arg for arg in kernel["args"]}
            expanded_args.update({k: v for k, v in args_dict.items() if v is not None})
    except Exception as e:
        raise ValueError(
            f"Failed to consolidate argument documentation from kernels:\n{kernels}"
        ) from e

    print(f"\n{heading(level + 1)} Arguments\n")
    for arg in expanded_args.values():
        print(
            f"- **{to_str(arg['name'])}** ({to_str(arg['type'])}): {to_str(arg['description'])}"
        )


def render_meta(raw_meta, level=1, usage=True, arguments=True):
    if "description" in raw_meta:
        render_description(raw_meta["description"])

    if "kernels" in raw_meta:
        for kernel in raw_meta["kernels"]:
            kernel["args"] = expand_args(kernel["args"])

        if usage:
            render_usage(raw_meta["title"], raw_meta["kernels"], level)

        if arguments:
            render_args(raw_meta["kernels"], level=level)


def heading(level):
    return "#" * level + " "


def to_str(v):
    if isinstance(v, str):
        return v

    if isinstance(v, dict):
        if v["t"] == "Str":
            return v["c"]
        if v["t"] == "Code":
            return f"`{v['c'][1]}`"
        elif v["t"] == "Space":
            return " "
        elif v["t"] == "Para":
            return "".join(to_str(item) for item in v["c"])
        else:
            raise ValueError(f"Unhandled type in Pandoc ast convert: {v}")
    else:
        return "".join(to_str(item) for item in v)


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
        render_meta(raw_meta)
