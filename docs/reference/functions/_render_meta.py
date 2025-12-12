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


def render_meta(raw_meta, level=1, usage=True, arguments=True):
    """Render parsed YAML frontmatter into a standardized template

    Output:

    ```
    <description>

    ## Usage

    return_type ST_Name(arg_name: arg_type)

    ## Arguments

    - **arg_name** (arg_type): Arg description
    ```

    """
    if "description" in raw_meta:
        render_description(raw_meta["description"])

    if "kernels" in raw_meta:
        for kernel in raw_meta["kernels"]:
            kernel["args"] = expand_args(kernel["args"])

        if usage:
            render_usage(raw_meta["title"], raw_meta["kernels"], level)

        if arguments:
            render_args(raw_meta["kernels"], level=level)


def render_description(description):
    print(to_str(description).strip())


def render_usage(name, kernels, level):
    print(f"\n{heading(level + 1)} Usage\n")
    print("\n```sql")
    for kernel in kernels:
        args = ", ".join(render_usage_arg(arg) for arg in kernel["args"])
        print(f"{to_str(kernel['returns'])} {to_str(name)}({args})")
    print("```")


def render_usage_arg(arg):
    if arg["default"] is not None:
        return f"{arg['name']}: {arg['type']} = {arg['default']}"
    else:
        return f"{arg['name']}: {arg['type']}"


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


def expand_args(args):
    """Normalizes frontmatter-specified arguments

    This enables shorthand argument definitions like "geometry" to coexist
    with more verbosely defined argument definitions.
    """
    args = [expand_arg(arg) for arg in args]
    return deduplicate_common_arg_combinations(args)


def deduplicate_common_arg_combinations(expanded_args):
    """Transform argument names for binary functions with multiple geometry inputs

    Functions like ST_Intersects() that accept two geometries or two geographies
    will both have auto-generated function names [geom, geom]. This function
    renames them to [geomA, geomB].
    """
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


def expand_arg(arg):
    """Ensure each arg definition is a dict with keys type, name, description,
    and default
    """
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


# Define some default argument names/descriptions so that we can abbreviate
# the ubiquitous "geometry" input type argument without repeating the string
# "geom: Input geometry" for every single function.
DEFAULT_ARG_NAMES = {
    "geometry": "geom",
    "geography": "geog",
    "raster": "rast",
}

DEFAULT_ARG_DESCRIPTIONS = {
    "geometry": "Input geometry",
    "geography": "Input geography",
    "raster": "Input raster",
}


def heading(level):
    return "#" * level + " "


def to_str(v):
    """Convert a value to a string

    Because we call this from within a Quarto filter and use pandoc's built-in
    JSON output, we might get a pandoc JSONified AST here. Parsing the AST here
    is slightly easier than getting the Quarto filter to recreate the initial
    YAML frontmatter.
    """
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
        help=(
            "Function yaml metadata (e.g., frontmatter for a function doc page). "
            "Use `-` to read stdin."
        ),
    )

    args = parser.parse_args(sys.argv[1:])
    if args.meta == "-":
        args.meta = sys.stdin.read()

    with io.StringIO(args.meta) as f:
        raw_meta = yaml.safe_load(f)
        render_meta(raw_meta)
