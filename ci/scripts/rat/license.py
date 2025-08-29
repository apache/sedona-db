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

from pathlib import Path
import shutil

HERE = Path(__file__).parent
SEDONADB = HERE.parent.parent.parent
LICENSES = {
    "license_bash.txt": ["*.sh"],
    "license_c.txt": ["*.c", "*.cc", "*.h", "*.rs"],
    "license_md.txt": ["*.md"],
    "license_py.txt": [
        ".clang-format",
        ".cmake-format",
        ".gitignore",
        "*.cmake",
        "*.gitmodules",
        "*.ps1",
        "*.py",
        "*.R",
        "*.toml",
        "*.yaml",
        "*.yml",
        "CMakeLists.txt",
    ],
}


def load_licenses():
    out = {}

    for license_path, patterns in LICENSES.items():
        with open(HERE / license_path) as f:
            license = f.read()

        for pattern in patterns:
            out[pattern] = license

    return out


def load_ignored_patterns():
    with open(HERE / "license_ignore.txt") as f:
        return [item.strip() for item in f.readlines()]


def needs_license(path: Path, license):
    with open(path) as f:
        return f.read(len(license)) != license


def apply_license(path: Path, licenses, ignored, verbose):
    for pattern in ignored:
        if path.match(pattern):
            if verbose:
                print(f"Skipping '{path}' (matched license ignore '{pattern}')")
            return

    for pattern, license in licenses.items():
        if not path.match(pattern):
            continue

        if needs_license(path, license):
            if verbose:
                print(f"Applying license to '{path}'")

            path_tmp = path.with_suffix(".bak")
            path.rename(path_tmp)
            try:
                with open(path_tmp) as src, open(path, "w") as dst:
                    dst.write(license)
                    shutil.copyfileobj(src, dst)
            except Exception as e:
                path.unlink()
                path_tmp.rename(path)
                raise e

            path_tmp.unlink()
            return
        else:
            if verbose:
                print(f"Skipping '{path}' (already licensed)")
            return

    if verbose:
        print(f"Skipping '{path}' (no license pattern match)")


def main():
    import sys

    verbose = "--verbose" in sys.argv
    paths = [arg for arg in sys.argv[1:] if arg != "--verbose"]
    licenses = load_licenses()
    ignored = load_ignored_patterns()

    for path in paths:
        path = Path(path).resolve(strict=True)
        apply_license(path, licenses=licenses, ignored=ignored, verbose=verbose)


if __name__ == "__main__":
    main()
