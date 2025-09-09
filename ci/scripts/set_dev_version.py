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

import os
import re
import subprocess
import tomllib


def git(*args):
    out = subprocess.run(
        ["git"] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {args} failed:\n{out.stderr.decode()}")

    return out.stdout.decode().strip().splitlines()


def src_path(*args):
    release_dir = os.path.dirname(__file__)
    relative_path = os.path.join(release_dir, "..", "..", *args)
    return os.path.abspath(relative_path)


def file_regex_replace(pattern, replacement, path):
    with open(path) as f:
        content = f.read()

    # It is usually good to know if zero items are about to be replaced
    if re.search(pattern, content) is None:
        raise ValueError(f"file {path} does not contain pattern '{pattern}'")

    content = re.sub(pattern, replacement, content)
    with open(path, "w") as f:
        f.write(content)


def find_last_dev_tag():
    """Finds the commit of the last version bump

    Note that this excludes changes that happened during the release
    process but were not picked into the release branch.
    """
    try:
        maybe_last_dev_tag = git(
            "describe", "--match", "apache-sedona-db-*.dev", "--tags", "--abbrev=0"
        )
    except RuntimeError:
        first_commit = git("rev-list", "--max-parents=0", "HEAD")[0]
        return ("0.0.0", first_commit)

    last_dev_tag = maybe_last_dev_tag[0]
    last_version = re.search(r"[0-9]+\.[0-9]+\.[0-9]+", last_dev_tag).group(0)
    sha = git("rev-list", "-n", "1", last_dev_tag)[0]
    return last_version, sha


def find_commits_since(begin_sha, end_sha="HEAD"):
    lines = git("log", "--pretty=oneline", f"{begin_sha}..{end_sha}")
    return lines


def main():
    _, last_dev_tag = find_last_dev_tag()
    dev_distance = len(find_commits_since(last_dev_tag))

    file_regex_replace(
        r'\nversion = "([0-9]+\.[0-9]+\.[0-9]+)"',
        f'\nversion = "\\1-alpha{dev_distance}"',
        src_path("Cargo.toml"),
    )

    with open(src_path("Cargo.toml"), "rb") as f:
        print(tomllib.load(f)["workspace"]["package"]["version"])


if __name__ == "__main__":
    main()
