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

"""Script to parse coordinates through debug log, and replace them with easier
names.

"""

import sys
import re


def main():
    pts = {}

    float_regex = r"[-+]?(?:\d*\.\d+|\d+)"
    coord_regex = f"({float_regex}),? ({float_regex})"
    regex = re.compile(coord_regex)

    line = sys.stdin.readline()
    while line:
        if line.startswith("input:"):
            print(line, end="")
        m = regex.search(line)
        while m:
            st = m.start()
            print(line[:st], end="")
            sig = m.expand(r"\1#\2")
            if sig not in pts:
                pts[sig] = len(pts)
            idx = pts[sig]
            print(f"⚝{idx}", end="")

            en = m.end()
            line = line[en:]
            m = regex.search(line)
        print(line, end="")
        line = sys.stdin.readline()

    print("end of input")
    print("points:")
    for sig in pts:
        x, y = sig.split("#")
        idx = pts[sig]
        print(f"\t{idx}: Pt({x} {y})")


if __name__ == "__main__":
    main()
