#!/usr/bin/env bash
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

# Used by both CI and local pre-commit hooks.

set -uo pipefail

mode="check"
ci_cleanup="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix)
            mode="fix"
            shift
            ;;
        --ci-cleanup)
            ci_cleanup="true"
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: ci/scripts/cargo-clippy.sh [--fix] [--ci-cleanup]

Options:
  --fix         Run clippy with --fix (best-effort; may not fix all violations).
  --ci-cleanup  Remove selected target/debug artifacts after clippy run.
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

status=0
if [[ "$mode" == "fix" ]]; then
    cargo clippy --fix --workspace --all-targets --all-features --allow-dirty --allow-staged -- -Dwarnings || status=$?
else
    cargo clippy --workspace --all-targets --all-features -- -Dwarnings || status=$?
fi

if [[ "$ci_cleanup" == "true" ]]; then
    rm -rf target/debug/deps
    rm -rf target/debug/incremental
    rm -rf target/debug/build
fi

exit "$status"
