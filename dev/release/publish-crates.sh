#!/bin/bash
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
#
# SedonaDB Crates.io Publishing Script
#
# This script publishes all SedonaDB crates to crates.io in the correct dependency order.
# Run with --dry-run first to validate everything before actual publishing.
#
# Usage:
#   ./scripts/publish-crates.sh --dry-run    # Validate only (recommended first)
#   ./scripts/publish-crates.sh --publish    # Actually publish to crates.io
#
# Prerequisites:
#   1. Create account at https://crates.io (via GitHub login)
#   2. Verify your email address
#   3. Generate API token at https://crates.io/settings/tokens
#   4. Run: cargo login <your-api-token>
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and workspace root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments
DRY_RUN=true
DRY_RUN_LOCAL=false
SKIP_TESTS=false
SKIP_VALIDATION=false
START_FROM=""

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dry-run         Validate packages without publishing (default)"
    echo "  --dry-run-local   Validate packages using local dependencies (more useful)"
    echo "  --publish         Actually publish to crates.io"
    echo "  --skip-tests      Skip running tests"
    echo "  --skip-validation Skip pre-publish validation"
    echo "  --start-from PKG  Start publishing from a specific package"
    echo "  --help            Show this help message"
    echo ""
    echo "Note: --dry-run will fail for crates with unpublished dependencies."
    echo "      Use --dry-run-local for validating the entire workspace first."
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            DRY_RUN_LOCAL=false
            shift
            ;;
        --dry-run-local)
            DRY_RUN=true
            DRY_RUN_LOCAL=true
            shift
            ;;
        --publish)
            DRY_RUN=false
            DRY_RUN_LOCAL=false
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Crates in dependency order (leaf crates first)
# This order ensures that when publishing a crate, all its dependencies are already published
CRATES=(
    # Tier 1 - Foundation crates (no internal dependencies)
    "rust/sedona-geo-traits-ext"
    "rust/sedona-geo-generic-alg"

    # Tier 2 - Core types (no internal deps)
    "rust/sedona-geometry"
    "rust/sedona-common"

    # Tier 3 - Schema (depends on common)
    "rust/sedona-schema"

    # Tier 4 - Expression (depends on common, geometry, schema)
    "rust/sedona-expr"

    # Tier 5 - Functions (depends on expr, geometry, schema, common)
    "rust/sedona-functions"

    # Tier 6 - C wrappers and sedona-geo (all depend on functions)
    "c/sedona-tg"
    "c/sedona-geos"
    "c/sedona-proj"
    "c/sedona-s2geography"
    "c/sedona-geoarrow-c"
    "rust/sedona-geo"

    # Tier 7 - Higher-level features
    "rust/sedona-geoparquet"
    "rust/sedona-raster"
    "rust/sedona-raster-functions"
    "rust/sedona-spatial-join"
    "rust/sedona-datasource"

    # Tier 8 - Testing utilities (depends on expr, geometry, schema, raster)
    "rust/sedona-testing"

    # Tier 9 - Main library (depends on most crates including sedona-testing)
    "rust/sedona"

    # Tier 10 - Crates that depend on main library
    "rust/sedona-adbc"
    "sedona-cli"
)

# Crates that should NOT be published
EXCLUDED_CRATES=(
    "python/sedonadb"      # Python bindings - use PyPI
    "r/sedonadb/src/rust"  # R bindings - use CRAN
    "rust/sedona-testing"  # Test utilities only
)

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  SedonaDB Crates.io Publishing Script${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    if [ "$DRY_RUN_LOCAL" = true ]; then
        echo -e "${YELLOW}Mode: DRY RUN LOCAL (package validation only)${NC}"
    else
        echo -e "${YELLOW}Mode: DRY RUN (full crates.io validation)${NC}"
    fi
else
    echo -e "${RED}Mode: PUBLISH (will upload to crates.io)${NC}"
fi
echo ""

cd "$WORKSPACE_ROOT"

# ============================================
# Step 1: Pre-flight checks
# ============================================
echo -e "${BLUE}Step 1: Pre-flight checks${NC}"
echo "----------------------------------------"

# Check if logged in to crates.io
if ! cargo login --help &>/dev/null; then
    echo -e "${RED}Error: cargo is not installed${NC}"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    git status --short
    echo ""
    if [ "$DRY_RUN" = false ]; then
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "Current branch: ${GREEN}$CURRENT_BRANCH${NC}"

# Check git tag
CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
echo -e "Workspace version: ${GREEN}$CURRENT_VERSION${NC}"
echo ""

# ============================================
# Step 2: Run tests (optional)
# ============================================
if [ "$SKIP_TESTS" = false ]; then
    echo -e "${BLUE}Step 2: Running tests${NC}"
    echo "----------------------------------------"

    echo "Running: cargo test --workspace --exclude sedonadb --exclude sedonadbr"
    if ! cargo test --workspace --exclude sedonadb --exclude sedonadbr --all-targets --all-features; then
        echo -e "${RED}Tests failed! Fix issues before publishing.${NC}"
        exit 1
    fi
    echo -e "${GREEN}All tests passed!${NC}"
    echo ""
else
    echo -e "${YELLOW}Step 2: Skipping tests (--skip-tests)${NC}"
    echo ""
fi

# ============================================
# Step 3: Build release
# ============================================
if [ "$SKIP_VALIDATION" = false ]; then
    echo -e "${BLUE}Step 3: Building release${NC}"
    echo "----------------------------------------"

    echo "Running: cargo build --workspace --exclude sedonadb --exclude sedonadbr --release"
    if ! cargo build --workspace --exclude sedonadb --exclude sedonadbr --release; then
        echo -e "${RED}Build failed!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Release build successful!${NC}"
    echo ""
else
    echo -e "${YELLOW}Step 3: Skipping build (--skip-validation)${NC}"
    echo ""
fi

# ============================================
# Step 4: Validate/Publish each crate
# ============================================
if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}Step 4: Validating packages (dry-run)${NC}"
else
    echo -e "${BLUE}Step 4: Publishing packages${NC}"
fi
echo "----------------------------------------"

STARTED=false
if [ -z "$START_FROM" ]; then
    STARTED=true
fi

FAILED_CRATES=()
PUBLISHED_CRATES=()
SKIPPED_CRATES=()

for crate_path in "${CRATES[@]}"; do
    crate_name=$(basename "$crate_path")

    # Handle --start-from
    if [ "$STARTED" = false ]; then
        if [ "$crate_name" = "$START_FROM" ] || [ "$crate_path" = "$START_FROM" ]; then
            STARTED=true
        else
            echo -e "${YELLOW}Skipping $crate_name (before --start-from)${NC}"
            SKIPPED_CRATES+=("$crate_name")
            continue
        fi
    fi

    # Get actual package name from Cargo.toml
    if [ -f "$WORKSPACE_ROOT/$crate_path/Cargo.toml" ]; then
        pkg_name=$(grep '^name = ' "$WORKSPACE_ROOT/$crate_path/Cargo.toml" | head -1 | sed 's/name = "\(.*\)"/\1/')
    else
        echo -e "${RED}Cargo.toml not found: $crate_path${NC}"
        FAILED_CRATES+=("$crate_path")
        continue
    fi

    echo ""
    echo -e "${BLUE}>>> Processing: $pkg_name ($crate_path)${NC}"

    cd "$WORKSPACE_ROOT/$crate_path"

    if [ "$DRY_RUN" = true ]; then
        if [ "$DRY_RUN_LOCAL" = true ]; then
            # Local dry run - use cargo check (compiles with local deps, no crates.io lookup)
            echo "    Running: cargo check --all-features"
            if cargo check --all-features 2>&1; then
                echo -e "    ${GREEN}Check passed${NC}"
                PUBLISHED_CRATES+=("$pkg_name")
            else
                echo -e "    ${RED}Check failed${NC}"
                FAILED_CRATES+=("$pkg_name")
            fi
        else
            # Full dry run - validates against crates.io
            echo "    Running: cargo publish --dry-run"
            if cargo publish --dry-run 2>&1; then
                echo -e "    ${GREEN}Validation passed${NC}"
                PUBLISHED_CRATES+=("$pkg_name")
            else
                echo -e "    ${RED}Validation failed${NC}"
                FAILED_CRATES+=("$pkg_name")
            fi
        fi
    else
        # Actual publish
        echo "    Running: cargo publish"
        if cargo publish 2>&1; then
            echo -e "    ${GREEN}Published successfully${NC}"
            PUBLISHED_CRATES+=("$pkg_name")

            # Wait for crates.io to index the new crate
            echo "    Waiting 120 seconds for crates.io to index..."
            sleep 120
        else
            echo -e "    ${RED}Publish failed${NC}"
            FAILED_CRATES+=("$pkg_name")

            # Ask whether to continue
            read -p "    Continue with remaining crates? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                break
            fi
        fi
    fi

    cd "$WORKSPACE_ROOT"
done

# ============================================
# Summary
# ============================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

if [ ${#PUBLISHED_CRATES[@]} -gt 0 ]; then
    if [ "$DRY_RUN" = true ]; then
        echo -e "${GREEN}Validated successfully (${#PUBLISHED_CRATES[@]} crates):${NC}"
    else
        echo -e "${GREEN}Published successfully (${#PUBLISHED_CRATES[@]} crates):${NC}"
    fi
    for crate in "${PUBLISHED_CRATES[@]}"; do
        echo "  - $crate"
    done
    echo ""
fi

if [ ${#SKIPPED_CRATES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Skipped (${#SKIPPED_CRATES[@]} crates):${NC}"
    for crate in "${SKIPPED_CRATES[@]}"; do
        echo "  - $crate"
    done
    echo ""
fi

if [ ${#FAILED_CRATES[@]} -gt 0 ]; then
    echo -e "${RED}Failed (${#FAILED_CRATES[@]} crates):${NC}"
    for crate in "${FAILED_CRATES[@]}"; do
        echo "  - $crate"
    done
    echo ""
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo -e "${GREEN}Dry run completed successfully!${NC}"
    echo ""
    echo "To actually publish, run:"
    echo "  $0 --publish"
else
    echo -e "${GREEN}All crates published successfully!${NC}"
fi
