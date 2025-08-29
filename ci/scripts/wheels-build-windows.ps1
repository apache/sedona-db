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

# If running locally:
# $env:VCPKG_ROOT="C:\Users\dewey\Documents\rscratch\vcpkg"
# $env:VCPKG_DEFAULT_TRIPLET="x64-windows-dynamic-release"
# $env:CIBW_BUILD="cp311-win_amd64"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$vcpkgBinDirectory = "$env:VCPKG_ROOT\installed\$env:VCPKG_DEFAULT_TRIPLET\bin"
$vcpkgLibDirectory = "$env:VCPKG_ROOT\installed\$env:VCPKG_DEFAULT_TRIPLET\lib"

# Put here/windows on PATH for our fake pkg-config and geos-config executables
$env:PATH += ";$scriptDirectory\windows"

# Give https://github.com/georust/geos/blob/47afbad2483e489911ddb456417808340e9342c3/sys/build.rs
# (well, specifically our dummy geos-config) the information it needs to build bindings
$env:GEOS_LIB_DIR = "$vcpkgLibDirectory"
$env:GEOS_VERSION = "3.13.0"
$originalDirectory = Get-Location

# Use delvewheel to copy any required dependencies from vcpkg into the wheel
$env:CIBW_REPAIR_WHEEL_COMMAND_WINDOWS="delvewheel repair -v --add-path=$vcpkgBinDirectory --wheel-dir={dest_dir} {wheel}"

# Quality of life: don't change the working directory of the calling script even when it fails
$parentDirectory = Split-Path -Parent (Split-Path -Parent $scriptDirectory)
try {
    # Compile windows/geos-config.rs into an executable using cargo. We need these because
    # vcpkg for GEOS does not build geos-config and because the rust geos crate requires
    # pkg-config to be available even if it cannot find GEOS.
    Push-Location "$scriptDirectory\windows"
	cargo build --release
	Copy-Item ".\target\release\geos-config.exe" ".\geos-config.exe" -Force
	Copy-Item ".\target\release\pkg-config.exe" ".\pkg-config.exe" -Force
	Pop-Location

	Push-Location "$parentDirectory"
	python -m cibuildwheel --output-dir python\sedonadb\dist python\sedonadb
}
finally {
	# Restore the original working directory
	Set-Location -Path $originalDirectory
}
